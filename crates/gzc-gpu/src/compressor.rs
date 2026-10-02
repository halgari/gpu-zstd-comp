//! The entry point of the crate: [`Compressor`], [`Level`], [`CompressorOptions`], [`Frames`]
//! and the typed streaming form ([`Stream`], [`Batch`], [`Payload`]).
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;

use gzc_core::config::BLOCK_SIZE;
use gzc_core::params::{LVL3, LVL9S12SEG, MatchParams, OPT14, OPT16P1};

use crate::context::{GpuContext, GpuOptions};
use crate::error::{Error, invalid_input};
use crate::kernels::{FRAME_STRIDE, GpuParams, check_matching};
use crate::pipeline::{
    FrameBatch, FrameStream, Pipeline, PipelineConfig, PipelineStats, Region, UploadSlot, max_batch_for_budget,
    payload_blocks,
};

/// The most a frame is longer than its block: the frame header, the block header and, for a
/// compressed block, its section headers.
const FRAME_OVERHEAD: usize = FRAME_STRIDE - BLOCK_SIZE;

/// A compression level, named after the libzstd level whose ratio it matches on 64 KiB blocks.
///
/// Each level is one fixed preset of [`gzc_core::params`]:
///
/// | Level | Preset | Parse |
/// |---|---|---|
/// | [`Level::Zstd3`] | `lvl3` | greedy, two hash chains |
/// | [`Level::Zstd9`] | `lvl9s12seg` | lazy, 12-bit hash key, 4 KiB segments |
/// | [`Level::Zstd14`] | `opt14` | optimal, two passes |
/// | [`Level::Zstd16`] | `opt16p1` | optimal, one pass, sparse long chains |
///
/// The output for a level is the same on every GPU, and equal to what the CPU reference
/// (`gzc_core::reference::compress_block_to_frame`) writes for the preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Level {
    /// Fastest. Ratio of `zstd -3`.
    Zstd3,
    /// Ratio of `zstd -9`.
    Zstd9,
    /// Ratio of `zstd -14`.
    Zstd14,
    /// Best ratio. Ratio of `zstd -16`.
    Zstd16,
}

impl Level {
    /// Every level, fastest first.
    pub const ALL: [Level; 4] = [Level::Zstd3, Level::Zstd9, Level::Zstd14, Level::Zstd16];

    /// The match parameters the level stands for.
    pub fn preset(self) -> MatchParams {
        match self {
            Level::Zstd3 => LVL3,
            Level::Zstd9 => LVL9S12SEG,
            Level::Zstd14 => OPT14,
            Level::Zstd16 => OPT16P1,
        }
    }

    /// The name of [`Level::preset`] in `gzc_core::params::PRESETS`.
    pub fn preset_name(self) -> &'static str {
        match self {
            Level::Zstd3 => "lvl3",
            Level::Zstd9 => "lvl9s12seg",
            Level::Zstd14 => "opt14",
            Level::Zstd16 => "opt16p1",
        }
    }
}

/// How a [`Compressor`] is built. Start from [`CompressorOptions::new`] and change fields:
///
/// ```
/// use gzc_gpu::{CompressorOptions, Level};
///
/// let options = CompressorOptions { vram_budget_mib: 2048, ..CompressorOptions::new(Level::Zstd9) };
/// assert_eq!(options.inflight, 3);
/// ```
#[derive(Clone, Debug)]
pub struct CompressorOptions {
    /// The match parameters. [`CompressorOptions::new`] sets them from the level; any set that
    /// [`crate::gpu_supports`] accepts works, such as another entry of
    /// `gzc_core::params::PRESETS`.
    pub preset: MatchParams,
    /// GPU memory the compressor may allocate, in MiB. The default, 6144, leaves headroom on an
    /// 8 GB card. It sizes the batch when `batch_blocks` is `None`.
    ///
    /// It is a cap on what the compressor asks for. It is not checked against the memory the
    /// adapter has or has free: on a smaller or busy card, building the compressor fails with
    /// [`Error::OutOfMemory`].
    pub vram_budget_mib: u64,
    /// Blocks per batch. `None` picks the largest batch that fits `vram_budget_mib` and the
    /// device's limits. `Some(n)` uses `n` and ignores the budget.
    pub batch_blocks: Option<u32>,
    /// Batches in flight. Each one adds an upload buffer and a readback buffer. The default is 3.
    pub inflight: u32,
    /// Device and kernel options. The default reads nothing from the environment.
    pub gpu: GpuOptions,
}

impl CompressorOptions {
    /// The defaults for `level`: a 6144 MiB budget, 3 batches in flight, [`GpuOptions::default`].
    pub fn new(level: Level) -> Self {
        Self { preset: level.preset(), vram_budget_mib: 6144, batch_blocks: None, inflight: 3, gpu: GpuOptions::default() }
    }

    /// [`CompressorOptions::new`] with [`GpuOptions::try_from_env`], so the `GZC_*` variables
    /// apply. A variable with a value it does not take is [`Error::InvalidInput`].
    pub fn try_from_env(level: Level) -> Result<Self, Error> {
        Ok(Self { gpu: GpuOptions::try_from_env()?, ..Self::new(level) })
    }

    /// [`CompressorOptions::try_from_env`] for programs that would stop on a bad value anyway.
    ///
    /// # Panics
    ///
    /// Panics where `try_from_env` returns an error, with the error's message.
    pub fn from_env(level: Level) -> Self {
        Self::try_from_env(level).unwrap_or_else(|e| panic!("{e}"))
    }
}

impl From<Level> for CompressorOptions {
    fn from(level: Level) -> Self {
        Self::new(level)
    }
}

/// Compresses 64 KiB blocks on the GPU. Every block becomes one complete zstd frame.
///
/// A compressor owns a GPU device, the compiled kernels and its buffers. Building one takes
/// about a second; keep it and reuse it. It is `Send + Sync`. Calls on one compressor run one
/// after another.
///
/// A frame declares its content size and has no checksum. Any zstd decoder reads it. Frames
/// are independent: decode any one alone, or decode their concatenation to get the whole input
/// back.
///
/// ```no_run
/// use gzc_gpu::{Compressor, Level};
///
/// let data = std::fs::read("input.bin")?;
/// let compressor = Compressor::new(Level::Zstd16)?;
/// let frames = compressor.compress(&data)?;
/// std::fs::write("output.zst", frames.as_bytes())?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Compressor {
    ctx: Arc<GpuContext>,
    pipe: Mutex<Pipeline>,
    /// The threads of the call that holds `pipe`, to refuse a call from inside its closures.
    busy: Mutex<Busy>,
    preset: MatchParams,
    batch: u32,
}

/// The threads a running call's closures run on.
#[derive(Default)]
struct Busy {
    /// The thread that called `compress` or `stream`; `produce` runs on it.
    caller: Option<ThreadId>,
    /// The thread `on_batch` runs on, once it has run.
    delivery: Option<ThreadId>,
}

/// The compressor's pipeline, held for one call.
struct Held<'c> {
    pipe: MutexGuard<'c, Pipeline>,
    busy: &'c Mutex<Busy>,
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        *self.busy.lock().unwrap_or_else(PoisonError::into_inner) = Busy::default();
    }
}

impl Compressor {
    /// Opens the default GPU and builds a compressor for `level` with the default options.
    pub fn new(level: Level) -> Result<Self, Error> {
        Self::with_options(CompressorOptions::new(level))
    }

    /// Opens the default GPU as `options.gpu` says and builds a compressor on it.
    pub fn with_options(options: CompressorOptions) -> Result<Self, Error> {
        let ctx = GpuContext::new(options.gpu.clone()).map_err(Error::from_anyhow)?;
        Self::with_context(Arc::new(ctx), &options)
    }

    /// Builds a compressor on a device that is already open. `options.gpu` is not used.
    ///
    /// A context that reads frames back through a transfer queue
    /// ([`GpuContext::transfer_readback`]) serves one compressor at a time: building a second
    /// one while the first is alive is [`Error::InvalidInput`]. Open the context with
    /// [`GpuOptions::transfer_queue`] off to share it between live compressors.
    pub fn with_context(ctx: Arc<GpuContext>, options: &CompressorOptions) -> Result<Self, Error> {
        Self::build(ctx, options).map_err(Error::from_anyhow)
    }

    fn build(ctx: Arc<GpuContext>, options: &CompressorOptions) -> anyhow::Result<Self> {
        let params = GpuParams { matching: options.preset, emit_frames: true, huffman: true };
        check_matching(&options.preset)?;
        if options.inflight < 1 {
            return Err(invalid_input("inflight must be at least 1"));
        }
        let batch = match options.batch_blocks {
            Some(n) => n,
            None => max_batch_for_budget(&ctx, params, options.inflight, options.vram_budget_mib)?,
        };
        let pipe = Pipeline::new(&ctx, &PipelineConfig { batch, inflight: options.inflight, params })?;
        Ok(Self { ctx, pipe: Mutex::new(pipe), busy: Mutex::default(), preset: options.preset, batch })
    }

    /// Compresses `data`. It is split into 64 KiB blocks; the last block may be shorter. Frame
    /// `i` holds bytes `i * 65536 ..` of `data`, and decodes to exactly those bytes. Empty input
    /// gives zero frames.
    ///
    /// The result holds every frame in memory, and its buffer is reserved up front at the size
    /// of `data` plus 64 bytes per block. For input that should not be held twice, use
    /// [`Compressor::stream`], which hands out each batch's frames as they finish.
    pub fn compress(&self, data: &[u8]) -> Result<Frames, Error> {
        let blocks: Vec<&[u8]> = data.chunks(BLOCK_SIZE).collect();
        self.compress_blocks(&blocks)
    }

    /// Compresses blocks that are already split. Frame `i` decodes to exactly `blocks[i]`.
    ///
    /// Every block holds 1 to 65536 bytes. Any block may be short, not only the last: blocks are
    /// compressed independently, so this also serves a list of unrelated chunks. An empty block
    /// or one above 64 KiB is [`Error::InvalidInput`]. Memory use is as for
    /// [`Compressor::compress`].
    pub fn compress_blocks(&self, blocks: &[&[u8]]) -> Result<Frames, Error> {
        if let Some(i) = blocks.iter().position(|b| !(1..=BLOCK_SIZE).contains(&b.len())) {
            let msg = format!("block {i} is {} bytes, expected 1..={BLOCK_SIZE}", blocks[i].len());
            return Err(Error::InvalidInput(msg));
        }
        let mut frames = Frames::default();
        if blocks.is_empty() {
            return Ok(frames);
        }
        frames.ends.reserve_exact(blocks.len());
        // No frame is longer than its block plus `FRAME_OVERHEAD`, so the buffer never moves.
        frames.bytes.reserve_exact(blocks.iter().map(|b| b.len() + FRAME_OVERHEAD).sum());
        let mut held = self.lock()?;
        let threads = held.pipe.copy_threads();
        let run = held.pipe.stream_frames(
            |batch| {
                frames.push_batch(&batch, threads);
                Ok(())
            },
            |stream| stream.upload_blocks(blocks),
        );
        drop(held);
        run.map_err(|e| self.error(e))?;
        Ok(frames)
    }

    /// Compresses a stream of batches without copying on either side.
    ///
    /// `produce` runs on the calling thread. It takes one [`Batch`] after another from the
    /// [`Stream`], reserves room for its payloads, writes them straight into GPU upload memory
    /// and submits the batch. `on_batch` runs on a second thread. It gets each finished batch as
    /// a [`FrameBatch`], in submission order, while `produce` fills the next ones. The frames
    /// point into the readback buffer; the buffer is reused once the `FrameBatch` is dropped,
    /// which may happen on any thread.
    ///
    /// Block indices count from 0 in each call. Returns once `produce` has returned and every
    /// batch was delivered and dropped. An error from either closure stops the stream and is
    /// returned as it was. The compressor stays usable.
    ///
    /// The compressor is busy until `stream` returns. A call to `compress`, `compress_blocks`
    /// or `stream` on the same compressor from inside either closure is
    /// [`Error::InvalidInput`]; from any other thread it waits.
    ///
    /// ```no_run
    /// use gzc_gpu::{Compressor, Error, Level};
    ///
    /// let files: Vec<Vec<u8>> = vec![vec![1; 100_000], vec![2; 5_000]];
    /// let compressor = Compressor::new(Level::Zstd9)?;
    /// let mut out = Vec::new();
    /// compressor.stream(
    ///     |batch| {
    ///         for (_index, frame) in batch.frames() {
    ///             out.extend_from_slice(frame);
    ///         }
    ///         Ok(())
    ///     },
    ///     |stream| {
    ///         let lens: Vec<usize> = files.iter().map(Vec::len).collect();
    ///         let mut batch = stream.next_batch()?;
    ///         for (payload, file) in batch.reserve(&lens)?.iter_mut().zip(&files) {
    ///             payload.write(file);
    ///         }
    ///         batch.submit()?;
    ///         Ok(())
    ///     },
    /// )?;
    /// # Ok::<(), Error>(())
    /// ```
    pub fn stream<F, P>(&self, mut on_batch: F, produce: P) -> Result<PipelineStats, Error>
    where
        F: FnMut(FrameBatch) -> Result<(), Error> + Send,
        P: FnOnce(&mut Stream<'_, '_>) -> Result<(), Error>,
    {
        let mut held = self.lock()?;
        let busy = &self.busy;
        let run = held.pipe.stream_frames(
            |batch| {
                // One thread delivers every batch of a stream.
                busy.lock().unwrap_or_else(PoisonError::into_inner).delivery = Some(std::thread::current().id());
                on_batch(batch).map_err(anyhow::Error::new)
            },
            |inner| produce(&mut Stream { inner }).map_err(anyhow::Error::new),
        );
        drop(held);
        run.map_err(|e| self.error(e))
    }

    /// Blocks per batch: what `CompressorOptions::batch_blocks` or the VRAM budget came to.
    pub fn batch_blocks(&self) -> usize {
        self.batch as usize
    }

    /// The match parameters the compressor was built for.
    pub fn preset(&self) -> MatchParams {
        self.preset
    }

    /// The GPU device the compressor runs on.
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    /// One line naming the adapter and what the device runs with.
    pub fn describe(&self) -> String {
        self.ctx.describe()
    }

    /// Takes the pipeline for one call. Another thread's call is waited for. A call from a
    /// thread that the running call's closures run on could never finish, so it is an error.
    fn lock(&self) -> Result<Held<'_>, Error> {
        let me = std::thread::current().id();
        let busy = self.busy.lock().unwrap_or_else(PoisonError::into_inner);
        if busy.caller == Some(me) || busy.delivery == Some(me) {
            return Err(Error::InvalidInput(
                "the compressor was called from inside its own stream's closure; it is busy until that stream returns"
                    .to_string(),
            ));
        }
        drop(busy);
        // A stream that panicked has cleaned the pipeline up before the panic went on.
        let pipe = self.pipe.lock().unwrap_or_else(PoisonError::into_inner);
        self.busy.lock().unwrap_or_else(PoisonError::into_inner).caller = Some(me);
        Ok(Held { pipe, busy: &self.busy })
    }

    /// The typed form of a pipeline error. A failure without a class on a lost device is the
    /// device loss.
    fn error(&self, e: anyhow::Error) -> Error {
        match (Error::from_anyhow(e), self.ctx.device_lost()) {
            (Error::Other(e), Some(why)) => Error::DeviceLost(format!("GPU device lost: {why} ({e})")),
            (e, _) => e,
        }
    }
}

/// The frames of one [`Compressor::compress`] call, in block order, in one buffer.
///
/// ```
/// # let frames = gzc_gpu::Frames::default();
/// for frame in &frames {
///     // write `frame` out
/// # let _ = frame;
/// }
/// assert_eq!(frames.len(), 0);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Frames {
    bytes: Vec<u8>,
    /// The end of each frame in `bytes`. A frame starts where the one before it ends.
    ends: Vec<usize>,
}

impl Frames {
    /// The number of frames, one per block.
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    /// True when there is no frame: the input was empty.
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Frame `i`, the complete zstd frame of block `i`. Panics when `i` is out of range.
    pub fn frame(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.ends[i - 1] };
        &self.bytes[start..self.ends[i]]
    }

    /// Frame `i`, or `None` when `i` is out of range.
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        (i < self.len()).then(|| self.frame(i))
    }

    /// The frames in block order.
    pub fn iter(&self) -> FramesIter<'_> {
        FramesIter { frames: self, next: 0, end: self.len() }
    }

    /// Every frame back to back: a zstd stream that decodes to the whole input. It is empty for
    /// empty input.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// [`Frames::as_bytes`], taking the buffer.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Appends a finished batch's frames, copying them with up to `threads` threads (this one
    /// included). The copy goes into fresh memory. On an RTX 5090 a `lvl9s12seg` batch of 5403
    /// blocks (265 MB of frames) takes one thread 43 ms, longer than the GPU needs for the next
    /// batch, and four threads 19 ms.
    fn push_batch(&mut self, batch: &FrameBatch, threads: usize) {
        let n = batch.len();
        let base = self.bytes.len();
        let first = self.ends.len();
        let mut end = base;
        for k in 0..n {
            end += batch.frame(k).len();
            self.ends.push(end);
        }
        self.bytes.reserve(end - base);
        let ends = &self.ends[first..];
        let mut rest = &mut self.bytes.spare_capacity_mut()[..end - base];
        // Shares of whole frames, at least 64 per thread.
        let per = n.div_ceil(threads.max(1)).max(64);
        let mut at = base;
        std::thread::scope(|s| {
            let mut shares = Vec::new();
            for k in (0..n).step_by(per) {
                let last = (k + per).min(n);
                let (mine, tail) = rest.split_at_mut(ends[last - 1] - at);
                shares.push((k..last, mine));
                rest = tail;
                at = ends[last - 1];
            }
            let copy = |(range, mut dst): (std::ops::Range<usize>, &mut [std::mem::MaybeUninit<u8>])| {
                for k in range {
                    let frame = batch.frame(k);
                    let (here, tail) = dst.split_at_mut(frame.len());
                    here.write_copy_of_slice(frame);
                    dst = tail;
                }
            };
            let mut shares = shares.into_iter();
            let mine = shares.next();
            for share in shares {
                s.spawn(move || copy(share));
            }
            mine.map(copy);
        });
        // SAFETY: the shares cover `base..end` of the buffer back to back, and each wrote every
        // byte of its part: frame `k` fills the bytes up to `ends[k]`.
        unsafe { self.bytes.set_len(end) };
    }
}

impl<'a> IntoIterator for &'a Frames {
    type Item = &'a [u8];
    type IntoIter = FramesIter<'a>;

    fn into_iter(self) -> FramesIter<'a> {
        self.iter()
    }
}

/// Iterator over the frames of a [`Frames`], in block order.
#[derive(Clone, Debug)]
pub struct FramesIter<'a> {
    frames: &'a Frames,
    next: usize,
    end: usize,
}

impl<'a> Iterator for FramesIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        (self.next < self.end).then(|| {
            self.next += 1;
            self.frames.frame(self.next - 1)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.end - self.next;
        (n, Some(n))
    }
}

impl DoubleEndedIterator for FramesIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        (self.next < self.end).then(|| {
            self.end -= 1;
            self.frames.frame(self.end)
        })
    }
}

impl ExactSizeIterator for FramesIter<'_> {}

/// The producer's side of [`Compressor::stream`]: hands out batches to fill.
pub struct Stream<'a, 'p> {
    inner: &'a mut FrameStream<'p>,
}

impl<'p> Stream<'_, 'p> {
    /// Blocks one batch holds.
    pub fn batch_blocks(&self) -> usize {
        self.inner.slot_capacity()
    }

    /// Blocks submitted so far. The next submitted block gets this index.
    pub fn submitted_blocks(&self) -> usize {
        self.inner.submitted_blocks()
    }

    /// The next batch to fill. Waits while every batch is in flight or still held as a
    /// [`FrameBatch`]. A batch dropped without [`Batch::submit`] is handed out again.
    pub fn next_batch(&mut self) -> Result<Batch<'_, 'p>, Error> {
        let slot = self.inner.next_upload_slot().map_err(Error::from_anyhow)?;
        Ok(Batch { slot, used: 0, reserved_bytes: 0, written_bytes: AtomicUsize::new(0) })
    }

    /// Copies `blocks` into as many batches as they need and submits them. Every block holds 1
    /// to 65536 bytes, as for [`Compressor::compress_blocks`]. Their frames arrive at `on_batch`
    /// like any other batch's.
    pub fn submit_blocks(&mut self, blocks: &[&[u8]]) -> Result<(), Error> {
        self.inner.upload_blocks(blocks).map_err(Error::from_anyhow)
    }
}

/// One batch of a [`Stream`]: GPU upload memory for up to [`Batch::capacity`] blocks.
///
/// [`Batch::reserve`] takes each payload's length before any byte is written, and
/// [`Batch::submit`] takes no block count. The batch therefore always knows the real length of
/// every block, and a short block always gets a frame of exactly its own bytes.
///
/// The memory is reused from batch to batch and is not cleared. Each [`Payload`] counts the
/// bytes written to it, and [`Batch::submit`] fails unless every reserved byte was written, so
/// a frame never holds bytes of an earlier batch.
#[must_use = "a batch does nothing until it is submitted"]
pub struct Batch<'s, 'p> {
    slot: UploadSlot<'s, 'p>,
    /// Blocks reserved so far.
    used: usize,
    /// Bytes of the payloads reserved so far.
    reserved_bytes: usize,
    /// Bytes written to those payloads. A payload only appends, inside its own range, so this
    /// equals `reserved_bytes` exactly when every payload was written to its end.
    written_bytes: AtomicUsize,
}

impl Batch<'_, '_> {
    /// Blocks the batch holds.
    pub fn capacity(&self) -> usize {
        self.slot.capacity()
    }

    /// Blocks reserved so far.
    pub fn len(&self) -> usize {
        self.used
    }

    /// True when nothing is reserved yet.
    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    /// Blocks still free.
    pub fn remaining(&self) -> usize {
        self.capacity() - self.used
    }

    /// Reserves room for one payload per entry of `lens`, each `lens[i]` bytes long, after what
    /// is already reserved. A payload is a file or an independent chunk: it starts on a block
    /// boundary, takes `lens[i].div_ceil(65536)` blocks, and only its last block may be short.
    ///
    /// Returns one [`Payload`] per entry, in order. They are disjoint, so several threads can
    /// fill them at once, one thread per payload. Every payload must be written to its end
    /// before [`Batch::submit`].
    ///
    /// A zero length, or payloads that do not fit [`Batch::remaining`], is
    /// [`Error::InvalidInput`]; nothing is reserved then.
    pub fn reserve(&mut self, lens: &[usize]) -> Result<Vec<Payload<'_>>, Error> {
        let regions = self.slot.payloads_from(self.used, lens).map_err(Error::from_anyhow)?;
        self.used += lens.iter().map(|&len| payload_blocks(len)).sum::<usize>();
        self.reserved_bytes += lens.iter().sum::<usize>();
        let batch_written = &self.written_bytes;
        Ok(regions.into_iter().map(|region| Payload { region, written: 0, batch_written }).collect())
    }

    /// Submits the reserved blocks. Returns the index of the batch's first block in the stream.
    ///
    /// An empty batch is [`Error::InvalidInput`]. So is a batch with a payload that was not
    /// written to its end; nothing is submitted then, and the stream hands the batch's memory
    /// out again with the next [`Stream::next_batch`].
    pub fn submit(self) -> Result<usize, Error> {
        self.submit_tagged(0)
    }

    /// [`Batch::submit`] with a tag that the batch's [`FrameBatch::tag`] returns, such as an
    /// index into the caller's table of what the batch holds.
    pub fn submit_tagged(self, tag: u64) -> Result<usize, Error> {
        if self.used == 0 {
            return Err(Error::InvalidInput("cannot submit an empty batch".to_string()));
        }
        let written = self.written_bytes.load(Ordering::Relaxed);
        if written != self.reserved_bytes {
            return Err(Error::InvalidInput(format!(
                "cannot submit the batch: only {written} of its {} reserved payload bytes were written",
                self.reserved_bytes
            )));
        }
        self.slot.submit_with(self.used, tag).map_err(Error::from_anyhow)
    }
}

/// Write-only GPU upload memory for one payload of a [`Batch`], exactly as long as the payload.
///
/// A payload is written front to back, with [`Payload::write`] or through [`std::io::Write`],
/// and counts what it was given. [`Batch::submit`] refuses a batch whose payloads were not
/// written to the end, because the memory is reused: bytes that nobody wrote would be an earlier
/// batch's.
///
/// The memory may be write-combined and cannot be read. Do not decode into it: a decompressor
/// reads its own output. Decode into ordinary memory and write the result here.
///
/// One payload is filled by one thread. To fill a large file from several threads, reserve it
/// as several payloads of whole blocks (multiples of 65536 bytes, the rest last): the frames are
/// the same.
pub struct Payload<'a> {
    region: Region<'a>,
    /// Bytes written so far; the next write lands here.
    written: usize,
    /// The batch's count of written payload bytes.
    batch_written: &'a AtomicUsize,
}

impl Payload<'_> {
    /// The payload's length in bytes.
    pub fn len(&self) -> usize {
        self.region.len()
    }

    /// True for a payload of no bytes. [`Batch::reserve`] never returns one.
    pub fn is_empty(&self) -> bool {
        self.region.is_empty()
    }

    /// Bytes written so far.
    pub fn written(&self) -> usize {
        self.written
    }

    /// Bytes still to write.
    pub fn remaining(&self) -> usize {
        self.len() - self.written
    }

    /// Appends `bytes` after what was written before.
    ///
    /// # Panics
    ///
    /// Panics when `bytes` is longer than [`Payload::remaining`].
    pub fn write(&mut self, bytes: &[u8]) {
        assert!(
            bytes.len() <= self.remaining(),
            "{} bytes do not fit the {} left of a {}-byte payload",
            bytes.len(),
            self.remaining(),
            self.len()
        );
        self.region.write(self.written, bytes);
        self.written += bytes.len();
        self.batch_written.fetch_add(bytes.len(), Ordering::Relaxed);
    }
}

/// Appends like [`Payload::write`], taking at most [`Payload::remaining`] bytes. A full payload
/// takes none, which `write_all` and `std::io::copy` report as an error.
impl std::io::Write for Payload<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let n = bytes.len().min(self.remaining());
        Payload::write(self, &bytes[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// Pins that one compressor can serve several threads.
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    const fn send<T: Send>() {}
    send_sync::<Compressor>();
    send_sync::<Frames>();
    send::<Payload<'static>>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_map_to_their_presets() {
        use gzc_core::params::preset;
        assert_eq!(Level::ALL.map(Level::preset_name), ["lvl3", "lvl9s12seg", "opt14", "opt16p1"]);
        for level in Level::ALL {
            assert_eq!(preset(level.preset_name()), Ok(level.preset()), "{level:?}");
            assert!(crate::gpu_supports(&level.preset()), "{level:?}");
        }
    }

    /// A failed transfer-queue allocation (injected: this machine cannot run out on demand) is
    /// `Error::OutOfMemory` from the constructor, whichever buffer it hits, and leaves the
    /// context usable.
    #[test]
    fn a_failed_transfer_allocation_is_out_of_memory() {
        use crate::transfer::tests::FAIL_ALLOCATION_AFTER;
        let _gpu = crate::testing::gpu_test_slot();
        let ctx = crate::testing::gpu();
        if !ctx.transfer_readback() {
            eprintln!("skipped: no transfer queue on this adapter");
            return;
        }
        let options = CompressorOptions { batch_blocks: Some(4), inflight: 2, ..CompressorOptions::new(Level::Zstd3) };
        // Allocations in order: frames, frame_len, then each slot's staging and resolve buffers.
        for nth in [0, 1, 2, 4] {
            FAIL_ALLOCATION_AFTER.set(Some(nth));
            let e = Compressor::with_context(ctx.clone(), &options).err().expect("the allocation failed");
            assert_eq!(FAIL_ALLOCATION_AFTER.get(), None, "allocation {nth} was never reached");
            match &e {
                Error::OutOfMemory(m) => {
                    assert!(m.contains("out of memory") && m.contains("vram_budget_mib"), "allocation {nth}: {m}")
                }
                other => panic!("allocation {nth}: {other:?}"),
            }
        }
        let compressor = Compressor::with_context(ctx, &options).expect("the context still works");
        assert_eq!(compressor.compress(&[5; 70_000]).unwrap().len(), 2);
    }

    #[test]
    fn default_options() {
        let o = CompressorOptions::new(Level::Zstd9);
        assert_eq!((o.preset, o.vram_budget_mib, o.batch_blocks, o.inflight), (LVL9S12SEG, 6144, None, 3));
        assert_eq!(o.gpu, GpuOptions::default());
        assert_eq!(CompressorOptions::from(Level::Zstd3).preset, LVL3);
    }

    #[test]
    fn frames_index_and_iterate() {
        let f = Frames { bytes: b"aabbbc".to_vec(), ends: vec![2, 5, 6] };
        assert_eq!((f.len(), f.is_empty()), (3, false));
        assert_eq!([f.frame(0), f.frame(1), f.frame(2)], [&b"aa"[..], b"bbb", b"c"]);
        assert_eq!((f.get(2), f.get(3)), (Some(&b"c"[..]), None));
        assert_eq!(f.iter().collect::<Vec<_>>(), [&b"aa"[..], b"bbb", b"c"]);
        assert_eq!(f.iter().rev().collect::<Vec<_>>(), [&b"c"[..], b"bbb", b"aa"]);
        assert_eq!(f.iter().len(), 3);
        assert_eq!((&f).into_iter().count(), 3);
        assert_eq!(f.as_bytes(), b"aabbbc");
        assert_eq!(f.into_bytes(), b"aabbbc");
        let empty = Frames::default();
        assert!(empty.is_empty() && empty.iter().next().is_none() && empty.as_bytes().is_empty());
    }
}
