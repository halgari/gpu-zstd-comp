//! In-flight streaming submission and GPU timestamp queries.
//!
//! `inflight` slots each own a mappable upload buffer and a mappable staging (readback) buffer.
//! Every device-side buffer, the kernels' input (`data`), scratch and outputs (`frames`,
//! `frame_len`) included, is shared by all slots: the queue runs one submission after another
//! and each submission consumes what it produced (its outputs are copied into the slot's staging
//! buffer before the next submission's kernels start, which wgpu orders with a barrier), so only
//! the host-visible buffers need one copy per batch in flight (see `vram_bytes`). A batch is
//! written into a free slot's upload buffer; the upload copy, the kernels and the readback into
//! the slot's staging buffer (plus resolved timestamps) are recorded in one submission, and the
//! staging buffer is mapped once. Completed slots are handed to the sink in submission order while
//! later batches keep the GPU busy.
//!
//! Threads: the caller's thread is the producer: it writes each batch into a slot's mapped upload
//! buffer (itself with `stream_frames`, or `FrameStream::upload_blocks` copying from `&[&[u8]]`)
//! and submits it. A completion thread (one per `run*` / `stream_frames` call) waits for the
//! batches in submission order and lends each one's staging bytes to the sink (`Lease`,
//! `FrameBatch`); a slot is reused once its batch is released, so delivery overlaps the next
//! uploads instead of following them on one thread. Slot states live in `Shared`.
//!
//! Direct upload (`GpuContext::direct_upload`, full ReBAR): there is no shared `data` and no
//! upload copy; each submission binds its slot's upload buffer (device-local, mapped for the
//! host between submissions) as `data`.
//!
//! Transfer readback (`GpuContext::transfer`, frame path; see `Xfer`): each batch
//! is two main-queue submissions (K1–K3, then K5/K4 and the timestamps) and a copy on the
//! transfer queue into the slot's host staging buffer, ordered by timeline semaphores; `frames`
//! and `frame_len` are buffers both queue families share. The readback then overlaps the next
//! batch's kernels instead of following its batch's K4 on the main queue. The `gpu_readback`
//! entry of `PipelineStats::transfer_ms` is K4's end to the batch's end marker, so it only counts
//! the main queue's share (about 0 on this path).
//!
//! Two output paths, chosen by `GpuParams::emit_frames` when the pipeline is built:
//! - parses (`run`, `BlockSink`): K1→K2→K3, staging holds `counts` and the full fixed-stride
//!   `seqs` region; the host decodes a `BlockOutput` per block, gathering its literals from the
//!   block (K3 writes no literals).
//! - frames (`run_frames`, `FrameSink`): K1→K2→K3→K5→K4, staging holds `frame_len` and the
//!   `frames` region, a copy of the fixed-stride buffer; the sink reads each frame in place
//!   (`FrameBatch`). K5 (the literals
//!   section, Huffman-coded with `GpuParams::huffman`) gathers the literals from `data` and writes
//!   into the same `frames` / `frame_len` buffers, so it adds no memory.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::Instant;

use anyhow::{Context as _, anyhow};
use gzc_core::config::BLOCK_SIZE;
use gzc_core::seq::BlockOutput;

use crate::kernels::{
    BatchBuffers, FRAME_STRIDE, GpuParams, KERNEL_QUERIES, Kernels, TRUNC_KERNEL_NAME, decode_output, k4_tables_bytes,
    max_seqs,
};
use crate::sizing::{
    counts_bytes, data_bytes, frame_len_bytes, frames_bytes, max_batch_blocks, scratch_bytes, seqs_bytes_for, slot_bytes,
};
use gzc_core::params::MatchParams;
use crate::context::{ErrorScopes, GpuContext};
use crate::transfer::{Commands, RawBuffer, StreamingGuard, Timeline, TransferQueue};

mod completion;
mod slots;
mod stream;
#[cfg(test)]
mod tests;

use completion::*;
use slots::*;
use stream::*;
pub use slots::FrameBatch;
pub(crate) use stream::check_blocks;
pub use stream::{FrameStream, Region, UploadSlot, payload_blocks, payload_real_lens};

#[derive(Clone, Copy, Debug)]
pub struct PipelineConfig {
    /// Blocks per batch (one submission).
    pub batch: u32,
    /// Batches in flight (slots).
    pub inflight: u32,
    pub params: GpuParams,
}

#[derive(Clone, Debug, Default)]
pub struct PipelineStats {
    /// GPU time summed over all batches per kernel ("k1_chains", "k2_best", "k3_parse", plus
    /// "k4_entropy" on the frame path and "k5_huffman" with Huffman literals, and last "k3_trunc"
    /// when some batch held a partial block and so ran K3t), in milliseconds;
    /// empty when the device has no timestamp queries; a kernel whose timestamps the device left
    /// unwritten (in any batch) is left out. That happens on Metal (an M4 Pro): with counters
    /// sampled at stage boundaries only (no `TIMESTAMP_QUERY_INSIDE_ENCODERS`) it left K4's pair,
    /// the last kernel's, unwritten in some batch of the optimal parse's frame-path test (batches
    /// of 16 and 2 blocks; the bench's larger batches time it), though that pass samples like
    /// every other kernel's: its own compute encoder, begin and end on the pass, every slot
    /// written before the resolve in the same submission.
    pub kernel_ms: Vec<(String, f64)>,
    /// Wall time from the first upload to the last block handed to the sink.
    pub wall_s: f64,
    /// Number of batches submitted.
    pub batches: u32,
    /// Where the time outside the kernels goes, summed over all batches, in milliseconds (see
    /// `TRANSFER_NAMES`). The `gpu_*` entries come from timestamps (left out without them, or
    /// when the device left a marker timestamp unwritten, which Metal apparently does for the
    /// empty marker passes); the `host_*` ones are wall time on the producer or the completion
    /// thread.
    pub transfer_ms: Vec<(String, f64)>,
}

/// Names of `PipelineStats::transfer_ms`, in order:
/// - `gpu_upload_copy`: the upload -> data copy (the batch's start marker to K1's begin).
/// - `gpu_readback`: the output -> staging copies (K4's end, or K3's on the parse path, to the
///   batch's end marker).
/// - `gpu_idle`: gaps between one batch's end marker and the next batch's start marker. This
///   includes the previous batch's timestamp resolve and its copy into staging (recorded after
///   its end marker; a few µs) and any barrier work at the start of a submission.
///
/// Producer thread (the one calling `run*` / `stream_frames`; its time is the critical path):
/// - `host_upload_wait`: waiting for the next slot: its batch completed and released by the sink,
///   its upload buffer mapped again.
/// - `host_upload_write`: from handing out the upload slot to its submission (the wrappers: the
///   copy into the mapped upload buffer).
/// - `host_submit`: unmapping, recording and submitting the batch, plus the map requests.
/// - `host_fill`: from the start to the first submission.
///
/// Completion thread (overlaps the producer):
/// - `host_wait`: blocked on the GPU for the oldest batch.
/// - `host_deliver`: handing a completed batch to the sink (for `stream_frames`: until `on_batch`
///   returns; a sink that keeps the `FrameBatch` does its work after that).
/// - `host_unmap`: releasing the batches (unmapping the staging buffer), on whichever thread.
/// - `host_drain`: from the last batch's GPU completion to the end (its delivery and release).
pub const TRANSFER_NAMES: [&str; 11] = [
    "gpu_upload_copy",
    "gpu_readback",
    "gpu_idle",
    "host_upload_wait",
    "host_upload_write",
    "host_submit",
    "host_wait",
    "host_deliver",
    "host_unmap",
    "host_fill",
    "host_drain",
];

/// Receives each block's parse; called exactly once per index, in arbitrary order. `Pipeline::run`
/// calls it on the pipeline's completion thread (hence `Send` there).
pub trait BlockSink {
    fn put(&mut self, index: usize, out: BlockOutput);
}

/// Receives each block's complete zstd frame; called exactly once per index. `Pipeline::run_frames`
/// calls it on the pipeline's completion thread (hence `Send` there), batch after batch in
/// submission order and in index order within a batch, while the calling thread uploads the next
/// batches. `frame` borrows the batch's staging buffer: copy (or write) it out before returning.
pub trait FrameSink {
    fn put(&mut self, index: usize, frame: &[u8]);
}

/// A frame sink that several delivery threads call at once (`Pipeline::run_frames_par`): exactly
/// once per index, in arbitrary order. `frame` borrows the staging buffer as in `FrameSink`.
pub trait ParFrameSink: Sync {
    fn put(&self, index: usize, frame: &[u8]);
}

/// Hands one completed batch to the caller's sink on the completion thread: `(lease of the
/// staging bytes, first block index, block count, the producer's tag)`.
type Handler<'d> = dyn FnMut(Lease, usize, u32, u64) -> anyhow::Result<()> + Send + 'd;

/// A submitted batch, sent to the completion thread.
struct Job {
    slot: usize,
    first: usize,
    n: u32,
    /// `UploadSlot::submit_with`'s tag (0 for `submit`).
    tag: u64,
    submission: wgpu::SubmissionIndex,
    /// The staging map's result (wgpu staging), or None (transfer readback: wait for `seq`).
    mapped: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    /// Transfer readback: the batch's value on the `Xfer` timelines.
    seq: u64,
    staging: Arc<Staging>,
    /// The batch held a partial block: it ran K3t, and its timestamps include K3t's pair.
    partial: bool,
}

/// Transfer readback (frame path, `GpuContext::transfer`): K4 writes `frames` /
/// `frame_len` (shared by both queue families, imported into `bufs`); each batch's main-queue work
/// is two submissions, K1–K3 and then K5/K4 (+ timestamps), the second signalling `k_done` = the
/// batch's `seq`; the transfer queue waits for that, copies frame_len, frames and the timestamps
/// into the slot's host staging buffer and signals `t_done` = `seq`, which the host waits for. The
/// K5/K4 submission of the next batch waits for the previous `t_done` before overwriting `frames`
/// (long passed by then: K1–K3 run in between). The readback so leaves the main queue.
struct Xfer {
    frames: RawBuffer,
    frame_len: RawBuffer,
    /// One command buffer per slot.
    cmds: Commands,
    k_done: Timeline,
    /// Shared with the completion thread, which waits on it.
    t_done: Arc<Timeline>,
    /// `seq` of the next batch (timeline values only grow, across runs).
    next_seq: u64,
    /// This pipeline's exclusive claim on the transfer queue (one transfer-readback `Pipeline`
    /// per context, see `GpuContext::transfer`); released when the pipeline drops.
    _streaming: StreamingGuard,
    /// Declared last: the objects above go first.
    tq: std::sync::Arc<TransferQueue>,
}

/// Compiled kernels plus the shared device buffers and `inflight` slots of `batch` blocks,
/// reusable across runs. It shares ownership of its context.
pub struct Pipeline {
    ctx: Arc<GpuContext>,
    cfg: PipelineConfig,
    kernels: Kernels,
    /// Every device-side buffer, shared by all slots (see the module docs).
    bufs: BatchBuffers,
    layout: StagingLayout,
    slots: Vec<Slot>,
    /// The slots' staging states (see `Shared`).
    shared: Arc<Shared>,
    /// `GpuContext::direct_upload`: `bufs.data` is the submitting slot's upload buffer (set per
    /// submission) and there is no upload copy.
    direct: bool,
    /// Threads `FrameStream::upload_blocks` copies a batch with (`upload_threads`).
    upload_threads: usize,
    /// Transfer readback (see `Xfer`); declared last so it drops after every wgpu import of its
    /// buffers.
    xfer: Option<Xfer>,
    /// Test hook: the delivery after this many more fails (then the hook clears).
    #[cfg(test)]
    fail_deliveries_after: Option<u32>,
    /// Test hook: the submission after this many more fails once its slot is marked in flight
    /// (then the hook clears).
    #[cfg(test)]
    fail_submits_after: Option<u32>,
    /// Test hook: the completion thread waits as on Metal (`Completion::poll_only`) whatever the
    /// backend.
    #[cfg(test)]
    poll_only: bool,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if let Some(x) = &self.xfer {
            // The raw buffers, semaphores and command buffers must outlive every GPU use.
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
            x.tq.idle();
        }
    }
}

impl Pipeline {
    /// Compiles the kernels and allocates the shared buffers and every slot, for the frame path
    /// when `cfg.params.emit_frames` and the parse path otherwise. Errors on `batch`/`inflight`
    /// of 0, a batch above `max_batch_blocks`, match params the GPU does not support (see
    /// `Kernels::new`), a wgpu out-of-memory/validation error, or a lost device; a failed
    /// allocation errors here ("GPU allocation of N MiB ... failed"), naming the buffer, never
    /// later as an invalid buffer.
    pub fn new(ctx: &Arc<GpuContext>, cfg: &PipelineConfig) -> anyhow::Result<Self> {
        let m = cfg.params.matching;
        let max = max_batch_blocks(&ctx.device.limits(), &m);
        anyhow::ensure!(cfg.inflight >= 1, "inflight must be at least 1");
        anyhow::ensure!(cfg.batch >= 1 && cfg.batch <= max, "batch {} not in 1..={max} for this device", cfg.batch);
        if let Some(why) = ctx.device_lost() {
            anyhow::bail!("GPU device lost: {why}");
        }

        let scopes = ErrorScopes::push(ctx);
        let frames = cfg.params.emit_frames;
        // Kernels::new validates the match params (`check_matching`) before any buffer is allocated.
        let kernels = Kernels::new(ctx, cfg.params)?;
        let layout = StagingLayout::new(cfg.batch, frames, &m);
        scopes.pop()?;
        // Every allocation below, checked once at the end (`ErrorScopes::pop_alloc`): an
        // out-of-memory or invalid buffer fails here, naming the buffer, not at its first use.
        let scopes = ErrorScopes::push(ctx);
        let staging_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let direct = ctx.direct_upload;
        let upload_usage = if direct {
            // The kernels bind it as `data` (MAPPABLE_PRIMARY_BUFFERS).
            wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::STORAGE
        } else {
            wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC
        };
        let tq = ctx.transfer.clone().filter(|_| frames);
        use ash::vk::BufferUsageFlags as U;
        let shared = U::STORAGE_BUFFER | U::TRANSFER_SRC | U::TRANSFER_DST;
        // The transfer readback's raw objects exist before any wgpu buffer imports them, so on an
        // early error return the imports (declared later) drop first.
        let xfer = match tq {
            Some(tq) => Some(Xfer {
                // First, so a second transfer-readback pipeline errors before allocating anything.
                _streaming: tq.begin_streaming()?,
                frames: tq.buffer(frames_bytes(cfg.batch) + ctx.poison_pad(), shared, false)?,
                frame_len: tq.buffer(frame_len_bytes(cfg.batch) + ctx.poison_pad(), shared, false)?,
                cmds: tq.commands(cfg.inflight)?,
                k_done: tq.timeline()?,
                t_done: Arc::new(tq.timeline()?),
                next_seq: 1,
                tq,
            }),
            None => None,
        };
        let mut slots = Vec::with_capacity(cfg.inflight as usize);
        for _ in 0..cfg.inflight {
            let staging = match &xfer {
                Some(x) => Staging::Host(x.tq.buffer(layout.size, U::TRANSFER_DST, true)?),
                None => Staging::Wgpu(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pipeline.staging"),
                    size: layout.size,
                    usage: staging_usage,
                    mapped_at_creation: false,
                })),
            };
            let queries = if ctx.timestamps {
                let set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("pipeline.timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: PIPELINE_QUERIES,
                });
                let size = PIPELINE_QUERIES as u64 * wgpu::QUERY_SIZE as u64;
                let usage = wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC;
                let (resolve, raw) = match &xfer {
                    Some(x) => {
                        let raw = x.tq.buffer(size, U::TRANSFER_SRC | U::TRANSFER_DST, false)?;
                        // SAFETY: as for `frames` above (QUERY_RESOLVE and COPY_SRC map to
                        // TRANSFER_DST and TRANSFER_SRC); the slot keeps `raw` beside the import.
                        (unsafe { raw.import(&ctx.device, "pipeline.resolve", usage) }, Some(raw))
                    }
                    None => (
                        ctx.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("pipeline.resolve"),
                            size,
                            usage,
                            mapped_at_creation: false,
                        }),
                        None,
                    ),
                };
                Some(Queries { set, resolve, raw })
            } else {
                None
            };
            // `data_bytes`: the batch plus one trailing word, which `UploadSlot::submit_with` zeroes
            // after the last submitted block (K3opt reads one word past each block; with the
            // direct upload this buffer is `data`, with the copy upload it is copied along).
            #[cfg(test)]
            let upload_size = data_bytes(cfg.batch) + ctx.poison_pad() + tests::EXTRA_UPLOAD_BYTES.get();
            #[cfg(not(test))]
            let upload_size = data_bytes(cfg.batch) + ctx.poison_pad();
            let upload = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pipeline.upload"),
                size: upload_size,
                usage: upload_usage,
                mapped_at_creation: true,
            });
            assert_eq!(upload.size(), upload_size, "upload slots keep the trailing zero word");
            slots.push(Slot {
                upload,
                upload_mapped: None,
                upload_unmapped: false,
                upload_submission: None,
                staging: Arc::new(staging),
                staging_requested: false,
                queries,
            });
        }
        // Direct upload: no shared `data`, each submission binds its slot's upload buffer (see
        // `submit`). Transfer readback: frames / frame_len are buffers both queue families may use.
        // Neither is allocated in wgpu then, so the allocations add up to `vram_bytes_with`.
        let data = direct.then(|| slots[0].upload.clone());
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        // SAFETY: `ctx.device` lives on the transfer queue's VkDevice; the buffers were created
        // STORAGE | TRANSFER_SRC | TRANSFER_DST and CONCURRENT over families 0 and the transfer
        // family; `xfer` outlives every GPU use (`Pipeline`'s Drop waits for both queues and `xfer`
        // is declared after `bufs`, here and in `Pipeline`).
        let frame_bufs = xfer.as_ref().map(|x| unsafe {
            let frames = x.frames.import(&ctx.device, "batch.frames", usage);
            (frames, x.frame_len.import(&ctx.device, "batch.frame_len", usage))
        });
        let bufs = BatchBuffers::with_parts(ctx, cfg.batch, frames, &m, data, frame_bufs)?;
        let what = format!("the pipeline (batch {}, inflight {})", cfg.batch, cfg.inflight);
        scopes.pop_alloc(&what, vram_bytes_with(cfg, direct))?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            cfg: *cfg,
            kernels,
            bufs,
            layout,
            shared: Shared::new(slots.len()),
            slots,
            direct,
            upload_threads: upload_threads(ctx),
            xfer,
            #[cfg(test)]
            fail_deliveries_after: None,
            #[cfg(test)]
            fail_submits_after: None,
            #[cfg(test)]
            poll_only: false,
        })
    }

    /// Every wait for the GPU polls without blocking (`GpuContext::poll_only`: Metal; or the
    /// test hook).
    fn poll_only(&self) -> bool {
        #[cfg(test)]
        return self.poll_only || self.ctx.poll_only();
        #[cfg(not(test))]
        return self.ctx.poll_only();
    }

    /// True when batches are read back through the transfer queue (see `Xfer`).
    pub fn transfer_readback(&self) -> bool {
        self.xfer.is_some()
    }

    /// How this pipeline's K3 runs (the mode its kernels were built with).
    pub fn k3_mode(&self) -> Option<crate::kernels::K3Mode> {
        self.kernels.k3_mode()
    }

    /// Bytes of every buffer the pipeline created; timestamp query sets and their resolve
    /// buffers are left out, as in `vram_bytes`.
    #[cfg(test)]
    fn allocated_bytes(&self) -> u64 {
        // Every buffer but the staging ones is `poison_pad` larger than its logical size.
        let pad = self.ctx.poison_pad();
        let size = |b: &wgpu::Buffer| b.size() - pad;
        let opt = |b: &Option<wgpu::Buffer>| b.as_ref().map_or(0, size);
        let s = &self.bufs;
        // Direct upload: `data` is a slot's upload buffer, counted with the slots.
        let data = if self.direct { 0 } else { size(&s.data) };
        let k3opt = s.opt.as_ref().map_or(0, |o| size(&o.prices) + size(&o.scratch) + size(&o.sched));
        let shared = data
            + [&s.head, &s.pred, &s.best, &s.seqs, &s.counts].iter().map(|b| size(b)).sum::<u64>()
            + opt(&s.frames)
            + opt(&s.frame_len)
            + opt(&s.lens)
            + k3opt;
        let per_slot: u64 = self.slots.iter().map(|slot| size(&slot.upload) + slot.staging.size()).sum();
        // Transfer readback: frames / frame_len are imports of `Xfer`'s buffers (same sizes).
        let own = self.kernels.own_buffer_bytes();
        shared + per_slot + if own > 0 { own - pad } else { 0 }
    }

    /// Streams `blocks` (each BLOCK_SIZE bytes) through the slots, handing every block's parse
    /// to `sink` exactly once, on the completion thread. Errors on a frame-path pipeline, and on
    /// wgpu validation or out-of-memory errors.
    pub fn run(&mut self, blocks: &[&[u8]], sink: &mut (impl BlockSink + Send)) -> anyhow::Result<PipelineStats> {
        anyhow::ensure!(!self.layout.frames, "pipeline built with emit_frames: use run_frames");
        check_blocks(blocks)?;
        let layout = self.layout;
        let max = max_seqs(&self.cfg.params.matching);
        let handler = move |lease: Lease, first: usize, n: u32, _tag: u64| -> anyhow::Result<()> {
            let words: &[u32] = bytemuck::cast_slice(lease.bytes());
            let seq_stride = 3 * max as usize;
            for b in 0..n as usize {
                let (n_seq, n_lit) = (words[2 * b], words[2 * b + 1]);
                anyhow::ensure!(
                    n_seq <= max && n_lit as usize <= BLOCK_SIZE,
                    "block {}: bad counts ({n_seq}, {n_lit})",
                    first + b
                );
                let s = (layout.a / 4) as usize + b * seq_stride;
                let out = decode_output(blocks[first + b], &words[s..s + seq_stride], n_seq);
                let got = out.literals.len();
                anyhow::ensure!(
                    got == n_lit as usize,
                    "block {}: K3 counted {n_lit} literals, the sequences leave {got}",
                    first + b
                );
                sink.put(first + b, out);
            }
            Ok(())
        };
        self.stream_with(Box::new(handler), |stream| stream.upload_blocks(blocks))
    }

    /// Streams `blocks` through the slots, handing every block's zstd frame (K4 output) to `sink`
    /// exactly once, on the completion thread (see `FrameSink`) while this thread uploads. A block
    /// is its real bytes, 1..=BLOCK_SIZE of them (`Block::real_len`: a file's last block may be
    /// short), and its frame decodes to exactly those. Errors on a parse-path pipeline, and on
    /// wgpu validation or out-of-memory errors.
    pub fn run_frames(&mut self, blocks: &[&[u8]], sink: &mut (impl FrameSink + Send)) -> anyhow::Result<PipelineStats> {
        check_frame_blocks(blocks)?;
        self.stream_frames(
            |batch| {
                batch.frames().for_each(|(i, frame)| sink.put(i, frame));
                Ok(())
            },
            |stream| stream.upload_blocks(blocks),
        )
    }

    /// `run_frames` with each completed batch's frames handed to `sink` from `threads` delivery
    /// threads at once (`FrameBatch::deliver_par`), so that copying or writing the frames out
    /// takes about a `threads`-th of the time.
    pub fn run_frames_par(
        &mut self,
        blocks: &[&[u8]],
        sink: &impl ParFrameSink,
        threads: usize,
    ) -> anyhow::Result<PipelineStats> {
        check_frame_blocks(blocks)?;
        self.stream_frames(
            |batch| {
                batch.deliver_par(sink, threads);
                Ok(())
            },
            |stream| stream.upload_blocks(blocks),
        )
    }

    /// The streaming frame API, with no host copy on either side. `produce` runs on this thread:
    /// it takes each upload slot in turn (`FrameStream::next_upload_slot`), writes blocks into its
    /// mapped memory (`UploadSlot::regions_mut`, or the `unsafe` `UploadSlot::blocks_mut`) and
    /// submits them (`UploadSlot::submit` / `submit_with`, any number up to the slot's capacity,
    /// e.g. when a flush timer fires); it blocks only when every slot is in flight or still lent
    /// out. A completion thread waits for the batches in submission order and hands each one to
    /// `on_batch` as a `FrameBatch`, whose frames point into the slot's staging buffer; the slot
    /// is reused once the batch is dropped, which may happen on any thread. Blocks count from 0
    /// per stream. Returns once `produce` has returned, every submitted batch went to `on_batch`
    /// and every `FrameBatch` was dropped.
    ///
    /// An error from either side aborts the stream and is returned (the completion side's first);
    /// every later `FrameStream` call errors. After an error from `on_batch` no further batch is
    /// delivered; after one from `produce` (or a failed submit) the completion thread may still
    /// deliver a batch or two it had already taken before it sees the abort. A panic on either side
    /// also aborts the stream and is re-raised (not returned) once the pipeline is cleaned up. The
    /// pipeline stays usable either way.
    pub fn stream_frames<F, P>(&mut self, mut on_batch: F, produce: P) -> anyhow::Result<PipelineStats>
    where
        F: FnMut(FrameBatch) -> anyhow::Result<()> + Send,
        P: FnOnce(&mut FrameStream<'_>) -> anyhow::Result<()>,
    {
        anyhow::ensure!(self.layout.frames, "pipeline built without emit_frames: use run");
        let layout = self.layout;
        let handler = move |lease: Lease, first: usize, n: u32, tag: u64| {
            on_batch(FrameBatch::new(lease, first, n, tag, &layout)?)
        };
        self.stream_with(Box::new(handler), produce)
    }

    /// Runs `produce` on this thread and the completion thread (`Completion`) beside it, which
    /// owns `handler` and drops it when done (so a sink that forwards batches to its own threads
    /// can release them on the channel's close); collects the profile. A panic on either side
    /// is re-raised once the pipeline is cleaned up (`abandon`).
    fn stream_with<P>(&mut self, handler: Box<Handler<'_>>, produce: P) -> anyhow::Result<PipelineStats>
    where
        P: FnOnce(&mut FrameStream<'_>) -> anyhow::Result<()>,
    {
        let ctx = Arc::clone(&self.ctx);
        let scopes = ErrorScopes::push(&ctx);
        let start = Instant::now();
        let names = self.kernels.names();
        // End query of the last kernel recorded: K4 on the frame path (K5 runs before it), K3 on
        // the parse path.
        let last = if self.layout.frames { "k4_entropy" } else { "k3_parse" };
        #[cfg(test)]
        let fail_after = self.fail_deliveries_after.take();
        #[cfg(not(test))]
        let fail_after = None;
        let poll_only = self.poll_only();
        let completion = Completion {
            ctx: &ctx,
            xfer: self.xfer.as_ref().map(|x| (x.tq.clone(), x.t_done.clone())),
            shared: self.shared.clone(),
            layout: self.layout,
            n_kernels: names.len(),
            last_kernel_end: 2 * names.iter().position(|&k| k == last).expect("last kernel is timed") + 1,
            timed: ctx.timestamps,
            poll_only,
            fail_after,
        };
        let shared = self.shared.clone();
        shared.release_ns.store(0, Ordering::Relaxed);
        let (produced, completed) = std::thread::scope(|s| {
            let (tx, rx) = mpsc::channel();
            let worker = s.spawn(move || completion.run(rx, handler));
            let mut stream = FrameStream {
                pipe: &mut *self,
                tx,
                next_slot: 0,
                next_index: 0,
                batches: 0,
                start,
                prof: ProducerProfile::default(),
                submit_error: None,
            };
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| produce(&mut stream)));
            let FrameStream { tx, batches, prof, submit_error, .. } = stream;
            // Closing the channel ends the completion thread once it has drained the queue.
            drop(tx);
            // A failed submission fails the stream even when `produce` swallowed its error.
            let r = match (r, submit_error) {
                (Ok(Ok(())), Some(e)) => Ok(Err(anyhow!("a submission failed (produce carried on): {e}"))),
                (r, _) => r,
            };
            if !matches!(r, Ok(Ok(()))) {
                shared.abort();
            }
            (r.map(|r| r.map(|()| (batches, prof))), worker.join())
        });
        let (produced, (completed, cprof)) = match (produced, completed) {
            (Ok(p), Ok(c)) => (p, c),
            (p, c) => {
                self.abandon();
                std::panic::resume_unwind(p.err().or(c.err()).expect("one side panicked"))
            }
        };
        #[cfg(test)]
        {
            self.fail_deliveries_after = cprof.fail_after;
        }
        // A completion-side error also explains the producer's "stream aborted".
        let result = completed.and(produced);
        if result.is_err() {
            self.abandon();
        } else {
            self.shared.wait_released();
        }
        let end = Instant::now();
        let wall_s = (end - start).as_secs_f64();
        // A failed stream reports its own error first: the scope may hold one raised by
        // `abandon`'s cleanup, e.g. after a lost device (wgpu-core destroys every buffer, so the
        // unmap of an in-flight staging buffer fails), which must not mask the cause.
        let (batches, pprof) = match (result, scopes.pop()) {
            (Err(e), Err(scope)) => return Err(e.context(format!("stream failed (wgpu errors meanwhile: {scope:#})"))),
            (Err(e), Ok(())) => return Err(e),
            (Ok(_), Err(scope)) => return Err(scope),
            (Ok(r), Ok(())) => r,
        };

        // Timers whose timestamps the device did not write are left out, as without timestamps.
        let (kernel_ms, gpu_ms) = if self.ctx.timestamps {
            let period_ns = self.ctx.queue.get_timestamp_period() as f64;
            let ms = |t: u64| t as f64 * period_ns / 1e6;
            let mut kernel_ms: Vec<(String, f64)> = names
                .iter()
                .zip(&cprof.ticks)
                .zip(&cprof.ticks_bad)
                .filter(|(_, bad)| !**bad)
                .map(|((name, &t), _)| (name.to_string(), ms(t)))
                .collect();
            // K3t ran only for the batches holding a partial block.
            if cprof.trunc_batches > 0 && !cprof.trunc_bad {
                kernel_ms.push((TRUNC_KERNEL_NAME.to_string(), ms(cprof.trunc_ticks)));
            }
            (
                kernel_ms,
                if cprof.markers_bad {
                    Vec::new()
                } else {
                    vec![ms(cprof.upload_copy), ms(cprof.readback), ms(cprof.idle)]
                },
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let drain = cprof.last_wait.map_or(0.0, |t| (end - t).as_secs_f64());
        let release = self.shared.release_ns.load(Ordering::Relaxed) as f64 / 1e9;
        let host_ms = [
            pprof.upload_wait,
            pprof.upload_write,
            pprof.submit,
            cprof.wait,
            cprof.deliver,
            release,
            pprof.fill,
            drain,
        ]
        .map(|s| s * 1e3);
        let transfer_ms = TRANSFER_NAMES[..3]
            .iter()
            .zip(gpu_ms)
            .chain(TRANSFER_NAMES[3..].iter().zip(host_ms))
            .map(|(n, ms)| (n.to_string(), ms))
            .collect();
        Ok(PipelineStats { kernel_ms, wall_s, batches, transfer_ms })
    }

    /// Records the upload copy, the kernels and the readback of slot `i`'s first `n` blocks into
    /// the slot's staging buffer (the fixed-stride copies, or with the transfer
    /// readback the copies on the transfer queue), plus the resolved timestamps, submits them and
    /// requests the maps; returns the batch for the completion thread. The slot's upload buffer
    /// holds the blocks (mapped); it is persistent rather than `queue.write_buffer`, which would
    /// allocate a fresh staging buffer per call that lives until the submission completes.
    fn submit(&mut self, i: usize, first: usize, n: u32, tag: u64, lens: &[u32]) -> anyhow::Result<Job> {
        debug_assert_eq!(lens.len(), n as usize);
        // A batch holding a partial block runs K3t (`Kernels::record_truncate`) on the blocks'
        // real lengths; a batch of full blocks records exactly what it always did.
        let partial = self.layout.frames && lens.iter().any(|&l| l < BLOCK_SIZE as u32);
        if partial {
            // Applied at the start of this batch's first submission below, so after every earlier
            // batch's K3t read `lens`.
            let buf = self.bufs.lens.as_ref().expect("frame-path buffers have lens");
            self.ctx.queue.write_buffer(buf, 0, bytemuck::cast_slice(lens));
        }
        if self.direct {
            // The kernels read this slot's upload buffer; the bind groups recorded below hold it.
            self.bufs.data = self.slots[i].upload.clone();
        }
        let (ctx, layout, bufs, direct): (&GpuContext, _, _, _) = (&self.ctx, self.layout, &self.bufs, self.direct);
        let slot = &mut self.slots[i];
        let bytes = n as usize * BLOCK_SIZE;
        slot.upload.unmap();
        slot.upload_unmapped = true;
        // From here on the GPU may write the slot's staging buffer (`abandon` cleans up on error).
        self.shared.set(i, SlotState::InFlight);
        #[cfg(test)]
        match &mut self.fail_submits_after {
            Some(0) => {
                self.fail_submits_after = None;
                anyhow::bail!("injected submit failure (test hook)");
            }
            Some(k) => *k -= 1,
            None => {}
        }

        let n_queries = 2 * self.kernels.names().len() as u32;
        // The queries a batch resolves: the kernels' pairs, the two markers, and K3t's pair
        // (after the markers) only when the batch runs K3t: a batch of full blocks neither writes
        // nor resolves it.
        let n_resolve = n_queries + 2 + if partial { 2 } else { 0 };
        let resolved = n_resolve as u64 * wgpu::QUERY_SIZE as u64;
        let queries = slot.queries.as_ref().map(|q| &q.set);
        let ts_trunc = || {
            queries.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(n_queries + 2),
                end_of_pass_write_index: Some(n_queries + 3),
            })
        };
        // Start/end markers: empty passes whose timestamps (written at BOTTOM_OF_PIPE on Vulkan,
        // i.e. once all earlier commands completed) bracket the batch's copies.
        let marker = |enc: &mut wgpu::CommandEncoder, index: u32| {
            if let Some(set) = queries {
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("marker"),
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: set,
                        beginning_of_pass_write_index: Some(index),
                        end_of_pass_write_index: None,
                    }),
                });
            }
        };
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pipeline") });
        marker(&mut enc, n_queries);
        if !direct {
            enc.copy_buffer_to_buffer(&slot.upload, 0, &bufs.data, 0, bytes as u64 + 4);
        }
        let (submission, mapped, seq) = match (&mut self.xfer, &*slot.staging) {
            (Some(x), Staging::Host(staging)) => {
                let seq = x.next_seq;
                x.next_seq += 1;
                // K1–K3, then K5/K4 in a second submission that waits for the previous batch's
                // readback (it overwrites `frames`) and signals `k_done`.
                self.kernels.record_front(ctx, &mut enc, bufs, n, queries)?;
                // Through `submit_wgpu` too (nothing staged), so it holds the same lock as the
                // second submission and can never pick up semaphores staged for it.
                // SAFETY: nothing is staged.
                unsafe { x.tq.submit_wgpu(&ctx.queue, enc.finish(), None, None)? };
                let mut enc =
                    ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pipeline") });
                if partial {
                    self.kernels.record_truncate(ctx, &mut enc, bufs, n, ts_trunc());
                }
                self.kernels.record_entropy(ctx, &mut enc, bufs, n, queries);
                marker(&mut enc, n_queries + 1);
                if let Some(q) = &slot.queries {
                    enc.resolve_query_set(&q.set, 0..n_resolve, &q.resolve, 0);
                }
                // SAFETY: both timelines live in `xfer` until `Pipeline`'s Drop waited for the queues.
                let submission = unsafe {
                    x.tq.submit_wgpu(&ctx.queue, enc.finish(), Some((&x.t_done, seq - 1)), Some((&x.k_done, seq)))?
                };
                let mut copies = vec![
                    (x.frame_len.buffer, 0, staging.buffer, 0, frame_len_bytes(n)),
                    (x.frames.buffer, 0, staging.buffer, layout.a, frames_bytes(n)),
                ];
                if let Some(raw) = slot.queries.as_ref().and_then(|q| q.raw.as_ref()) {
                    copies.push((raw.buffer, 0, staging.buffer, layout.ts, resolved));
                }
                // SAFETY: slot i's command buffer is idle (the slot is free, so its last copy
                // signalled t_done); the sources and staging live in `xfer` / the slot until the
                // queues are idle; K4 and the resolve that write the sources are ordered before
                // `k_done = seq` by the submission above.
                unsafe { x.tq.copy(x.cmds.buffers[i], &copies, &x.k_done, seq, &x.t_done, seq)? };
                (submission, None, seq)
            }
            (None, Staging::Wgpu(staging)) => {
                // record_timed binds exactly counts_bytes(n) (K3) and frame_len_bytes(n) (K5, K4),
                // so no kernel processes the stale blocks of a partial batch.
                if partial {
                    self.kernels.record_front(ctx, &mut enc, bufs, n, queries)?;
                    self.kernels.record_truncate(ctx, &mut enc, bufs, n, ts_trunc());
                    self.kernels.record_entropy(ctx, &mut enc, bufs, n, queries);
                } else {
                    self.kernels.record_timed(ctx, &mut enc, bufs, n, queries)?;
                }
                if layout.frames {
                    let (frames, frame_len) = (bufs.frames.as_ref().unwrap(), bufs.frame_len.as_ref().unwrap());
                    enc.copy_buffer_to_buffer(frame_len, 0, staging, 0, frame_len_bytes(n));
                    enc.copy_buffer_to_buffer(frames, 0, staging, layout.a, frames_bytes(n));
                } else {
                    enc.copy_buffer_to_buffer(&bufs.counts, 0, staging, 0, counts_bytes(n));
                    enc.copy_buffer_to_buffer(&bufs.seqs, 0, staging, layout.a, seqs_bytes_for(n, &self.cfg.params.matching));
                }
                marker(&mut enc, n_queries + 1);
                if let Some(q) = &slot.queries {
                    enc.resolve_query_set(&q.set, 0..n_resolve, &q.resolve, 0);
                    enc.copy_buffer_to_buffer(&q.resolve, 0, staging, layout.ts, resolved);
                }
                let submission = ctx.queue.submit([enc.finish()]);
                let (tx, mapped) = mpsc::channel();
                staging.map_async(wgpu::MapMode::Read, .., move |r| {
                    let _ = tx.send(r);
                });
                (submission, Some(mapped), 0)
            }
            _ => unreachable!("host staging iff transfer readback"),
        };
        slot.staging_requested = mapped.is_some();
        let (tx, upload_mapped) = mpsc::channel();
        slot.upload.map_async(wgpu::MapMode::Write, .., move |r| {
            let _ = tx.send(r);
        });
        slot.upload_mapped = Some(upload_mapped);
        slot.upload_unmapped = false;
        slot.upload_submission = Some(submission.clone());
        Ok(Job { slot: i, first, n, tag, submission, mapped, seq, staging: slot.staging.clone(), partial })
    }

    /// After an error: wait for the GPU, unmap and free every slot still in flight so the
    /// pipeline stays usable, then wait for the sink to drop the batches it still holds.
    fn abandon(&mut self) {
        let _ = self.ctx.wait_idle(self.poll_only());
        if let Some(x) = &self.xfer {
            x.tq.idle();
            // A batch whose readback was never submitted leaves t_done behind (and one whose second
            // submission failed leaves k_done behind: `submit_wgpu` unstaged its signal): catch the
            // timelines up so the next batch's waits hold.
            // SAFETY: both queues are idle (waited for just above), so nothing signals them, and
            // nothing is left staged on the wgpu queue (`submit_wgpu` unstages on failure).
            unsafe {
                let _ = x.tq.catch_up(&x.k_done, x.next_seq - 1);
                let _ = x.tq.catch_up(&x.t_done, x.next_seq - 1);
            }
        }
        {
            let mut g = self.shared.lock();
            for (i, slot) in self.slots.iter_mut().enumerate() {
                if g.slots[i] == SlotState::InFlight {
                    if slot.staging_requested
                        && let Staging::Wgpu(staging) = &*slot.staging
                    {
                        staging.unmap();
                    }
                    slot.staging_requested = false;
                    g.slots[i] = SlotState::Free;
                }
            }
            g.abort = false;
        }
        self.shared.cv.notify_all();
        self.shared.wait_released();
    }
}

/// Bytes a slot owns: its upload buffer and its staging buffer (both mappable; counted although
/// drivers may place them in host memory).
fn per_slot_bytes(batch: u32, frames: bool, m: &MatchParams) -> u64 {
    data_bytes(batch) + StagingLayout::new(batch, frames, m).size
}

/// Device memory a `Pipeline` for `cfg` allocates: the shared buffers once (the scratch buffers,
/// their hash chain buffers sized by `cfg.params.matching`: one chain for single-hash presets, two
/// for Dfast and Opt3; the optimal parse's 8 B per position of candidates and of trace, its
/// larger `seqs` and K3opt's prices and scratch; plus `data` and, on the frame path, `frames` and
/// `frame_len`), per slot its upload and staging buffers, plus K4's constant tables. K5 has no buffers of its own. Uploads go
/// through the persistent upload buffers only, so no transient staging adds to this. Timestamp
/// query sets are not counted.
/// This is the copy-upload footprint, an upper bound for any context (`vram_bytes_with`).
pub fn vram_bytes(cfg: &PipelineConfig) -> u64 {
    vram_bytes_with(cfg, false)
}

/// `vram_bytes` for a context with (`GpuContext::direct_upload`) or without the direct upload,
/// which has no shared `data` buffer (−`data_bytes(batch)`).
pub fn vram_bytes_with(cfg: &PipelineConfig, direct_upload: bool) -> u64 {
    let frames = cfg.params.emit_frames;
    scratch_bytes(cfg.batch, &cfg.params.matching) + slot_bytes(cfg.batch, frames)
        - if direct_upload { data_bytes(cfg.batch) } else { 0 }
        + cfg.inflight as u64 * per_slot_bytes(cfg.batch, frames, &cfg.params.matching)
        + if frames { k4_tables_bytes() } else { 0 }
}
