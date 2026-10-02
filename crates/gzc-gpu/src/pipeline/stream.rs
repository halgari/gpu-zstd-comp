//! The producer's side of a stream: upload slots, their write-only regions and the block copy.
use super::*;

/// The producer's side of a stream (`Pipeline::stream_frames`): hands out the slots' mapped upload
/// buffers in turn and submits them. Block indices count from 0 per stream, in submission order.
pub struct FrameStream<'p> {
    pub(super) pipe: &'p mut Pipeline,
    pub(super) tx: mpsc::Sender<Job>,
    pub(super) next_slot: usize,
    pub(super) next_index: usize,
    pub(super) batches: u32,
    pub(super) start: Instant,
    pub(super) prof: ProducerProfile,
    /// The first failed submission's error: the stream fails even if `produce` swallows it.
    pub(super) submit_error: Option<String>,
}

impl<'p> FrameStream<'p> {
    /// Blocks per upload slot (the pipeline's batch size).
    pub fn slot_capacity(&self) -> usize {
        self.pipe.cfg.batch as usize
    }

    /// Blocks submitted so far: the index the next submitted block gets.
    pub fn submitted_blocks(&self) -> usize {
        self.next_index
    }

    /// The next upload slot, once it is free: the batch it held `inflight` submissions ago has
    /// completed and its `FrameBatch` was dropped. Errors once the stream is aborted (the
    /// completion side failed) and on a failed map.
    pub fn next_upload_slot(&mut self) -> anyhow::Result<UploadSlot<'_, 'p>> {
        let i = self.next_slot;
        let t = Instant::now();
        self.pipe.shared.wait_free(i)?;
        let poll_only = self.pipe.poll_only();
        let ctx: &GpuContext = &self.pipe.ctx;
        let slot = &mut self.pipe.slots[i];
        slot.staging_requested = false;
        if slot.upload_unmapped {
            let (tx, rx) = mpsc::channel();
            slot.upload.map_async(wgpu::MapMode::Write, .., move |r| {
                let _ = tx.send(r);
            });
            slot.upload_mapped = Some(rx);
            slot.upload_unmapped = false;
            slot.upload_submission = None;
        }
        if let Some(rx) = slot.upload_mapped.take() {
            // Its submission has completed (the slot is free), so the callback is normally in. On
            // Metal (`poll_only`) the wait polls without blocking, as the completion thread does:
            // a `PollType::Wait` here would run beside the completion thread's polls and, with a
            // shared context, other threads' submits.
            let r = ctx
                .wait_callback(&rx, slot.upload_submission.clone(), poll_only)
                .context("waiting for the upload buffer's map")?;
            if r.is_err() {
                slot.upload_unmapped = true;
            }
            r.context("map upload buffer")?;
        }
        let view = slot.upload.get_mapped_range_mut(..).context("upload mapped range")?;
        self.prof.upload_wait += t.elapsed().as_secs_f64();
        let lens = vec![BLOCK_SIZE as u32; self.slot_capacity()];
        Ok(UploadSlot { stream: self, slot: i, view, lens, acquired: Instant::now() })
    }

    /// Copies `blocks` into as many slots as they need and submits them: the `&[&[u8]]` form of
    /// the API. The copy is split over `GpuOptions::upload_threads` threads. On the frame path a block is
    /// its real bytes (1..=BLOCK_SIZE, zero-padded in the slot, see `run_frames`); on the parse
    /// path every block is BLOCK_SIZE bytes.
    pub fn upload_blocks(&mut self, blocks: &[&[u8]]) -> anyhow::Result<()> {
        if self.pipe.layout.frames { check_frame_blocks(blocks)? } else { check_blocks(blocks)? }
        let threads = self.pipe.upload_threads;
        for chunk in blocks.chunks(self.slot_capacity()) {
            let mut slot = self.next_upload_slot()?;
            let region = slot.regions_mut(&[chunk.len()])?.pop().expect("one region");
            copy_blocks(region, chunk, threads);
            for (k, b) in chunk.iter().enumerate() {
                slot.set_real_len(k, b.len())?;
            }
            slot.submit(chunk.len())?;
        }
        Ok(())
    }

    /// Submits slot `i`'s first `n` blocks. Any failure aborts the stream: a slot whose
    /// submission failed after it was marked in flight would never come free, so every later
    /// call must error rather than wait for it.
    pub(super) fn submit(&mut self, i: usize, n: u32, tag: u64, lens: &[u32]) -> anyhow::Result<()> {
        let t = Instant::now();
        let job = match self.pipe.submit(i, self.next_index, n, tag, lens) {
            Ok(job) => job,
            Err(e) => {
                self.pipe.shared.abort();
                self.submit_error.get_or_insert_with(|| format!("{e:#}"));
                return Err(e);
            }
        };
        let sent = self.tx.send(job);
        self.prof.submit += t.elapsed().as_secs_f64();
        if self.batches == 0 {
            self.prof.fill = self.start.elapsed().as_secs_f64();
        }
        self.batches += 1;
        self.next_index += n as usize;
        self.next_slot = (i + 1) % self.pipe.slots.len();
        sent.map_err(|_| {
            self.pipe.shared.abort();
            anyhow!("stream aborted: the completion thread stopped")
        })
    }
}

/// A free slot's mapped upload buffer (`FrameStream::next_upload_slot`): write up to `capacity()`
/// blocks into it (`regions_mut`, or the `unsafe` `blocks_mut`), then `submit` the first `n`.
/// Dropped without submitting, the slot is handed out again by the next `next_upload_slot`.
///
/// The memory is mapped MAP_WRITE memory: device-local write-combined memory with the direct
/// upload (ReBAR), and possibly uncached / write-combined host memory with the copy upload too.
/// Write it sequentially and never read it back. A decompressor reads its own recent output for
/// its matches, so do not decode into the slot: decode into a cached buffer (or a streaming
/// decoder's window) and copy the result in.
#[must_use = "an upload slot is wasted unless submitted"]
pub struct UploadSlot<'s, 'p> {
    stream: &'s mut FrameStream<'p>,
    slot: usize,
    view: wgpu::BufferViewMut,
    /// Each block's real length (`set_real_len`, `Region::pad`), BLOCK_SIZE until set.
    lens: Vec<u32>,
    acquired: Instant,
}

impl UploadSlot<'_, '_> {
    /// Blocks the slot holds (the pipeline's batch size).
    pub fn capacity(&self) -> usize {
        self.stream.slot_capacity()
    }

    /// The slot's upload memory as a plain `&mut [u8]`: `capacity()` blocks of BLOCK_SIZE bytes,
    /// block k at `k * BLOCK_SIZE` (split it with `chunks_mut` to fill it from several threads).
    /// Every byte of each submitted block must be written: the bytes are whatever an earlier
    /// batch left there. Prefer the safe `regions_mut`; this is for code that needs a slice (a
    /// `Read::read` into it, say). Never read it: see the type's docs.
    ///
    /// # Safety
    ///
    /// wgpu hands out mapped-for-write memory only as `wgpu::WriteOnly`, whose contract says its
    /// pointer must not be turned into a `&mut` or read. This does both (a `&mut [u8]` may be
    /// read), which is sound only because of wgpu-core implementation details: its native
    /// backends map a real host pointer that stays valid until `unmap`, and they zero-fill a
    /// MAP_WRITE range that was never written when mapping it, so every byte is initialized (the
    /// rest holds earlier batches' bytes); `upload_slot_bytes_are_initialized` pins this. The
    /// caller must not use the slice to read uninitialized memory on any other wgpu backend, and
    /// this must be rechecked on every wgpu upgrade.
    pub unsafe fn blocks_mut(&mut self) -> &mut [u8] {
        let len = self.capacity() * BLOCK_SIZE;
        let mut w = self.view.slice(..len);
        // SAFETY: the view maps at least `len` bytes (`data_bytes(batch)`), which stay mapped and
        // are ours alone while `self` is borrowed (only `submit` drops the view); initialized per
        // the caller's contract above.
        unsafe { &mut *w.as_raw_ptr().as_ptr() }
    }

    /// Block `k`'s BLOCK_SIZE bytes of `blocks_mut`.
    ///
    /// # Safety
    ///
    /// As for `blocks_mut`.
    pub unsafe fn block_mut(&mut self, k: usize) -> &mut [u8] {
        // SAFETY: the caller's contract.
        unsafe { &mut self.blocks_mut()[k * BLOCK_SIZE..(k + 1) * BLOCK_SIZE] }
    }

    /// Splits the slot's memory, from block 0, into consecutive disjoint write-only regions of
    /// `blocks[i]` whole blocks each: one per payload (a file, or a decompressed chunk of one),
    /// which several threads can fill at once (`Region` is `Send`). Finish each payload with
    /// `Region::pad`; submit the sum of the blocks. Errors if the regions exceed `capacity()`.
    pub fn regions_mut(&mut self, blocks: &[usize]) -> anyhow::Result<Vec<Region<'_>>> {
        let (total, cap) = (blocks.iter().sum::<usize>(), self.capacity());
        anyhow::ensure!(total <= cap, "regions of {total} blocks exceed the slot's {cap}");
        let mut rest = Region { bytes: self.view.slice(..total * BLOCK_SIZE), lens: &mut self.lens[..total] };
        let mut out = Vec::with_capacity(blocks.len());
        for &b in blocks {
            let (region, tail) = rest.split_at(b * BLOCK_SIZE);
            out.push(region);
            rest = tail;
        }
        Ok(out)
    }

    /// Sets block `k`'s real length to `len` (1..=BLOCK_SIZE; every block starts at BLOCK_SIZE):
    /// its frame then declares and holds only the block's first `len` bytes. For blocks written
    /// through `blocks_mut`; the rest of the block must be zero, as `chunk_file` pads a file's last
    /// block. `Region::pad` sets it for a region's payload.
    pub fn set_real_len(&mut self, k: usize, len: usize) -> anyhow::Result<()> {
        let cap = self.capacity();
        anyhow::ensure!(k < cap, "block {k} is past the slot's {cap}");
        anyhow::ensure!((1..=BLOCK_SIZE).contains(&len), "a block's real length is 1..={BLOCK_SIZE}, not {len}");
        self.lens[k] = len as u32;
        Ok(())
    }

    /// Submits the slot's first `n` blocks (1..=capacity(): a partial batch, e.g. when a flush
    /// timer fires, is fine); returns the index of its first block. Tag 0 (see `submit_with`).
    pub fn submit(self, n: usize) -> anyhow::Result<usize> {
        self.submit_with(n, 0)
    }

    /// `submit`, tagging the batch with `tag`, which its `FrameBatch::tag` returns: e.g. an index
    /// into the producer's table of which payloads (and real lengths) the batch holds.
    pub fn submit_with(self, n: usize, tag: u64) -> anyhow::Result<usize> {
        let cap = self.capacity();
        anyhow::ensure!((1..=cap).contains(&n), "cannot submit {n} blocks: not in 1..={cap}");
        let UploadSlot { stream, slot, mut view, lens, acquired } = self;
        // Trailing zero word after the last block (the slot may hold stale blocks beyond it).
        view.slice(n * BLOCK_SIZE..n * BLOCK_SIZE + 4).copy_from_slice(&[0u8; 4]);
        drop(view);
        stream.prof.upload_write += acquired.elapsed().as_secs_f64();
        let first = stream.next_index;
        stream.submit(slot, n as u32, tag, &lens[..n])?;
        Ok(first)
    }
}

/// A write-only piece of an upload slot (`UploadSlot::regions_mut`), a whole number of blocks.
pub struct Region<'a> {
    bytes: wgpu::WriteOnly<'a, [u8]>,
    /// The real lengths of the region's blocks (`UploadSlot::lens`).
    lens: &'a mut [u32],
}

// SAFETY: `WriteOnly<[u8]>` lacks `Send` only because wgpu's impl needs a sized `T`; like a
// `&mut [u8]`, a byte range of it may move to another thread, and `regions_mut` hands out
// disjoint ranges.
unsafe impl Send for Region<'_> {}

impl<'a> Region<'a> {
    /// Bytes in the region.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Writes `bytes` at `offset` (panics past the end).
    pub fn write(&mut self, offset: usize, bytes: &[u8]) {
        self.bytes.slice(offset..offset + bytes.len()).copy_from_slice(bytes);
    }

    /// The region as wgpu's `WriteOnly`, for writers that take one.
    pub fn write_only(&mut self) -> wgpu::WriteOnly<'_, [u8]> {
        self.bytes.slice(..)
    }

    /// Finishes a `len`-byte payload written at the start of the region: zero-fills the rest of
    /// its last block (the padding `chunk_file` gives a file's last block; the GPU parses the
    /// padded block), records the blocks' real lengths (`payload_real_lens(len)`), so the last
    /// block's frame declares and holds only its real bytes, and returns the blocks the payload
    /// occupies (`payload_blocks(len)`). Errors if they do not fit in the region.
    #[must_use = "the returned block count is what to submit"]
    pub fn pad(&mut self, len: usize) -> anyhow::Result<usize> {
        let blocks = payload_blocks(len);
        let end = blocks * BLOCK_SIZE;
        anyhow::ensure!(
            end <= self.len(),
            "a {len}-byte payload needs {blocks} blocks, the region has {}",
            self.len() / BLOCK_SIZE
        );
        self.bytes.slice(len..end).fill(0);
        for (l, real) in self.lens.iter_mut().zip(payload_real_lens(len)) {
            *l = real as u32;
        }
        Ok(blocks)
    }

    /// Splits at byte `mid`, a multiple of BLOCK_SIZE.
    pub(super) fn split_at(self, mid: usize) -> (Region<'a>, Region<'a>) {
        debug_assert_eq!(mid % BLOCK_SIZE, 0);
        let (a, b) = self.bytes.split_at(mid);
        let (la, lb) = self.lens.split_at_mut(mid / BLOCK_SIZE);
        (Region { bytes: a, lens: la }, Region { bytes: b, lens: lb })
    }
}

/// Blocks a payload of `len` bytes occupies, as `gzc_core::block::chunk_file` splits a file (0
/// for an empty one).
pub fn payload_blocks(len: usize) -> usize {
    len.div_ceil(BLOCK_SIZE)
}

/// The real length (`Block::real_len`) of each block of a `len`-byte payload: BLOCK_SIZE, and the
/// rest for the last one.
pub fn payload_real_lens(len: usize) -> impl ExactSizeIterator<Item = usize> {
    (0..payload_blocks(len)).map(move |k| (len - k * BLOCK_SIZE).min(BLOCK_SIZE))
}

/// Every block must be exactly BLOCK_SIZE bytes.
pub(crate) fn check_blocks(blocks: &[&[u8]]) -> anyhow::Result<()> {
    if let Some(i) = blocks.iter().position(|b| b.len() != BLOCK_SIZE) {
        anyhow::bail!("block {i} is {} bytes, expected BLOCK_SIZE {BLOCK_SIZE}", blocks[i].len());
    }
    Ok(())
}

/// Every block must hold 1..=BLOCK_SIZE bytes (its real bytes; the frame path).
pub(super) fn check_frame_blocks(blocks: &[&[u8]]) -> anyhow::Result<()> {
    if let Some(i) = blocks.iter().position(|b| !(1..=BLOCK_SIZE).contains(&b.len())) {
        anyhow::bail!("block {i} is {} bytes, expected 1..=BLOCK_SIZE ({BLOCK_SIZE})", blocks[i].len());
    }
    Ok(())
}

/// Copies `blocks` back to back into `dst`, split over `threads` threads (this one
/// included): one thread's stores into the (write-combined, ReBAR) upload buffer run at ~18 GB/s,
/// two or more at the link's ~26 GB/s (RTX 5090).
pub(super) fn copy_blocks(dst: Region<'_>, blocks: &[&[u8]], threads: usize) {
    let copy = |mut dst: Region<'_>, src: &[&[u8]]| {
        for (k, b) in src.iter().enumerate() {
            dst.write(k * BLOCK_SIZE, b);
            // A short (partial) block is zero-padded, as `chunk_file` pads it.
            dst.bytes.slice(k * BLOCK_SIZE + b.len()..(k + 1) * BLOCK_SIZE).fill(0);
        }
    };
    let per = blocks.len().div_ceil(threads.max(1)).max(64);
    std::thread::scope(|s| {
        let mut rest = dst;
        let mut shares = Vec::new();
        for src in blocks.chunks(per) {
            let (mine, tail) = rest.split_at(src.len() * BLOCK_SIZE);
            shares.push((mine, src));
            rest = tail;
        }
        let mut shares = shares.into_iter();
        let first = shares.next();
        for (dst, src) in shares {
            s.spawn(move || copy(dst, src));
        }
        if let Some((dst, src)) = first {
            copy(dst, src);
        }
    });
}

/// Threads writing a batch into its upload buffer: `GpuOptions::upload_threads`, else 4 (at most
/// the available parallelism).
pub(super) fn upload_threads(ctx: &GpuContext) -> usize {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    ctx.opts.upload_threads.unwrap_or(4).clamp(1, cores.max(1))
}
