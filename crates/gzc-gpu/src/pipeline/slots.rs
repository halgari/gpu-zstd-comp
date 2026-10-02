//! A pipeline's slots: their staging buffers, the states the producer and the completion thread
//! share, and the batches lent out of them (`Lease`, `FrameBatch`).
use super::*;

/// Timestamp queries per slot: the kernels' begin/end pairs, then a start and an end marker, then
/// K3t's pair, written and resolved only by a batch that holds a partial block (see
/// `Pipeline::submit`).
pub(super) const PIPELINE_QUERIES: u32 = KERNEL_QUERIES + 4;

/// Alignment of each region of a slot's staging buffer (see `StagingLayout::new`).
pub(super) const STAGING_ALIGN: u64 = 256;

/// Byte offsets of one slot's staging buffer, laid out for `cap` blocks. Parse path:
/// `[counts][seqs][timestamps]` (`seqs` of `max_seqs(m)` per block); frame path:
/// `[frame_len][frames][timestamps]`; the regions keep the GPU buffers' fixed per-block stride.
#[derive(Clone, Copy)]
pub(super) struct StagingLayout {
    pub(super) frames: bool,
    /// Parse path: seqs region. Frame path: frames region.
    pub(super) a: u64,
    pub(super) ts: u64,
    pub(super) size: u64,
}

impl StagingLayout {
    pub(super) fn new(cap: u32, frames: bool, m: &MatchParams) -> Self {
        // Every region starts STAGING_ALIGN-aligned: a GPU->staging copy to a destination that is
        // only 4-byte aligned runs several times slower (RTX 5090 / Vulkan: 6-10 ms more per
        // batch for the ~200 MB frames region).
        let al = |x: u64| x.next_multiple_of(STAGING_ALIGN);
        let (a, ts) = if frames {
            let a = al(frame_len_bytes(cap));
            (a, al(a + frames_bytes(cap)))
        } else {
            let a = al(counts_bytes(cap));
            (a, al(a + seqs_bytes_for(cap, m)))
        };
        Self { frames, a, ts, size: ts + PIPELINE_QUERIES as u64 * wgpu::QUERY_SIZE as u64 }
    }
}

/// A slot's staging buffer: mapped through wgpu after the submission, or (transfer readback)
/// host memory the transfer queue writes and that stays mapped. Shared between the slot, the
/// completion thread and the batch lent out of it (`Lease`).
pub(super) enum Staging {
    Wgpu(wgpu::Buffer),
    Host(RawBuffer),
}

#[cfg(test)]
impl Staging {
    pub(super) fn size(&self) -> u64 {
        match self {
            Staging::Wgpu(b) => b.size(),
            Staging::Host(b) => b.size,
        }
    }
}

/// A slot's timestamp query set and the buffer it resolves into (transfer readback: shared with
/// the transfer queue, `raw` keeps it alive).
pub(super) struct Queries {
    pub(super) set: wgpu::QuerySet,
    pub(super) resolve: wgpu::Buffer,
    pub(super) raw: Option<RawBuffer>,
}

/// The producer's side of a slot; the staging side's state lives in `Shared`.
pub(super) struct Slot {
    /// Persistent upload buffer (MAP_WRITE | COPY_SRC, `data_bytes(batch)`): the producer writes a
    /// batch into it while it is mapped, and the submission copies it into the shared `data` (direct
    /// upload: MAP_WRITE | STORAGE, bound as `data` itself).
    /// After each submission it is re-mapped; `upload_mapped` receives that map's result (None:
    /// mapped, or `upload_unmapped`).
    pub(super) upload: wgpu::Buffer,
    pub(super) upload_mapped: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    /// Unmapped with no re-map requested (a submission failed in between): mapped on next use.
    pub(super) upload_unmapped: bool,
    /// The submission that last used `upload` (to wait for its re-map).
    pub(super) upload_submission: Option<wgpu::SubmissionIndex>,
    pub(super) staging: Arc<Staging>,
    /// A wgpu staging map was requested since the slot was last free (`abandon` unmaps it).
    pub(super) staging_requested: bool,
    /// Query set and its resolve buffer, when timestamps are enabled.
    pub(super) queries: Option<Queries>,
}

/// Where a slot's staging buffer is: free for the next submission, written by an in-flight batch,
/// or lent out (`Lease`, e.g. inside a `FrameBatch`) until the sink drops it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SlotState {
    Free,
    InFlight,
    Leased,
}

/// Slot states shared by the producer (the thread in `run*` / `stream_frames`), the completion
/// thread and the leases, which may be dropped on any thread.
pub(super) struct Shared {
    pub(super) state: Mutex<SharedState>,
    pub(super) cv: Condvar,
    /// Nanoseconds spent releasing leases (unmap), for `host_unmap`.
    pub(super) release_ns: AtomicU64,
}

pub(super) struct SharedState {
    pub(super) slots: Vec<SlotState>,
    /// Set when either side of a stream failed: the producer stops waiting for slots and the
    /// completion thread stops delivering.
    pub(super) abort: bool,
}

impl Shared {
    pub(super) fn new(slots: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(SharedState { slots: vec![SlotState::Free; slots], abort: false }),
            cv: Condvar::new(),
            release_ns: AtomicU64::new(0),
        })
    }

    pub(super) fn lock(&self) -> std::sync::MutexGuard<'_, SharedState> {
        // No critical section below can leave the state inconsistent.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(super) fn set(&self, slot: usize, to: SlotState) {
        self.lock().slots[slot] = to;
        self.cv.notify_all();
    }

    pub(super) fn abort(&self) {
        self.lock().abort = true;
        self.cv.notify_all();
    }

    pub(super) fn aborted(&self) -> bool {
        self.lock().abort
    }

    /// Blocks until `slot` is free; errors once the stream is aborted.
    pub(super) fn wait_free(&self, slot: usize) -> anyhow::Result<()> {
        let mut g = self.lock();
        loop {
            anyhow::ensure!(!g.abort, "stream aborted");
            if g.slots[slot] == SlotState::Free {
                return Ok(());
            }
            g = self.wait_logged(g, "a free upload slot");
        }
    }

    /// Blocks until no slot is lent out.
    pub(super) fn wait_released(&self) {
        let mut g = self.lock();
        while g.slots.contains(&SlotState::Leased) {
            g = self.wait_logged(g, "the stream's end");
        }
    }

    /// One condvar wait; a wait of `STALL_WARN` or longer logs how many batches the sink still
    /// holds, so a leaked or over-held `FrameBatch` shows up instead of a silent hang.
    pub(super) fn wait_logged<'g>(
        &self,
        g: std::sync::MutexGuard<'g, SharedState>,
        what: &str,
    ) -> std::sync::MutexGuard<'g, SharedState> {
        const STALL_WARN: std::time::Duration = std::time::Duration::from_secs(10);
        let (g, timeout) = self.cv.wait_timeout(g, STALL_WARN).unwrap_or_else(|e| e.into_inner());
        if timeout.timed_out() {
            let held = g.slots.iter().filter(|&&s| s == SlotState::Leased).count();
            eprintln!(
                "gzc: pipeline waiting {}s+ for {what}: {held} FrameBatch(es) still held by the sink, {} batch(es) in flight",
                STALL_WARN.as_secs(),
                g.slots.iter().filter(|&&s| s == SlotState::InFlight).count()
            );
        }
        g
    }
}

/// Aborts the stream if the completion thread unwinds, so the producer never waits for a slot
/// that will not come free.
pub(super) struct AbortOnPanic<'s>(pub(super) &'s Shared);

impl Drop for AbortOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.abort();
        }
    }
}

/// A completed batch's staging bytes, lent out until dropped: the slot takes no new batch before
/// that. Dropping it unmaps a wgpu staging buffer and frees the slot.
pub(super) struct Lease {
    pub(super) shared: Arc<Shared>,
    pub(super) slot: usize,
    pub(super) staging: Arc<Staging>,
    /// wgpu staging: its mapped range, which `ptr`/`len` point into.
    pub(super) view: Option<wgpu::BufferView>,
    pub(super) ptr: *const u8,
    pub(super) len: usize,
}

// SAFETY: the bytes are immutable while the lease lives (the slot is not resubmitted, so neither
// the GPU nor the transfer queue writes them, and wgpu keeps the range mapped); the other fields
// are `Send + Sync`.
unsafe impl Send for Lease {}
unsafe impl Sync for Lease {}

impl Lease {
    pub(super) fn bytes(&self) -> &[u8] {
        // SAFETY: see the `Send` impl; `ptr`/`len` cover the mapping, alive as long as `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let t = Instant::now();
        self.view.take();
        // wgpu error scopes are thread-local: the stream's scopes (`stream_with`) only see the
        // producer thread, so an error from this unmap (on the completion thread or a writer
        // thread) goes to the device's uncaptured-error handler. Unmapping a mapped buffer raises
        // none.
        if let Staging::Wgpu(b) = &*self.staging {
            b.unmap();
        }
        self.shared.release_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.shared.set(self.slot, SlotState::Free);
    }
}

/// A completed batch of zstd frames lent to the caller (`Pipeline::stream_frames`) without a copy:
/// each frame points into the slot's staging buffer. The slot takes a new batch only once this is
/// dropped, so a sink may hand it to writer threads (it is `Send + Sync`) and drop it when their
/// writes are done; the producer blocks in `FrameStream::next_upload_slot` meanwhile. Holding
/// `inflight` batches stalls the stream (every slot is lent out); holding `inflight - 1` keeps it
/// going but serialises it (one batch in flight at a time), so release batches promptly. A
/// batch never released makes `stream_frames` wait forever at its end (with a warning every 10 s).
pub struct FrameBatch {
    lease: Lease,
    first: usize,
    tag: u64,
    /// Per frame: byte offset in the staging buffer, and length.
    spans: Vec<(usize, u32)>,
}

impl FrameBatch {
    /// Checks every frame length (1..=FRAME_STRIDE) and locates the frames, at their fixed stride.
    pub(super) fn new(lease: Lease, first: usize, n: u32, tag: u64, layout: &StagingLayout) -> anyhow::Result<Self> {
        let bytes = lease.bytes();
        let lens: &[u32] = bytemuck::cast_slice(&bytes[..frame_len_bytes(n) as usize]);
        let mut spans = Vec::with_capacity(n as usize);
        for (b, &len) in lens.iter().enumerate() {
            anyhow::ensure!(len > 0 && len as usize <= FRAME_STRIDE, "block {}: bad frame length {len}", first + b);
            spans.push((layout.a as usize + b * FRAME_STRIDE, len));
        }
        Ok(Self { lease, first, tag, spans })
    }

    /// Index of the batch's first block in the stream (blocks count from 0 per stream).
    pub fn first_index(&self) -> usize {
        self.first
    }

    /// The tag the producer submitted the batch with (`UploadSlot::submit_with`; 0 for
    /// `submit`), e.g. an index into its own table of the batch's payloads.
    pub fn tag(&self) -> u64 {
        self.tag
    }

    /// Number of frames (blocks) in the batch.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// True for a batch of no frames. A stream never delivers one.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Frame `k` of the batch (block `first_index() + k`).
    pub fn frame(&self, k: usize) -> &[u8] {
        let (at, len) = self.spans[k];
        &self.lease.bytes()[at..at + len as usize]
    }

    /// `(block index, frame)` for every frame, in index order.
    pub fn frames(&self) -> impl ExactSizeIterator<Item = (usize, &[u8])> + '_ {
        (0..self.len()).map(|k| (self.first + k, self.frame(k)))
    }

    /// Hands every frame to `sink` from `threads` threads (this one included), each taking a
    /// contiguous share of the batch.
    pub fn deliver_par(&self, sink: &impl ParFrameSink, threads: usize) {
        let n = self.len();
        let per = n.div_ceil(threads.clamp(1, n.max(1))).max(1);
        std::thread::scope(|s| {
            let mut shares = (0..n).step_by(per).map(|k| k..(k + per).min(n));
            let mine = shares.next();
            for share in shares {
                s.spawn(move || share.for_each(|k| sink.put(self.first + k, self.frame(k))));
            }
            mine.into_iter().flatten().for_each(|k| sink.put(self.first + k, self.frame(k)));
        });
    }
}
