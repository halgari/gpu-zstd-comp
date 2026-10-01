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
//! Transfer readback (`GpuContext::transfer`, frame path without packing; see `Xfer`): each batch
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
//!   `frames` region, either as a copy of the fixed-stride buffer or, with `GZC_PACK` (see
//!   `PackKernel`), packed by a kernel that writes the frames contiguously straight into the
//!   (mappable) staging buffer; the sink reads each frame in place (`FrameBatch`). K5 (the literals
//!   section, Huffman-coded with `GpuParams::huffman`) gathers the literals from `data` and writes
//!   into the same `frames` / `frame_len` buffers, so it adds no memory.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::Instant;

use anyhow::{Context as _, anyhow};
use gzc_core::config::BLOCK_SIZE;
use gzc_core::seq::BlockOutput;

use crate::compressor::{
    BatchBuffers, FRAME_STRIDE, GpuParams, KERNEL_QUERIES, Kernels, counts_bytes, data_bytes, decode_output,
    frame_len_bytes, frames_bytes, k4_tables_bytes, max_batch_blocks, max_seqs, scratch_bytes, seqs_bytes_for,
    slot_bytes,
};
use gzc_core::params::MatchParams;
use crate::context::GpuContext;
use crate::transfer::{Commands, RawBuffer, StreamingGuard, Timeline, TransferQueue};

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
    /// "k4_entropy" on the frame path and "k5_huffman" with Huffman literals), in milliseconds;
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
/// - `gpu_readback`: the output -> staging copies, or the pack kernel (K4's end, or K3's on
///   the parse path, to the batch's end marker).
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

/// Timestamp queries per slot: the kernels' begin/end pairs, then a start and an end marker (see
/// `Pipeline::submit`).
const PIPELINE_QUERIES: u32 = KERNEL_QUERIES + 2;

/// Alignment of each region of a slot's staging buffer (see `StagingLayout::new`).
const STAGING_ALIGN: u64 = 256;

/// Byte offsets of one slot's staging buffer, laid out for `cap` blocks. Parse path:
/// `[counts][seqs][timestamps]` (`seqs` of `max_seqs(m)` per block); frame path:
/// `[frame_len][frames][timestamps]`; the regions keep the GPU buffers' fixed per-block stride
/// (packed frames need at most that much).
#[derive(Clone, Copy)]
struct StagingLayout {
    frames: bool,
    /// Parse path: seqs region. Frame path: frames region.
    a: u64,
    ts: u64,
    size: u64,
}

impl StagingLayout {
    fn new(cap: u32, frames: bool, m: &MatchParams) -> Self {
        // Every region starts STAGING_ALIGN-aligned: a GPU->staging copy to a destination that is
        // only 4-byte aligned runs several times slower (RTX 5090 / Vulkan: 6-10 ms more per
        // batch for the ~200 MB frames region), which showed up as a batch-size-dependent loss.
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
}

/// A slot's staging buffer: mapped through wgpu after the submission, or (transfer readback)
/// host memory the transfer queue writes and that stays mapped. Shared between the slot, the
/// completion thread and the batch lent out of it (`Lease`).
enum Staging {
    Wgpu(wgpu::Buffer),
    Host(RawBuffer),
}

#[cfg(test)]
impl Staging {
    fn size(&self) -> u64 {
        match self {
            Staging::Wgpu(b) => b.size(),
            Staging::Host(b) => b.size,
        }
    }
}

/// A slot's timestamp query set and the buffer it resolves into (transfer readback: shared with
/// the transfer queue, `raw` keeps it alive).
struct Queries {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    raw: Option<RawBuffer>,
}

/// The producer's side of a slot; the staging side's state lives in `Shared`.
struct Slot {
    /// Persistent upload buffer (MAP_WRITE | COPY_SRC, `data_bytes(batch)`): the producer writes a
    /// batch into it while it is mapped, and the submission copies it into the shared `data` (direct
    /// upload: MAP_WRITE | STORAGE, bound as `data` itself).
    /// After each submission it is re-mapped; `upload_mapped` receives that map's result (None:
    /// mapped, or `upload_unmapped`).
    upload: wgpu::Buffer,
    upload_mapped: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    /// Unmapped with no re-map requested (a submission failed in between): mapped on next use.
    upload_unmapped: bool,
    /// The submission that last used `upload` (to wait for its re-map).
    upload_submission: Option<wgpu::SubmissionIndex>,
    staging: Arc<Staging>,
    /// A wgpu staging map was requested since the slot was last free (`abandon` unmaps it).
    staging_requested: bool,
    /// Query set and its resolve buffer, when timestamps are enabled.
    queries: Option<Queries>,
}

/// Where a slot's staging buffer is: free for the next submission, written by an in-flight batch,
/// or lent out (`Lease`, e.g. inside a `FrameBatch`) until the sink drops it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    Free,
    InFlight,
    Leased,
}

/// Slot states shared by the producer (the thread in `run*` / `stream_frames`), the completion
/// thread and the leases, which may be dropped on any thread.
struct Shared {
    state: Mutex<SharedState>,
    cv: Condvar,
    /// Nanoseconds spent releasing leases (unmap), for `host_unmap`.
    release_ns: AtomicU64,
}

struct SharedState {
    slots: Vec<SlotState>,
    /// Set when either side of a stream failed: the producer stops waiting for slots and the
    /// completion thread stops delivering.
    abort: bool,
}

impl Shared {
    fn new(slots: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(SharedState { slots: vec![SlotState::Free; slots], abort: false }),
            cv: Condvar::new(),
            release_ns: AtomicU64::new(0),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SharedState> {
        // No critical section below can leave the state inconsistent.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn set(&self, slot: usize, to: SlotState) {
        self.lock().slots[slot] = to;
        self.cv.notify_all();
    }

    fn abort(&self) {
        self.lock().abort = true;
        self.cv.notify_all();
    }

    fn aborted(&self) -> bool {
        self.lock().abort
    }

    /// Blocks until `slot` is free; errors once the stream is aborted.
    fn wait_free(&self, slot: usize) -> anyhow::Result<()> {
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
    fn wait_released(&self) {
        let mut g = self.lock();
        while g.slots.contains(&SlotState::Leased) {
            g = self.wait_logged(g, "the stream's end");
        }
    }

    /// One condvar wait; a wait of `STALL_WARN` or longer logs how many batches the sink still
    /// holds, so a leaked or over-held `FrameBatch` shows up instead of a silent hang.
    fn wait_logged<'g>(
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
struct AbortOnPanic<'s>(&'s Shared);

impl Drop for AbortOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.abort();
        }
    }
}

/// A completed batch's staging bytes, lent out until dropped: the slot takes no new batch before
/// that. Dropping it unmaps a wgpu staging buffer and frees the slot.
struct Lease {
    shared: Arc<Shared>,
    slot: usize,
    staging: Arc<Staging>,
    /// wgpu staging: its mapped range, which `ptr`/`len` point into.
    view: Option<wgpu::BufferView>,
    ptr: *const u8,
    len: usize,
}

// SAFETY: the bytes are immutable while the lease lives (the slot is not resubmitted, so neither
// the GPU nor the transfer queue writes them, and wgpu keeps the range mapped); the other fields
// are `Send + Sync`.
unsafe impl Send for Lease {}
unsafe impl Sync for Lease {}

impl Lease {
    fn bytes(&self) -> &[u8] {
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
    /// Checks every frame length (1..=FRAME_STRIDE) and locates the frames: at their fixed stride,
    /// or (`packed`) one after another at `PACK_ALIGN` boundaries.
    fn new(lease: Lease, first: usize, n: u32, tag: u64, layout: &StagingLayout, packed: bool) -> anyhow::Result<Self> {
        let bytes = lease.bytes();
        let lens: &[u32] = bytemuck::cast_slice(&bytes[..frame_len_bytes(n) as usize]);
        let mut spans = Vec::with_capacity(n as usize);
        let mut at = layout.a as usize;
        for (b, &len) in lens.iter().enumerate() {
            anyhow::ensure!(len > 0 && len as usize <= FRAME_STRIDE, "block {}: bad frame length {len}", first + b);
            if packed {
                spans.push((at, len));
                at += (len as usize).next_multiple_of(PACK_ALIGN);
            } else {
                spans.push((layout.a as usize + b * FRAME_STRIDE, len));
            }
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

/// Transfer readback (frame path, `GpuContext::transfer`, no packing): K4 writes `frames` /
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
/// reusable across runs.
pub struct Pipeline<'a> {
    ctx: &'a GpuContext,
    cfg: PipelineConfig,
    kernels: Kernels,
    /// Every device-side buffer, shared by all slots (see the module docs).
    bufs: BatchBuffers,
    layout: StagingLayout,
    slots: Vec<Slot>,
    /// The slots' staging states (see `Shared`).
    shared: Arc<Shared>,
    /// Frame path with `GpuContext::pack_frames` (opt-in, `GZC_PACK`): packs the frames into the
    /// staging buffer instead of copying the fixed-stride region.
    pack: Option<PackKernel>,
    /// `GpuContext::direct_upload`: `bufs.data` is the submitting slot's upload buffer (set per
    /// submission) and there is no upload copy.
    direct: bool,
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

impl Drop for Pipeline<'_> {
    fn drop(&mut self) {
        if let Some(x) = &self.xfer {
            // The raw buffers, semaphores and command buffers must outlive every GPU use.
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
            x.tq.idle();
        }
    }
}

/// `pack_frames.wgsl`: after K4, writes a batch's frames contiguously (16-byte aligned) straight
/// into the slot's mappable staging buffer, so only the frames' bytes cross the bus. Needs
/// `MAPPABLE_PRIMARY_BUFFERS` (staging is a storage buffer then). Off by default: on the RTX 5090
/// (PCIe 5 x16) the shader's stores into host memory move ~33 GB/s against the copy engine's
/// ~50 GB/s, so packing the ~74 % of the bytes that are real takes longer than copying them all.
struct PackKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

const PACK_WGSL: &str = include_str!("shaders/pack_frames.wgsl");

/// Packed frames start on multiples of this many bytes (`pack_frames.wgsl` stores 16 at once).
const PACK_ALIGN: usize = 16;

impl PackKernel {
    fn new(ctx: &GpuContext, staging: &StagingLayout) -> Self {
        let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pack"),
            entries: &[entry(0, true), entry(1, true), entry(2, false)],
        });
        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pack"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        const _: () =
            assert!(FRAME_STRIDE.is_multiple_of(PACK_ALIGN) && (STAGING_ALIGN as usize).is_multiple_of(PACK_ALIGN));
        let src = format!(
            "const FRAME_WORDS: u32 = {}u;\nconst PACK_BASE: u32 = {}u;\n{PACK_WGSL}",
            FRAME_STRIDE / 4,
            staging.a / 4
        );
        let module = ctx.wgsl_module("pack_frames", &src, wgpu::ShaderRuntimeChecks::checked());
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pack_frames"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: crate::context::compile_opts(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// Records the kernel on the first `n` blocks of `bufs` (which must hold frames), into `staging`.
    fn record(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &BatchBuffers,
        n: u32,
        staging: &wgpu::Buffer,
    ) {
        let (frames, frame_len) = (bufs.frames.as_ref().unwrap(), bufs.frame_len.as_ref().unwrap());
        // The kernel takes the batch size from the bound length of `frame_len`.
        let frame_len =
            wgpu::BufferBinding { buffer: frame_len, offset: 0, size: wgpu::BufferSize::new(frame_len_bytes(n)) };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pack"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(frame_len) },
                wgpu::BindGroupEntry { binding: 1, resource: frames.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: staging.as_entire_binding() },
            ],
        });
        let mut pass =
            enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pack"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n, 1, 1);
    }
}

impl<'a> Pipeline<'a> {
    /// Compiles the kernels and allocates the shared buffers and every slot, for the frame path
    /// when `cfg.params.emit_frames` and the parse path otherwise. Errors on `batch`/`inflight`
    /// of 0, a batch above `max_batch_blocks`, match params the GPU does not support (see
    /// `Kernels::new`), or a wgpu out-of-memory/validation error. The frame path packs frames
    /// (`PackKernel`) iff the context has `GpuContext::pack_frames` (asked for
    /// packing: `GZC_PACK`, or `GpuContext::with_options`).
    pub fn new(ctx: &'a GpuContext, cfg: &PipelineConfig) -> anyhow::Result<Self> {
        let m = cfg.params.matching;
        let max = max_batch_blocks(&ctx.device.limits(), &m);
        anyhow::ensure!(cfg.inflight >= 1, "inflight must be at least 1");
        anyhow::ensure!(cfg.batch >= 1 && cfg.batch <= max, "batch {} not in 1..={max} for this device", cfg.batch);

        let scopes = ErrorScopes::push(ctx);
        let frames = cfg.params.emit_frames;
        // Kernels::new validates the match params (`check_matching`) before any buffer is allocated.
        let kernels = Kernels::new(ctx, cfg.params)?;
        let layout = StagingLayout::new(cfg.batch, frames, &m);
        let pack = (frames && ctx.pack_frames).then(|| PackKernel::new(ctx, &layout));
        let mut staging_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        if pack.is_some() {
            staging_usage |= wgpu::BufferUsages::STORAGE;
        }
        let direct = ctx.direct_upload;
        let upload_usage = if direct {
            // The kernels bind it as `data` (MAPPABLE_PRIMARY_BUFFERS).
            wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::STORAGE
        } else {
            wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC
        };
        let tq = ctx.transfer.clone().filter(|_| frames && pack.is_none());
        use ash::vk::BufferUsageFlags as U;
        let shared = U::STORAGE_BUFFER | U::TRANSFER_SRC | U::TRANSFER_DST;
        // The transfer readback's raw objects exist before any wgpu buffer imports them, so on an
        // early error return the imports (declared later) drop first.
        let xfer = match tq {
            Some(tq) => Some(Xfer {
                // First, so a second transfer-readback pipeline errors before allocating anything.
                _streaming: tq.begin_streaming()?,
                frames: tq.buffer(frames_bytes(cfg.batch), shared, false)?,
                frame_len: tq.buffer(frame_len_bytes(cfg.batch), shared, false)?,
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
            let upload = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pipeline.upload"),
                size: data_bytes(cfg.batch),
                usage: upload_usage,
                mapped_at_creation: true,
            });
            assert_eq!(upload.size(), data_bytes(cfg.batch), "upload slots keep the trailing zero word");
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
        let bufs = BatchBuffers::with_parts(ctx, cfg.batch, frames, &m, data, frame_bufs);
        scopes.pop()?;
        Ok(Self {
            ctx,
            cfg: *cfg,
            kernels,
            bufs,
            layout,
            shared: Shared::new(slots.len()),
            slots,
            pack,
            direct,
            xfer,
            #[cfg(test)]
            fail_deliveries_after: None,
            #[cfg(test)]
            fail_submits_after: None,
            #[cfg(test)]
            poll_only: false,
        })
    }

    /// True when batches are read back through the transfer queue (see `Xfer`).
    pub fn transfer_readback(&self) -> bool {
        self.xfer.is_some()
    }

    /// How this pipeline's K3 runs (the mode its kernels were built with).
    pub fn k3_mode(&self) -> crate::compressor::K3Mode {
        self.kernels.k3_mode()
    }

    /// Bytes of every buffer the pipeline created; timestamp query sets and their resolve
    /// buffers are left out, as in `vram_bytes`.
    #[cfg(test)]
    fn allocated_bytes(&self) -> u64 {
        let opt = |b: &Option<wgpu::Buffer>| b.as_ref().map_or(0, |b| b.size());
        let s = &self.bufs;
        // Direct upload: `data` is a slot's upload buffer, counted with the slots.
        let data = if self.direct { 0 } else { s.data.size() };
        let k3opt = s.opt.as_ref().map_or(0, |o| o.prices.size() + o.scratch.size());
        let shared = data
            + [&s.head, &s.pred, &s.best, &s.seqs, &s.counts].iter().map(|b| b.size()).sum::<u64>()
            + opt(&s.frames)
            + opt(&s.frame_len)
            + k3opt;
        let per_slot: u64 = self.slots.iter().map(|slot| slot.upload.size() + slot.staging.size()).sum();
        // Transfer readback: frames / frame_len are imports of `Xfer`'s buffers (same sizes).
        shared + per_slot + self.kernels.own_buffer_bytes()
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

    /// Streams `blocks` (each BLOCK_SIZE bytes) through the slots, handing every block's zstd
    /// frame (K4 output) to `sink` exactly once, on the completion thread (see `FrameSink`) while
    /// this thread uploads. Errors on a parse-path pipeline, and on wgpu validation or
    /// out-of-memory errors.
    pub fn run_frames(&mut self, blocks: &[&[u8]], sink: &mut (impl FrameSink + Send)) -> anyhow::Result<PipelineStats> {
        check_blocks(blocks)?;
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
        check_blocks(blocks)?;
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
        P: FnOnce(&mut FrameStream<'_, 'a>) -> anyhow::Result<()>,
    {
        anyhow::ensure!(self.layout.frames, "pipeline built without emit_frames: use run");
        let (layout, packed) = (self.layout, self.pack.is_some());
        let handler = move |lease: Lease, first: usize, n: u32, tag: u64| {
            on_batch(FrameBatch::new(lease, first, n, tag, &layout, packed)?)
        };
        self.stream_with(Box::new(handler), produce)
    }

    /// Runs `produce` on this thread and the completion thread (`Completion`) beside it, which
    /// owns `handler` and drops it when done (so a sink that forwards batches to its own threads
    /// can release them on the channel's close); collects the profile. A panic on either side
    /// is re-raised once the pipeline is cleaned up (`abandon`).
    fn stream_with<P>(&mut self, handler: Box<Handler<'_>>, produce: P) -> anyhow::Result<PipelineStats>
    where
        P: FnOnce(&mut FrameStream<'_, 'a>) -> anyhow::Result<()>,
    {
        let scopes = ErrorScopes::push(self.ctx);
        let start = Instant::now();
        let names = self.kernels.names();
        // End query of the last kernel recorded: K4 on the frame path (K5 runs before it), K3 on
        // the parse path.
        let last = if self.layout.frames { "k4_entropy" } else { "k3_parse" };
        #[cfg(test)]
        let fail_after = self.fail_deliveries_after.take();
        #[cfg(not(test))]
        let fail_after = None;
        let metal = self.ctx.adapter_info.backend == wgpu::Backend::Metal;
        #[cfg(test)]
        let poll_only = metal || self.poll_only;
        #[cfg(not(test))]
        let poll_only = metal;
        let completion = Completion {
            ctx: self.ctx,
            xfer: self.xfer.as_ref().map(|x| (x.tq.clone(), x.t_done.clone())),
            shared: self.shared.clone(),
            layout: self.layout,
            n_kernels: names.len(),
            last_kernel_end: 2 * names.iter().position(|&k| k == last).expect("last kernel is timed") + 1,
            timed: self.ctx.timestamps,
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
            (
                names
                    .iter()
                    .zip(&cprof.ticks)
                    .zip(&cprof.ticks_bad)
                    .filter(|(_, bad)| !**bad)
                    .map(|((name, &t), _)| (name.to_string(), ms(t)))
                    .collect(),
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
    /// the slot's staging buffer (the fixed-stride copies, or the pack kernel, or with the transfer
    /// readback the copies on the transfer queue), plus the resolved timestamps, submits them and
    /// requests the maps; returns the batch for the completion thread. The slot's upload buffer
    /// holds the blocks (mapped); it is persistent rather than `queue.write_buffer`, which would
    /// allocate a fresh staging buffer per call that lives until the submission completes.
    fn submit(&mut self, i: usize, first: usize, n: u32, tag: u64) -> anyhow::Result<Job> {
        if self.direct {
            // The kernels read this slot's upload buffer; the bind groups recorded below hold it.
            self.bufs.data = self.slots[i].upload.clone();
        }
        let (ctx, layout, bufs, direct) = (self.ctx, self.layout, &self.bufs, self.direct);
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
        let resolved = (n_queries + 2) as u64 * wgpu::QUERY_SIZE as u64;
        let queries = slot.queries.as_ref().map(|q| &q.set);
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
                self.kernels.record_entropy(ctx, &mut enc, bufs, n, queries);
                marker(&mut enc, n_queries + 1);
                if let Some(q) = &slot.queries {
                    enc.resolve_query_set(&q.set, 0..n_queries + 2, &q.resolve, 0);
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
                self.kernels.record_timed(ctx, &mut enc, bufs, n, queries)?;
                if let Some(pack) = &self.pack {
                    pack.record(ctx, &mut enc, bufs, n, staging);
                } else if layout.frames {
                    let (frames, frame_len) = (bufs.frames.as_ref().unwrap(), bufs.frame_len.as_ref().unwrap());
                    enc.copy_buffer_to_buffer(frame_len, 0, staging, 0, frame_len_bytes(n));
                    enc.copy_buffer_to_buffer(frames, 0, staging, layout.a, frames_bytes(n));
                } else {
                    enc.copy_buffer_to_buffer(&bufs.counts, 0, staging, 0, counts_bytes(n));
                    enc.copy_buffer_to_buffer(&bufs.seqs, 0, staging, layout.a, seqs_bytes_for(n, &self.cfg.params.matching));
                }
                marker(&mut enc, n_queries + 1);
                if let Some(q) = &slot.queries {
                    enc.resolve_query_set(&q.set, 0..n_queries + 2, &q.resolve, 0);
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
        Ok(Job { slot: i, first, n, tag, submission, mapped, seq, staging: slot.staging.clone() })
    }

    /// After an error: wait for the GPU, unmap and free every slot still in flight so the
    /// pipeline stays usable, then wait for the sink to drop the batches it still holds.
    fn abandon(&mut self) {
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
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

/// The completion thread of a stream: waits for each batch in submission order, adds its
/// timestamps to the profile and lends its staging bytes to the handler.
struct Completion<'c> {
    ctx: &'c GpuContext,
    /// Transfer readback: the queue and the `t_done` timeline to wait on.
    xfer: Option<(Arc<TransferQueue>, Arc<Timeline>)>,
    shared: Arc<Shared>,
    layout: StagingLayout,
    n_kernels: usize,
    /// Index of the last kernel's end query.
    last_kernel_end: usize,
    timed: bool,
    /// Wait for a batch by polling (`PollType::Poll`) until its staging map completes, never with
    /// `PollType::Wait`; on Metal. wgpu-hal 30's Metal `Device::wait` errors with
    /// `DeviceError::Lost` ("No active command buffers for fence value") when it runs while a
    /// `queue.submit` on another thread (the producer's next batch) is between `Fence::maintain`,
    /// which drops command buffers whose status is already `Completed` although their completion
    /// handler has not yet raised the fence value, and pushing its own command buffer: it then
    /// finds the fence below the value and no pending command buffer that will reach it.
    /// wgpu-core turns that into a lost device and destroys every buffer (which surfaced as
    /// "Buffer with 'pipeline.staging' label has been destroyed" from `abandon`'s unmap). A
    /// non-blocking poll never calls `Device::wait`.
    poll_only: bool,
    /// Test hook: the delivery after this many more fails (then the hook clears).
    fail_after: Option<u32>,
}

/// The completion thread's share of `PipelineStats::transfer_ms` (GPU ticks, host seconds).
#[derive(Default)]
struct CompletionProfile {
    /// Kernel ticks, per `Kernels::names`.
    ticks: Vec<u64>,
    /// Per kernel: some batch had an unwritten begin or end timestamp (`unwritten_stamp`).
    ticks_bad: Vec<bool>,
    /// Some batch had an unwritten marker timestamp: `upload_copy`, `readback` and `idle` are
    /// meaningless (Metal apparently writes none for the empty marker passes).
    markers_bad: bool,
    upload_copy: u64,
    readback: u64,
    idle: u64,
    /// End marker of the last batch finished (ticks), for `idle`.
    last_end: Option<u64>,
    wait: f64,
    deliver: f64,
    /// When the last wait for the GPU returned, for `host_drain`.
    last_wait: Option<Instant>,
    fail_after: Option<u32>,
}

/// The producer's share of `PipelineStats::transfer_ms` (host seconds).
#[derive(Default)]
struct ProducerProfile {
    upload_wait: f64,
    upload_write: f64,
    submit: f64,
    fill: f64,
}

impl Completion<'_> {
    /// Delivers every job; drops `handler` before returning.
    fn run(mut self, jobs: mpsc::Receiver<Job>, mut handler: Box<Handler<'_>>) -> (anyhow::Result<()>, CompletionProfile) {
        let shared = self.shared.clone();
        let _abort = AbortOnPanic(&shared);
        let mut prof =
            CompletionProfile { ticks: vec![0; self.n_kernels], ticks_bad: vec![false; self.n_kernels], ..Default::default() };
        let r = self.drain(jobs, &mut *handler, &mut prof);
        drop(handler);
        if r.is_err() {
            shared.abort();
        }
        prof.fail_after = self.fail_after;
        (r, prof)
    }

    fn drain(
        &mut self,
        jobs: mpsc::Receiver<Job>,
        handler: &mut Handler<'_>,
        prof: &mut CompletionProfile,
    ) -> anyhow::Result<()> {
        // Ends once the producer has dropped its sender and every job is taken.
        for job in jobs {
            if self.shared.aborted() {
                // The producer failed: deliver nothing more (`abandon` frees the slots).
                break;
            }
            let t = Instant::now();
            // The staging map's result, when the wait already took it.
            let mut mapped = None;
            match (&self.xfer, &job.mapped) {
                (Some((tq, t_done)), _) => {
                    tq.wait(t_done, job.seq)?;
                    // Deliver the upload buffers' map callbacks of completed submissions.
                    self.ctx.device.poll(wgpu::PollType::Poll).context("device poll")?;
                }
                (None, Some(rx)) if self.poll_only => {
                    // The map completes with the submission (the staging buffer's last use).
                    mapped = Some(loop {
                        self.ctx.device.poll(wgpu::PollType::Poll).context("device poll")?;
                        match rx.recv_timeout(std::time::Duration::from_micros(200)) {
                            Ok(r) => break r,
                            // The producer failed: deliver nothing more, as above.
                            Err(mpsc::RecvTimeoutError::Timeout) if self.shared.aborted() => return Ok(()),
                            Err(mpsc::RecvTimeoutError::Timeout) => {}
                            Err(mpsc::RecvTimeoutError::Disconnected) => anyhow::bail!("staging map callback dropped"),
                        }
                    });
                }
                (None, _) => {
                    let wait = wgpu::PollType::Wait { submission_index: Some(job.submission.clone()), timeout: None };
                    self.ctx.device.poll(wait).context("device poll")?;
                }
            }
            let done = Instant::now();
            prof.wait += (done - t).as_secs_f64();
            prof.last_wait = Some(done);
            let lease = self.lease(&job, mapped)?;
            if self.timed {
                self.add_timestamps(lease.bytes(), prof);
            }
            if let Some(k) = &mut self.fail_after {
                if *k == 0 {
                    self.fail_after = None;
                    anyhow::bail!("injected delivery failure (test hook)");
                }
                *k -= 1;
            }
            let t = Instant::now();
            handler(lease, job.first, job.n, job.tag)?;
            prof.deliver += t.elapsed().as_secs_f64();
        }
        Ok(())
    }

    /// Lends out `job`'s staging bytes (its batch completed); `mapped`: the staging map's result
    /// if the wait already received it.
    fn lease(&self, job: &Job, mapped: Option<Result<(), wgpu::BufferAsyncError>>) -> anyhow::Result<Lease> {
        let (view, ptr, len) = match &*job.staging {
            Staging::Wgpu(staging) => {
                let r = match mapped {
                    Some(r) => r,
                    None => {
                        let rx = job.mapped.as_ref().context("wgpu staging without a map request")?;
                        // The submission completed, so its map callback has run or is running
                        // (on whichever thread polled: the producer polls too).
                        rx.recv().context("staging map callback dropped")?
                    }
                };
                r.context("map staging buffer")?;
                let view = staging.get_mapped_range(..).map_err(|e| anyhow!("mapped range: {e}"))?;
                let (ptr, len) = (view.as_ptr(), view.len());
                (Some(view), ptr, len)
            }
            Staging::Host(staging) => {
                // SAFETY: the transfer queue's copy into it completed (waited for t_done >=
                // job.seq), and the slot's next copy is only submitted once the lease is dropped.
                let bytes = unsafe { staging.mapped() };
                (None, bytes.as_ptr(), bytes.len())
            }
        };
        self.shared.set(job.slot, SlotState::Leased);
        Ok(Lease { shared: self.shared.clone(), slot: job.slot, staging: job.staging.clone(), view, ptr, len })
    }

    fn add_timestamps(&self, view: &[u8], prof: &mut CompletionProfile) {
        let nk = self.n_kernels;
        // Only 4-byte aligned in general (seqs_bytes(1) is not a multiple of 8).
        let t = self.layout.ts as usize;
        let stamps: Vec<u64> =
            view[t..t + (nk + 1) * 16].chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        for (k, acc) in prof.ticks.iter_mut().enumerate() {
            *acc += stamps[2 * k + 1].saturating_sub(stamps[2 * k]);
            prof.ticks_bad[k] |= unwritten_stamp(stamps[2 * k]) || unwritten_stamp(stamps[2 * k + 1]);
        }
        let (m0, m1) = (stamps[2 * nk], stamps[2 * nk + 1]);
        prof.markers_bad |= [m0, m1, stamps[0], stamps[self.last_kernel_end]].into_iter().any(unwritten_stamp);
        prof.upload_copy += stamps[0].saturating_sub(m0);
        prof.readback += m1.saturating_sub(stamps[self.last_kernel_end]);
        if let Some(end) = prof.last_end {
            prof.idle += m0.saturating_sub(end);
        }
        prof.last_end = Some(m1);
    }
}

/// A timestamp the GPU did not write: resolved as 0 (an M4 Pro printed a `gpu_upload_copy` of
/// about 1.9e6 ms, the absolute GPU clock minus a zero start marker: Metal apparently samples
/// nothing for the empty marker passes) or as Metal's `MTLCounterErrorValue` (all ones).
fn unwritten_stamp(t: u64) -> bool {
    t == 0 || t == u64::MAX
}

/// The producer's side of a stream (`Pipeline::stream_frames`): hands out the slots' mapped upload
/// buffers in turn and submits them. Block indices count from 0 per stream, in submission order.
pub struct FrameStream<'p, 'a> {
    pipe: &'p mut Pipeline<'a>,
    tx: mpsc::Sender<Job>,
    next_slot: usize,
    next_index: usize,
    batches: u32,
    start: Instant,
    prof: ProducerProfile,
    /// The first failed submission's error: the stream fails even if `produce` swallows it.
    submit_error: Option<String>,
}

impl<'p, 'a> FrameStream<'p, 'a> {
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
    pub fn next_upload_slot(&mut self) -> anyhow::Result<UploadSlot<'_, 'p, 'a>> {
        let i = self.next_slot;
        let t = Instant::now();
        self.pipe.shared.wait_free(i)?;
        let ctx = self.pipe.ctx;
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
            // Its submission has completed (the slot is free), so the callback is normally in.
            let r = match rx.try_recv() {
                Ok(r) => r,
                Err(_) => {
                    let wait = match slot.upload_submission.clone() {
                        Some(s) => wgpu::PollType::Wait { submission_index: Some(s), timeout: None },
                        None => wgpu::PollType::wait_indefinitely(),
                    };
                    ctx.device.poll(wait).context("device poll")?;
                    rx.recv().context("upload map callback dropped")?
                }
            };
            if r.is_err() {
                slot.upload_unmapped = true;
            }
            r.context("map upload buffer")?;
        }
        let view = slot.upload.get_mapped_range_mut(..).context("upload mapped range")?;
        self.prof.upload_wait += t.elapsed().as_secs_f64();
        Ok(UploadSlot { stream: self, slot: i, view, acquired: Instant::now() })
    }

    /// Copies `blocks` (each BLOCK_SIZE bytes) into as many slots as they need and submits them:
    /// the `&[&[u8]]` form of the API. The copy is split over `GZC_UPLOAD_THREADS` threads.
    pub fn upload_blocks(&mut self, blocks: &[&[u8]]) -> anyhow::Result<()> {
        check_blocks(blocks)?;
        for chunk in blocks.chunks(self.slot_capacity()) {
            let mut slot = self.next_upload_slot()?;
            let region = slot.regions_mut(&[chunk.len()])?.pop().expect("one region");
            copy_blocks(region, chunk);
            slot.submit(chunk.len())?;
        }
        Ok(())
    }

    /// Submits slot `i`'s first `n` blocks. Any failure aborts the stream: a slot whose
    /// submission failed after it was marked in flight would never come free, so every later
    /// call must error rather than wait for it.
    fn submit(&mut self, i: usize, n: u32, tag: u64) -> anyhow::Result<()> {
        let t = Instant::now();
        let job = match self.pipe.submit(i, self.next_index, n, tag) {
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
pub struct UploadSlot<'s, 'p, 'a> {
    stream: &'s mut FrameStream<'p, 'a>,
    slot: usize,
    view: wgpu::BufferViewMut,
    acquired: Instant,
}

impl UploadSlot<'_, '_, '_> {
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
        let mut rest = self.view.slice(..total * BLOCK_SIZE);
        let mut out = Vec::with_capacity(blocks.len());
        for &b in blocks {
            let (region, tail) = rest.split_at(b * BLOCK_SIZE);
            out.push(Region(region));
            rest = tail;
        }
        Ok(out)
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
        let UploadSlot { stream, slot, mut view, acquired } = self;
        // Trailing zero word after the last block (the slot may hold stale blocks beyond it).
        view.slice(n * BLOCK_SIZE..n * BLOCK_SIZE + 4).copy_from_slice(&[0u8; 4]);
        drop(view);
        stream.prof.upload_write += acquired.elapsed().as_secs_f64();
        let first = stream.next_index;
        stream.submit(slot, n as u32, tag)?;
        Ok(first)
    }
}

/// A write-only piece of an upload slot (`UploadSlot::regions_mut`), a whole number of blocks.
pub struct Region<'a>(wgpu::WriteOnly<'a, [u8]>);

// SAFETY: `WriteOnly<[u8]>` lacks `Send` only because wgpu's impl needs a sized `T`; like a
// `&mut [u8]`, a byte range of it may move to another thread, and `regions_mut` hands out
// disjoint ranges.
unsafe impl Send for Region<'_> {}

impl<'a> Region<'a> {
    /// Bytes in the region.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Writes `bytes` at `offset` (panics past the end).
    pub fn write(&mut self, offset: usize, bytes: &[u8]) {
        self.0.slice(offset..offset + bytes.len()).copy_from_slice(bytes);
    }

    /// The region as wgpu's `WriteOnly`, for writers that take one.
    pub fn write_only(&mut self) -> wgpu::WriteOnly<'_, [u8]> {
        self.0.slice(..)
    }

    /// Finishes a `len`-byte payload written at the start of the region: zero-fills the rest of
    /// its last block, the padding `chunk_file` gives a file's last block (the slot's old bytes
    /// would otherwise be compressed with it), and returns the blocks it occupies
    /// (`payload_blocks(len)`; `payload_real_lens(len)` gives their real lengths). Errors if they
    /// do not fit in the region.
    pub fn pad(&mut self, len: usize) -> anyhow::Result<usize> {
        let blocks = payload_blocks(len);
        let end = blocks * BLOCK_SIZE;
        anyhow::ensure!(
            end <= self.len(),
            "a {len}-byte payload needs {blocks} blocks, the region has {}",
            self.len() / BLOCK_SIZE
        );
        self.0.slice(len..end).fill(0);
        Ok(blocks)
    }

    fn split_at(self, mid: usize) -> (Region<'a>, Region<'a>) {
        let (a, b) = self.0.split_at(mid);
        (Region(a), Region(b))
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
fn check_blocks(blocks: &[&[u8]]) -> anyhow::Result<()> {
    if let Some(i) = blocks.iter().position(|b| b.len() != BLOCK_SIZE) {
        anyhow::bail!("block {i} is {} bytes, expected BLOCK_SIZE {BLOCK_SIZE}", blocks[i].len());
    }
    Ok(())
}

/// Copies `blocks` back to back into `dst`, split over `upload_threads()` threads (this one
/// included): one thread's stores into the (write-combined, ReBAR) upload buffer run at ~18 GB/s,
/// two or more at the link's ~26 GB/s (RTX 5090).
fn copy_blocks(dst: Region<'_>, blocks: &[&[u8]]) {
    let copy = |mut dst: Region<'_>, src: &[&[u8]]| {
        for (k, b) in src.iter().enumerate() {
            dst.write(k * BLOCK_SIZE, b);
        }
    };
    let per = blocks.len().div_ceil(upload_threads()).max(64);
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

/// Threads writing a batch into its upload buffer: `GZC_UPLOAD_THREADS`, else 4 (at most the
/// available parallelism).
fn upload_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        std::env::var("GZC_UPLOAD_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4).clamp(1, cores.max(1))
    })
}

/// Out-of-memory and validation error scopes, popped together.
struct ErrorScopes {
    // Fields drop in declaration order and wgpu requires scopes to pop in reverse push order:
    // `validation` (pushed last) must come first, so an early return drops them correctly.
    validation: wgpu::ErrorScopeGuard,
    oom: wgpu::ErrorScopeGuard,
}

impl ErrorScopes {
    fn push(ctx: &GpuContext) -> Self {
        let oom = ctx.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        Self { oom, validation }
    }

    fn pop(self) -> anyhow::Result<()> {
        let validation = pollster::block_on(self.validation.pop());
        let oom = pollster::block_on(self.oom.pop());
        if let Some(e) = validation {
            return Err(anyhow!("wgpu validation error: {e}"));
        }
        if let Some(e) = oom {
            return Err(anyhow!("wgpu out of memory: {e}"));
        }
        Ok(())
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

/// Builds a parse-path `Pipeline` for `cfg` (whose `emit_frames` must be false) and streams
/// `blocks` through it (see `Pipeline::run`).
pub fn compress_stream(
    ctx: &GpuContext,
    cfg: &PipelineConfig,
    blocks: &[&[u8]],
    sink: &mut (impl BlockSink + Send),
) -> anyhow::Result<PipelineStats> {
    Pipeline::new(ctx, cfg)?.run(blocks, sink)
}

/// Builds a frame-path `Pipeline` for `cfg` (`emit_frames` is forced on; `huffman` is kept) and
/// streams `blocks` through it (see `Pipeline::run_frames`).
pub fn compress_stream_frames(
    ctx: &GpuContext,
    cfg: &PipelineConfig,
    blocks: &[&[u8]],
    sink: &mut (impl FrameSink + Send),
) -> anyhow::Result<PipelineStats> {
    let cfg = PipelineConfig { params: GpuParams { emit_frames: true, ..cfg.params }, ..*cfg };
    Pipeline::new(ctx, &cfg)?.run_frames(blocks, sink)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chains::{head_bytes, pred_bytes};
    use gzc_core::block::chunk_file;
    use gzc_core::frame::write_frame;
    use gzc_core::params::{LVL3, LVL9, LVL9S12SEG, MatchParams, OPT14, OPT16, RUNG1};
    use gzc_core::reference::compress_block;
    use gzc_core::synth::test_cases;

    struct Collect(Vec<Option<BlockOutput>>);

    impl BlockSink for Collect {
        fn put(&mut self, index: usize, out: BlockOutput) {
            assert!(self.0[index].is_none(), "index {index} delivered twice");
            self.0[index] = Some(out);
        }
    }

    /// The frame path's kernel timers (Huffman literals): all five, each positive. The one
    /// exception: on an adapter that samples timestamps at pass boundaries only (no
    /// `TIMESTAMP_QUERY_INSIDE_ENCODERS`: Metal on Apple GPUs), the device may leave the last
    /// kernel's (K4's) pair unwritten, so `kernel_ms` leaves K4 out (`PipelineStats::kernel_ms`);
    /// every other timer must still be there.
    fn assert_frame_timers(ctx: &GpuContext, stats: &PipelineStats, what: &str) {
        if !ctx.timestamps {
            assert!(stats.kernel_ms.is_empty(), "{what}: {:?}", stats.kernel_ms);
            return;
        }
        let all = ["k1_chains", "k2_best", "k3_parse", "k4_entropy", "k5_huffman"];
        let names: Vec<&str> = stats.kernel_ms.iter().map(|(n, _)| n.as_str()).collect();
        if ctx.timestamps_inside_encoders || names != ["k1_chains", "k2_best", "k3_parse", "k5_huffman"] {
            assert_eq!(names, all, "{what}");
        }
        assert!(stats.kernel_ms.iter().all(|&(_, ms)| ms > 0.0), "{what}: {:?}", stats.kernel_ms);
    }

    fn cfg(batch: u32, inflight: u32) -> PipelineConfig {
        PipelineConfig { batch, inflight, params: GpuParams { matching: LVL3, emit_frames: false, huffman: true } }
    }

    #[test]
    fn stream_matches_reference_every_index_once() {
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        let distinct: Vec<Vec<u8>> =
            test_cases().into_iter().flat_map(|(_, bytes)| chunk_file(&bytes)).map(|b| b.data).collect();
        let want: Vec<BlockOutput> = distinct.iter().map(|b| compress_block(b, LVL3)).collect();
        // 1000 = 15 * 64 + 40: the last batch is partial and reuses a slot holding stale blocks.
        let blocks: Vec<&[u8]> = (0..1000).map(|i| distinct[i % distinct.len()].as_slice()).collect();

        let mut sink = Collect(vec![None; blocks.len()]);
        let stats = compress_stream(&ctx, &cfg(64, 3), &blocks, &mut sink).expect("compress_stream");

        for (i, got) in sink.0.iter().enumerate() {
            let got = got.as_ref().unwrap_or_else(|| panic!("index {i} never delivered"));
            assert!(*got == want[i % distinct.len()], "index {i}: GPU != reference");
        }
        assert_eq!(stats.batches, 16);
        assert!(stats.wall_s > 0.0);
        if ctx.timestamps {
            let names: Vec<&str> = stats.kernel_ms.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(names, ["k1_chains", "k2_best", "k3_parse"]);
            assert!(stats.kernel_ms.iter().all(|&(_, ms)| ms > 0.0), "{:?}", stats.kernel_ms);
        }
    }

    #[test]
    fn stream_odd_batch_single_slot_and_reuse() {
        // Odd batch: the staging timestamp region is only 4-byte aligned. One slot: every batch
        // waits for the previous one. The pipeline is reused for a second run.
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        let distinct: Vec<Vec<u8>> =
            test_cases().into_iter().flat_map(|(_, bytes)| chunk_file(&bytes)).map(|b| b.data).collect();
        let blocks: Vec<&[u8]> = distinct.iter().map(|b| b.as_slice()).collect();
        let mut pipe = Pipeline::new(&ctx, &cfg(7, 1)).unwrap();
        for _ in 0..2 {
            let mut sink = Collect(vec![None; blocks.len()]);
            let stats = pipe.run(&blocks, &mut sink).unwrap();
            assert_eq!(stats.batches as usize, blocks.len().div_ceil(7));
            for (i, got) in sink.0.into_iter().enumerate() {
                assert!(got.unwrap() == compress_block(blocks[i], LVL3), "index {i}");
            }
        }
        let mut sink = Collect(vec![None; 1]);
        assert!(pipe.run(&[&[0u8; 3]], &mut sink).is_err(), "short block rejected");
    }

    #[test]
    fn stream_handles_empty_input_and_rejects_bad_config() {
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        let mut sink = Collect(Vec::new());
        let stats = compress_stream(&ctx, &cfg(8, 2), &[], &mut sink).unwrap();
        assert_eq!(stats.batches, 0);
        assert!(compress_stream(&ctx, &cfg(0, 2), &[], &mut sink).is_err());
        assert!(compress_stream(&ctx, &cfg(8, 0), &[], &mut sink).is_err());
        assert!(compress_stream(&ctx, &cfg(u32::MAX, 1), &[], &mut sink).is_err());
        let bad = MatchParams { lazy: 3, ..LVL9 };
        let bad = PipelineConfig { params: GpuParams { matching: bad, ..cfg(8, 2).params }, ..cfg(8, 2) };
        let e = Pipeline::new(&ctx, &bad).err().expect("lazy 3 is invalid");
        assert!(e.to_string().contains("lazy 3"), "{e}");
    }

    struct CollectFrames(Vec<Option<Vec<u8>>>);

    impl FrameSink for CollectFrames {
        fn put(&mut self, index: usize, frame: &[u8]) {
            assert!(self.0[index].is_none(), "index {index} delivered twice");
            self.0[index] = Some(frame.to_vec());
        }
    }

    fn distinct_blocks() -> Vec<Vec<u8>> {
        test_cases().into_iter().flat_map(|(_, bytes)| chunk_file(&bytes)).map(|b| b.data).collect()
    }

    fn cpu_frame(block: &[u8], params: GpuParams) -> Vec<u8> {
        write_frame(block, &compress_block(block, params.matching), params.frame_options())
    }

    #[test]
    fn staging_regions_are_aligned() {
        for cap in [1, 7, 1365, 1535, 1890] {
            for frames in [true, false] {
                let l = StagingLayout::new(cap, frames, &LVL3);
                assert!([l.a, l.ts].iter().all(|x| x % STAGING_ALIGN == 0), "cap {cap} frames {frames}");
                let end = l.a + if frames { frames_bytes(cap) } else { seqs_bytes_for(cap, &LVL3) };
                assert!(l.a >= if frames { frame_len_bytes(cap) } else { counts_bytes(cap) } && end <= l.ts);
            }
        }
    }

    #[test]
    fn vram_counts_scratch_once_and_slots_per_inflight() {
        let frames = |batch, inflight| PipelineConfig {
            params: GpuParams { matching: LVL3, emit_frames: true, huffman: true },
            ..cfg(batch, inflight)
        };
        let one = vram_bytes(&frames(100, 1));
        let per_slot = vram_bytes(&frames(100, 2)) - one;
        assert_eq!(vram_bytes(&frames(100, 4)), one + 3 * per_slot);
        // A slot owns only its upload and staging buffers; data, frames and frame_len are shared.
        assert_eq!(per_slot, data_bytes(100) + StagingLayout::new(100, true, &LVL3).size);
        assert_eq!(one, scratch_bytes(100, &LVL3) + slot_bytes(100, true) + per_slot + k4_tables_bytes());
        // The parse path reads back the fixed-stride seqs instead of the frames.
        assert!(vram_bytes(&cfg(100, 2)) > vram_bytes(&frames(100, 2)));
        // The direct upload drops the shared `data` buffer only.
        assert_eq!(vram_bytes_with(&frames(100, 2), true), vram_bytes(&frames(100, 2)) - data_bytes(100));
        // K5 (Huffman literals) needs no buffers of its own.
        let raw_lits =
            PipelineConfig { params: GpuParams { huffman: false, ..frames(100, 2).params }, ..frames(100, 2) };
        assert_eq!(vram_bytes(&raw_lits), vram_bytes(&frames(100, 2)));
        #[cfg(feature = "block-128k")]
        {
            // ~2.4 MiB of scratch per block, ~0.25 MiB per block per slot on the frame path.
            let mib = |b: u64| b as f64 / (1u64 << 20) as f64 / 100.0;
            assert!((2.3..2.5).contains(&mib(scratch_bytes(100, &LVL3))), "{}", mib(scratch_bytes(100, &LVL3)));
            assert!((0.24..0.26).contains(&mib(per_slot)), "{}", mib(per_slot));
        }
        #[cfg(feature = "block-64k")]
        {
            // ~1.44 MiB of scratch per block, ~0.125 MiB per block per slot on the frame path.
            let mib = |b: u64| b as f64 / (1u64 << 20) as f64 / 100.0;
            assert!((1.4..1.5).contains(&mib(scratch_bytes(100, &LVL3))), "{}", mib(scratch_bytes(100, &LVL3)));
            assert!((0.12..0.13).contains(&mib(per_slot)), "{}", mib(per_slot));
        }
    }

    /// `vram_bytes` equals the bytes of every buffer a `Pipeline` actually creates (shared scratch
    /// once), per preset: single-hash presets allocate one chain's head/pred; the optimal parse
    /// adds its second candidate word, the 8 B-per-position trace (in `pred`), the larger `seqs`
    /// and K3opt's prices and scratch.
    #[test]
    fn vram_matches_params() {
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        for matching in [LVL3, RUNG1, LVL9, OPT16, OPT14] {
            // opt14/opt16 only implement at blocks of at most 64 KiB.
            if matching.opt.is_some() && gzc_core::config::LOG2_BLOCK > 16 {
                continue;
            }
            for (emit_frames, batch, inflight) in [(true, 7, 1), (true, 16, 3), (false, 5, 2)] {
                let cfg =
                    PipelineConfig { batch, inflight, params: GpuParams { matching, emit_frames, huffman: true } };
                let pipe = Pipeline::new(&ctx, &cfg).unwrap();
                assert_eq!(
                    pipe.allocated_bytes(),
                    vram_bytes_with(&cfg, ctx.direct_upload),
                    "{matching:?} {emit_frames} b{batch} i{inflight}"
                );
            }
        }
        let scratch = |m: MatchParams| {
            vram_bytes(&PipelineConfig {
                batch: 10,
                inflight: 1,
                params: GpuParams { matching: m, emit_frames: true, huffman: true },
            })
        };
        // One chain instead of two: head and pred halve.
        assert_eq!(scratch(LVL3) - scratch(RUNG1), head_bytes(10, 1) + pred_bytes(10, 1));
        // opt14/opt16 only implement at blocks of at most 64 KiB.
        if gzc_core::config::LOG2_BLOCK <= 16 {
            // The optimal parse over lvl3 (two chains too): candidates, seqs (in `slots` staging
            // only on the parse path), K3opt's prices and scratch.
            let opt_extra = crate::compressor::best_bytes(10)
                + crate::compressor::seqs_bytes_for(10, &OPT16)
                - crate::compressor::seqs_bytes(10)
                + crate::compressor::opt_bytes(10, &OPT16);
            assert_eq!(scratch(OPT16) - scratch(LVL3), opt_extra);
            assert_eq!(scratch(OPT14), scratch(OPT16));
        }
    }

    /// M5 T5: the optimal parse (K1 Opt3 → K2opt → K3opt passes → K5 → K4) through the streaming
    /// pipeline in every upload/readback mode, over partial batches and reuse, equals the CPU
    /// oracle's frames (which libzstd decodes); the parse path too. `allocated_bytes` equals
    /// `vram_bytes` in every mode.
    #[test]
    fn opt_stream_every_mode_matches_cpu() {
        // opt14/opt16 only implement at blocks of at most 64 KiB.
        if gzc_core::config::LOG2_BLOCK > 16 {
            return;
        }
        let distinct = distinct_blocks();
        let blocks: Vec<&[u8]> = (0..50).map(|i| distinct[i % distinct.len()].as_slice()).collect();
        let modes = mode_contexts();
        for matching in [OPT14, OPT16] {
            let params = GpuParams { matching, emit_frames: true, huffman: true };
            let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
            for (w, b) in want.iter().zip(&distinct) {
                assert_eq!(zstd::bulk::decompress(w, BLOCK_SIZE).unwrap(), *b, "libzstd decodes the oracle's frame");
            }
            for (name, ctx) in &modes {
                let pcfg = PipelineConfig { batch: 16, inflight: 3, params };
                let mut pipe = Pipeline::new(ctx, &pcfg).unwrap();
                assert_eq!(pipe.transfer_readback(), ctx.transfer.is_some(), "{name}");
                assert_eq!(pipe.allocated_bytes(), vram_bytes_with(&pcfg, ctx.direct_upload), "{name}");
                for _ in 0..2 {
                    let mut sink = CollectFrames(vec![None; blocks.len()]);
                    let stats = pipe.run_frames(&blocks, &mut sink).unwrap();
                    assert_eq!(stats.batches, 4, "{name}");
                    for (i, got) in sink.0.into_iter().enumerate() {
                        assert!(got.unwrap() == want[i % distinct.len()], "{name} {matching:?}: index {i}");
                    }
                    assert_frame_timers(ctx, &stats, name);
                }
            }
            // The parse path: the `seqs` readback at MAX_SEQS_OPT per block.
            let ctx = &modes[0].1;
            let pcfg = PipelineConfig { batch: 7, inflight: 2, params: GpuParams { emit_frames: false, ..params } };
            let mut pipe = Pipeline::new(ctx, &pcfg).unwrap();
            assert_eq!(pipe.allocated_bytes(), vram_bytes_with(&pcfg, ctx.direct_upload));
            let mut sink = Collect(vec![None; blocks.len()]);
            pipe.run(&blocks, &mut sink).unwrap();
            for (i, got) in sink.0.into_iter().enumerate() {
                assert!(got.unwrap() == compress_block(blocks[i], matching), "parse path {matching:?}: index {i}");
            }
        }
    }

    #[test]
    fn stream_frames_match_cpu_every_index_once() {
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        let distinct = distinct_blocks();
        // Huffman literals (cfg's default).
        let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, cfg(1, 1).params)).collect();
        // 1000 = 15 * 64 + 40: the last batch is partial and reuses a slot holding stale blocks.
        let blocks: Vec<&[u8]> = (0..1000).map(|i| distinct[i % distinct.len()].as_slice()).collect();

        let mut sink = CollectFrames(vec![None; blocks.len()]);
        // emit_frames: false in cfg is overridden by compress_stream_frames.
        let stats = compress_stream_frames(&ctx, &cfg(64, 3), &blocks, &mut sink).expect("compress_stream_frames");

        for (i, got) in sink.0.iter().enumerate() {
            let got = got.as_ref().unwrap_or_else(|| panic!("index {i} never delivered"));
            assert!(*got == want[i % distinct.len()], "index {i}: GPU frame != CPU frame");
        }
        assert_eq!(stats.batches, 16);
        assert_frame_timers(&ctx, &stats, "lvl3");
    }

    /// The packing readback (`GZC_PACK`) delivers the same frames: Huffman and raw literals,
    /// raw (largest) frames from the random block, partial batches, odd batch sizes.
    #[test]
    fn stream_frames_packed_match_cpu() {
        let ctx = GpuContext::with_options(true, true).expect("GPU required for gzc-gpu tests");
        if !ctx.pack_frames {
            eprintln!("skipped: no MAPPABLE_PRIMARY_BUFFERS");
            return;
        }
        let distinct = distinct_blocks();
        let blocks: Vec<&[u8]> = (0..300).map(|i| distinct[i % distinct.len()].as_slice()).collect();
        for (huffman, batch, inflight) in [(true, 64, 3), (false, 7, 1), (true, 13, 2)] {
            let params = GpuParams { matching: LVL9, emit_frames: true, huffman };
            let pcfg = PipelineConfig { batch, inflight, params };
            let mut pipe = Pipeline::new(&ctx, &pcfg).unwrap();
            assert!(pipe.pack.is_some());
            assert!(!pipe.transfer_readback(), "packing reads back on the main queue");
            assert_eq!(
                pipe.allocated_bytes(),
                vram_bytes_with(&pcfg, ctx.direct_upload),
                "packing needs no extra memory"
            );
            let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
            let mut sink = CollectFrames(vec![None; blocks.len()]);
            pipe.run_frames(&blocks, &mut sink).unwrap();
            for (i, got) in sink.0.into_iter().enumerate() {
                assert!(got.unwrap() == want[i % distinct.len()], "index {i} huffman {huffman} b{batch}");
            }
        }
    }

    /// Contexts for every upload/readback mode the adapter supports, whatever the environment:
    /// (copy upload, main-queue readback), (direct, main), (copy, transfer), (direct, transfer).
    fn mode_contexts() -> Vec<(String, GpuContext)> {
        use crate::context::GpuOptions;
        let mut out = Vec::new();
        for transfer_queue in [false, true] {
            for direct in [false, true] {
                let opts = GpuOptions { direct_upload: Some(direct), transfer_queue, ..GpuOptions::default() };
                let ctx = GpuContext::with_gpu_options(opts).expect("GPU required for gzc-gpu tests");
                if ctx.direct_upload != direct || ctx.transfer.is_some() != transfer_queue {
                    eprintln!("mode direct={direct} transfer={transfer_queue} unsupported here: skipped");
                    continue;
                }
                out.push((format!("direct={direct} transfer={transfer_queue}"), ctx));
            }
        }
        out
    }

    /// Every upload/readback mode delivers the CPU's frames, over partial batches and reuse, for
    /// the chain finder (lvl9) and the bucket-sorted finder with the segmented parse (lvl9s12seg).
    #[test]
    fn stream_frames_every_mode_matches_cpu() {
        let distinct = distinct_blocks();
        let blocks: Vec<&[u8]> = (0..200).map(|i| distinct[i % distinct.len()].as_slice()).collect();
        let modes = mode_contexts();
        assert!(modes.iter().any(|(_, c)| c.transfer.is_none() && !c.direct_upload), "the fallback mode always exists");
        for matching in [LVL9, LVL9S12SEG] {
            let params = GpuParams { matching, emit_frames: true, huffman: true };
            let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
            for (name, ctx) in &modes {
                let pcfg = PipelineConfig { batch: 23, inflight: 3, params };
                let mut pipe = Pipeline::new(ctx, &pcfg).unwrap();
                assert_eq!(pipe.transfer_readback(), ctx.transfer.is_some(), "{name}");
                assert_eq!(pipe.allocated_bytes(), vram_bytes_with(&pcfg, ctx.direct_upload), "{name}");
                for _ in 0..2 {
                    let mut sink = CollectFrames(vec![None; blocks.len()]);
                    pipe.run_frames(&blocks, &mut sink).unwrap();
                    for (i, got) in sink.0.into_iter().enumerate() {
                        assert!(got.unwrap() == want[i % distinct.len()], "{name} {matching:?}: index {i}");
                    }
                }
            }
        }
    }

    /// One transfer-readback pipeline per context (`GpuContext::transfer`): a second one errors
    /// while the first is alive, and works once it is dropped. Pipelines that do not use the
    /// transfer queue (the parse path) are not affected.
    #[test]
    fn second_transfer_pipeline_errors() {
        let Some((name, ctx)) = mode_contexts().into_iter().find(|(_, c)| c.transfer.is_some()) else {
            eprintln!("no transfer queue on this adapter: skipped");
            return;
        };
        let distinct = distinct_blocks();
        let blocks: Vec<&[u8]> = distinct.iter().map(|b| b.as_slice()).collect();
        let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
        let pcfg = PipelineConfig { params, ..cfg(7, 2) };
        let mut first = Pipeline::new(&ctx, &pcfg).unwrap();
        assert!(first.transfer_readback(), "{name}");
        let e = Pipeline::new(&ctx, &pcfg).err().expect("a second transfer pipeline must error");
        assert!(e.to_string().contains("another transfer-readback Pipeline"), "{name}: {e}");
        // The parse path does not touch the transfer queue.
        let mut parse_pipe = Pipeline::new(&ctx, &cfg(7, 1)).unwrap();
        parse_pipe.run(&blocks, &mut Collect(vec![None; blocks.len()])).unwrap();
        drop(parse_pipe);
        // The refused attempt left the first pipeline intact.
        let mut sink = CollectFrames(vec![None; blocks.len()]);
        first.run_frames(&blocks, &mut sink).unwrap();
        for (i, got) in sink.0.into_iter().enumerate() {
            assert!(got.unwrap() == cpu_frame(blocks[i], params), "{name}: index {i}");
        }
        drop(first);
        let mut second = Pipeline::new(&ctx, &pcfg).expect("the queue is free again once the first pipeline dropped");
        assert!(second.transfer_readback());
        let mut sink = CollectFrames(vec![None; blocks.len()]);
        second.run_frames(&blocks, &mut sink).unwrap();
        assert!(sink.0.iter().all(|f| f.is_some()));
    }

    /// A delivery that fails mid-stream, with batches still in flight (and, where supported, on the
    /// transfer readback), leaves the pipeline usable: the next run delivers every frame right.
    #[test]
    fn stream_frames_recovers_from_failed_delivery() {
        let distinct = distinct_blocks();
        let blocks: Vec<&[u8]> = (0..300).map(|i| distinct[i % distinct.len()].as_slice()).collect();
        let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
        let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
        for (name, ctx) in &mode_contexts() {
            let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 16, inflight: 3, params }).unwrap();
            for fail_at in [0u32, 3] {
                pipe.fail_deliveries_after = Some(fail_at);
                let mut sink = CollectFrames(vec![None; blocks.len()]);
                let e = pipe.run_frames(&blocks, &mut sink).expect_err("delivery failure must surface");
                assert!(e.to_string().contains("injected"), "{name}: {e}");
                assert!(pipe.fail_deliveries_after.is_none());
                let mut sink = CollectFrames(vec![None; blocks.len()]);
                pipe.run_frames(&blocks, &mut sink).unwrap();
                for (i, got) in sink.0.into_iter().enumerate() {
                    assert!(got.unwrap() == want[i % distinct.len()], "{name} fail_at {fail_at}: index {i}");
                }
            }
        }
    }

    #[test]
    fn stream_frames_odd_batch_single_slot_and_reuse() {
        // Raw literals (K5 writes Raw sections only): the huffman: false frame path stays covered
        // end to end.
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        let distinct = distinct_blocks();
        let blocks: Vec<&[u8]> = distinct.iter().map(|b| b.as_slice()).collect();
        let params = GpuParams { matching: LVL3, emit_frames: true, huffman: false };
        let frames_cfg = PipelineConfig { params, ..cfg(7, 1) };
        let mut pipe = Pipeline::new(&ctx, &frames_cfg).unwrap();
        // The frame path reads back through the transfer queue whenever the context has one.
        assert_eq!(pipe.transfer_readback(), ctx.transfer.is_some());
        for _ in 0..2 {
            let mut sink = CollectFrames(vec![None; blocks.len()]);
            let stats = pipe.run_frames(&blocks, &mut sink).unwrap();
            assert_eq!(stats.batches as usize, blocks.len().div_ceil(7));
            for (i, got) in sink.0.into_iter().enumerate() {
                assert!(got.unwrap() == cpu_frame(blocks[i], params), "index {i}");
            }
        }
        // The frame pipeline has no parse output, and the parse pipeline no frames.
        assert!(pipe.run(&blocks, &mut Collect(vec![None; blocks.len()])).is_err());
        let mut parse_pipe = Pipeline::new(&ctx, &cfg(7, 1)).unwrap();
        assert!(parse_pipe.run_frames(&blocks, &mut CollectFrames(vec![None; blocks.len()])).is_err());
    }

    /// Frames of one `stream_frames` run, keyed by block index; checks exactly-once and batch order.
    #[derive(Default)]
    struct Batches {
        frames: Vec<Option<Vec<u8>>>,
        /// `(first index, len)` per batch, in delivery order.
        order: Vec<(usize, usize)>,
    }

    impl Batches {
        fn take(&mut self, batch: &FrameBatch) {
            self.order.push((batch.first_index(), batch.len()));
            for (i, frame) in batch.frames() {
                if self.frames.len() <= i {
                    self.frames.resize(i + 1, None);
                }
                assert!(self.frames[i].is_none(), "index {i} delivered twice");
                self.frames[i] = Some(frame.to_vec());
            }
        }

        /// Every index below `n` once, equal to `want(i)`; batches in submission order, contiguous.
        fn check(&self, n: usize, want: impl Fn(usize) -> Vec<u8>, what: &str) {
            assert_eq!(self.frames.len(), n, "{what}");
            for (i, f) in self.frames.iter().enumerate() {
                assert!(*f.as_ref().unwrap_or_else(|| panic!("{what}: index {i} never delivered")) == want(i), "{what}: index {i}");
            }
            let mut next = 0;
            for &(first, len) in &self.order {
                assert_eq!(first, next, "{what}: batches out of order: {:?}", self.order);
                next += len;
            }
        }
    }

    /// The zero-copy API in every upload/readback mode: the producer writes blocks straight into
    /// the slots (block by block, and some partial batches as a flush timer would submit), the
    /// sink hands each `FrameBatch` to a writer thread that releases it later, and slots are
    /// recycled many times over (far more batches than slots). Frames match the CPU's, each index
    /// once, batches in submission order; the pipeline then runs again.
    #[test]
    fn stream_frames_zero_copy_every_mode() {
        let distinct = distinct_blocks();
        // M5 T5: the optimal parse too (its K3opt reads one word past each block: the slot's
        // trailing zero word after partial batches of stale blocks). opt14 only implements at
        // blocks of at most 64 KiB.
        for matching in [LVL9S12SEG, OPT14] {
            if matching.opt.is_some() && gzc_core::config::LOG2_BLOCK > 16 {
                continue;
            }
            zero_copy_every_mode(&distinct, GpuParams { matching, emit_frames: true, huffman: true });
        }
    }

    fn zero_copy_every_mode(distinct: &[Vec<u8>], params: GpuParams) {
        let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
        // Batch sizes per submission: full (16), and partial ones as a flush timer would send.
        let sizes = [16usize, 5, 16, 1, 16, 16, 9, 16, 16, 16, 3, 16];
        let total: usize = sizes.iter().sum();
        for (name, ctx) in &mode_contexts() {
            let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 16, inflight: 3, params }).unwrap();
            for round in 0..2 {
                let (tx, rx) = mpsc::channel::<FrameBatch>();
                let collected = std::thread::scope(|s| {
                    // The writer: takes each batch, copies its frames, drops it (releasing the slot).
                    let writer = s.spawn(move || {
                        let mut got = Batches::default();
                        for batch in rx {
                            std::thread::sleep(std::time::Duration::from_millis(2));
                            got.take(&batch);
                        }
                        got
                    });
                    let stats = pipe
                        .stream_frames(
                            move |batch| {
                                tx.send(batch).map_err(|_| anyhow!("writer gone"))?;
                                Ok(())
                            },
                            |stream| {
                                assert_eq!(stream.slot_capacity(), 16);
                                for &n in &sizes {
                                    let first = stream.submitted_blocks();
                                    let mut slot = stream.next_upload_slot()?;
                                    for k in 0..n {
                                        // SAFETY: native wgpu-core backend (`blocks_mut`'s
                                        // contract; pinned by `upload_slot_bytes_are_initialized`).
                                        let block = unsafe { slot.block_mut(k) };
                                        block.copy_from_slice(&distinct[(first + k) % distinct.len()]);
                                    }
                                    assert_eq!(slot.submit(n)?, first);
                                }
                                Ok(())
                            },
                        )
                        .unwrap_or_else(|e| panic!("{name}: {e:#}"));
                    assert_eq!(stats.batches as usize, sizes.len(), "{name}");
                    writer.join().unwrap()
                });
                assert_eq!(collected.order.len(), sizes.len(), "{name}");
                collected.check(total, |i| want[i % distinct.len()].clone(), &format!("{name} round {round}"));
            }
        }
    }

    /// Lease lifetimes under stress, in every mode and with both completion waits (`poll_only`,
    /// Metal's): the sink hands every `FrameBatch` to a holder thread that keeps up to
    /// `inflight - 1` of them and drops them out of order after random delays (a batch whenever
    /// none arrives for up to a millisecond: the producer may be waiting for it), and still holds
    /// the last ones when the producer finishes and the completion thread closes the channel
    /// (the stream's end waits for them, `wait_released`); some streams fail mid-way
    /// (`fail_deliveries_after`), so `abandon` runs while batches are held. Every frame read
    /// right before its batch drops equals the CPU's, every stream's errors are its own, and the
    /// pipeline is reused and dropped right after a stream whose last batches another thread
    /// released.
    #[test]
    fn stream_frames_leases_held_across_stream_end() {
        let distinct = distinct_blocks();
        let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
        let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
        let blocks: Vec<&[u8]> = (0..120).map(|i| distinct[i % distinct.len()].as_slice()).collect();
        let mut seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64 | 1;
        eprintln!("seed {seed}");
        for (name, ctx) in &mode_contexts() {
            let poll_modes: &[bool] = if ctx.transfer.is_some() { &[false] } else { &[false, true] };
            for &poll_only in poll_modes {
                let inflight = 3;
                let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 4, inflight, params }).unwrap();
                pipe.poll_only = poll_only;
                for round in 0..6 {
                    let fail_at = (round % 3 == 2).then_some(round as u32 * 2);
                    pipe.fail_deliveries_after = fail_at;
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let mut rng = seed;
                    let (tx, rx) = mpsc::channel::<FrameBatch>();
                    let what = format!("{name} poll_only={poll_only} round {round}");
                    let (r, seen) = std::thread::scope(|s| {
                        let want = &want;
                        let what = &what;
                        let holder = s.spawn(move || {
                            let mut next = || {
                                rng ^= rng << 13;
                                rng ^= rng >> 7;
                                rng ^= rng << 17;
                                rng
                            };
                            let mut held: Vec<FrameBatch> = Vec::new();
                            let mut seen = 0;
                            let release = |b: FrameBatch, seen: &mut usize| {
                                for (i, f) in b.frames() {
                                    assert!(f == want[i % want.len()], "{what}: index {i}");
                                }
                                *seen += b.len();
                                drop(b);
                            };
                            // The producer reuses the slots in turn, so holding the oldest
                            // batch stalls it: release a random held batch whenever none
                            // arrives for a while, and whenever `inflight - 1` are held.
                            loop {
                                match rx.recv_timeout(std::time::Duration::from_micros(300 + next() % 700)) {
                                    Ok(b) => held.push(b),
                                    Err(mpsc::RecvTimeoutError::Timeout) if !held.is_empty() => {
                                        let k = next() as usize % held.len();
                                        release(held.swap_remove(k), &mut seen);
                                    }
                                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                                }
                                while held.len() >= inflight as usize {
                                    let k = next() as usize % held.len();
                                    std::thread::sleep(std::time::Duration::from_micros(next() % 500));
                                    release(held.swap_remove(k), &mut seen);
                                }
                            }
                            // The channel closed: the producer is done and the stream waits for
                            // these.
                            std::thread::sleep(std::time::Duration::from_millis(5));
                            while !held.is_empty() {
                                let k = next() as usize % held.len();
                                std::thread::sleep(std::time::Duration::from_micros(next() % 500));
                                release(held.swap_remove(k), &mut seen);
                            }
                            seen
                        });
                        let r = pipe.stream_frames(
                            move |b| tx.send(b).map_err(|_| anyhow!("holder gone")),
                            |stream| stream.upload_blocks(&blocks),
                        );
                        (r, holder.join().unwrap())
                    });
                    match fail_at {
                        Some(_) => {
                            let e = format!("{:#}", r.expect_err("injected delivery failure"));
                            assert!(e.contains("injected delivery failure"), "{what}: {e}");
                        }
                        None => {
                            let stats = r.unwrap_or_else(|e| panic!("{what}: {e:#}"));
                            assert_eq!(stats.batches as usize, blocks.len().div_ceil(4), "{what}");
                            assert_eq!(seen, blocks.len(), "{what}");
                        }
                    }
                }
                // Dropped right after a stream whose last batches the holder thread released.
                drop(pipe);
            }
        }
    }

    /// A device lost mid-stream (here `Device::destroy`; on Metal, wgpu-hal's fence race in
    /// `Device::wait` loses the device from a `poll`): wgpu-core then destroys every buffer, so
    /// `abandon`'s unmap of the in-flight staging buffers raises "Buffer with 'pipeline.staging'
    /// label has been destroyed" in the stream's error scope. The stream must report the failure
    /// itself (the lost device), with that validation error only as context.
    #[test]
    fn stream_frames_reports_device_loss_not_the_cleanup_error() {
        use crate::context::GpuOptions;
        let opts = GpuOptions { direct_upload: Some(false), transfer_queue: false, ..GpuOptions::default() };
        let ctx = GpuContext::with_gpu_options(opts).expect("GPU required for gzc-gpu tests");
        let distinct = distinct_blocks();
        let blocks: Vec<&[u8]> = (0..400).map(|i| distinct[i % distinct.len()].as_slice()).collect();
        let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
        let mut pipe = Pipeline::new(&ctx, &PipelineConfig { batch: 8, inflight: 3, params }).unwrap();
        let mut delivered = 0;
        let e = pipe
            .stream_frames(
                |_batch| {
                    delivered += 1;
                    if delivered == 2 {
                        ctx.device.destroy();
                    }
                    Ok(())
                },
                |stream| stream.upload_blocks(&blocks),
            )
            .expect_err("a stream on a lost device fails");
        let msg = format!("{e:#}");
        eprintln!("stream error: {msg}");
        assert!(!msg.starts_with("wgpu validation error"), "the cleanup's validation error masks the cause: {msg}");
    }

    /// Payloads of arbitrary size spanning several blocks (and one of 0 bytes), written from
    /// several threads into disjoint regions of a slot that holds stale bytes, then padded: the
    /// frames equal the CPU's for `chunk_file` of each payload, and `payload_real_lens` gives
    /// `chunk_file`'s real lengths.
    #[test]
    fn stream_frames_multi_block_payloads() {
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        let params = GpuParams { matching: LVL9S12SEG, emit_frames: true, huffman: true };
        let source: Vec<u8> = test_cases().into_iter().flat_map(|(_, bytes)| bytes).collect();
        let bs = BLOCK_SIZE;
        let sizes = [3 * bs + 100, 2 * bs, 1, 0, 5 * bs - 1, bs + 7];
        let mut at = 0;
        let payloads: Vec<&[u8]> = sizes
            .iter()
            .map(|&n| {
                let p = &source[at % (source.len() - 6 * bs)..][..n];
                at += n + 12345;
                p
            })
            .collect();
        let (mut want, mut want_lens) = (Vec::new(), Vec::new());
        for p in &payloads {
            let blocks = chunk_file(p);
            want_lens.extend(blocks.iter().map(|b| b.real_len));
            want.extend(blocks.iter().map(|b| cpu_frame(&b.data, params)));
            assert_eq!(payload_blocks(p.len()), blocks.len());
        }
        let lens: Vec<usize> = payloads.iter().flat_map(|p| payload_real_lens(p.len())).collect();
        assert_eq!(lens, want_lens, "per-block real lengths");
        let blocks_each: Vec<usize> = payloads.iter().map(|p| payload_blocks(p.len())).collect();
        let total: usize = blocks_each.iter().sum();

        let mut pipe = Pipeline::new(&ctx, &PipelineConfig { batch: 16, inflight: 2, params }).unwrap();
        // Dirty both slots.
        pipe.stream_frames(
            |_batch| Ok(()),
            |stream| {
                for _ in 0..2 {
                    let mut slot = stream.next_upload_slot()?;
                    slot.regions_mut(&[16])?[0].write_only().fill(0xAB);
                    slot.submit(16)?;
                }
                Ok(())
            },
        )
        .unwrap();
        let mut got = Batches::default();
        pipe.stream_frames(
            |batch| {
                got.take(&batch);
                Ok(())
            },
            |stream| {
                for _ in 0..2 {
                    let mut slot = stream.next_upload_slot()?;
                    assert!(slot.regions_mut(&[slot.capacity() + 1]).is_err());
                    let regions = slot.regions_mut(&blocks_each)?;
                    std::thread::scope(|s| {
                        for (mut region, p) in regions.into_iter().zip(&payloads) {
                            s.spawn(move || {
                                region.write(0, p);
                                assert_eq!(region.pad(p.len()).unwrap(), payload_blocks(p.len()));
                                assert!(region.pad(region.len() + 1).is_err());
                            });
                        }
                    });
                    slot.submit(total)?;
                }
                Ok(())
            },
        )
        .unwrap();
        got.check(2 * total, |i| want[i % total].clone(), "payloads");
    }

    /// Pins what `UploadSlot::blocks_mut`'s safety relies on, in every upload/readback mode: the
    /// mapping is real host memory whose bytes are initialized: a new slot reads back all zeros,
    /// and a reused one reads back exactly what was written into it before (and zeros where
    /// nothing was), not garbage. Recheck on every wgpu upgrade.
    #[test]
    fn upload_slot_bytes_are_initialized() {
        let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
        let bs = BLOCK_SIZE;
        for (name, ctx) in &mode_contexts() {
            let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 4, inflight: 2, params }).unwrap();
            pipe.stream_frames(
                |_batch| Ok(()),
                |stream| {
                    // First use of both slots: zeros. Write block 0 only, submit it.
                    for s in 0..2u8 {
                        let mut slot = stream.next_upload_slot()?;
                        // SAFETY: native wgpu-core backend; this test pins the contract.
                        let bytes = unsafe { slot.blocks_mut() };
                        assert!(bytes.iter().all(|&b| b == 0), "{name}: new slot {s} not zeroed");
                        bytes[..bs].fill(0x50 + s);
                        slot.submit(1)?;
                    }
                    // Reused: block 0 as written, the trailing zero word `submit` writes at
                    // block 1's start, the rest still zero.
                    for s in 0..2u8 {
                        let mut slot = stream.next_upload_slot()?;
                        // SAFETY: as above.
                        let bytes = unsafe { slot.blocks_mut() };
                        assert!(bytes[..bs].iter().all(|&b| b == 0x50 + s), "{name}: slot {s} lost its bytes");
                        assert!(bytes[bs..].iter().all(|&b| b == 0), "{name}: slot {s} holds garbage");
                        // Submitted, not dropped: a dropped slot would be handed out again.
                        slot.submit(1)?;
                    }
                    Ok(())
                },
            )
            .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        }
    }

    /// A submission that fails after its slot was marked in flight aborts the stream: the
    /// producer's later calls error instead of waiting for a slot that never comes free, even if
    /// it swallowed the first error; `stream_frames` returns the error; the pipeline is reusable.
    #[test]
    fn failed_submit_is_sticky() {
        let distinct = distinct_blocks();
        let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
        let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
        let blocks: Vec<&[u8]> = (0..64).map(|i| distinct[i % distinct.len()].as_slice()).collect();
        for (name, ctx) in &mode_contexts() {
            let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 16, inflight: 3, params }).unwrap();
            pipe.fail_submits_after = Some(1);
            let e = pipe
                .stream_frames(
                    |_batch| Ok(()),
                    |stream| {
                        stream.upload_blocks(&blocks[..16])?;
                        let e = stream.upload_blocks(&blocks[16..32]).expect_err("second submit fails");
                        assert!(e.to_string().contains("injected submit"), "{name}: {e:#}");
                        // Swallowed: every later call errors at once.
                        for _ in 0..4 {
                            let e = stream.next_upload_slot().err().expect("aborted stream hands out no slot");
                            assert!(e.to_string().contains("aborted"), "{name}: {e:#}");
                        }
                        Ok(())
                    },
                )
                .expect_err("a swallowed submit failure still fails the stream");
            assert!(format!("{e:#}").contains("injected submit"), "{name}: {e:#}");
            let mut sink = CollectFrames(vec![None; blocks.len()]);
            pipe.run_frames(&blocks, &mut sink).unwrap();
            for (i, got) in sink.0.into_iter().enumerate() {
                assert!(got.unwrap() == want[i % distinct.len()], "{name}: index {i}");
            }
        }
    }

    /// `submit_with` tags reach `FrameBatch::tag` with their batch; `submit` tags 0.
    #[test]
    fn frame_batches_carry_their_tags() {
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        let distinct = distinct_blocks();
        let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
        let mut pipe = Pipeline::new(&ctx, &PipelineConfig { batch: 8, inflight: 3, params }).unwrap();
        let mut seen = Vec::new();
        pipe.stream_frames(
            |batch| {
                seen.push((batch.first_index(), batch.len(), batch.tag()));
                Ok(())
            },
            |stream| {
                for (k, n) in [8usize, 3, 8, 5, 1, 8].into_iter().enumerate() {
                    let mut slot = stream.next_upload_slot()?;
                    for (b, mut region) in slot.regions_mut(&vec![1; n])?.into_iter().enumerate() {
                        region.write(0, &distinct[(k + b) % distinct.len()]);
                    }
                    if k == 4 {
                        slot.submit(n)?;
                    } else {
                        slot.submit_with(n, 1000 + k as u64)?;
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(seen, [(0, 8, 1000), (8, 3, 1001), (11, 8, 1002), (19, 5, 1003), (24, 1, 0), (25, 8, 1005)]);
    }

    /// A batch kept alive holds its slot: its frames stay intact while later batches reuse the
    /// other slots, and the stream only waits for it once it needs that slot again. Holding
    /// `inflight - 1` batches does not stall. An upload slot dropped without a submit is handed
    /// out again.
    #[test]
    fn stream_frames_held_batches_keep_their_bytes() {
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        let distinct = distinct_blocks();
        let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
        let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
        let blocks: Vec<&[u8]> = (0..8 * 10).map(|i| distinct[i % distinct.len()].as_slice()).collect();
        let mut pipe = Pipeline::new(&ctx, &PipelineConfig { batch: 8, inflight: 3, params }).unwrap();
        let (tx, rx) = mpsc::channel::<FrameBatch>();
        let got = std::thread::scope(|s| {
            // Keeps the last two batches; releases the rest once the channel closes, which happens
            // when the pipeline drops `on_batch` (before `stream_frames` waits for the releases).
            let writer = s.spawn(move || {
                let mut held = std::collections::VecDeque::new();
                let mut got = Batches::default();
                let mut check = |(batch, snap): (FrameBatch, Vec<Vec<u8>>)| {
                    for (k, f) in snap.iter().enumerate() {
                        assert!(batch.frame(k) == f.as_slice(), "held batch {} changed", batch.first_index());
                    }
                    got.take(&batch);
                };
                for batch in rx {
                    // Snapshot at delivery; compare again once two more batches went by.
                    let snap: Vec<Vec<u8>> = (0..batch.len()).map(|k| batch.frame(k).to_vec()).collect();
                    held.push_back((batch, snap));
                    if held.len() > 2 {
                        check(held.pop_front().unwrap());
                    }
                }
                held.into_iter().for_each(&mut check);
                got
            });
            let stats = pipe
                .stream_frames(
                    move |batch| tx.send(batch).map_err(|_| anyhow!("writer gone")),
                    |stream| {
                        // Taken and dropped without a submit: the same slot comes back.
                        drop(stream.next_upload_slot()?);
                        stream.upload_blocks(&blocks)
                    },
                )
                .unwrap();
            assert_eq!(stats.batches, 10);
            writer.join().unwrap()
        });
        got.check(blocks.len(), |i| want[i % distinct.len()].clone(), "held");
    }

    /// Errors mid-stream, from either side, surface from `stream_frames`, stop the other side and
    /// leave the pipeline usable (every upload/readback mode): the sink failing on its third
    /// batch; the producer failing after four submissions; a bad submit size; a panicking sink.
    #[test]
    fn stream_frames_errors_mid_stream() {
        let distinct = distinct_blocks();
        let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
        let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
        let blocks: Vec<&[u8]> = (0..200).map(|i| distinct[i % distinct.len()].as_slice()).collect();
        for (name, ctx) in &mode_contexts() {
            let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 16, inflight: 3, params }).unwrap();
            // The sink fails on its third batch: the producer, blocked or not, gets an error.
            let mut seen = 0;
            let e = pipe
                .stream_frames(
                    |_batch| {
                        seen += 1;
                        anyhow::ensure!(seen < 3, "sink failed");
                        Ok(())
                    },
                    |stream| stream.upload_blocks(&blocks),
                )
                .expect_err("sink error must surface");
            assert!(e.to_string().contains("sink failed"), "{name}: {e:#}");
            assert_eq!(seen, 3, "{name}: no delivery after the failing one");
            // The producer fails after four submissions; batches may still be in flight.
            let e = pipe
                .stream_frames(
                    |_batch| Ok(()),
                    |stream| {
                        stream.upload_blocks(&blocks[..64])?;
                        anyhow::bail!("producer failed")
                    },
                )
                .expect_err("producer error must surface");
            assert!(e.to_string().contains("producer failed"), "{name}: {e:#}");
            // Bad submit sizes are refused, and the slot stays usable.
            let e = pipe
                .stream_frames(
                    |_batch| Ok(()),
                    |stream| {
                        let slot = stream.next_upload_slot()?;
                        assert!(slot.capacity() == 16);
                        slot.submit(17)
                    }
                    .map(|_| ()),
                )
                .expect_err("17 > capacity");
            assert!(e.to_string().contains("not in 1..=16"), "{name}: {e:#}");
            let e = pipe
                .stream_frames(|_batch| Ok(()), |stream| stream.next_upload_slot()?.submit(0).map(|_| ()))
                .expect_err("0 blocks");
            assert!(e.to_string().contains("not in 1..=16"), "{name}: {e:#}");
            // A panic on either side propagates without leaving the other side waiting.
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pipe.stream_frames(|_batch| panic!("sink panicked"), |stream| stream.upload_blocks(&blocks))
            }));
            assert!(r.is_err(), "{name}: the sink's panic propagates");
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pipe.stream_frames(
                    |_batch| Ok(()),
                    |stream| {
                        stream.upload_blocks(&blocks[..48])?;
                        panic!("producer panicked")
                    },
                )
            }));
            assert!(r.is_err(), "{name}: the producer's panic propagates");
            // After all of that the same pipeline delivers every frame right.
            let mut sink = CollectFrames(vec![None; blocks.len()]);
            pipe.run_frames(&blocks, &mut sink).unwrap();
            for (i, got) in sink.0.into_iter().enumerate() {
                assert!(got.unwrap() == want[i % distinct.len()], "{name}: index {i}");
            }
        }
    }

    /// Parallel delivery (`run_frames_par`): every index exactly once and right, whatever the
    /// thread count (more threads than frames in a batch included), and actually from several
    /// threads.
    #[test]
    fn run_frames_par_every_index_once() {
        use std::sync::atomic::AtomicU32;
        struct Par {
            hits: Vec<AtomicU32>,
            frames: Vec<Mutex<Vec<u8>>>,
            threads: Mutex<std::collections::HashSet<std::thread::ThreadId>>,
        }
        impl ParFrameSink for Par {
            fn put(&self, index: usize, frame: &[u8]) {
                self.hits[index].fetch_add(1, Ordering::Relaxed);
                *self.frames[index].lock().unwrap() = frame.to_vec();
                self.threads.lock().unwrap().insert(std::thread::current().id());
            }
        }
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        let distinct = distinct_blocks();
        let params = GpuParams { matching: LVL9, emit_frames: true, huffman: true };
        let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
        let blocks: Vec<&[u8]> = (0..150).map(|i| distinct[i % distinct.len()].as_slice()).collect();
        let mut pipe = Pipeline::new(&ctx, &PipelineConfig { batch: 32, inflight: 2, params }).unwrap();
        for threads in [1, 3, 8, 64] {
            let sink = Par {
                hits: (0..blocks.len()).map(|_| AtomicU32::new(0)).collect(),
                frames: (0..blocks.len()).map(|_| Mutex::new(Vec::new())).collect(),
                threads: Default::default(),
            };
            let stats = pipe.run_frames_par(&blocks, &sink, threads).unwrap();
            assert_eq!(stats.batches, 5);
            for i in 0..blocks.len() {
                assert_eq!(sink.hits[i].load(Ordering::Relaxed), 1, "threads {threads}: index {i}");
                assert!(*sink.frames[i].lock().unwrap() == want[i % distinct.len()], "threads {threads}: index {i}");
            }
            let used = sink.threads.lock().unwrap().len();
            assert!(if threads == 1 { used == 1 } else { used > 1 }, "threads {threads}: {used} delivery threads");
        }
    }
}
