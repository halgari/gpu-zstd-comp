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
//!   (mappable) staging buffer; the host only copies each frame's bytes out. K5 (the literals
//!   section, Huffman-coded with `GpuParams::huffman`) gathers the literals from `data` and writes
//!   into the same `frames` / `frame_len` buffers, so it adds no memory.
use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{Context as _, anyhow};
use gzc_core::config::BLOCK_SIZE;
use gzc_core::seq::BlockOutput;

use crate::compressor::{
    BatchBuffers, FRAME_STRIDE, GpuParams, KERNEL_QUERIES, Kernels, MAX_SEQS, counts_bytes, data_bytes, decode_output,
    frame_bytes, frame_len_bytes, frames_bytes, k4_tables_bytes, max_batch_blocks, scratch_bytes, seqs_bytes,
    slot_bytes,
};
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
    /// empty when the device has no timestamp queries.
    pub kernel_ms: Vec<(String, f64)>,
    /// Wall time from the first upload to the last block handed to the sink.
    pub wall_s: f64,
    /// Number of batches submitted.
    pub batches: u32,
    /// Where the time outside the kernels goes, summed over all batches, in milliseconds (see
    /// `TRANSFER_NAMES`). The `gpu_*` entries come from timestamps (left out without them); the
    /// `host_*` ones are wall time on the pipeline thread.
    pub transfer_ms: Vec<(String, f64)>,
}

/// Names of `PipelineStats::transfer_ms`, in order:
/// - `gpu_upload_copy`: the upload -> data copy (the batch's start marker to K1's begin).
/// - `gpu_readback`: the output -> staging copies, or the pack kernel (K4's end, or K3's on
///   the parse path, to the batch's end marker).
/// - `gpu_idle`: gaps between one batch's end marker and the next batch's start marker. This
///   includes the previous batch's timestamp resolve and its copy into staging (recorded after
///   its end marker; a few µs) and any barrier work at the start of a submission.
/// - `host_upload_wait`: waiting for a slot's upload buffer to be mapped again.
/// - `host_upload_write`: writing the blocks into the mapped upload buffer, plus unmap.
/// - `host_submit`: recording and submitting the batch, plus the map requests.
/// - `host_wait`: blocked in `device.poll` for the oldest batch.
/// - `host_deliver`: handing a completed batch to the sink.
/// - `host_unmap`: unmapping the staging buffer.
/// - `host_fill`: from the start to the first submission.
/// - `host_drain`: from the last `device.poll` returning to the end.
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

/// Accumulators behind `PipelineStats::transfer_ms`: GPU ticks and host seconds.
#[derive(Default)]
struct Profile {
    /// Kernel ticks, per `Kernels::names`.
    ticks: Vec<u64>,
    upload_copy: u64,
    readback: u64,
    idle: u64,
    /// End marker of the last batch finished (ticks), for `idle`.
    last_end: Option<u64>,
    upload_wait: f64,
    upload_write: f64,
    submit: f64,
    wait: f64,
    deliver: f64,
    unmap: f64,
    fill: f64,
    drain: f64,
}

/// Receives each block's parse; called exactly once per index, in arbitrary order.
pub trait BlockSink {
    fn put(&mut self, index: usize, out: BlockOutput);
}

/// Receives each block's complete zstd frame; called exactly once per index, in arbitrary order.
/// `frame` borrows the mapped staging buffer: copy it out before returning.
pub trait FrameSink {
    fn put(&mut self, index: usize, frame: &[u8]);
}

/// Timestamp queries per slot: the kernels' begin/end pairs, then a start and an end marker (see
/// `Pipeline::submit`).
const PIPELINE_QUERIES: u32 = KERNEL_QUERIES + 2;

/// Alignment of each region of a slot's staging buffer (see `StagingLayout::new`).
const STAGING_ALIGN: u64 = 256;

/// Byte offsets of one slot's staging buffer, laid out for `cap` blocks. Parse path:
/// `[counts][seqs][timestamps]`; frame path: `[frame_len][frames][timestamps]`; the regions keep
/// the GPU buffers' fixed per-block stride (packed frames need at most that much).
#[derive(Clone, Copy)]
struct StagingLayout {
    frames: bool,
    /// Parse path: seqs region. Frame path: frames region.
    a: u64,
    ts: u64,
    size: u64,
}

impl StagingLayout {
    fn new(cap: u32, frames: bool) -> Self {
        // Every region starts STAGING_ALIGN-aligned: a GPU->staging copy to a destination that is
        // only 4-byte aligned runs several times slower (RTX 5090 / Vulkan: 6-10 ms more per
        // batch for the ~200 MB frames region), which showed up as a batch-size-dependent loss.
        let al = |x: u64| x.next_multiple_of(STAGING_ALIGN);
        let (a, ts) = if frames {
            let a = al(frame_len_bytes(cap));
            (a, al(a + frames_bytes(cap)))
        } else {
            let a = al(counts_bytes(cap));
            (a, al(a + seqs_bytes(cap)))
        };
        Self { frames, a, ts, size: ts + PIPELINE_QUERIES as u64 * wgpu::QUERY_SIZE as u64 }
    }
}

/// Hands one completed batch to the caller's sink: `(first block index, block count, mapped
/// staging bytes)`.
type Deliver<'d> = dyn FnMut(usize, u32, &[u8]) -> anyhow::Result<()> + 'd;

/// A submitted batch awaiting its staging map.
struct Job {
    first: usize,
    n: u32,
    submission: wgpu::SubmissionIndex,
    /// The staging map's result (wgpu staging), or None (transfer readback: wait for `seq`).
    mapped: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    /// Transfer readback: the batch's value on the `Xfer` timelines.
    seq: u64,
}

/// A slot's staging buffer: mapped through wgpu after the submission, or (transfer readback)
/// host memory the transfer queue writes and that stays mapped.
enum Staging {
    Wgpu(wgpu::Buffer),
    Host(Box<RawBuffer>),
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

struct Slot {
    /// Persistent upload buffer (MAP_WRITE | COPY_SRC, `data_bytes(batch)`): the host writes a
    /// batch into it while it is mapped, and the submission copies it into the shared `data` (direct
    /// upload: MAP_WRITE | STORAGE, bound as `data` itself).
    /// After each submission it is re-mapped; `upload_mapped` receives that map's result (None:
    /// mapped).
    upload: wgpu::Buffer,
    upload_mapped: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    /// The submission that last used `upload` (to wait for its re-map).
    upload_submission: Option<wgpu::SubmissionIndex>,
    staging: Staging,
    /// Query set and its resolve buffer, when timestamps are enabled.
    queries: Option<Queries>,
    job: Option<Job>,
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
    t_done: Timeline,
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
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pack_frames"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pack_frames"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
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
        let layout = StagingLayout::new(cfg.batch, frames);
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
                t_done: tq.timeline()?,
                next_seq: 1,
                tq,
            }),
            None => None,
        };
        let mut slots = Vec::with_capacity(cfg.inflight as usize);
        for _ in 0..cfg.inflight {
            let staging = match &xfer {
                Some(x) => Staging::Host(Box::new(x.tq.buffer(layout.size, U::TRANSFER_DST, true)?)),
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
            slots.push(Slot {
                upload: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pipeline.upload"),
                    size: data_bytes(cfg.batch),
                    usage: upload_usage,
                    mapped_at_creation: true,
                }),
                upload_mapped: None,
                upload_submission: None,
                staging,
                queries,
                job: None,
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
            slots,
            pack,
            direct,
            xfer,
            #[cfg(test)]
            fail_deliveries_after: None,
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
        let shared = data
            + [&s.head, &s.pred, &s.best, &s.seqs, &s.counts].iter().map(|b| b.size()).sum::<u64>()
            + opt(&s.frames)
            + opt(&s.frame_len);
        let per_slot: u64 = self.slots.iter().map(|slot| slot.upload.size() + slot.staging.size()).sum();
        // Transfer readback: frames / frame_len are imports of `Xfer`'s buffers (same sizes).
        shared + per_slot + self.kernels.own_buffer_bytes()
    }

    /// Streams `blocks` (each BLOCK_SIZE bytes) through the slots, handing every block's parse
    /// to `sink` exactly once. Errors on a frame-path pipeline, and on wgpu validation or
    /// out-of-memory errors.
    pub fn run(&mut self, blocks: &[&[u8]], sink: &mut impl BlockSink) -> anyhow::Result<PipelineStats> {
        anyhow::ensure!(!self.layout.frames, "pipeline built with emit_frames: use run_frames");
        let layout = self.layout;
        self.run_with(blocks, &mut |first, n, view| {
            let words: &[u32] = bytemuck::cast_slice(view);
            let seq_stride = 3 * MAX_SEQS as usize;
            for b in 0..n as usize {
                let (n_seq, n_lit) = (words[2 * b], words[2 * b + 1]);
                anyhow::ensure!(
                    n_seq <= MAX_SEQS && n_lit as usize <= BLOCK_SIZE,
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
        })
    }

    /// Streams `blocks` (each BLOCK_SIZE bytes) through the slots, handing every block's zstd
    /// frame (K4 output) to `sink` exactly once. Errors on a parse-path pipeline, and on wgpu
    /// validation or out-of-memory errors.
    pub fn run_frames(&mut self, blocks: &[&[u8]], sink: &mut impl FrameSink) -> anyhow::Result<PipelineStats> {
        anyhow::ensure!(self.layout.frames, "pipeline built without emit_frames: use run");
        let layout = self.layout;
        let packed = self.pack.is_some();
        self.run_with(blocks, &mut |first, n, view| {
            let lens: &[u32] = bytemuck::cast_slice(&view[..frame_len_bytes(n) as usize]);
            let frames = &view[layout.a as usize..layout.a as usize + n as usize * FRAME_STRIDE];
            // Packed: each frame starts at the PACK_ALIGN boundary after the previous one, so
            // (with every length at most FRAME_STRIDE) all of them lie inside `frames`.
            let mut at = 0usize;
            for (b, &len) in lens.iter().enumerate() {
                let frame = if packed {
                    let ok = len > 0 && len as usize <= FRAME_STRIDE;
                    anyhow::ensure!(ok, "block {}: bad frame length {len}", first + b);
                    let f = &frames[at..at + len as usize];
                    at += (len as usize).next_multiple_of(PACK_ALIGN);
                    f
                } else {
                    frame_bytes(frames, b, len).with_context(|| format!("block {}", first + b))?
                };
                sink.put(first + b, frame);
            }
            Ok(())
        })
    }

    fn run_with(&mut self, blocks: &[&[u8]], deliver: &mut Deliver<'_>) -> anyhow::Result<PipelineStats> {
        if let Some(i) = blocks.iter().position(|b| b.len() != BLOCK_SIZE) {
            anyhow::bail!("block {i} is {} bytes, expected BLOCK_SIZE {BLOCK_SIZE}", blocks[i].len());
        }
        let scopes = ErrorScopes::push(self.ctx);
        let start = Instant::now();
        let names = self.kernels.names();
        let mut prof = Profile { ticks: vec![0u64; names.len()], ..Default::default() };
        let result = self.stream(blocks, deliver, &mut prof, start);
        let wall_s = start.elapsed().as_secs_f64();
        if result.is_err() {
            self.abandon();
        }
        scopes.pop()?;
        let batches = result?;

        let (kernel_ms, gpu_ms) = if self.ctx.timestamps {
            let period_ns = self.ctx.queue.get_timestamp_period() as f64;
            let ms = |t: u64| t as f64 * period_ns / 1e6;
            (
                names.iter().zip(&prof.ticks).map(|(name, &t)| (name.to_string(), ms(t))).collect(),
                vec![ms(prof.upload_copy), ms(prof.readback), ms(prof.idle)],
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let host_ms = [
            prof.upload_wait,
            prof.upload_write,
            prof.submit,
            prof.wait,
            prof.deliver,
            prof.unmap,
            prof.fill,
            prof.drain,
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

    /// The submit / wait / deliver loop; returns the number of batches submitted.
    fn stream(
        &mut self,
        blocks: &[&[u8]],
        deliver: &mut Deliver<'_>,
        prof: &mut Profile,
        start: Instant,
    ) -> anyhow::Result<u32> {
        let batch = self.cfg.batch as usize;
        let mut next = 0usize;
        let mut batches = 0u32;
        let mut last_wait = start;
        // Busy slots, oldest submission first.
        let mut busy: VecDeque<usize> = VecDeque::new();
        loop {
            while next < blocks.len() {
                let Some(i) = self.slots.iter().position(|s| s.job.is_none()) else { break };
                let n = batch.min(blocks.len() - next);
                self.submit(i, next, &blocks[next..next + n], prof)?;
                if batches == 0 {
                    prof.fill = start.elapsed().as_secs_f64();
                }
                busy.push_back(i);
                next += n;
                batches += 1;
            }
            let Some(&oldest) = busy.front() else { break };
            // Every slot is busy (or nothing is left to submit): block on the oldest batch...
            let job = self.slots[oldest].job.as_ref().unwrap();
            let t = Instant::now();
            match &self.xfer {
                Some(x) => {
                    x.tq.wait(&x.t_done, job.seq)?;
                    // Deliver the upload buffers' map callbacks of completed submissions.
                    self.ctx.device.poll(wgpu::PollType::Poll).context("device poll")?;
                }
                None => {
                    let submission = job.submission.clone();
                    self.ctx
                        .device
                        .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: None })
                        .context("device poll")?;
                }
            }
            last_wait = Instant::now();
            prof.wait += (last_wait - t).as_secs_f64();
            // ...then drain it and every later batch that has also completed.
            let mut waited_for = true;
            while let Some(&i) = busy.front() {
                let job = self.slots[i].job.as_ref().unwrap();
                let ready = match (&self.xfer, &job.mapped) {
                    (Some(x), _) => waited_for || x.tq.value(&x.t_done)? >= job.seq,
                    (None, Some(rx)) => match rx.try_recv() {
                        Ok(r) => {
                            r.context("map staging buffer")?;
                            true
                        }
                        Err(mpsc::TryRecvError::Empty) if !waited_for => false,
                        Err(_) => anyhow::bail!("staging map callback not delivered after waiting for its submission"),
                    },
                    (None, None) => unreachable!("wgpu staging without a map request"),
                };
                if !ready {
                    break;
                }
                self.finish(i, deliver, prof)?;
                busy.pop_front();
                waited_for = false;
            }
        }
        prof.drain = last_wait.elapsed().as_secs_f64();
        Ok(batches)
    }

    /// Uploads `blocks` into slot `i` and submits the upload copy, the kernels and the readback
    /// into the slot's staging buffer (the fixed-stride copies, or the pack kernel, or with the
    /// transfer readback the copies on the transfer queue), plus the resolved timestamps. Uses
    /// the slot's persistent upload buffer rather than `queue.write_buffer`, which would allocate
    /// a fresh staging buffer per call that lives until the submission completes.
    fn submit(&mut self, i: usize, first: usize, blocks: &[&[u8]], prof: &mut Profile) -> anyhow::Result<()> {
        if self.direct {
            // The kernels read this slot's upload buffer; the bind groups recorded below hold it.
            self.bufs.data = self.slots[i].upload.clone();
        }
        let (ctx, layout, bufs, direct) = (self.ctx, self.layout, &self.bufs, self.direct);
        let slot = &mut self.slots[i];
        let n = blocks.len() as u32;
        let t0 = Instant::now();
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
            r.context("map upload buffer")?;
        }
        let t1 = Instant::now();
        let bytes = n as usize * BLOCK_SIZE;
        {
            let mut view = slot.upload.get_mapped_range_mut(..bytes as u64 + 4).context("upload mapped range")?;
            // Several threads write their share of the blocks: one thread's stores into the
            // (write-combined, ReBAR) upload buffer run at ~20 GB/s, which made this copy the
            // pipeline thread's largest cost once the readback left the GPU's critical path.
            let per = blocks.len().div_ceil(upload_threads()).max(64);
            let mut rest = view.slice(..);
            std::thread::scope(|s| {
                for chunk in blocks.chunks(per) {
                    let (mine, tail) = std::mem::take(&mut rest).split_at(chunk.len() * BLOCK_SIZE);
                    rest = tail;
                    let mine = SendWriteOnly(mine);
                    s.spawn(move || {
                        let mut mine = mine.into_inner();
                        for (k, b) in chunk.iter().enumerate() {
                            mine.slice(k * BLOCK_SIZE..(k + 1) * BLOCK_SIZE).copy_from_slice(b);
                        }
                    });
                }
                // Trailing zero word after the last block (`data` may hold stale blocks beyond it).
                rest.copy_from_slice(&[0u8; 4]);
            });
        }
        slot.upload.unmap();
        let t2 = Instant::now();

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
        let (submission, mapped, seq) = match (&mut self.xfer, &slot.staging) {
            (Some(x), Staging::Host(staging)) => {
                let seq = x.next_seq;
                x.next_seq += 1;
                // K1–K3, then K5/K4 in a second submission that waits for the previous batch's
                // readback (it overwrites `frames`) and signals `k_done`.
                self.kernels.record_front(ctx, &mut enc, bufs, n, queries);
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
                self.kernels.record_timed(ctx, &mut enc, bufs, n, queries);
                if let Some(pack) = &self.pack {
                    pack.record(ctx, &mut enc, bufs, n, staging);
                } else if layout.frames {
                    let (frames, frame_len) = (bufs.frames.as_ref().unwrap(), bufs.frame_len.as_ref().unwrap());
                    enc.copy_buffer_to_buffer(frame_len, 0, staging, 0, frame_len_bytes(n));
                    enc.copy_buffer_to_buffer(frames, 0, staging, layout.a, frames_bytes(n));
                } else {
                    enc.copy_buffer_to_buffer(&bufs.counts, 0, staging, 0, counts_bytes(n));
                    enc.copy_buffer_to_buffer(&bufs.seqs, 0, staging, layout.a, seqs_bytes(n));
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
        let (tx, upload_mapped) = mpsc::channel();
        slot.upload.map_async(wgpu::MapMode::Write, .., move |r| {
            let _ = tx.send(r);
        });
        slot.upload_mapped = Some(upload_mapped);
        slot.upload_submission = Some(submission.clone());
        slot.job = Some(Job { first, n, submission, mapped, seq });
        let t3 = Instant::now();
        prof.upload_wait += (t1 - t0).as_secs_f64();
        prof.upload_write += (t2 - t1).as_secs_f64();
        prof.submit += (t3 - t2).as_secs_f64();
        Ok(())
    }

    /// Hands completed slot `i` to `deliver`, adds its timestamps to `prof`, unmaps (wgpu staging)
    /// and frees the slot.
    fn finish(&mut self, i: usize, deliver: &mut Deliver<'_>, prof: &mut Profile) -> anyhow::Result<()> {
        let layout = self.layout;
        // End query of the last kernel recorded: K4 on the frame path (K5 runs before it), K3 on
        // the parse path.
        let names = self.kernels.names();
        let last = if layout.frames { "k4_entropy" } else { "k3_parse" };
        let last_kernel_end = 2 * names.iter().position(|&k| k == last).expect("last kernel is timed") + 1;
        #[cfg(test)]
        let fail_now = match &mut self.fail_deliveries_after {
            Some(0) => {
                self.fail_deliveries_after = None;
                true
            }
            Some(k) => {
                *k -= 1;
                false
            }
            None => false,
        };
        #[cfg(not(test))]
        let fail_now = false;
        let slot = &mut self.slots[i];
        let job = slot.job.take().unwrap();
        let t0 = Instant::now();
        let mut t1 = t0;
        let timed = slot.queries.is_some();
        let mut use_view = |view: &[u8]| -> anyhow::Result<()> {
            anyhow::ensure!(!fail_now, "injected delivery failure (test hook)");
            deliver(job.first, job.n, view)?;
            t1 = Instant::now();
            if timed {
                let nk = prof.ticks.len();
                // Only 4-byte aligned in general (seqs_bytes(1) is not a multiple of 8).
                let t = layout.ts as usize;
                let stamps: Vec<u64> = view[t..t + (nk + 1) * 16]
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                for (k, acc) in prof.ticks.iter_mut().enumerate() {
                    *acc += stamps[2 * k + 1].saturating_sub(stamps[2 * k]);
                }
                let (m0, m1) = (stamps[2 * nk], stamps[2 * nk + 1]);
                prof.upload_copy += stamps[0].saturating_sub(m0);
                prof.readback += m1.saturating_sub(stamps[last_kernel_end]);
                if let Some(end) = prof.last_end {
                    prof.idle += m0.saturating_sub(end);
                }
                prof.last_end = Some(m1);
            }
            Ok(())
        };
        let result = match &slot.staging {
            Staging::Wgpu(staging) => {
                let r = match staging.get_mapped_range(..) {
                    Ok(view) => use_view(&view[..]),
                    Err(e) => Err(anyhow!("mapped range: {e}")),
                };
                staging.unmap();
                r
            }
            // SAFETY: the transfer queue's copy into it completed (`stream` waited for t_done >=
            // job.seq) and the slot's next copy is only submitted after this delivery.
            Staging::Host(staging) => use_view(unsafe { staging.mapped() }),
        };
        let t2 = Instant::now();
        prof.deliver += (t1 - t0).as_secs_f64();
        prof.unmap += (t2 - t1).as_secs_f64();
        result
    }

    /// After an error: wait for the GPU and unmap every busy slot so the pipeline stays usable.
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
        for slot in &mut self.slots {
            if slot.job.take().is_some()
                && let Staging::Wgpu(staging) = &slot.staging
            {
                staging.unmap();
            }
        }
    }
}

/// A disjoint piece of a mapped upload range, handed to one writer thread. `WriteOnly<[u8]>` is
/// not `Send` only because its `Send` impl needs a sized `T`; like `&mut [u8]`, a byte slice of
/// it is safe to move to another thread.
struct SendWriteOnly<'a>(wgpu::WriteOnly<'a, [u8]>);

// SAFETY: see above; the pieces come from `split_at`, so no two threads write the same bytes.
unsafe impl Send for SendWriteOnly<'_> {}

impl<'a> SendWriteOnly<'a> {
    /// A method, so that a closure captures the whole wrapper rather than its non-`Send` field.
    fn into_inner(self) -> wgpu::WriteOnly<'a, [u8]> {
        self.0
    }
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
fn per_slot_bytes(batch: u32, frames: bool) -> u64 {
    data_bytes(batch) + StagingLayout::new(batch, frames).size
}

/// Device memory a `Pipeline` for `cfg` allocates: the shared buffers once (the scratch buffers,
/// their hash chain buffers sized by `cfg.params.matching`: one chain for single-hash presets, two
/// for Dfast; plus `data` and, on the frame path, `frames` and `frame_len`), per slot its upload
/// and staging buffers, plus K4's constant tables. K5 has no buffers of its own. Uploads go
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
        + cfg.inflight as u64 * per_slot_bytes(cfg.batch, frames)
        + if frames { k4_tables_bytes() } else { 0 }
}

/// Builds a parse-path `Pipeline` for `cfg` (whose `emit_frames` must be false) and streams
/// `blocks` through it (see `Pipeline::run`).
pub fn compress_stream(
    ctx: &GpuContext,
    cfg: &PipelineConfig,
    blocks: &[&[u8]],
    sink: &mut impl BlockSink,
) -> anyhow::Result<PipelineStats> {
    Pipeline::new(ctx, cfg)?.run(blocks, sink)
}

/// Builds a frame-path `Pipeline` for `cfg` (`emit_frames` is forced on; `huffman` is kept) and
/// streams `blocks` through it (see `Pipeline::run_frames`).
pub fn compress_stream_frames(
    ctx: &GpuContext,
    cfg: &PipelineConfig,
    blocks: &[&[u8]],
    sink: &mut impl FrameSink,
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
    use gzc_core::params::{LVL3, LVL9, LVL9S12SEG, MatchParams, RUNG1};
    use gzc_core::reference::compress_block;
    use gzc_core::synth::test_cases;

    struct Collect(Vec<Option<BlockOutput>>);

    impl BlockSink for Collect {
        fn put(&mut self, index: usize, out: BlockOutput) {
            assert!(self.0[index].is_none(), "index {index} delivered twice");
            self.0[index] = Some(out);
        }
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
                let l = StagingLayout::new(cap, frames);
                assert!([l.a, l.ts].iter().all(|x| x % STAGING_ALIGN == 0), "cap {cap} frames {frames}");
                let end = l.a + if frames { frames_bytes(cap) } else { seqs_bytes(cap) };
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
        assert_eq!(per_slot, data_bytes(100) + StagingLayout::new(100, true).size);
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
    /// once), per preset: single-hash presets allocate one chain's head/pred.
    #[test]
    fn vram_matches_params() {
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        for matching in [LVL3, RUNG1, LVL9] {
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
        if ctx.timestamps {
            let names: Vec<&str> = stats.kernel_ms.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(names, ["k1_chains", "k2_best", "k3_parse", "k4_entropy", "k5_huffman"]);
            assert!(stats.kernel_ms.iter().all(|&(_, ms)| ms > 0.0), "{:?}", stats.kernel_ms);
        }
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
}
