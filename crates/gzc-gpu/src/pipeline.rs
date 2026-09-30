//! In-flight streaming submission and GPU timestamp queries.
//!
//! `inflight` slots each own an input (`data`) buffer, the kernels' output buffers they read
//! back and a mappable staging buffer; the kernels' scratch buffers (hash chains, matches, parse)
//! are shared by all slots, since the queue runs one batch's kernels after another anyway (see
//! `vram_bytes`). A batch is uploaded into a free slot; the kernels plus copies of their outputs (and resolved timestamps)
//! into the slot's staging buffer are recorded in one submission, and the staging buffer is
//! mapped once. Completed slots are handed to the sink in submission order while later batches
//! keep the GPU busy.
//!
//! Two output paths, chosen by `GpuParams::emit_frames` when the pipeline is built:
//! - parses (`run`, `BlockSink`): K1→K2→K3, staging holds `counts` and the full fixed-stride
//!   `seqs` and `lits` regions; the host decodes a `BlockOutput` per block.
//! - frames (`run_frames`, `FrameSink`): K1→K2→K3(→K5)→K4, staging holds `frame_len` and the full
//!   fixed-stride `frames` region; the host only copies each frame's bytes out. K5 (Huffman
//!   literals, `GpuParams::huffman`) writes into the same `frames` / `frame_len` buffers, so it
//!   adds no memory.
use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{Context as _, anyhow};
use gzc_core::config::BLOCK_SIZE;
use gzc_core::seq::BlockOutput;

use crate::compressor::{
    BatchBuffers, FRAME_STRIDE, GpuParams, KERNEL_QUERIES, Kernels, MAX_SEQS, counts_bytes, decode_output,
    data_bytes, frame_bytes, frame_len_bytes, frames_bytes, k4_tables_bytes, lits_bytes, max_batch_blocks,
    scratch_bytes, seqs_bytes, slot_bytes,
};
use crate::context::GpuContext;

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
    /// `TRANSFER_NAMES`). The `gpu_*` entries come from timestamps (empty without them); the
    /// `host_*` ones are wall time on the pipeline thread.
    pub transfer_ms: Vec<(String, f64)>,
}

/// Names of `PipelineStats::transfer_ms`, in order:
/// - `gpu_upload_copy`: upload -> data copy (from the batch's start marker to K1's begin).
/// - `gpu_readback_copy`: output -> staging copies (from the last kernel's end to the end marker).
/// - `gpu_idle`: gaps between one batch's end marker and the next batch's start marker.
/// - `gpu_k2_k3_gap`: K3's begin minus K2's end (where the deferred readback copies sit).
/// - `host_upload_wait`: waiting for a slot's upload buffer to be mapped again.
/// - `host_upload_write`: writing the blocks into the mapped upload buffer, plus unmap.
/// - `host_submit`: recording and submitting the batch, plus the map requests.
/// - `host_wait`: blocked in `device.poll` for the oldest batch.
/// - `host_deliver`: handing a completed batch to the sink.
/// - `host_unmap`: unmapping the staging buffer.
/// - `host_fill`: from the start to the first submission.
/// - `host_drain`: from the last `device.poll` returning to the end.
pub const TRANSFER_NAMES: [&str; 12] = [
    "gpu_upload_copy",
    "gpu_readback_copy",
    "gpu_idle",
    "gpu_k2_k3_gap",
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
    readback_copy: u64,
    idle: u64,
    k2_k3_gap: u64,
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
/// `[counts][seqs][lits][timestamps]`; frame path: `[frame_len][frames][timestamps]`; the
/// regions keep the GPU buffers' fixed per-block stride.
#[derive(Clone, Copy)]
struct StagingLayout {
    frames: bool,
    /// Parse path: seqs region. Frame path: frames region.
    a: u64,
    /// Parse path: lits region. Frame path: unused (== ts).
    b: u64,
    ts: u64,
    size: u64,
}

impl StagingLayout {
    fn new(cap: u32, frames: bool) -> Self {
        // Every region starts STAGING_ALIGN-aligned: a GPU->staging copy to a destination that is
        // only 4-byte aligned runs several times slower (RTX 5090 / Vulkan: 6-10 ms more per
        // batch for the ~200 MB frames region), which showed up as a batch-size-dependent loss.
        let al = |x: u64| x.next_multiple_of(STAGING_ALIGN);
        let (a, b, ts) = if frames {
            let a = al(frame_len_bytes(cap));
            let ts = al(a + frames_bytes(cap));
            (a, ts, ts)
        } else {
            let a = al(counts_bytes(cap));
            let b = al(a + seqs_bytes(cap));
            (a, b, al(b + lits_bytes(cap)))
        };
        Self { frames, a, b, ts, size: ts + PIPELINE_QUERIES as u64 * wgpu::QUERY_SIZE as u64 }
    }
}

/// Hands one completed batch to the caller's sink: `(first block index, block count, mapped
/// staging bytes)`.
type Deliver<'d> = dyn FnMut(usize, u32, &[u8]) -> anyhow::Result<()> + 'd;

/// A submitted batch awaiting its staging map.
struct Job {
    first: usize,
    n: u32,
    /// The submission holding the batch's readback copies and the staging map's result; None
    /// while those copies are not recorded yet (see `Pipeline::submit`).
    readback: Option<(wgpu::SubmissionIndex, mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>)>,
}

struct Slot {
    bufs: BatchBuffers,
    /// Persistent upload buffer (MAP_WRITE | COPY_SRC, `data_bytes(batch)`): the host writes a
    /// batch into it while it is mapped, and the submission copies it into `bufs.data`. After
    /// each submission it is re-mapped; `upload_mapped` receives that map's result (None: mapped).
    upload: wgpu::Buffer,
    upload_mapped: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    staging: wgpu::Buffer,
    /// Query set and its resolve buffer, when timestamps are enabled.
    queries: Option<(wgpu::QuerySet, wgpu::Buffer)>,
    job: Option<Job>,
}

/// Compiled kernels plus `inflight` slots of `batch` blocks, reusable across runs.
pub struct Pipeline<'a> {
    ctx: &'a GpuContext,
    cfg: PipelineConfig,
    kernels: Kernels,
    layout: StagingLayout,
    slots: Vec<Slot>,
    /// Record each batch's readback copies into the *next* submission, between its K2 and K3,
    /// so they overlap that batch's K3 instead of running after this batch's K4 (see `submit`).
    /// `GZC_READBACK_INLINE` (anything but 0) turns it off.
    defer_readback: bool,
    /// The slot whose batch is submitted but whose readback is not recorded yet.
    pending: Option<usize>,
}

impl<'a> Pipeline<'a> {
    /// Compiles the kernels and allocates every slot, for the frame path when
    /// `cfg.params.emit_frames` and the parse path otherwise. Errors on `batch`/`inflight` of 0,
    /// a batch above `max_batch_blocks`, match params the GPU does not support (see
    /// `Kernels::new`), or a wgpu out-of-memory/validation error.
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
        let mut bufs = vec![BatchBuffers::new(ctx, cfg.batch, frames, &m)];
        for _ in 1..cfg.inflight {
            bufs.push(BatchBuffers::new_sharing(ctx, &bufs[0], frames));
        }
        let slots = bufs
            .into_iter()
            .map(|bufs| Slot {
                bufs,
                upload: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pipeline.upload"),
                    size: data_bytes(cfg.batch),
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                }),
                upload_mapped: None,
                staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pipeline.staging"),
                    size: layout.size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                queries: ctx.timestamps.then(|| {
                    let set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("pipeline.timestamps"),
                        ty: wgpu::QueryType::Timestamp,
                        count: PIPELINE_QUERIES,
                    });
                    let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("pipeline.resolve"),
                        size: PIPELINE_QUERIES as u64 * wgpu::QUERY_SIZE as u64,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });
                    (set, resolve)
                }),
                job: None,
            })
            .collect();
        scopes.pop()?;
        let defer_readback = !std::env::var("GZC_READBACK_INLINE").is_ok_and(|v| v != "0");
        Ok(Self { ctx, cfg: *cfg, kernels, layout, slots, defer_readback, pending: None })
    }

    /// Bytes of every buffer the pipeline created, the shared scratch buffers once; timestamp
    /// query sets and their resolve buffers are left out, as in `vram_bytes`.
    #[cfg(test)]
    fn allocated_bytes(&self) -> u64 {
        let opt = |b: &Option<wgpu::Buffer>| b.as_ref().map_or(0, |b| b.size());
        let s = &self.slots[0].bufs;
        let scratch = [&s.head, &s.pred, &s.best, &s.seqs, &s.lits, &s.counts].iter().map(|b| b.size()).sum::<u64>();
        let per_slot: u64 = self
            .slots
            .iter()
            .map(|slot| {
                let b = &slot.bufs;
                b.data.size() + opt(&b.frames) + opt(&b.frame_len) + slot.upload.size() + slot.staging.size()
            })
            .sum();
        scratch + per_slot + self.kernels.own_buffer_bytes()
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
            let lit_stride = BLOCK_SIZE / 4;
            for b in 0..n as usize {
                let (n_seq, n_lit) = (words[2 * b], words[2 * b + 1]);
                anyhow::ensure!(
                    n_seq <= MAX_SEQS && n_lit as usize <= BLOCK_SIZE,
                    "block {}: bad counts ({n_seq}, {n_lit})",
                    first + b
                );
                let s = (layout.a / 4) as usize + b * seq_stride;
                let l = (layout.b / 4) as usize + b * lit_stride;
                sink.put(first + b, decode_output(&words[s..s + seq_stride], &words[l..l + lit_stride], n_seq, n_lit));
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
        self.run_with(blocks, &mut |first, n, view| {
            let lens: &[u32] = bytemuck::cast_slice(&view[..frame_len_bytes(n) as usize]);
            let frames = &view[layout.a as usize..layout.a as usize + n as usize * FRAME_STRIDE];
            for (b, &len) in lens.iter().enumerate() {
                sink.put(first + b, frame_bytes(frames, b, len).with_context(|| format!("block {}", first + b))?);
            }
            Ok(())
        })
    }

    fn run_with(
        &mut self,
        blocks: &[&[u8]],
        deliver: &mut Deliver<'_>,
    ) -> anyhow::Result<PipelineStats> {
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
                vec![ms(prof.upload_copy), ms(prof.readback_copy), ms(prof.idle), ms(prof.k2_k3_gap)],
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
        let transfer_ms = TRANSFER_NAMES[..4]
            .iter()
            .zip(gpu_ms)
            .chain(TRANSFER_NAMES[4..].iter().zip(host_ms))
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
            // No later batch will carry the pending readback (nothing left to submit, or every
            // slot is busy and the oldest is the pending one): submit it on its own.
            if self.pending.is_some()
                && (next == blocks.len() || self.slots[oldest].job.as_ref().unwrap().readback.is_none())
            {
                self.flush()?;
            }
            // Every slot is busy (or nothing is left to submit): block on the oldest batch...
            let submission = self.slots[oldest].job.as_ref().unwrap().readback.as_ref().unwrap().0.clone();
            let t = Instant::now();
            self.ctx
                .device
                .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: None })
                .context("device poll")?;
            last_wait = Instant::now();
            prof.wait += (last_wait - t).as_secs_f64();
            // ...then drain it and every later batch that has also completed.
            let mut waited_for = true;
            while let Some(&i) = busy.front() {
                let Some((_, mapped)) = &self.slots[i].job.as_ref().unwrap().readback else { break };
                match mapped.try_recv() {
                    Ok(r) => r.context("map staging buffer")?,
                    Err(mpsc::TryRecvError::Empty) if !waited_for => break,
                    Err(_) => anyhow::bail!("staging map callback not delivered after waiting for its submission"),
                }
                self.finish(i, deliver, prof)?;
                busy.pop_front();
                waited_for = false;
            }
        }
        prof.drain = last_wait.elapsed().as_secs_f64();
        Ok(batches)
    }

    /// Records the copies of slot `slot`'s batch of `n` blocks (outputs and resolved timestamps)
    /// into its staging buffer.
    fn record_readback(slot: &Slot, layout: StagingLayout, n_queries: u32, n: u32, enc: &mut wgpu::CommandEncoder) {
        if layout.frames {
            let (frames, frame_len) = (slot.bufs.frames.as_ref().unwrap(), slot.bufs.frame_len.as_ref().unwrap());
            enc.copy_buffer_to_buffer(frame_len, 0, &slot.staging, 0, frame_len_bytes(n));
            enc.copy_buffer_to_buffer(frames, 0, &slot.staging, layout.a, frames_bytes(n));
        } else {
            enc.copy_buffer_to_buffer(&slot.bufs.counts, 0, &slot.staging, 0, counts_bytes(n));
            enc.copy_buffer_to_buffer(&slot.bufs.seqs, 0, &slot.staging, layout.a, seqs_bytes(n));
            enc.copy_buffer_to_buffer(&slot.bufs.lits, 0, &slot.staging, layout.b, lits_bytes(n));
        }
        if let Some((set, resolve)) = &slot.queries {
            let q = n_queries + 2;
            enc.resolve_query_set(set, 0..q, resolve, 0);
            enc.copy_buffer_to_buffer(resolve, 0, &slot.staging, layout.ts, q as u64 * wgpu::QUERY_SIZE as u64);
        }
    }

    /// Maps slot `i`'s staging buffer once `submission` (which holds its readback copies) is done.
    fn map_readback(&mut self, i: usize, submission: wgpu::SubmissionIndex) {
        let slot = &mut self.slots[i];
        let (tx, mapped) = mpsc::channel();
        slot.staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        slot.job.as_mut().unwrap().readback = Some((submission, mapped));
    }

    /// Submits the pending batch's readback copies on their own.
    fn flush(&mut self) -> anyhow::Result<()> {
        let Some(p) = self.pending.take() else { return Ok(()) };
        let n_queries = 2 * self.kernels.names().len() as u32;
        let n = self.slots[p].job.as_ref().unwrap().n;
        let mut enc =
            self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pipeline.readback") });
        Self::record_readback(&self.slots[p], self.layout, n_queries, n, &mut enc);
        let submission = self.ctx.queue.submit([enc.finish()]);
        self.map_readback(p, submission);
        Ok(())
    }

    /// Uploads `blocks` into slot `i` and submits the kernels, plus the staging copies: this
    /// batch's own (after K4) without `defer_readback`; with it, the pending previous batch's,
    /// recorded between this batch's K2 and K3 so the GPU runs them alongside K3 (K3's barrier
    /// only waits for compute work, and the copies read that batch's own per-slot buffers).
    /// Uses the slot's persistent upload buffer rather than `queue.write_buffer`, which would
    /// allocate a fresh staging buffer per call that lives until the submission completes.
    fn submit(&mut self, i: usize, first: usize, blocks: &[&[u8]], prof: &mut Profile) -> anyhow::Result<()> {
        let (ctx, layout) = (self.ctx, self.layout);
        let n = blocks.len() as u32;
        let t0 = Instant::now();
        {
            let slot = &mut self.slots[i];
            if let Some(rx) = slot.upload_mapped.take() {
                // Its submission has completed (the slot is free), so the callback is normally in.
                let r = match rx.try_recv() {
                    Ok(r) => r,
                    Err(_) => {
                        ctx.device.poll(wgpu::PollType::wait_indefinitely()).context("device poll")?;
                        rx.recv().context("upload map callback dropped")?
                    }
                };
                r.context("map upload buffer")?;
            }
        }
        let t1 = Instant::now();
        let bytes = n as usize * BLOCK_SIZE;
        {
            let slot = &mut self.slots[i];
            let mut view = slot.upload.get_mapped_range_mut(..bytes as u64 + 4).context("upload mapped range")?;
            for (k, b) in blocks.iter().enumerate() {
                view.slice(k * BLOCK_SIZE..(k + 1) * BLOCK_SIZE).copy_from_slice(b);
            }
            // Trailing zero word after the last block (`data` may hold stale blocks beyond it).
            view.slice(bytes..bytes + 4).copy_from_slice(&[0u8; 4]);
            drop(view);
            slot.upload.unmap();
        }
        let t2 = Instant::now();

        let n_queries = 2 * self.kernels.names().len() as u32;
        let pending = if self.defer_readback { self.pending.take() } else { None };
        let submission = {
            let slot = &self.slots[i];
            // Start/end markers: empty passes whose timestamps (written at BOTTOM_OF_PIPE on
            // Vulkan, i.e. once all earlier commands completed) bracket the batch.
            let marker = |enc: &mut wgpu::CommandEncoder, index: u32| {
                if let Some((set, _)) = &slot.queries {
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
            let mut enc =
                ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pipeline") });
            marker(&mut enc, n_queries);
            enc.copy_buffer_to_buffer(&slot.upload, 0, &slot.bufs.data, 0, bytes as u64 + 4);
            // record_timed binds exactly counts_bytes(n) (K3) and frame_len_bytes(n) (K5, K4), so
            // no kernel processes the stale blocks of a partial batch.
            let slots = &self.slots;
            self.kernels.record_timed_with(ctx, &mut enc, &slot.bufs, n, slot.queries.as_ref().map(|q| &q.0), &mut |enc| {
                if let Some(p) = pending {
                    let pn = slots[p].job.as_ref().unwrap().n;
                    Self::record_readback(&slots[p], layout, n_queries, pn, enc);
                }
            });
            marker(&mut enc, n_queries + 1);
            if !self.defer_readback {
                Self::record_readback(slot, layout, n_queries, n, &mut enc);
            }
            ctx.queue.submit([enc.finish()])
        };

        let slot = &mut self.slots[i];
        let (tx, upload_mapped) = mpsc::channel();
        slot.upload.map_async(wgpu::MapMode::Write, .., move |r| {
            let _ = tx.send(r);
        });
        slot.upload_mapped = Some(upload_mapped);
        slot.job = Some(Job { first, n, readback: None });
        if self.defer_readback {
            if let Some(p) = pending {
                self.map_readback(p, submission);
            }
            self.pending = Some(i);
        } else {
            self.map_readback(i, submission);
        }
        let t3 = Instant::now();
        prof.upload_wait += (t1 - t0).as_secs_f64();
        prof.upload_write += (t2 - t1).as_secs_f64();
        prof.submit += (t3 - t2).as_secs_f64();
        Ok(())
    }

    /// Hands mapped slot `i` to `deliver`, adds its timestamps to `prof`, unmaps and frees the slot.
    fn finish(&mut self, i: usize, deliver: &mut Deliver<'_>, prof: &mut Profile) -> anyhow::Result<()> {
        let layout = self.layout;
        // Query of the last kernel's end: K4 (6/7) on the frame path, K3 (4/5) on the parse path.
        let last_kernel_end = if layout.frames { 7 } else { 5 };
        let slot = &mut self.slots[i];
        let job = slot.job.take().unwrap();
        let t0 = Instant::now();
        let mut t1 = t0;
        let result = (|| -> anyhow::Result<()> {
            let view = slot.staging.get_mapped_range(..).context("mapped range")?;
            deliver(job.first, job.n, &view[..])?;
            t1 = Instant::now();
            if slot.queries.is_some() {
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
                prof.readback_copy += m1.saturating_sub(stamps[last_kernel_end]);
                // K3's begin minus K2's end: holds the previous batch's readback with
                // `defer_readback` if timestamps serialize it.
                prof.k2_k3_gap += stamps[4].saturating_sub(stamps[3]);
                if let Some(end) = prof.last_end {
                    prof.idle += m0.saturating_sub(end);
                }
                prof.last_end = Some(m1);
            }
            Ok(())
        })();
        slot.staging.unmap();
        let t2 = Instant::now();
        prof.deliver += (t1 - t0).as_secs_f64();
        prof.unmap += (t2 - t1).as_secs_f64();
        result
    }

    /// After an error: wait for the GPU and unmap every busy slot so the pipeline stays usable.
    fn abandon(&mut self) {
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
        self.pending = None;
        for slot in &mut self.slots {
            if slot.job.take().is_some_and(|job| job.readback.is_some()) {
                slot.staging.unmap();
            }
        }
    }
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

/// Device memory a `Pipeline` for `cfg` allocates: the shared scratch buffers once (their hash
/// chain buffers sized by `cfg.params.matching`: one chain for single-hash presets, two for
/// Dfast), and per slot its data/output buffers plus its upload and readback staging buffers
/// (mappable; counted although drivers may place them in host memory), plus K4's constant
/// tables. K5 has no buffers of its own. Uploads go through the persistent upload buffers only,
/// so no transient staging adds to this. Timestamp query sets are not counted.
pub fn vram_bytes(cfg: &PipelineConfig) -> u64 {
    let frames = cfg.params.emit_frames;
    let per_slot =
        slot_bytes(cfg.batch, frames) + data_bytes(cfg.batch) + StagingLayout::new(cfg.batch, frames).size;
    scratch_bytes(cfg.batch, &cfg.params.matching)
        + cfg.inflight as u64 * per_slot
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
    use gzc_core::block::chunk_file;
    use crate::chains::{head_bytes, pred_bytes};
    use gzc_core::params::{LVL3, LVL9, MatchParams, RUNG1};
    use gzc_core::reference::compress_block;
    use gzc_core::frame::write_frame;
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
                assert!([l.a, l.b, l.ts].iter().all(|x| x % STAGING_ALIGN == 0), "cap {cap} frames {frames}");
                let end = if frames { l.a + frames_bytes(cap) } else { l.b + lits_bytes(cap) };
                assert!(l.a >= if frames { frame_len_bytes(cap) } else { counts_bytes(cap) } && end <= l.ts);
                assert!(frames || l.a + seqs_bytes(cap) <= l.b);
            }
        }
    }

    #[test]
    fn vram_counts_scratch_once_and_slots_per_inflight() {
        let frames = |batch, inflight| {
            PipelineConfig { params: GpuParams { matching: LVL3, emit_frames: true, huffman: true }, ..cfg(batch, inflight) }
        };
        let one = vram_bytes(&frames(100, 1));
        let per_slot = vram_bytes(&frames(100, 2)) - one;
        assert_eq!(vram_bytes(&frames(100, 4)), one + 3 * per_slot);
        assert_eq!(per_slot, slot_bytes(100, true) + data_bytes(100) + StagingLayout::new(100, true).size);
        assert_eq!(one, scratch_bytes(100, &LVL3) + per_slot + k4_tables_bytes());
        assert!(one > scratch_bytes(100, &LVL3) + per_slot, "scratch counted once, plus the K4 tables");
        // The parse path reads back the fixed-stride seqs and lits instead of the frames.
        assert!(vram_bytes(&cfg(100, 2)) > vram_bytes(&frames(100, 2)));
        // K5 (Huffman literals) needs no buffers of its own.
        let raw_lits = PipelineConfig { params: GpuParams { huffman: false, ..frames(100, 2).params }, ..frames(100, 2) };
        assert_eq!(vram_bytes(&raw_lits), vram_bytes(&frames(100, 2)));
        #[cfg(feature = "block-128k")]
        {
            // ~2.5 MiB of scratch per block, ~0.5 MiB per block per slot on the frame path.
            let mib = |b: u64| b as f64 / (1u64 << 20) as f64 / 100.0;
            assert!((2.4..2.6).contains(&mib(scratch_bytes(100, &LVL3))), "{}", mib(scratch_bytes(100, &LVL3)));
            assert!((0.49..0.51).contains(&mib(per_slot)), "{}", mib(per_slot));
        }
    }

    /// `vram_bytes` equals the bytes of every buffer a `Pipeline` actually creates (shared scratch
    /// once), per preset: single-hash presets allocate one chain's head/pred.
    #[test]
    fn vram_matches_params() {
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        for matching in [LVL3, RUNG1, LVL9] {
            for (emit_frames, batch, inflight) in [(true, 7, 1), (true, 16, 3), (false, 5, 2)] {
                let cfg = PipelineConfig { batch, inflight, params: GpuParams { matching, emit_frames, huffman: true } };
                let pipe = Pipeline::new(&ctx, &cfg).unwrap();
                assert_eq!(pipe.allocated_bytes(), vram_bytes(&cfg), "{matching:?} {emit_frames} b{batch} i{inflight}");
            }
        }
        let scratch = |m: MatchParams| {
            vram_bytes(&PipelineConfig { batch: 10, inflight: 1, params: GpuParams { matching: m, emit_frames: true, huffman: true } })
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

    #[test]
    fn stream_frames_odd_batch_single_slot_and_reuse() {
        // Raw literals (no K5): the huffman: false frame path stays covered end to end.
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        let distinct = distinct_blocks();
        let blocks: Vec<&[u8]> = distinct.iter().map(|b| b.as_slice()).collect();
        let params = GpuParams { matching: LVL3, emit_frames: true, huffman: false };
        let frames_cfg = PipelineConfig { params, ..cfg(7, 1) };
        let mut pipe = Pipeline::new(&ctx, &frames_cfg).unwrap();
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
