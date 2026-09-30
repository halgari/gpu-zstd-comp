//! In-flight streaming submission and GPU timestamp queries.
//!
//! `inflight` slots each own a `BatchBuffers` and a mappable staging buffer. A batch is
//! uploaded into a free slot; K1→K2→K3 plus copies of `counts`, the full fixed-stride `seqs`
//! and `lits` regions (and resolved timestamps) into the slot's staging buffer are recorded in
//! one submission, and the staging buffer is mapped once. Completed slots are decoded into the
//! sink in submission order while later batches keep the GPU busy.
use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{Context as _, anyhow};
use gzc_core::config::BLOCK_SIZE;
use gzc_core::seq::BlockOutput;

use crate::compressor::{
    BatchBuffers, GpuParams, KERNEL_NAMES, KERNEL_QUERIES, Kernels, MAX_SEQS, counts_bytes, decode_output,
    lits_bytes, max_batch_blocks, seqs_bytes,
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
    /// GPU time summed over all batches per kernel ("k1_chains", "k2_best", "k3_parse"), in
    /// milliseconds; empty when the device has no timestamp queries.
    pub kernel_ms: Vec<(String, f64)>,
    /// Wall time from the first upload to the last block handed to the sink.
    pub wall_s: f64,
    /// Number of batches submitted.
    pub batches: u32,
}

/// Receives each block's parse; called exactly once per index, in arbitrary order.
pub trait BlockSink {
    fn put(&mut self, index: usize, out: BlockOutput);
}

/// Byte offsets of one slot's staging buffer, laid out for `cap` blocks:
/// `[counts][seqs][lits][timestamps]`, seqs and lits at the GPU buffers' fixed stride.
#[derive(Clone, Copy)]
struct StagingLayout {
    seqs: u64,
    lits: u64,
    ts: u64,
    size: u64,
}

impl StagingLayout {
    fn new(cap: u32) -> Self {
        let seqs = counts_bytes(cap);
        let lits = seqs + seqs_bytes(cap);
        let ts = lits + lits_bytes(cap);
        Self { seqs, lits, ts, size: ts + KERNEL_QUERIES as u64 * wgpu::QUERY_SIZE as u64 }
    }
}

/// A submitted batch awaiting its staging map.
struct Job {
    first: usize,
    n: u32,
    submission: wgpu::SubmissionIndex,
    mapped: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

struct Slot {
    bufs: BatchBuffers,
    staging: wgpu::Buffer,
    /// Query set and its resolve buffer, when timestamps are enabled.
    queries: Option<(wgpu::QuerySet, wgpu::Buffer)>,
    job: Option<Job>,
}

/// Compiled kernels plus `inflight` slots of `batch` blocks, reusable across `run` calls.
pub struct Pipeline<'a> {
    ctx: &'a GpuContext,
    cfg: PipelineConfig,
    kernels: Kernels,
    layout: StagingLayout,
    slots: Vec<Slot>,
}

impl<'a> Pipeline<'a> {
    /// Compiles the kernels and allocates every slot. Errors on `batch`/`inflight` of 0, a
    /// batch above `max_batch_blocks`, or a wgpu out-of-memory/validation error.
    pub fn new(ctx: &'a GpuContext, cfg: &PipelineConfig) -> anyhow::Result<Self> {
        let max = max_batch_blocks(&ctx.device.limits());
        anyhow::ensure!(cfg.inflight >= 1, "inflight must be at least 1");
        anyhow::ensure!(cfg.batch >= 1 && cfg.batch <= max, "batch {} not in 1..={max} for this device", cfg.batch);

        let scopes = ErrorScopes::push(ctx);
        let kernels = Kernels::new(ctx, cfg.params);
        let layout = StagingLayout::new(cfg.batch);
        let slots = (0..cfg.inflight)
            .map(|_| Slot {
                bufs: BatchBuffers::new(ctx, cfg.batch),
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
                        count: KERNEL_QUERIES,
                    });
                    let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("pipeline.resolve"),
                        size: KERNEL_QUERIES as u64 * wgpu::QUERY_SIZE as u64,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });
                    (set, resolve)
                }),
                job: None,
            })
            .collect();
        scopes.pop()?;
        Ok(Self { ctx, cfg: *cfg, kernels, layout, slots })
    }

    /// Streams `blocks` (each BLOCK_SIZE bytes) through the slots, handing every block's parse
    /// to `sink` exactly once. wgpu validation/out-of-memory errors become `Err`.
    pub fn run(&mut self, blocks: &[&[u8]], sink: &mut impl BlockSink) -> anyhow::Result<PipelineStats> {
        if let Some(i) = blocks.iter().position(|b| b.len() != BLOCK_SIZE) {
            anyhow::bail!("block {i} is {} bytes, expected BLOCK_SIZE {BLOCK_SIZE}", blocks[i].len());
        }
        let scopes = ErrorScopes::push(self.ctx);
        let start = Instant::now();
        let mut ticks = [0u64; KERNEL_NAMES.len()];
        let result = self.stream(blocks, sink, &mut ticks);
        let wall_s = start.elapsed().as_secs_f64();
        if result.is_err() {
            self.abandon();
        }
        scopes.pop()?;
        let batches = result?;

        let kernel_ms = if self.ctx.timestamps {
            let period_ns = self.ctx.queue.get_timestamp_period() as f64;
            KERNEL_NAMES.iter().zip(ticks).map(|(name, t)| (name.to_string(), t as f64 * period_ns / 1e6)).collect()
        } else {
            Vec::new()
        };
        Ok(PipelineStats { kernel_ms, wall_s, batches })
    }

    /// The submit / wait / decode loop; returns the number of batches submitted.
    fn stream(&mut self, blocks: &[&[u8]], sink: &mut impl BlockSink, ticks: &mut [u64]) -> anyhow::Result<u32> {
        let batch = self.cfg.batch as usize;
        let mut next = 0usize;
        let mut batches = 0u32;
        // Busy slots, oldest submission first.
        let mut busy: VecDeque<usize> = VecDeque::new();
        loop {
            while next < blocks.len() {
                let Some(i) = self.slots.iter().position(|s| s.job.is_none()) else { break };
                let n = batch.min(blocks.len() - next);
                self.submit(i, next, &blocks[next..next + n]);
                busy.push_back(i);
                next += n;
                batches += 1;
            }
            let Some(&oldest) = busy.front() else { break };
            // Every slot is busy (or nothing is left to submit): block on the oldest batch...
            let submission = self.slots[oldest].job.as_ref().unwrap().submission.clone();
            self.ctx
                .device
                .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: None })
                .context("device poll")?;
            // ...then drain it and every later batch that has also completed.
            let mut waited_for = true;
            while let Some(&i) = busy.front() {
                match self.slots[i].job.as_ref().unwrap().mapped.try_recv() {
                    Ok(r) => r.context("map staging buffer")?,
                    Err(mpsc::TryRecvError::Empty) if !waited_for => break,
                    Err(_) => anyhow::bail!("staging map callback not delivered after waiting for its submission"),
                }
                self.finish(i, sink, ticks)?;
                busy.pop_front();
                waited_for = false;
            }
        }
        Ok(batches)
    }

    /// Uploads `blocks` into slot `i` and submits K1→K2→K3 plus the staging copies.
    fn submit(&mut self, i: usize, first: usize, blocks: &[&[u8]]) {
        let (ctx, layout) = (self.ctx, self.layout);
        let slot = &mut self.slots[i];
        let n = blocks.len() as u32;
        for (k, b) in blocks.iter().enumerate() {
            ctx.queue.write_buffer(&slot.bufs.data, (k * BLOCK_SIZE) as u64, b);
        }
        // Trailing zero word after the last block (the slot may hold stale blocks beyond it).
        ctx.queue.write_buffer(&slot.bufs.data, n as u64 * BLOCK_SIZE as u64, &[0u8; 4]);

        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pipeline") });
        // record_timed binds exactly counts_bytes(n), so K3 never parses stale blocks.
        self.kernels.record_timed(ctx, &mut enc, &slot.bufs, n, slot.queries.as_ref().map(|q| &q.0));
        enc.copy_buffer_to_buffer(&slot.bufs.counts, 0, &slot.staging, 0, counts_bytes(n));
        enc.copy_buffer_to_buffer(&slot.bufs.seqs, 0, &slot.staging, layout.seqs, seqs_bytes(n));
        enc.copy_buffer_to_buffer(&slot.bufs.lits, 0, &slot.staging, layout.lits, lits_bytes(n));
        if let Some((set, resolve)) = &slot.queries {
            enc.resolve_query_set(set, 0..KERNEL_QUERIES, resolve, 0);
            enc.copy_buffer_to_buffer(resolve, 0, &slot.staging, layout.ts, resolve.size());
        }
        let submission = ctx.queue.submit([enc.finish()]);

        let (tx, mapped) = mpsc::channel();
        slot.staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        slot.job = Some(Job { first, n, submission, mapped });
    }

    /// Decodes mapped slot `i` into `sink`, adds its kernel ticks, unmaps and frees the slot.
    fn finish(&mut self, i: usize, sink: &mut impl BlockSink, ticks: &mut [u64]) -> anyhow::Result<()> {
        let layout = self.layout;
        let slot = &mut self.slots[i];
        let job = slot.job.take().unwrap();
        let result = (|| -> anyhow::Result<()> {
            let view = slot.staging.get_mapped_range(..).context("mapped range")?;
            let words: &[u32] = bytemuck::cast_slice(&view[..]);
            let seq_stride = 3 * MAX_SEQS as usize;
            let lit_stride = BLOCK_SIZE / 4;
            for b in 0..job.n as usize {
                let (n_seq, n_lit) = (words[2 * b], words[2 * b + 1]);
                anyhow::ensure!(
                    n_seq <= MAX_SEQS && n_lit as usize <= BLOCK_SIZE,
                    "block {}: bad counts ({n_seq}, {n_lit})",
                    job.first + b
                );
                let s = (layout.seqs / 4) as usize + b * seq_stride;
                let l = (layout.lits / 4) as usize + b * lit_stride;
                let out = decode_output(&words[s..s + seq_stride], &words[l..l + lit_stride], n_seq, n_lit);
                sink.put(job.first + b, out);
            }
            if slot.queries.is_some() {
                // Only 4-byte aligned in general (seqs_bytes(1) is not a multiple of 8).
                let t = layout.ts as usize;
                let stamps: Vec<u64> = view[t..t + KERNEL_QUERIES as usize * 8]
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                for (k, acc) in ticks.iter_mut().enumerate() {
                    *acc += stamps[2 * k + 1].saturating_sub(stamps[2 * k]);
                }
            }
            Ok(())
        })();
        slot.staging.unmap();
        result
    }

    /// After an error: wait for the GPU and unmap every busy slot so the pipeline stays usable.
    fn abandon(&mut self) {
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
        for slot in &mut self.slots {
            if slot.job.take().is_some() {
                slot.staging.unmap();
            }
        }
    }
}

/// Out-of-memory and validation error scopes, popped together.
struct ErrorScopes {
    oom: wgpu::ErrorScopeGuard,
    validation: wgpu::ErrorScopeGuard,
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

/// Builds a `Pipeline` for `cfg` and streams `blocks` through it (see `Pipeline::run`).
pub fn compress_stream(
    ctx: &GpuContext,
    cfg: &PipelineConfig,
    blocks: &[&[u8]],
    sink: &mut impl BlockSink,
) -> anyhow::Result<PipelineStats> {
    Pipeline::new(ctx, cfg)?.run(blocks, sink)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gzc_core::block::chunk_file;
    use gzc_core::reference::{LVL3, compress_block};
    use gzc_core::synth::test_cases;

    struct Collect(Vec<Option<BlockOutput>>);

    impl BlockSink for Collect {
        fn put(&mut self, index: usize, out: BlockOutput) {
            assert!(self.0[index].is_none(), "index {index} delivered twice");
            self.0[index] = Some(out);
        }
    }

    fn cfg(batch: u32, inflight: u32) -> PipelineConfig {
        PipelineConfig { batch, inflight, params: GpuParams { depth: 1 } }
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
    }
}
