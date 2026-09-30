//! GPU compressor benchmark runs.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, mpsc};
use std::time::Instant;

use rayon::prelude::*;

use gzc_core::config::BLOCK_SIZE;
use gzc_core::frame::{FrameOptions, write_frame};
use gzc_core::seq::BlockOutput;
use gzc_gpu::context::GpuContext;
use gzc_gpu::pipeline::{BlockSink, Pipeline, PipelineConfig};

use crate::corpus::Corpus;
use crate::result::{RunResult, per_kind};

/// Forwards every block's parse to the frame-writer workers.
struct ChannelSink(mpsc::Sender<(usize, BlockOutput)>);

impl BlockSink for ChannelSink {
    fn put(&mut self, index: usize, out: BlockOutput) {
        // The receiver lives until every worker exits, which is after this sink is dropped.
        self.0.send((index, out)).expect("frame writers alive");
    }
}

struct Discard;

impl BlockSink for Discard {
    fn put(&mut self, _index: usize, _out: BlockOutput) {}
}

/// Compresses every block of `corpus` with the streaming GPU pipeline; `writer_threads` rayon
/// workers turn the parses into zstd frames (`write_frame`). One untimed warmup batch runs
/// first on the same pipeline; the timed region spans from the first upload to the last frame
/// written. With `verify`, every frame is then decompressed with libzstd and compared with its
/// padded block. `kernel_ms` holds the timed run's summed per-kernel GPU time (when the device
/// supports timestamps); a per-batch breakdown is printed to stderr.
pub fn run_gpu(corpus: &Corpus, cfg: &PipelineConfig, writer_threads: usize, verify: bool) -> anyhow::Result<RunResult> {
    anyhow::ensure!(writer_threads >= 1, "writer_threads must be at least 1");
    let ctx = GpuContext::new()?;
    let mut pipe = Pipeline::new(&ctx, cfg)?;
    let blocks: Vec<&[u8]> = corpus.blocks.iter().map(|b| b.data.as_slice()).collect();
    let opts = FrameOptions::default();

    pipe.run(&blocks[..blocks.len().min(cfg.batch as usize)], &mut Discard)?;

    // 0 = not written yet (a frame is never empty).
    let sizes: Vec<AtomicU64> = blocks.iter().map(|_| AtomicU64::new(0)).collect();
    let frames: Vec<OnceLock<Vec<u8>>> = if verify { blocks.iter().map(|_| OnceLock::new()).collect() } else { Vec::new() };
    let duplicates = AtomicU64::new(0);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(writer_threads).build()?;
    let (tx, rx) = mpsc::channel::<(usize, BlockOutput)>();
    let rx = Mutex::new(rx);

    let start = Instant::now();
    let stats = pool.in_place_scope(|s| {
        for _ in 0..writer_threads {
            s.spawn(|_| {
                loop {
                    let msg = rx.lock().unwrap().recv();
                    let Ok((i, out)) = msg else { break };
                    let frame = write_frame(blocks[i], &out, opts);
                    if sizes[i].swap(frame.len() as u64, Ordering::Relaxed) != 0 {
                        duplicates.fetch_add(1, Ordering::Relaxed);
                    }
                    if verify {
                        let _ = frames[i].set(frame);
                    }
                }
            });
        }
        let mut sink = ChannelSink(tx);
        let stats = pipe.run(&blocks, &mut sink);
        drop(sink); // closes the channel: workers drain it and exit
        stats
    })?;
    let seconds = start.elapsed().as_secs_f64();

    let sizes: Vec<u64> = sizes.into_iter().map(AtomicU64::into_inner).collect();
    anyhow::ensure!(duplicates.into_inner() == 0, "a block was delivered more than once");
    if let Some(i) = sizes.iter().position(|&s| s == 0) {
        anyhow::bail!("block {i} was never delivered");
    }
    if verify {
        pool.install(|| {
            frames.par_iter().zip(&corpus.blocks).enumerate().try_for_each(|(i, (frame, block))| {
                let dec = zstd::bulk::decompress(frame.get().expect("frame kept"), BLOCK_SIZE)?;
                anyhow::ensure!(dec == block.data, "block {i} did not round-trip through the GPU frame");
                Ok(())
            })
        })?;
    }

    let real_bytes = corpus.real_bytes();
    let mb = real_bytes as f64 / 1e6;
    eprintln!(
        "  gpu b{} i{} w{writer_threads}: {} batches, pipeline {:.3}s ({:.1} MB/s), end-to-end {seconds:.3}s ({:.1} MB/s)",
        cfg.batch,
        cfg.inflight,
        stats.batches,
        stats.wall_s,
        mb / stats.wall_s,
        mb / seconds
    );
    let batches = stats.batches.max(1) as f64;
    for (name, ms) in &stats.kernel_ms {
        eprintln!("    {name:<10} {ms:>10.1} ms total {:>8.2} ms/batch {:>9.1} MB/s", ms / batches, mb / (ms / 1e3));
    }
    let gpu_ms: f64 = stats.kernel_ms.iter().map(|k| k.1).sum();
    if gpu_ms > 0.0 {
        eprintln!("    {:<10} {gpu_ms:>10.1} ms total {:>8.2} ms/batch {:>9.1} MB/s", "sum", gpu_ms / batches, mb / (gpu_ms / 1e3));
    }

    Ok(RunResult {
        engine: "gpu".to_string(),
        config: format!("lvl3-greedy b{} i{}", cfg.batch, cfg.inflight),
        threads: Some(writer_threads),
        real_bytes,
        compressed_bytes: sizes.iter().sum(),
        seconds,
        per_kind: per_kind(corpus, &sizes),
        kernel_ms: stats.kernel_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gzc_gpu::compressor::GpuParams;

    #[test]
    fn gpu_run_synthetic_verifies() {
        let corpus = Corpus::synthetic();
        let cfg = PipelineConfig { batch: 4, inflight: 2, params: GpuParams { depth: 1 } };
        let r = run_gpu(&corpus, &cfg, 2, true).unwrap();
        assert_eq!(r.engine, "gpu");
        assert_eq!(r.config, "lvl3-greedy b4 i2");
        assert_eq!(r.threads, Some(2));
        assert_eq!(r.real_bytes, corpus.real_bytes());
        assert!(r.ratio() > 1.0, "ratio was {}", r.ratio());
        let kind_sum: u64 = r.per_kind.iter().map(|k| k.compressed_bytes).sum();
        assert_eq!(kind_sum, r.compressed_bytes);
        // Same parse and frame writer as the CPU reference: identical output size.
        let cpu = crate::refrun::run_ref(&corpus, 1, false).unwrap();
        assert_eq!(r.compressed_bytes, cpu.compressed_bytes);
    }
}
