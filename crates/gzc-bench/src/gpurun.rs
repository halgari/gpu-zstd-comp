//! GPU compressor benchmark runs (frame path: the GPU emits complete zstd frames).
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use rayon::prelude::*;

use gzc_core::config::BLOCK_SIZE;
use gzc_gpu::compressor::GpuParams;
use gzc_gpu::context::GpuContext;
use gzc_gpu::pipeline::{FrameSink, ParFrameSink, Pipeline, PipelineConfig, vram_bytes_with};

use crate::corpus::Corpus;
use crate::result::{RunResult, per_kind};

/// Per-block results shared by the sink and the writer threads.
struct Frames {
    /// Frame size per block; 0 = not delivered yet (a frame is never empty).
    sizes: Vec<AtomicU64>,
    /// Frame bytes per block, kept only with `verify`.
    kept: Vec<OnceLock<Vec<u8>>>,
    duplicates: AtomicU64,
}

impl Frames {
    fn new(n: usize, verify: bool) -> Self {
        Self {
            sizes: (0..n).map(|_| AtomicU64::new(0)).collect(),
            kept: if verify { (0..n).map(|_| OnceLock::new()).collect() } else { Vec::new() },
            duplicates: AtomicU64::new(0),
        }
    }

    /// Records block `i`'s frame (the stand-in for writing it out).
    fn record(&self, i: usize, frame: Vec<u8>) {
        if self.sizes[i].swap(frame.len() as u64, Ordering::Relaxed) != 0 {
            self.duplicates.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(slot) = self.kept.get(i) {
            let _ = slot.set(frame);
        } else {
            black_box(frame);
        }
    }
}

/// Copies each frame out of the staging buffer and records it, on the pipeline's completion
/// thread.
struct InlineSink<'a>(&'a Frames);

impl FrameSink for InlineSink<'_> {
    fn put(&mut self, index: usize, frame: &[u8]) {
        self.0.record(index, frame.to_vec());
    }
}

/// `InlineSink` for several delivery threads at once.
struct ParSink<'a>(&'a Frames);

impl ParFrameSink for ParSink<'_> {
    fn put(&self, index: usize, frame: &[u8]) {
        self.0.record(index, frame.to_vec());
    }
}

struct Discard;

impl FrameSink for Discard {
    fn put(&mut self, _index: usize, _frame: &[u8]) {}
}

/// Compresses every block of `corpus` with the streaming GPU pipeline on the frame path: the GPU
/// emits each block's complete zstd frame (Huffman literals with `cfg.params.huffman`, else raw)
/// and the host only copies the bytes out of the staging buffer (the stand-in for writing them
/// out). The calling thread uploads; with `writer_threads == 0` the pipeline's completion thread
/// copies and records every frame (`Pipeline::run_frames`), with N > 0 N delivery threads share
/// each batch (`Pipeline::run_frames_par`). One untimed warmup batch runs first on the same pipeline; the
/// timed region spans from the first upload to the last frame recorded. Each block goes up as its
/// real bytes (`Block::real_len`). With `verify`, every frame is then decompressed with libzstd and
/// compared with the block's real bytes. `kernel_ms` holds the
/// timed run's summed per-kernel GPU time (when the device supports timestamps); a per-batch
/// breakdown is printed to stderr. `cfg.params.emit_frames` is forced on. `preset` names
/// `cfg.params.matching` in the run's config label.
pub fn run_gpu(
    ctx: &GpuContext,
    corpus: &Corpus,
    preset: &str,
    cfg: &PipelineConfig,
    writer_threads: usize,
    verify: bool,
) -> anyhow::Result<RunResult> {
    let cfg = PipelineConfig { params: GpuParams { emit_frames: true, ..cfg.params }, ..*cfg };
    let mut pipe = Pipeline::new(ctx, &cfg)?;
    eprintln!(
        "  k3 mode: {:?}; direct upload: {}; transfer readback: {}; allocated {} MiB",
        pipe.k3_mode(),
        ctx.direct_upload,
        pipe.transfer_readback(),
        vram_bytes_with(&cfg, ctx.direct_upload).div_ceil(1 << 20)
    );
    let blocks: Vec<&[u8]> = corpus.blocks.iter().map(|b| b.real()).collect();

    pipe.run_frames(&blocks[..blocks.len().min(cfg.batch as usize)], &mut Discard)?;

    let frames = Frames::new(blocks.len(), verify);
    let start = Instant::now();
    let stats = if writer_threads == 0 {
        pipe.run_frames(&blocks, &mut InlineSink(&frames))?
    } else {
        pipe.run_frames_par(&blocks, &ParSink(&frames), writer_threads)?
    };
    let seconds = start.elapsed().as_secs_f64();

    let Frames { sizes, kept, duplicates } = frames;
    let sizes: Vec<u64> = sizes.into_iter().map(AtomicU64::into_inner).collect();
    anyhow::ensure!(duplicates.into_inner() == 0, "a block was delivered more than once");
    if let Some(i) = sizes.iter().position(|&s| s == 0) {
        anyhow::bail!("block {i} was never delivered");
    }
    if verify {
        kept.par_iter().zip(&corpus.blocks).enumerate().try_for_each(|(i, (frame, block))| {
            let dec = zstd::bulk::decompress(frame.get().expect("frame kept"), BLOCK_SIZE)?;
            anyhow::ensure!(dec == block.real(), "block {i} did not round-trip through the GPU frame");
            Ok(())
        })?;
        crate::result::log_frame_digest(corpus, kept.iter().map(|f| f.get().expect("frame kept").as_slice()));
    }

    let real_bytes = corpus.real_bytes();
    let mb = real_bytes as f64 / 1e6;
    eprintln!(
        "  gpu {preset} b{} i{} w{writer_threads}: {} batches, pipeline {:.3}s ({:.1} MB/s), end-to-end {seconds:.3}s ({:.1} MB/s)",
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
        eprintln!(
            "    {:<10} {gpu_ms:>10.1} ms total {:>8.2} ms/batch {:>9.1} MB/s",
            "sum",
            gpu_ms / batches,
            mb / (gpu_ms / 1e3)
        );
    }
    // Where the rest of the time goes (see gzc_gpu::pipeline::TRANSFER_NAMES).
    for (name, ms) in &stats.transfer_ms {
        eprintln!("    {name:<18} {ms:>10.1} ms total {:>8.2} ms/batch", ms / batches);
    }

    Ok(RunResult {
        engine: "gpu".to_string(),
        config: format!("{preset} {}b{} i{}", if cfg.params.huffman { "" } else { "rawlit " }, cfg.batch, cfg.inflight),
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
    use gzc_core::frame::FrameOptions;
    use gzc_core::params::LVL3;
    use gzc_core::reference::compress_block_to_frame;

    #[test]
    fn gpu_run_synthetic_verifies() {
        let corpus = Corpus::synthetic();
        // Same parse and frame writer as the CPU reference: identical sizes, with Huffman
        // literals and with raw ones.
        let cpu_bytes = |huffman| -> u64 {
            let opts = FrameOptions { checksum: false, huffman };
            corpus.blocks.iter().map(|b| compress_block_to_frame(b.real(), LVL3, opts).len() as u64).sum()
        };
        let ctx = GpuContext::new().unwrap();
        for (writers, huffman) in [(0, true), (2, true), (0, false)] {
            let params = GpuParams { matching: LVL3, emit_frames: false, huffman };
            let cfg = PipelineConfig { batch: 4, inflight: 2, params };
            let r = run_gpu(&ctx, &corpus, "lvl3", &cfg, writers, true).unwrap();
            assert_eq!(r.engine, "gpu");
            assert_eq!(r.config, if huffman { "lvl3 b4 i2" } else { "lvl3 rawlit b4 i2" });
            assert_eq!(r.threads, Some(writers));
            assert_eq!(r.real_bytes, corpus.real_bytes());
            assert!(r.ratio() > 1.0, "ratio was {}", r.ratio());
            let kind_sum: u64 = r.per_kind.iter().map(|k| k.compressed_bytes).sum();
            assert_eq!(kind_sum, r.compressed_bytes);
            assert_eq!(r.compressed_bytes, cpu_bytes(huffman));
        }
    }
}
