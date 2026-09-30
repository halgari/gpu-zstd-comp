//! CPU reference compressor benchmark runs.
use std::time::Instant;

use rayon::prelude::*;

use gzc_core::config::BLOCK_SIZE;
use gzc_core::frame::FrameOptions;
use gzc_core::params::MatchParams;
use gzc_core::reference::compress_block_to_frame;

use crate::corpus::Corpus;
use crate::result::{per_kind, RunResult};

/// Compresses every block in `corpus` to a zstd frame using the CPU reference
/// compressor with `params` (the preset called `name`, which labels the run), using a rayon
/// pool pinned to `threads` threads. Runs one untimed warmup pass over the
/// first `min(corpus.len(), 64)` blocks (to pay JIT/allocator/cache warmup
/// cost outside the timed region), then times a full pass over every block;
/// per-block frame sizes come from that timed pass. When `verify` is set,
/// every frame produced by the timed pass is decompressed with libzstd
/// afterward and checked against the original (padded) block; any mismatch
/// is an error.
/// `params` must be `cpu_supports`ed (`compress_block` panics otherwise).
pub fn run_ref(corpus: &Corpus, name: &str, params: MatchParams, threads: usize, verify: bool) -> anyhow::Result<RunResult> {
    let opts = FrameOptions::default();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;

    let warmup_n = corpus.blocks.len().min(64);
    pool.install(|| {
        corpus.blocks[..warmup_n].par_iter().for_each(|block| {
            let _ = compress_block_to_frame(&block.data, params, opts);
        });
    });

    let start = Instant::now();
    let frames: Vec<Vec<u8>> = pool.install(|| {
        corpus.blocks.par_iter().map(|block| compress_block_to_frame(&block.data, params, opts)).collect()
    });
    let seconds = start.elapsed().as_secs_f64();

    if verify {
        pool.install(|| -> anyhow::Result<()> {
            corpus.blocks.par_iter().zip(frames.par_iter()).enumerate().try_for_each(
                |(i, (block, frame))| -> anyhow::Result<()> {
                    let dec = zstd::bulk::decompress(frame, BLOCK_SIZE)?;
                    if dec != block.data {
                        anyhow::bail!("block {i} did not round-trip through the reference frame");
                    }
                    Ok(())
                },
            )
        })?;
    }

    let sizes: Vec<u64> = frames.iter().map(|f| f.len() as u64).collect();
    let compressed_bytes: u64 = sizes.iter().sum();
    let real_bytes = corpus.real_bytes();
    let per_kind_stats = per_kind(corpus, &sizes);

    Ok(RunResult {
        engine: "cpu-ref".to_string(),
        config: name.to_string(),
        threads: Some(threads),
        real_bytes,
        compressed_bytes,
        seconds,
        per_kind: per_kind_stats,
        kernel_ms: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::Corpus;

    #[test]
    fn ref_run_ratio_and_verify() {
        let corpus = Corpus::synthetic();
        let result = run_ref(&corpus, "lvl3", gzc_core::params::LVL3, 2, true).unwrap();

        assert_eq!(result.engine, "cpu-ref");
        assert_eq!(result.config, "lvl3");
        assert_eq!(result.threads, Some(2));
        assert!(result.ratio() > 1.0, "ratio was {}", result.ratio());
    }
}
