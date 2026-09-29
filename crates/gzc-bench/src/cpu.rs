//! CPU baseline runs using libzstd for comparison.
use std::time::Instant;

use rayon::prelude::*;

use crate::corpus::Corpus;
use crate::result::{per_kind, RunResult};

fn compress_block(compressor: &mut zstd::bulk::Compressor<'static>, block: &gzc_core::block::Block) -> anyhow::Result<u64> {
    let compressed = compressor.compress(&block.data)?;
    Ok(compressed.len() as u64)
}

/// Compresses every block in `corpus` with libzstd at `level`, using a rayon
/// pool pinned to `threads` threads. Runs one untimed warmup pass over the
/// first `min(corpus.len(), 64)` blocks (to pay JIT/allocator/cache warmup
/// cost outside the timed region), then times a full pass over every block;
/// per-block compressed sizes come from that timed pass.
pub fn run_cpu(corpus: &Corpus, level: i32, threads: usize) -> anyhow::Result<RunResult> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;

    let warmup_n = corpus.blocks.len().min(64);
    pool.install(|| -> anyhow::Result<()> {
        corpus.blocks[..warmup_n]
            .par_iter()
            .map_init(
                || zstd::bulk::Compressor::new(level).expect("zstd compressor"),
                |compressor, block| compress_block(compressor, block),
            )
            .collect::<anyhow::Result<Vec<u64>>>()?;
        Ok(())
    })?;

    let start = Instant::now();
    let sizes: Vec<u64> = pool.install(|| -> anyhow::Result<Vec<u64>> {
        corpus
            .blocks
            .par_iter()
            .map_init(
                || zstd::bulk::Compressor::new(level).expect("zstd compressor"),
                |compressor, block| compress_block(compressor, block),
            )
            .collect::<anyhow::Result<Vec<u64>>>()
    })?;
    let seconds = start.elapsed().as_secs_f64();

    let compressed_bytes: u64 = sizes.iter().sum();
    let real_bytes = corpus.real_bytes();
    let per_kind_stats = per_kind(corpus, &sizes);

    Ok(RunResult {
        engine: "cpu-libzstd".to_string(),
        config: format!("L{level}"),
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
    use gzc_core::config::BLOCK_SIZE;

    #[test]
    fn cpu_run_ratio_and_round_trip() {
        let corpus = Corpus::synthetic();
        let result = run_cpu(&corpus, 3, 2).unwrap();

        assert_eq!(result.engine, "cpu-libzstd");
        assert_eq!(result.config, "L3");
        assert_eq!(result.threads, Some(2));
        assert!(result.ratio() > 1.0, "ratio was {}", result.ratio());

        let mut compressor = zstd::bulk::Compressor::new(3).unwrap();
        for block in &corpus.blocks {
            let compressed = compressor.compress(&block.data).unwrap();
            let decompressed = zstd::bulk::decompress(&compressed, BLOCK_SIZE).unwrap();
            assert_eq!(decompressed, block.data);
        }
    }
}
