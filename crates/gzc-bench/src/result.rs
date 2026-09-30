//! RunResult types shared by all benchmark backends.
use crate::corpus::{Corpus, Kind};

#[derive(serde::Serialize, Clone, Debug)]
pub struct KindStat {
    pub kind: Kind,
    pub real_bytes: u64,
    pub compressed_bytes: u64,
}

#[derive(serde::Serialize, Clone, Debug)]
pub struct RunResult {
    /// "cpu-libzstd" | "cpu-ref" | "gpu"
    pub engine: String,
    /// e.g. "L3", "lvl3", "lvl3 b512 i3"
    pub config: String,
    /// CPU threads used (for gpu: CPU threads used for host-side work).
    pub threads: Option<usize>,
    pub real_bytes: u64,
    pub compressed_bytes: u64,
    pub seconds: f64,
    pub per_kind: Vec<KindStat>,
    /// Named GPU kernel timings in milliseconds; empty for CPU runs.
    pub kernel_ms: Vec<(String, f64)>,
}

impl RunResult {
    /// Throughput over real (unpadded) bytes, MB = 10^6 bytes.
    pub fn mb_per_s(&self) -> f64 {
        if self.seconds <= 0.0 {
            return 0.0;
        }
        (self.real_bytes as f64 / 1_000_000.0) / self.seconds
    }

    pub fn ratio(&self) -> f64 {
        if self.compressed_bytes == 0 {
            return 0.0;
        }
        self.real_bytes as f64 / self.compressed_bytes as f64
    }
}

/// Builds per-kind stats from per-block compressed sizes (`compressed_sizes[i]`
/// is the compressed size of `corpus.blocks[i]`).
pub fn per_kind(corpus: &Corpus, compressed_sizes: &[u64]) -> Vec<KindStat> {
    // Fixed, stable order: Dds, Nif, Other.
    let mut dds = KindStat { kind: Kind::Dds, real_bytes: 0, compressed_bytes: 0 };
    let mut nif = KindStat { kind: Kind::Nif, real_bytes: 0, compressed_bytes: 0 };
    let mut other = KindStat { kind: Kind::Other, real_bytes: 0, compressed_bytes: 0 };

    for ((block, kind), &compressed) in
        corpus.blocks.iter().zip(corpus.kinds.iter()).zip(compressed_sizes.iter())
    {
        let stat = match kind {
            Kind::Dds => &mut dds,
            Kind::Nif => &mut nif,
            Kind::Other => &mut other,
        };
        stat.real_bytes += block.real_len as u64;
        stat.compressed_bytes += compressed;
    }

    vec![dds, nif, other].into_iter().filter(|s| s.real_bytes > 0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_kind_sums_match_totals() {
        let corpus = Corpus::synthetic();
        let sizes: Vec<u64> =
            corpus.blocks.iter().map(|b| (b.real_len as u64 / 2) + 1).collect();
        let stats = per_kind(&corpus, &sizes);

        let real_sum: u64 = stats.iter().map(|s| s.real_bytes).sum();
        let compressed_sum: u64 = stats.iter().map(|s| s.compressed_bytes).sum();

        assert_eq!(real_sum, corpus.real_bytes());
        assert_eq!(compressed_sum, sizes.iter().sum::<u64>());
    }

    #[test]
    fn ratio_and_throughput() {
        let r = RunResult {
            engine: "cpu-libzstd".into(),
            config: "L3".into(),
            threads: Some(8),
            real_bytes: 2_000_000,
            compressed_bytes: 1_000_000,
            seconds: 1.0,
            per_kind: vec![],
            kernel_ms: vec![],
        };
        assert_eq!(r.ratio(), 2.0);
        assert_eq!(r.mb_per_s(), 2.0);
    }

    #[test]
    fn zero_seconds_and_zero_compressed_are_safe() {
        let r = RunResult {
            engine: "cpu-libzstd".into(),
            config: "L3".into(),
            threads: Some(1),
            real_bytes: 0,
            compressed_bytes: 0,
            seconds: 0.0,
            per_kind: vec![],
            kernel_ms: vec![],
        };
        assert_eq!(r.ratio(), 0.0);
        assert_eq!(r.mb_per_s(), 0.0);
    }
}
