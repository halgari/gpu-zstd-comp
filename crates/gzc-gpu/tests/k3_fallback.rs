//! The cooperative K3's in-kernel fallback: when the lane-layout guard fails, lane 0 of each block
//! runs the sequential parse. `GZC_K3_FORCE_FALLBACK=1` (test-only) makes every workgroup take that
//! branch; its output must still equal the CPU oracle. A separate test binary, because the
//! variable is read by `Kernels::new` and must not leak into other tests.
use gzc_core::block::chunk_file;
use gzc_core::lazy::cases::lazy_test_cases;
use gzc_core::lazy::lazy_parse;
use gzc_core::params::{LVL3, LVL9, RUNG1, RUNG2};
use gzc_core::reference::{Match, compress_block};
use gzc_core::synth::test_cases;
use gzc_gpu::compressor::{GpuParams, K3Mode, Kernels, compress_batch, parses_from_best};
use gzc_gpu::context::GpuContext;

#[test]
fn forced_fallback_matches_cpu() {
    // SAFETY: the only test in this binary, so no other thread reads the environment.
    unsafe { std::env::set_var("GZC_K3_FORCE_FALLBACK", "1") };
    let ctx = GpuContext::new().expect("GPU required");
    let blocks: Vec<(String, Vec<u8>)> = test_cases()
        .into_iter()
        .flat_map(|(name, bytes)| {
            chunk_file(&bytes).into_iter().enumerate().map(move |(i, b)| (format!("{name}[{i}]"), b.data))
        })
        .collect();
    for m in [LVL9, RUNG2, RUNG1, LVL3] {
        let kernels = Kernels::new(&ctx, GpuParams { matching: m, emit_frames: false, huffman: false }).unwrap();
        if !matches!(kernels.k3_mode(), K3Mode::Coop { .. }) {
            eprintln!("sequential K3 ({:?}): no cooperative kernel to force", kernels.k3_mode());
            return;
        }
        let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
        let got = compress_batch(&ctx, &kernels, &refs).expect("compress_batch");
        for ((name, b), got) in blocks.iter().zip(&got) {
            assert!(*got == compress_block(b, m), "{name} ({m:?}): forced fallback != compress_block");
        }
        if m.lazy > 0 {
            let cases = lazy_test_cases();
            let bl: Vec<&[u8]> = cases.iter().map(|c| c.block.as_slice()).collect();
            let bests: Vec<Vec<Match>> = cases.iter().map(|c| c.best.clone()).collect();
            let got = parses_from_best(&ctx, &kernels, &bl, &bests).expect("parses_from_best");
            for (c, got) in cases.iter().zip(&got) {
                assert!(*got == lazy_parse(&c.block, &c.best, &m), "{} ({m:?}): forced fallback != lazy_parse", c.name);
            }
        }
        eprintln!("{m:?}: {} blocks equal with the forced fallback", blocks.len());
    }
}
