//! The cooperative K3's in-kernel fallback: when the lane-layout guard fails, lane 0 of each block
//! runs the sequential parse. `GpuOptions::k3_force_fallback` (test-only) makes every workgroup
//! take that branch; its output must still equal the CPU oracle.
use gzc_core::block::chunk_file;
use gzc_core::fixtures::RUNG1;
use gzc_core::params::LVL3;
use gzc_core::reference::compress_block;
use gzc_core::synth::test_cases;
use gzc_gpu::GpuOptions;
use gzc_gpu::compressor::{GpuParams, K3Mode, Kernels, compress_batch};

#[test]
fn forced_fallback_matches_cpu() {
    let _gpu = gzc_gpu::testing::gpu_test_slot();
    let ctx = gzc_gpu::testing::gpu_with(GpuOptions { k3_force_fallback: true, ..GpuOptions::from_env() });
    let blocks: Vec<(String, Vec<u8>)> = test_cases()
        .into_iter()
        .flat_map(|(name, bytes)| {
            chunk_file(&bytes).into_iter().enumerate().map(move |(i, b)| (format!("{name}[{i}]"), b.data))
        })
        .collect();
    for m in [RUNG1, LVL3] {
        let kernels = Kernels::new(&ctx, GpuParams { matching: m, emit_frames: false, huffman: false }).unwrap();
        if !matches!(kernels.k3_mode(), Some(K3Mode::Coop { .. })) {
            eprintln!("sequential K3 ({:?}): no cooperative kernel to force", kernels.k3_mode());
            return;
        }
        let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
        let got = compress_batch(&ctx, &kernels, &refs).expect("compress_batch");
        for ((name, b), got) in blocks.iter().zip(&got) {
            assert!(*got == compress_block(b, m), "{name} ({m:?}): forced fallback != compress_block");
        }
        eprintln!("{m:?}: {} blocks equal with the forced fallback", blocks.len());
    }
}
