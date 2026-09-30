//! K1 differential test: GPU hash chains must equal `gzc_core::reference::chains` (Dfast: the
//! long then short `compute_preds`; Single: one chain over `hash_width(.., min_match)`).
use gzc_core::block::chunk_file;
use gzc_core::hash::{compute_preds, hash_long, hash_short, hash_width};
use gzc_core::params::{Hashes, LVL3, MatchParams, RUNG1};
use gzc_core::reference::chains;
use gzc_core::synth::test_cases;
use gzc_gpu::chains::gpu_preds;
use gzc_gpu::context::GpuContext;

/// Every test case chunked into padded blocks, labelled "name[i]".
fn all_blocks() -> Vec<(String, Vec<u8>)> {
    test_cases()
        .into_iter()
        .flat_map(|(name, bytes)| {
            chunk_file(&bytes)
                .into_iter()
                .enumerate()
                .map(move |(i, b)| (format!("{name}[{i}]"), b.data))
        })
        .collect()
}

fn check(ctx: &GpuContext, blocks: &[(String, Vec<u8>)], params: &MatchParams) {
    let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
    let preds = gpu_preds(ctx, &refs, params).expect("gpu_preds");
    assert_eq!(preds.len(), blocks.len());
    for ((name, block), got) in blocks.iter().zip(&preds) {
        let want = chains(block, params);
        assert_eq!(got.len(), want.len(), "{name}: chain count");
        for (h, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.len(), w.len(), "{name}: chain {h} length");
            if let Some(p) = (0..w.len()).find(|&p| g[p] != w[p]) {
                panic!("{name}: chain {h} pred mismatch at {p}: gpu {} cpu {}", g[p], w[p]);
            }
        }
    }
}

#[test]
fn gpu_preds_match_cpu_all_blocks_one_batch() {
    let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
    // The Dfast chains are the long- and short-hash preds, in that order.
    let block = &all_blocks()[0].1;
    assert_eq!(chains(block, &LVL3), vec![compute_preds(block, hash_long), compute_preds(block, hash_short)]);
    check(&ctx, &all_blocks(), &LVL3);
}

#[test]
fn gpu_preds_match_cpu_batch_of_one() {
    let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
    for b in all_blocks() {
        check(&ctx, std::slice::from_ref(&b), &LVL3);
    }
}

/// Single-hash presets build one chain over `hash_width(min_match)`, for every allowed width
/// (4 is rung1's; 8 takes the mask(4) = 0xFFFFFFFF branch).
#[test]
fn k1_single_hash_preds_match_cpu() {
    let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
    let blocks = all_blocks();
    let block = &blocks[0].1;
    assert_eq!(chains(block, &RUNG1), vec![compute_preds(block, |b: &[u8], p: usize| hash_width(b, p, 4))]);
    check(&ctx, &blocks, &RUNG1);
    for b in blocks.iter().step_by(3) {
        check(&ctx, std::slice::from_ref(b), &RUNG1);
    }
    for min_match in 5..=8 {
        let p = MatchParams { hashes: Hashes::Single, min_match, ..RUNG1 };
        check(&ctx, &blocks, &p);
    }
}

#[test]
fn pack_blocks_appends_zero_word() {
    let a = vec![1u8; gzc_core::config::BLOCK_SIZE];
    let packed = gzc_gpu::context::pack_blocks(&[&a, &a]);
    assert_eq!(packed.len(), 2 * gzc_core::config::BLOCK_SIZE / 4 + 1);
    assert_eq!(packed[0], 0x0101_0101);
    assert_eq!(*packed.last().unwrap(), 0);
}
