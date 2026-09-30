//! K1 differential test: GPU hash chains must equal gzc_core::hash::compute_preds.
use gzc_core::block::chunk_file;
use gzc_core::hash::{compute_preds, hash_long, hash_short};
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

fn check(ctx: &GpuContext, blocks: &[(String, Vec<u8>)]) {
    let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
    let preds = gpu_preds(ctx, &refs).expect("gpu_preds");
    assert_eq!(preds.len(), blocks.len());
    for ((name, block), got) in blocks.iter().zip(&preds) {
        let long = compute_preds(block, hash_long);
        let short = compute_preds(block, hash_short);
        if let Some(p) = (0..long.len()).find(|&p| got.long[p] != long[p]) {
            panic!("{name}: long pred mismatch at {p}: gpu {} cpu {}", got.long[p], long[p]);
        }
        if let Some(p) = (0..short.len()).find(|&p| got.short[p] != short[p]) {
            panic!("{name}: short pred mismatch at {p}: gpu {} cpu {}", got.short[p], short[p]);
        }
        assert_eq!(got.long.len(), long.len(), "{name}");
        assert_eq!(got.short.len(), short.len(), "{name}");
    }
}

#[test]
fn gpu_preds_match_cpu_all_blocks_one_batch() {
    let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
    check(&ctx, &all_blocks());
}

#[test]
fn gpu_preds_match_cpu_batch_of_one() {
    let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
    for b in all_blocks() {
        check(&ctx, std::slice::from_ref(&b));
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
