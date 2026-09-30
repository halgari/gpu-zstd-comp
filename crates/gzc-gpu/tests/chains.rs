//! K1 differential test: GPU hash chains must equal `gzc_core::reference::chains` (Dfast: the
//! long then short `compute_preds`; Single: one chain over `hash_width(.., min_match)`). Every
//! check runs both K1 kernels: the subgroup kernel (when the adapter has subgroups) and the
//! workgroup-sort fallback (`GpuContext::with_subgroups(false)`).
use gzc_core::block::chunk_file;
use gzc_core::config::{BLOCK_SIZE, HASHED_POSITIONS};
use gzc_core::hash::{compute_preds, hash_long, hash_short, hash_width};
use gzc_core::params::{Hashes, LVL3, LVL9, MatchParams, RUNG1};
use gzc_core::reference::chains;
use gzc_core::synth::test_cases;
use gzc_gpu::chains::{
    ChainsKernel, ChainsOptions, PRED_POS, gpu_preds, gpu_preds_with, head_bytes, pred_bytes, pred_fp, pred_of_word,
};
use gzc_gpu::context::{GpuContext, pack_blocks};

/// Every test case chunked into padded blocks, labelled "name[i]", plus small-alphabet blocks
/// (many distinct repeated hashes per tile: several equal-hash groups per subgroup tile).
fn all_blocks() -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = test_cases()
        .into_iter()
        .flat_map(|(name, bytes)| {
            chunk_file(&bytes)
                .into_iter()
                .enumerate()
                .map(move |(i, b)| (format!("{name}[{i}]"), b.data))
        })
        .collect();
    for alphabet in [2u64, 3, 5] {
        let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ alphabet;
        let block = (0..BLOCK_SIZE)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 32) % alphabet) as u8
            })
            .collect();
        out.push((format!("alphabet{alphabet}"), block));
    }
    out
}

/// The default context (subgroup kernel when supported) and the fallback context.
fn contexts() -> Vec<GpuContext> {
    let sg = GpuContext::new().expect("GPU required for gzc-gpu tests");
    let fallback = GpuContext::with_subgroups(false).expect("GPU required for gzc-gpu tests");
    assert!(!fallback.subgroups);
    if !sg.subgroups {
        eprintln!("adapter without subgroups (or GZC_NO_SUBGROUPS set): only the fallback K1 is tested");
    }
    vec![sg, fallback]
}

fn compare(name: &str, got: &[Vec<u32>], block: &[u8], params: &MatchParams) {
    let want = chains(block, params);
    assert_eq!(got.len(), want.len(), "{name}: chain count");
    for (h, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.len(), w.len(), "{name}: chain {h} length");
        if let Some(p) = (0..w.len()).find(|&p| g[p] != w[p]) {
            panic!("{name}: chain {h} pred mismatch at {p}: gpu {} cpu {}", g[p], w[p]);
        }
    }
}

fn check(ctx: &GpuContext, blocks: &[(String, Vec<u8>)], params: &MatchParams) {
    let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
    let preds = gpu_preds(ctx, &refs, params).expect("gpu_preds");
    assert_eq!(preds.len(), blocks.len());
    for ((name, block), got) in blocks.iter().zip(&preds) {
        let path = if ctx.subgroups { "subgroup" } else { "fallback" };
        compare(&format!("{path} {name}"), got, block, params);
    }
}

#[test]
fn gpu_preds_match_cpu_all_blocks_one_batch() {
    // The Dfast chains are the long- and short-hash preds, in that order.
    let block = &all_blocks()[0].1;
    assert_eq!(chains(block, &LVL3), vec![compute_preds(block, hash_long), compute_preds(block, hash_short)]);
    for ctx in contexts() {
        check(&ctx, &all_blocks(), &LVL3);
    }
}

#[test]
fn gpu_preds_match_cpu_batch_of_one() {
    for ctx in contexts() {
        for b in all_blocks() {
            check(&ctx, std::slice::from_ref(&b), &LVL3);
        }
    }
}

/// Single-hash presets build one chain over `hash_width(min_match)`, for every allowed width
/// (4 is rung1's; 8 takes the mask(4) = 0xFFFFFFFF branch).
#[test]
fn k1_single_hash_preds_match_cpu() {
    let blocks = all_blocks();
    let block = &blocks[0].1;
    assert_eq!(chains(block, &RUNG1), vec![compute_preds(block, |b: &[u8], p: usize| hash_width(b, p, 4))]);
    for ctx in contexts() {
        check(&ctx, &blocks, &RUNG1);
        for b in blocks.iter().step_by(3) {
            check(&ctx, std::slice::from_ref(b), &RUNG1);
        }
        for min_match in 5..=8 {
            let p = MatchParams { hashes: Hashes::Single, min_match, ..RUNG1 };
            check(&ctx, &blocks, &p);
        }
    }
}

/// The context picks the kernel: the subgroup one exactly when the device has subgroups.
#[test]
fn k1_kernel_follows_context() {
    for ctx in contexts() {
        let i = &ctx.adapter_info;
        eprintln!("{}: subgroups {} (sizes {}..={})", i.name, ctx.subgroups, i.subgroup_min_size, i.subgroup_max_size);
        assert_eq!(ChainsKernel::new(&ctx, &LVL9).unwrap().uses_subgroups(), ctx.subgroups);
    }
}

/// Runs `kernel` on `blocks` with the given head/pred buffers and checks every chain.
fn run_check(ctx: &GpuContext, kernel: &ChainsKernel, params: &MatchParams, blocks: &[&(String, Vec<u8>)], head: &wgpu::Buffer, pred: &wgpu::Buffer, what: &str) {
    let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
    let packed = pack_blocks(&refs);
    let data = ctx.storage_buffer("test.data", (packed.len() * 4) as u64, false);
    ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&packed));
    let got = kernel.run(ctx, &data, head, pred, refs.len() as u32);
    for ((name, block), g) in blocks.iter().zip(&got) {
        compare(&format!("sg={} {what} {name}", kernel.uses_subgroups()), g, block, params);
    }
}

/// `head` is per-dispatch scratch: one head buffer, shared by kernels of different presets and
/// reused across dispatches of varying size (stale entries of other blocks and presets in it),
/// must give the CPU chains every time. `groups` 1 makes one workgroup build every chain of the
/// dispatch in a row (tags 1..=n on one table), 3 a few each; `wide_masks` covers the kernel's
/// path for subgroups wider than 32 lanes.
#[test]
fn k1_head_reuse_across_dispatches() {
    let blocks = all_blocks();
    for ctx in contexts() {
        let cap = blocks.len() as u32;
        let head = ctx.storage_buffer("test.head", head_bytes(cap, 2), false);
        let pred = ctx.storage_buffer("test.pred", pred_bytes(cap, 2), true);
        let opts = |groups, wide_masks| ChainsOptions { groups, wide_masks, ..ChainsOptions::default() };
        let kernels: Vec<(MatchParams, ChainsKernel)> = [
            (LVL3, opts(None, false)),
            (RUNG1, opts(Some(1), false)),
            (LVL3, opts(Some(3), true)),
            (LVL9, opts(Some(1), false)),
            (RUNG1, opts(None, true)),
            (LVL3, opts(Some(1), false)),
        ]
        .into_iter()
        .map(|(p, o)| (p, ChainsKernel::with_options(&ctx, &p, o).unwrap()))
        .collect();
        for round in 0..18usize {
            let (params, kernel) = &kernels[round % kernels.len()];
            // Rotating, varying-size subsets, so stale head entries of other blocks are present.
            let n = [blocks.len(), 5, 1, blocks.len() - 3, 9][round % 5];
            let subset: Vec<&(String, Vec<u8>)> = (0..n).map(|i| &blocks[(i + 7 * round) % blocks.len()]).collect();
            run_check(&ctx, kernel, params, &subset, &head, &pred, &format!("round {round}"));
        }
    }
}

/// Two live contexts (separate wgpu instances, whose buffer ids can coincide) interleave
/// dispatches on their own head buffers; neither may disturb the other.
#[test]
fn k1_two_contexts_interleaved() {
    let blocks = all_blocks();
    let ctxs = [GpuContext::new().unwrap(), GpuContext::new().unwrap()];
    let bufs: Vec<(wgpu::Buffer, wgpu::Buffer)> = ctxs
        .iter()
        .map(|c| (c.storage_buffer("head", head_bytes(4, 2), false), c.storage_buffer("pred", pred_bytes(4, 2), true)))
        .collect();
    let kernels: Vec<ChainsKernel> = ctxs
        .iter()
        .map(|c| ChainsKernel::with_options(c, &LVL3, ChainsOptions { groups: Some(1), ..ChainsOptions::default() }).unwrap())
        .collect();
    for round in 0..8usize {
        let i = round % 2;
        let subset: Vec<&(String, Vec<u8>)> = (0..4).map(|k| &blocks[(k + 5 * round) % blocks.len()]).collect();
        run_check(&ctxs[i], &kernels[i], &LVL3, &subset, &bufs[i].0, &bufs[i].1, &format!("ctx {i} round {round}"));
    }
}

/// K1's raw pred words: the predecessor (PRED_POS for none) in bits 0..17 and `pred_fp` of the
/// word's own position above, for every hashed position; the unhashed tail holds "none" and no
/// fingerprint. Both kernels, Dfast (both chains carry the position's fingerprint) and Single.
#[test]
fn k1_pred_words_carry_fingerprints() {
    let blocks = all_blocks();
    let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
    let packed = pack_blocks(&refs);
    let n = refs.len() as u32;
    for ctx in contexts() {
        let data = ctx.storage_buffer("test.data", (packed.len() * 4) as u64, false);
        ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&packed));
        for params in [LVL3, LVL9] {
            let kernel = ChainsKernel::new(&ctx, &params).unwrap();
            let nh = kernel.n_hashes() as usize;
            let head = ctx.storage_buffer("test.head", head_bytes(n, nh as u32), false);
            let pred = ctx.storage_buffer("test.pred", pred_bytes(n, nh as u32), true);
            let words = kernel.run_words(&ctx, &data, &head, &pred, n);
            for (b, (name, block)) in blocks.iter().enumerate() {
                let want = chains(block, &params);
                for (c, want) in want.iter().enumerate() {
                    let got = &words[(b * nh + c) * BLOCK_SIZE..][..BLOCK_SIZE];
                    for p in 0..BLOCK_SIZE {
                        let w = got[p];
                        let expect = if p < HASHED_POSITIONS { pred_fp(block, p) } else { 0 };
                        let what = format!("sg={} {params:?} {name} chain {c} p {p}: word {w:#x}", kernel.uses_subgroups());
                        assert_eq!(w & !PRED_POS, expect, "{what}: fingerprint");
                        assert_eq!(pred_of_word(w), want[p], "{what}: predecessor");
                    }
                }
            }
        }
    }
}

/// A subgroup kernel that fails its self-test is replaced by the fallback, which still builds the
/// right chains.
#[test]
fn k1_failed_self_test_falls_back() {
    let ctx = GpuContext::new().unwrap();
    if !ctx.subgroups {
        return;
    }
    let good = ChainsKernel::with_options(&ctx, &LVL9, ChainsOptions::default()).unwrap();
    assert!(good.uses_subgroups(), "self-test failed on this adapter");
    let opts = ChainsOptions { break_subgroup_kernel: true, ..ChainsOptions::default() };
    let broken = ChainsKernel::with_options(&ctx, &LVL9, opts).unwrap();
    assert!(!broken.uses_subgroups());
    let blocks = all_blocks();
    let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
    let preds = gpu_preds_with(&ctx, &broken, &refs).unwrap();
    for ((name, block), got) in blocks.iter().zip(&preds) {
        compare(name, got, block, &LVL9);
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

/// The chains of a short key (`MatchParams::hash_bits`), from both K1 kernels.
#[test]
fn chains_over_short_keys() {
    for ctx in contexts() {
        for params in [MatchParams { hash_bits: 13, ..LVL9 }, MatchParams { hash_bits: 11, min_match: 6, ..LVL9 }] {
            let blocks = all_blocks();
            let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
            let preds = gpu_preds(&ctx, &refs, &params).unwrap();
            for ((name, block), got) in blocks.iter().zip(&preds) {
                compare(name, got, block, &params);
            }
        }
    }
}

/// The bucket-sorted K1 (speed2 E2): every slot equals `gzc_core::hash::bucket_sort` (position
/// and fingerprint bits), for 11..=13-bit keys and min_match 4 and 6, in one batch and block by
/// block. It is selected for the sorted presets whenever the device has >= 32-lane subgroups and
/// the table fits its workgroup memory, and never without subgroups or for 16-bit keys.
#[test]
fn sorted_k1_matches_bucket_sort() {
    use gzc_core::params::{LVL9S12, LVL9S13};
    use gzc_gpu::sorted::{SortKernel, sorted_words, table_bytes};
    let blocks = all_blocks();
    let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
    let fallback = GpuContext::with_subgroups(false).expect("GPU required for gzc-gpu tests");
    assert!(SortKernel::new(&fallback, &LVL9S13).unwrap().is_none(), "sorted K1 without subgroups");
    assert!(SortKernel::new(&ctx, &LVL9).unwrap().is_none(), "sorted K1 for a 16-bit key");
    for params in [LVL9S13, LVL9S12, MatchParams { hash_bits: 11, min_match: 6, ..LVL9 }] {
        let usable = ctx.subgroups
            && ctx.adapter_info.subgroup_min_size >= 32
            && ctx.device.limits().max_compute_workgroup_storage_size >= table_bytes(&params);
        let Some(k1) = SortKernel::new(&ctx, &params).unwrap() else {
            assert!(!usable, "{params:?}: sorted K1 not selected on a device that supports it");
            eprintln!("{params:?}: sorted K1 unavailable on this device");
            continue;
        };
        let n = blocks.len() as u32;
        let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
        let packed = pack_blocks(&refs);
        let data = ctx.storage_buffer("test.data", (packed.len() * 4) as u64, false);
        ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&packed));
        let sorted = ctx.storage_buffer("test.sorted", pred_bytes(n, 1), true);
        let rank = ctx.storage_buffer("test.rank", pred_bytes(n, 1), false);
        let got = k1.run(&ctx, &data, &sorted, &rank, n);
        for (b, (name, block)) in blocks.iter().enumerate() {
            let want = sorted_words(block, &params);
            let g = &got[b * BLOCK_SIZE..][..HASHED_POSITIONS];
            if let Some(s) = (0..HASHED_POSITIONS).find(|&s| g[s] != want[s]) {
                panic!("{params:?} {name}: slot {s} gpu {:#x} cpu {:#x}", g[s], want[s]);
            }
        }
        // Block by block: a lone block's words past its end come from the buffer's last word.
        for (name, block) in blocks.iter().step_by(3) {
            let packed = pack_blocks(&[block.as_slice()]);
            let data = ctx.storage_buffer("test.data1", (packed.len() * 4) as u64, false);
            ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&packed));
            let got = k1.run(&ctx, &data, &sorted, &rank, 1);
            assert!(got[..HASHED_POSITIONS] == sorted_words(block, &params)[..HASHED_POSITIONS], "{params:?} {name} alone");
        }
    }
}
