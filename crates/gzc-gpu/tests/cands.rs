//! K1 (Opt3 chains) + K2opt differential test: the GPU candidate words must equal
//! `gzc_core::reference::find_cands` word for word. Every check runs with the subgroup K1 (when the
//! adapter has subgroups) and the fallback K1 (`GpuContext::with_subgroups(false)`).
use gzc_core::block::chunk_file;
use gzc_core::config::BLOCK_SIZE;
use gzc_core::hash::hash_width;
use gzc_core::params::{SparseChain, MatchParams, OPT14, OPT16, OPT16P1, OptParams};
use gzc_core::reference::{CandWords, chains, find_cands, unpack_cands};
use gzc_core::synth::test_cases;
use gzc_gpu::compressor::{OptCandKernel, cands_from_blocks};
use gzc_gpu::context::GpuContext;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
}

/// Every test case chunked into padded blocks, plus small-alphabet blocks (many 3-byte matches).
fn all_blocks() -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = test_cases()
        .into_iter()
        .flat_map(|(name, bytes)| {
            chunk_file(&bytes).into_iter().enumerate().map(move |(i, b)| (format!("{name}[{i}]"), b.data))
        })
        .collect();
    for alphabet in [2u64, 3, 5, 17] {
        let mut r = Rng(0x9E37_79B9_7F4A_7C15 ^ alphabet);
        out.push((format!("alphabet{alphabet}"), (0..BLOCK_SIZE).map(|_| (r.next() >> 32) as u8 % alphabet as u8).collect()));
    }
    out
}

fn contexts() -> Vec<GpuContext> {
    let sg = GpuContext::new().expect("GPU required for gzc-gpu tests");
    let fallback = GpuContext::with_subgroups(false).expect("GPU required for gzc-gpu tests");
    if !sg.subgroups {
        eprintln!("adapter without subgroups (or GZC_NO_SUBGROUPS set): only the fallback K1 is tested");
    }
    vec![sg, fallback]
}

fn first_diff(got: &[CandWords], want: &[CandWords]) -> Option<String> {
    let p = (0..want.len()).find(|&p| got[p] != want[p])?;
    Some(format!("p {p}: gpu {:?} cpu {:?}", unpack_cands(got[p]), unpack_cands(want[p])))
}

fn check(ctx: &GpuContext, blocks: &[(String, Vec<u8>)], params: &MatchParams) {
    let kernel = OptCandKernel::new(ctx, params).expect("OptCandKernel::new");
    // The subgroup K1 passes its self-test (built for these chains) wherever it may run.
    assert_eq!(kernel.uses_subgroups(), gzc_gpu::chains::ChainsKernel::subgroup_kernel_possible(ctx), "{params:?}");
    let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
    let got = cands_from_blocks(ctx, &kernel, &refs).expect("cands_from_blocks");
    assert_eq!(got.len(), blocks.len());
    for ((name, block), got) in blocks.iter().zip(&got) {
        let want = find_cands(block, &chains(block, params), params);
        if let Some(d) = first_diff(got, &want) {
            panic!("sg={} depth {} {name}: {d}", kernel.uses_subgroups(), params.depth);
        }
    }
}

/// K2opt's h4 fingerprint skip (k2_opt.wgsl header): the 16-bit 4-byte hash is injective in
/// byte 3 for fixed bytes 0..3, so an h4-chain entry whose first 4 bytes differ from p's shares
/// fewer than 3 bytes with it (the spec's fingerprint caveat never applies). Exhaustive over byte 3
/// for random prefixes.
#[test]
fn h4_hash_is_injective_in_byte_3() {
    let mut r = Rng(7);
    for _ in 0..20_000 {
        let abc = [r.byte(), r.byte(), r.byte()];
        let mut seen = [false; 1 << 16];
        for v in 0..=255u8 {
            let h = hash_width(&[abc[0], abc[1], abc[2], v, 0, 0, 0, 0], 0, 4) as usize;
            assert!(!seen[h], "{abc:?}: byte 3 collision");
            seen[h] = true;
        }
    }
}

#[test]
fn gpu_cands_match_cpu_opt16() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let blocks = all_blocks();
    for ctx in contexts() {
        check(&ctx, &blocks, &OPT16);
        check(&ctx, &blocks[..1], &OPT16);
    }
}

/// opt14 has the same candidates; other depths of the h4 walk (1: h3-dominated, 64: the maximum).
#[test]
fn gpu_cands_match_cpu_other_depths() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let blocks = all_blocks();
    for ctx in contexts() {
        check(&ctx, &blocks, &OPT14);
        for depth in [1, 4, 64] {
            check(&ctx, &blocks, &MatchParams { depth, ..OPT16 });
        }
    }
}

/// M6 `opt16p1` (S3 candidates): h4 depth 8, h3 depth 4 and the 6-, 10- and 12-byte sparse
/// chains on every 4th position, 16 deep, in one merged walk; and other long chain shapes (
/// one, two or three chains, stride 8, depths 1 and 64, widths 5 to 12) and h4 depths.
#[test]
fn gpu_cands_match_cpu_opt16p1() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let lc = |width, stride, depth| Some(SparseChain { width, stride, depth });
    let with = |depth, sparse_chains| MatchParams {
        depth,
        opt: Some(OptParams { sparse_chains, ..OPT16P1.opt.unwrap() }),
        ..OPT16P1
    };
    let blocks = all_blocks();
    for ctx in contexts() {
        check(&ctx, &blocks, &OPT16P1);
        check(&ctx, &blocks[..1], &OPT16P1);
        check(&ctx, &blocks, &with(1, [lc(5, 8, 1), lc(12, 4, 64), None]));
        check(&ctx, &blocks, &with(32, [lc(8, 4, 3), lc(7, 8, 16), None]));
        check(&ctx, &blocks, &with(64, [lc(9, 4, 16), None, None]));
    }
}

/// K1 builds sparse chains of word-aligned slots only (stride 4 or 8).
#[test]
fn opt_cand_kernel_rejects_unaligned_long_chains() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let ctx = GpuContext::new().unwrap();
    let o = OPT16P1.opt.unwrap();
    let m = MatchParams { opt: Some(OptParams { sparse_chains: [Some(SparseChain { width: 10, stride: 2, depth: 16 }), None, None], ..o }), ..OPT16P1 };
    assert!(m.validate().is_ok());
    assert!(OptCandKernel::new(&ctx, &m).is_err());
}

#[test]
fn opt_cand_kernel_rejects_non_opt_params() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let ctx = GpuContext::new().unwrap();
    assert!(OptCandKernel::new(&ctx, &gzc_core::params::LVL9).is_err());
}

/// Informal (reads the real corpus): K1 + K2opt against `find_cands` on real .dds/.nif blocks.
/// `GZC_CORPUS=/path/to/data/corpus cargo test --release -p gzc-gpu --test cands corpus -- --ignored --nocapture`
/// `GZC_CANDS_PRESET` (comma-separated) names the opt presets (default `opt16,opt16p1`).
/// Takes up to `GZC_CORPUS_BLOCKS` (default 4000) blocks sampled uniformly over the whole corpus:
/// every k-th of all (file, block) pairs, files in sorted path order, k = total blocks / wanted.
/// Skips (with a message) when the corpus directory does not exist.
#[test]
#[ignore]
fn corpus_cands_match_cpu() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let want_blocks: usize = std::env::var("GZC_CORPUS_BLOCKS").map(|v| v.parse().unwrap()).unwrap_or(4000);
    let Some(blocks) = gzc_core::testdata::corpus_sample_named(want_blocks) else { return };
    let names = std::env::var("GZC_CANDS_PRESET").unwrap_or_else(|_| "opt16,opt16p1".to_string());
    let ctx = GpuContext::new().expect("GPU required");
    for name in names.split(',') {
        let params = gzc_core::params::preset(name).unwrap();
        eprintln!("{name}, subgroups: {}", ctx.subgroups);
        for chunk in blocks.chunks(500) {
            check(&ctx, chunk, &params);
        }
        eprintln!("{name}: {} blocks equal", blocks.len());
    }
}
