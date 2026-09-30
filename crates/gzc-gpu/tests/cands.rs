//! K1 (Opt3 chains) + K2opt differential test: the GPU candidate words must equal
//! `gzc_core::reference::find_cands` word for word. Every check runs with the subgroup K1 (when the
//! adapter has subgroups) and the fallback K1 (`GpuContext::with_subgroups(false)`).
use gzc_core::block::chunk_file;
use gzc_core::config::BLOCK_SIZE;
use gzc_core::hash::hash_width;
use gzc_core::params::{MatchParams, OPT14, OPT16};
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
    let blocks = all_blocks();
    for ctx in contexts() {
        check(&ctx, &blocks, &OPT16);
        check(&ctx, &blocks[..1], &OPT16);
    }
}

/// opt14 has the same candidates; other depths of the h4 walk (1: h3-dominated, 64: the maximum).
#[test]
fn gpu_cands_match_cpu_other_depths() {
    let blocks = all_blocks();
    for ctx in contexts() {
        check(&ctx, &blocks, &OPT14);
        for depth in [1, 4, 64] {
            check(&ctx, &blocks, &MatchParams { depth, ..OPT16 });
        }
    }
}

#[test]
fn opt_cand_kernel_rejects_non_opt_params() {
    let ctx = GpuContext::new().unwrap();
    assert!(OptCandKernel::new(&ctx, &gzc_core::params::LVL9).is_err());
}

/// Informal (reads the real corpus): K1 + K2opt against `find_cands` on real .dds/.nif blocks.
/// `GZC_CORPUS=/path/to/data/corpus cargo test --release -p gzc-gpu --test cands corpus -- --ignored --nocapture`
/// Takes up to `GZC_CORPUS_BLOCKS` (default 4000) blocks, spread over the files in sorted order.
#[test]
#[ignore]
fn corpus_cands_match_cpu() {
    use std::path::{Path, PathBuf};
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            let ext = p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
            if p.is_dir() {
                walk(&p, out);
            } else if matches!(ext.as_str(), "dds" | "nif") {
                out.push(p);
            }
        }
    }
    let root = std::env::var("GZC_CORPUS")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/corpus").to_string());
    let want_blocks: usize = std::env::var("GZC_CORPUS_BLOCKS").map(|v| v.parse().unwrap()).unwrap_or(4000);
    let mut files = Vec::new();
    walk(Path::new(&root), &mut files);
    files.sort();
    let step = (files.len() / want_blocks).max(1);
    let mut blocks: Vec<(String, Vec<u8>)> = Vec::new();
    for f in files.iter().step_by(step) {
        if blocks.len() >= want_blocks {
            break;
        }
        let bytes = std::fs::read(f).unwrap();
        for (i, b) in chunk_file(&bytes).into_iter().take(4).enumerate() {
            blocks.push((format!("{}[{i}]", f.display()), b.data));
        }
    }
    blocks.truncate(want_blocks);
    eprintln!("{} blocks from {} files", blocks.len(), files.len());
    let ctx = GpuContext::new().expect("GPU required");
    eprintln!("subgroups: {}", ctx.subgroups);
    for chunk in blocks.chunks(500) {
        check(&ctx, chunk, &OPT16);
    }
    eprintln!("{} blocks equal", blocks.len());
}
