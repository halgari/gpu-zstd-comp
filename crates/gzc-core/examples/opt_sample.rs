//! Optimal-parse sample runs on the corpus (M5 T1): ratios of the `opt*` presets on every
//! `every`-th block, and the prior tables of `codes::OPT_PRIOR_*`.
//!
//! Blocks are numbered in corpus order: files under the corpus directory sorted recursively by
//! path, extensions `dds` and `nif` only, each chunked by `block::chunk_file` (the order of
//! `gzc-bench --ext dds,nif` and of the m5-opt-design prototype). Block `i` is in the sample
//! when `i % every == offset`. The design's evaluation sample is `every 50, offset 0`.
//!
//! ```text
//! cargo run --release -p gzc-core --example opt_sample -- eval  <corpus> <every> <offset> [zstd]
//! cargo run --release -p gzc-core --example opt_sample -- train <corpus> <every> <offset>
//! cargo run --release -p gzc-core --example opt_sample -- train-s3 <corpus> <every> <offset>
//! ```
//! `eval` prints the ratio (real bytes / frame bytes) of opt14 and opt16 (and libzstd L14/L16
//! with `zstd`), and checks every opt frame with libzstd. `train` sums the LL/ML/OF code
//! histograms of opt16's output over the sample and prints them scaled to 65536 per table
//! (round to nearest): the `OPT_PRIOR_*` constants. `train-s3` does the same for `s3_train_params`
//! (the opt16 schedule over opt16p1's candidates and segment ends): the `OPT_PRIOR_S3_*`
//! constants. `eval` also prints opt16p1. `THREADS` (default 16) sets the thread count.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use gzc_core::block::chunk_file;
use gzc_core::config::BLOCK_SIZE;
use gzc_core::frame::{write_frame, FrameOptions};
use gzc_core::opt::Hist;
use gzc_core::params::{MatchParams, OptParams, PriorTables, Seed, OPT14, OPT16, OPT16P1};
use gzc_core::reference::{chains, compress_block, find_cands};

struct Blk {
    data: Vec<u8>,
    real: usize,
}

fn load(dir: &Path, every: usize, offset: usize) -> Vec<Blk> {
    let files = gzc_core::testdata::corpus_files(dir);
    let mut out = Vec::new();
    let mut idx = 0usize;
    for f in files {
        let bytes = std::fs::read(&f).unwrap();
        for b in chunk_file(&bytes) {
            if idx % every == offset {
                out.push(Blk { data: b.data, real: b.real_len });
            }
            idx += 1;
        }
    }
    eprintln!("sampled {} of {idx} blocks (every {every}, offset {offset}), block {BLOCK_SIZE} B", out.len());
    out
}

/// `f` over every block on `THREADS` threads, results in block order.
fn par_map<T: Send>(blocks: &[Blk], f: impl Fn(&Blk) -> T + Sync) -> Vec<T> {
    let threads: usize = std::env::var("THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let next = AtomicUsize::new(0);
    let mut all: Vec<(usize, T)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= blocks.len() {
                            return mine;
                        }
                        mine.push((i, f(&blocks[i])));
                    }
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    all.sort_by_key(|x| x.0);
    all.into_iter().map(|x| x.1).collect()
}

fn ratio(blocks: &[Blk], sizes: &[usize]) -> f64 {
    let real: usize = blocks.iter().map(|b| b.real).sum();
    real as f64 / sizes.iter().sum::<usize>() as f64
}

fn eval(blocks: &[Blk], zstd: bool) {
    for (name, p) in [("opt14", OPT14), ("opt16", OPT16), ("opt16p1", OPT16P1)] {
        let t = std::time::Instant::now();
        let sizes = par_map(blocks, |b| {
            let out = compress_block(&b.data, p);
            let fr = write_frame(&b.data, &out, FrameOptions::default());
            let dec = zstd::bulk::decompress(&fr, BLOCK_SIZE).expect("libzstd rejected an opt frame");
            assert!(dec == b.data, "{name}: libzstd round trip mismatch");
            fr.len()
        });
        println!("{name:<8} ratio {:.5}  ({:.1} s)", ratio(blocks, &sizes), t.elapsed().as_secs_f64());
    }
    if zstd {
        for level in [14, 16] {
            let sizes = par_map(blocks, |b| zstd::bulk::compress(&b.data, level).unwrap().len());
            println!("zstd L{level} ratio {:.5}", ratio(blocks, &sizes));
        }
    }
}

/// The parse the `OPT_PRIOR_S3_*` tables are trained on (M6 B0): `OPT16P1`'s candidates (h4 8
/// deep, h3, the S3 sparse chains) and gap3 segment ends, with `OPT16`'s schedule (`BlockInit`
/// seed, 3 cheap passes, the optLevel-2 final pass), no relaxation pruning and no drop pass.
fn s3_train_params() -> MatchParams {
    let o = OptParams { passes: 3, seed: Seed::BlockInit, prior: PriorTables::M5, relax_lengths: None, drop_max_len: 0, ..OPT16P1.opt.unwrap() };
    MatchParams { opt: Some(o), ..OPT16P1 }
}

fn train(blocks: &[Blk], p: MatchParams, prefix: &str) {
    let hs = par_map(blocks, |b| {
        let cands = find_cands(&b.data, &chains(&b.data, &p), &p);
        Hist::of_output(&gzc_core::opt::parse(&b.data, &cands, &p))
    });
    let mut t = (vec![0u64; 36], vec![0u64; 53], vec![0u64; 32]);
    for h in &hs {
        t.0.iter_mut().zip(h.ll).for_each(|(a, x)| *a += x as u64);
        t.1.iter_mut().zip(h.ml).for_each(|(a, x)| *a += x as u64);
        t.2.iter_mut().zip(h.of).for_each(|(a, x)| *a += x as u64);
    }
    let scale = |v: &[u64]| -> Vec<u32> {
        let s: u64 = v.iter().sum::<u64>().max(1);
        v.iter().map(|&x| ((x * 65536 + s / 2) / s) as u32).collect()
    };
    println!("pub const {prefix}_LL: [u32; 36] = {:?};", scale(&t.0));
    println!("pub const {prefix}_ML: [u32; 53] = {:?};", scale(&t.1));
    println!("pub const {prefix}_OF: [u32; 32] = {:?};", scale(&t.2));
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let usage = "usage: opt_sample eval|train|train-s3 <corpus> <every> <offset> [zstd]";
    let (mode, dir) = (a.get(1).expect(usage), PathBuf::from(a.get(2).expect(usage)));
    let every: usize = a.get(3).expect(usage).parse().unwrap();
    let offset: usize = a.get(4).expect(usage).parse().unwrap();
    let blocks = load(&dir, every, offset);
    match mode.as_str() {
        "eval" => eval(&blocks, a.get(5).is_some_and(|s| s == "zstd")),
        "train" => train(&blocks, OPT16, "OPT_PRIOR"),
        "train-s3" => train(&blocks, s3_train_params(), "OPT_PRIOR_S3"),
        _ => panic!("{usage}"),
    }
}
