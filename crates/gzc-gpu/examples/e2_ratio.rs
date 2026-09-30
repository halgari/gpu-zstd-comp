//! speed2 E2: CPU-oracle corpus ratio for lvl9 variants with other key widths / depths.
//!
//! `cargo run --release -p gzc-gpu --example e2_ratio -- <corpus dir> <hash_bits:depth>...`
//! Compresses every block of the corpus's .dds/.nif files with `reference::compress_block_to_frame`
//! (lvl9 with the given `hash_bits` and `depth`) and prints real bytes / frame bytes, like
//! `gzc-bench ref`. Files are streamed (per-file parallelism) instead of loaded at once.
use gzc_core::block::chunk_file;
use gzc_core::frame::FrameOptions;
use gzc_core::params::{LVL9, MatchParams};
use gzc_core::reference::compress_block_to_frame;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk(&p, out);
        } else if matches!(p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(), Some("dds" | "nif")) {
            out.push(p);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut files = Vec::new();
    walk(Path::new(&args[1]), &mut files);
    let variants: Vec<MatchParams> = args[2..]
        .iter()
        .map(|v| {
            let (h, d) = v.split_once(':').expect("hash_bits:depth");
            MatchParams { hash_bits: h.parse().unwrap(), depth: d.parse().unwrap(), ..LVL9 }
        })
        .collect();
    let real = AtomicU64::new(0);
    let comp: Vec<AtomicU64> = variants.iter().map(|_| AtomicU64::new(0)).collect();
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(f) = files.get(i) else { break };
                    let data = std::fs::read(f).unwrap();
                    real.fetch_add(data.len() as u64, Ordering::Relaxed);
                    for blk in chunk_file(&data) {
                        for (v, c) in variants.iter().zip(&comp) {
                            let frame = compress_block_to_frame(&blk.data, *v, FrameOptions::default());
                            c.fetch_add(frame.len() as u64, Ordering::Relaxed);
                        }
                    }
                }
            });
        }
    });
    let real = real.load(Ordering::Relaxed);
    for (v, c) in variants.iter().zip(&comp) {
        let c = c.load(Ordering::Relaxed);
        println!("hash_bits {:2} depth {:2}: {real} / {c} = {:.5}", v.hash_bits, v.depth, real as f64 / c as f64);
    }
}
