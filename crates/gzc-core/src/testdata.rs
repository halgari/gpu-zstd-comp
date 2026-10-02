//! Corpus access for the ignored corpus tests, probes and examples (not part of the API).
//!
//! The corpus is the `.dds` / `.nif` files under `GZC_CORPUS` (default `data/corpus` at the
//! repository root), in sorted path order: the order `gzc-bench --ext dds,nif` loads them in.
use std::path::{Path, PathBuf};

use crate::block::chunk_file;
use crate::config::BLOCK_SIZE;

/// The corpus directory: `GZC_CORPUS`, else `data/corpus` at the repository root. `None` (after
/// a message on stderr) when it does not exist, so a corpus test can skip.
pub fn corpus_dir() -> Option<PathBuf> {
    let root = std::env::var("GZC_CORPUS")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/corpus").to_string());
    let root = PathBuf::from(root);
    if root.is_dir() {
        Some(root)
    } else {
        eprintln!("corpus directory {} not found (set GZC_CORPUS): skipping", root.display());
        None
    }
}

/// Every `.dds` / `.nif` file (extension case-insensitive) under `dir`, recursively, in sorted
/// path order.
pub fn corpus_files(dir: &Path) -> Vec<PathBuf> {
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
    let mut files = Vec::new();
    walk(dir, &mut files);
    files.sort();
    files
}

/// Up to `n` corpus blocks sampled uniformly over the whole corpus, each named
/// `path[block index]`: every k-th of all (file, block) pairs in `corpus_files` order,
/// k = total blocks / n. Blocks are `chunk_file` blocks (a file's last block zero-padded to
/// BLOCK_SIZE). `None` when the corpus is absent (`corpus_dir`).
pub fn corpus_sample_named(n: usize) -> Option<Vec<(String, Vec<u8>)>> {
    let files = corpus_files(&corpus_dir()?);
    let counts: Vec<usize> =
        files.iter().map(|f| (std::fs::metadata(f).unwrap().len() as usize).div_ceil(BLOCK_SIZE)).collect();
    let total: usize = counts.iter().sum();
    let step = (total / n.max(1)).max(1);
    let mut blocks = Vec::new();
    let mut first = 0usize; // global index of the file's first block
    for (f, &c) in files.iter().zip(&counts) {
        if blocks.len() >= n {
            break;
        }
        // The file's blocks whose global index first + i is a multiple of step.
        let picked: Vec<usize> = (first.div_ceil(step) * step..first + c).step_by(step).map(|g| g - first).collect();
        if !picked.is_empty() {
            let chunks = chunk_file(&std::fs::read(f).unwrap());
            assert_eq!(chunks.len(), c, "{}", f.display());
            for i in picked {
                blocks.push((format!("{}[{i}]", f.display()), chunks[i].data.clone()));
            }
        }
        first += c;
    }
    blocks.truncate(n);
    eprintln!("{} corpus blocks (every {step}th of {total}) from {} files", blocks.len(), files.len());
    Some(blocks)
}

/// `corpus_sample_named` without the names.
pub fn corpus_sample(n: usize) -> Option<Vec<Vec<u8>>> {
    Some(corpus_sample_named(n)?.into_iter().map(|(_, b)| b).collect())
}
