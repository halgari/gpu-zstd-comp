//! Corpus loading and file discovery per `corpus.toml`.
use std::path::{Path, PathBuf};

use gzc_core::block::{chunk_file, Block};
use gzc_core::synth;

/// Coarse content-type classification, driven by file extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Kind {
    Dds,
    Nif,
    Other,
}

impl Kind {
    fn from_ext(ext: &str) -> Kind {
        match ext.to_ascii_lowercase().as_str() {
            "dds" => Kind::Dds,
            "nif" => Kind::Nif,
            _ => Kind::Other,
        }
    }
}

/// A loaded corpus: every file's blocks, flattened, with a per-block `Kind`.
pub struct Corpus {
    pub blocks: Vec<Block>,
    /// `kinds[i]` is the kind of `blocks[i]`.
    pub kinds: Vec<Kind>,
    pub files: usize,
}

pub struct LoadOpts {
    pub input: PathBuf,
    /// Only these extensions (case-insensitive, no leading dot) are loaded. `None` loads everything.
    pub exts: Option<Vec<String>>,
    /// Stop adding files once the running total of loaded bytes reaches this cap.
    pub max_bytes: Option<u64>,
}

impl Corpus {
    pub fn real_bytes(&self) -> u64 {
        self.blocks.iter().map(|b| b.real_len as u64).sum()
    }

    /// A data-free corpus built from `gzc_core::synth::test_cases`, all blocks kind `Other`.
    pub fn synthetic() -> Corpus {
        let mut blocks = Vec::new();
        let mut kinds = Vec::new();
        let mut files = 0usize;
        for (_name, data) in synth::test_cases() {
            for b in chunk_file(&data) {
                kinds.push(Kind::Other);
                blocks.push(b);
            }
            files += 1;
        }
        Corpus { blocks, kinds, files }
    }
}

/// Recursively collects file paths under `dir`, depth-first, each directory's
/// entries visited in sorted order, so the overall traversal is deterministic.
fn collect_paths(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_paths(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

/// Loads a corpus from `opts.input`, recursing into subdirectories, visiting
/// paths in sorted order for a deterministic block sequence.
pub fn load(opts: &LoadOpts) -> anyhow::Result<Corpus> {
    let mut paths = Vec::new();
    collect_paths(&opts.input, &mut paths)?;

    let mut blocks = Vec::new();
    let mut kinds = Vec::new();
    let mut files = 0usize;
    let mut total: u64 = 0;

    for path in paths {
        if let Some(exts) = &opts.exts {
            let matches = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| exts.iter().any(|x| x.eq_ignore_ascii_case(e)))
                .unwrap_or(false);
            if !matches {
                continue;
            }
        }

        if let Some(max) = opts.max_bytes
            && total >= max
        {
            break;
        }

        let kind = path
            .extension()
            .and_then(|e| e.to_str())
            .map(Kind::from_ext)
            .unwrap_or(Kind::Other);

        let bytes = std::fs::read(&path)?;
        total += bytes.len() as u64;
        for b in chunk_file(&bytes) {
            kinds.push(kind);
            blocks.push(b);
        }
        files += 1;
    }

    Ok(Corpus { blocks, kinds, files })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gzc_core::config::BLOCK_SIZE;

    fn temp_subdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gzc-bench-corpus-test-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_file(dir: &Path, name: &str, len: usize) {
        std::fs::write(dir.join(name), vec![0x55u8; len]).unwrap();
    }

    #[test]
    fn load_discovers_files_sorted_with_kinds() {
        let dir = temp_subdir("basic");
        write_file(&dir, "a.dds", BLOCK_SIZE + 5);
        write_file(&dir, "b.nif", 10);
        write_file(&dir, "c.txt", 7);

        let corpus = load(&LoadOpts { input: dir.clone(), exts: None, max_bytes: None }).unwrap();

        assert_eq!(corpus.blocks.len(), 4);
        assert_eq!(corpus.kinds, vec![Kind::Dds, Kind::Dds, Kind::Nif, Kind::Other]);
        assert_eq!(corpus.real_bytes(), (BLOCK_SIZE + 5 + 10 + 7) as u64);
        assert_eq!(corpus.files, 3);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_filters_by_extension() {
        let dir = temp_subdir("ext");
        write_file(&dir, "a.dds", BLOCK_SIZE + 5);
        write_file(&dir, "b.nif", 10);
        write_file(&dir, "c.txt", 7);

        let corpus = load(&LoadOpts {
            input: dir.clone(),
            exts: Some(vec!["dds".to_string()]),
            max_bytes: None,
        })
        .unwrap();

        assert_eq!(corpus.blocks.len(), 2);
        assert_eq!(corpus.files, 1);
        assert!(corpus.kinds.iter().all(|k| *k == Kind::Dds));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_stops_once_running_total_reaches_max_bytes() {
        let dir = temp_subdir("max");
        write_file(&dir, "a.dds", 10);
        write_file(&dir, "b.nif", 10);
        write_file(&dir, "c.txt", 10);

        // After a.dds (10 bytes) the running total (10) already reaches the
        // cap, so b.nif and c.txt are never added.
        let corpus =
            load(&LoadOpts { input: dir.clone(), exts: None, max_bytes: Some(10) }).unwrap();

        assert_eq!(corpus.files, 1);
        assert_eq!(corpus.real_bytes(), 10);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn synthetic_corpus_is_all_other_kind() {
        let corpus = Corpus::synthetic();
        assert!(!corpus.blocks.is_empty());
        assert!(corpus.kinds.iter().all(|k| *k == Kind::Other));
        assert_eq!(corpus.files, synth::test_cases().len());
        assert_eq!(corpus.blocks.len(), corpus.kinds.len());
    }
}
