//! Shared logic for the `fetch-corpus` dev tool: manifest parsing, on-disk
//! layout helpers and "already done" skip logic. Kept in a lib target so it
//! is unit-testable without touching the network.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

/// `corpus.toml`: a list of `[[mod]]` entries pinning exact files to fetch.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    pub r#mod: Vec<ModEntry>,
}

/// One pinned Nexus Mods file.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ModEntry {
    /// Local corpus subdirectory name, e.g. `data/corpus/<name>/…`.
    pub name: String,
    /// Nexus game domain, e.g. `skyrimspecialedition`.
    pub game: String,
    pub mod_id: u32,
    pub file_id: u32,
}

/// Parse a `corpus.toml` manifest from its text contents.
pub fn parse_manifest(text: &str) -> Result<Manifest> {
    toml::from_str(text).context("failed to parse manifest toml")
}

/// Load and parse a manifest file from disk.
pub fn load_manifest(path: &Path) -> Result<Manifest> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read manifest: {}", path.display()))?;
    parse_manifest(&text)
}

/// Where a downloaded archive for `file_name` lives under `data_dir`.
pub fn archive_path(data_dir: &Path, file_name: &str) -> PathBuf {
    data_dir.join("archives").join(file_name)
}

/// The `.part` path used while an archive is still downloading.
pub fn archive_part_path(data_dir: &Path, file_name: &str) -> PathBuf {
    let mut path = archive_path(data_dir, file_name).into_os_string();
    path.push(".part");
    PathBuf::from(path)
}

/// Whether the final archive file already exists (download can be skipped).
pub fn archive_present(data_dir: &Path, file_name: &str) -> bool {
    archive_path(data_dir, file_name).is_file()
}

/// The extracted-corpus directory for a mod entry named `name`.
pub fn corpus_dir(data_dir: &Path, name: &str) -> PathBuf {
    data_dir.join("corpus").join(name)
}

/// The `.done` marker path for a mod entry named `name`.
pub fn done_marker_path(data_dir: &Path, name: &str) -> PathBuf {
    corpus_dir(data_dir, name).join(".done")
}

/// Whether extraction for `name` has already completed (extraction can be skipped).
pub fn is_done(data_dir: &Path, name: &str) -> bool {
    done_marker_path(data_dir, name).is_file()
}

/// Write the `.done` marker for `name`, creating the corpus dir if needed.
pub fn mark_done(data_dir: &Path, name: &str) -> Result<()> {
    let dir = corpus_dir(data_dir, name);
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create corpus dir: {}", dir.display()))?;
    fs::write(done_marker_path(data_dir, name), b"")
        .context("failed to write .done marker")?;
    Ok(())
}

/// Derive a file name from a Nexus CDN download URI (last path segment,
/// percent-decoded, query string stripped).
pub fn filename_from_uri(uri: &str) -> Result<String> {
    let without_query = uri.split(['?', '#']).next().unwrap_or(uri);
    let last_segment = without_query
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .with_context(|| format!("could not derive a file name from URI: {uri}"))?;
    Ok(percent_decode(last_segment))
}

/// Minimal percent-decoding (no external dependency): decodes `%XX` triples,
/// passes everything else through unchanged.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3])
            && let Ok(byte) = u8::from_str_radix(hex, 16)
        {
            out.push(byte);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "fetch-corpus-test-{tag}-{}-{}",
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parses_manifest_with_multiple_mods() {
        let text = r#"
            [[mod]]
            name = "smim"
            game = "skyrimspecialedition"
            mod_id = 659
            file_id = 111

            [[mod]]
            name = "noble-skyrim"
            game = "skyrimspecialedition"
            mod_id = 21423
            file_id = 222
        "#;

        let manifest = parse_manifest(text).expect("manifest should parse");

        assert_eq!(manifest.r#mod.len(), 2);
        assert_eq!(
            manifest.r#mod[0],
            ModEntry {
                name: "smim".to_string(),
                game: "skyrimspecialedition".to_string(),
                mod_id: 659,
                file_id: 111,
            }
        );
        assert_eq!(manifest.r#mod[1].name, "noble-skyrim");
        assert_eq!(manifest.r#mod[1].mod_id, 21423);
        assert_eq!(manifest.r#mod[1].file_id, 222);
    }

    #[test]
    fn rejects_malformed_manifest() {
        let text = "not = [valid";
        assert!(parse_manifest(text).is_err());
    }

    #[test]
    fn filename_from_uri_strips_query_and_decodes() {
        let uri = "https://cdn.example.com/path/SMIM%20SE%202.08.7z?md5=abc&expires=123";
        assert_eq!(filename_from_uri(uri).unwrap(), "SMIM SE 2.08.7z");
    }

    #[test]
    fn filename_from_uri_no_query() {
        let uri = "https://cdn.example.com/path/plain-file.zip";
        assert_eq!(filename_from_uri(uri).unwrap(), "plain-file.zip");
    }

    #[test]
    fn archive_present_false_when_missing() {
        let data_dir = unique_temp_dir("archive-missing");
        assert!(!archive_present(&data_dir, "some-file.7z"));
    }

    #[test]
    fn archive_present_true_when_file_exists() {
        let data_dir = unique_temp_dir("archive-present");
        let archives = data_dir.join("archives");
        fs::create_dir_all(&archives).unwrap();
        fs::write(archives.join("some-file.7z"), b"data").unwrap();

        assert!(archive_present(&data_dir, "some-file.7z"));
        assert!(!archive_present(&data_dir, "other-file.7z"));
    }

    #[test]
    fn is_done_false_before_marker_written() {
        let data_dir = unique_temp_dir("done-missing");
        assert!(!is_done(&data_dir, "smim"));
    }

    #[test]
    fn is_done_true_after_mark_done() {
        let data_dir = unique_temp_dir("done-present");
        assert!(!is_done(&data_dir, "smim"));

        mark_done(&data_dir, "smim").unwrap();

        assert!(is_done(&data_dir, "smim"));
        assert!(corpus_dir(&data_dir, "smim").is_dir());
    }

    #[test]
    fn is_done_is_per_mod_name() {
        let data_dir = unique_temp_dir("done-per-name");
        mark_done(&data_dir, "smim").unwrap();

        assert!(is_done(&data_dir, "smim"));
        assert!(!is_done(&data_dir, "noble-skyrim"));
    }
}
