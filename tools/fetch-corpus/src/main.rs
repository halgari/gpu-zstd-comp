//! `fetch-corpus`: dev-only tool that downloads and unpacks a pinned Nexus
//! Mods benchmark corpus (see `corpus.toml`) into `data/corpus/<name>/…`.
//!
//! Not part of the compression pipeline itself — nothing in the workspace
//! depends on this crate.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use fetch_corpus::{
    archive_part_path, archive_path, archive_present, done_marker_path, is_done, load_manifest,
    mark_done, ModEntry,
};

const NEXUS_API_BASE: &str = "https://api.nexusmods.com/v1";

#[derive(Parser)]
#[command(name = "fetch-corpus", about = "Download/build the gpu-zstd-comp benchmark corpus")]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
enum CliCommand {
    /// List files for a Nexus mod (used to pick which file_id to pin).
    List {
        /// Nexus game domain, e.g. skyrimspecialedition
        game: String,
        /// Nexus mod id
        mod_id: u32,
    },
    /// Fetch and unpack every mod pinned in the manifest.
    Fetch {
        /// Path to the manifest (defaults to ./corpus.toml)
        #[arg(long, default_value = "corpus.toml")]
        manifest: PathBuf,
        /// Root data directory (archives/ and corpus/ live under here)
        #[arg(long, default_value = "data")]
        data: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CliCommand::List { game, mod_id } => cmd_list(&game, mod_id),
        CliCommand::Fetch { manifest, data } => cmd_fetch(&manifest, &data),
    }
}

fn require_api_key() -> Result<String> {
    std::env::var("NEXUS_API_KEY")
        .context("NEXUS_API_KEY is not set in the environment; get one from https://next.nexusmods.com/settings/api-keys and export it before running fetch-corpus")
}

// --- Nexus API response shapes (only the fields we use) ---

#[derive(Debug, Deserialize)]
struct FilesResponse {
    files: Vec<NexusFile>,
}

#[derive(Debug, Deserialize)]
struct NexusFile {
    file_id: u64,
    category_name: Option<String>,
    size_kb: Option<f64>,
    file_name: String,
}

#[derive(Debug, Deserialize)]
struct DownloadLinkEntry {
    #[serde(rename = "URI")]
    uri: String,
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    short_name: String,
}

fn cmd_list(game: &str, mod_id: u32) -> Result<()> {
    let api_key = require_api_key()?;
    let client = reqwest::blocking::Client::new();
    let url = format!("{NEXUS_API_BASE}/games/{game}/mods/{mod_id}/files.json");

    let resp = client
        .get(&url)
        .header("apikey", &api_key)
        .send()
        .with_context(|| format!("request failed: {url}"))?;

    if !resp.status().is_success() {
        bail!("Nexus API returned {} for {url}", resp.status());
    }

    let body: FilesResponse = resp
        .json()
        .context("failed to parse files.json response")?;

    println!("file_id, category_name, size_kb, file_name");
    for f in &body.files {
        println!(
            "{}, {}, {}, {}",
            f.file_id,
            f.category_name.as_deref().unwrap_or(""),
            f.size_kb.map(|v| v.to_string()).unwrap_or_default(),
            f.file_name
        );
    }

    Ok(())
}

/// Look up a single file's metadata (in particular its real file_name) by
/// id. This does not require premium and does not count as a download.
fn fetch_file_name(
    client: &reqwest::blocking::Client,
    api_key: &str,
    game: &str,
    mod_id: u32,
    file_id: u32,
) -> Result<String> {
    let url = format!("{NEXUS_API_BASE}/games/{game}/mods/{mod_id}/files/{file_id}.json");
    let resp = client
        .get(&url)
        .header("apikey", api_key)
        .send()
        .with_context(|| format!("request failed: {url}"))?;

    if !resp.status().is_success() {
        bail!("Nexus API returned {} for {url}", resp.status());
    }

    let file: NexusFile = resp
        .json()
        .with_context(|| format!("failed to parse file metadata response from {url}"))?;
    Ok(file.file_name)
}

fn fetch_download_uri(
    client: &reqwest::blocking::Client,
    api_key: &str,
    game: &str,
    mod_id: u32,
    file_id: u32,
) -> Result<String> {
    let url =
        format!("{NEXUS_API_BASE}/games/{game}/mods/{mod_id}/files/{file_id}/download_link.json");
    let resp = client
        .get(&url)
        .header("apikey", api_key)
        .send()
        .with_context(|| format!("request failed: {url}"))?;

    if !resp.status().is_success() {
        bail!(
            "Nexus API returned {} for {url} (premium membership required for download_link)",
            resp.status()
        );
    }

    let links: Vec<DownloadLinkEntry> = resp
        .json()
        .with_context(|| format!("failed to parse download_link.json response from {url}"))?;
    let first = links
        .into_iter()
        .next()
        .with_context(|| format!("no download links returned for file_id {file_id}"))?;
    Ok(first.uri)
}

fn download_archive(
    client: &reqwest::blocking::Client,
    uri: &str,
    data_dir: &Path,
    file_name: &str,
) -> Result<()> {
    let dest = archive_path(data_dir, file_name);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create dir: {}", parent.display()))?;
    }
    let part_path = archive_part_path(data_dir, file_name);

    println!("  downloading -> {}", dest.display());
    let mut resp = client
        .get(uri)
        .send()
        .with_context(|| "download request failed".to_string())?;
    if !resp.status().is_success() {
        bail!("download returned status {}", resp.status());
    }

    let mut file = fs::File::create(&part_path)
        .with_context(|| format!("failed to create {}", part_path.display()))?;
    let bytes = resp
        .copy_to(&mut file)
        .context("failed while streaming download to disk")?;
    file.flush().ok();
    drop(file);

    fs::rename(&part_path, &dest)
        .with_context(|| format!("failed to rename {} -> {}", part_path.display(), dest.display()))?;

    println!("  downloaded {bytes} bytes");
    Ok(())
}

fn run_7z_extract(archive: &Path, out_dir: &Path) -> Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create dir: {}", out_dir.display()))?;

    let out_arg = format!("-o{}", out_dir.display());
    let status = Command::new("7z")
        .arg("x")
        .arg("-y")
        .arg(&out_arg)
        .arg(archive)
        .status()
        .with_context(|| format!("failed to spawn 7z for {}", archive.display()))?;

    if !status.success() {
        bail!("7z extraction of {} failed: {status}", archive.display());
    }
    Ok(())
}

/// Recursively find all `*.bsa` files under `dir` (case-insensitive extension).
fn find_bsas(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read dir {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            find_bsas(&path, out)?;
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("bsa"))
        {
            out.push(path);
        }
    }
    Ok(())
}

/// Extract every file in a TES4/SSE BSA archive into `dest_root`, using the
/// `ba2` crate, decompressing any compressed entries, then delete the BSA.
fn extract_bsa(bsa_path: &Path, dest_root: &Path) -> Result<()> {
    use ba2::prelude::*;
    use ba2::tes4::{Archive, FileCompressionOptions};
    use ba2::ByteSlice;

    let (archive, options) =
        Archive::read(bsa_path).with_context(|| format!("failed to read BSA: {}", bsa_path.display()))?;
    let compression_options = FileCompressionOptions::builder()
        .version(options.version())
        .build();

    for (dir_key, directory) in &archive {
        let dir_name = bstr_to_relative_path(dir_key.name().as_bytes());
        for (file_key, file) in directory {
            let file_name = bstr_to_relative_path(file_key.name().as_bytes());
            let out_path = dest_root.join(&dir_name).join(&file_name);
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create dir: {}", parent.display()))?;
            }
            let mut out_file = fs::File::create(&out_path)
                .with_context(|| format!("failed to create {}", out_path.display()))?;
            file.write(&mut out_file, &compression_options)
                .with_context(|| format!("failed to extract {} from {}", out_path.display(), bsa_path.display()))?;
        }
    }

    fs::remove_file(bsa_path)
        .with_context(|| format!("failed to remove {} after extraction", bsa_path.display()))?;
    Ok(())
}

/// Convert a BSA-internal path (backslash separated, possibly empty) into a
/// relative filesystem path segment.
fn bstr_to_relative_path(name: &[u8]) -> PathBuf {
    let s = String::from_utf8_lossy(name);
    let normalized = s.replace('\\', "/");
    PathBuf::from(normalized)
}

fn cmd_fetch(manifest_path: &Path, data_dir: &Path) -> Result<()> {
    let api_key = require_api_key()?;
    let manifest = load_manifest(manifest_path)
        .with_context(|| format!("failed to load manifest: {}", manifest_path.display()))?;
    let client = reqwest::blocking::Client::new();

    for entry in &manifest.r#mod {
        println!("== {} (mod_id={}, file_id={}) ==", entry.name, entry.mod_id, entry.file_id);
        fetch_one(&client, &api_key, data_dir, entry)?;
    }

    Ok(())
}

fn fetch_one(
    client: &reqwest::blocking::Client,
    api_key: &str,
    data_dir: &Path,
    entry: &ModEntry,
) -> Result<()> {
    let file_name = fetch_file_name(client, api_key, &entry.game, entry.mod_id, entry.file_id)?;
    let archive = archive_path(data_dir, &file_name);

    if archive_present(data_dir, &file_name) {
        println!("  archive already present: {}", archive.display());
    } else {
        let uri = fetch_download_uri(client, api_key, &entry.game, entry.mod_id, entry.file_id)?;
        download_archive(client, &uri, data_dir, &file_name)?;
    }

    if is_done(data_dir, &entry.name) {
        println!("  already extracted (found {})", done_marker_path(data_dir, &entry.name).display());
        return Ok(());
    }

    let out_dir = data_dir.join("corpus").join(&entry.name);
    println!("  extracting archive -> {}", out_dir.display());
    run_7z_extract(&archive, &out_dir)?;

    let mut bsas = Vec::new();
    find_bsas(&out_dir, &mut bsas)?;
    for bsa in &bsas {
        println!("  unpacking BSA: {}", bsa.display());
        extract_bsa(bsa, &out_dir)?;
    }

    mark_done(data_dir, &entry.name)?;
    println!("  done: {}", out_dir.display());
    Ok(())
}
