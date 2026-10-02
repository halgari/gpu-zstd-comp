//! Benchmark CLI: compares CPU baseline, CPU reference and GPU compressors over a corpus.
mod corpus;
mod cpu;
mod result;
mod report;
mod refrun;
mod gpurun;

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, Parser, Subcommand};
use gzc_core::params::{MatchParams, PRESETS};
use gzc_gpu::{GpuParams, gpu_supports, max_batch_blocks};
use gzc_gpu::{GpuContext, GpuOptions};
use gzc_gpu::pipeline::{PipelineConfig, vram_bytes_with};

use corpus::{Corpus, LoadOpts};

/// Default libzstd levels benchmarked by `cpu` and `all`.
const DEFAULT_LEVELS: &str = "1,2,3,4,5,6";
/// Highest libzstd level the benchmark accepts: levels above this are out of scope.
const MAX_LEVEL: i64 = 16;

fn level_parser() -> clap::builder::RangedI64ValueParser<i32> {
    clap::value_parser!(i32).range(..=MAX_LEVEL)
}

/// Default `--preset` list for `ref`, `gpu` and `all`.
const DEFAULT_PRESETS: &str = "lvl3";

/// A named `gzc_core::params` preset, as selected by `--preset`.
#[derive(Clone, Copy, Debug)]
struct Preset {
    name: &'static str,
    params: MatchParams,
}

/// `--preset` value parser: one of `gzc_core::params::PRESETS` by name.
fn parse_preset(name: &str) -> Result<Preset, String> {
    let params = gzc_core::params::preset(name)?;
    let name = PRESETS.iter().find(|(n, _)| *n == name).map(|(n, _)| *n).expect("preset() found it");
    Ok(Preset { name, params })
}

/// Errors on the first preset the selected engines (`cpu`: the reference compressor, `gpu`: the
/// GPU kernels) do not implement yet. Run before any work starts.
fn check_presets(presets: &[Preset], cpu: bool, gpu: bool) -> anyhow::Result<()> {
    for p in presets {
        anyhow::ensure!(!cpu || p.params.validate().is_ok(), "preset '{}' is not implemented yet on cpu", p.name);
        anyhow::ensure!(!gpu || gpu_supports(&p.params), "preset '{}' is not implemented yet on gpu", p.name);
    }
    Ok(())
}

#[derive(Parser)]
#[command(name = "gzc-bench", about = "Benchmark CPU/GPU zstd compression over a corpus")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// CPU libzstd baseline.
    Cpu(CpuArgs),
    /// CPU reference compressor (the algorithm the GPU mirrors).
    Ref(RefArgs),
    /// Streaming GPU compressor (match finding, parse and complete zstd frames, Huffman
    /// literals included, on the GPU).
    Gpu(GpuArgs),
    /// Every engine (cpu-libzstd, cpu-ref, gpu) into one report.
    All(AllArgs),
}

#[derive(Args)]
struct CorpusArgs {
    /// Directory to load the corpus from (recursive).
    #[arg(long)]
    input: Option<PathBuf>,
    /// Use the built-in synthetic corpus instead of `--input`.
    #[arg(long)]
    synthetic: bool,
    /// Comma-separated list of file extensions to include (default: all).
    #[arg(long, value_delimiter = ',')]
    ext: Option<Vec<String>>,
    /// Stop loading files once this many bytes have been read.
    #[arg(long)]
    max_bytes: Option<u64>,
}

#[derive(Args)]
struct CpuArgs {
    #[command(flatten)]
    corpus: CorpusArgs,
    /// Comma-separated zstd compression levels (at most 16).
    #[arg(long, value_delimiter = ',', default_value = DEFAULT_LEVELS, value_parser = level_parser())]
    levels: Vec<i32>,
    /// Comma-separated thread counts.
    #[arg(long, value_delimiter = ',', default_value = "1,8,16,32")]
    threads: Vec<usize>,
    /// Output directory for the JSON/HTML reports.
    #[arg(long, default_value = "out")]
    out: PathBuf,
}

#[derive(Args)]
struct RefArgs {
    #[command(flatten)]
    corpus: CorpusArgs,
    /// Comma-separated match presets (lvl3, lvl9seg, lvl9s12seg, opt14, opt16, opt16p1).
    #[arg(long, value_delimiter = ',', default_value = DEFAULT_PRESETS, value_parser = parse_preset)]
    preset: Vec<Preset>,
    /// Comma-separated thread counts.
    #[arg(long, value_delimiter = ',', default_value = "1,8")]
    threads: Vec<usize>,
    /// Decompress every produced frame with libzstd after the timed pass and
    /// error on any mismatch against the original block.
    #[arg(long)]
    verify: bool,
    /// Output directory for the JSON/HTML reports.
    #[arg(long, default_value = "out")]
    out: PathBuf,
}

/// A `--batch` list entry: an exact block count, or `max` (the largest batch that fits
/// `--vram-budget-mb` at a given preset and `--inflight`, capped by the device's own limit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchSpec {
    N(u32),
    Max,
}

/// `--batch` value parser: an unsigned integer, or `max` (case-insensitive).
fn parse_batch_spec(s: &str) -> Result<BatchSpec, String> {
    if s.eq_ignore_ascii_case("max") {
        return Ok(BatchSpec::Max);
    }
    s.parse::<u32>().map(BatchSpec::N).map_err(|e| format!("'{s}': {e} (expected a number or 'max')"))
}

/// Resolves `BatchSpec::Max` to the largest batch that fits `budget_mb` at `inflight` for match
/// params `m`, capped by `device_max` (`gzc_gpu::sizing::max_batch_blocks`). `vram_bytes` is
/// non-decreasing in `batch` (every buffer it counts scales with `batch`, at fixed `inflight`), so
/// this binary searches rather than scanning every batch size. `direct_upload`: the context reads
/// batches from the upload buffers (`GpuContext::direct_upload`, no shared `data` buffer).
fn resolve_max_batch(
    m: MatchParams,
    inflight: u32,
    vram_budget_mb: u64,
    device_max: u32,
    direct_upload: bool,
) -> anyhow::Result<u32> {
    let fits =
        |batch: u32| vram_bytes_with(&sweep_cfg(m, batch, inflight), direct_upload).div_ceil(1 << 20) <= vram_budget_mb;
    anyhow::ensure!(
        device_max >= 1 && fits(1),
        "--batch max: even batch 1 at inflight {inflight} does not fit {vram_budget_mb} MiB (--vram-budget-mb) \
         or this device's limits"
    );
    if fits(device_max) {
        return Ok(device_max);
    }
    let (mut lo, mut hi) = (1u32, device_max); // fits(lo) held, fits(hi) does not.
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) { lo = mid } else { hi = mid }
    }
    Ok(lo)
}

/// `spec` as an exact batch count: `N` as is, `Max` resolved via `resolve_max_batch`.
fn resolve_batch(
    spec: BatchSpec,
    m: MatchParams,
    inflight: u32,
    vram_budget_mb: u64,
    device_max: u32,
    direct_upload: bool,
) -> anyhow::Result<u32> {
    match spec {
        BatchSpec::N(n) => Ok(n),
        BatchSpec::Max => resolve_max_batch(m, inflight, vram_budget_mb, device_max, direct_upload),
    }
}

#[derive(Args)]
struct GpuSweepArgs {
    /// Comma-separated blocks per GPU batch, or `max` (the largest batch fitting
    /// `--vram-budget-mb` for a given preset and `--inflight`, capped by the device's own limit;
    /// resolved per preset, so `lvl3` and the single-hash presets can resolve to different
    /// numbers). Mixed lists like `512,max` are allowed.
    #[arg(long, value_delimiter = ',', default_value = "512", value_parser = parse_batch_spec)]
    batch: Vec<BatchSpec>,
    /// Comma-separated number of batches in flight.
    #[arg(long, value_delimiter = ',', default_value = "3")]
    inflight: Vec<u32>,
    /// Comma-separated number of CPU frame-writer threads. The GPU emits finished frames, so a
    /// writer only copies (in a real tool: writes out) the bytes it is handed; 0 = the pipeline's
    /// completion thread does that itself, N > 0 = N threads share each completed batch.
    #[arg(long, value_delimiter = ',', default_value = "0")]
    writer_threads: Vec<usize>,
    /// GPU memory budget in MiB (default: an 8 GB card minus headroom). Every (batch, inflight)
    /// config's pipeline footprint (`gzc_gpu::pipeline::vram_bytes_with`, for the context's upload mode
    /// once the device is open) must fit, else the run errors.
    #[arg(long, default_value_t = 6144)]
    vram_budget_mb: u64,
}

#[derive(Args)]
struct GpuArgs {
    #[command(flatten)]
    corpus: CorpusArgs,
    /// Comma-separated match presets (lvl3, lvl9seg, lvl9s12seg, opt14, opt16, opt16p1).
    #[arg(long, value_delimiter = ',', default_value = DEFAULT_PRESETS, value_parser = parse_preset)]
    preset: Vec<Preset>,
    #[command(flatten)]
    sweep: GpuSweepArgs,
    /// Decompress every produced frame with libzstd after the timed pass and
    /// error on any mismatch against the original block.
    #[arg(long)]
    verify: bool,
    /// Output directory for the JSON/HTML reports.
    #[arg(long, default_value = "out")]
    out: PathBuf,
}

#[derive(Args)]
struct AllArgs {
    #[command(flatten)]
    corpus: CorpusArgs,
    /// Comma-separated match presets for cpu-ref and gpu (lvl3, lvl9seg, lvl9s12seg, opt14, opt16, opt16p1).
    #[arg(long, value_delimiter = ',', default_value = DEFAULT_PRESETS, value_parser = parse_preset)]
    preset: Vec<Preset>,
    /// Comma-separated zstd compression levels (cpu-libzstd only; at most 16).
    #[arg(long, value_delimiter = ',', default_value = DEFAULT_LEVELS, value_parser = level_parser())]
    levels: Vec<i32>,
    /// Comma-separated thread counts, used by both cpu-libzstd and cpu-ref.
    #[arg(long, value_delimiter = ',', default_value = "1,8,16,32")]
    threads: Vec<usize>,
    #[command(flatten)]
    gpu: GpuSweepArgs,
    /// Decompress every cpu-ref and gpu frame with libzstd after the timed pass and
    /// error on any mismatch against the original block.
    #[arg(long)]
    verify: bool,
    /// Output directory for the JSON/HTML reports.
    #[arg(long, default_value = "out")]
    out: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Cpu(args) => run_cpu_cmd(args),
        Command::Ref(args) => run_ref_cmd(args),
        Command::Gpu(args) => run_gpu_cmd(args),
        Command::All(args) => run_all_cmd(args),
    }
}

fn load_corpus(args: &CorpusArgs) -> anyhow::Result<Corpus> {
    if args.synthetic {
        Ok(Corpus::synthetic())
    } else {
        let input = args
            .input
            .clone()
            .ok_or_else(|| anyhow::anyhow!("either --input DIR or --synthetic is required"))?;
        corpus::load(&LoadOpts { input, exts: args.ext.clone(), max_bytes: args.max_bytes })
    }
}

fn log_corpus(corpus: &Corpus) {
    eprintln!(
        "corpus: {} files, {} blocks, {} real bytes",
        corpus.files,
        corpus.blocks.len(),
        corpus.real_bytes()
    );
}

fn write_reports(results: &[result::RunResult], out: &PathBuf) -> anyhow::Result<()> {
    report::print_table(results);
    std::fs::create_dir_all(out)?;
    let json_path = report::write_json(results, out)?;
    let html_path = report::write_html(results, out)?;
    println!("JSON report: {}", json_path.display());
    println!("HTML report: {}", html_path.display());
    Ok(())
}

fn run_cpu_cmd(args: CpuArgs) -> anyhow::Result<()> {
    let corpus = load_corpus(&args.corpus)?;
    log_corpus(&corpus);

    let mut results = Vec::new();
    for &threads in &args.threads {
        for &level in &args.levels {
            eprintln!("running cpu-libzstd L{level} @ {threads} threads...");
            results.push(cpu::run_cpu(&corpus, level, threads)?);
        }
    }

    write_reports(&results, &args.out)
}

/// Runs the CPU reference for every preset and thread count, appending to `results`.
fn run_ref_sweep(
    corpus: &Corpus,
    presets: &[Preset],
    threads: &[usize],
    verify: bool,
    results: &mut Vec<result::RunResult>,
) -> anyhow::Result<()> {
    for p in presets {
        for &threads in threads {
            eprintln!("running cpu-ref {} @ {threads} threads (verify={verify})...", p.name);
            results.push(refrun::run_ref(corpus, p.name, p.params, threads, verify)?);
        }
    }
    Ok(())
}

fn run_ref_cmd(args: RefArgs) -> anyhow::Result<()> {
    check_presets(&args.preset, true, false)?;
    let corpus = load_corpus(&args.corpus)?;
    log_corpus(&corpus);

    let mut results = Vec::new();
    run_ref_sweep(&corpus, &args.preset, &args.threads, args.verify, &mut results)?;

    write_reports(&results, &args.out)
}

/// GPU memory of the frame-path pipeline for `cfg` (with the direct upload's footprint when
/// `direct_upload`), checked against `budget_mb`.
fn check_vram(cfg: &PipelineConfig, budget_mb: u64, direct_upload: bool) -> anyhow::Result<u64> {
    let cfg = PipelineConfig { params: GpuParams { emit_frames: true, ..cfg.params }, ..*cfg };
    let mib = vram_bytes_with(&cfg, direct_upload).div_ceil(1 << 20);
    anyhow::ensure!(
        mib <= budget_mb,
        "gpu b{} i{} needs {mib} MiB of GPU memory, over the {budget_mb} MiB budget (--vram-budget-mb)",
        cfg.batch,
        cfg.inflight
    );
    Ok(mib)
}

fn sweep_cfg(matching: MatchParams, batch: u32, inflight: u32) -> PipelineConfig {
    PipelineConfig { batch, inflight, params: GpuParams { matching, emit_frames: true, huffman: true } }
}

/// Checks every preset is implemented on the GPU and validates every explicit (preset, batch,
/// inflight) config of `sweep` against the VRAM budget (an `N` entry needs no device, so this
/// much fails before opening one); then opens the GPU, resolves every `max` entry (per preset and
/// `--inflight`, since a batch fitting the budget depends on both) and checks every resolved
/// batch against the device's own limit and the VRAM budget again. Run before any timed work so a
/// bad config or a missing adapter fails fast.
fn gpu_preflight(presets: &[Preset], sweep: &GpuSweepArgs) -> anyhow::Result<Arc<GpuContext>> {
    check_presets(presets, false, true)?;
    anyhow::ensure!(sweep.inflight.iter().all(|&i| i >= 1), "--inflight must be at least 1");
    for p in presets {
        for &spec in &sweep.batch {
            if let BatchSpec::N(batch) = spec {
                for &inflight in &sweep.inflight {
                    check_vram(&sweep_cfg(p.params, batch, inflight), sweep.vram_budget_mb, false)?;
                }
            }
        }
    }
    let ctx = Arc::new(GpuContext::new(GpuOptions::from_env())?);
    eprintln!("{}", ctx.describe());
    for p in presets {
        let max = max_batch_blocks(&ctx.device().limits(), &p.params);
        for &spec in &sweep.batch {
            for &inflight in &sweep.inflight {
                let batch = resolve_batch(spec, p.params, inflight, sweep.vram_budget_mb, max, ctx.direct_upload())?;
                anyhow::ensure!(
                    batch >= 1 && batch <= max,
                    "--batch {batch} not in 1..={max} for preset '{}' on this device",
                    p.name
                );
                check_vram(&sweep_cfg(p.params, batch, inflight), sweep.vram_budget_mb, ctx.direct_upload())?;
            }
        }
    }
    Ok(ctx)
}

/// Runs every (preset, batch, inflight, writer_threads) combination on `ctx`, appending to
/// `results`. Configs must have passed `gpu_preflight`. `--batch max` is re-resolved here (cheap:
/// no GPU dispatch) rather than threaded through from `gpu_preflight`, so the two stay in sync by
/// construction.
fn run_gpu_sweep(
    ctx: &Arc<GpuContext>,
    corpus: &Corpus,
    presets: &[Preset],
    sweep: &GpuSweepArgs,
    verify: bool,
    results: &mut Vec<result::RunResult>,
) -> anyhow::Result<()> {
    for p in presets {
        let max = max_batch_blocks(&ctx.device().limits(), &p.params);
        for &spec in &sweep.batch {
            for &inflight in &sweep.inflight {
                let batch = resolve_batch(spec, p.params, inflight, sweep.vram_budget_mb, max, ctx.direct_upload())?;
                for &writers in &sweep.writer_threads {
                    let cfg = sweep_cfg(p.params, batch, inflight);
                    let mib = check_vram(&cfg, sweep.vram_budget_mb, ctx.direct_upload())?;
                    eprintln!(
                        "running gpu {} b{batch} i{inflight} ({mib} MiB GPU memory) @ {writers} writer threads (verify={verify})...",
                        p.name
                    );
                    results.push(gpurun::run_gpu(ctx, corpus, p.name, &cfg, writers, verify)?);
                }
            }
        }
    }
    Ok(())
}

fn run_gpu_cmd(args: GpuArgs) -> anyhow::Result<()> {
    let ctx = gpu_preflight(&args.preset, &args.sweep)?;
    let corpus = load_corpus(&args.corpus)?;
    log_corpus(&corpus);

    let mut results = Vec::new();
    run_gpu_sweep(&ctx, &corpus, &args.preset, &args.sweep, args.verify, &mut results)?;

    write_reports(&results, &args.out)
}

fn run_all_cmd(args: AllArgs) -> anyhow::Result<()> {
    // Fail on an unimplemented preset, a bad GPU config or a missing adapter before the (long)
    // CPU sweeps.
    check_presets(&args.preset, true, true)?;
    let ctx = gpu_preflight(&args.preset, &args.gpu)?;
    let corpus = load_corpus(&args.corpus)?;
    log_corpus(&corpus);

    // One Vec<RunResult> holds every engine's runs so `write_reports` produces
    // a single combined report.
    let mut results = Vec::new();

    for &threads in &args.threads {
        for &level in &args.levels {
            eprintln!("running cpu-libzstd L{level} @ {threads} threads...");
            results.push(cpu::run_cpu(&corpus, level, threads)?);
        }
    }

    run_ref_sweep(&corpus, &args.preset, &args.threads, args.verify, &mut results)?;

    // A GPU failure still leaves the CPU results (and any finished GPU runs) on disk.
    if let Err(gpu_err) = run_gpu_sweep(&ctx, &corpus, &args.preset, &args.gpu, args.verify, &mut results) {
        eprintln!("gpu sweep failed: {gpu_err:#}; writing the results gathered so far");
        write_reports(&results, &args.out)?;
        return Err(gpu_err);
    }

    write_reports(&results, &args.out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gzc_gpu::pipeline::vram_bytes;

    #[test]
    fn vram_budget_rejects_configs_that_do_not_fit() {
        let params = GpuParams { matching: gzc_core::params::LVL3, emit_frames: false, huffman: true };
        let cfg = PipelineConfig { batch: 64, inflight: 2, params };
        let mib = check_vram(&cfg, 1 << 20, false).unwrap();
        assert!(mib > 0);
        assert_eq!(check_vram(&cfg, mib, false).unwrap(), mib, "a config exactly at the budget fits");
        let err = check_vram(&cfg, mib - 1, false).unwrap_err().to_string();
        assert!(err.contains("--vram-budget-mb") && err.contains("b64 i2"), "{err}");
    }

    #[test]
    fn batch_spec_parses_numbers_and_max_case_insensitively() {
        assert_eq!(parse_batch_spec("512"), Ok(BatchSpec::N(512)));
        assert_eq!(parse_batch_spec("0"), Ok(BatchSpec::N(0)));
        assert_eq!(parse_batch_spec("max"), Ok(BatchSpec::Max));
        assert_eq!(parse_batch_spec("Max"), Ok(BatchSpec::Max));
        assert_eq!(parse_batch_spec("MAX"), Ok(BatchSpec::Max));
        let err = parse_batch_spec("bogus").unwrap_err();
        assert!(err.contains("bogus") && err.contains("max"), "{err}");
        assert!(parse_batch_spec("-1").is_err());
        assert!(parse_batch_spec("").is_err());
    }

    #[test]
    fn batch_flag_parses_mixed_number_and_max_lists() {
        for cmd in ["gpu", "all"] {
            let batches = |extra: &[&str]| -> Result<Vec<BatchSpec>, clap::Error> {
                let argv = [&["gzc-bench", cmd, "--synthetic"], extra].concat();
                Ok(match Cli::try_parse_from(argv)?.command {
                    Command::Gpu(a) => a.sweep.batch,
                    Command::All(a) => a.gpu.batch,
                    _ => unreachable!(),
                })
            };
            assert_eq!(batches(&[]).unwrap(), [BatchSpec::N(512)], "{cmd}: default");
            assert_eq!(
                batches(&["--batch", "256,max,1024"]).unwrap(),
                [BatchSpec::N(256), BatchSpec::Max, BatchSpec::N(1024)],
                "{cmd}"
            );
            assert_eq!(batches(&["--batch", "max"]).unwrap(), [BatchSpec::Max], "{cmd}");
            assert!(batches(&["--batch", "bogus"]).is_err(), "{cmd}");
        }
    }

    /// `resolve_max_batch` binary-searches `vram_bytes`; a linear scan over every batch from
    /// `device_max` down to 1 must land on the same answer (the largest batch that fits).
    #[test]
    fn resolve_max_batch_matches_linear_scan() {
        let m = gzc_core::params::LVL3;
        for (budget_mb, inflight, device_max) in [(64u64, 1u32, 64u32), (128, 2, 200), (6144, 3, 4000)] {
            let want = (1..=device_max)
                .rev()
                .find(|&b| vram_bytes(&sweep_cfg(m, b, inflight)).div_ceil(1 << 20) <= budget_mb);
            let got = resolve_max_batch(m, inflight, budget_mb, device_max, false).ok();
            assert_eq!(got, want, "budget_mb={budget_mb} inflight={inflight} device_max={device_max}");
        }
    }

    #[test]
    fn resolve_max_batch_returns_device_max_when_it_fits() {
        let m = gzc_core::params::LVL3;
        let mib = vram_bytes(&sweep_cfg(m, 8, 1)).div_ceil(1 << 20);
        assert_eq!(resolve_max_batch(m, 1, mib, 8, false).unwrap(), 8, "device_max itself fits: use it");
        assert_eq!(resolve_max_batch(m, 1, mib + 1_000_000, 8, false).unwrap(), 8, "a huge budget: still capped at device_max");
    }

    #[test]
    fn resolve_max_batch_errors_when_even_batch_1_does_not_fit() {
        let m = gzc_core::params::LVL3;
        let err = resolve_max_batch(m, 1, 0, 4096, false).unwrap_err().to_string();
        assert!(err.contains("--batch max") && err.contains("--vram-budget-mb"), "{err}");
    }

    #[test]
    fn resolve_batch_passes_through_n_and_resolves_max() {
        let m = gzc_core::params::LVL3;
        assert_eq!(resolve_batch(BatchSpec::N(77), m, 1, 1, 1, false).unwrap(), 77, "N is never validated by resolve_batch itself");
        let mib = vram_bytes(&sweep_cfg(m, 16, 2)).div_ceil(1 << 20);
        assert_eq!(resolve_batch(BatchSpec::Max, m, 2, mib, 16, false).unwrap(), 16);
    }

    /// Different presets can resolve `max` to different numbers at the same budget/inflight:
    /// LVL3's two hash chains cost more scratch memory per block than a single-hash preset's one.
    #[test]
    fn resolve_max_batch_differs_per_preset() {
        let (budget_mb, inflight, device_max) = (6144u64, 3u32, 100_000u32);
        let lvl3 = resolve_max_batch(gzc_core::params::LVL3, inflight, budget_mb, device_max, false).unwrap();
        let lvl9 = resolve_max_batch(gzc_core::params::LVL9SEG, inflight, budget_mb, device_max, false).unwrap();
        assert!(lvl3 < lvl9, "lvl3 {lvl3} should resolve smaller than lvl9seg {lvl9} at the same budget");
        // The direct upload has no shared `data` buffer: a larger batch fits.
        let direct = resolve_max_batch(gzc_core::params::LVL3, inflight, budget_mb, device_max, true).unwrap();
        assert!(direct > lvl3, "direct upload {direct} vs copy upload {lvl3}");
    }

    #[test]
    fn preset_flag_parses_list_and_rejects_unknown() {
        for cmd in ["ref", "gpu", "all"] {
            let presets = |extra: &[&str]| -> Result<Vec<Preset>, clap::Error> {
                let argv = [&["gzc-bench", cmd, "--synthetic"], extra].concat();
                Ok(match Cli::try_parse_from(argv)?.command {
                    Command::Ref(a) => a.preset,
                    Command::Gpu(a) => a.preset,
                    Command::All(a) => a.preset,
                    Command::Cpu(_) => unreachable!(),
                })
            };
            let names = |extra: &[&str]| presets(extra).unwrap().iter().map(|p| p.name).collect::<Vec<_>>();
            assert_eq!(names(&[]), ["lvl3"], "{cmd}: default");
            assert_eq!(names(&["--preset", "lvl3,lvl9seg"]), ["lvl3", "lvl9seg"], "{cmd}");
            assert_eq!(presets(&["--preset", "lvl9s12seg"]).unwrap()[0].params, gzc_core::params::LVL9S12SEG, "{cmd}");
            let err = presets(&["--preset", "lvl3,bogus"]).unwrap_err().to_string();
            assert!(err.contains("bogus") && err.contains("lvl3") && err.contains("lvl9seg"), "{cmd}: {err}");
            for removed in ["rung1", "rung2", "lvl9", "lvl9s12", "lvl9s12d16seg"] {
                assert!(presets(&["--preset", removed]).is_err(), "{cmd}: {removed} is no longer a preset");
            }
        }
        let lvl3 = parse_preset("lvl3").unwrap();
        assert!(check_presets(&[lvl3], true, true).is_ok());
        // An unsegmented lazy parse is CPU-only (no preset has one; the GPU refuses the params).
        let cpu_only = Preset { name: "lazy2-unsegmented", params: gzc_core::fixtures::LVL9 };
        assert!(check_presets(&[cpu_only], true, false).is_ok());
        assert!(check_presets(&[cpu_only], false, true).is_err());
        let all: Vec<Preset> = PRESETS.iter().map(|(n, _)| parse_preset(n).unwrap()).collect();
        assert!(check_presets(&all, true, false).is_ok(), "the cpu implements every preset");
        // M5 T5: the GPU implements the optimal parse too, and since M6 B4 the M6 options
        // (opt16p1).
        assert!(check_presets(&all, true, true).is_ok(), "cpu and gpu implement every preset");
        let p1 = parse_preset("opt16p1").unwrap();
        assert!(check_presets(&[p1], true, true).is_ok(), "cpu and gpu implement opt16p1");
        let opt16 = parse_preset("opt16").unwrap();
        assert!(check_presets(&[opt16], false, true).is_ok(), "opt16 runs on the gpu");
    }

    /// `--batch max` for the optimal parse under the 6 GiB budget at `--inflight 3`: its larger
    /// per-block footprint (8 B of candidates and of trace per position, `MAX_SEQS_OPT` seqs,
    /// K3opt's prices and scratch, all in `vram_bytes`) resolves to a smaller batch than
    /// lvl9s12seg: about 1.78 MiB per block at b3458 with three slots' upload and staging
    /// buffers (≈ 1.82 at b1000, where the `head` tables weigh more), so 3458 blocks (copy upload).
    #[test]
    fn resolve_max_batch_shrinks_for_opt() {
        use gzc_core::params::{LVL9S12SEG, OPT14, OPT16, OPT16P1};
        let (budget_mb, inflight, device_max) = (6144u64, 3u32, 100_000u32);
        let lvl = resolve_max_batch(LVL9S12SEG, inflight, budget_mb, device_max, false).unwrap();
        let o16 = resolve_max_batch(OPT16, inflight, budget_mb, device_max, false).unwrap();
        let o14 = resolve_max_batch(OPT14, inflight, budget_mb, device_max, false).unwrap();
        eprintln!("--batch max at {budget_mb} MiB, i{inflight}: lvl9s12seg {lvl}, opt {o16}");
        assert_eq!(o16, o14, "opt14 and opt16 allocate the same buffers");
        assert!(o16 < lvl, "opt {o16} should resolve below lvl9s12seg {lvl}");
        let fits = |b: u32| vram_bytes(&sweep_cfg(OPT16, b, inflight)).div_ceil(1 << 20) <= budget_mb;
        assert!(fits(o16) && !fits(o16 + 1));
        let per_block = vram_bytes(&sweep_cfg(OPT16, 1000, inflight)) as f64 / 1000.0 / (1u64 << 20) as f64;
        assert!((1.8..1.85).contains(&per_block), "{per_block} MiB per block");
        assert!((3400..3500).contains(&o16), "opt --batch max {o16} at 6 GiB, i3");
        // M6 opt16p1: three sparse chains add 3 * BLOCK_SIZE / 4 pred words per block (+192 KiB),
        // so the budget allows fewer: 3125 blocks. On a device whose storage
        // bindings stop at 2 GiB (an RTX 5090 under wgpu) `device_max` is lower still: the pred
        // buffer, 704 KiB per block, caps the batch at 2978 (`max_batch_blocks`).
        let p1 = resolve_max_batch(OPT16P1, inflight, budget_mb, device_max, false).unwrap();
        let fits = |b: u32| vram_bytes(&sweep_cfg(OPT16P1, b, inflight)).div_ceil(1 << 20) <= budget_mb;
        assert!(p1 < o16 && fits(p1) && !fits(p1 + 1), "opt16p1 {p1}, opt16 {o16}");
        assert_eq!(p1, 3125, "opt16p1 --batch max at 6 GiB, i3");
    }

    #[test]
    fn preset_help_lists_every_preset() {
        use clap::CommandFactory;
        let mut cli = Cli::command();
        for cmd in ["ref", "gpu", "all"] {
            let help = cli.find_subcommand_mut(cmd).unwrap().render_long_help().to_string();
            for (name, _) in PRESETS {
                assert!(help.contains(name), "{cmd} --help does not list {name}");
            }
        }
    }

    #[test]
    fn levels_default_to_1_through_6_and_reject_above_16() {
        for cmd in ["cpu", "all"] {
            let levels = |extra: &[&str]| -> Result<Vec<i32>, clap::Error> {
                let argv = [&["gzc-bench", cmd, "--synthetic"], extra].concat();
                Ok(match Cli::try_parse_from(argv)?.command {
                    Command::Cpu(a) => a.levels,
                    Command::All(a) => a.levels,
                    _ => unreachable!(),
                })
            };
            assert_eq!(levels(&[]).unwrap(), (1..=6).collect::<Vec<_>>(), "{cmd}");
            assert_eq!(levels(&["--levels", "3,16"]).unwrap(), vec![3, 16], "{cmd}");
            assert!(levels(&["--levels", "17"]).is_err(), "{cmd}: 17 must be rejected");
            assert!(levels(&["--levels", "1,19"]).is_err(), "{cmd}: 19 must be rejected");
        }
    }
}
