//! Benchmark CLI: compares CPU baseline, CPU reference and GPU compressors over a corpus.
mod corpus;
mod cpu;
mod result;
mod report;
mod refrun;
mod gpurun;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use gzc_core::params::{MatchParams, PRESETS, cpu_supports};
use gzc_gpu::compressor::{GpuParams, gpu_supports, max_batch_blocks};
use gzc_gpu::context::GpuContext;
use gzc_gpu::pipeline::{PipelineConfig, vram_bytes};

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
        anyhow::ensure!(!cpu || cpu_supports(&p.params), "preset '{}' is not implemented yet on cpu", p.name);
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
    /// Comma-separated match presets (lvl3, rung1, rung2, lvl9).
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

#[derive(Args)]
struct GpuSweepArgs {
    /// Comma-separated blocks per GPU batch.
    #[arg(long, value_delimiter = ',', default_value = "512")]
    batch: Vec<u32>,
    /// Comma-separated number of batches in flight.
    #[arg(long, value_delimiter = ',', default_value = "3")]
    inflight: Vec<u32>,
    /// Comma-separated number of CPU frame-writer threads. The GPU emits finished frames, so a
    /// writer only records (in a real tool: writes out) the bytes it is handed; 0 = the pipeline
    /// thread does that itself, N > 0 = N threads fed through a bounded channel.
    #[arg(long, value_delimiter = ',', default_value = "0")]
    writer_threads: Vec<usize>,
    /// GPU memory budget in MiB (default: an 8 GB card minus headroom). Every (batch, inflight)
    /// config's pipeline footprint (`gzc_gpu::pipeline::vram_bytes`) must fit, else the run errors.
    #[arg(long, default_value_t = 6144)]
    vram_budget_mb: u64,
}

#[derive(Args)]
struct GpuArgs {
    #[command(flatten)]
    corpus: CorpusArgs,
    /// Comma-separated match presets (lvl3, rung1, rung2, lvl9).
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
    /// Comma-separated match presets for cpu-ref and gpu (lvl3, rung1, rung2, lvl9).
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

/// GPU memory of the frame-path pipeline for `cfg`, checked against `budget_mb`.
fn check_vram(cfg: &PipelineConfig, budget_mb: u64) -> anyhow::Result<u64> {
    let cfg = PipelineConfig { params: GpuParams { emit_frames: true, ..cfg.params }, ..*cfg };
    let mib = vram_bytes(&cfg).div_ceil(1 << 20);
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

/// Checks every preset is implemented on the GPU and validates every (preset, batch, inflight)
/// config of `sweep` against the VRAM budget, then opens the GPU and checks each batch against
/// the device's limit. Run before any timed work so a bad config or a missing adapter fails fast.
fn gpu_preflight(presets: &[Preset], sweep: &GpuSweepArgs) -> anyhow::Result<GpuContext> {
    check_presets(presets, false, true)?;
    for p in presets {
        for &batch in &sweep.batch {
            for &inflight in &sweep.inflight {
                anyhow::ensure!(inflight >= 1, "--inflight must be at least 1");
                check_vram(&sweep_cfg(p.params, batch, inflight), sweep.vram_budget_mb)?;
            }
        }
    }
    let ctx = GpuContext::new()?;
    let max = max_batch_blocks(&ctx.device.limits());
    for &batch in &sweep.batch {
        anyhow::ensure!(batch >= 1 && batch <= max, "--batch {batch} not in 1..={max} for this device");
    }
    Ok(ctx)
}

/// Runs every (preset, batch, inflight, writer_threads) combination on `ctx`, appending to
/// `results`. Configs must have passed `gpu_preflight`.
fn run_gpu_sweep(
    ctx: &GpuContext,
    corpus: &Corpus,
    presets: &[Preset],
    sweep: &GpuSweepArgs,
    verify: bool,
    results: &mut Vec<result::RunResult>,
) -> anyhow::Result<()> {
    for p in presets {
        for &batch in &sweep.batch {
            for &inflight in &sweep.inflight {
                for &writers in &sweep.writer_threads {
                    let cfg = sweep_cfg(p.params, batch, inflight);
                    let mib = check_vram(&cfg, sweep.vram_budget_mb)?;
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

    #[test]
    fn vram_budget_rejects_configs_that_do_not_fit() {
        let params = GpuParams { matching: gzc_core::params::LVL3, emit_frames: false, huffman: true };
        let cfg = PipelineConfig { batch: 64, inflight: 2, params };
        let mib = check_vram(&cfg, 1 << 20).unwrap();
        assert!(mib > 0);
        assert_eq!(check_vram(&cfg, mib).unwrap(), mib, "a config exactly at the budget fits");
        let err = check_vram(&cfg, mib - 1).unwrap_err().to_string();
        assert!(err.contains("--vram-budget-mb") && err.contains("b64 i2"), "{err}");
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
            assert_eq!(names(&["--preset", "lvl3,lvl9"]), ["lvl3", "lvl9"], "{cmd}");
            assert_eq!(presets(&["--preset", "rung1"]).unwrap()[0].params, gzc_core::params::RUNG1, "{cmd}");
            let err = presets(&["--preset", "lvl3,bogus"]).unwrap_err().to_string();
            assert!(err.contains("bogus") && err.contains("lvl3") && err.contains("lvl9"), "{cmd}: {err}");
        }
        let lvl9 = parse_preset("lvl9").unwrap();
        let lvl3 = parse_preset("lvl3").unwrap();
        assert!(check_presets(&[lvl3], true, true).is_ok());
        let err = check_presets(&[lvl3, lvl9], true, false).unwrap_err().to_string();
        assert_eq!(err, "preset 'lvl9' is not implemented yet on cpu");
        let err = check_presets(&[lvl9], false, true).unwrap_err().to_string();
        assert_eq!(err, "preset 'lvl9' is not implemented yet on gpu");
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
