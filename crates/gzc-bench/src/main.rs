//! Benchmark CLI: compares CPU baseline, CPU reference and GPU compressors over a corpus.
mod corpus;
mod cpu;
mod result;
mod report;
mod refrun;
mod gpurun;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use gzc_gpu::compressor::GpuParams;
use gzc_gpu::pipeline::{PipelineConfig, vram_bytes};

use corpus::{Corpus, LoadOpts};

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
    /// Streaming GPU compressor (level-3 greedy parse and complete zstd frames on the GPU;
    /// literals stay raw until the GPU does Huffman).
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
    /// Comma-separated zstd compression levels.
    #[arg(long, value_delimiter = ',', default_value = "1,3,5,7,9,12,15,19")]
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
    /// Comma-separated zstd compression levels (cpu-libzstd only).
    #[arg(long, value_delimiter = ',', default_value = "1,3,5,7,9,12,15,19")]
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

fn run_ref_cmd(args: RefArgs) -> anyhow::Result<()> {
    let corpus = load_corpus(&args.corpus)?;
    log_corpus(&corpus);

    let mut results = Vec::new();
    for &threads in &args.threads {
        eprintln!("running cpu-ref lvl3-greedy @ {threads} threads (verify={})...", args.verify);
        results.push(refrun::run_ref(&corpus, threads, args.verify)?);
    }

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

/// Runs every (batch, inflight, writer_threads) combination, appending to `results`. Every
/// config is checked against the VRAM budget before anything runs.
fn run_gpu_sweep(corpus: &Corpus, sweep: &GpuSweepArgs, verify: bool, results: &mut Vec<result::RunResult>) -> anyhow::Result<()> {
    let cfg = |batch, inflight| {
        PipelineConfig { batch, inflight, params: GpuParams { depth: 1, emit_frames: true, huffman: true } }
    };
    for &batch in &sweep.batch {
        for &inflight in &sweep.inflight {
            check_vram(&cfg(batch, inflight), sweep.vram_budget_mb)?;
        }
    }
    for &batch in &sweep.batch {
        for &inflight in &sweep.inflight {
            for &writers in &sweep.writer_threads {
                let cfg = cfg(batch, inflight);
                let mib = check_vram(&cfg, sweep.vram_budget_mb)?;
                eprintln!(
                    "running gpu lvl3-greedy b{batch} i{inflight} ({mib} MiB GPU memory) @ {writers} writer threads (verify={verify})..."
                );
                results.push(gpurun::run_gpu(corpus, &cfg, writers, verify)?);
            }
        }
    }
    Ok(())
}

fn run_gpu_cmd(args: GpuArgs) -> anyhow::Result<()> {
    let corpus = load_corpus(&args.corpus)?;
    log_corpus(&corpus);

    let mut results = Vec::new();
    run_gpu_sweep(&corpus, &args.sweep, args.verify, &mut results)?;

    write_reports(&results, &args.out)
}

fn run_all_cmd(args: AllArgs) -> anyhow::Result<()> {
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

    for &threads in &args.threads {
        eprintln!("running cpu-ref lvl3-greedy @ {threads} threads (verify={})...", args.verify);
        results.push(refrun::run_ref(&corpus, threads, args.verify)?);
    }

    run_gpu_sweep(&corpus, &args.gpu, args.verify, &mut results)?;

    write_reports(&results, &args.out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vram_budget_rejects_configs_that_do_not_fit() {
        let cfg = PipelineConfig { batch: 64, inflight: 2, params: GpuParams { depth: 1, emit_frames: false, huffman: true } };
        let mib = check_vram(&cfg, 1 << 20).unwrap();
        assert!(mib > 0);
        assert_eq!(check_vram(&cfg, mib).unwrap(), mib, "a config exactly at the budget fits");
        let err = check_vram(&cfg, mib - 1).unwrap_err().to_string();
        assert!(err.contains("--vram-budget-mb") && err.contains("b64 i2"), "{err}");
    }
}
