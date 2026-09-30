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
use gzc_gpu::pipeline::PipelineConfig;

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
    /// Streaming GPU compressor (level-3 greedy parse on the GPU, frames written on the CPU).
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
    /// Comma-separated number of CPU frame-writer threads.
    #[arg(long, value_delimiter = ',', default_value = "2,4,8")]
    writer_threads: Vec<usize>,
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

/// Runs every (batch, inflight, writer_threads) combination, appending to `results`.
fn run_gpu_sweep(corpus: &Corpus, sweep: &GpuSweepArgs, verify: bool, results: &mut Vec<result::RunResult>) -> anyhow::Result<()> {
    for &batch in &sweep.batch {
        for &inflight in &sweep.inflight {
            for &writers in &sweep.writer_threads {
                let cfg = PipelineConfig { batch, inflight, params: GpuParams { depth: 1 } };
                eprintln!("running gpu lvl3-greedy b{batch} i{inflight} @ {writers} writer threads (verify={verify})...");
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
