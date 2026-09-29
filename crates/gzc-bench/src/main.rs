//! Benchmark CLI: compares CPU baseline, CPU reference and GPU compressors over a corpus.
mod corpus;
mod cpu;
mod result;
mod report;
mod refrun;
mod gpurun;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

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
    /// Every engine (cpu-libzstd, cpu-ref, and gpu once T9 lands) into one report.
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
struct AllArgs {
    #[command(flatten)]
    corpus: CorpusArgs,
    /// Comma-separated zstd compression levels (cpu-libzstd only).
    #[arg(long, value_delimiter = ',', default_value = "1,3,5,7,9,12,15,19")]
    levels: Vec<i32>,
    /// Comma-separated thread counts, used by both cpu-libzstd and cpu-ref.
    #[arg(long, value_delimiter = ',', default_value = "1,8,16,32")]
    threads: Vec<usize>,
    /// Decompress every cpu-ref frame with libzstd after the timed pass and
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

fn run_all_cmd(args: AllArgs) -> anyhow::Result<()> {
    let corpus = load_corpus(&args.corpus)?;
    log_corpus(&corpus);

    // One Vec<RunResult> holds every engine's runs so `write_reports` produces
    // a single combined report. T9 (gpu engine) appends its RunResults here
    // too, before `write_reports` is called.
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

    write_reports(&results, &args.out)
}
