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
}

#[derive(Args)]
struct CpuArgs {
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

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Cpu(args) => run_cpu_cmd(args),
    }
}

fn run_cpu_cmd(args: CpuArgs) -> anyhow::Result<()> {
    let corpus = if args.synthetic {
        Corpus::synthetic()
    } else {
        let input = args
            .input
            .clone()
            .ok_or_else(|| anyhow::anyhow!("either --input DIR or --synthetic is required"))?;
        corpus::load(&LoadOpts { input, exts: args.ext.clone(), max_bytes: args.max_bytes })?
    };

    eprintln!(
        "corpus: {} files, {} blocks, {} real bytes",
        corpus.files,
        corpus.blocks.len(),
        corpus.real_bytes()
    );

    let mut results = Vec::new();
    for &threads in &args.threads {
        for &level in &args.levels {
            eprintln!("running cpu-libzstd L{level} @ {threads} threads...");
            results.push(cpu::run_cpu(&corpus, level, threads)?);
        }
    }

    report::print_table(&results);

    std::fs::create_dir_all(&args.out)?;
    let json_path = report::write_json(&results, &args.out)?;
    let html_path = report::write_html(&results, &args.out)?;
    println!("JSON report: {}", json_path.display());
    println!("HTML report: {}", html_path.display());

    Ok(())
}
