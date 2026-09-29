# gpu-zstd-comp

A prototype GPU (wgpu/compute-shader) zstd compressor: match finding, parsing and
entropy coding run on the GPU in fixed-size blocks, producing standard zstd frames
that decode with libzstd. `gzc-core` holds the shared, CPU-checkable primitives
(block chunking, the sequence/repeat-offset model, hashing, synthetic test corpora,
and — as later tasks land — the CPU reference encoder); `gzc-gpu` holds the wgpu
kernels and host-side orchestration; `gzc-bench` compares CPU-baseline, CPU-reference
and GPU throughput/ratio over a corpus; `tools/fetch-corpus` downloads/builds the
benchmark corpus described by `corpus.toml`.

Block size is a compile-time feature on `gzc-core`/`gzc-gpu`/`gzc-bench`: exactly one
of `block-16k`, `block-32k`, `block-64k`, `block-128k` (default `block-128k`).

## Build

```sh
cargo build --workspace
```

## Test

```sh
cargo test --workspace
```

Non-default block size (all three feature-gated crates must agree):

```sh
cargo test --workspace --no-default-features \
  --features gzc-core/block-16k,gzc-gpu/block-16k,gzc-bench/block-16k
```

## Benchmark CLI

`gzc-bench cpu` runs the CPU libzstd baseline over a corpus, across the given
compression levels and thread counts, and writes a results table, a JSON
dump and a self-contained HTML report (throughput-vs-ratio chart, headline
comparison at the 1/10 Gbit marks, and a per-kind ratio table).

```sh
# Data-free smoke run (no corpus on disk required):
cargo run --release -p gzc-bench -- cpu --synthetic --levels 1,3 --threads 1,8

# Against a real corpus directory, filtered to DDS/NIF files:
cargo run --release -p gzc-bench -- cpu \
  --input data/corpus --ext dds,nif \
  --levels 1,3,5,7,9,12,15,19 --threads 1,8,16,32 \
  --out out
```

`--max-bytes N` caps the amount of corpus data loaded (stops adding files
once the running total reaches `N`). Reports land in `--out` (default
`out/`, gitignored): `results.json` and `report.html`.
