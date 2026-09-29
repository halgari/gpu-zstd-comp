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

## Corpus (`tools/fetch-corpus`, dev-only)

`gzc-bench` needs a realistic byte corpus to measure ratio/throughput on. Nothing
in the workspace depends on this tool — it just populates `data/corpus/` on disk.
`data/` and `out/` are gitignored; nothing under them is ever committed.

Requires `NEXUS_API_KEY` (a Nexus Mods Premium account, for `download_link`) and
`7z` on `PATH`.

```sh
# List a mod's files to find a file_id to pin (prints file_id, category_name,
# size_kb, file_name):
NEXUS_API_KEY=... cargo run -p fetch-corpus -- list skyrimspecialedition 659

# Fetch + unpack everything pinned in corpus.toml into data/archives and
# data/corpus/<name>/… (loose files; any .bsa found after 7z extraction is
# unpacked with the `ba2` crate and then deleted). Re-running skips archives
# that are already downloaded and corpus dirs that already have a `.done`
# marker.
NEXUS_API_KEY=... cargo run --release -p fetch-corpus -- fetch
```

`corpus.toml` pins exact Nexus file ids (not "latest") so the corpus is
reproducible; see the comments there for which mods/files were chosen and why.

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

`gzc-bench ref` runs the CPU reference compressor (`gzc_core::reference`, engine
`cpu-ref`, config `lvl3-greedy`: level-3-style greedy parse, predefined FSE
tables, raw literals) over a corpus, across the given thread counts, and writes
the same table/JSON/HTML report shape. `--verify` decompresses every produced
frame with libzstd after the timed pass and errors on any mismatch against the
original block.

```sh
cargo run --release -p gzc-bench -- ref --synthetic --threads 1,8 --verify

cargo run --release -p gzc-bench -- ref \
  --input data/corpus --ext dds,nif --max-bytes 2000000000 \
  --threads 8 --verify --out out
```

`gzc-bench all` runs cpu-libzstd (across `--levels` x `--threads`) then cpu-ref
(across `--threads`, with `--verify`) — and, once the GPU engine lands, gpu too
— into a single combined report, so all engines' ratio-vs-throughput points sit
on one chart:

```sh
cargo run --release -p gzc-bench -- all \
  --input data/corpus --ext dds,nif --max-bytes 2000000000 \
  --levels 1,3,5,7,9,12,15,19 --threads 1,8,16,32 --verify --out out
```
