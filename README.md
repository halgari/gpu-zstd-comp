# gpu-zstd-comp

A prototype GPU (wgpu/compute-shader) zstd compressor: match finding, parsing and
entropy coding run on the GPU in fixed-size blocks, producing standard zstd frames
that decode with libzstd. `gzc-core` holds the shared, CPU-checkable primitives
(block chunking, the sequence/repeat-offset model, hashing, synthetic test corpora,
the frame writer and the CPU reference encoder the GPU mirrors); `gzc-gpu` holds the wgpu
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
  --levels 1,2,3,4,5,6 --threads 1,8,16,32 \
  --out out
```

`--levels` defaults to `1,2,3,4,5,6`; levels above 16 are rejected.

`--max-bytes N` caps the amount of corpus data loaded (stops adding files
once the running total reaches `N`). Reports land in `--out` (default
`out/`, gitignored): `results.json` and `report.html`.

`gzc-bench ref` runs the CPU reference compressor (`gzc_core::reference`, engine
`cpu-ref`, config `<preset>`: hash-chain match finding and parse, sequences with
per-stream predefined / RLE / computed FSE tables, Huffman (or RLE / raw)
literals; the same frames the GPU emits, byte for byte) over a corpus, across
the given thread counts, and writes the same table/JSON/HTML report shape.
`--verify` decompresses every produced frame with libzstd after the timed pass
and errors on any mismatch against the original block.

`--preset <list>` (on `ref`, `gpu` and `all`; default `lvl3`) picks the match
parameters (`gzc_core::params::PRESETS`), one run per preset:

| Preset | hashes | min match | depth | parse | Compare against |
|---|---|---|---|---|---|
| `lvl3` | dfast (8 B + 5 B) | 5 | 1 | greedy | M3 output (byte-identical) / L3 |
| `rung1` | single 4 B | 4 | 8 | greedy | L5 |
| `rung2` | single 4 B | 4 | 8 | lazy | L6 |
| `lvl9` | single 4 B | 4 | 32 | lazy2 | L9 |

All four presets run on both `cpu-ref` and the GPU. See `docs/results/2026-09-29-m4.md`
for the M4 measurement and `docs/results/2026-09-30-speed.md` for the speed-phase
headline: GPU `lvl9` now reaches **4107 MB/s** at `--batch max --inflight 3` (**4260
MB/s** at `--inflight 2`), ratio 1.355 — matching libzstd L9's ratio (1.3532) at 669
MB/s on 8 threads of the same CPU. All these numbers are measured on an RTX 5090; the
speed-phase doc also has a projection for an 8 GB-class card.

```sh
cargo run --release -p gzc-bench -- ref --synthetic --threads 1,8 --verify

cargo run --release -p gzc-bench -- ref \
  --input data/corpus --ext dds,nif --max-bytes 2000000000 \
  --threads 8 --verify --out out
```

`gzc-bench gpu` runs the streaming GPU compressor (engine `gpu`, config
`<preset> b<batch> i<inflight>`, e.g. `lvl9 b2559 i3`): the GPU runs the whole parse and emits
complete zstd frames (Huffman literals included), byte-identical to `cpu-ref`;
the host only uploads blocks and copies finished frames out. Each comma-separated
list is swept:

- `--preset P` match presets (default `lvl3`; see above).
- `--batch N` blocks per GPU batch (default 512), or `max`: the largest batch that
  fits `--vram-budget-mb` at a given preset and `--inflight` (capped by the
  device's own limit), resolved separately per preset — the max batch depends on
  how much scratch memory the preset's hash chains need per block, so it is not
  the same number for every preset. Lists can mix the two, e.g. `--batch 512,max`.
  The resolved number is what shows up in the run's config label.
- `--inflight K` batches in flight (default 3).
- `--writer-threads W` CPU threads that receive finished frames (default 0: the
  pipeline thread does it).
- `--vram-budget-mb M` (default 6144, i.e. an ~8 GB card minus headroom): every
  (batch, inflight) config must fit, checked before anything runs.
- `--verify` decompresses every frame with libzstd after the timed pass.

At the default 6144 MiB budget, `--batch max` resolves (RTX 5090) to b2047 for
`lvl3` (its two hash chains cost more scratch per block) and b2559 for the
single-hash presets (`rung1`, `rung2`, `lvl9`) at `--inflight 3`; at `--inflight 2`
it's b2047 for `lvl3` and b2860 for the single-hash presets. These depend on the
pipeline's VRAM footprint (`gzc_gpu::pipeline::vram_bytes`), not just the device, so
re-derive them for your own card/build with e.g.:

```sh
cargo run --release -p gzc-bench -- gpu --synthetic --preset lvl3,rung1,lvl9 --batch max --inflight 3
```

```sh
cargo run --release -p gzc-bench -- gpu --synthetic --verify   # smoke run

cargo run --release -p gzc-bench -- gpu \
  --input data/corpus --ext dds,nif --preset lvl9 --batch max --inflight 3 --verify --out out
```

`gzc-bench all` runs cpu-libzstd (across `--levels` x `--threads`), cpu-ref
(across `--preset` x `--threads`) and gpu (across the GPU flags above) into a single combined
report, so all engines' ratio-vs-throughput points sit on one chart. The GPU
configs and adapter are checked before the CPU runs start; if the GPU sweep
fails, the CPU results are still written.

```sh
cargo run --release -p gzc-bench -- all \
  --input data/corpus --ext dds,nif --max-bytes 2000000000 \
  --levels 1,2,3,4,5,6 --threads 1,8,16,32 --preset lvl3,rung1,rung2,lvl9 \
  --batch max --inflight 3 --verify --out out
```

## Tuning / diagnostics

Environment knobs for the GPU kernels (`crates/gzc-gpu`), all read once at startup or
kernel construction, never per record:

- `GZC_NO_SUBGROUPS` (anything but `0`): forces the fallback K1 (`k1_chains.wgsl`)
  **and** the sequential K3 (`k3_parse.wgsl`/`k3_lazy.wgsl`) — this is the code path
  GPUs without subgroup support run, and the floor for throughput on this card (see
  `docs/results/2026-09-30-speed.md`).
- `GZC_K1_GROUPS=N`: live head tables (persistent workgroups) the subgroup K1 kernel
  keeps resident; default 128, sized for the ~32 MB L2 of 8 GB-class cards. 256 is
  about +7% on an RTX 5090 (96 MB L2).
- `GZC_K3_MODE=seq|coop`: forces the sequential or subgroup-cooperative K3 kernel
  (`coop` errors if the device/probe can't support it); default: auto-detected.
- `GZC_K3_W=4|8|16|32|64`: cooperative K3's lanes per block (at most the device's
  minimum subgroup size).
- `GZC_K3_BPW=2`: two blocks per cooperative-K3 workgroup (needs min == max subgroup
  size == W); opt-in, worthwhile on Ada's 24-workgroups-per-SM limit.
- `GZC_PACK` (anything but `0`): GPU-side frame packing instead of the fixed-stride
  copy; opt-in, since it's slower than the copy on an RTX 5090 but saves PCIe traffic
  worth it on ×8-lane cards.
- `GZC_NO_TIMESTAMPS` (anything but `0`): leaves `Features::TIMESTAMP_QUERY` off, to
  time runs without per-kernel timestamp queries.
- `GZC_K3_FORCE_FALLBACK=1` — **test only**, not a tuning knob: makes every workgroup of
  the cooperative K3 kernel take its in-kernel sequential fallback path (the one a
  failed lane-layout guard takes), so tests can exercise it without a device that
  actually fails the guard.
