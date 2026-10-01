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
of `block-16k`, `block-32k`, `block-64k`, `block-128k` (default `block-64k`, the largest block
size the downloader uses; blocks are always independent).

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

The portable (no-subgroup) kernels: `GZC_NO_SUBGROUPS=1 cargo test --workspace --release`.

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
parameters (`gzc_core::params::PRESETS`), one run per preset. Every preset runs on both `cpu-ref`
and the GPU, byte for byte the same frames:

| Preset | What it is | Compare against | ≥ libzstd L9 at |
|---|---|---|---|
| `lvl3` | dfast chains (8 B + 5 B), min match 5, depth 1, greedy | M3 output (byte-identical) / L3 | – |
| `rung1` | single 4 B hash chain, depth 8, greedy | L5 | – |
| `rung2` | `rung1` with a lazy parse | L6 | – |
| `lvl9` | single 4 B hash chain, depth 32, lazy2 | L9 | 16, 32, 64, 128 KiB |
| `lvl9seg` | `lvl9` with the parse split into independent 4 KiB segments (speed-2 E1) | L9 | 16, 32, 64, 128 KiB |
| `lvl9s12` | `lvl9` with a 12-bit hash key; the GPU finder bucket-sorts candidates per block (E2) | L9 | 16, 32, 64, 128 KiB |
| `lvl9s12seg` | `lvl9s12` + segmented parse: **the fastest preset validated at every block size** | L9 | 16, 32, 64, 128 KiB |
| `lvl9s12d16seg` | `lvl9s12seg` walking 16 candidates instead of 32 | L9 | 16, 32, 64 KiB only (below L9 at 128 KiB) |
| `opt14` | M5 optimal parse (3-byte matches, priced DP per 4 KiB segment), prior seed + 1 re-pricing pass | L14 | ≥ L14 at 16, 32, 64 KiB (the GPU runs it at ≤ 64 KiB) |
| `opt16` | M5 optimal parse, block-init seed + 3 re-pricing passes | L16 | ≥ L16 at 16, 32, 64 KiB (the GPU runs it at ≤ 64 KiB) |

Full-corpus ratios (`gzc-bench ref`, which the GPU matches byte for byte) against libzstd L9 on
the same block size:

| Block | L9 | lvl9 | lvl9seg | lvl9s12 | lvl9s12seg | lvl9s12d16seg |
|---|---:|---:|---:|---:|---:|---:|
| 16 KiB | 1.30009 | 1.30024 | 1.30042 | 1.30023 | 1.30040 | 1.30017 |
| 32 KiB | 1.31996 | 1.32131 | 1.32140 | 1.32128 | 1.32137 | 1.32107 |
| 64 KiB | 1.33786 | 1.33932 | 1.33931 | 1.33927 | 1.33926 | 1.33860 |
| 128 KiB | 1.35317 | 1.35489 | 1.35478 | 1.35468 | 1.35456 | 1.35159 |

Headline (speed phase 2, `docs/results/2026-09-30-speed2.md`, RTX 5090, 64 KiB blocks, full
corpus, `--batch max --inflight 3`): GPU **`lvl9s12seg` reaches 10588 MB/s** (median of 8 runs) at
ratio **1.33926**, above libzstd L9's 1.3379 on the same blocks (libzstd L9: 1754 MB/s on 32
threads). `lvl9` does 5767 MB/s and `lvl9seg` 8809 MB/s; with `GZC_NO_SUBGROUPS=1`, `lvl9s12seg`
does 8722 MB/s. At that point the pipeline was **host-bound** (one thread wrote uploads and
delivered frames; the GPU waited ~2 ms per batch); since the host track (`docs/results/host-log.md`)
frames are delivered on a completion thread beside the uploading thread and the GPU is the
bottleneck again, with the host's share of the wall time down from ~40–120 ms to ~25 ms per run. The earlier phases are in `docs/results/2026-09-29-m4.md` and
`docs/results/2026-09-30-speed.md` (128 KiB blocks). All these numbers are measured on an RTX
5090. The quality presets `opt14`/`opt16` (M5, `docs/results/2026-09-30-m5.md`) reach libzstd L14/L16's
ratio at 64 KiB (1.37064 / 1.37144) at 2.17 / 1.43 GB/s (batch 2900; 1.75 / 1.09 GB/s at the default
`--batch max` on a 6 GiB budget, which spills past one wave).

```sh
cargo run --release -p gzc-bench -- ref --synthetic --threads 1,8 --verify

cargo run --release -p gzc-bench -- ref \
  --input data/corpus --ext dds,nif --max-bytes 2000000000 \
  --threads 8 --verify --out out
```

`gzc-bench gpu` runs the streaming GPU compressor (engine `gpu`, config
`<preset> b<batch> i<inflight>`, e.g. `lvl9s12seg b5403 i3`): the GPU runs the whole parse and emits
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
- `--writer-threads W` CPU threads that copy out finished frames (default 0: the
  pipeline's completion thread does it, beside the uploading thread; N > 0: N threads share each
  completed batch, `Pipeline::run_frames_par`). On the RTX 5090 box 4 threads measured 0–3 % over
  0 at `--inflight 3` and 3 % at `--inflight 2` (`docs/results/host-log.md`).
- `--vram-budget-mb M` (default 6144, i.e. an ~8 GB card minus headroom): every
  (batch, inflight) config must fit, checked before anything runs.
- `--verify` decompresses every frame with libzstd after the timed pass.

At the default 6144 MiB budget and 64 KiB blocks, `--batch max` resolves (RTX 5090, with the
direct upload active) to b4095 for `lvl3` (the device's own limit) and b5403 for every
single-hash preset (`rung1` … `lvl9s12d16seg`) at `--inflight 3`; at `--inflight 2` it's b4095
and b6078. Without the direct upload (`GZC_DIRECT_UPLOAD=0`, or no full ReBAR) the pipeline keeps a
shared `data` buffer and the single-hash presets resolve to b5118 at `--inflight 3`. These depend
on the pipeline's VRAM footprint (`gzc_gpu::pipeline::vram_bytes_with`), the block size and the
device, so re-derive them for your own card/build with e.g.:

```sh
cargo run --release -p gzc-bench -- gpu --synthetic \
  --preset lvl3,rung1,rung2,lvl9,lvl9seg,lvl9s12,lvl9s12seg,lvl9s12d16seg --batch max --inflight 3
```

```sh
cargo run --release -p gzc-bench -- gpu --synthetic --verify   # smoke run

cargo run --release -p gzc-bench -- gpu \
  --input data/corpus --ext dds,nif --preset lvl9s12seg --batch max --inflight 3 --verify --out out
```

`gzc-bench all` runs cpu-libzstd (across `--levels` x `--threads`), cpu-ref
(across `--preset` x `--threads`) and gpu (across the GPU flags above) into a single combined
report, so all engines' ratio-vs-throughput points sit on one chart. The GPU
configs and adapter are checked before the CPU runs start; if the GPU sweep
fails, the CPU results are still written.

```sh
cargo run --release -p gzc-bench -- all \
  --input data/corpus --ext dds,nif --max-bytes 2000000000 \
  --levels 1,2,3,4,5,6,9 --threads 1,8,16,32 --preset lvl3,lvl9,lvl9s12seg \
  --batch max --inflight 3 --verify --out out
```

## Streaming API (`gzc_gpu::pipeline`)

A `Pipeline` owns `inflight` slots (a mapped upload buffer and a staging buffer each). The calling
thread is the producer; a completion thread per run waits for the batches in submission order
and hands them to the sink, so frame delivery overlaps the next uploads.

- `Pipeline::stream_frames(on_batch, produce)`: the zero-copy form. `produce` gets a
  `FrameStream`; `next_upload_slot()` blocks until a slot is free and returns an `UploadSlot`
  whose `regions_mut(&[blocks…])` splits the mapped upload memory itself into write-only,
  `Send` regions (write blocks, or payloads spanning many blocks, straight into them; finish a
  payload with `Region::pad`, whose per-block real lengths `payload_real_lens` gives), then
  `submit(n)` / `submit_with(n, tag)` any `n` up to the capacity (a partial batch, e.g. on a
  flush timer, is fine). `unsafe fn blocks_mut()` gives the same memory as a `&mut [u8]` (sound
  only on wgpu-core's native backends; see its docs). `on_batch` receives each completed batch
  as a `FrameBatch` (`first_index()`, `tag()`, `frame(k)`), whose frames point into the staging
  buffer; the slot is reused once the batch is dropped, which may happen on a writer thread.
  Holding `inflight` batches stalls the stream and `inflight - 1` serialises it. Errors on
  either side abort the stream (returned); panics are re-raised; the pipeline stays usable.
- `run_frames(&blocks, &mut FrameSink)` / `run_frames_par(&blocks, &ParFrameSink, threads)` /
  `run(&blocks, &mut BlockSink)`: the `&[&[u8]]` wrappers (the producer copies the blocks in,
  `GZC_UPLOAD_THREADS` threads). **Sinks passed to `run`, `run_frames` and `compress_stream*`
  must now be `Send`**: they are called on the completion thread.

The upload slot is MAP_WRITE memory: write-combined device memory with the direct upload, and
possibly uncached or write-combined host memory with the copy upload too. Write it sequentially
and never read it. A decompressor reads its own output back for matches, so do not decode into
the slot: decode into a cached buffer and copy the result in.

## Tuning / diagnostics

Environment knobs for the GPU kernels and host pipeline (`crates/gzc-gpu`), all read once at
startup or kernel construction, never per record. Boolean knobs follow one convention: a knob
that is off by default is turned on by any value but `0`; one that is on by default is turned off
by `0` only.

- `GZC_NO_SUBGROUPS` (anything but `0`): a device without subgroups, i.e. the portable kernels
  every GPU can run. Chain presets use the fallback K1 (`k1_chains.wgsl`); the sorted presets
  (`lvl9s12*`) use the workgroup-memory bucket sort (`k1_sort.wgsl`) instead of
  `k1_sort_sg.wgsl`, with the same window K2. K3 is the sequential kernel
  (`k3_parse.wgsl`/`k3_lazy.wgsl`) for unsegmented presets; the segmented parse (`*seg`) never
  uses subgroups. Output is byte-identical either way; `lvl9s12seg` does 8722 MB/s this way on the
  RTX 5090 (see `docs/results/2026-09-30-speed2.md`).
- `GZC_SORTED=0`: the sorted presets use the hash-chain K1/K2 over the same 12-bit key instead of
  the bucket-sorted finder (byte-identical, slower); for comparisons.
- `GZC_K1_GROUPS=N`: live head tables (persistent workgroups) the subgroup chain K1 keeps
  resident; default 128, sized for the ~32 MB L2 of 8 GB-class cards. 256 is about +7% on an RTX
  5090 (96 MB L2).
- `GZC_K3_MODE=seq|coop`: forces the sequential or subgroup-cooperative K3 kernel
  (`coop` errors if the device/probe can't support it); default: auto-detected.
- `GZC_K3_W=4|8|16|32|64`: cooperative K3's lanes per block (at most the device's
  minimum subgroup size).
- `GZC_K3_BPW=2`: two blocks per cooperative-K3 workgroup (needs min == max subgroup
  size == W); opt-in, worthwhile on Ada's 24-workgroups-per-SM limit.
- `GZC_TRANSFER_QUEUE=0`: turns off the transfer-queue readback (speed-2 E3). By default, on a
  Vulkan 1.2+ adapter with timeline semaphores and a transfer-only queue family, the frame path
  reads each batch back on that dedicated copy queue, overlapping the next batch's kernels. Only
  one such pipeline may exist per `GpuContext` (a second `Pipeline::new` errors), and nothing else
  may submit to the context's queue from another thread while it exists (see
  `GpuContext::transfer`).
- `GZC_DIRECT_UPLOAD=0|1`: the kernels read each batch straight from its mapped upload buffer, with
  no upload copy and no shared `data` buffer (E8). Default: on when host-visible device-local
  memory covers VRAM (full ReBAR / SAM); `0` turns it off, `1` forces it on wherever
  `MAPPABLE_PRIMARY_BUFFERS` exists (without ReBAR the kernels would then read over PCIe).
- `GZC_UPLOAD_THREADS=N`: threads writing a batch into its upload buffer (default 4, at most the
  available cores).
- `GZC_CHECKED_SHADERS` (anything but `0`): builds K2/K4/K5 with naga's bounds checks and loop
  bounding (E9 builds them unchecked); a debugging aid.
- `GZC_PACK` (anything but `0`): GPU-side frame packing instead of the fixed-stride
  copy; opt-in, since it's slower than the copy on an RTX 5090 but saves PCIe traffic
  worth it on ×8-lane cards. Packing **disables the transfer-queue readback** (packed frames are
  read back on the main queue).
- `GZC_NO_TIMESTAMPS` (anything but `0`): leaves `Features::TIMESTAMP_QUERY` off, to
  time runs without per-kernel timestamp queries.
- `GZC_EMULATE_SHIFT_MOD32`, `GZC_EMULATE_VEC_RMW` (anything but `0`) — **test only**: every
  shader is rewritten (through naga) to behave as on other GPUs: shifts take their amount mod 32
  (Apple, AMD; NVIDIA gives 0 for a shift by 32 or more), and a store to one component of a
  workgroup vector is a read-modify-write of the whole vector (Apple's Metal, which corrupted
  every frame on an M4 Pro until K4 stopped doing it). `tests/differential_emulated.rs` runs the
  differential suite with both on (`gzc_gpu::emulate`).
- `GZC_K3_FORCE_FALLBACK=1` — **test only**, not a tuning knob: makes every workgroup of
  the cooperative K3 kernel take its in-kernel sequential fallback path (the one a
  failed lane-layout guard takes), so tests can exercise it without a device that
  actually fails the guard.

**Backends:** the E3 paths (transfer-queue readback, and the direct upload's ReBAR detection) are
Vulkan-only. On DX12 (wgpu's default on Windows) and Metal they are inactive: readback runs on the
main queue and the upload is copied (unless `GZC_DIRECT_UPLOAD=1`), which gives correct output at
lower throughput. Pin the Vulkan backend (e.g. `WGPU_BACKEND=vulkan`) to get them on Windows.
