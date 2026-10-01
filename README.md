# gpu-zstd-comp

A GPU zstd compressor written in Rust with wgpu compute shaders (WGSL). Input is cut into
independent blocks (64 KiB by default). The GPU does the match finding, the parse and the entropy
coding, and writes one standard zstd frame per block. Every frame decodes with stock libzstd.

It was built for one job: recompressing Skyrim mod data (mostly DDS textures, plus NIF meshes)
while it downloads, on a gaming PC, at up to 10 Gbit/s (1250 MB/s). It is a prototype. There is
no GPU decompression, and blocks never reference each other.

## Status (2026-10-01)

- **Ten presets** run on the GPU, from libzstd-level-3 ratio up to libzstd-level-16 ratio. The
  `gzc-core` crate holds a CPU reference encoder for each preset. GPU output is byte-identical
  to it, and tests check that on every supported backend.
- **Fastest preset:** `lvl9s12seg` beats libzstd level 9's ratio at about 10.4 GB/s on an
  RTX 5090.
- **Highest-ratio presets:** `opt14` and `opt16` beat libzstd levels 14 and 16 at
  1.8–2.3 and 1.1–1.5 GB/s on the same card.
- **Platforms:** Linux/Vulkan on NVIDIA is the main development platform. macOS/Metal works and
  was benchmarked on an M4 Pro. Windows/Vulkan works but has an open bug (see below).
  Windows/DX12 does not work.
- **Next steps:** a research round on making `opt16` faster has finished. Its recommendations
  have not been implemented yet:
  - kernel changes that keep the output byte-identical (fewer registers, skipping positions
    that can't start a match, a persistent heaviest-block-first parse kernel);
  - a 2-pass `opt16` that gets its ratio margin from splitting each frame into several zstd
    blocks and from a longer-match hash chain.

  See `docs/superpowers/m6/synthesis.md` and the twelve research reports next to it.

## Performance: RTX 5090 vs Ryzen 9 9950X3D

Measured 2026-10-01 at commit `ba061ce` (`opt14`/`opt16` rows re-measured at `d75bebd` on branch `m6`, after the M6 Track A kernel work: `docs/results/2026-10-01-m6-trackA.md`) on one Linux machine: an RTX 5090 (Vulkan, driver 610)
and a Ryzen 9 9950X3D (16 cores, 32 threads).

- **Corpus:** the full corpus, `--ext dds,nif`: 3172 files, 6.49 GB, 100,754 blocks of 64 KiB.
- **What MB/s means:** uncompressed bytes per second of wall time, including upload and
  readback.
- **Ratio:** uncompressed size divided by compressed size.
- **GPU runs:** median of 3 runs at `--batch max --inflight 3` and the default 6 GiB
  VRAM budget. A separate `--verify` run decoded every frame of every preset with libzstd, with
  no mismatches.
- **CPU runs:** libzstd (`gzc-bench cpu`), one run per cell.
- **Background load:** a browser and two idle VMs were running.

| GPU preset | GPU MB/s | Ratio | Same-ratio libzstd level | CPU MB/s, 8 / 16 / 32 threads | libzstd ratio |
|---|---:|---:|---|---:|---:|
| `lvl3` | 7,447 | 1.264 | L3 | 3,841 / 7,945 / 9,360 | 1.262 |
| `rung1` | 9,750 | 1.330 | L6 (closest) | 964 / 1,911 / 2,265 | 1.335 |
| `rung2` | 7,503 | 1.337 | L6 | 964 / 1,911 / 2,265 | 1.335 |
| `lvl9` | 5,794 | 1.33932 | L9 | 731 / 1,420 / 1,747 | 1.33786 |
| `lvl9seg` | 9,134 | 1.33931 | L9 | 731 / 1,420 / 1,747 | 1.33786 |
| `lvl9s12` | 6,217 | 1.33927 | L9 | 731 / 1,420 / 1,747 | 1.33786 |
| `lvl9s12seg` | **10,396** | 1.33926 | L9 | 731 / 1,420 / 1,747 | 1.33786 |
| `lvl9s12d16seg` | 11,086 | 1.33860 | L9 | 731 / 1,420 / 1,747 | 1.33786 |
| `opt14` | 2,935 (2,862 at batch 2900) | 1.37064 | L14 | 224 / 415 / 590 | 1.36827 |
| `opt16` | 2,003 (1,914 at batch 2900) | 1.37144 | L16 | 190 / 384 / 501 | 1.37100 |

How to read it:

- **Against all 32 CPU threads:** at level-9 ratio the GPU is about 6× faster (`lvl9s12seg`,
  10.4 GB/s against 1.75 GB/s). At level 14 it is about 5× faster and at level 16 about 4×.
  At level 3 the CPU wins (9.4 GB/s against 7.4).
- **The 8-thread column** is the closest thing here to the target machine, an 8-core gaming
  PC. Against it the GPU is 14× faster at level 9, 13× at level 14 and 10× at level 16.
  The 9950X3D's cores are faster than a typical gaming CPU's.
- **`opt14` and `opt16` at `--batch max`:** since the M6 kernel work (register diet, persistent
  heaviest-first parse) a 3586-block batch fits one wave and is the fastest setting.
- **Without subgroups:** `lvl9s12seg` runs at 8,569 MB/s with `GZC_NO_SUBGROUPS=1`, the
  portable kernels every GPU can run.
- **Small cards:** an RTX 4060 has about 14 % of a 5090's compute. Projections, not
  measurements, put `opt16` at about 150–215 MB/s and `opt14` at 250–365 MB/s on an RTX 4060.

Ratios with five decimals come from `gzc-bench ref` (the CPU reference, which the GPU matches
byte for byte); the others are rounded from the run output. Full write-ups:
`docs/results/2026-09-30-speed2.md` (the level-9 presets) and `docs/results/2026-09-30-m5.md`
(`opt14`/`opt16`).

## Other hardware

**Apple M4 Pro** (16-core GPU, 12 CPU cores, 24 GB, macOS 15.1, Metal), on the same corpus.
Details are in `docs/results/2026-09-30-m4pro.md`.

- **Correctness:** every preset verified.
- **The GPU loses to the CPU at every level on this machine:**

  | GPU preset | GPU MB/s | libzstd level | CPU MB/s, 12 threads |
  |---|---:|---|---:|
  | `lvl3` | 491 | L3 | 4,821 |
  | `lvl9seg` | 576 | L9 | 692 (721 at 8 threads) |
  | `opt14` | 221 | L14 | 295 |
  | `opt16` | 161 | L16 | 259 |

- **GPU and CPU together:** with the GPU and 8 CPU threads each compressing their own share of
  the blocks, level-16 ratio reached 366 MB/s, 42 % more than the CPU alone.
- **Metal tuning:** two settings on the `metal-exp` branch (subgroup width 32, 32 K1
  workgroups) add 4–27 %. They are not on master yet.

**GTX 1660 Super** (Windows 11, Vulkan, Ryzen 5 7600X), same corpus, two verified runs each
(`docs/results/2026-10-01-gtx1660s.md`): `lvl9s12seg` 552 MB/s, `lvl9seg` 370, `opt14` 95,
`opt16` 58. The parse is 80 % of `opt16`'s time on this card. The fastest preset clears 1 Gbit but
not 10 Gbit; `opt14`/`opt16` are below 1 Gbit. CPU baseline not measured yet.

## What has been tested

- **Linux, RTX 5090, Vulkan:**
  - the full test suite, with and without subgroups;
  - full-corpus `--verify` for every preset;
  - GPU vs CPU-reference differential tests on synthetic and corpus blocks;
  - extra test modes that make shaders behave like Apple and AMD GPUs, fill every buffer with
    garbage before each batch (`GZC_POISON`), and stall threads at random (`GZC_EMULATE_SKEW`).
- **macOS, M4 Pro, Metal:** the full test suite and a verified full-corpus benchmark of every
  preset.
- **Windows, GTX 1660 Super, Vulkan:** build, the test suite (with the failure below), and the
  smoke test.
- **Linux, AMD Ryzen iGPU, OpenGL backend:** one quick run, before the Metal-era fixes.

## What has not been tested

- **AMD discrete cards** (RDNA 2/3) on Vulkan, Windows or Linux. AMD's Vulkan driver has not
  run this code at all.
- **Intel Arc.**
- **The common 8 GB NVIDIA cards** (RTX 3060, RTX 4060). Every number for them is a
  projection.
- **The CPU baseline on the GTX 1660 Super machine** (Ryzen 5 7600X).
- **Linux distributions other than the dev machine's (Arch-based),** and other driver versions.
- **Other Apple chips** (M1–M3, base M4).

## Known problems

- **Wrong output on the GTX 1660 Super under load.** On Windows/Vulkan with subgroups on, a few
  frames came out the right length with their last 1–3 bytes wrong. It happened only when
  other work was using the GPU at the same time, and no error was reported.
  - It has not reproduced on the RTX 5090, including under the poison, skew and contention
    test modes.
  - Two rounds of hardening have gone in since: subgroup operations moved out of branches, and
    allocation and device-loss errors are now reported. Neither has been re-tested on that card.
  - Until it is, use `--verify` (or decode and check) on anything other than the dev machine.
- **DX12 does not work.** On Windows with `WGPU_BACKEND=dx12`, the smoke test reports
  "Data corruption detected", and startup takes several minutes. Vulkan is wgpu's default on
  the machines tried, but nothing stops wgpu from picking DX12. Use `WGPU_BACKEND=vulkan` on
  Windows.
- **Driver-killed jobs hang the process.** If the driver kills a GPU job for running too long
  (NVIDIA Xid 109), the process waits forever instead of returning an error. A watchdog is not
  written yet.
- **Many devices per process on Metal.** Creating many GPU devices at once in one process can
  lose a device. Only the test suite does that, and it limits itself to 2 at a time
  (`GZC_GPU_TEST_SLOTS`).
- **Metal throughput:** subgroup kernels fall back to their slower versions on Metal, because
  wgpu reports Apple's subgroup width as 4–64. The fix is on the `metal-exp` branch.

## Layout

- `crates/gzc-core`: the CPU side, checkable without a GPU. Block chunking, the sequence and
  repeat-offset model, hashing, synthetic test data, the frame writer, and the CPU reference
  encoder that the GPU mirrors.
- `crates/gzc-gpu`: the WGSL kernels (`src/shaders/`), the pipeline that streams batches
  through the GPU, and the GPU tests.
- `crates/gzc-bench`: the benchmark CLI. It compares libzstd, the CPU reference and the GPU on
  a corpus, and writes a table, JSON and an HTML report.
- `tools/fetch-corpus`: downloads the benchmark corpus pinned in `corpus.toml` from Nexus Mods.
- `docs/results/`: dated results for each phase. `docs/superpowers/`: specs, plans and design
  notes.

Blocks are always 64 KiB and independent: each is one standard zstd frame.

## Build

```sh
cargo build --workspace
```

## Test

```sh
cargo test --workspace
```

The portable (no-subgroup) kernels: `GZC_NO_SUBGROUPS=1 cargo test --workspace --release`.

GPU tests that open their own device and build full-size pipelines take one of
`GZC_GPU_TEST_SLOTS` (default 2) slots per test binary (`gzc_gpu::test_support`), so a parallel
run does not put a dozen devices on one GPU at once; light tests run beside them unthrottled.

## Corpus (`tools/fetch-corpus`, dev-only)

`gzc-bench` needs a realistic byte corpus to measure ratio/throughput on. Nothing
in the workspace depends on this tool; it just populates `data/corpus/` on disk.
`data/` and `out/` are gitignored; nothing under them is ever committed.

Requires `NEXUS_API_KEY` (a Nexus Mods Premium account, for `download_link`) and
`7z` on `PATH` (Linux: `p7zip`/`7zip` package; macOS: `brew install p7zip`; Windows: 7-Zip).

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

| Preset | What it is | Compare against |
|---|---|---|
| `lvl3` | dfast chains (8 B + 5 B), min match 5, depth 1, greedy | M3 output (byte-identical) / L3 |
| `rung1` | single 4 B hash chain, depth 8, greedy | L5 |
| `rung2` | `rung1` with a lazy parse | L6 |
| `lvl9` | single 4 B hash chain, depth 32, lazy2 | L9 |
| `lvl9seg` | `lvl9` with the parse split into independent 4 KiB segments (speed-2 E1) | L9 |
| `lvl9s12` | `lvl9` with a 12-bit hash key; the GPU finder bucket-sorts candidates per block (E2) | L9 |
| `lvl9s12seg` | `lvl9s12` + segmented parse: the fastest preset at L9 ratio | L9 |
| `lvl9s12d16seg` | `lvl9s12seg` walking 16 candidates instead of 32 | L9 |
| `opt14` | M5 optimal parse (3-byte matches, priced DP per 4 KiB segment), prior seed + 1 re-pricing pass | L14 |
| `opt16` | M5 optimal parse, block-init seed + 3 re-pricing passes | L16 |

Full-corpus ratios at 64 KiB (`gzc-bench ref`, which the GPU matches byte for byte):

| Preset | Ratio | libzstd | libzstd ratio |
|---|---:|---|---:|
| `lvl9` | 1.33932 | L9 | 1.33786 |
| `lvl9seg` | 1.33931 | L9 | 1.33786 |
| `lvl9s12` | 1.33927 | L9 | 1.33786 |
| `lvl9s12seg` | 1.33926 | L9 | 1.33786 |
| `lvl9s12d16seg` | 1.33860 | L9 | 1.33786 |
| `opt14` | 1.37064 | L14 | 1.36827 |
| `opt16` | 1.37144 | L16 | 1.37100 |

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
  device's own limit), resolved separately per preset; the max batch depends on
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
- `GZC_DUMP_WGSL=<dir>`: writes every shader module's final WGSL to `<dir>/<label>.<n>.wgsl`
  (a dev aid for offline register and shared-memory statistics).
- `GZC_PACK` (anything but `0`): GPU-side frame packing instead of the fixed-stride
  copy; opt-in, since it's slower than the copy on an RTX 5090 but saves PCIe traffic
  worth it on ×8-lane cards. Packing **disables the transfer-queue readback** (packed frames are
  read back on the main queue).
- `GZC_NO_TIMESTAMPS` (anything but `0`): leaves `Features::TIMESTAMP_QUERY` off, to
  time runs without per-kernel timestamp queries.
- `GZC_EMULATE_SHIFT_MOD32`, `GZC_EMULATE_VEC_RMW` (anything but `0`), **test only**: every
  shader is rewritten (through naga) to behave as on other GPUs: shifts take their amount mod 32
  (Apple, AMD; NVIDIA gives 0 for a shift by 32 or more), and a store to one component of a
  workgroup vector is a read-modify-write of the whole vector (Apple's Metal, which corrupted
  every frame on an M4 Pro until K4 stopped doing it). `tests/differential_emulated.rs` runs the
  differential suite with both on (`gzc_gpu::emulate`). `GZC_EMULATE_SKEW`, **test only**:
  timing skew, every invocation stalls pseudo-randomly at entry, after each barrier and before
  each subgroup operation, for races a slow or preempted GPU would expose (much slower). Meant
  for fast GPUs: on a GTX 1660 Super the stalled kernels run long enough to lose the device.
  `cargo run --release -p gzc-gpu --example gpu_hog -- [GiB] [s]` is a second GPU tenant for
  contention runs.
- `GZC_K3_FORCE_FALLBACK=1`, **test only**, not a tuning knob: makes every workgroup of
  the cooperative K3 kernel take its in-kernel sequential fallback path (the one a
  failed lane-layout guard takes), so tests can exercise it without a device that
  actually fails the guard.
- `GZC_POISON` (anything but `0`), **test only**: memory poisoning (`gzc_gpu::poison`). Every
  buffer gets 4 KiB of padding, and before every batch garbage fills every scratch and output
  buffer, the input past the batch's trailing zero word and the padding; workgroup memory loses
  its zero-init and is dirtied by a garbage kernel. Output must stay byte-identical: no kernel may
  read memory it did not write in that batch. `GZC_POISON_SEED=N` fixes the patterns.
  `tests/poison.rs` runs a differential subset poisoned; `GZC_POISON=1 cargo test` the whole suite.

**Backends.** The transfer-queue readback and the direct-upload ReBAR detection only work on
Vulkan. On Metal they are off: readback runs on the main queue and uploads are copied, unless
`GZC_DIRECT_UPLOAD=1`. That gives correct output at lower speed. DX12 currently gives wrong
output (see Known problems), so set `WGPU_BACKEND=vulkan` on Windows.
