# gpu-zstd-comp reference

Details behind the README: the corpus tool, the benchmark CLI, the streaming API and the tuning and test switches.

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
and the GPU, byte for byte the same frames. (The GPU parses greedily over the whole block, or lazily
and optimally in 4 KiB segments; a lazy parse over the whole block exists only in the CPU
reference, and no preset uses it.)

| Preset | What it is | Compare against |
|---|---|---|
| `lvl3` | dfast chains (8 B + 5 B), min match 5, depth 1, greedy | M3 output (byte-identical) / L3 |
| `lvl9seg` | single 4 B hash chain, depth 32, lazy2, the parse split into independent 4 KiB segments (speed-2 E1) | L9 |
| `lvl9s12seg` | `lvl9seg` with a 12-bit hash key; the GPU finder bucket-sorts candidates per block (E2): the fastest preset at L9 ratio, except on Apple GPUs, where `lvl9seg` is faster | L9 |
| `opt14` | M5 optimal parse (3-byte matches, priced DP per 4 KiB segment), prior seed + 1 re-pricing pass | L14 |
| `opt16` | M5 optimal parse, block-init seed + 3 re-pricing passes | L16 |
| `opt16p1` | M6: one optimal-parse pass over more candidates (three extra hash chains on every 4th position), a retrained prior seed, then short matches turned back into literals where cheaper | L16 |

Full-corpus ratios at 64 KiB (`gzc-bench ref`, which the GPU matches byte for byte):

| Preset | Ratio | libzstd | libzstd ratio |
|---|---:|---|---:|
| `lvl9seg` | 1.33931 | L9 | 1.33786 |
| `lvl9s12seg` | 1.33926 | L9 | 1.33786 |
| `opt14` | 1.37064 | L14 | 1.36827 |
| `opt16` | 1.37144 | L16 | 1.37100 |
| `opt16p1` | 1.37229 | L16 | 1.37100 |

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
- `--facade` runs the same sweep through the library's `Compressor::compress_blocks` (engine
  `gpu-facade`), which returns every frame in one buffer. `--writer-threads` is not used.

After each run the bench prints the GPU time per kernel to stderr, from timestamp queries:
`k1_chains` (or `k1_sort`), `k2_best` (or `k2_window`), `k3_parse`, `k4_entropy`, `k5_huffman`,
then their `sum`. A `k3_trunc` row follows when the input has short blocks. It is K3t, the kernel
that cuts a short block's parse to the block's real length. K3t runs only in batches that hold a
short block; its per-batch figure is its total divided by all batches. The `gpu_*` and `host_*`
rows say where the time outside the kernels went (`gzc_gpu::pipeline::TRANSFER_NAMES`).

At the default 6144 MiB budget and 64 KiB blocks, `--batch max` resolves (RTX 5090, with the
direct upload active) to b4095 for `lvl3` (the device's own limit) and b5403 for the
single-hash presets (`lvl9seg`, `lvl9s12seg`) at `--inflight 3`; at `--inflight 2` it's b4095
and b6078. Without the direct upload (`GZC_DIRECT_UPLOAD=0`, or no full ReBAR) the pipeline keeps a
shared `data` buffer and the single-hash presets resolve to b5118 at `--inflight 3`. These depend
on the pipeline's VRAM footprint (`gzc_gpu::pipeline::vram_bytes_with`) and the
device, so re-derive them for your own card/build with e.g.:

```sh
cargo run --release -p gzc-bench -- gpu --synthetic \
  --preset lvl3,lvl9seg,lvl9s12seg,opt14,opt16,opt16p1 --batch max --inflight 3
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
  --levels 1,2,3,4,5,6,9 --threads 1,8,16,32 --preset lvl3,lvl9seg,lvl9s12seg \
  --batch max --inflight 3 --verify --out out
```

## Library API (`gzc_gpu`)

`cargo doc --no-deps -p gzc-gpu --open` has the full API with examples. This is the overview.

### Compressor

```rust
use gzc_gpu::{Compressor, CompressorOptions, Level};

let compressor = Compressor::new(Level::Zstd16)?;
let frames = compressor.compress(&data)?;
for (i, frame) in frames.iter().enumerate() {
    // `frame` is the complete zstd frame of block `i`
}
```

- `Level::{Zstd3, Zstd9, Zstd14, Zstd16}` are the presets `lvl3`, `lvl9s12seg`, `opt14` and
  `opt16p1`. The output for a level is the same on every GPU, and equal to `gzc-bench ref` for
  the preset.
- `Compressor::compress(&data)` splits `data` into 64 KiB blocks. The last block may be short;
  its frame holds exactly its bytes. Empty input gives zero frames.
- `Compressor::compress_blocks(&blocks)` takes blocks of 1 to 65536 bytes. Any block may be
  short, so it also serves independent chunks.
- `Frames` holds every frame in one buffer: `len()`, `frame(i)`, `iter()`, and `as_bytes()`,
  which is a zstd stream of the whole input. `compress` keeps all frames in memory and reserves
  that buffer up front at the input's size plus 64 bytes per block. Use `stream` for input that
  should not be held twice.
- `CompressorOptions` has the match parameters (any preset of `gzc_core::params`), the VRAM
  budget (6144 MiB), an explicit batch size, the batches in flight (3) and the `GpuOptions`.
  Without an explicit batch the compressor takes the largest batch that fits the budget and the
  device, as `--batch max` does (`gzc_gpu::pipeline::max_batch_for_budget`). The budget caps
  what the compressor asks for. It is not checked against the memory the adapter has or has
  free; on a smaller or busy card, building the compressor fails with `OutOfMemory`.
- A `Compressor` is `Send + Sync`; calls on one compressor run one after another. Build it once:
  it owns the device, the kernels and the buffers.
- A context that reads frames back through a transfer queue serves one compressor at a time.
  `Compressor::with_context` on such a context while another compressor is alive is
  `InvalidInput`.

### Streaming

`Compressor::stream(on_batch, produce)` compresses while data arrives, with no copy on either
side.

```rust
compressor.stream(
    |batch| {
        // On a second thread, in submission order. Frames point into the readback buffer.
        for (index, frame) in batch.frames() {
            write_out(index, frame);
        }
        Ok(())
    },
    |stream| {
        // On the calling thread.
        let mut batch = stream.next_batch()?;
        let mut payloads = batch.reserve(&[file_a.len(), file_b.len()])?;
        payloads[0].write(&file_a);
        payloads[1].write(&file_b);
        drop(payloads);
        batch.submit()?;
        Ok(())
    },
)?;
```

- `Stream::next_batch()` waits for a free batch of `batch_blocks()` blocks.
- `Batch::reserve(&lens)` takes each payload's length and returns one write-only `Payload` per
  entry, exactly that long. A payload is a file or an independent chunk: it starts on a block
  boundary and only its last block may be short. The batch zeroes the padding and keeps every
  block's real length, and `Batch::submit()` takes no block count. A short block therefore
  always gets a frame of exactly its own bytes.
- A `Payload` is written front to back: `write(&bytes)` appends, and it implements
  `std::io::Write`. It counts what it was given. Upload memory is reused and never cleared, so
  `Batch::submit()` is `InvalidInput` unless every reserved payload was written to its end;
  nothing is submitted then. A frame can therefore never hold bytes of an earlier batch.
- Payloads are disjoint and `Send`: several threads can fill them at once, one thread per
  payload. To fill one large file from several threads, reserve it as several payloads of whole
  blocks; the frames are the same. A batch need not be full, so a flush timer can submit what
  it has.
- `FrameBatch` (`first_index()`, `tag()`, `len()`, `frame(k)`, `frames()`) is `Send + Sync`. Its
  readback buffer is reused once it is dropped, which may happen on a writer thread. Holding
  as many batches as are in flight stalls the stream.
- `Stream::submit_blocks(&blocks)` is the copying form inside a stream.
- An error from either closure stops the stream and is returned as it was. A panic is re-raised.
  The compressor stays usable.
- The compressor is busy until `stream` returns. Calling it from inside either closure is
  `InvalidInput`; a call from another thread waits.

A `Payload` is mapped GPU upload memory, which may be write-combined and cannot be read. A
decompressor reads its own output back for matches, so do not decode into a payload: decode into
ordinary memory and write the result in.

### Errors

`gzc_gpu::Error` is `NoAdapter`, `Unsupported`, `OutOfMemory`, `DeviceLost`, `InvalidInput` or
`Other` (with its source). On `NoAdapter` or `DeviceLost` a program can fall back to the CPU:
`gzc_core::reference::compress_block_to_frame` writes the same frames. On `OutOfMemory` it can
lower the VRAM budget.

### Device options

`GpuOptions` says how the device is opened and which kernels are built. `GpuOptions::default()`
reads nothing from the environment. `GpuOptions::try_from_env()` (and
`CompressorOptions::try_from_env(level)`) applies the `GZC_*` variables below and returns
`InvalidInput` for a value a variable does not take. `from_env()` is the same, panicking on
such a value; `gzc-bench` and the tests use these. A `GpuOptions` value out of range
(`k3_width: Some(7)`, `k1_groups: Some(0)`) is `InvalidInput` when the device is opened. No
option changes the compressed output. wgpu's own `WGPU_BACKEND` variable picks
the backend either way.

### The pipeline underneath (`gzc_gpu::pipeline`)

`Compressor` is built on `Pipeline`, which stays public for programs that need more: several
pipelines on one `Arc<GpuContext>`, per-kernel timing (`PipelineStats`), the parse-only path
(`run`), raw literals, and delivery from several threads (`run_frames_par`).

- A `Pipeline` owns `inflight` slots, each a mapped upload buffer and a staging buffer. The
  calling thread is the producer. A completion thread per run waits for the batches in
  submission order and hands them to the sink, so delivery overlaps the next uploads.
- `Pipeline::stream_frames(on_batch, produce)` is what `Compressor::stream` wraps. Its
  `UploadSlot` hands out whole-block regions (`regions_mut`) and leaves the lengths to the
  caller: finish each payload with `Region::pad(len)`, or call `set_real_len` after writing
  through the `unsafe` `blocks_mut`, then `submit(n)`. A caller that skips this gets a 64 KiB
  frame for a short block. This layer does not track writes either: a submitted block that was
  not fully written holds an earlier batch's bytes. `Batch` and `Payload` exist to rule both
  out.
- `run_frames(&blocks, &mut FrameSink)`, `run_frames_par(&blocks, &ParFrameSink, threads)` and
  `run(&blocks, &mut BlockSink)` copy the blocks in. Sinks are called on the completion thread,
  so they must be `Send`.

## Tuning / diagnostics

Environment knobs for the GPU kernels and host pipeline (`crates/gzc-gpu`). `gzc-bench` and the
tests read them once, through `GpuOptions::try_from_env()` / `from_env()`; each one sets a field
of `GpuOptions`. A program that builds `GpuOptions::default()` is not affected by them. Boolean
knobs follow one convention: a knob that is off by default is turned on by any value but `0`;
one that is on by default is turned off by `0` only. Every other variable takes the values
listed with it. Any other value is an error that names the variable (`gzc-bench` exits with
it); none is silently ignored.

- `GZC_NO_SUBGROUPS` (anything but `0`): a device without subgroups, i.e. the portable kernels
  every GPU can run. Chain presets use the fallback K1 (`k1_chains.wgsl`); the sorted preset
  (`lvl9s12seg`) uses the workgroup-memory bucket sort (`k1_sort.wgsl`) instead of
  `k1_sort_sg.wgsl`, with the same window K2. K3 is the sequential kernel
  (`k3_parse.wgsl`) for `lvl3`'s unsegmented greedy parse; the segmented parse (`*seg`) never
  uses subgroups. Output is byte-identical either way; `lvl9s12seg` does 8722 MB/s this way on the
  RTX 5090 (see `docs/results/2026-09-30-speed2.md`).
- `GZC_SORTED=0`: the sorted preset uses the hash-chain K1/K2 over the same 12-bit key instead of
  the bucket-sorted finder (byte-identical, slower); for comparisons.
- `GZC_K1_GROUPS=N`: live head tables (persistent workgroups) the subgroup chain K1 keeps
  resident; default 128, sized for the ~32 MB L2 of 8 GB-class cards. 256 is about +7% on an RTX
  5090 (96 MB L2).
- `GZC_K3_MODE=seq|coop`: forces the sequential or subgroup-cooperative K3 kernel of the
  unsegmented greedy parse (`lvl3`; `coop` errors if the device/probe can't support it);
  default: auto-detected. The segmented and optimal parses have one K3 each.
- `GZC_K3_W=4|8|16|32|64`: cooperative K3's lanes per block (at most the device's
  minimum subgroup size).
- `GZC_TRANSFER_QUEUE=0`: turns off the transfer-queue readback (speed-2 E3). By default, on a
  Vulkan 1.2+ adapter with timeline semaphores and a transfer-only queue family, the frame path
  reads each batch back on that dedicated copy queue, overlapping the next batch's kernels. Only
  one such pipeline may exist per `GpuContext` (a second `Pipeline::new` errors), and nothing else
  may submit to the context's queue from another thread while it exists (see
  `GpuContext`).
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
- `GZC_NO_TIMESTAMPS` (anything but `0`): leaves `Features::TIMESTAMP_QUERY` off, to
  time runs without per-kernel timestamp queries.
- `GZC_EMULATE_SHIFT_MOD32`, `GZC_EMULATE_VEC_RMW` (anything but `0`), **test only**: every
  shader is rewritten (through naga) to behave as on other GPUs: shifts take their amount mod 32
  (Apple, AMD; NVIDIA gives 0 for a shift by 32 or more), and a store to one component of a
  workgroup vector is a read-modify-write of the whole vector (Apple's Metal, which corrupted
  every frame on an M4 Pro until K4 stopped doing it). `tests/differential_emulated.rs` runs the
  differential suite with both on (`GpuOptions::emulate`). `GZC_EMULATE_SKEW`, **test only**:
  timing skew, every invocation stalls pseudo-randomly at entry, after each barrier and before
  each subgroup operation, for races a slow or preempted GPU would expose (much slower). Meant
  for fast GPUs: on a GTX 1660 Super the stalled kernels run long enough to lose the device.
  `cargo run --release -p gzc-gpu --example gpu_hog -- [GiB] [s]` is a second GPU tenant for
  contention runs.
- `GZC_K3_FORCE_FALLBACK` (anything but `0`), **test only**, not a tuning knob: makes every workgroup of
  the cooperative K3 kernel take its in-kernel sequential fallback path (the one a
  failed lane-layout guard takes), so tests can exercise it without a device that
  actually fails the guard.
- `GZC_POISON` (anything but `0`), **test only**: memory poisoning (`GpuOptions::poison`). Every
  buffer gets 4 KiB of padding, and before every batch garbage fills every scratch and output
  buffer, the input past the batch's trailing zero word and the padding; workgroup memory loses
  its zero-init and is dirtied by a garbage kernel. Output must stay byte-identical: no kernel may
  read memory it did not write in that batch. `GZC_POISON_SEED=N` fixes the patterns.
  `tests/poison.rs` runs a differential subset poisoned; `GZC_POISON=1 cargo test` the whole suite.

**Backends.** The transfer-queue readback and the direct-upload ReBAR detection only work on
Vulkan. On Metal they are off: readback runs on the main queue and uploads are copied, unless
`GZC_DIRECT_UPLOAD=1`. That gives correct output at lower speed. DX12 currently gives wrong
output (see Known problems), so set `WGPU_BACKEND=vulkan` on Windows.

**Library users:** the `block-16k`/`32k`/`64k`/`128k` Cargo features were removed in M6;
the block size is always 64 KiB. Drop any `features = ["block-…"]` from dependent manifests.
