# GPU zstd compression prototype — design

Date: 2026-09-29
Status: approved in conversation; implementation authorized

## 1. Purpose

Compress data on the GPU while it is being downloaded, so a machine with a gaming
GPU and ~8 CPU cores can keep up with a fast connection *and* get a better
compression ratio than its CPU could at the same speed.

- Workload: ~300 GB of Skyrim mod data per result set, mostly `.dds` textures and
  `.nif` meshes (the data that compresses best in this use case).
- Throughput targets: 1 Gbit/s ≈ **125 MB/s**; 10 Gbit/s ≈ **1.25 GB/s** (the goal).
- Question to answer: **at ≥ 1.25 GB/s, does the GPU produce a better compression
  ratio than 8 CPU cores running libzstd?** zstd level 3 is the calibration point
  where correctness and like-for-like parity are established; the GPU's expected
  win is level 9–19-class ratio at 10 Gbit speed.
- Only compression is in scope. Output must be standard zstd decodable by libzstd.

## 2. Fixed decisions

| Decision | Choice |
|---|---|
| Language / GPU API | Rust (stable, 1.96), `wgpu` compute shaders in WGSL |
| Block size | Compile-time Cargo feature: `block-16k`, `block-32k`, `block-64k`, `block-128k` (default 128k) |
| Chunking | Each file split independently into `BLOCK_SIZE` blocks; last block zero-padded; blocks never span files |
| Output format | One zstd frame per block, each containing exactly one zstd block |
| Block independence | No shared context or dictionaries between blocks |
| CPU baseline | libzstd via the `zstd` crate, per-block, levels 1–19, thread counts 1/8/16/32 |
| Dev corpus | SMIM SE + one texture pack from Nexus, in gitignored `data/`; never required by tests |
| Test data | Deterministic synthetic generators only |
| Integers in WGSL | u32 only; no dependency on `shader-int64` |

Reference hardware: AMD Ryzen 9 9950X3D (16C/32T), NVIDIA RTX 5090 (32 GB), Linux, Vulkan.

## 3. Architecture

```
gpu-zstd-comp/
  Cargo.toml                 workspace
  crates/
    gzc-core/                no GPU. config, chunking, zstd frame/bitstream writer,
                             sequence types, CPU reference compressor, synthetic data
    gzc-gpu/                 wgpu device, WGSL shaders, batch pipeline
    gzc-bench/               CLI: corpus loading, CPU baseline, GPU runs, reports
  tools/fetch-corpus/        dev-only Nexus downloader/unpacker
  data/  out/                gitignored
```

### 3.1 gzc-core

- `BLOCK_SIZE: usize` from the block-size feature (exactly one must be active;
  default `block-128k`). `gzc-gpu` and `gzc-bench` forward the features.
- `chunk_file(&[u8]) -> Vec<Block>`: each `Block` is `BLOCK_SIZE` bytes plus
  `real_len` (bytes of real data before padding).
- `Sequence { lit_len: u32, match_len: u32, off_base: u32 }` using zstd's
  offBase convention (1–3 = repeat codes, `offset + 3` = explicit offset).
- `BlockOutput { sequences: Vec<Sequence>, literals: Vec<u8> }` — the interface
  between any match-finder/parser (CPU reference or GPU) and the frame writer.
  Trailing literals after the last sequence are in `literals`.
- Frame writer: `write_frame(block: &[u8], out: &BlockOutput, opts) -> Vec<u8>`.
  - Frame header: magic `0xFD2FB528`, single-segment flag, frame content size =
    `BLOCK_SIZE`, no dictionary ID, checksum optional (off by default; xxhash64).
  - One block, last-block flag set. Block type chosen:
    - RLE if the input block is a single repeated byte;
    - Compressed (literals section + sequences section) if smaller than raw;
    - Raw otherwise.
  - M1 literals: raw literals section. M3: Huffman (4 streams when ≥ 256
    literals, 1 stream otherwise), with raw/RLE fallback.
  - M1 sequences: predefined FSE tables for LL/ML/OF. M3: per-symbol-type
    choice of predefined / RLE / computed FSE by libzstd-style cost heuristic.
- Bit writer (little-endian, forward) and backward FSE bitstream writer with the
  zstd end-mark convention.
- CPU reference compressor: the exact algorithm the GPU runs (§3.2), in plain
  Rust, producing a `BlockOutput`. It is the oracle for GPU differential tests.
- Synthetic generators (seeded): zeros, random, repetitive text, DDS-like
  (DDS header + BC1/BC3 4×4 blocks with repeated patterns), NIF-like (header
  strings + float arrays), plus size edge cases (1 byte, `BLOCK_SIZE`,
  `BLOCK_SIZE + 1`).

### 3.2 Match-finding and parsing algorithm (shared by CPU reference and GPU)

Parameters for the level-3 calibration: `min_match = 5`, hash widths 8 bytes
("long") and 5 bytes ("short"), chain depth `D = 1` per hash, greedy parse.

1. **Key + sort (K1).** For each position `p` in `0..BLOCK_SIZE - 8`, compute
   `h = hash_w(bytes[p..p+w])` with `HASH_BITS = 32 - log2(BLOCK_SIZE)` bits, and
   `key = (h << log2(BLOCK_SIZE)) | p`. Sort keys ascending per block. Positions
   whose hash window would read past the block get no key (sentinel). Each run of
   equal `h` is that hash's chain in increasing position order; the chain
   predecessors of `p` are the preceding entries in its run. Done for each hash
   width → `pred_long[p]`, `pred_short[p]` (up to `D` predecessors each; the
   sorted array itself serves as the chain).
2. **Match find (K2).** For each `p`: for each candidate `q` among its
   predecessors (long first, then short), compute the forward match length
   `len` (bounded by `BLOCK_SIZE - p`). Keep the longest; ties go to the
   nearest `q`. Store `best[p] = (offset = p - q, len)` if `len ≥ min_match`,
   else none.
3. **Greedy parse (K3), one pass per block.** `anchor = 0`, `p = 0`,
   `rep = [1, 4, 8]`, limit `real_end = BLOCK_SIZE`:
   - Check a repeat match at `p` with `rep[0]` (if `p ≥ rep[0]`): byte-compare;
     if length ≥ `min_match`, take it (repeat code).
   - Else if `best[p]` exists, take it (explicit offset; the offBase is still
     mapped to a repeat code if it equals one of `rep`).
   - Else advance `p` by `1 + ((p - anchor) >> 8)` (literal-run acceleration),
     bounded to not skip past the end.
   - On a match: emit `Sequence { lit_len: p - anchor, match_len, off_base }`,
     update `rep` exactly per the zstd spec (including the `lit_len == 0`
     repeat-code shift rule), set `p += match_len`, `anchor = p`.
   - Stop at `BLOCK_SIZE - 8`; remaining bytes are trailing literals.
   The parse must produce zstd-valid rep-code semantics; the frame writer does
   not re-derive them.

GPU and CPU reference must produce **identical** `BlockOutput`s.

Later stages (M4) change only parameters and parse: chain depth `D > 1`, lazy
matching (check `p+1` before committing), then a price-based optimal parse over
top-K candidates per position. A suffix-array match finder is an M4 option.

### 3.3 gzc-gpu

- Device: request a high-performance adapter; request the adapter's max
  storage-buffer binding size and workgroup storage; optionally
  `TIMESTAMP_QUERY`. Fail with a clear error if no adapter.
- Block bytes are uploaded as `array<u32>` (WGSL has no u8 storage); shaders
  extract bytes with shift/mask.
- Kernels per batch of `N` blocks:
  - K1: per-block key generation + radix sort (one workgroup per block per hash
    width, operating in global memory scratch).
  - K2: one thread per position, `best[]` as two u32 (offset, len).
  - K3: one thread per block, greedy parse → sequences + literals into
    per-block output regions sized for the worst case, plus counts.
  - M3 adds: literal histogram + Huffman table + parallel 4-stream encode (prefix
    sum over code lengths, atomicOr bit placement); per-block FSE sequence
    encode (one thread per block); final block bytes assembled on GPU.
- Pipeline: `--batch N` blocks per submission (default 512), `--inflight K`
  batches (default 3) overlapping upload / compute / readback. Buffers
  allocated once per in-flight slot and reused.
- API: `GpuCompressor::new(config) -> Result<Self>`;
  `compress_batch(&[Block]) -> Vec<BlockOutput>` (M2) /
  `-> Vec<Vec<u8>>` frames (M3+); a streaming `compress_all` that drives the
  in-flight pipeline and returns per-kernel timings.
- Shader constants (`BLOCK_SIZE`, `LOG2_BLOCK`, `HASH_BITS`, `MIN_MATCH`, `D`)
  are injected at pipeline creation via WGSL `override` constants or source
  templating.

Memory budget: ~2.5 MB scratch per 128 KB block → ~1.3 GB for N = 512.

### 3.4 gzc-bench

- `gzc-bench cpu --input DIR [--ext dds,nif] [--max-bytes N] [--levels 1,3,...] [--threads 1,8,16,32]`
- `gzc-bench gpu --input DIR [...] [--batch N] [--inflight K] [--config lvl3|...]`
- `gzc-bench all ...` runs both and writes the report.
- Corpus: recursive load into RAM, per-file chunking, file-type tag
  (dds / nif / other) on each block.
- Metrics per run: throughput in MB/s of **real** input bytes (not padding),
  ratio = real bytes / compressed bytes, MB/s per thread (CPU), per-kernel GPU
  time (timestamp queries when available). Broken out per file type and total.
- GPU timing: warmup, then steady-state wall clock over the full corpus
  including upload, readback and any CPU-side frame assembly. Kernel-only time
  reported separately.
- `--verify`: libzstd round-trip of every output, excluded from timing.
- Outputs: stdout table; `out/results-<ts>.json`; `out/report-<ts>.html`
  (self-contained inline-SVG chart: x = throughput, y = ratio; CPU 8-thread
  curve by level; CPU 1-thread × 8 dashed projection; GPU points; vertical lines
  at 125 MB/s and 1250 MB/s; headline: best ratio at ≥ 1.25 GB/s for CPU-8T vs GPU).

### 3.5 tools/fetch-corpus

- Reads `NEXUS_API_KEY`; manifest `corpus.toml` pins `{game, mod_id, file_id}`
  entries (SMIM SE = skyrimspecialedition 659, plus one texture pack).
- Premium `download_link` API → `data/archives/`; extract with system `7z` into
  `data/corpus/<name>/`; unpack any `.bsa` via the `ba2` crate.
- Idempotent (skips existing downloads). Not a dependency of any other crate.

## 4. Testing

1. gzc-core unit tests: bit writers, predefined FSE encoding, frame header
   fields, hand-built `BlockOutput`s (zero literals, match to end of block, no
   sequences, all three repeat codes, lit_len == 0 repeat shift) decoded by
   libzstd and compared.
2. CPU reference round-trips every synthetic block through libzstd.
3. GPU differential: GPU `BlockOutput` == CPU reference for every synthetic block.
4. GPU round-trip of every synthetic block.
5. Bench `--verify` against the real corpus (manual, dev only).

GPU tests require an adapter and fail loudly without one.

## 5. Milestones

- **M0** Scaffold, block-size features, corpus fetcher, CPU baseline + report.
  Done: CPU ratio/throughput curve on DDS/NIF; known 8-thread crossover of 1.25 GB/s.
- **M1** Frame writer (raw literals, predefined FSE, raw/RLE fallback) + CPU
  reference compressor. Done: all synthetic + corpus blocks round-trip; ratio vs
  libzstd lvl 3 known.
- **M2** GPU K1–K3 + batched pipeline. Done: GPU == CPU reference on all test
  blocks; round-trip; end-to-end and per-kernel throughput reported.
- **M3** GPU Huffman literals + computed FSE sequences, GPU-side block assembly.
  Done: round-trip; ratio within a few % of libzstd lvl 3.
- **M4** Deeper search, lazy, optimal parse; config sweep. Done: headline
  question answered with data.

## 6. Out of scope

GPU decompression; dictionaries or cross-block context; multi-GPU; downloader
integration; container formats beyond one frame per block.
