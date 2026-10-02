# Speed phase: context for idea generation

## Goal
Make the GPU zstd compressor much faster at libzstd-level-9 ratio (preset `lvl9`) on typical hardware:
an ~8 GB gaming GPU (RTX 4060 / RX 7600 class: roughly 24–34 SMs/CUs, ~270–290 GB/s, 32 MB L2 or
Infinity Cache) alongside an 8-core CPU. The only dev GPU is an RTX 5090 (170 SMs, ~1.8 TB/s,
96 MB L2), so ideas that scale down well (use fewer SMs efficiently, less VRAM, less bandwidth)
matter more than ideas that only help a huge GPU.

## Hard constraints
- Output must be standard zstd frames (one frame per 128 KiB block) that libzstd decodes exactly.
- GPU output must be **byte-identical to a CPU oracle** (`crates/gzc-core`). If an idea changes the
  algorithm (for example a different parse, candidate selection or hash), the CPU oracle changes in
  lockstep and a new preset is defined. `lvl9` ratio on the corpus must stay ≥ libzstd L9 (1.3532;
  ours is currently 1.3549).
- VRAM budget 6144 MiB for the whole pipeline (every buffer, including staging).
- wgpu 30 / WGSL only, and u32 arithmetic (no shader-int64). Subgroup operations are available as a
  wgpu feature on Vulkan (check adapter support; there must be a fallback or a hard requirement that
  is reasonable on 4060/7600-class GPUs). Default workgroup storage is 16 KiB; more can be requested
  up to the adapter limit (typically 32–48 KiB on consumer GPUs).
- Blocks are independent (no cross-block context). The input is millions of blocks (~300 GB), so
  cross-block parallelism is unlimited; latency does not matter, only throughput.

## Current pipeline (per batch of N blocks, e.g. N=1638, inflight 3, shared scratch)
1. **K1 hash chains** (21 % of GPU time for lvl9, 23.3 ms/batch). One workgroup of 256 threads per
   (block, hash). It walks the block in 512 tiles of 256 positions, sequentially. Per tile it does a
   shared-memory bitonic sort of (hash16 << 8 | lane), then finds the in-tile predecessor, reads the
   global per-block `head[h]` table (2^16 u32, stored as pos+1, cleared per batch) for the first of each
   run, and writes `head[h]` for the last. Barriers between tiles. Output: `pred[p]` = previous position
   with the same hash (a hash chain), u32 per position. Hash = 4 bytes, multiply-xor, 16 bits.
2. **K2 best match** (15 %, 16.6 ms/batch). One thread per position (2D dispatch). It walks
   `pred` up to depth 32. Each step is a dependent global load of pred[q], then a capped compare
   (≤ 64 bytes, u32 loads at unaligned offsets, two-word funnel shift) against the block data.
   Keeps the longest (tie → nearest). Writes `best[p] = (offset, len)`, 8 B per position.
3. **K3 parse** (55 %, 61.3 ms/batch). **One thread per block, workgroup size 1**, a sequential
   lazy2 parse (a port of zstd's lazy_generic) over `best[]`. It extends capped matches uncapped,
   checks rep offsets, catches up backwards, and emits sequences (3 u32 each) and literals (byte
   accumulator flushed as u32) to per-block regions. Workgroup size 1 was 6× faster than 64
   (one block per subgroup/warp).
4. **K5 Huffman literals** (3 %). 256 threads per block: histogram, thread-0 table build, parallel
   4-stream encode with a prefix sum and atomicOr.
5. **K4 sequence entropy + frame** (6 %). 64 threads per block: histograms, thread-0 FSE
   normalize/table build/backward bitstream, cooperative literal copy, frame assembly.
- Host: per-slot persistent upload buffer (map, write, copy), then submit, then map the readback of
  fixed-stride frame buffers. End to end is 1611 MB/s against 1819 MB/s kernel-only (≈ 11 % host
  overhead). Kernels within a batch run serially (separate compute passes). Batches share scratch
  (head/pred/best/seqs/lits/counts) on one queue, so batch i+1's K1 cannot overlap batch i's K3.
- VRAM per 128 KiB block: shared scratch ≈ 2.4–2.9 MiB (data 128 KiB, head 256 KiB, pred 512 KiB,
  best 1 MiB, seqs ~384 KiB, lits 128 KiB, …) + ≈ 0.5 MiB per in-flight slot.

## Measured (RTX 5090, full 6.49 GB corpus, 95 % DDS textures)
- GPU lvl9: 1611 MB/s end-to-end, ratio 1.3549. Per batch: K1 23.3, K2 16.6, K3 61.3, K4 7.1, K5 3.2 ms.
- libzstd L9 on 8 CPU threads: 669 MB/s, ratio 1.3532. L4 on 8 threads: 2101 MB/s, 1.334.

## Files
crates/gzc-gpu/src/{compressor.rs, pipeline.rs, chains.rs, context.rs},
crates/gzc-gpu/src/shaders/{common.wgsl, k1_chains.wgsl, k2_best.wgsl, k3_parse.wgsl, k3_lazy.wgsl,
k4_seq_entropy.wgsl, k5_huffman.wgsl}; CPU oracle crates/gzc-core/src/{reference.rs, lazy.rs, hash.rs,
params.rs}. Spec: docs/design/specs/2026-09-29-m4-lazy-lvl9-design.md. Results:
docs/results/2026-09-29-m4.md.
