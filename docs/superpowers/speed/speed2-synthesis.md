# Speed phase 2: research synthesis

Inputs are ten independent reports (`.superpowers/speed2/r1…r10`). Baseline on `master` @ 6bedcff: lvl9 **4107 MB/s** (b2559 i3) on the RTX 5090. Kernel time per batch
is K3 31.0, K1 16.0, K2 10.4, K5 5.4 and K4 2.6 ms, plus 9 ms of serial copies.

## Where the reports converge

- **K3's cost is sequential depth, not lane width.** R1, R8 and R10 all show that on the 5090 the batch waits
  on its densest block (wall time is 2× the work). R4 measured that splitting the parse into 4 KiB segments
  cuts the longest dependent chain by about 20× for −0.014 % ratio (still +0.11 % over L9).
- **The match finder is saturated, but its memory access pattern is not.** R5 shows an ideal suffix-array
  finder gains only +0.06 %, and depth 16 costs only −0.02 % (R5, R10). A bucket-sorted per-block candidate
  array turns K2's dependent random loads into a contiguous window read. With a 13-bit key it also puts K1's
  table in workgroup memory. R1 recommends the same sort formulation independently.
- **Overlap is the portable structural win, and wgpu blocks it by design.** R3 found that wgpu chains every
  submission with a TOP_OF_PIPE semaphore. A second (async) compute queue plus a transfer queue can be added
  through wgpu-hal hooks without leaving wgpu: est. +20–30 %, and more on PCIe ×8 cards. R2 reaches the same
  conclusion from CUDA's side (streams are half of CUDA's projected gain).
- **Fusion and megakernels are traps.** R6: K2 is fast *because* the whole GPU works on 1–2 blocks at a time,
  which keeps the working set in L2. R5 and R6 also reject a lazy K2 inside K3.
- **Small-GPU defaults are wrong today** (R7, with real register counts):
  - K1 needs 72–96 groups on a 4060, not 128.
  - `GZC_K3_BPW=2` should be the NVIDIA default (48 vs 24 warps per SM on Ada).
  - K4 and K5 are limited by workgroup memory and registers.
  - Heaviest-first block ordering recovers the tail on multi-wave cards (R8, R10).
- **CUDA is worth a time-boxed proof of concept at most.** Its unique wins are `__match_any_sync` for K1 and
  register control for K3. Decide after the Vulkan multi-queue result (R2). HIP: no.
- **The biggest *ratio* lever is the frame format, not the parse** (R10):
  - One frame per file with a sliding window over the previous block gives +2.0 % (128 KiB window) to +4.7 %
    (1 MiB), at L9 parameters.
  - Parsing stays block-parallel, because the history is known input.
  - Spent as speed, rung1 plus a window still beats L9.
  - This changes the output contract: blocks are no longer independently decodable. **That is the user's
    decision.**
- **Domain structure pays** (R9): DDS here is 69 % DXT5 and 31 % DXT1, with no BC5/BC7. DXT1 matches sit on the
  8-byte grid, and grid-aligned search *raises* DXT1's ratio 1.374 → 1.391 at ~0.08× the chain steps. A
  per-format preset gives ratio 1.3582 with ~⅓ of the K2 work.
- **Dead ends, measured:**
  - Adaptive per-block level (R10).
  - Pre-classifying blocks (R8: 0 RLE blocks, 0.04 % incompressible).
  - Position-parallel parsing with pointer jumping (R4: breaks rep codes and falls below L9).
  - Optimal-parse-lite with one candidate per position (R4: no gain).
  - Suffix-array and binary-tree finders (R5).
  - Hand-written SPIR-V for speed alone (R3: ≤ 5 %).

## Ranked experiment list

| # | Experiment | Source | Output | Est. gain (5090 / 4060) | Effort |
|---|---|---|---|---|---|
| E1 | **Segmented parse** (4 KiB segments, one lane each; new preset) | R4 (R1, R8) | new preset, ratio ≥ L9 | K3 31 → 2–7 ms; **+40–70 % e2e** / K3 ~105 → 6–15 ms | L |
| E2 | **Bucket-sorted candidate array**: K1 counting sort (13-bit key, workgroup memory), K2 window read (new preset; a byte-identical 16-bit-key variant as fallback) | R5 (R1) | new preset, ratio ≥ L9 | K1+K2 26 → ~10 ms; **+30 %** / large | L |
| E3 | **Multi-queue via wgpu-hal**: async compute queue (K3 of batch i ∥ K1/K2 of batch i+1) + transfer queue for copies | R3 (R2) | identical | **+20–30 %** / +25–35 % | M–L |
| E4 | Depth-16 lvl9 preset | R5, R10 | new preset (−0.02 %) | K2 −25 % (~+3 %) | S |
| E5 | Small-GPU defaults and proxy: K1 groups from residency, BPW=2 on NVIDIA, heaviest-first order, `--proxy 4060` | R7, R8, R10 | identical | 5090 ~0 / 4060 +25–40 % on K1/K3 | M |
| E6 | Share buffers by lifetime (bigger batches) | R6 | identical | +6–9 % / ~0 | S |
| E7 | K5 residency / serial-chain fix | R1, R7 | identical | +5 % / +10–15 % | M |
| E8 | Skip the upload copy (read the ReBAR upload buffer directly) | R1, R6, R8 | identical | +3 % / +1–8 % | S |
| E9 | WGSL bounds checks off (unchecked shader modules) | R2 | identical | unknown, free test | S |
| E10 | ~~Multi-block frames with a sliding window~~ | R10 | **rejected by user**: blocks stay independent and ≤ 64 KiB | – | – |
| E11 | CUDA K3 proof of concept | R2 | identical | 1.1–1.8× K3 (after E1, likely moot) | M |
| E13 | **Per-format preset `lvl9dds`**: selected per block from the DDS header. DXT1 hashes/searches only phase {0,4} mod 8 at depth 2; DXT5 depth 8; NIF depth 8 | R9 | new preset, **ratio 1.3582** (> lvl9) | K2 walk 0.29–0.38×, K1 inserts 0.78× (≈ −13–15 % GPU) | M |
| E12 | Optimal parse beyond L9 (btopt-class) | R8 | new preset | +3 % ratio (L19), free on 7z-bound installs | L |

## Execution plan

Parallel tracks on branch `speed2` touch disjoint files. Each is measured with the same protocol and reviewed:

- **Track A (K3):** E1 segmented parse.
- **Track B (K1/K2):** E2 bucket-sorted finder, then E4, then E13 (per-format preset; same kernels).
- **Track C (host/queues):** E3 multi-queue + E8 + E9.
- **After A–C merge:** E6, E7, E5 (E5 needs the proxy mode; the defaults touch `chains.rs` after B).
- **Rejected (user, 2026-09-30):** E10. Blocks stay independent and **at most 64 KiB**. From now on, block-64k is
  the default and the primary benchmark. The ratio target is libzstd L9 on 64 KiB blocks: **1.3379** on the full
  corpus (DDS 1.3277, NIF 1.6016); our CPU lvl9 at 64 KiB gives 1.3393.
- **Deferred:** E11, E12.

New presets are combined at the end into one fast preset. E1+E2 (+E4) is checked for ratio ≥ L9 on the full
corpus.
