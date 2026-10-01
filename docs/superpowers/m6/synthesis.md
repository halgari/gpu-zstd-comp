# M6 research synthesis: making opt16 (L16 ratio) faster

Date: 2026-09-30. There are ten Opus research reports in this directory (a01–a10), with artifacts under `artifacts/`.

**Measurement caveats:**
- All GPU timings were taken under contention (load 20–140, with 3–8 other agents on the 5090). They are interleaved ratios against an unmodified baseline.
- Ratios are CPU-oracle measurements. "Full" means the whole corpus (6.49 GB) and "sample" means the 1/50 sample.

## Baseline (M5)

opt16 runs at 44.7 µs/block, or 1426 MB/s at b2900. At `--batch max` (b3586) it runs 57 µs/block (1090 MB/s).

| Stage | µs/block |
|---|---:|
| K1 | 6.9 |
| K2opt | 2.6 |
| K3 (4 passes, each 8.2–9.2) | 33.3 |
| K4 + K5 | 1.9 |

Ratio at 64 KiB is +0.032 % over L16; at 16 KiB it is +0.015 %.

## A. Exact speedups (byte-identical to today's oracle; no ratio risk)

| # | Change | Source | Measured effect on K3 | Notes |
|---|---|---|---|---|
| A1 | **Register diet**: pack records into 1 word, derive lane constants, drop live epilogue args. Plus **u16 LL/ML/OF price tables** | a02 I1+I2, a03 v_mlc2/3, a01 (three agents converged on this) | −4…−8 % at b2900; −11…−12 % per cheap pass in throughput regime; −31…−36 % at b3586–4080 (one wave of ~3900–4080 instead of 3230) | Gate: ≤ ~88 regs on **every** pass kernel, smem ≤ 4266 B. v_mlc3 broke the private-ring fallback, which must be fixed |
| A2 | **Divergence**: rep-length memo (L−d), merged lockstep rep extension (cap 36 + exact tail), flat phase-2 emit | a01 P1/P2/P5 | −12 % over the 4 opt16 passes (0.880); 0.78–0.83 past one wave | Needs A1's register room. Control-flow divergence drops from 1.31–1.37× to 1.09–1.15×, so little branch gain is left |
| A3 | **Dead-position skip**: 61 % of positions have no earlier same-3-byte prefix. Also fold literal-only positions in cheap passes | a09 | −10 % (skip), −13 % (with fold) | Dead-run lengths are host-computed in the test. Building them in K2 is estimated at < 0.3 µs/block |
| A4 | **Persistent-workgroup K3 + heavy-first order** (atomic block counter; proxy = positions with longest candidate 3..32, ρ 0.925) | a07, a08 | b3586: 51.1 → 31.1 µs/block; neutral at one wave; 8 GB model −27…−32 % (−18…−23 % from heavy-first alone) | Root cause: the NVIDIA Vulkan driver runs a > 1-wave dispatch of the 93-reg kernel as synchronous waves (no backfill). Recompute `lane` in the loop to keep registers at 93 |

These overlap: A2 and A3 both remove rep-probe and trip work, and A1, A2 and A4 interact through occupancy. A plausible combined K3 cut is 30–40 % at one wave, and more at large batches and on small GPUs. This needs measuring after each port.

## B. Ratio levers (new oracle and preset), used to buy fewer passes

| # | Lever | Source | Ratio vs L16 (64 KiB full) | 16 KiB | Cost |
|---|---|---|---:|---:|---|
| B1 | **Multi-block frames**: split each 64 KiB frame into 2–4 zstd blocks, each with its own Huffman/FSE tables (repeat and treeless modes allowed). Still one frame per 64 KiB chunk; no cross-chunk references | a10 | opt14 parse + split: **+0.161 %**; opt16 + split: +0.211 % | +0.048 % (sample) | GPU split stage est. +1…4 µs/block; libzstd decode 4–6 % slower. Also +0.23 % for lvl9s12seg |
| B2 | **h10 chain**: a 10-byte hash at every 4th position, depth 16; h4 depth 32 → 8 | a06 | opt14 schedule: **+0.101 %** | +0.057 % (sample) | K1+K2 cost unchanged (K1 ×1.07, K2 ×0.75–0.79). Needs retrained priors |
| B3 | **In-pass price refresh**: cheap pass in 8–32 slices, rebuilding prices per slice, deterministically | a04 | B.Ew16.f (2 passes): **+0.084 %**; B.Ew32u16.f (1.5): +0.058 % | +0.050 % / +0.015 % | Barrier cost modelled at +5…13 % per cheap pass (needs a GPU measurement). No trained tables needed, which makes it robust to unseen mods |
| B4 | **gap3**: inner segments may start matches up to 3 B before the segment end (now 8) | a05 | +0.022 % | +0.016 % | One-line oracle change, two-line kernel change; GPU prototype exact |
| B5 | 3-pass opt16: Prior + 2 cheap + final | a05 | +0.023 % (+0.045 % with gap3) | +0.023 % | Preset-only; K3 −21.5 % (measured) |
| B6 | Final-pass `sufficient_len` 16 for DDS | a09 | spends margin: +0.024 % | +0.008 % | −4 % K3. Use only if margin is left over |

The levers are largely independent: B1 is entropy, B2 is candidates, B3 is prices and B4 is boundaries. They should mostly stack, but this has not been measured together. Each one alone already makes a 2-pass schedule clear L16 at 64 KiB.

## C. Dead ends (measured; don't revisit)

- Parallel or speculative DP: path-dependent, and the tail only falls to 0.59–0.80.
- Rep-history approximations: −0.1…−0.4 %.
- Subgroup-cooperative relaxation and subgroup shuffles.
- wg32: 13–17 % slower.
- A CUDA/HIP port: no lever beyond WGSL.
- `dot4U8Packed`.
- Data or candidates in shared memory: they don't fit.
- Fused passes: ≤ 2 %.
- Segment work queues: already balanced.
- Async compute overlap.
- Hybrid CPU DP: the CPU DP costs 1.5× libzstd L16. 8 cores give about 154 MB/s.
- One-pass L16: −0.14 % even with split. It reaches L14 only.
- Lazy or greedy price seeds.
- BC-aligned search: −0.3…−2.3 %.
- Routing blocks to lvl9.
- Price-convergence early exit: < 0.6 % of blocks.
- Fused K1 (1.3× slower).
- Chain-depth cuts on today's recipe: these eat the margin.
- Suffix-array candidates: 3–5× K1 for ≤ 0.2 %.

## D. Where this lands (projected; one combined measurement needed)

- **5090:** a 2-pass L16-class preset (B1/B2/B3/B4 headroom) with A1–A4 gives K3 ≈ 2 × ~6 µs ≈ 12, K1+K2 ≈ 9.5 and K4/K5 + split ≈ 3–5. That is about 25–27 µs/block, ≈ 2.4–2.6 GB/s (1.7–1.8× today), sustained at `--batch max`.
- **4060-class:** about 0.35–0.5 GB/s, which is 3–4 Gbit, still not 10 Gbit.
- **1660 Super, M4:** about 0.1–0.2 GB/s.
- **10 Gbit on 8 GB cards** still means lvl9s12seg. With B1 split it gains +0.23 % ratio at a small cost.
- **Cost/benefit (a10):** opt16 saves 5.26 GB per 300 GB over lvl9s12seg; opt14→opt16 saves only 0.13 GB.
