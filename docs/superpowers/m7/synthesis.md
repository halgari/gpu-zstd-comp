# M7 research: GPU-friendly ways to reach libzstd-L16 ratio

Date: 2026-10-01. Six Opus agents worked on this, and their reports (r1–r6) are in this folder.

Unless marked otherwise, ratios are from the CPU oracle on the full corpus at 64 KiB, measured against libzstd L16 (1.37100). The baseline, today's `opt16`, uses 4 DP passes and scores 1.37144 (+0.032 %). It runs at 44.7 µs/block on an RTX 5090.

## Results

| Agent | Idea | Passes | vs L16 | GPU cost (model) |
|---|---|---:|---:|---|
| r1 | AWR: start from a lazy parse, then rerun the DP in independent 512 B windows, shifting the anchors each pass | 2 | +0.079 % | Gives 8–16× more lanes. K3 ≈ 17–19 µs/block |
| r3 | S3 candidates (h4 d8, h3 d4, sparse 6/10/12-byte chains), plus split | 1 | +0.076 % | ≈ 23–26 µs/block |
| r4 | h10 candidates, then a drop pass, rep re-optimisation and a shared-table split | 1 | **+0.237 %** | ≈ 24–28 µs/block |
| r5 | File-window statistics seeding one final pass, plus split (adding gap3 and h10 reaches +0.333 %) | 1 | +0.217 % | ≈ 1.9–2.0× opt16 |
| r2 | Rep lists from a parallel pre-pass | 4 | +0.021 % | K3 −9…−11 % |
| r2 | **Top-4 length pruning** | – | −0.0025 % | L2 pass about 12 % faster |
| r6 | Diagonal candidates: top-K offsets from a histogram, then scan those diagonals | 2 | +0.02–0.06 % on top of h10 | +0.3–1.3 µs/block |

Each of these produces standard zstd. Every frame was decoded with libzstd, and each lever has a byte-exact CPU-oracle design.

**Chosen for M6 Track B (B0):** `opt16p1`. It combines:
- the S3 candidates;
- a retrained prior;
- 1 DP pass;
- gap3;
- the drop pass;
- top-4 pruning.

It scores 1.37235 (+0.098 %) with no change to the frame writer.

**Held in reserve:**
- the frame split, worth +0.14–0.2 % on top;
- AWR as the next parse structure if K3 still dominates on 8 GB cards;
- file-window statistics.

## Dead ends (measured)

- **Exotic hardware (r6).** Tensor cores are reachable from WGSL, but only in f16, and a match test costs 32 MACs where one XOR does the job. Building RT-core acceleration structures costs 109–231 µs per block. DPX instructions are not in hardware on sm_120. NVENC and NVOFA are approximate.
- **Min-plus parallel prefix (r1).** Parse paths already merge within 64 B, so windowing gets the same result; the prefix version costs about 700× the work.
- **Cheaper literal-length pricing (r2).** Costs −0.02…−1.0 %.
- **Lazy parses (r4).** Even with every format-side gain added, they stay 1.15 % below L16.
- **GPU suffix array (r3).** Costs 2–3× K1+K2 and gains only +0.03 % over S3.
- **Cheap-parse statistics with a correction, and stratified sampling (r5).**
