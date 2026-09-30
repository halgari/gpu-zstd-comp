# Speed phase 2 log

Protocol: `.superpowers/speed2/impl-common.md`. Full corpus `data/corpus --ext dds,nif` (6.49 GB), `--batch max
--inflight 3`, 6144 MiB budget, RTX 5090, three runs per row. Each timed run started only after 5 quiet seconds: no other
gzc-bench, GPU test, rustc, or process above 150 % CPU. Block size is 64 KiB (primary, per the user's constraint) unless
marked 128 KiB.

## E1: segmented parse, preset `lvl9seg` (branch of `speed2` @ 265cec2), 2026-09-30 07:57–08:42

`lvl9seg` is lvl9 with the parse split into independent 4 KiB segments:

- Match finding (K1/K2) is unchanged.
- Each segment starts with empty reps.
- Matches are clamped to the segment end, with no skip acceleration.
- `off_base` is then re-encoded with the block's true rep history.

K3 runs one lane per segment (`k3_seg.wgsl`), followed by a per-block fix-up dispatch.

| Block / preset | Config | E2E MB/s (3 runs) | Median | K1 | K2 | K3 | K4 | K5 | Sum ms/batch | Ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64K lvl9 | b5118 i3 | 5324.3 / 5373.9 / 5496.3 | 5373.9 | 13.87 | 8.73 | 20.86 | 2.05 | 4.62 | 50.13 | 1.339323 |
| **64K lvl9seg** | b5118 i3 | 7080.6 / 7067.8 / 7047.3 | **7067.8 (+31.5 %)** | 13.84 | 8.83 | **6.41** | 1.97 | 4.52 | 35.58 | 1.339309 |
| 128K lvl9 | b2559 i3 | 4209.4 / 4214.9 / 4241.0 | 4214.9 | | | 30.89 | | | | 1.354889 |
| **128K lvl9seg** | b2559 i3 | 6347.9 / 6356.0 / 6338.6 | **6347.9 (+50.6 %)** | | | **6.70** | | | | 1.354776 |

Ratio checks:

- **64 KiB:** libzstd L9 is 1.3379 (controller's number). lvl9seg is +0.105 % above it and −0.001 % below lvl9.
- **128 KiB:** L9 is 1.3532. lvl9seg is +0.12 % above it and −0.008 % below lvl9.
- **Equality and decoding:** `gzc-bench ref --preset lvl9seg --verify` at 64K produced exactly the GPU's bytes (4 848 642 982). `gpu --verify` at 64K passed.

Stages (K3 ms/batch):

| Stage | 128K K3 | 64K K3 | Notes |
|---|---:|---:|---|
| lvl9 (coop K3) | 30.9 | 20.9 | baseline |
| lane per segment (naive `k3_lazy` port) + lane-0 walk re-encoding every `off_base` | 14.6 | | fix-up alone 3.1 ms |
| + per-lane speculative encode, lane 0 re-encodes only until the true reps meet the lane's | 13.3 | | fix-up 2.2 ms |
| + the parse lanes store their own `off_base` (the speculative encode is free) | 12.1 | | fix-up 0.77 ms (0.53 at 64K) |
| + literal scan tests 8 positions per step | 7.1 | 6.67 | |
| + 32-lane workgroups (was 64) | 6.70 | **6.41** | kept |

Rejected (64K, K3 ms/batch, one run each unless noted):

- Scan width 4 gave 7.60, 12 gave 7.65 and 16 gave 9.14, against 6.67 for width 8.
- A 16-byte-per-step match extension gave 6.89 against 6.65.
- Reusing the scan's hit masks to skip deferral loads gave 6.61 against 6.67, which is neutral, so it was dropped.
- Segment size: 2 KiB gave 6.52 against 6.70 for 4 KiB, which is not worth the ratio. 8 KiB gave 11.39. At 128K, 1 KiB gave 10.30.
- Workgroup size: 64 gave 6.67 and 128 gave 6.65, against 6.40 for 32 (two runs each).

Where the rest of K3 goes (64K, temporary switches, output invalid): the fix-up dispatch takes 0.53 ms and the lazy
deferral about 2.2 ms. Catch-up and the immediate-repcode loop are negligible.
