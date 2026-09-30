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
- **Equality and decoding:**
  - `gzc-bench ref` and `gzc-bench gpu` report the same total size at 64K (4 848 642 982 bytes).
  - Each engine's `--verify` passed. It round-trips that engine's own frames through libzstd; it does not compare the two
    engines' bytes.
  - Byte identity was checked per frame: at 64K, `corpus_blocks_match_cpu_per_preset` (`GZC_CORPUS_BLOCKS=4000
    GZC_CORPUS_PRESETS=lvl9seg`) compared GPU and CPU frames on 4 000 real blocks from 3 172 files, and all were equal.

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

## E9 + E8 + E3: unchecked shaders, ReBAR upload, transfer queue (branch of `speed2` @ 265cec2 + 517f426), 2026-09-30 07:47–10:20

Full report: `.superpowers/speed2/e3-report.md`.

- **E9:** K2, K4 and K5 are built with `create_shader_module_trusted`, with bounds checks and loop bounding off and
  integer-division checks on. K1 stays checked, because it is 7 % slower unchecked at 64K.
- **E8:** kernels bind the slot's ReBAR upload buffer as `data`, so there is no upload copy and 320 MiB less VRAM.
- **E3:** the readback goes to a transfer-only queue, through wgpu-hal `device_from_raw`, ash buffers shared
  `CONCURRENT` between the families, and timeline semaphores.
- The async-compute overlap was measured and not integrated (see below).

64K, b5118 i3, one quiet window (load average about 1), three runs each:

| Preset / step | E2E MB/s (3 runs) | Median | K1 | K2 | K3 | K4 | K5 | Sum ms/b | Upload / readback ms/b | Ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lvl9seg base (517f426) | 7048.8 / 7047.8 / 7037.1 | 7047.8 | 13.84 | 8.73 | 6.43 | 2.01 | 4.61 | 35.60 | 2.47 / 6.57 | 1.339 |
| + E9 | 7286.2 / 7307.8 / 7302.9 | 7302.9 (+3.6 %) | 13.85 | 7.95 | 6.43 | 1.59 | 4.22 | 34.04 | 2.49 / 6.50 | 1.339 |
| + E8 | 7727.9 / 7721.1 / 7698.5 | 7721.1 (+9.6 %) | 13.87 | 8.05 | 6.44 | 1.58 | 4.14 | 34.03 | 0.00 / 6.57 | 1.339 |
| **+ E3 transfer queue** | 9081.7 / 9123.6 / 9120.3 | **9120.3 (+29.4 %)** | 13.89 | 7.96 | 6.41 | 1.63 | 4.15 | 33.93 | 0.04 / 0.00 | 1.339 |
| lvl9 base (517f426) | 5414.9 / 5571.2 / 5363.2 | 5414.9 | 13.87 | 8.78 | 20.14 | 1.96 | 4.62 | 49.54 | 2.46 / 6.46 | 1.339 |
| + E9 | 5571.4 / 5526.7 / 5480.0 | 5526.7 (+2.1 %) | 13.86 | 8.00 | 20.59 | 1.59 | 4.20 | 48.32 | 2.46 / 6.40 | 1.339 |
| + E8 | 5939.4 / 5852.2 / 5811.4 | 5852.2 (+8.1 %) | 13.85 | 7.96 | 19.73 | 1.59 | 4.24 | 47.35 | 0.00 / 6.49 | 1.339 |
| **+ E3 transfer queue** | 6594.9 / 6525.3 / 6461.2 | **6525.3 (+20.5 %)** | 13.87 | 7.99 | 20.40 | 1.58 | 4.23 | 47.97 | 0.03 / 0.00 | 1.339 |

Notes:

- lvl9's K3 varies from 18.6 to 26.1 ms/batch between runs. A second lvl9 pass gave medians of 5350 / 5340 / 5408 /
  6631.
- The lvl9seg compressed size equals E1's (4 848 642 982 bytes).
- With E8 the footprint is 5824 MiB at b5118. `gzc-bench --batch max` now uses `vram_bytes_with(ctx.direct_upload)`,
  so it resolves to b5403.
- 128K lvl9 b2559 (before the E1 merge): 4241, then 4411 with E9 (+4.0 %; K1 was still unchecked then), then 4547
  with E8 (+3.1 %).

Async compute, measured with `multiqueue::tests::overlap_probe`. K3 of set A runs on the main queue and K1+K2 of set B
on the async-compute family 2. The table gives concurrent time over the serial sum:

| Case | K3 submitted first | K1+K2 submitted first |
|---|---:|---:|
| 128K lvl9, n = 2559 | 0.99–1.17 | 0.73–0.81 |
| 64K lvl9, n = 3900 | 1.32 | 0.62–0.76 |
| 64K lvl9, n = 3900, K2 then K1 (pipeline order) | 1.00–1.19 | 1.02–1.13 |
| 64K lvl9seg, n = 5118 | 1.11 | 0.87 |

A second family-0 queue gives 1.00 in every case. Queue priorities and swapping the queues do not change the picture.
The criterion was ≤ 0.75, so the compute overlap was not integrated.

Transfer queue, measured with `transfer_probe` at 64K, n = 3900. A 255 MB readback copy takes 4.72 ms. K3 plus the
copy takes 16.28 ms against 16.20 for K3 alone, and K1+K2 plus the copy takes 18.07 ms against 17.95 alone, so the
copy overlaps fully. It was integrated.

Follow-up: the pipeline thread's upload write is split over 4 threads, which takes it from 20 to 14 ms/batch. End to
end is unchanged because the run is GPU-bound again; it is kept as headroom.

## E2 + E4: bucket-sorted finder (`lvl9s13`, `lvl9s12`) and depth 16 (`lvl9d16`), 2026-09-30 07:45–10:25

The E2 branch is based on `speed2` @ 265cec2, with E1 (517f426) merged. The full report is
`.superpowers/speed2/e2-report.md`.

New oracle parameter: `MatchParams.hash_bits`, where key = `hash16 >> (16 - hash_bits)`. `find_best` still walks
chains over the key.

The GPU builds a per-block bucket-sorted candidate array in two steps:

- **K1:** a counting sort in workgroup memory, one 32-lane subgroup per block. It uses 16-bit counters at 64K, so
  12 bits need 8 KiB and 13 bits 16 KiB. It ranks positions into `best`, then a block-major scatter places them into
  `pred`. A workgroup-memory version covers devices without subgroups of at least 32 lanes.
- **K2:** one thread per slot. It stages the window and the keys in workgroup memory, then walks the entries below
  its slot that share its key. That walk is exactly the chain.

All new presets are byte-identical to the oracle.

Full corpus dds+nif, 64K, `--batch max` (b5118) i3, RTX 5090. Median of 3 runs, taken under load average 7–8 from
other agents, with every run gated on no other gzc-bench/test process. Compare rows within this table; the
lvl9/lvl9seg baselines ran 10–13 % below the controller's numbers here.

| 64K preset | E2E MB/s (3 runs) | Median | K1 | K2 | K3 | K4 | K5 | Sum ms/batch | Ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lvl9 | 4782 / 4656 / 4689 | 4689 | 14.08 | 9.54 | 26.62 | 2.31 | 5.43 | 58.21 | 1.33932 |
| lvl9s13 | 5108 / 5462 / 5541 | 5462 (+16 %) | 8.42 | 9.69 | 23.60 | 2.23 | 4.94 | 48.79 | 1.33929 |
| lvl9s12 | 5277 / 5453 / 5121 | 5277 (+13 %) | 6.17 | 10.92 | 24.89 | 2.31 | 5.38 | 49.66 | 1.33927 |
| lvl9d16 | 4693 / 5376 / 5281 | 5281 (+13 %) | 13.99 | 6.43 | 23.50 | 2.16 | 4.90 | 50.73 | 1.33902 |
| lvl9seg | 6606 / 6699 / 6526 | 6606 | 14.09 | 9.48 | 6.62 | 2.17 | 5.31 | 37.60 | 1.33931 |
| lvl9s13seg | 7046 / 7556 / 7749 | 7556 (+14 %) | 8.59 | 9.87 | 6.64 | 2.05 | 4.99 | 32.14 | 1.33927 |
| **lvl9s12seg** | 7145 / 7860 / 7705 | **7705 (+17 %)** | 6.11 | 10.95 | 6.64 | 2.12 | 5.04 | 30.80 | 1.33926 |
| lvl9d16seg | 6541 / 7117 / 6914 | 6914 (+5 %) | 14.14 | 6.84 | 6.61 | 2.27 | 5.65 | 35.51 | 1.33901 |
| **lvl9s12d16seg** | 7515 / 8383 / 8044 | **8044 (+22 %)** | 6.09 | 8.71 | 6.59 | 2.08 | 4.94 | 28.42 | 1.33860 |

The percentages compare against lvl9 for the unsegmented presets and against lvl9seg for the segmented ones. The
unsegmented rows are noisy because the coop K3 swings between 23 and 28 ms/batch from run to run.

Ratio checks:

- **Floor:** libzstd L9 at 64K is 1.3379, and every preset is above it. The smallest margin is lvl9s12d16seg at
  +0.05 %.
- **Totals:** the GPU byte totals equal the oracle's, both from `gzc-bench ref` and from `examples/e2_ratio`. For
  example, lvl9s12seg is 4 848 823 883 bytes in both.
- **Decoding:** `--verify` passed for all 7 new presets.
- **Per frame:** `corpus_blocks_match_cpu_per_preset` compared 4000 real blocks and found every GPU frame equal to
  the CPU frame.

Oracle key/depth grid at 64K, full corpus (hash_bits:depth → ratio):

| hash_bits:depth | 16:32 | 16:16 | 13:32 | 13:16 | 12:32 | 12:16 | 12:48 | 11:32 | 11:48 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ratio | 1.33932 | 1.33902 | 1.33929 | 1.33893 | 1.33927 | 1.33861 | 1.33942 | 1.33911 | 1.33940 |

With segmentation (seg 12), the ratio barely moves: lvl9seg 1.33931, 13:32 1.33927, 12:32 1.33926, 12:16 1.33860.

Other configurations:

- **128K** (one run, b2559):
  - E2E: lvl9 4143, lvl9s12 4421, lvl9d16 4299, lvl9seg 6039, and lvl9s12seg 7183 (+19 %) MB/s. K1 drops from 15.95
    to 7.88 ms.
  - lvl9s13 needs a 32 KiB table there, over the 16 KiB default limit. It therefore runs the chain fallback: 3580
    MB/s, with K2 at 22 ms.
  - Ratios: s12 1.35468, s12seg 1.35456, d16 1.35451.
- **No subgroups** (`GZC_NO_SUBGROUPS=1`, 64K, one run):
  - E2E: lvl9seg 5658, lvl9s12seg 6950, lvl9s12d16seg 7283 MB/s.
  - K1: 24.6 ms for the chain kernel, 12.6 ms for the workgroup-memory sort.
  - Before the workgroup-memory sort existed, the chain fallback for 12-bit keys ran at 3533 MB/s, because K2 took
    30.7 ms.
- **VRAM:** unchanged, with the same batch sizes. The unused 64 MiB `head` buffer could be dropped for about +1 %
  batch.

Rejected, with numbers:

- **K2, one thread per position walking 32 raw window entries** (128K): 52.8 ms/batch.
- **K1 scattering straight from the sort kernel** (128K, 13-bit): 38–48 ms. The cause was DRAM write amplification
  with roughly 1000 blocks in flight; the same kernel without the store took 9.6 ms. The fix was rank + block-major
  scatter, which brought K1 to 12.2 ms at 128K.
- **Single-pass ranking with a global start table** (64K): s13 9.2–9.5 and s12 6.8–7.1 ms, against 8.1–8.7 and
  5.8–6.2.
- **16-bit packed slots:** no gain.
- **Prefetching 4 chunks ahead** (128K): 14.1 ms, against 12.2.
- **Hoisting a chunk's ballots** (128K): 17.2 ms, against 12.2.
- **K2 bucket bounds from a START bit or a bitmap:** 9.8 and 10.2 ms, against 9.2–9.8.
- **K2 eager loads of p:** no gain.

R5's K2 window model (6–7.5 ms at 128K) did not hold. The window K2 costs about the same as the chain K2 (16-bit:
7.9–8.9 against 8.7–9.3 at 64K), because the compares, not the pointer chase, set its cost. The gain comes from K1.
Option B (a byte-identical 16-bit radix sort) was not needed.

**Kept:** `lvl9s12seg` (+17 %) and `lvl9s12d16seg` (+22 %). `lvl9s13`, `lvl9s12`, `lvl9d16`, `lvl9s13seg` and
`lvl9d16seg` are kept as measured variants, for the controller to prune.

**E2 fix round 1:** the presets were pruned to `lvl9s12`, `lvl9s12seg` and `lvl9s12d16seg`. The rows for lvl9s13, lvl9s13seg, lvl9d16 and lvl9d16seg above record measured variants only. The no-subgroup K1 now fits its workgroup memory exactly: 12-bit keys fit at every block size, and at 128K only the subgroup version runs. At 128K, `lvl9s12d16seg` is 1.35159 on the full corpus, below the L9 floor of 1.3532, so only `lvl9s12seg` (1.35456) qualifies at 128K.

## E3 fix round 1 + merge with E2 (2c18c75): headline numbers, 2026-09-30 11:03–11:07

Setup:

- 64K blocks, `--batch max`, i3.
- Runs alternated between `speed2` @ 2c18c75 (b5118) and the E3 branch (b5403).
- Each run was gated on an idle GPU; load average was 2–7.
- Every E3 run used the direct upload and the transfer readback.

| Preset | speed2 2c18c75 (3 runs) | Median | + E9/E8/E3 (3 runs) | Median | Change | Ratio |
|---|---|---:|---|---:|---:|---:|
| lvl9seg | 7009.7 / 7061.2 / 7063.9 | 7061.2 | 9154.4 / 9176.9 / 9138.7 | **9154.4** | +29.6 % | 1.339 |
| lvl9s12seg | 8085.5 / 8240.1 / 7985.8 | 8085.5 | 10535.9 / 10840.7 / 10742.5 | **10742.5** | +32.9 % | 1.339 |
| lvl9s12d16seg | 8308.8 / 8535.1 / 8435.0 | 8435.0 | 10821.1 / 10338.9 / 10699.4 | **10699.4** | +26.8 % | 1.339 |

Kernel sum per block, from the median run:

| Preset | speed2 (µs/block) | E3 branch (µs/block) |
|---|---:|---:|
| lvl9seg | 6.92 | 6.59 |
| lvl9s12seg | 5.73 | 5.38 |
| lvl9s12d16seg | 5.33 | 4.96 |

- `--verify` passed for all three presets.
- The compressed sizes equal speed2's: 4 848 642 982, 4 848 823 883 and 4 851 213 203 bytes.
- `k2_window` is now also built unchecked (−2.5 %).
- **Metric note:** with the transfer readback, `gpu_readback` measures only the main queue's share (K4's end to the
  batch's end marker), which is about 0. The copy itself runs on the transfer queue and is not timed.
