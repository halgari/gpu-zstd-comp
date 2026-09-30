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
