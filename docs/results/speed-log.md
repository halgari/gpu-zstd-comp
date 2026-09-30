# Speed phase log

Protocol: `docs/superpowers/plans/2026-09-30-speed-phase.md`. Full corpus (`--ext dds,nif`, 6.49 GB), `--batch max --inflight 3`,
6144 MiB budget, RTX 5090, median of 3 runs. Output must stay byte-identical to the CPU oracle.

## S0 — Baseline (branch `speed` @ 4c029c2 = m4 8894816 + plan), 2026-09-30 00:05, load avg ~4 (two idle VMs, desktop)

| Build / preset | Config | End-to-end MB/s (3 runs) | Median | Ratio |
|---|---|---|---:|---:|
| M3 (c43eee2) lvl3 | b1388 i3 | 1545.7 / 1545.8 / 1543.3 | 1545.7 | 1.271 |
| M4 lvl3 | b1365 i3 | 1448.7 / 1444.2 / 1447.0 | 1447.0 | 1.271 |
| **M4 lvl9** | b1638 i3 | 1608.2 / 1605.1 / 1603.3 | **1605.1** | 1.355 |

lvl9 per-kernel ms/batch (median): K1 23.36 · K2 16.63 · K3 61.26 · K4 7.07 · K5 3.22 · **sum 111.7** (kernel-only 1817 MB/s; host/transfer ≈ 12 %).

The lvl3 regression from M3 to M4 (−6.4 %, back to back) is real, not noise. It is to be investigated in S1, which touches the
same buffers. Suspects: the larger `MAX_SEQS` seqs stride and the resulting readback/staging layout, or the move of
`MIN_MATCH`/`DEPTH` from global to per-Kernels constants.

## S1 — K2 rework + best[] packing + staging alignment (kept), 2026-09-30 00:10–00:50, load avg 1.5–3, another agent sometimes on the GPU (contended runs re-run)

Changes, each committed and measured separately (lvl9, full corpus, i3; the "b1638" rows keep the baseline batch):

| Step | Batch | E2E MB/s (median) | K2 ms/b | K3 ms/b | Kernel sum ms/b | Kernel total ms |
|---|---|---:|---:|---:|---:|---:|
| S0 baseline (this worktree) | b1638 | 1632.7 | 15.73 | 61.12 | 110.19 | 3526 |
| + K2 cap early-out | b1638 | 1696.3 | 10.94 | 61.17 | 105.61 | 3380 |
| + streaming capped compare (1 load/side/4 B) | b1638 | 1717.6 | 9.47 | 61.05 | 104.05 | 3330 |
| + p's first 8 bytes in registers | b1638 | 1727.0 | 8.66 | 61.14 | 103.33 | 3307 |
| + best[] packed to one u32 | b1638 | 1767.6 | 8.60 | 58.57 | 100.65 | 3221 |
| same, `--batch max` (VRAM freed) | **b1890** | 1845.3 | 9.80 | 58.91 | 109.66 | 3071 |
| + 256-byte-aligned staging regions | b1890 | **1981.2** | 9.86 | 58.95 | 109.60 | 3069 |

Final per-kernel ms/batch (lvl9 b1890): K1 27.95 · K2 9.86 · K3 58.95 · K4 9.45 · K5 3.56 · sum 109.6 (kernel-only 2116 MB/s).

| Preset | Before (S0) | After (S1) |
|---|---|---|
| lvl9 | b1638 1632.7 MB/s | b1890 **1981.2** MB/s (+21.3 %), ratio 1.355 (compressed bytes identical) |
| lvl3 | b1365 1461.3 | b1535 **1669.3** (+14.2 %; M3 was 1545.7) |
| rung1 | b1638 1978.6 | b1890 **2299.2** (+16.2 %) |

`--verify` lvl9: all 51216 blocks round-trip (1967.7 MB/s). Compressed bytes equal S0 for lvl9, lvl3 and rung1.

**M3→M4 lvl3 regression, explained.** M3 (c43eee2) and M4 run at the *same* batch give the same speed (M3 b1365 1462.0 / 1460.8 MB/s, M4 b1365 1461.3). The kernels did not regress. M4's larger `MAX_SEQS` (BLOCK_SIZE/4+1 instead of /5+1) moved lvl3's `--batch max` from 1388 to 1365, and the steady-state GPU time spent outside the kernels depends on the batch size in steps: 9 ms/batch at b1388 but 15 ms at b1365. It is 15–19 ms whenever the staging buffer's frames region starts at `frame_len_bytes(cap) = 4*cap`, which is only 4-byte aligned, and 4–10 ms when that offset is 16-byte (or better) aligned. The GPU→staging copy of about 200 MB into a destination that is only 4-byte aligned is that much slower. Aligning every staging region to 256 bytes gives 6–7 ms per batch at any batch size, which fixes lvl3 and also gains 7 % on lvl9.

Kept: all of it. Notes: batch size itself matters on the 5090 because K3 (one lane per block) and K4 are far from filling the GPU, so K3's time per batch barely grows with blocks. Freeing VRAM therefore turns directly into MB/s here, but not on a card whose warp slots K3 already fills.
## S2 — K1 rework (branch of `speed` @ 50d8710: 0c5019b + e511d6a), 2026-09-30 00:50–01:35, load avg 1.1–2.9

Diagnosis (K1-only microbenchmark, 1638 corpus blocks): K1 was **DRAM-bound**, not barrier-bound. One 256 KiB
head table per block (1638 × 256 KiB = 410 MB live) against 96 MB of L2. A one-warp-per-block ballot kernel was
*slower* (35 ms), because it kept all tables live. With 64 shared tables (16 MB), the same kernel took 3.9 ms.

What changed:
- **0c5019b.** New `k1_chains_sg.wgsl` (needs `SUBGROUP` + `IMMEDIATES` and subgroup sizes 32..=128):
  - a persistent grid of 256 workgroups, each reusing one L2-resident head table;
  - tag-stamped entries, so head is never cleared (the host clears only when the 2^15 tags run out);
  - 256-position tiles, with equal hashes matched by ballot bit-slicing inside 32-lane chunks and cross-chunk
    links through ballots published to workgroup memory;
  - 2 barriers per tile instead of 38.

  The old kernel is kept as the fallback (`GZC_NO_SUBGROUPS=1`).
- **e511d6a.** The fallback is persistent too, so head is capped at 256 tables (64 MiB) for both paths instead of
  256 KiB per chain. `--batch max` for lvl9 grows from 1638 to 1736 (lvl3: 1365 → 1519).

`pred` is identical: the chain tests run both kernels, and the differential tests pass on both paths at both block
sizes (plus the 2000-block corpus test). A `--verify` run passed.

Same-binary A/B (old kernel = `GZC_NO_SUBGROUPS=1` on 0c5019b), 3 clean runs each (no other GPU process), median:

| Build / preset | Batch | End-to-end MB/s (3 runs) | Median | K1 ms/batch | Kernel sum ms/batch (µs/block) |
|---|---|---|---:|---:|---:|
| old K1, lvl9 | 1638 | 1623.2 / 1625.3 / 1626.0 | 1625.3 | 23.47 | 110.90 (67.7) |
| 0c5019b, lvl9 | 1638 | 1879.7 / 1881.9 / 1879.3 | 1879.7 | 6.76 | 94.16 (57.5) |
| **e511d6a, lvl9** | 1736 | 1977.7 / 1973.8 / 1961.9 | **1973.8** | 7.99 | 98.05 (56.5) |
| e511d6a fallback, lvl9 | 1736 | 1817.9 / 1817.3 / 1817.3 | 1817.3 | 16.72 | 107.42 (61.9) |
| old K1, rung1 | 1638 | 1978.5 / 1979.0 / 1976.0 | 1978.5 | 23.39 | 88.64 |
| e511d6a, rung1 | 1736 | 2490.7 / 2484.1 / 2470.2 | 2484.1 | 8.01 | 75.61 |
| old K1, lvl3 | 1365 | 1462.1 / 1456.2 / 1457.6 | 1457.6 | 42.72 | 101.55 |
| e511d6a, lvl3 | 1519 | 2145.2 / 2140.5 / 2145.1 | 2145.1 | 12.74 | 73.06 |

lvl9 per-kernel at e511d6a (median run): K1 8.00 · K2 16.93 · K3 61.53 · K4 8.39 · K5 3.20. `--verify` run: 1961.1 MB/s.

**Kept.** lvl9 is +21.4 % end-to-end over the same-session baseline (+23 % over the S0 log), with the kernel sum per
block −17 %. rung1 is +25.6 % and lvl3 +47 %; lvl3 is now above M3's 1545.7. Tuning note: `GZC_K1_GROUPS` sets the
live-table count. 224 is ~0.5 ms faster for lvl9 and 256 is 1 ms faster for lvl3; on 32 MB-L2 cards ~64–96 should
be right (untested).

### S2 fix round 1 + S1+S2 merged (merge aa17791 of b7dc7db, fixes c2bcd03), 2026-09-30 01:40–01:55, load avg 1.0–2.1

Fix round 1 changes:
- K1's head is now per-dispatch scratch. Each workgroup clears its table in-kernel at the start of a dispatch, and tags
  are chain ordinal + 1, so there is no host tag state and no IMMEDIATES feature.
- The subgroup kernel self-tests at `ChainsKernel::new` and falls back on mismatch.
- Default live tables: 128 (32 MiB) for 32 MB-L2 target cards.

"S1+S2 merged", lvl9, 3 clean runs each (no other GPU process), median:

| Config | Batch | End-to-end MB/s (3 runs) | Median | K1 | K2 | K3 | K4 | K5 | Sum ms/batch (µs/block) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| default (128 tables), `--batch max` | 2026 | 2384.8 / 2390.2 / 2402.7 | **2390.2** | 13.20 | 10.71 | 59.22 | 9.99 | 3.48 | 96.60 (47.7) |
| `GZC_K1_GROUPS=256`, `--batch max` | 2026 | 2463.6 / 2477.8 / 2481.0 | 2477.8 | 9.80 | 10.60 | 59.23 | 9.64 | 3.61 | 93.00 (45.9) |
| default (128 tables), b1638 | 1638 | 2140.5 / 2140.5 / 2139.6 | 2140.5 | 10.83 | 8.75 | 58.83 | 7.02 | 3.00 | 88.40 (54.0) |
| fallback K1 (`GZC_NO_SUBGROUPS=1`), b1638 | 1638 | 2019.8 / 2017.6 / 2021.9 | 2019.8 | 16.42 | | | | | 93.92 |

- The 5090 cost of the 128-table default (the controller's ruling for 8 GB-class cards) against 256 is +3.4 ms/batch of K1,
  which is −3.5 % end-to-end (2390.2 against 2477.8 MB/s). `GZC_K1_GROUPS=256` recovers it on 96 MB-L2 GPUs.
- The in-kernel table clear per dispatch was not A/B-timed separately. It writes 32 MiB per dispatch (64 MiB at 256
  tables); K1 at b1638 with 128 tables is 10.8 ms, in line with the pre-fix microbenchmark at 128 groups (about 10–11 ms),
  so any cost is within noise.
- `--verify` of the merged default passed: 2383.2 MB/s.

## S3 — Subgroup-cooperative K3 (kept), 2026-09-30 00:45–03:10, load avg 0.5–2.9, other agents on the GPU at times (contended runs re-run; a "c" marks a contended run kept in the three-run list)

**Diagnosis, before building.** A CPU instrumentation of `lazy_parse` over all 51,216 corpus blocks shows that K3 was dominated by
literal scanning, not by flat-block extensions:
- Skip iterations per block: mean 73 K, max 121 K. Deferral visits: 17.5 K. Sequences: 8.5 K.
- The dependent-step cost of the sequential parse is narrow, from mean 245 K to max 290 K.

Measured on the GPU, with K3 on batches of one replicated block:
- The sequential kernel takes 50–59 ms on *every* kind of block.
- The cooperative kernel's time is set by sequence-dense blocks: 43 ms for a block with 20 K sequences.
- K3 is **per-warp latency-bound**: the dense block still takes 38 ms at 1 warp per SM.

A microbenchmark of WGSL dependent-op latencies on the 5090: L1 load 19 ns, L2 137 ns, `subgroupShuffle` +18 ns, `subgroupBallot` +32 ns.
Per-sequence time after the scan change is therefore set by the length of the op chain, not by memory round trips.

What changed:
- `k3_coop.wgsl`: one block per W lanes, where W = the minimum subgroup size (32 here).
  - Replicated parse state and uniform control flow.
  - Cooperative match_len (`subgroupMin`), catch-up, literal scan (W skip-sequence candidates per step), literal copy, and a deferral window taken from the scan lanes.
  - A cooperative greedy parse for lvl3/rung1.
- Lane probe; the sequential K3 as fallback (`GZC_K3_MODE=seq`, no subgroups, `GZC_NO_SUBGROUPS=1`).
- Both K3 modules are built without naga's forced loop bounding. Every K3 loop provably terminates; bounds checks stay on.
- Report: `.superpowers/sdd/2026-09-30-speed-phase/s3-report.md`.

Stages (lvl9, b1890, i3, before the S2 merge, median of 3):

| Step | E2E MB/s (3 runs) | Median | K3 ms/b | Sum ms/b | |
|---|---|---:|---:|---:|---|
| seq baseline | 1993.0 / 1984.4 / 1980.1 | 1984.4 | 58.75 | 109.60 | |
| A as designed | 1648.2 / 1662.5 / 1654.8 | 1654.8 | 81.32 | 132.81 | slower |
| A + uniform first-word probe | 1997.0 / 1998.8 / 2015.4 | 1998.8 | 57.81 | 108.61 | kept (base for B) |
| B cooperative literal scan | 2438.9 / 2429.0 / 2439.1 | 2438.9 | 37.88 | 87.72 | kept |
| C as designed (byte head/tail) | 2339.3 / 2085.4c / 2348.5 | 2339.3 | 41.60 | 91.67 | dropped |
| C′ one load per literal word | 2485.9 / 2474.1 / 2259.5c | 2474.1 | 36.11 | 86.30 | kept |
| D window | 2486.1 / 2488.8 / 2489.6 | 2488.8 | 35.62 | 85.81 | |
| + known first word, wide extension | 2493.7 / 2501.7 / 2513.7 | 2501.7 | 35.10 | 85.24 | |
| + window from the scan lanes | 1984.3c / 2516.8 / 2510.7 | 2510.7 | 34.59 | 84.98 | |
| + scan loads before the immediate probe | 2529.3 / 2507.3 / 2510.0 | 2510.0 | 34.55 | 84.97 | dropped |
| + `subgroupMin` in match_len | 2577.2 / 2574.4 / 2547.6 | 2574.4 | 32.81 | 82.66 | D kept |
| `subgroupMin` in catch-up | 2563.3 / 2558.0 / 2563.3 | 2563.3 | 33.28 | 83.16 | dropped |
| `subgroupMin` in scan + catch-up | 2553.7 / 2546.0 / 2553.1 | 2553.1 | 33.42 | 83.35 | dropped |
| E: 2 blocks / workgroup | 2552.6 / 1711.7c / 2227.9c | — | 33.34 | 83.32 | opt-in `GZC_K3_BPW=2` |
| no forced loop bounding | 2618.0 / 2613.8 / 2594.0 | 2613.8 | 31.50 | 81.31 | kept |
| peel first match step | 2608.1 / 2610.1 / 2597.3 | 2608.1 | 31.68 | 81.52 | dropped |

b1638 at stage B: seq 1891.4 / 1891.9 / 1892.7 (K3 58.51) → coop 2352.6 / 2353.3 / 2355.0 (K3 37.66).

Final, merged with S2 (d49d4d3 + bc7ff57), i3, median of 3. "seq + LB" is the pre-S3 K3 in the same binary.

| Preset / config | Batch | E2E MB/s (3 runs) | Median | K3 ms/b | Sum ms/b |
|---|---:|---|---:|---:|---:|
| lvl9 seq + LB | 2026 | 2403.2 / 2406.3 / 2406.5 | 2406.3 | 58.98 | 96.07 |
| lvl9 seq | 2026 | 2454.6 / 2460.2 / 2459.0 | 2459.0 | 56.70 | 93.73 |
| **lvl9 coop W32** | 2026 | 3278.8 / 3282.9 / 3268.4 | **3278.8** | **31.97** | **68.42** |
| lvl9 coop W16 | 2026 | 3266.8 / 3286.3 / 3279.5 | 3279.5 | 32.03 | 68.50 |
| lvl9 coop W8 | 2026 | 3254.0 / 3290.6 / 3228.8 | 3254.0 | 32.36 | 69.02 |
| lvl9 seq | 1638 | 2201.9 / 2203.5 / 2200.4 | 2201.9 | 56.35 | 85.73 |
| **lvl9 coop** | 1638 | 3029.4 / 3031.9 / 3032.5 | **3031.9** | 31.41 | 60.70 |
| rung1 seq + LB | 2026 | 2828.2 / 2874.6 / 2864.8 | 2864.8 | 46.64 | 79.46 |
| rung1 seq | 2026 | 3437.5 / 3471.7 / 3428.6 | 3437.5 | 32.50 | 65.01 |
| **rung1 coop** | 2026 | 4902.5 / 4852.3 / 4881.4 | **4881.4** | 12.01 | 43.35 |
| lvl3 seq + LB | 1736 | 2418.2 / 2416.6 / 2418.1 | 2418.1 | 46.48 | 82.79 |
| lvl3 seq | 1736 | 2887.9 / 2878.7 / 2886.0 | 2886.0 | 32.28 | 68.35 |
| **lvl3 coop** | 1736 | 4211.1 / 4220.3 / 4110.9 | **4211.1** | 8.82 | 44.71 |

lvl9 per-kernel ms/batch (b2026): K1 13.17 · K2 10.57 · K3 31.98 · K4 9.11 · K5 3.58 · sum 68.42 (33.8 µs/block, was 47.4).

`--verify` passed: lvl9 3260.7, rung1 4736.5, lvl3 4165.7 MB/s. Compressed bytes are identical in every mode, width and batch
(lvl9 4,792,885,250; rung1 4,834,508,359; lvl3 5,108,985,909).

**Kept.** Against the S1+S2 log:
- lvl9: 2390.2 → 3278.8 MB/s (**+37 %**); b1638: 2140.5 → 3031.9 (+42 %).
- Kernel sum: 96.6 → 68.4 ms/batch.
- rung1 +70 % and lvl3 +74 % against the same binary's pre-S3 K3.

E (2 blocks per workgroup) cannot help on the 5090: 32 resident workgroups per SM are never the limit. It stays opt-in for
Ada-class cards, where 24 resident workgroups per SM means about 3.3 K3 waves per batch. It is unmeasured there.

4060 view: the gain is a shorter dependent chain per block, not width or bandwidth (W8 ≈ W32; K3 reads < 5 GB/s). So it
should carry over roughly 1:1 per block: K3 about 190 → about 105 ms/batch for lvl9 at ~3.3 waves, and 2.5–3.5× for greedy.

## S6 — Transfer path (branch of `speed` @ 966a24f), 2026-09-30 02:00–03:30, load avg 0.9–2.9, runs gated on no other gzc process on the GPU

### Where the time outside the kernels goes (before S6: lvl9, b2026 i3, 26 batches)

New instrumentation (`PipelineStats::transfer_ms`, printed by `gzc-bench gpu`). Each submission is bracketed by two
empty marker passes whose timestamps are written at BOTTOM_OF_PIPE (i.e. after all earlier commands finish), and
the host thread is timed with `Instant`. `GZC_NO_TIMESTAMPS=1` turns all queries off. It gives the same MB/s
(2398.5 / 2410.6 / 2400.1 against 2397.8 / 2398.4 with timestamps), so the markers do not perturb the run.

| Item (median run) | ms/batch | Notes |
|---|---:|---|
| kernels K1–K5 | 96.2 | |
| GPU upload copy (upload → `data`) | 1.97 | 265 MB at ~135 GB/s: the MAP_WRITE upload buffer is device-local (ReBAR), so this is a VRAM → VRAM copy |
| GPU readback copy (`frames` → staging) | 5.27 | 266 MB at ~50 GB/s: PCIe 5 ×16 into host-cached memory |
| GPU idle between submissions | 0.03 | the queue never runs dry |
| host: upload memcpy + unmap | 12.8 | overlapped with GPU work |
| host: frame delivery (`to_vec` per frame) | 8.2 | overlapped |
| host: submit, upload-map wait, unmap | 0.3 | |
| host: fill (first upload) / drain | 13.3 / 2.7 ms once | 0.6 % of the run |

After S6 (lvl9 b2431 i3, 22 batches, median run 2579.9 MB/s):

| Item | ms/batch | Notes |
|---|---:|---|
| kernels K1–K5 | 105.33 | 43.3 µs/block against 47.4 before |
| GPU upload copy | 2.32 | |
| GPU readback copy | 5.98 | |
| GPU idle between submissions | 0.02 | includes the previous batch's timestamp resolve + copy |
| host: upload memcpy + unmap | 14.95 | overlapped |
| host: frame delivery | 9.57 | overlapped |
| host: submit, upload-map wait, unmap | 0.26 | |
| host: fill / drain | 15.6 / 0.6 ms once | |

Wall time is 114.4 ms/batch, which equals kernels + copies (113.6) plus fill/drain. The copies are still 7.3 % of
the time, all of it serial with the kernels.

Before S6, wall time per batch was 104.1 ms, which equals kernels + copies (103.5) plus fill/drain. **The run is GPU-bound and the
host is idle ~80 % of the time.** The whole "host overhead" is the two copies, which the single wgpu queue runs
serially with the kernels (7.2 ms/batch, 7 %).

### Sub-changes

| Step | Batch | E2E MB/s (3 runs) | Median | Kernel sum ms/b (µs/block) | Readback ms/b | Kept? |
|---|---:|---|---:|---:|---:|---|
| baseline 966a24f | 2026 | 2395.1 / 2405.4 / 2394.3 | **2395.1** | 95.98 (47.4) | 5.27 | |
| deferred readback: batch i's copies recorded between K2 and K3 of submission i+1 | 2026 | 2371.8 / 2405.0; no-ts 2386.8 / 2388.3 | ~2388 | 95.8 | 0 (K2→K3 gap 5.1–5.5) | no: the copy still serializes with K3 |
| pack kernel inside K3's pass (deferred, into the MAPPABLE_PRIMARY staging buffer) | 2026 | 2226.7 / 2320.4; no-ts 2325.7 / 2341.9 | ~2320 | K3 +8.4 | — | no: serialized with K3 |
| pack kernel inside K1's pass | 2026 | 2273.7 / 2253.5 | ~2264 | K1 +10.0 | — | no: serialized with K1 |
| **shared `data`/`frames`/`frame_len` across slots** (a slot keeps only its upload + staging) | **2431** | 2579.9 / 2586.0 / 2573.9 | **2579.9** | 105.33 (43.3) | 6.09 | **yes (+7.7 %)** |
| + `GZC_PACK=1` (pack after K4 into the staging buffer, 16-byte stores) | 2431 | 2557.1 / 2480.2 / 2488.1 | 2488.1 | 104.9–107.9 | 6.8–7.1 (+0.7 idle) | opt-in only (−1 to −3.6 %) |

rung1: baseline b2026 2869.9 / 2855.3 / 2853.7 (median **2855.3**) → S6 b2431 3077.0 / 3106.0 / 3099.9 (median **3099.9**, +8.6 %).
`--verify` lvl9 at S6: all blocks round-trip, 2569.1 MB/s. Compressed bytes equal the baseline (lvl9 4792885250, rung1 4834508359).

Informational, `--inflight 2`: with only upload + staging per slot, a slot costs ~0.25 MiB/block, so i2 fits b2701 and
reaches 2769.8 / 2767.0 / 2794.9 (median **2769.8**, +7.4 % over S6 i3) with the GPU still never idle (host work per
batch ~28 ms against ~120 ms of GPU). The baseline at i2 fits b2431 and gives 2575.8, the same as S6 at i3 with the
same batch: the S6 gain is the batch-size gain from freed VRAM.

Findings:
- **No copy/compute overlap on one wgpu queue here.** A readback copy recorded between K2 and K3 (whose barrier only waits
  on compute) still ran serially with K3. A pack dispatch placed in K3's or K1's pass right after the kernel's dispatch
  (so no barrier separates them: frames/frame_len are moved to read-only at the start of the submission, and staging's
  first use is a host → compute transition in the pre-pass) also ran serially: the pass grew by the pack's full time.
  This holds with and without timestamps. For K3 the likely cause is that its 1-lane workgroups fill every SM's
  registers (2026 WGs in ~1.5 waves), so a second grid only starts in K3's tail. Overlap would need a second queue,
  which wgpu does not expose, or S5/S7-style merging inside kernels.
- **Packing is slower than copying on this card.** The shader's 16-byte stores into host memory reach ~33 GB/s (196 MB of
  real frame bytes in ~5.9 ms), against the copy engine's ~50 GB/s for the full 266 MB fixed-stride region (5.3 ms).
  The first version, with 4-byte stores, took ~9 ms. Packing is kept behind `GZC_PACK=1` (tested; staging size is
  unchanged, since the worst case is still one raw frame per block) for measurement on a PCIe ×8 card, where both
  paths should be link-bound and packing sends 26 % fewer bytes.
- **Per-slot device buffers were redundant.** Each submission copies its own outputs to staging before the next
  submission's kernels start, and wgpu orders the upload copy after the previous K4's reads with a barrier. So `data`,
  `frames` and `frame_len` can be shared the way the scratch buffers already are. This frees 2 × 530 MB at i3, and
  `--batch max` grows from 2026 to 2431.
- The upload copy (2.3 ms/batch now) could go away if the kernels read `data` straight from the (ReBAR, device-local)
  upload buffer. It was not tried: without ReBAR, that buffer lives in host memory and every kernel would read over PCIe.

**`--inflight 2` observation (for the controller's protocol decision).** With slots now cheap, lvl9 `--batch max
--inflight 2` gets b2701 and **2769.8 MB/s** (2769.8 / 2767.0 / 2794.9), +7.4 % over S6 at i3. The protocol is still i3.

**`GZC_PACK` is an explicit exception to the keep/revert rule (controller ruling).** It is slower on the 5090 but kept
opt-in, to be re-measured on the PCIe 4.0 ×8 target card. `MAPPABLE_PRIMARY_BUFFERS`, a native-only feature, is
requested only when packing is asked for: `GZC_PACK`, or `GpuContext::with_options(_, true)` in its test. The
default path does not enable it.

**Kept:** the profiler, `GZC_NO_TIMESTAMPS`, the shared device buffers, and `GZC_PACK` as opt-in. **Reverted:** the
deferred-readback and in-pass-pack scheduling (commits 301fd30, and the hooks in compressor.rs/chains.rs, which are
not in the final diff).

4060-class carry-over. The shared buffers cut the footprint from ~3.03 to ~2.53 MiB per block at i3, so an 8 GB card's
batch grows by the same +20 %. That is worth less there, because K3 fills a 24–34 SM card's warp slots well before
~1150 blocks; the headroom is better spent on S5's per-slot scratch or on more batches in flight. The transfer share,
however, grows on such cards. A 4060 is PCIe 4.0 ×8 (~13 GB/s), so 128 KiB up and 128 KiB down per block cost ~20 µs
per block, against a projected 60–80 µs of kernel time: 20–25 % of the wall time, all of it serial with the kernels on
one queue. That is where `GZC_PACK` (26 % fewer readback bytes) should be re-measured, together with a non-ReBAR check of
the upload path.

### S3 fix round 1 + S1+S2+S3+S6 merged (a3a5708 fixes, a1aeb86 merge of cb26ed0), 2026-09-30, load avg 0.8–2.7, runs gated on an idle GPU

Fix round 1:
- `coop_push_lits` is wrap-safe. Before, a final push from an anchor past BLOCK_SIZE (only reachable with a best[] word
  that claims a match past the block end; K2 never writes one) wrapped `end - start` to about 2^32 bytes.
- Every `k3_coop.wgsl` loop carries a termination note (variant and bound), and the unsafe justification for building K3 without
  naga's loop bounding references them.
- `gzc-bench` prints the pipeline's actual K3 mode.
- New tests: an anchor past the block end (fails without the clamp); scans restarting mid-regime; a full literal region
  next to fast blocks; the in-kernel sequential fallback forced on (`GZC_K3_FORCE_FALLBACK=1`, test-only); the probe at 2
  blocks per workgroup.

**Stage E (`GZC_K3_BPW=2`) is kept opt-in, default off, as an explicit exception to the ≥ 3 % rule** (controller ruling). It
targets Ada's limit of 24 resident workgroups per SM and cannot be measured on the 5090. The full suite passes with it.

lvl9, `--batch max` (b2431), i3, median of 3. "seq" is the sequential K3 in the same binary (without loop bounding):

| Config | E2E MB/s (3 runs) | Median | K1 | K2 | K3 | K4 | K5 | Kernel sum ms/b |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seq K3 | 2642.3 / 2642.7 / 2666.8 | 2642.7 | 15.39 | 12.45 | 58.95 | 11.81 | 3.97 | 102.57 |
| **coop K3 (default)** | 3497.3 / 3526.7 / 3551.6 | **3526.7** | 15.37 | 12.56 | 32.61 | 9.95 | 4.14 | 74.62 |

`--verify`: 3481.5 MB/s, passed. Compressed bytes are equal in both modes (4,792,885,250).

### S4 — literals gathered from (data, seqs) (7b61477), 2026-09-30, load avg 0.6–2.0, every run gated on an idle GPU

What changed:
- K3 (sequential, lazy and cooperative) no longer writes literals: `push_lits` only counts, `coop_push_lits` and the
  `lits` binding are gone. `counts` stays (n_seq, n_lit).
- K5 now runs whenever frames are emitted and writes every literals section, Raw included (`HUFFMAN` false: Raw only).
  It gathers the literals from `data`. A workgroup prefix sum over groups of G = ceil(n_seq / 256) sequences per thread
  leaves each group's first literal index and block byte in 2 × 257 u32 of workgroup memory (2 KiB, for any n_seq up
  to MAX_SEQS). A thread seeks its first literal with a binary search plus a walk over at most G runs, then streams a
  contiguous literal range with a run cursor, loading an unaligned word when 4 bytes lie in one run (histogram, bit
  counts, the backward encode, the Raw copy).
- K4 no longer reads literals (no `lits` binding, no Raw copy, no `RAW_SECTION`).
- The `lits` buffer is removed: −128 KiB per block (−5 % of the ~2.53 MiB per block at i3), so `--batch max` grows
  from b2431 to **b2559**. The parse path reads back only `seqs` and gathers the literals on the host from the block
  (`decode_output` / `gather_literals`).
- K5 workgroup memory: ~9.3 → ~11.3 KiB (under 16 KiB).

lvl9 `--batch max`, i3, baseline (b2ee6c7) and S4 runs interleaved:

| Config | Batch | E2E MB/s (3 runs) | Median | K1 | K2 | K3 | K4 | K5 | Kernel sum ms/b (µs/block) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2431 | 3516.5 / 3496.2 / 3512.3 | 3512.3 | 15.37 | 12.51 | 32.62 | 10.37 | 4.01 | 74.88 (30.8) |
| **S4** | **2559** | 3684.6 / 3543.9 / 3676.6 | **3676.6 (+4.7 %)** | 16.08 | 13.04 | 30.86 | 9.75 | 4.75 | 74.48 (29.1) |
| S4, same batch | 2431 | 3627.7 / 3606.9 / 3639.8 | 3627.7 (+3.3 %) | 15.40 | 12.59 | 30.48 | 9.33 | 4.50 | 72.11 (29.7) |

(Per-kernel columns are from each row's median run.) Kernel time over the whole corpus: 1647 ms → 1564 ms (−5.1 %):
K3 −70 ms (−9.7 %), K4 −23 ms, K5 +11 ms. The middle S4 run (3543.9) had a noisy K4/K5 (11.2 / 5.8 ms/b).

rung1 `--batch max`, i3, interleaved: baseline 5084.6 / 5056.7 / 5083.0 (median **5083.0**, b2431) → S4 5176.8 / 5218.1 /
5247.5 (median **5218.1**, +2.7 %, b2559).

`--verify` lvl9: 3622.8 MB/s, every block round-trips. Compressed bytes equal the baseline (lvl9 4,792,885,250;
rung1 4,834,508,359).

Kept (+4.7 % lvl9, +2.7 % rung1, 5 % more blocks per batch).

Intermediate K5 versions (lvl9 b2559): byte-at-a-time gather 3558.9 (K5 6.9 ms/b); word loads in the histogram, bit
counts and Raw copy 3631.5 (K5 5.5); plus word loads in the backward encode (final) 3656–3677 (K5 4.5–4.75).

4060-class carry-over. The K3 gain (a store stream and the literal copy off the parse's critical path) should carry
over roughly in proportion, since K3 is the kernel least hurt by fewer SMs. K5's extra work (index scan, seeks) is
parallel across 256 threads per block and scales with bandwidth like the rest of K5; its share stays small. The
VRAM saving is worth more on an 8 GB card: 128 KiB per block is 5 % more blocks per batch, or headroom for
per-slot scratch.

### S9 — K4 sequence encode chunked and parallel (2e8a04f), 2026-09-30 04:28–04:36, load avg 1.2–1.6

Profile first (lvl9 b2559, K4 = 10.2 ms/batch, each phase disabled or run twice in a scratch build):

- the backward sequence encode loop on thread 0 took 9.3 ms (92 %). It was bound by global-load latency: 3 `seqs`
  loads, then dependent `tab` lookups, then bit writes, for each sequence;
- the full-block RLE check (every thread reads 1/64 of the 128 KiB block) took 0.46 ms;
- the FSE table builds took about 0.3 ms (sequential spread plus state-table loops, 3 tables of up to 512 cells);
- the histograms, mode choice, normalization, cost and ncount took about 0.25 ms together.

K4 is latency-bound: per batch it takes 1.0 ms at b320 and b640 (under one wave), 1.65 ms at b1280 and 2.3 ms at b2559.
The kernel time is set by the slowest blocks' serial path, not by bandwidth.

What changed (K4 only; the output is unchanged):

- **Chunked backward encode.** Chunks of 256 sequences, last chunk first:
  1. all threads compute the codes, from sequences held in registers and loaded one chunk ahead;
  2. threads 0..2 run only the FSE state transitions, one stream each, unrolled by four so the state-independent
     `tt` loads go first, and record each step's (value, nbits);
  3. all threads compute the per-sequence bit counts and a workgroup prefix sum (one barrier plus 16 `vec4` reads,
     replacing a 12-barrier Hillis–Steele scan);
  4. all threads place the bits with `atomicOr` into workgroup staging words and write the complete words to the
     frame; the partial last word carries into the next chunk.

  The Raw decision is the same: encoding stops once the sequences pass a Raw block's size.
- **Parallel FSE table builds.** Visit j of the spread walk lands on the j-th `t` whose `(t * step) & mask` is at or
  below `high`. Each thread takes a range of `t`, and a prefix sum is needed only when -1 symbols hold the top cells.
  Thread s then fills symbol s's state-table slots in increasing cell order.
- **RLE check with early exit.** It runs in rounds of 8 words per thread and stops after the first round that finds a
  byte different from the first.
- Workgroup memory: ~9.3 → ~14.9 KiB, still under 16 KiB.

Step by step, K4 ms/batch at b2559 (single runs): 10.2 → chunked encode 3.85 → chain unrolled ×4 2.85 → RLE early exit
2.38 → parallel builds 2.28–2.38 (sequential builds in the same kernel: 2.49) → barrier-light scan 2.16–2.25.

Tried and dropped:

- **Speculative segmented state chains** (16 segments per stream from guessed states, then a sequential fix-up until
  the true chain meets the recorded one): 3.8 ms. The tANS chains almost never meet within 16 steps, and a warm-up
  of 4 or 16 steps made no difference.
- **Per-stream parallel mode choice** (threads 0..2): no change.
- **Unrolled histogram loop**: no change.
- **Codes of the next chunk computed in step 4**, one barrier interval fewer: 2.8 ms, slower.
- **C = 128**, 3.2 KiB less workgroup memory: no change.

lvl9 `--batch max` (b2559), i3, base (a3c5f8b) and S9 interleaved. The per-kernel columns come from each row's median
run:

| Config | E2E MB/s (3 runs) | Median | K1 | K2 | K3 | K4 | K5 | Kernel sum ms/b |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | 3676.9 / 3679.5 / 3673.1 | 3676.9 | 16.12 | 13.07 | 30.83 | 9.86 | 4.65 | 74.54 |
| **S9** | 4048.5 / 4041.7 / 4033.7 | **4041.7 (+9.9 %)** | 16.10 | 12.96 | 30.86 | **2.28** | 4.74 | **66.94 (−10.2 %)** |

K4 over the whole corpus: 207–212 ms → 47–48 ms (−77 %).

rung1 `--batch max`, i3, interleaved:

- base: 5225.4 / 5275.0 / 5227.4, median **5227.4** (K4 9.5–10.1 ms/b, kernel sum 49.6);
- S9: 6024.1 / 5993.8 / 5989.7, median **5993.8** (+14.7 %; K4 2.24–2.30, kernel sum 41.7).

Below one wave (b640), K4 per batch goes 5.56 → 0.99 ms, so a block's serial path is ~5.6× shorter.

`--verify` lvl9: 3979.6 MB/s, and every block round-trips. A first `--verify` run measured 2140 MB/s while another
GPU job was running. Compressed bytes are identical: lvl9 4,792,885,250; rung1 4,834,508,359.

Tests: the full workspace suite passes at 128K and at 16K, with `GZC_PACK=1`, with `GZC_K3_MODE=seq`, and on the
ignored corpus test (300 MB, every preset). A new test, `k4_chunk_boundaries_match_cpu`, covers:

- n_seq 1 to 1025 around multiples of 256;
- 16-bit extra fields.

It catches a wrong chunk split and a corrupted carry word.

**Kept** (+9.9 % lvl9, +14.7 % rung1).

4060-class carry-over. K4 was, and still is, latency-bound per block. On a 24-SM card with 100 KB of shared memory
per SM:

- residency falls from ~9 to ~6 K4 workgroups per SM, because of the larger workgroup memory plus the 1 KiB per
  block that the driver reserves. That means ~12 → ~18 waves for 2559 blocks;
- each block's serial path is ~5.6× shorter.

The net should be ~3.5× less K4 time. Cutting workgroup memory, for example a u16 state table or aliasing
`hist`/`sp` with the staging words, would win back residency there. The only such cut tried here, C = 128
(−3.2 KiB), measured neutral on the 5090.
## S8 — K1/K2 second pass (branch of `speed` @ bc2d809; 468834a, 8d42f97, d153d8a, b4db3f5), 2026-09-30 03:15–04:40, load avg 0.8–3.3, runs gated on no other gzc process and an idle GPU ("c" = a run another agent's job overlapped anyway)

### What bounds K2

A CPU simulation of K2's walk over 513–2049 corpus blocks (lvl9) and GPU diagnostics, before any change:
- Candidates: 3.68 per position. 22 % are hash collisions (len < 4), 61 % stop at 4..7 bytes, 15 % at 8..63, 2 % cap.
- SIMT waste is large: a subgroup of 32 consecutive positions steps as long as its longest walk (16.1 steps against a
  mean of 3.7; 5.7 of 32 lanes active at depth ≥ 16).
- But K2's time follows the **number of candidates, not subgroup steps**. Capping the walk at depth d gives K2 0.85
  (d0), 3.2 (d1), 4.5 (d2), 5.9 (d4), 7.4 (d8), 9.1 (d16) and 12.5 ms (d32). One extra `pred` load per candidate in the
  *same* 32-byte sector costs +0.5 ms; one in *another* line costs +3.6–3.9 ms. So K2 is bound by the sectors it
  pulls into the SMs: about 1 for `pred[q]` plus 1.23 for q's data per candidate.

Consequently, the ideas from the dispatch that cut subgroup steps or add memory-level parallelism did not help
(lvl9 b2431, pre-merge, base K2 12.56 ms/b):

| K2 variant (byte-identical) | K2 ms/b | |
|---|---:|---|
| prefetch `pred[q]` before q's compare | 12.48 | no change |
| third q word loaded only after the first 4 bytes match | 12.55 | no change |
| two walks per invocation (p, p + 32), loads of both issued before either compare | 13.55 | slower |
| per-workgroup work queue (atomic counter, 4096 positions per workgroup) | 14.97 | slower: init/steps no longer coalesce |
| the same, static 4/16 positions per lane | 14.00 / 16.27 | slower |
| two phases: 4 lockstep steps, then leftover walks compacted in workgroup memory | 23.1 | much slower (1 run) |

### What changed (kept)

- **Fingerprinted pred words** (468834a, 8d42f97). K1 (both kernels) stores `pred[p]` as a word: bits 0..17 hold the
  predecessor (0x1FFFF = none), and bits 17..32 hold a fingerprint of p itself: 7 hash bits of bytes p..p+4 and
  byte p+4 (common.wgsl `pred_fp`, `pred_word`). K2 loads q's word anyway, for the next candidate. It now compares
  q's fingerprint with p's before touching q's data. The search result is the lexicographic max of (len, q) over the
  visited candidates, or none below MIN_MATCH. So K2 skips q when:
  - the lo bits differ: len < 4;
  - or byte 4 differs (len ≤ 4) and MIN_MATCH > 4, best_len > 4, or best_len == 4 with q < best_q.

  Skipped candidates still count toward DEPTH and can never be cap-length, so the walk and its early-out are unchanged.
  58 % of lvl9 candidates are skipped (22 % collisions, 36.5 % byte-4 mismatches). The `pred` buffer and VRAM are
  unchanged (u32 per position), so `--batch max` is unchanged. `ChainsKernel::run` decodes the words, and `run_words`
  returns them raw.
- **Next pred word loaded before the compare** (d153d8a). This is worth it only now that the compare depends on the
  pred load: K2 10.67 → 9.99 ms/b in the step runs below.

Rejected: a finer fingerprint (3 lo bits + byte 4 + a nibble of byte 5, which could skip another ~7 % of candidates)
gave the same K2 time (10.70 against 10.67 ms/b).

Step runs (not interleaved; lvl9 `--batch max` b2559, i3):

| Step | E2E MB/s (3 runs) | Median | K1 | K2 | K3 | K4 | K5 | Sum ms/b |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| merged baseline bc2d809 | 3698.4 / 3666.6 / 3686.3 | 3686.3 | 16.08 | 13.17 | 30.80 | 9.77 | 4.58 | 74.39 |
| + fingerprints | 3776.6 / 3772.6 / 3779.3 | 3776.6 | 15.85 | 10.67 | 30.82 | 9.84 | 4.71 | 71.88 |
| + next pred word first | 2548.0c / 3838.7 / 3836.6 | 3836.6 | 15.90 | 9.99 | 30.89 | 9.61 | 4.66 | 71.05 |

**Final, baseline and S8 binaries interleaved run by run** (i3):

| Preset / batch | Build | E2E MB/s (3 runs) | Median | K1 | K2 | K3 | K4 | K5 | Sum ms/b |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| lvl9 max (b2559) | bc2d809 | 3696.5 / 3640.3 / 3664.6 | 3664.6 | 16.14 | 13.10 | 30.87 | 9.97 | 4.70 | 74.78 |
| lvl9 max (b2559) | **S8** | 2199.4c / 3834.8 / 3851.6 | **3834.8 (+4.6 %)** | 15.88 | **9.84** | 30.88 | 9.74 | 4.60 | **70.95 (−5.1 %)** |
| lvl9 b2431 | bc2d809 | 3585.9 / 3597.8 / 3616.9 | 3597.8 | 15.40 | 12.51 | 30.50 | 9.91 | 4.58 | 72.90 |
| lvl9 b2431 | **S8** | 3764.7 / 3728.6 / 3762.1 | **3762.1 (+4.6 %)** | 15.19 | 9.44 | 30.52 | 9.58 | 4.58 | 69.30 (−4.9 %) |
| rung1 max (b2559) | bc2d809 | 5288.9 / 5259.1 / 5229.6 | 5259.1 | 16.14 | 7.51 | 11.36 | 9.70 | 4.61 | 49.32 |
| rung1 max (b2559) | **S8** | 5451.3 / 5450.7 / 5465.3 | **5451.3 (+3.7 %)** | 15.87 | 5.42 | 11.38 | 10.05 | 4.48 | 47.19 (−4.3 %) |

K2 −25 % (lvl9) and −28 % (rung1). `--verify` lvl9 (S8): every block round-trips, 3627.5 MB/s. Compressed bytes equal
the baseline (lvl9 4,792,885,250; rung1 4,834,508,359).

**Kept** (lvl9 +4.6 % end-to-end, kernel sum −5 %).

### K1: table groups, and where a tile's time goes (nothing kept)

K1 is a persistent grid, so its time is (rounds = ⌈chains / groups⌉) × (time per round). A round takes ~0.8 ms while
each SM holds at most one workgroup (≤ 160 groups on the 170-SM 5090), and ~0.95–1.1 ms beyond that. That is about
1.6 µs per 256-position tile, latency-bound with 8 warps per SM.

`GZC_K1_GROUPS` sweep, lvl9 `--batch max` (b2559, 2559 chains), final S8 tree, median of 3:

| Groups | Rounds | K1 ms/b | E2E MB/s |
|---:|---:|---:|---:|
| 96 | 27 | 21.08 | 3588.9 |
| **128 (default)** | 20 | 15.90 | 3828.2 |
| 160 | 16 | 13.00 | 3957.0 |
| 192 | 14 | 13.28 | 3939.8 |
| 224 | 12 | 11.41 | 4044.5 |
| 256 | 10 | 11.35 | 4049.6 |

On the 5090, 224–256 groups is +5.6 % end-to-end over the 128 default, and 160 (40 MiB of tables) gets +3.4 % of that.
The default stays at 128 (the controller's ruling for 32 MB-L2 cards). A 4060 (24 SMs, 32 MB L2) needs ≥ 4–6 workgroups
per SM to hide the tile latency, i.e. 96–144 groups, and ≤ ~96–128 tables (24–32 MiB) to leave L2 room for the data
and pred streams. So 96–128 is right there, to be measured.

Ablations of the subgroup kernel (diagnostic builds, output wrong on purpose, self-test bypassed, 1 run each, lvl9
b2431, K1 15.35 ms/b):

| Removed | K1 ms/b |
|---|---:|
| device-scope fence per tile (`storageBarrier` → `workgroupBarrier`) | 13.61 |
| the speculative `head` load | 13.97 |
| the `head` store | 12.78 |
| fence + load | 11.73 |
| cross-chunk matching (8 × 5 vec4 of ballots per chunk-first lane) | 11.27 |
| fence + load + store + matching | 6.75 |
| + 15 of the 16 hash-bit ballots | 4.37 |

Two restructurings from these, both byte-identical and both slower, so dropped:
- Pipelined head loads: tile k + 1's hash and head load are issued in tile k, right after its fence. The first lane
  of a hash also matches against the previous tile's published ballots. K1 went to 18.95 ms/b, because the extra
  matching costs more than the hidden load. The store still sits right before the fence.
- A per-tile uniqueness filter: two workgroup bitmaps (8192 slots, triple-buffered) let lanes whose hash is unique
  in the tile skip the cross-chunk matching. K1 went to 17.09 ms/b: the shared-memory atomics cost more than the
  matching they save on this corpus.

Fusing K1 + K2 (ideas-sonnet 5) was not tried. K2 is bound by the sectors of `pred` and data it reads, and fusion
still has to write and re-read `pred`. It would also run the whole K2 walk inside K1's 128 persistent workgroups,
about 1/20 of K2's parallelism. That is not a simple change.

4060-class carry-over:
- The fingerprints remove sector traffic, about a third of K2's per-candidate reads, rather than latency. The
  4060's K2 should be at least as transaction-bound (a smaller L2, and each batch's per-block working set of data
  plus pred is 640 KiB), so K2 should drop by a similar ~20–25 %.
- The pred-word prefetch is latency hiding, which helps more when fewer warps are resident.
- VRAM is unchanged.
