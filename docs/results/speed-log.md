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
