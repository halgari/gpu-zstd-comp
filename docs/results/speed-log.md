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
