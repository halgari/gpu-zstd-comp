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
