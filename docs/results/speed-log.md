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
