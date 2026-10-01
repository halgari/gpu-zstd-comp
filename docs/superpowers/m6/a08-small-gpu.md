# a08-small-gpu: opt16 on the cards people actually own

Agent a08, 2026-09-30. Research only: no repository file changed. The scratch hooks are in
`/tmp/claude-1000/m6/a08-small-gpu/proxy-hooks.diff`, the raw run logs in `.../runs/`, and the worktree has been removed.

## TL;DR

1. **I built a measured 4060 proxy on the 5090, and it fits the 5090's own numbers.**
   - A workgroup-memory pad caps K3opt residency per SM. I checked every level against wave boundaries: pads of 41456 / 28456 / 20756 / 12556 / 6832 B give 2 / 3 / 4 / 6 / 9 blocks per SM, and no pad gives 20.
   - With 3 blocks per SM the 5090 runs 510 blocks at once, against about 432 on a 4060.
   - A second measurement gives per-block latency against warps per SM, using one uniform block.
   - Combining the two predicts the 5090 at 9 and 20 per SM to within 10 % and 1 %. It also predicts opt16/opt14 on the 5090 at 1423/2240 MB/s, against 1426/2169 measured.
2. **Projections (opt16 / opt14, MB/s, kernel-bound, with heavy-first order):**
   - RTX 4060: **150–215 / 250–365**
   - RTX 3060: 80–170 / 130–290
   - RX 7600: 100–230 / 170–400 (wave32)
   - RX 6600: 85–190 / 140–330
   - Arc A750: 55–180 / 95–300
   - GTX 1660 Super: **40–105 / 70–185**
   - M4 Pro (20-core): 50–105 / 80–180
   - M4: 25–50 / 40–90

   Without heavy-first order (today's pipeline), opt16 is about 18 % lower.
   - **1 Gbit (125 MB/s):** opt14 reaches it on every discrete card except possibly the 1660 Super and Arc. opt16 reaches it clearly only on the 4060, and on the 3060 and RDNA cards only near the top of their ranges.
   - **10 Gbit (1250 MB/s): no small card, with either preset.** The best case (4060, opt14) is about 30 % of line rate. On a 3060 or 1660 Super, an 8-core CPU running libzstd L16 is roughly as fast as the GPU.
3. **Heavy-first (LPT) order is the biggest small-GPU lever.** At 4060-like concurrency it cut K3 time by **18–23 %** (sorted 125 against unsorted 150–163 µs/block), and the gain does not shrink between 4 and 8 waves. On the 5090 at one wave it gains nothing, which is why nobody has pushed it.
4. **Batch size stops mattering after about 4 waves once blocks are sorted.**
   - Measured: 2 waves cost 10–17 % more; 4, 5.7 and 7.8 waves were equal.
   - On a 4060 that is about 1800 blocks, or 3.1 GiB. **VRAM never binds.** A 6 GB card can budget 3–4 GiB without losing throughput.
   - **PCIe never binds either:** about 0.4 GB/s of traffic at the projected rates, against 6–13 GB/s for 3.0 ×8 to 4.0 ×8.
5. **K3 saturates per lane, not per warp instruction.**
   - wg32 (2 blocks per warp) gets 0.96× wg16's saturated throughput, even though it issues about half the warp instructions. So the limit is per-lane memory transactions, not ALU issue.
   - Two consequences:
     - (a) Ada/Ampere's half-width INT32 path costs less than feared: arch factor 0.7–1.0.
     - (b) **wg32 is not a small-GPU win on NVIDIA.** The perf study's "wg32 better in multi-wave" result was for the pre-T3b kernel.
6. **A 2-pass schedule that still meets the L16 gate** (a04's scratch results show `K.c.f` = 1.37116 at 64 KiB, +0.012 % over L16) would make "opt16-class ratio" cost what opt14 costs. That is **×1.6–1.7 on every small card.** This is the recommended small-GPU preset, pending a04's report.
7. **Portability blocker (from the ledger, not mine).** On Metal all presets currently corrupt output, probably from variable shifts ≥ 32, and AMD is likely affected too. None of the AMD, Arc or Apple numbers above mean anything until that is fixed.

## 1. Method and what was measured

### Contention

About 3–6 other gzc/k3opt processes were on the GPU during every run: probe, k3opt, differential and chains binaries from other agents. Utilisation was mostly 1–10 %, with spikes to 58 %. Each run's log line records the other processes present.
- **Every absolute number here was measured under contention.**
- I interleaved proxy and baseline runs, repeated each twice, and report ratios.
- The unpadded baseline at N = 2900 stayed at 34.0–34.5 µs/block (opt16 K3, 4 passes plus fix-up) across the session, against 34.07 on an idle GPU earlier. So contention moved the results by about 1–2 %.
- Samples that ran next to a heavy neighbour were re-run: the first r = 2 run (555 µs) and one G = 256 K1 run.

### Harness

This is `k3opt_passes_timing` from a detached worktree of 6fe7fa5 (master), with four scratch hooks:

| Hook | What it does |
|---|---|
| `GZC_K3OPT_PAD=bytes` | Adds `var<workgroup> gzc_pad` to `main_opt`, touched under `arrayLength(&data) == 0u`. vkstats shows the registers are unchanged (97–98) and shared memory grows by the pad. |
| `GZC_K3OPT_WG` | Sets the K3opt workgroup size. |
| `GZC_REPEAT` / `GZC_REPEAT_N` | Runs N copies of a single block. |
| `GZC_DUMP_ALL` | Dumps the composed WGSL, which the perf study's `w2s` and `vkstats` then turn into register counts. |

- Corpus: the harness's uniform-stride sample, N = 2900 unless stated (N = 1020, 2040 and 4000 for the batch sweep), 64 KiB blocks, opt16 unless stated. `GZC_K3OPT_SORT` is the heavy-first order.
- The K1 group sweep used `gzc-bench gpu --max-bytes 600000000` (9280 blocks, the first 157 files) with `--batch 2900`.

### vkstats (sm_120, current master K3opt)

| Kernel | Registers | Shared memory (B) | Resident per SM on the 5090 (measured) |
|---|---:|---:|---:|
| K3opt wg16 (all 3 kernels) | 97–98 | 4520 | 20 blocks (wave boundary between 3400 and 3570) |
| K3opt wg32 | 97 | 8848 | 11 WGs = 22 blocks (boundary between 3740 and 4080) |
| K3opt wg64 | 97 | 17504 | – |

### Residency calibration

I timed N = 170·k copies of block 1000 and looked for the step between waves:

| Pad (B) | 0 | 6832 | 12556 | 20756 | 28456 | 41456 |
|---|---:|---:|---:|---:|---:|---:|
| Blocks per SM | 20 | 9 | 6 | 4 | 3 | 2 |

### Latency against residency

Block 1000 is a typical block. L(r) is its per-pass latency with r blocks resident on every SM:

| r (warps per SM) | 1 | 2 | 3 | 4 | 6 | 9 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| L(r), ms | 10.0 | 11.3 | 11.7 | 11.9 | 12.5 | 13.2 | 17.5 | 18.9 | 19.8 | 21.2 |
| SM throughput r / L (blocks·pass per ms) | 0.10 | 0.18 | 0.26 | 0.34 | 0.48 | 0.68 | 0.97 | 0.95 | 0.96 | 0.94 |

The SM saturates at about 17 warps. Ada and Ampere sit at 16–19 warps, which is right at the knee.

### The proxy (corpus, heavy-first, 510 resident blocks ≈ a 4060's 432)

opt16 K3, 4 passes plus fix-up, µs/block, two interleaved repetitions:

| N (waves at 510) | 1020 (2) | 2040 (4) | 2900 (5.7) | 4000 (7.8) |
|---|---:|---:|---:|---:|
| sorted | 137–146 | 122–125 | 127.6 | 124.5–125.8 |
| unsorted | 149 | 148–151 | 155–174 | 157–163 |
| unpadded baseline (N 2900) | | | 34.2–34.5 | |

- opt14 at the same setting: sorted 65.5–65.9, unsorted 83–88, baseline 18.7.
- Proxy / baseline ratio: opt16 3.73, opt14 3.51.

### Model

```
K3(card) = P × (510 / C_card) × (L(r_card) / L(3)) / (f_card / 2.87 GHz) / a
```

- P is the proxy value at 510 blocks: 127.5 (opt16) or 65.7 (opt14).
- The 5090 sustained 2.87 GHz under K3 load (nvidia-smi sampled).
- C_card is the card's resident-block capacity, r_card its residency per SM, and a the per-SM, per-clock architecture factor.

Validation:
- At 9 per SM the model predicts 47.9 µs; the measured value is 51–55 (1.9 waves, so it includes a tail).
- At 20 per SM it predicts 34.6; the measured value is 34.1.

### Other kernels

**K1** time scales exactly as 1/G up to 170 groups: G = 24 / 48 / 85 / 96 / 128 / 170 give 82.6 / 41–47 / 22.3 / 21.0 / 16.0 / 12.5 ms per batch. Each SM holds one 256-thread workgroup, so the kernel is purely latency-bound.
- At G = 256 (about 1.5 per SM) it took 10.8 ms, against 8.3 for perfect scaling. That is a contention factor of 1.3 at 2 per SM.
- On a 4060 (96 groups, 4 per SM), I bracket the contention factor at 1.3–2.2. That gives 13–21 µs/block with G = 96.
- The default G = 128 is worse on a 4060: 32 groups run as a second round, about 1.5× slower, as R7 found.

**K2, K4 and K5** use the same residency per SM on both cards, so I scale them by SMs × clock:
- K2 × 7.08 / f / a: 20–28 µs.
- K4 + K5: 14–20 µs (latency-bound, R7's residencies).

The other kernels' total on a 4060 is 47–69 µs.

## 2. Per-kernel boundedness

| Kernel | 5090 (µs/block, opt16) | Bound by | Small-GPU scaling |
|---|---:|---|---|
| K1 (two chains) | 6.86 | **Latency**: the persistent groups' serial task chain (1/G exact). On cards with little L2, the atomics on the head tables (G × 256 KiB) also miss L2. | × (170 / G_card) × 128/170 × contention (1.3–2.2) / f. On the 3060 (3 MB L2) and 1660 Super (1.5 MB) the head tables live in DRAM: +0–80 %. |
| K2opt | 2.61 | Issue and L2 transactions at full occupancy (R7) | × SM × clock (≈ 7.5× on a 4060) |
| K3opt passes | 33.30 | Below 12 warps per SM: latency (divergent serial chains; L(1) = 10 ms per pass). At 17–20 warps per SM: saturated **per-lane memory transactions**. Not ALU, since wg32 gains nothing. Not DRAM: about 1.2 MiB per block-pass is 11 % of the 5090's DRAM at saturation, and about 7 % of a 4060's. | The model above. On the 5090 one wave is tail-bound (the heaviest block); on small cards it is throughput plus a tail, and heavy-first order removes the tail. |
| K4 | 0.60 | Latency; workgroup memory caps residency (6 per SM on Ada, 4 per CU on RDNA3) | × SM ratio (about 17.8 waves on a 4060) |
| K5 | 1.31 | Latency; registers cap residency (5 per SM) | × SM ratio (about 21 waves) |

**DRAM.** About 6–7 MiB per block per opt16 is a floor of 25 µs on a 4060 (272 GB/s) and 20 µs on a 1660 Super. That is less than 10 % of the projected time, so it never binds on discrete cards. The M4 (120 GB/s) floor is about 55 µs, against about 1300 µs of compute, so it doesn't bind there either.

**The small-L2 risk (3060, 1660 Super).** K3 keeps its ring payload in global scratch (6.3 KB per block), which only helps while that scratch stays in L1/L2.
- On a 3060, 448 resident blocks × 6.3 KB = 2.8 MB, against a 3 MB L2.
- On a 1660 Super, 242 × 6.3 KB = 1.5 MB, against a 1.5 MB L2, with only 32 KB of L1 per SM because shared memory takes 64 KB.
- I therefore widened a down to 0.5 (3060) and 0.35 (1660 Super). This is the one place where the 5090 proxy cannot see the effect, because it has 96 MB of L2.

## 3. Per-card projection (kernel-bound, heavy-first order, 64 KiB, µs per block → MB/s)

| Card | Units × clock (load) | K3 residency (limit) | Concurrent blocks | a | K3 opt16 | K3 opt14 | Other kernels | **opt16 MB/s** | **opt14 MB/s** | opt16 unsorted |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 (model check) | 170 × 2.87 | 20 (regs/smem) | 3400 | 1 | 35 | 18 | 11 | 1423 (meas. 1426) | 2240 (meas. 2169) | – |
| **RTX 4060** | 24 × ~2.7 | 18 (smem 100 KB; regs 19) | 432 | 0.7–1.0 | 258–369 | 133–190 | 47–69 | **150–215** | **250–365** | 126–181 |
| RTX 3060 12 GB | 28 × ~1.85 | 16 (16-blocks-per-SM cap) | 448 | 0.5–1.0 | 326–650 | 168–340 | 58–155 | **80–170** | **130–290** | 70–145 |
| GTX 1660 Super | 22 × ~1.9 | 11 (64 KB smem) | 242 | 0.35–0.9 | 550–1410 | 284–730 | 72–215 | **40–105** | **70–185** | 35–90 |
| RX 7600 (wave32) | 32 CU × ~2.6 | 14 (64 KB LDS per CU) | 448 | 0.4–0.9 | 242–544 | 125–280 | 40–107 | **100–230** | **170–400** | 85–195 |
| RX 6600 (wave32) | 28 CU × ~2.45 | 14 (LDS) | 392 | 0.4–0.9 | 293–660 | 151–340 | 49–130 | **85–190** | **140–330** | 70–160 |
| Arc A750 | 28 Xe × ~2.3 | ~14 (SLM; large-GRF mode) | ~392 | 0.3–0.9 | 312–937 | 161–483 | 56–208 (K1 falls back to sort) | **55–180** | **95–300** | 48–150 |
| M4 Pro, 20-core | 20 × 1.58 | ~18 (dynamic caching) | ~360 | 0.5–1.0 | 530–1060 | 273–546 | 96–283 | **50–105** | **80–180** | 42–88 |
| M4, 10-core | 10 × 1.58 | ~18 | ~180 | 0.5–1.0 | 1060–2120 | 546–1092 | 192–566 | **25–50** | **40–90** | 21–44 |

- The discrete NVIDIA rows are the best-founded: the same ISA family, measured residency rules, and a measured contention curve.
- **Everything outside NVIDIA is an estimate with no measurement behind it.** There are no vkstats for RADV, ANV or Metal, and the AMD, Arc and Apple builds currently produce corrupt output (ledger).
- **RDNA running wave64** (RADV has historically used wave64 for compute; Windows drivers usually use wave32): wg16 fills only 16 of 64 lanes, and the VGPR budget halves the waves. K3 gets about 1.5–2× slower, so RX 6600/7600 opt16 drops to about 50–150 MB/s.
- For the M4 Pro, use the peer session's numbers once they exist. These rows are placeholders built from core count × clock.

### Targets

| Target | opt16 | opt14 (and the 2-pass ≥ L16 schedule) |
|---|---|---|
| **125 MB/s (1 Gbit)** | 4060 **yes** (1.2–1.7×). 3060, 7600, 6600: **borderline**, with ranges straddling 125. Arc: unclear. 1660 Super, M4 Pro, M4: **no**. | 4060, 7600: **yes**. 3060, 6600: yes except at the bottom of their ranges. 1660 Super, A750, M4 Pro: borderline. M4: no. |
| **1250 MB/s (10 Gbit)** | **No card.** The 4060 is 6–8× short. | **No card.** The best case (4060) is 3.4–5× short. |

**CPU comparison (scaled, not measured).** libzstd L16 gives 497 MB/s on the 9950X3D's 16 cores and 32 threads, which suggests about 150–220 MB/s on a typical 8-core. On a 3060 or 1660 Super the CPU alone matches or beats GPU opt16, and on a 4060 it is roughly equal. Any small-GPU plan should use both (§4, item 5).

## 4. Ideas and recommendations

| # | Idea | Mechanism | Speedup (small GPU) | Ratio | Exactness | Effort | Risk |
|---|---|---|---|---|---|---|---|
| 1 | **Heavy-first (LPT) block order in K3**: a GPU cost pass plus a sort, or a persistent grid pulling blocks in descending cost from an atomic queue. Cost proxy: Σ min(max(lenA, lenB), 32) − 2, or the previous pass's time. | Removes the heavy-block tail, which does not amortise with batch size on small cards | **Measured: −18 to −23 % K3** at 510 concurrency (N 2040–4000); opt14 −21 to −26 %. About −15 to −20 % end to end. 0 % on the 5090 at one wave. | none | byte-identical (order only; outputs are per block) | S–M | Low. Coordinate with a07-scheduling. |
| 2 | **A 2-pass (or 3-pass) schedule that meets the L16 gate as the "opt16" of small GPUs**: a04's `K.c.f` / `K.c.c.f`, a new seed with 1 or 2 cheap passes | Fewer DP passes: K3 ×0.52 at 2 passes (measured opt14/opt16 proxy ratio 65.7 / 127.5) | **×1.6–1.7** end to end on every small card (4060: 150–215 → 250–365 MB/s). The 3-pass variant gives ×1.3. | a04 scratch: 64 KiB **1.37116** (+0.012 % over L16) for 2-pass, 1.37157 for 3-pass; 16 KiB 1.32809 / 1.32830 (≥ L16 1.32774) | new oracle / preset | M (a04's lane) | The margin is thin (+0.012 %). Needs the full-corpus gate at 16/32/64 KiB. |
| 3 | **Residency-derived defaults** (R7, now with measured K3 numbers): K1 G = min(resident capacity, L2 budget) (96 on a 4060); K3 stays at wg16 on NVIDIA; batch = a whole number of K3 waves, at least 4 | Avoids K1's second round; nothing gained beyond 4 waves | K1 about −30 % on a 4060 (R7; the 1/G law here confirms the mechanism). Batch: 2 → 4 waves gives −10 to −17 % K3 (measured). | none | byte-identical | S | Low. Needs `C_K3` from a startup probe (R7 §3) or vkstats. |
| 4 | **Rate-governed preset ladder**: per batch, pick opt16 → opt14 → lvl9s12seg so that compression keeps up with the download rate | Throughput on demand | Keeps line rate on any card | Mixed. The corpus total falls below L16 once any lower rung is used. **User decision.** | each rung stays exact to its oracle | M | Policy, not tech: it breaks "≥ L16" as a guarantee. |
| 5 | **Hybrid CPU + GPU** with a deterministic block split (for example every k-th block to CPU threads), tuned at startup | Adds the CPU's 150–220 MB/s | about ×1.7–2.5 total on 3060/1660-class cards | libzstd L16 on CPU blocks: the corpus ratio stays between L16 and opt16, so it passes the gate | CPU blocks are libzstd frames, not the oracle, so the output is still deterministic if the split is fixed | M | The CPU also has download, hash and IO work to do. |
| 6 | **Trim K3 workgroup memory** (4520 → about 3.5 KB): alias `hist` (1 KiB) onto `ring_p`, which is dead during the epilogue and prologue | Raises residency where shared memory is the limit | Turing 11 → 14 per SM, about +20 % K3 (from the L(r) curve). RDNA LDS 14 → 18 per CU (then VGPR-bound), about +15–20 %. Ada 18 → 19, about +3 %. 5090: 0. | none | byte-identical | S | Low (the T3b study already proposed the aliasing). |
| 7 | **Wave-size-aware K3 workgroup on AMD**: read `subgroup_size` in the lane probe; under wave64 use wg64 (4 blocks per wave) or wg32 | Recovers 48 idle lanes and the VGPR occupancy lost to wave64 | Estimated 1.5–2× K3 on wave64 compiles; 0 on wave32 | none | byte-identical (W and wg are output-invariant) | S | It is guesswork until someone runs RADV. |
| 8 | **A small-L2 guard for K3's scratch payload**: on 3060/1660-class cards, check whether moving the ring payload back into shared memory, or into the private-ring variant, beats global scratch | T3b's −32 % assumed an L2-resident scratch | Unknown: 0 to +30 % on small-L2 cards | none | byte-identical | M | Needs a real card; the 5090 cannot emulate a 1.5–3 MB L2. |

### Checked and rejected

- **wg32 (2 blocks per warp) on NVIDIA.**
  - Saturated uniform-block throughput: 0.913 against 0.952 blocks·pass/ms/SM (−4 %).
  - The lone-block latency is +19 % (19.6 against 16.5 ms per pass, 64 blocks).
  - The perf study's multi-wave win was on the old shared-memory ring. Keep wg16.
- **Larger batches for small cards.** Beyond about 4 sorted waves: 0 % (measured). Without sorting the tail stays at +20–25 % even at 7.8 waves, so sort rather than grow the batch.
- **VRAM as a limit.** 4 waves on a 4060 need 3.1 GiB (1.78 MiB per block). A 6 GB 1660 Super needs about 1.7 GiB for 4 waves of 242 blocks. The 6 GiB budget can drop to 3–4 GiB on 6–8 GB cards without cost.
- **PCIe.** 64 KiB up and about 48 KiB down per block is about 0.4 GB/s at 215 MB/s of input. That is under 7 % of PCIe 3.0 ×8 (~6.5 GB/s effective) and 3 % of 4.0 ×8. Even 1250 MB/s would need only about 2.2 GB/s.
- **Clock-locking proxy** (R7's `:tput` mode): no root (`sudo -n` fails), and not needed, because K3's per-SM saturation curve gives the issue side directly.
- **Emulating 24 SMs at full per-SM contention:** not possible. The block scheduler spreads breadth-first, and batches above about 4000 blocks exceed the 2 GiB binding limit (`max_storage_buffer_binding_size`; N = 6800 failed). The concurrency proxy plus the L(r) curve replaces it, validated above.
- **"Turing lacks independent thread scheduling":** it doesn't. Volta and later, including TU116, have it; Pascal does not. The 1660 Super's real limits are 64 KB of shared memory per SM, 32 warps per SM, 16 blocks per SM, 1.5 MB of L2, and a 16-lane INT32 path per SMSP.

## 5. Top 3

1. **Heavy-first order in the pipeline (idea 1)**: measured −18 to −23 % K3 at 4060 concurrency; byte-identical; S–M. Do it together with **batch = whole K3 waves, at least 4** and **K1 G from residency** (idea 3).
2. **Ship the ≥ L16 2-pass schedule (a04's `K.c.f`) as the small-GPU / default opt16 (idea 2)**: ×1.6–1.7 everywhere, and the only change that moves the 4060 from about 1.2–1.7 Gbit to about 2–3 Gbit at L16 ratio. It needs the full gate at every block size, because its margin is +0.012 %.
3. **Get one real small card measured before any more tuning.** Buy or borrow a 4060 and one RDNA card, and run R7's kstats plus `k3opt_passes_timing` (with heavy-first order). That pins a (currently 0.7–1.0 on Ada, 0.4–0.9 on RDNA) and the small-L2 question (idea 8). Also fix the Metal/AMD shift corruption first, since no non-NVIDIA number means anything until then.

**Bottom line for the product question.** At 10 Gbit no 8 GB card can run opt16 or opt14; they are 3.4–8× short. At 1 Gbit a 4060 or RX 7600 can run opt16, and with top-3 items 1 and 2 it runs at about 2–3 Gbit. 1660-class and M4-class machines need the rate-governed ladder or the CPU hybrid (ideas 4 and 5) to keep up even at 1 Gbit.
