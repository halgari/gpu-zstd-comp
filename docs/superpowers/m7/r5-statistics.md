# r5-statistics: cheap symbol statistics so one final DP pass reaches L16

Research only. No repository file was changed.

- Scratch workspace: `~/.cache/gzc-m7/r5-statistics/`. Build outputs and stats caches were deleted.
- Artifacts: `.superpowers/m7-research/artifacts/r5-statistics/`:
  - `core-vs-repo.diff`: the scratch `gzc-core` against the repo. It is a10's copy, which has the split-writer exports, plus my additions to `opt.rs`:
    - `segment_part`, a resumable segment;
    - an in-pass slice histogram;
    - `dp_pass_sliced`, which slices a pass with a price-update callback per slice;
    - `dp_pass_ranges`, a DP over arbitrary sub-ranges;
    - a `GAP3` switch.
  - `exp/src/main.rs`: the experiment driver, with a schedule language, a stats-block cache, file and window grouping, and lazy/lvl9 corrections. `split.rs` is a10's exact multi-block frame writer, and `cands_multi` is a06's h10 candidate finder.
  - `res/*.txt`: raw outputs. `comb.py` sums the full-corpus chunks.

## TL;DR

1. **The encoder can learn statistics per file, or per ≤ 1 MiB window of a file, instead of per block.**
   - The pre-pass is one cheap in-pass-refresh pass (a04's `Ew16`). It runs on 2 evenly spread blocks per 16-block window at 64 KiB, or 8 per 64-block window at 16 KiB.
   - That is 13 % of blocks, so about **0.13 of a cheap pass per block (≈ 1.2 µs on the 5090, estimate)**.
   - Its pooled histogram prices the single final pass of every block in the window.
   - Full corpus, **1 DP pass**: **1.37147 (+0.035 % vs L16) at 64 KiB and 1.32794 (+0.015 %) at 16 KiB**. That is exactly opt16's ratio, for one pass instead of four.
   - Output blocks stay independent. Only the encoder's work depends on neighbouring blocks.
2. **Adding zstd-style adaptive prices inside the final pass** raises the single pass above opt16.
   - The pass runs in 8 slices; each slice is priced from `¼·seed + own cumulative counts`.
   - Full corpus: +0.047 % vs L16 at 64 KiB and +0.029 % at 16 KiB, with windowed stats.
   - With whole-file stats from 2 blocks per file (0.057 of a pass): +0.037 % / +0.022 %.
   - **On its own it does not clear L16 + 0.05 %.**
3. **The fewest passes that clear L16 + 0.05 % at both 64 and 16 KiB is 1 final DP pass plus the stats stage, once split (B1) is used.** Full corpus, vs L16:

   | 1-pass schedule | 64 KiB | 16 KiB |
   |---|---:|---:|
   | windowed static stats + split | **+0.217 %** | **+0.062 %** |
   | windowed adaptive (D8) + split | +0.231 % | +0.075 % |
   | whole-file adaptive + split + gap3 | +0.242 % | +0.085 % |
   | whole-file static + split + h10 | +0.300 % | +0.108 % |
   | whole-file adaptive + split + h10 | +0.333 % | +0.128 % |
   | *no stats stage (opt14 prior, `P.f`) + split + h10* | *−0.016 %* | *−0.095 %* |

   The last row shows that the stats stage is what makes a single pass work.
4. **Dead ends:**
   - lazy/greedy/lvl9 statistics with an offline correction: −0.16 % to −1.4 %, even when trained on the same mod;
   - stratified within-block sampling: −0.06 % to −0.58 %;
   - a file seed in front of a ≥ 1.5-pass schedule: no gain, the refresh dominates;
   - groups that ignore file boundaries: −0.11 % at 32 blocks.
5. **GPU caveat.**
   - The stats pass touches only 6–13 % of the blocks, but a DP pass is latency-bound: one lane walks 4 KiB serially.
   - As a separate dispatch it would cost a full pass of wall time.
   - It must ride as extra workgroups in another K3 dispatch: the previous batch's final pass, which is a lookahead.

## Method

- **Oracle.** The real `gzc_core::opt` DP, `Hist`, `Prices` and `write_frame`, and a10's split writer. Ratio = real bytes / frame bytes.
  - Sanity checks on the full corpus reproduce the published numbers: `B.c.c.c.f` = opt16 1.37144 / 1.32794, `B.E16.f` = 1.37216 / 1.32840, `B.E32u16.f` = 1.37180, and split opt16 (est_cuts) +0.173 %.
  - Finalist frames round-trip through libzstd (`VERIFY=1`, including split frames).
- **Samples.** Exploration used the 1/50 sample (2016 blocks at 64 KiB, 7959 at 16 KiB). On it opt16 = 1.37211 and L16 = 1.37175. Finalists used the **full corpus** (100,754 blocks / 397,925 blocks), run in 3–4 interleaved chunks and summed.
- **Gates.** L16 = 1.37100 / 1.32774; L14 = 1.36827 / 1.32606.
- **Schedule notation.** The seed comes first, then the passes.
  - Seeds:
    - `B`: block init;
    - `P`: opt14 prior plus cover literals;
    - `FM<m><src>f`: the mean histogram of `m` evenly spread stats blocks per file (`j_i = ⌊(2i+1)n/2m⌋`), with the file's literals;
    - `S<src>`: the block's own stats.
  - Stats sources: `e` = Ew16 pass output, `h` = Ew32u16 pooled slices, `o` = converged opt16, `c` = one `B.c` pass, `l`/`g` = price-aware lazy/greedy over the K2opt records, `n` = lvl9s12seg, and `x`/`y`/`z` = l/n/g with a cross-mod correction.
  - Passes:
    - `f`: final optLevel-2 pass;
    - `c`: cheap pass;
    - `E<n>[u<k>]`: a04's refresh pass;
    - `D<n>a<p>`: the final pass in `n` slices, each slice priced from `p%·seed + cumulative own slice counts`; `g1` means one-slice lag;
    - `Z<k>w<p>`: stratified sub-range pass.
  - `+S` means with split (est_cuts, the min of the two frames as in a10). `FWIN=W` means windows of W blocks inside a file (the last window absorbs the tail). `GROUP=G` means fixed corpus groups that ignore files.
- **Corpus shape.** 3172 files: 601 single-block files (0.6 % of 64 KiB blocks), and 89 % of blocks in files of 32 blocks or more.
  - Whole-file stats: FM1 = 0.031, FM2 = 0.057 and FM4 = 0.099 stats blocks per block at 64 KiB; FM8 = 0.053 at 16 KiB.
  - Windowed (W16/FM2 at 64 KiB, W64/FM8 at 16 KiB): 0.131 / 0.128.

## 1. File-level statistics from a GPU pre-pass

**Mechanism.**
- A cheap Ew16 pass (block-init seed, 16 refresh slices) runs on the stats blocks.
- Their output histograms (literals, LL, ML, OF) are summed per file or window and divided by the number of stats blocks.
- Every block of the file then runs **one** final pass priced with `Prices::from_freqs(c + (c > 0))` of that table.

**Which source to pool** (sample, 64 KiB, all blocks of each file as stats blocks, `F1`):

| file stats source | vs opt16 | vs L16 |
|---|---:|---:|
| `e` (Ew16 output) | **−0.016 %** | +0.010 % |
| `h` (Ew32u16 half pass) | −0.029 % | −0.003 % |
| `o` (opt16 converged, 4 passes) | −0.030 % | −0.004 % |
| `c` (one B.c pass) | −0.485 % | −0.458 % |
| `l`/`g` lazy/greedy | −0.34 % / −0.38 % | |
| `n` lvl9s12seg | −1.52 % | |

- Own cover literals instead of the file's literals: −0.01 %. The block's raw byte histogram: −0.4 to −0.5 %. Keep the file literals.
- Pooled Ew16 output is as good as converged opt16 statistics, at a quarter of the cost.

**How many stats blocks** (sample, 64 KiB, static `.f`):
- FM1 −0.077 %, FM2 −0.028 %, FM4 −0.024 %, F8 −0.024 % and F1 −0.016 % vs opt16.
- **Locality matters more than count.** Windows inside a file:
  - W16/FM2: +0.001 %;
  - W8/FM2: +0.013 %;
  - W32/FM2: −0.026 %.

**Full corpus, 1 DP pass, static final:**

| schedule | stats cost (cheap passes/block) | 64 KiB | vs L16 | +S vs L16 | 16 KiB schedule | 16 KiB | vs L16 | +S vs L16 |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| whole file, FM2ef.f | 0.057 | 1.37111 | +0.008 % | +0.188 % | FM8ef.f (0.053) | 1.32774 | −0.000 % | +0.047 % |
| **window, W16 FM2ef.f** | 0.131 | **1.37147** | **+0.035 %** | **+0.217 %** | W64 FM8ef.f (0.128) | **1.32794** | **+0.015 %** | **+0.062 %** |
| whole file + gap3 | 0.057 | 1.37141 | +0.030 % | +0.211 % | FM8ef.f + gap3 | 1.32796 | +0.016 % | +0.064 % |
| whole file + h10 | 0.057 | 1.37264 | +0.120 % | +0.300 % | FM8ef.f + h10 | 1.32854 | +0.060 % | +0.108 % |
| *no stats: `P.f`* | 0 | 1.36676 | −0.310 % | −0.133 % | `P.f` | 1.32488 | −0.216 % | −0.168 % |
| *opt16* | 3 | 1.37144 | +0.032 % | +0.206 % | | 1.32794 | +0.015 % | +0.063 % |

**Stability across file sizes and kinds** (full corpus, vs opt16; chunk averages).

- 64 KiB, whole-file FM2ef.f:

  | class | vs opt16 |
  |---|---:|
  | 1-block files | +0.027 % (the seed is the block's own Ew16, so this equals B.E16.f) |
  | 2–3 blocks | −0.044 % |
  | 4–7 blocks | −0.140 % |
  | 8–31 blocks | −0.076 % |
  | 32+ blocks | −0.018 % |
  | NIF | −0.258 % |
  | DXT1 | +0.021 % |

- The adaptive final pass (§4) repairs most of this: 4–7 blocks −0.042 %, NIF −0.050 %.
- At 16 KiB the same pattern holds: FM8ef.f is −0.099 % on 4–7-block files, and FM8ef.D8a25 is −0.028 %.
- Small files are a small share of the bytes, so they barely move the total. The weak point is medium files (4–31 blocks) of mixed mips, which windowing fixes.

**Grouping that ignores file boundaries**, for example a stream with no file information (sample, FM2ef.f):
- G8: −0.006 %;
- **G32: −0.107 %**;
- G128: −0.351 %.

Groups must respect file (format) boundaries. A streaming API without file boundaries would need a DDS-header or format-change detector, or windows of 8 blocks or fewer, which costs 0.25 of a pass.

**GPU cost of the stats stage** (estimate, 5090, per 64 KiB block, from M5's 8.2 µs cheap pass):
- An Ew16 cheap pass costs 8.2 × 1.08 (a04's 16-slice barrier imbalance) plus about 0.3 µs of rebuilds, ≈ 9.2 µs per stats block.
  - Whole-file FM2: 0.057 × 9.2 ≈ **0.52 µs/block**.
  - W16/FM2: 0.131 × 9.2 ≈ **1.2 µs/block**.
  - The `h` source (half pass) halves this, at about −0.01 to −0.02 %.
- Pooling is one tiny workgroup per window (≤ 8 histogram adds of 377 entries) plus a 377-entry price build, < 0.05 µs/block. Memory: 377 u32 counts plus 377 u16 prices per window, about 2.3 KB.
- **Chain length.** The stats pass is a full 4 KiB-per-lane dependent chain. As its own dispatch over 6–13 % of the batch it is latency-bound: one wave costs about a whole pass of wall time.
  - It must therefore run as **extra workgroups inside the previous batch's final-pass dispatch** (lookahead).
  - That needs the stats blocks' K1/K2opt candidates one batch early: either recompute them (+6–13 % of K1+K2 ≈ +0.54 / +1.24 µs/block) or keep their candidate buffers (+6–13 % candidate VRAM).
  - Total stats stage: **≈ 1.0 µs/block (whole file) to ≈ 2.4 µs/block (windowed)**, against 8.2 µs per cheap pass removed. The final pass needs a per-block `window id → price table` indirection; its prices still come from a table, as with today's PRICE_MODE.
- **5090 end-to-end** (today's kernels, before A1–A4; estimate): non-K3 11.38 µs + final 9.46 µs + stats 1.0–2.4 µs ≈ **21.9–23.2 µs/block ≈ 2.7–2.9 GB/s**. That is **≈ 1.9–2.0× opt16**, against ≈ 1.48× for a04's `B.E16.f`. The split stage adds 1–4 µs.
- **8 GB class** (a04's model: other ≈ 65 µs of which K1+K2 ≈ 54, L0 ≈ 42, L2 ≈ 48; opt16 ≈ 239 µs):
  - W16 static ≈ 65 + 48 + 0.131 × (45 + 54) ≈ 126 µs, which is **≈ 1.9× opt16, about 0.38–0.55 GB/s** (from 0.20–0.29).
  - Whole-file stats ≈ 119 µs (2.0×).

**Byte-exact oracle.**
- Stats blocks are chosen by an integer rule: window `w` of `W` blocks (the last window absorbs the tail), and `m = min(M, len)` blocks at `lo + ⌊(2i+1)·len/2m⌋`.
- Each one runs the a04 `dp_pass_refresh` oracle (to be ported; mine is `dp_pass_sliced` in the scratch core). Its histogram is `Hist::of_output`.
- The window table is `(Σ counts + m/2) / m` in u32. The final pass is `dp_pass` with `Prices::from_hist`-style frequencies from that table.
- Everything is integer, and K3's price build already exists, so the GPU is exact by construction.
- The host supplies `block → window` and the stats-block list.

**Effort and risk.**
- Effort M:
  - oracle: about 60 lines plus a04's refresh port;
  - K3: window-table indirection, and stats blocks as extra workgroups with an Ew16 PRICE_MODE;
  - host: windowing and lookahead batching;
  - a tiny pool kernel.
- Risks:
  - lookahead batching and the memory for one batch of extra candidates;
  - the dependence on file boundaries (streams);
  - **the static-only variant clears 16 KiB + 0.05 % only with split (+0.062 %)**: a thin margin, though + gap3 or h10 widen it.

## 2. Statistics from a cheaper parse plus an offline correction

**Mechanism.**
- Take the block's (or file's) histogram from a price-aware lazy parse over the K2opt records (3-byte matches, a04's `greedy(lazy)`), from greedy, or from the real lvl9s12seg parse.
- Multiply each table entry by a fixed ratio `Σ opt16-final / Σ cheap`, trained on every 200th block of **the other mod** (smim-se ↔ noble-skyrim-2k). This is a diagonal correction.

**Sample results, 64 KiB, one final pass, vs opt16:**

| source | raw | corrected cross-mod | corrected same-mod |
|---|---:|---:|---:|
| own lazy (`Slf` / `Sxf`) | −0.324 % | **−0.190 %** | −0.155 % |
| own greedy (`z`) | −0.363 % | −0.228 % | −0.185 % |
| own lvl9s12seg (`n` / `y`) | −1.543 % | −1.438 % | |
| file FM4 lazy (`x`) | −0.351 % | −0.230 % | −0.196 % |
| own lazy, corrected, + adaptive final `D16a25` | | −0.084 % | |

- The correction closes about 40 % of the gap. Cross-mod transfer costs only 0.03–0.04 %, so the transform is portable.
- The **model itself is the limit**: same-mod training still leaves −0.155 %.
- The lazy parse does not make the DP's start/stop decisions (a09: 7 % of the final sequences are shorter than the longest match available), so its LL/ML shape is wrong in ways a per-code scale cannot fix.

**GPU cost.** About 0.1–0.2 of a pass with no chain (a04). Irrelevant, because the ratio fails. **Dead end.**

## 3. Sampling within a block

**Mechanism.**
- `Z<k>w<p>`: a cheap DP over `k` centred strata per 4 KiB segment, covering `p` % of the segment. Each stratum starts like a segment (reps 0, literals anchored at its start; matches may reference earlier block data).
- The histogram is scaled by 1/p and prices the final pass.

**Sample results, vs opt16:**

| schedule | 64 KiB | 16 KiB |
|---|---:|---:|
| `B.Z1w25.f` / `B.Z4w25.f` / `B.Z16w25.f` | −0.54 / −0.52 / −0.58 % | −0.25 / −0.24 / −0.30 % |
| `B.Z8w50.f` | −0.50 % | −0.20 % |
| `P.Z8w50.f` | −0.10 % | −0.05 % |
| `FM4ef.Z8w50.f` (file seed, then strata) | −0.08 % | −0.04 % |
| *for comparison:* `FM4ef.f`, `B.E32u8.f` | −0.024 %, −0.035 % | −0.016 %, −0.051 % |

- The 16 KiB segment-0 problem goes away, since every segment is sampled.
- But a single static cheap pass from a poor seed gives poor statistics, and short strata bias the counts toward literals and short matches. Stratification over more strata makes this worse: Z16 is below Z4.
- Started from a good file seed, strata *degrade* it.
- Entropy weighting cannot fix a biased estimator, so it was not pursued. Anything below a full refresh pass loses to the file seed. **Dead end** (it confirms a04).

## 4. Adaptive price updates inside the final pass

**Mechanism.**
- The final optLevel-2 pass runs in `n` slices per segment, with a04's slice mechanics: a lane stops at its first series start past the boundary, and the workgroup barriers.
- After slice `j`, the prices for slice `j+1` come from `round(a·seed + Σ_{i≤j} slice_i)`, where `seed` is the file or window table.
- This is zstd's `ZSTD_updateStats`, done deterministically at fixed points. a04's refresh used only the last slice; that is too noisy for a *final* pass, which this cumulative-plus-prior form fixes.

**Sample results** (64 KiB, FM4ef seed, vs opt16):

| variant | vs opt16 |
|---|---:|
| D16 a = 0 / 10 / 25 / 50 / 100 % | −0.029 / −0.008 / **−0.002** / 0.000 / −0.003 % |
| a25 with D2 / D4 / D8 / D16 / D32 / D64 | −0.012 / −0.007 / −0.004 / −0.002 / −0.001 / −0.007 % |
| one-slice lag (`g1`; the barrier only waits for slice `j−1`), D16 / D8 | −0.006 / −0.010 % |
| `FM4ef.c.D16a25` (a full cheap pass, then the adaptive final) | +0.005 % (worse than `FM4ef.c.f`, +0.025 %) |

As a04 found, adaptivity helps only when the seed is not the block's own fresh statistics.

**Full corpus, 1 DP pass:**

| schedule | 64 KiB vs L16 | +S | 16 KiB schedule | vs L16 | +S |
|---|---:|---:|---|---:|---:|
| FM2ef.D8a25 (whole file) | **+0.037 %** (+0.005 % vs opt16) | +0.220 % | FM8ef.D8a25 | **+0.022 %** | +0.068 % |
| FM2ef.D4a25 | +0.033 % | +0.215 % | FM8ef.D4a25 | +0.018 % | +0.065 % |
| FM4ef.D8a25 | +0.036 % | +0.221 % | FM16ef.D8a25 | +0.024 % | +0.071 % |
| **W16 FM2ef.D8a25 (windowed)** | **+0.047 %** | **+0.231 %** | W64 FM8ef.D8a25 | **+0.029 %** | **+0.075 %** |
| FM2ef.D8a25 + gap3 | +0.059 % | +0.242 % | FM8ef.D8a25 + gap3 | +0.038 % | +0.085 % |
| FM2ef.D8a25 + h10 | +0.151 % | +0.333 % | FM8ef.D8a25 + h10 | +0.081 % | +0.128 % |

**Answer: yes, with a good file seed one adaptive final pass reaches L16, and in fact opt16's ratio or better.** It clears L16 + 0.05 % without split only at 64 KiB with gap3 (+0.059 %); at 16 KiB it needs split or h10.

**GPU cost** (estimate). a04's lane-imbalance proxy for an optLevel-2 pass gives +10.7 % at 8 slices and +6.5 % at 4. Each rebuild is about 0.1 % of a pass.
- D8 final ≈ 9.46 × 1.11 ≈ **10.5 µs**; D4 ≈ 10.1 µs; static 9.46 µs.
- The seed table is read from global memory at each rebuild (377 loads). The cumulative counts live in the existing workgroup `hist`, so there is no new shared memory.
- The lag-1 variant turns each barrier into a "slice j−1 done" wait, for −0.004 to −0.006 %. It is worth having if the GPU barrier cost exceeds the proxy.

**Oracle.** Integer `a = 1/4`: per-entry frequency `⌊(seed_sum + 4m·cum + 2m) / 4m⌋`, then `+ (c > 0)`. My prototype rounds the file mean in f64 first, which is a negligible difference to pin down in the oracle. Slicing follows a04's `dp_pass_refresh`.

**Effort and risk.**
- Effort M: the same slice machinery as a04's refresh, applied to the L2 kernel, plus a seed read.
- Risks:
  - L2 barrier imbalance (+17 % at 16 slices, which is why D8 or D4 is chosen);
  - registers in the L2 kernel for the slice stop.

## 5. Combinations and the fewest passes (full corpus, vs L16)

| DP passes | schedule | 64 KiB | 16 KiB | clears +0.05 % at both? |
|---|---|---:|---:|---|
| 1 (+0.13 stats) | W16/W64 static + split | +0.217 % | +0.062 % | **yes** (thin at 16 KiB) |
| 1 (+0.06 stats) | whole-file static + split + gap3 | +0.211 % | +0.064 % | **yes** |
| 1 (+0.13 stats) | windowed D8a25 + split | +0.231 % | +0.075 % | **yes** |
| 1 (+0.06 stats) | whole-file D8a25 + split + gap3 | +0.242 % | +0.085 % | **yes** |
| 1 (+0.06 stats) | whole-file static + split + h10 | +0.300 % | +0.108 % | **yes** |
| 1 (+0.06 stats) | whole-file D8a25 + split + h10 | +0.333 % | +0.128 % | **yes**, best margin |
| 1 (+0.06 stats) | whole-file D8a25 + gap3, no split | +0.059 % | +0.038 % | no (16 KiB) |
| 1 | `P.f` + split + h10 (no stats stage) | −0.016 % | −0.095 % | no |
| 1.5 | `B.E32u16.f` + split | +0.239 % | +0.063 % | yes, but costs 0.5 pass more |
| 2 | `B.E16.f` + split | +0.267 % | +0.098 % | yes |

- The whole-file and h10 rows were run with whole-file stats. Windowed stats would add about +0.03 % (64 KiB) and +0.01 % (16 KiB).
- Split, h10 and gap3 stack with the stats stage almost additively. Split's gain on a 1-pass parse (+0.18 %) is the same as on opt16 (+0.17 %).

## Dead ends checked

- **Cheap-parse statistics** (lazy, greedy, lvl9s12seg) with a per-entry correction trained cross-mod: −0.19 % at best, −0.155 % even same-mod. Model-limited (§2).
- **Statistics from one block-init cheap pass** (`c`) pooled per file: −0.49 %. The pre-pass must itself refresh, as Ew16 does.
- **Converged opt16 statistics as the file source** (`o`): no better than Ew16 (−0.030 % vs −0.016 %), at 4× the cost.
- **Stratified or entropy-sampled within-block partial passes** (§3): −0.06 % to −0.58 %. Worse than the file seed they start from.
- **A file seed in front of ≥ 1.5-pass schedules:**
  - `FM4ef.E32u16.f` +0.020 % = `B.E32u16.f` +0.021 %;
  - `FM4ef.c.f` +0.025 % < `B.E16.f` +0.050 % (sample).
  - In-pass refresh already makes the seed irrelevant.
- **An adaptive final pass after a full cheap pass**: −0.02 % against a static final pass.
- **A cumulative-adaptive cheap half pass** (`C32a25u16`, `G32a25u16`): equal to a04's `E32u16`.
- **Groups that ignore file boundaries**: −0.11 % (32 blocks) and −0.35 % (128 blocks). They must respect file or format boundaries.
- **The block's raw byte histogram as literal prices**: −0.4 to −0.5 %.

## Top 3

1. **Windowed file statistics plus a static single final pass, with split.**
   - Recipe: Ew16 on 2 of every 16 blocks (64 KiB) or 8 of every 64 (16 KiB) inside each file; pool; one optLevel-2 pass per block; split frames.
   - Ratio: +0.217 % / +0.062 % vs L16 (full corpus), +0.24 % / +0.08 % with gap3.
   - No new barriers in the final kernel; the stats stage ≈ 1.2 µs (+1.2 µs if K1/K2 are duplicated for the lookahead). Estimated **≈ 1.9–2.0× opt16 on both the 5090 and 8 GB cards**, against 1.48× for the 2-pass `B.E16.f`.
   - It needs lookahead batching so the stats pass rides inside the previous batch's final dispatch.
2. **Add 8-slice adaptive pricing to that final pass (`D8a25`).**
   - It buys +0.015–0.03 % for about +11 % of the final pass (≈ 1 µs).
   - It is the margin insurance at 16 KiB (+0.075 % with split) and the only 1-pass form at opt16-or-better ratio *without* split (+0.047 % / +0.029 %).
   - Use the lag-1 variant if the GPU barrier costs more than the proxy.
3. **Stack h10 (and gap3) on the 1-pass preset rather than spending them on fewer passes elsewhere.**
   - Whole-file D8a25 + split + h10 is +0.333 % / +0.128 % in one DP pass, the widest margin measured.
   - For the cheapest stats, the whole-file FM2 source (0.057 pass) is enough once h10 is in.

**Open GPU measurements needed:**
- the L2 slice-barrier cost (D8, D4, lag-1);
- the latency of stats workgroups appended to a final-pass dispatch;
- VRAM for the lookahead candidates on 8 GB cards.
