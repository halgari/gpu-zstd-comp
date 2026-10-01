# a04-passes: fewer DP passes through better price seeding and convergence

Research only. No repository file was changed. Scratch code is in `/tmp/claude-1000/m6/a04-passes/`:
- `exp/`: experiment binary. It uses the **unmodified** `gzc-core` as a library: `dp_pass`, `Prices`, `Hist`,
  `cover_literals`, `find_cands`, `write_frame`.
- `core/gzc-core/`: a scratch copy of the oracle with two additions:
  - `dp_pass_refresh`, the exact in-pass price-refresh pass (§1). It needs a resumable `segment_run`.
  - a DP work counter, used for the lane-imbalance estimate.
- `exp2/`: the experiment binary built on that copy.
- Raw outputs: `full64.txt`, `full16_q*.txt`, `xmod_h*.txt`, `e64_t*.txt`, `f64_t*.txt`, `f16_q*.txt`, `g64.txt`,
  `g16.txt`. `comb.py` sums the partial runs.

## TL;DR

1. **In-pass price refresh solves the problem.** The cheap pass is split into `w` slices per 4 KiB segment. After
   each slice the workgroup builds new price tables from that slice's histogram alone (unscaled), then continues.
   - One such pass plus the normal optLevel-2 final pass beats opt16's 4-pass ratio at both block sizes.
   - It uses today's block-init seed and needs no prior tables.
   - `B.Ew16.f` (2 passes) on the full corpus:
     - 64 KiB: **1.37216**, +0.084 % over L16 and +0.052 % over opt16.
     - 16 KiB: **1.32840**, +0.050 % over L16.
   - `B.Ew32u16.f` (1.5 passes; the cheap pass stops halfway through each segment):
     - 64 KiB: **1.37180**, +0.058 % over L16.
     - 16 KiB: **1.32794**, equal to opt16 and +0.015 % over L16.
2. **Per-kind priors** (DXT1 / DXT5 / DXT5 `_n` / NIF from the DDS header and file name) make the plain 2-pass
   schedule `K.c.f` clear L16, but only just:
   - full corpus with held-out priors: 1.37116 (+0.012 %) at 64 KiB and 1.32788 (+0.010 %) at 16 KiB;
   - with priors trained on the *other* mod: 1.37100 (+0.000 %). That margin is too fragile for a gate.
   - Per-kind priors do not get a single pass to L16: `K.f` is −0.136 % vs L16.
3. **One pass reaches L14, not L16.** `K.Dw16` is a final pass with in-pass refresh and per-kind seed. It gives
   1.37045 at 64 KiB (+0.159 % over L14) and 1.32644 at 16 KiB (+0.029 % over L14). That is an opt14 class at half
   of opt14's K3 cost.
4. **Exact early termination is a dead end.** Only 0.00 %, 0.10 % and 0.45 % of blocks have unchanged prices going
   into opt16's cheap passes 1, 2 and 3.

## How it was measured

- **Oracle.** The real `gzc_core::opt::dp_pass` and frame writer are used, so ratio = real bytes / frame bytes, as
  `gzc-bench ref` computes it. Sanity checks on the full corpus:
  - the `B.c.c.c.f` schedule reproduces opt16 exactly (1.37144 at 64 KiB, 1.32794 at 16 KiB);
  - `P.c.f` reproduces opt14 (1.37064 at 64 KiB, 1.32765 at 16 KiB).
  - Refresh-schedule frames round-trip through libzstd (`VERIFY=1` on the 1/50 sample at offset 7).
- **Exploration** used the 1/50 sample: every 50th block at offset 0, 2016 blocks at 64 KiB. On it, opt16 = 1.37211
  and L16 = 1.37175, so L16 is −0.027 % vs opt16.
- **Final candidates** ran on the **full corpus** (100,754 blocks at 64 KiB, 397,925 at 16 KiB), split into thirds
  or quarters that were then summed.
- **Priors were trained with block-disjoint cross-fitting.** Kind tables are opt16's final LL/ML/OF histograms
  (and literals), summed per kind and scaled to 65536.
  - Blocks with `idx % 50 < 25` use tables trained on offsets {35, 40, 45}.
  - The other blocks use tables trained on {10, 15, 20}.
  - So no block is ever scored with tables that saw it.
  - A harsher check trains on one mod and evaluates on the other (noble-skyrim-2k ↔ smim-se; NIF exists only in
    smim, so there NIF falls back to the global table).
  - 16 KiB has its own tables.
- **Corpus kinds** (blocks): DXT5 `_n` 55.5 k, DXT1 29.2 k, DXT5 other 10.7 k, NIF 5.2 k, uncompressed 0.2 k. The
  corpus has **no BC7 or other DX10 formats**, so per-kind gains for those are unmeasured.
- **Schedule notation.** The seed comes first, then each pass, then the final pass `f` (optLevel 2).
  - Seeds:
    - `B`: block-init, as opt16;
    - `P`: `OPT_PRIOR` plus cover literals, as opt14;
    - `K`: per-kind prior plus cover literals;
    - `H`: kind from the header only, so DXT5 and DXT5 `_n` are merged.
  - Passes:
    - `c`: cheap optLevel-0 pass;
    - `Ew<n>`: cheap pass with `n` refresh slices per segment;
    - `u<k>`: run only the first `k` slices;
    - `Dw<n>`: final pass with refresh;
    - `ck<n>` / `ch<n>`: histogram over 1/n of the segments, or the first 1/n of each segment.
- **GPU.** Not run. Every GPU number below is an estimate from the measured M5 per-pass times plus the CPU-measured
  lane-imbalance factor, and is labelled as such. No GPU contention applies.

## 1. The mechanism: in-pass price refresh (new oracle)

**Exact definition** (`dp_pass_refresh` in the scratch core):
- Each segment's series run in slices. Slice `j` holds the series whose **start** has an in-segment offset in
  `[j·L/n, (j+1)·L/n)`, with L = 4096. A lane stops at its first series start at or past the boundary.
- After slice `j`, the sequences that all 16 segments committed in that slice are histogrammed. They use
  segment-local offBase codes and include their literal bytes; a segment that finished also adds its trailing
  literals.
- Slice `j+1` runs with `Prices::from_hist(slice j's histogram alone)`. Pooling all earlier slices was measured worse:
  −0.02 % on the sample.
- The pass output is the normal fixed-up block parse. The final pass is priced from it exactly as today, by
  `Hist::of_output`. Pricing from the pooled slice histograms instead costs only 0.005 % (sample: +0.044 vs +0.050 %),
  so either is fine.

**Why it works.**
- What separates a pass from convergence is the block's own LL/ML/OF statistics, not the literals. Measured on the
  sample, one final pass with each part of the converged statistics swapped in:

  | prices of the single final pass | vs opt16 |
  |---|---:|
  | everything converged | +0.010 % |
  | converged codes, cover literals | −0.014 % |
  | converged literals, prior codes | −0.139 % |
  | kind prior plus cover literals | −0.183 % |

- A refresh every 1/16 segment gives 15 extra convergence steps for the cost of a histogram and a price-table rebuild.
  This is in effect zstd's `ZSTD_updateStats`, pooled over the workgroup at fixed points so it stays deterministic.
- Unscaled slice counts matter: the `+1` smoothing on small counts helps. The same slices scaled ×n gave only
  +0.008 % instead of +0.026 % (w8, sample). Smoothing a full-block histogram (dividing its counts) changed nothing.
- With the refresh the seed hardly matters. On the full corpus at 64 KiB: `B.Ew16.f` 1.37216, `P.Ew16.f` 1.37211, and
  `K.Ew16.f` 1.37213. So the oracle needs no trained tables, which makes it robust to unseen mods. The seed does
  matter for one-pass schedules: `B.Dw16` is −0.38 %.

**GPU fit.**
- K3opt already histograms in-kernel per series (`emit_series` → `hist_seq` into the workgroup `hist`), and the
  `PRICE_MODE 3` prologue already turns a histogram into price tables.
- A refresh is therefore:
  - a workgroup barrier at the slice boundary;
  - `slice = cumulative − snapshot`, or a second 256-word buffer;
  - a 377-entry price rebuild across 16 lanes, about 24 `frac_weight`s per lane;
  - a barrier.
- The cost that matters is **lane imbalance at the barriers**. It was measured with a DP work counter (trips plus
  relaxations, attributed to series starts) on the 1/50 sample, as Σ_slices max_lane against max_lane Σ, the lockstep
  model:

  | slices per segment | 2 | 4 | 8 | 16 | 32 |
  |---|---:|---:|---:|---:|---:|
  | optLevel-0 pass | +1.4 % | +2.9 % | +4.9 % | +8.0 % | +13.3 % |
  | optLevel-2 pass | +3.2 % | +6.5 % | +10.7 % | +16.8 % | +26.0 % |

- The rebuild itself is about 600 lane instructions, roughly 0.1 % of a pass per refresh (estimate).
- Shared memory grows by one slice histogram: 1 KB as u32, 0.5 KB as u16. Today's footprint is 4.6 KB, and
  registers, not shared memory, are the binding limit (estimate: no occupancy loss, to be confirmed with
  `vkstats`).
- A one-slice-lag variant would loosen the barrier: slice j+1 is priced from slice j−1. It was not measured.

## 2. Ranked schedules (full corpus, held-out where priors are used)

Gate at 64 KiB: L16 = 1.37100. Gate at 16 KiB: L16 = 1.32774.

| # | schedule | DP passes | 64 KiB | vs L16 | 16 KiB | vs L16 | est. 5090 µs/block (64 KiB) | est. MB/s, 5090 | vs opt16 throughput | effort |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **`B.Ew16.f`**: block-init seed, cheap pass with 16 refresh slices (256 B), final | 2 | **1.37216** | **+0.084 %** | **1.32840** | **+0.050 %** | ≈ 30.1 | ≈ 2100 | ≈ 1.48× | M |
| 2 | **`B.Ew32u16.f`**: 32 slices (128 B), cheap pass stops at mid-segment, final | 1.5 | **1.37180** | **+0.058 %** | **1.32794** | **+0.015 %** | ≈ 25.9 | ≈ 2450 | ≈ 1.72× | M |
| 3 | `B.Ew8.f`: 8 slices (512 B), less barrier imbalance | 2 | 1.37186 | +0.063 % | 1.32828 | +0.041 % | ≈ 29.7 | ≈ 2140 | ≈ 1.50× | M |
| 4 | `B.Ew32.f` | 2 | 1.37230 | +0.095 % | 1.32842 | +0.051 % | ≈ 30.8 | ≈ 2060 | ≈ 1.45× | M |
| 5 | `B.Ew16u8.f` (half pass, 16 slices) | 1.5 | 1.37152 | +0.038 % | 1.32788 | +0.011 % | ≈ 25.6 | ≈ 2480 | ≈ 1.74× | M |
| 6 | `K.c.c.f`: per-kind prior plus 2 cheap passes | 3 | 1.37157 | +0.041 % | 1.32810 | +0.027 % | ≈ 37.5 | ≈ 1700 | ≈ 1.19× | S |
| 7 | `K.c.f`: per-kind prior, opt14's structure | 2 | 1.37116 | +0.012 % (cross-mod +0.000 %) | 1.32788 | +0.010 % | ≈ 29.3 (= opt14, measured) | ≈ 2170 | ≈ 1.52× | S |
| – | `B.Ew32u12.f` / `B.Ew64u24.f` | 1.375 | 1.37158 / 1.37172 | +0.042 / +0.052 % | 1.32768 / 1.32754 | **−0.004 / −0.015 %** | ≈ 24.5 | | | fails at 16 KiB |
| – | `B.Ew64u16.f` / `B.Ew32u8.f` | 1.25 | 1.37130 / 1.37108 | +0.022 / +0.006 % | 1.32709 / 1.32722 | −0.049 / −0.039 % | ≈ 23.5 | | | fails at 16 KiB |
| – | opt16 today (`B.c.c.c.f`) | 4 | 1.37144 | +0.032 % | 1.32794 | +0.015 % | 44.68 (measured) | 1426 (measured) | 1.00× | – |
| – | opt14 today (`P.c.f`) | 2 | 1.37064 | −0.027 % | 1.32765 | −0.007 % | 29.3 (measured) | 2169 (measured) | 1.52× | – |

**Assumptions behind the 5090 estimates.** All are estimates except the two "measured" rows.
- Non-K3 kernels cost 11.38 µs (K1 6.86, K2opt 2.61, K4 0.60, K5 1.31, all measured).
- A cheap optLevel-0 pass costs 8.2 µs and the final pass with fix-up 9.46 µs (measured, M5).
- A refresh pass costs 8.2 × (1 + imbalance from the table) plus 0.2–0.5 µs of rebuilds.
- A half pass costs half the DP.
- B seeding avoids the 0.86 µs cover-literal step that opt14's first pass pays.
- MB/s = 65536 / µs × 0.97 (opt16's end-to-end factor).

**8 GB class (4060 / 3060), throughput-bound.** Using the k3opt-perf model (other kernels ≈ 65 µs, L0 ≈ 42 µs, L2
≈ 48 µs per block):
- `B.Ew16.f` ≈ 1.5× opt16, about **0.30–0.44 GB/s** from opt16's projected 0.20–0.29.
- `B.Ew32u16.f` ≈ 1.75× opt16, about **0.35–0.51 GB/s**.
- Line rate (1.25 GB/s) is still out of reach from pass count alone. This angle removes about 45 % of opt16's K3 work;
  K1, K2opt and the DP body are now most of the remaining time.

**What else the refresh unlocks.**
- An opt14 class at 1 DP pass: `K.Dw16`, at 1.37045 (+0.159 % over L14) and 1.32644 (+0.029 % over L14). The 16 KiB
  margin is thin; `P.Dw16` reaches +0.136 % and +0.021 %. Estimated K3 ≈ 11–12 µs against opt14's 18.6.
- Schedules above the opt16 ratio for the same 2-pass cost: `B.Ew32.f` is +0.095 % over L16.

## 3. Ideas, one by one

| idea | mechanism | speedup | ratio impact | exactness | effort | risk |
|---|---|---|---|---|---|---|
| **In-pass refresh** (§1), `B.Ew16.f` | refresh prices 15× inside the cheap pass, then the final pass | K3 33.3 → ≈ 18.7 µs; total ≈ 1.48× (estimate) | +0.052 % / +0.035 % vs opt16 (64 / 16 KiB, measured) | new oracle and preset | M: oracle `segment_run` plus slice loop (done in scratch); K3: slice stop, barrier, rebuild, slice histogram | barrier imbalance higher than the CPU proxy; registers for the slice-stop test |
| **Half refresh pass**, `B.Ew32u16.f` | as above; the cheap pass stops at mid-segment | ≈ 1.72× (estimate) | +0.026 % / +0.000 % vs opt16 | new oracle | M (same plus one bound) | 16 KiB has only opt16's margin |
| **Per-kind priors** (DDS FourCC, `_n` suffix, NIF) | 5 kind tables of 121 code entries (plus literals); seed = kind table plus cover literals | none by itself; enables 2-pass `K.c.f` (= opt14 cost, measured 1.52×) or 3-pass `K.c.c.f` (≈ 1.19×) | `K.c.f` +0.012 % vs L16 held-out, **+0.000 % cross-mod**; `K.c.c.f` above opt16 at both sizes | new oracle (tables) | S | priors overfit the corpus; BC7/DX10 unmeasured; header-only (`H`) costs 0.003 % |
| Per-kind priors with refresh | | – | +0.002 % over `B.Ew16.f`: noise | – | – | not worth it |
| Corpus-trained prior with literals | blend a kind literal table into cover literals | – | ±0.003 %: nothing | – | – | – |
| **Exact early termination** | skip later cheap passes when the prices repeat | 0.45 % of blocks at best | none | exact | S | **dead end** |
| Tolerance early termination (`a` pass: skip when the cross-entropy change < ε) | adaptive cheap-pass count | 2.25 average passes at −0.012 % (sample) | worse than the refresh at equal work | new oracle | S | dominated |
| Partial cheap pass over a segment subset (`ck4.ck2.f`) | DP over 1/4 then 1/2 of the segments, then final | 1.75 passes (estimate ≈ 1.6×) | 64 KiB +0.027 % vs L16, but **16 KiB −0.20 %**: at 16 KiB, k4 is segment 0 only, and segment 0 alone is −2.3 % unrepresentative (shortest window) | new oracle | M–L (lane repacking) | **dead end**: block-size dependent and dominated by the refresh |
| Prefix partial passes (`ch4.ch2.f`) | the first 1/k of each segment | 1.75 passes | 64 KiB +0.020 %, 16 KiB −0.016 % vs L16 | | | fails at 16 KiB |
| Price-aware greedy / lazy over K2opt records as the intermediate pass | cheap parse histogram instead of a DP | ≈ 0.1–0.2 of a pass (estimate) | −0.25 % to −0.6 % vs opt16 (sample) | | | **dead end** |
| K2opt-only histogram (record codes, raw or calibrated per kind) as the code seed | no parse at all | ≈ free | −0.7 % (NIF loses rep codes) | | | **dead end** |
| Neighbour seed: the previous block's *converged* histogram | sequential within a file | – | `Nf.f` −0.043 % vs opt16 (1 pass, sample); `Nc.c.f` −0.009 % vs `K.c.f` −0.031 % | output stays block-independent, but **creates a sequential dependency across a batch** (block i waits for block i−1's final pass) | M | **dead end** for a batch GPU |
| Neighbour pooling, parallel-safe (add block i−1's same-pass histogram) | | – | −0.011 % vs own only (sample) | | | dead end |
| Blend the prior into the pass-1 histogram | | – | −0.015 % to −0.044 % (sample) | | | dead end |
| Extrapolate (h1 + a(h1 − prior)) | | – | −0.05 % to −0.35 % | | | dead end |
| optLevel-2 cheap pass (`K.f.f`) | the cheap pass with the btultra flow | +1 µs | +0.002 % to +0.006 % | | | negligible |
| Refresh inside the final pass after a good cheap pass (`P.Ew16.Dw2`) | | – | −0.02 % to −0.1 % | | | hurts: slice noise beats staleness once prices are good |

## 4. Dead ends checked (summary)

- **Exact termination.** Prices are identical going into opt16's cheap pass 1, 2 and 3 for 0.00 %, 0.10 % and
  0.45 % of blocks.
- **One pass reaching L16 from any static seed.** Even the block's own converged statistics give only +0.010 % in
  one pass. Per-kind priors give −0.168 % vs opt16. Neighbour-converged statistics give −0.043 %.
- **Cheap parse-free or parse-light estimators** (greedy, lazy, candidate histogram): far too inaccurate on code
  statistics.
- **Neighbour seeding.** It is either sequential (useless for a batch) or noisy (worse).
- **Segment-subset partial passes.** Segment 0's statistics are unrepresentative, which breaks 16 KiB.
- **Blending, extrapolation and smoothing of full-pass histograms.**

## 5. Top-3 recommendation

1. **Build `B.Ew16.f` as the new L16 preset** (or `B.Ew8.f` if the GPU barrier cost turns out higher than the CPU
   proxy says).
   - It needs one cheap refresh pass plus the final pass: 2 DP passes instead of 4.
   - It stays above opt16's ratio at every size: +0.084 % / +0.050 % over L16, against opt16's +0.032 % / +0.015 %.
   - It needs no trained tables.
   - Estimated 1.48× opt16 throughput on the 5090 (≈ 2.1 GB/s) and about 1.5× on 8 GB cards.
   - First step: port `dp_pass_refresh` (scratch, about 120 lines) into the oracle. Then add slice stop, barrier and
     rebuild to K3opt behind `PRICE_MODE`, and measure the barrier cost on the GPU against `k3opt_passes_timing`.
2. **Then try the 1.5-pass `B.Ew32u16.f`** for the 8 GB class: ≈ 1.72× opt16, with ratio equal to opt16 at 16 KiB
   (+0.015 % over L16) and +0.058 % at 64 KiB.
   - If 16 KiB must keep more margin, pick `u16` at 64 KiB and full `Ew16` at 16 KiB. The schedule is a per-preset
     constant.
3. **Use the refresh for opt14 too** (`K.Dw16`, or `P.Dw16` without new tables).
   - That is 1 DP pass at L14 + 0.13–0.16 % (64 KiB) and +0.02–0.03 % (16 KiB).
   - Estimated K3 ≈ 11–12 µs against 18.6, so roughly 1.25–1.3× opt14 throughput.
   - Per-kind priors are worth adding only here, where a one-pass schedule makes the seed matter: +0.023 % at
     64 KiB. With any refresh pass in the schedule they add nothing measurable.

Not recommended:
- `K.c.f`, the cheapest change (S effort, opt14 speed). It passes L16 only with in-corpus priors (+0.012 %) and ties
  it with cross-mod priors.
- `K.c.c.f` (S, ≥ opt16 ratio). Fine as a stop-gap, but the refresh dominates it.
