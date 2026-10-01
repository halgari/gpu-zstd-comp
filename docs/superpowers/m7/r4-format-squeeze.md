# r4-format-squeeze: get the ratio from the format so the parse can be cheap

Date: 2026-10-01. Agent r4-format-squeeze. CPU research only; no repository file was changed.

**Where things are.**
- Scratch workspace: `~/.cache/gzc-m7/r4-format-squeeze/`. It holds a copy of `gzc-core` plus the harness crate `r4`.
- Artifacts: `.superpowers/m7-research/artifacts/r4-format-squeeze/`:
  - `r4-src/` — the harness: `ent.rs` (frame writer), `reopt.rs` (post-parse passes), `lzc.rs` (lazy parse), `cands.rs` (candidates), `main.rs`;
  - `gzc-core-scratch.diff` — a10's helpers plus one environment-gated price knob used for a dead-end test;
  - `runs/` — every full-corpus and sample log.

## Headline

**The cheapest parse that reaches L16 is one DP pass: Prior seed, one level-2 pass, on a06's h10 candidates.** Three cheap stages follow it:

- a new post-parse **"drop" pass**, which removes short matches that cost more than the literals they replace;
- an optional rep-offset re-optimisation;
- a **multi-block frame with shared tables**, which builds on a10's split.

Full corpus, every frame decoded by libzstd:

| block | 1-pass DP + h10 + drop/rep pass + shared split | vs L16 | vs L14 | opt16 today (4 passes) |
|---|---:|---:|---:|---:|
| 64 KiB | **1.37425** | **+0.237 %** | +0.437 % | +0.032 % |
| 32 KiB | 1.35313 | +0.214 % | +0.383 % | +0.099 % |
| 16 KiB | **1.32896** | **+0.092 %** | +0.218 % | +0.015 % |

- That is **one DP pass instead of four, at 3–7× opt16's ratio margin.**
- Projected on the 5090: about 24–28 µs/block against 44.7, which is **1.6–1.9×**, before any of the A1–A4 kernel speedups. 8 GB cards gain more, about 2×, because K3 is a larger share of their time.
- Stock libzstd decodes these frames 2–4 % slower than opt16's single-block frames.

**A surprising finding.** All the DP parses select too many 3-byte explicit matches. A cheap pass after the parse drops them when, at the parse's own final prices, the literals are cheaper. It gains:

| parse | gain from the drop pass alone |
|---|---:|
| 1-pass DP | +0.165 % (full corpus) |
| opt14 (`pcf`) | +0.098 % (sample) |
| opt16 | +0.056 % (sample) |

Its work is about 1 % of one DP pass. Each 4 KiB segment can be one lane, and the segment-local version loses nothing.

## Method

**Harness `r4`.**
- It runs the oracle's own `find_cands`, `seed_prices` and `dp_pass` (the a10 schedule DSL: `p`/`b` seed, then `c`/`f` passes). `pcf` is opt14 byte for byte: its full-corpus base is 4 737 822 257 B, the same as documented.
- h10 candidates use a06's recipe (h4 d8 + h3 d4 + h10 at every 4th position, d16). `pcf10` reproduces a06's full-corpus 1.37239 exactly.
- `l9` is `LVL9S12SEG`. Two lazy and greedy parses over opt candidates are in `lzc.rs`.
- The post-parse passes are in `reopt.rs`. The frame writers are in `ent.rs`; they extend a10's `split.rs`.

**Samples and corpus.**
- Exploration used the 1/50 sample: 2016 blocks at 64 KiB (L16 1.37175, L14 1.36911), and 7959 blocks at 16 KiB (L16 1.32513, L14 1.32345).
- Finalists ran on the **full corpus**: 100 754 blocks at 64 KiB and 397 925 at 16 KiB, against L16/L14 of 1.37100/1.36827, 1.35025/1.34797 and 1.32774/1.32606.

**Verification.** Every frame in every sample and full run was decompressed by libzstd and compared with the source.

**Threads and load.** Everything ran on 6 threads. The machine load average was 25–45. CPU timings are thread CPU time, min of 3. Decode timings are best-of-7–9 and interleaved.

**Configuration names used below.**
- Post-parse passes: `ds` = drop only (segment-local); `dsg` = drop, then R1 (beam 4, nearest-cover offsets), R2 and R1 again, all segment-local; `dsg2` = `dsg` plus a second drop round.
- Frame configurations:
  - `sp.a10` — a10's split writer;
  - `sp.all` — the split writer with every per-table option in §2b;
  - `sh` — the shared-table writer: estimated cuts on a 2 KiB grid with header base 25 B, per-stream table segments, all options.

## 1. Post-parse re-optimisation (fixed match/literal segmentation, then small moves)

**Mechanism.** All of these passes run on the parse's own (ll, ml, offset) list and use integer frac prices taken from that parse's histogram (`Prices::from_hist`, already in gzc-core). The output is re-encoded with true reps.

- **R1, rep-aware offset choice.** For each match, the choices are:
  - the original offset;
  - the nearest offset in the chains that covers the whole match (`nearest_cover`);
  - any current rep that reproduces the match, including the `ll0` cases: rep1/rep2 shift and `rep0 − 1`.

  A Viterbi pass over rep-history states, keeping the best 4–16 states, picks the cheapest offBase sequence. Writing an offset explicitly when it equals a rep is never better, so it is not offered.
- **R2, boundary moves.**
  - Extend a match backwards into its literal run.
  - Insert rep matches (any of the three, with ll0 semantics where it applies) into literal runs, trimming the start of the next match if needed.
  - Each move is greedy with a strict gain, under true reps.
- **Drop, the new pass.** For each explicit match of 6 bytes or less, compare:
  - keep: price(seq j) + price(seq j+1 with the reps after j);
  - drop: the literal prices of its bytes + price(seq j+1 with the merged literal run and the old reps).

  Drop when that is strictly cheaper.
  - 97 % of drops are 3-byte matches.
  - pf10 drops about 250 matches per block, 4 % of its sequences.
  - About 10 % of drops turn the next sequence into a cheaper rep.

  The rest come from prices: the DP priced each 3-byte match with the previous pass's (or the prior's) statistics, which make them look cheaper than they are in the final parse. A blanket 3-byte penalty inside the DP does not reproduce this; it costs 0.01–1.3 % (dead ends). The decision has to be local.

**Ratio.** 1/50 sample, 64 KiB, base frames; Δ is against the parse alone.

| parse | alone | + R1 (alias only) | + R1 (+ nearest cover) | + R1/R2 | + drop | + drop and R (`dsg`) |
|---|---:|---:|---:|---:|---:|---:|
| lvl9s12seg | 1.34004 | +0.047 % | +0.048 % | +0.171 % | +0.019 % | **+0.189 %** |
| pf10 (1 pass) | 1.36898 | +0.010 % | +0.015 % | +0.034 % | +0.167 % | **+0.208 %** |
| pf (1 pass, opt candidates) | 1.36727 | | | +0.030 % | +0.164 % | |
| pcf (opt14) | 1.37118 | +0.013 % | +0.018 % | +0.028 % | +0.098 % | **+0.131 %** |
| opt16 | 1.37211 | | | +0.025 % | +0.056 % | |

**Other R1 findings.**
- Beam width on `pcf`: 1 → +0.004 %, 2 → +0.012 %, 4 → +0.016 %, 16 → +0.018 %. History shaping matters, so use beam 4.
- The segment-local versions (4 KiB, incoming reps taken from the original parse) lose nothing: `pcf+dsg` 1.37297 against 1.37301 for whole-block.
- Accurate Huffman/FSE prices in the drop pass add nothing (−0.003 %).
- A second drop round (`dsg2`) adds +0.016–0.026 %.

**Full corpus, 1-pass DP (pf10), 64 / 16 KiB:**

| step | 64 KiB | 16 KiB |
|---|---:|---:|
| base | 1.36839 | 1.32584 |
| `ds` | 1.37065 (+0.165 %) | 1.32738 |
| `dsg` | 1.37121 (+0.206 %) | 1.32784 |
| `dsg2` | 1.37147 | 1.32801 |

On opt14 (`pcf`), `dsg` takes 1.37064 to 1.37239 at 64 KiB (+0.128 %) and 1.32765 to 1.32892 at 16 KiB.

**GPU cost.** Measured work per 64 KiB block for pf10+dsg, against a DP-pass proxy of 484 k units (3 rep probes + (lenB − 2) length evaluations per position):

| step | work per block | per lane (16 segments) | parallel shape | est. µs (5090) |
|---|---|---|---|---:|
| drop | 6.3 k sequence visits, about 40 ops each | about 390 visits, serial | about 1 % of a DP pass's dependent chain | **0.2–0.4** |
| R1, beam 4, run twice | 17 k transitions + 45 k cover checks | about 1 k transitions | the 4 beam states can run on 4 threads | 0.5–1 |
| nearest-cover walk | 25 k chain steps | – | parallel per sequence, K2-like | 0.2–0.4 |
| R2 | 105 k rep probes over 35 k literal positions | – | probes parallel per position; only the pick is serial | 0.4–0.8 |
| price rebuild (histogram + `from_hist`) | | | the same step as between DP passes | ≈ 0.3 |

- Total for `dsg`: **about 1.5–3 µs**. Drop only: **about 0.5 µs**.
- A DP pass costs 8.2–9.2 µs.
- Memory: the sequence list (already in K3/K4) and the chains (already in K1), plus 4 × 3 u32 of rep states per lane.

**Byte-exact oracle.**
- Integer prices.
- One lane per 4 KiB segment of sequence start positions. Rep state at a segment's first sequence comes from the parse's own true-rep encoding; K3 already produces it.
- Strict-gain rules and fixed tie order: for R1 the beam is sorted by (cost, reps lexicographic), offsets ascending.
- Compaction (merging dropped matches into the next literal run), then the existing `encode_raw` true-rep re-encode.

**Effort and risk.** Drop alone is S. R1/R2 is M. The risk is low: the passes cannot break a frame, because offsets are only replaced by ones verified to cover the match.

## 2. Block splitting done well

### 2a. Cut choice

1/50 sample at 64 KiB. All rows use the per-table options; Δ is against a10's estimator.

| cut method (`pcf`) | ratio | Δ | blocks/frame |
|---|---:|---:|---:|
| a10 estimate DP (4 KiB cells, header 40 + 0.3/sym + 0.4/code) | 1.37392 | – | 1.95 |
| same on 2 KiB cells | 1.37404 | +0.009 % | 1.99 |
| at most 2 / 3 / 4 / 8 parts | 1.37347 / 375 / 384 / 391 | −0.033 / −0.012 / −0.006 / −0.001 % | |
| 8 KiB cells | 1.37374 | −0.013 % | |
| header base 30 or 55 | ±0.002 % | robust | |
| even halves / quarters / best-of-{1, 2, 4} | 1.37219 / 1.36982 / 1.37275 | −0.13 / −0.30 / −0.09 % | |
| exact DP on 4 KiB cells (pf10+r) | | +0.006 % over the estimate | |
| estimate + exact local refine (±2 KiB, 512 B steps) | | +0.016 % over the estimate | |

**Shared tables, the new writer (`write_shared_frame`).**
- Given the cuts, each stream gets its own DP over the blocks. The streams are literals, LL, OF and ML.
- A segment of consecutive blocks shares one table built from their merged histogram, with best depth, log and normaliser:
  - literals use type 2 in the segment's first block and treeless in the rest;
  - FSE streams use Compressed in the first block and Repeat in the rest;
  - a lone block may use its own Raw, RLE or Huffman section, and FSE streams may use RLE.
- Because sharing makes a cut cheaper, the cut estimator's header base drops from 40 to 25 B.

| writer (sample, 64 KiB) | pf10+r | pcf | pf10+dsg (16 KiB) |
|---|---:|---:|---:|
| `sp.all` (a10 cuts, independent tables) | 1.37202 | 1.37392 | 1.32628 |
| `sh.all` (a10 cuts, shared) | 1.37251 (+0.036 %) | 1.37436 (+0.032 %) | 1.32632 |
| **`sh.g2k.h25`** (2 KiB cells, base 25, shared) | **1.37274 (+0.053 %)** | **1.37460 (+0.050 %)** | **1.32640** |
| `sh` with base 15 (3.3 blocks/frame) | 1.37277 | 1.37457 | 1.32577 (worse) |
| per-stream DP cuts, union | 1.37181–1.37246 | | – (dead) |

**Full corpus, `pcf`:**

| block | `sp.a10` | `sp.all` | `sh` |
|---|---:|---:|---:|
| 64 KiB | 1.37320 | 1.37336 | **1.37405** (+0.050 % over a10) |
| 16 KiB | 1.32838 | 1.32865 | 1.32878 |

**Blocks per frame.**
- The estimator picks 2.1–2.2 blocks per frame at 64 KiB and 1.2 at 16 KiB.
- Capping at 4 parts costs 0.01 %, and 2 parts costs 0.03 %.
- Shared tables use treeless literals in about 33 % of blocks, and Repeat in about 6 % of FSE streams.

**GPU cost of each cut method.** Measured against K4+K5's 1.9 µs/block.

| method | GPU structure | est. cost |
|---|---|---|
| even cuts | none | 0, but it loses 0.1–0.3 % |
| a10 estimate on 16 cells | per-cell histograms (16 × 377 bins, u16 in workgroup memory: 12 KB) → cumulative → 136 interval costs (`log2_x256` table) → 16² DP on one thread | ≈ 0.2–0.4 µs |
| 32 cells (2 KiB grid) | 528 intervals, about 0.2 M ops, 24 KB of u16 histograms, or two passes | ≈ 0.4–0.8 µs |
| exact DP or refine | 10–20 full table builds and encodes per block | ≥ +5 µs; not worth +0.006–0.016 % |
| shared-table DP, at most 4 parts | ≤ 10 candidate segments per stream, each one table build from a merged histogram plus a cross-entropy sum | +1–2 µs; the Huffman builds are serial per candidate but independent, so one subgroup per candidate |

On the CPU (single thread, unoptimised, f64), per block:

| step | time |
|---|---:|
| `write_frame` | 292 µs |
| split writer, all options | 589 µs |
| shared writer, all options | 1073 µs |
| estimated cuts, 4 KiB / 2 KiB cells | 336 / 1156 µs |

The GPU estimates above come from the parallel structure, not from these CPU ratios.

**Byte-exact oracle.**
- Port the estimator to integer `log2_x256` costs in 1/256 bytes.
- Fixed tie order: earliest cut, fewest parts.
- The shared DP's costs are already integers (Huffman payload bits, `cost_x256`).
- **Bug to carry over correctly.** After a part is emitted as a Raw block, later blocks must re-encode their offBases against the *decoder's* reps, because Raw blocks do not update them.
  - a10's `split.rs` did not do this.
  - Its output was still correct in practice, because a10 always took the smaller of the split and the single-block frame and verified the result.
  - It shows up with finer grids. The `r4` writer re-encodes per part.

**Effort and risk.** Split is M. Shared tables add M on top of it. There is no extra risk beyond a10's split. Decode cost is below.

### 2b. Per-block mode choices

All rows are relative to `sp.all`. The 64 KiB column is the sample for `pcf`; the 16 KiB column is the sample for pf10+dsg inside `sh`.

| option removed | 64 KiB | 16 KiB |
|---|---:|---:|
| FSE cost-optimal normalisation | −0.012 % | −0.018 % |
| FSE table-log sweep (5..max) | −0.003 % | −0.018 % |
| Huffman depth sweep (5..11) | −0.006 % | −0.017 % |
| single Huffman stream | −0.001 % | −0.004 % |
| treeless or Repeat, outside the shared writer | −0.001 % each | |
| best weight description (direct or FSE) | 0 | |
| all of the above (libzstd-like choices, still shared) | −0.020 % (pf10+dsg) | **−0.056 %** |

- **At 16 KiB, per-table choices are worth half the entropy-side gain**, because frames rarely split there.
- Full corpus, pf10+dsg2 with libzstd-like choices: 64 KiB +0.233 %, 16 KiB only +0.049 % over L16.

## 3. Literal section

| choice | finding |
|---|---|
| **1 vs 4 streams** | One stream is legal only with Size_Format 00, where both sizes are below 1024. So it can only apply to small parts (256–1023 literals). It is picked in about 1–3 % of sections and gains +0.001 % (64 KiB) or +0.004 % (16 KiB). It slows decode for those sections only. **Not worth a code path.** |
| **Huffman max depth** | Sweeping 5..11 (with the smaller description) gives +0.006 % at 64 KiB and +0.017 % at 16 KiB. On the GPU, the tree build is shared and only `setMaxHeight` plus a 256-symbol bit count runs per depth: 7 candidates, one per subgroup, about +0.2–0.4 µs. **Keep it for 16 KiB.** |
| **Raw vs compressed** | Already chosen by exact size in every writer. Raw is picked in fewer than 1 % of sections; nothing to gain. |
| order-0 gap | a10 measured 0.64 % between Huffman and the order-0 bound. It is integer code lengths plus headers, and the format cannot close it. |

## 4. Sequence section

- **FSE table log and normalisation.**
  - A cost-optimal normalisation (greedy marginal gain over `count × log2((n+1)/n)`, then cell swaps) gains +0.012 % (64 KiB) and +0.018 % (16 KiB).
  - Adding a log sweep of 5..max gains +0.003 % and +0.018 %.
  - An integer port uses the `FRAC`/`log2_x256` table.
  - On the GPU this is 3 streams × 5 logs × 2 normalisers = 30 independent candidates, one thread each, about 2–6 k ops each: about +0.3–0.6 µs.
- **Predefined vs compressed vs RLE vs Repeat.** These are chosen exactly per block from the cost estimate. Predefined wins in fewer than 0.3 % of streams. Repeat matters only inside the shared writer.
- **Accuracy.** The FSE table log is capped at 9/8 by the format. Nothing further is available.

## 5. Combinations with cheap parses

"Everything" means the parse, then `dsg`, then the shared-table writer `sh` (2 KiB cells, header 25, all options). All rows are full corpus unless marked (s), which means the 1/50 sample against the sample's own L16/L14. Each ratio cell gives the ratio, then (vs L16 / vs L14).

| parse | DP passes | 64 KiB, parse only | 64 KiB, + everything | 16 KiB, parse only | 16 KiB, + everything |
|---|---:|---:|---:|---:|---:|
| lvl9s12seg | 0 | 1.33926 | 1.34540 (−1.87 / −1.67 %) | 1.30040 | 1.30531 (−1.69 / −1.56 %) |
| lazy2, 3-byte, h10 candidates (s, HB3 = 13) | 0 | 1.35096 | 1.35592 (−1.15 / −0.96 %) | 1.31413 | 1.31800 (−0.54 / −0.41 %) |
| greedy-longest over opt candidates (s) | 0 | 1.33345 | 1.33626 (−2.6 %) | | |
| 1-pass DP `pf` (opt candidates) | 1 | 1.36727 (s) | 1.37250 (+0.110 / +0.309 %) | | 1.32794 (+0.015 / +0.142 %) |
| **1-pass DP `pf10` (h10)** | **1** | **1.36839** | **1.37425 (+0.237 / +0.437 %)** | **1.32584** | **1.32896 (+0.092 / +0.218 %)** |
| 1-pass `pf10`, drop only + `sh` | 1 | | 1.37370 (+0.197 / +0.397 %) | | 1.32851 (+0.058 / +0.185 %) |
| 1-pass `pf10` + `dsg2` + `sh` | 1 | | 1.37446 (+0.252 / +0.452 %) | | 1.32912 (+0.104 / +0.231 %) |
| 1-pass `pf10` + `sh` only (no post-parse pass) | 1 | | 1.37167 (+0.049 %) | | 1.32700 (**−0.056 %**) |
| opt14 `pcf` | 2 | 1.37064 | 1.37558 (+0.334 / +0.534 %) | 1.32765 | 1.33002 (+0.172 / +0.299 %) |
| opt14 + h10 `pcf10` | 2 | 1.37239 | **1.37743 (+0.469 / +0.670 %)** | 1.32977 (+dsg) | **1.33089 (+0.237 / +0.364 %)** |

At 32 KiB: `pf10` + everything is 1.35313 (+0.214 %), and `pf10` + `ds` + `sh` is 1.35263 (+0.176 %).

- Other 1-pass variants on the 64 KiB sample: level-0 pass `pc10` +0.208 %; 2 KiB DP segments +0.202 %; targetLength 16 +0.220 %. **All of them clear L16 by about 0.2 %.**
- **The lazy family cannot reach L16.**
  - Even with 3-byte matches, h10 candidates, all-rep probes and everything else, it is about 1.2 % short at 64 KiB.
  - The drop pass does almost nothing for lazy parses (+0.02 %), because they have no 3-byte matches.
  - For the 10 Gbit preset, lvl9s12seg + everything is still +0.46 % over lvl9s12seg alone, which is +0.56 % over libzstd L9.
- **The 1-pass DP needs both h10 and the post-parse pass at 16 KiB.**
  - Without h10, 16 KiB is only +0.015 %.
  - Without the post-parse pass, 16 KiB is −0.056 %.
- The opt14 + h10 + everything row (+0.47 % / +0.24 %) has margin to spend elsewhere, for example shallower chains, 2 KiB DP segments or libzstd-like table choices.

## 6. Cost of the frame/entropy stage on the GPU, and the whole pipeline

K4+K5 today: 1.9 µs/block on the 5090.

| addition | est. µs/block | basis |
|---|---:|---|
| post-parse price rebuild + drop | 0.5 | about 1 % of a DP pass's dependent chain |
| R1 (beam 4) + nearest cover + R2 | +1–2.5 | parallel probes plus short serial picks (§1 counts) |
| cut estimate, 2 KiB cells | 0.4–0.8 | ≈ 0.2 M integer ops; 24 KB of workgroup memory |
| per-part table builds (×2.2 parts) | +0.6–1.0 | K5's serial thread-0 Huffman build and K4's FSE builds, repeated per part |
| Huffman depth sweep + FSE log/normaliser sweep | +0.5–1.0 | 7 + 30 independent candidates, one subgroup or thread each |
| shared-table DP, at most 4 parts | +1–2 | ≤ 10 merged-histogram table builds per stream |
| **entropy stage total** | **≈ 4–6.5** (today 1.9) | |

**Pipeline on the 5090** (projected, not measured):

| preset | K1+K2 | K3 | post-parse | K4/K5 + split | total | throughput |
|---|---:|---:|---:|---:|---:|---:|
| opt16 today | 9.5 | 33.3 | – | 1.9 | 44.7 µs | 1.43 GB/s |
| **pf10 + dsg + sh** | 9.2 (a06) | 9.2 (1 pass) | 1.5–3 | 4–6.5 | **≈ 24–28 µs** | **≈ 2.3–2.7 GB/s (1.6–1.9×)** |
| pf10 + ds + shared split with libzstd-like table choices | 9.2 | 9.2 | 0.5 | 3–4 | ≈ 22–23 µs | ≈ 2.9 GB/s (2×). Ratio is +0.15–0.23 % at 32–64 KiB, but ≤ +0.05 % at 16 KiB (pf10 + dsg2 with those choices measured +0.049 %), so not robust |

- A1–A4's K3 cut (30–40 %) applies to the single pass as well.
- **8 GB class.** K3 is about 75 % of opt16's time there, so the new preset runs at roughly 0.45–0.5× opt16's time, about 2×: **≈ 0.4–0.6 GB/s against 0.20–0.29**.

## Decode speed (stock libzstd, 1 thread)

Measured on the 1/50 sample, interleaved, best of 7–9, on a loaded machine (±3 %).

| frames | 64 KiB | 16 KiB |
|---|---:|---:|
| opt16, 1 block (reference) | 975 MB/s | 847 MB/s |
| pf10 + dsg, 1 block | 983 (+1 %) | 863 (+2 %) |
| **pf10 + dsg + `sh` (2.1 / 1.2 blocks/frame)** | **954 (−2 %)** | **826 (−2.5 %)** |
| any parse: `sh` against its own 1-block frame | −3 to −5 % | −1 to −4 % |
| `sh` with header 15 (3.3 blocks/frame) | −7 % | |
| even quarters (4 blocks/frame) | −13 % | |

- The drop pass has fewer sequences (6060 against 6308 per block), so it decodes slightly faster. That cancels part of the split's cost.
- Keep cuts at about 2 blocks per frame. Cap at 4 parts (−0.006 % ratio) if the downloader's decode budget is tight.

## Dead ends (measured)

- **Lazy2 or greedy over opt candidates**, even with 3-byte matches, h10 and every format squeeze: −1.15 % (lazy) and −2.6 % (greedy) against L16 at 64 KiB.
- **A 3-byte-match price penalty inside the DP**, as an alternative to the drop pass: −0.01 % at 256/256 bit, and −0.1 % to −1.3 % with larger penalties.
- **Accurate (real Huffman/FSE) prices for the drop pass:** −0.003 %.
- **Per-stream union cuts, and low header constants that give 3–5 blocks per frame:** ratio is worse (−0.03 % to −0.07 % at 16 KiB) and decode is slower.
- **Exact cut DP or exact refine:** +0.006–0.016 % for ≥ 10× the writer work.
- **Even cuts:** −0.13 % to −0.30 %.
- **Single Huffman stream:** ≤ +0.004 %.
- **Frame header without FCS** (window descriptor instead): +0.002 % at 64 KiB. It needs a decoder that does not depend on `ZSTD_getFrameContentSize`. Not worth it.
- **R1 with beam 1:** only +0.004 %. **R1 without the drop pass on DP parses:** ≤ +0.03 %.

## Top 3

1. **New L16 preset: 1-pass DP on h10 candidates, then drop + R1/R2, then shared-table split.**
   - Full corpus: +0.237 % (64 KiB), +0.214 % (32 KiB), +0.092 % (16 KiB) over L16.
   - 1 DP pass, about 1.6–1.9× opt16 on the 5090 and about 2× on 8 GB cards (projected).
   - Decode −2 %.
   - Effort M–L. It needs a new post-parse kernel (segment lanes), a cut estimator, a multi-block K4/K5, and an integer oracle for each.
2. **Ship the drop pass (with R1/R2) first, on today's presets.**
   - opt14 1.37064 → 1.37239 (+0.128 %), already above L16 with no entropy changes.
   - lvl9s12seg +0.19 %.
   - About 0.5–3 µs, and it does not touch K3. It is the cheapest ratio per µs found.
3. **Entropy stage = a10's split + shared tables + per-table sweeps** (FSE optimal normalisation and log sweep, Huffman depth).
   - +0.05 % over a10's split at 64 KiB.
   - At 16 KiB the sweeps are worth +0.05 % and decide whether the 1-pass preset clears L16 there.
   - Cap the frame at 4 parts.

## Reproduce

```sh
cd ~/.cache/gzc-m7/r4-format-squeeze
CARGO_TARGET_DIR=$PWD/t64 cargo build --release --offline -p r4      # 16 KiB: --no-default-features --features block-16k
./t64/release/r4 eval pcf pf10+dsg                                  # all frame configs, 1/50 sample, verified
ONLY=sh.g2k.h25 ./t64/release/r4 eval pf10 pf10+ds pf10+dsg l9+dsg  # parse x post-pass
VERIFY=1 EVERY=1 ONLY=sp.a10,sp.all,sh.g2k.h25 ./t64/release/r4 full pf10+dsg   # full corpus
./t64/release/r4 work pf10+dsg; ./t64/release/r4 ptime; ./t64/release/r4 dec2 opt16 pf10+dsg
```
