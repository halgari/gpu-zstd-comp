# r1-parallel-parse: parse algorithms for parallel hardware

Agent r1-parallel-parse, 2026-10-01. Research only: no repository file was changed.

**Code.** The source and raw outputs are in `.superpowers/m7-research/artifacts/r1-parallel-parse/`.
- `src/dp.rs` is a parameterised, instrumented port of `opt::segment_ring`. It takes an arbitrary
  window (ip0, anchor0, reps0, iend, ilimit).
  - `validate` mode checks that 4 KiB windows reproduce `opt::passes` exactly, every pass: **2016/2016 sample
    blocks identical**.
- `src/plan.rs` holds the window plans (fixed, overlapping, anchored) and the pass schedules.
- `src/stat.rs` holds the static min-plus model, the lazy start parse and the coalescence statistics.
- Every new schedule's frames were decoded by libzstd: 0 mismatches on the sample (`verify` mode).

**Samples and baselines.**
- "Sample" is the 1/50 sample (2016 blocks). On it, L16 = 1.37175 and L14 = 1.36911.
- My reproduction of the known baselines matches exactly:
  - opt16: 1.37211;
  - opt16 + gap3: 1.37241;
  - Prior x2 + gap3: 1.37215 on the sample, and on the full corpus 1.37162 at 64 KiB and 1.32826 at 16 KiB.
    These are a05's numbers.

**Schedule notation.**
- `P` / `B` is the seed (Prior or BlockInit). Each `level:plan` that follows is one pass.
- `S<W>g3` = fixed W-byte segments with gap3.
- `Gl2,s4096` = a priced lazy2 start parse over the opt candidates, with reps, in 4 KiB segments.
- `A<W>p<phase>sak` = anchored-window refinement (§1).
  - `s` = window start reps from the previous parse.
  - `a` = start anchor at the previous literal run.
  - `k` = keep-if-cheaper.
  - `~n` = n passes with alternating phase.

**Cost units.**
- "cost" = DP positions stepped + relaxation iterations, the same unit as a05.
- "chain" = per-block sum over passes of the costliest window/lane, i.e. the dependent chain.
- opt16 today: cost 416 K per block, chain 30.8 K on average and 82 K at most (sample).

## TL;DR

1. **Anchored-window refinement (AWR) is the finding.** It is a monotone local-search parse made of small,
   independent DP windows, and it beats opt16's ratio with fewer DP passes and 8–60× more lanes.
   - Start from a cheap lazy2 parse. Then run k passes in which every window of W bytes re-parses with today's
     DP, between two **nodes of the previous parse**.
     - Its start state (reps and literal-run anchor) comes from the previous parse.
     - The window keeps its new path only if that path prices ≤ the old one.
     - Windows are independent, one lane each. Boundaries alternate by W/2 between passes.

   | | 64 KiB full corpus | 16 KiB full corpus |
   |---|---|---|
   | lazy + 1 AWR pass | **−0.035 %** vs L16 | −0.003 % |
   | lazy + 2 AWR passes (W = 512) | **+0.079 %** | +0.079 % |
   | lazy + 3 AWR passes | **+0.124 %** | +0.110 % |

   - For comparison:
     - opt16 (4 DP passes) is +0.032 % / +0.015 %;
     - Prior x2 + gap3 (3 DP passes) is +0.045 % / +0.039 %.
   - On the sample, the curve keeps rising: +0.136 % after 4 rounds and +0.161 % after 6.
2. **Static min-plus parse: the parallel prefix isn't needed.**
   - With static prices, shortest paths **coalesce within ≤ 64 bytes**: p50 = 0, p99 = 14 bytes, 100 % of
     256 K chunk starts.
   - So exact parallelism needs only windows with a 64-byte left overlap. A min-plus scan would cost about 700×
     the work for nothing.
   - But the rep-free model loses **−0.19 … −0.57 %**. A sequential rep fix-up wins back about 0.1 %. It is only
     useful as a start parse.
3. **Overlapping windows with stitching (idea 1) work, but AWR dominates them.**
   - Best results: W = 1024 / O = 64 gives +0.059 % (sample); W = 512 / O = 64 gives +0.045 % (full 64 KiB) and
     +0.023 % (full 16 KiB).
   - This costs 1.12–1.5× opt16's work.

## 1. Anchored-window refinement (ideas 1 + 3 merged): finalist

### Mechanism (one pass)

1. **Boundaries.** With phase φ ∈ {W/2, 0}, alternating:
   - `b_k` = the first node of the previous parse at or after `k·W + φ`. A node is a position not strictly inside
     a match.
   - The windows are `[b_k, b_{k+1})`.
   - The previous path is therefore always feasible inside every window, and no match the previous pass chose
     is cut.
2. **Window DP.** This is today's `segment_ring`, unchanged, with:
   - `ip0 = b_k`;
   - `anchor0` = start of the previous parse's literal run that contains `b_k`;
   - `reps0` = the previous parse's decoder reps at `b_k`;
   - `iend = b_{k+1}`, with `ilimit = iend − 3` (gap3), or `− 8` at the block end.
3. **Keep.**
   - Integer-price the new and the old path over the window with the same price tables and start state: literal
     prices, `ll_price(litlen)` and `match_price(offBase under the running reps)`.
   - Keep the new path iff it is ≤ the old one. This makes each pass a descent step in the model.
4. **Assembly.** Concatenate the windows, then `encode_raw` with the true reps. That is today's fix-up. The
   histogram gives the next pass's prices.

### Ratio

All AWR rows below use W = 512 and the lazy2 start parse in 4 KiB segments.

| schedule | DP passes | sample | full 64 KiB | vs L16 | full 16 KiB | vs L16 |
|---|---:|---:|---:|---:|---:|---:|
| opt16 today | 4 | 1.37211 | 1.37144 | +0.032 % | 1.32794 | +0.015 % |
| Prior x2 + gap3 (a05) | 3 | 1.37215 | 1.37162 | +0.045 % | 1.32826 | +0.039 % |
| lazy2 alone | 0 | 1.36122 (−0.77 %) | | | | |
| **lazy + 1 AWR** | 1 | 1.37108 | **1.37052** | **−0.035 %** | 1.32770 | −0.003 % |
| **lazy + 2 AWR** | 2 | 1.37266 | **1.37209** | **+0.079 %** | **1.32879** | **+0.079 %** |
| lazy + 2 AWR, W = 256 | 2 | 1.37260 | 1.37203 | +0.075 % | 1.32876 | +0.077 % |
| **lazy + 3 AWR** | 3 | 1.37329 | **1.37270** | **+0.124 %** | **1.32920** | **+0.110 %** |
| lazy + 4 / + 6 AWR | 4 / 6 | 1.37362 / 1.37395 | | | | |
| lazy + 6 AWR, W = 64 (1024 lanes per block) | 6 | 1.37389 | | | | |
| lazy + 6 AWR, W = 32 | 6 | 1.37315 | | | | |

**Ablations (sample, lazy + 2 AWR at W = 512, 1.37266 with every flag).** Every flag matters:
- with no keep: 1.37163;
- with no anchor-literal start: 1.37206;
- with no seeded reps (BlockInit seed, segment start): −0.10…−0.13 %;
- a lazy start in 256-byte segments instead of 4 KiB: 1.37227;
- level 2 instead of 0 on the middle pass: no gain.

**Other start parses (sample, the same 2 AWR passes).**
- A fresh `S512` DP: 1.37226. This costs a full DP pass, so the lazy start is the better choice.
- Greedy-longest: −0.095 %, which is bad.
- The static model with rep fix-up: about equal to lazy, but costlier.

**Why it beats the full DP.**
- Today, every pass re-parses from scratch with new prices. The series heuristics (`sufficient_len` commits,
  level-0 aborts) re-introduce the same local mistakes each time.
- With keep, the improvements accumulate.
- The `fb` column counts changed windows: about 2/3 of windows change in the first two passes (168 of 248
  per block), falling to about 30 % on average over 5 rounds at W = 256 (379 of 1280). Skipping windows that
  have converged is a possible later work saving; it was not measured.

### GPU cost against opt16 (model, from measured op counts)

**Per-pass DP work is unchanged.**
- An AWR pass costs 58.3 K trips, 42 K relaxations and 175 K rep probes per block. A 4 KiB opt16 pass costs
  57 K / 37 K / 170 K, so the AWR pass is about 1.02× the work.
- The extra steps are tiny and independent per window:
  - the boundary search: a binary search in the previous sequence list, or a covered-bit scan;
  - the start-state gather;
  - the keep pricing, O(seqs per window), about 6 K sequences per block in total.

**The start reps need a rep-history scan over the previous parse.**
- Today's fix-up already computes it sequentially.
- It is also a cheap parallel scan: each sequence's effect on (rep0, rep1, rep2) is a "constant or input index"
  map, and such maps compose in O(1).

**The lazy start costs 46.9 K trips with no ring, which is well under one DP pass.**
- lvl9s12seg's whole pipeline runs at 6.3 µs/block.
- So K3 lazy is ≲ 2 µs/block (estimate).

| | opt16 today | Prior x2 + gap3 | **lazy + 2 AWR (W = 512)** | lazy + 1 AWR |
|---|---|---|---|---|
| DP passes | 4 | 3 | 2 (+ lazy) | 1 (+ lazy) |
| lanes per 64 KiB block | 16 | 16 | 128 (256 at W = 256) | 128 |
| chain per DP pass (sample mean / max) | 7.7 K / 20.6 K | 7.9 K / 24 K | ~1.3 K / ~4 K | ~1.3 K / ~4 K |
| total cost per block | 416 K | 318 K | 280 K, of which lazy is 63 K but cheaper per unit | 181 K |
| ratio, full 64 KiB | +0.032 % | +0.045 % | **+0.079 %** | −0.035 % |
| K3, 5090, estimated | 33.3 µs (measured) | 27 µs (measured) | **≈ 17–19 µs** at today's per-pass throughput | ≈ 10 µs |
| K3, 8 GB cards (throughput-bound, ∝ work) | 1.0 | 0.81 | **≈ 0.55–0.6** | ≈ 0.3 |

**Throughput regime on the 5090.**
- 128 lanes × 2900 blocks = 371 K lanes, about 3.4× the ~109 K resident at 93 registers. So the 5090 also
  becomes throughput-bound instead of chain-bound, and the long-tail segment problem disappears:
  - a05's heaviest segment costs 7× a typical one;
  - here the per-pass window maximum is ~4 K.
- If occupancy rises from today's ~43 % at one wave, the 5090 figure could fall further, to about 10–12 µs.
  This is an estimate and needs a kernel.

**Memory.**
- Today's per-position trace stays.
- New per pass: the previous sequence list with per-sequence reps, about 6 K × 16 B ≈ 100 KB per block. That is
  a ~15–20 % larger K3 footprint per block.
- Window outputs need a prefix-sum compaction. Today's fix-up already does an equivalent.

**With the in-progress wins.** "lazy + 1 AWR" (−0.035 % / −0.003 %) plus split (+0.16 %) or h10 (+0.10 %) would
clear L16 at about 1.5 DP-pass-equivalents. Stacking has not been measured.

### Byte-exact oracle

Every step is integer and deterministic:
- the lazy start (priced lazy2, the prototype's `greedy_pass`, or the existing lazy kernel run over the opt
  candidates);
- boundaries (`node_at_or_after`);
- start state (`Prev::state`);
- the window DP (`segment_ring` with an explicit `Win`: the GPU kernel's per-segment prologue gets inputs instead
  of `k << 12`);
- keep (`path_price`, i32/i64 sums);
- assembly (existing).

The CPU oracle is the prototype's `run_plan(Anc)`. There are no speculation or merge paths, so the GPU and CPU
match by construction.

### Effort and risk

**Effort M.**
- Today's K3 kernel, with segment prologue inputs.
- A new boundary/start-state kernel.
- A keep step: the same kernel's epilogue, or a small kernel.
- A lazy start kernel.
- Assembly with variable-length windows.

**Risks.**
- Register pressure if keep is fused into the K3 kernel. Keep it separate.
- Variable window lengths: a window can be up to W plus one long match.
- Stacking with split, h10 and refresh is unmeasured.
- The prior and lazy start are tuned on this corpus.

## 2. Overlapping windows with stitching (idea 1 as posed)

**Mechanism.**
- Window `[kW − O, (k+1)W + O)` runs the DP from a fixed state:
  - fresh reps, or reps from the previous pass (`s`);
  - ilimit = iend − 3.
- At each boundary, splice at the common node (not inside a match in either path) nearest to b, within the
  overlap. Otherwise, splice at a node of the right window and truncate the left window's crossing match.
- About 20 % of boundaries need the truncation fallback, mostly long matches spanning the overlap. It costs
  little.

Sample, BlockInit + 3 cheap + final (4 passes):

| W / O | ratio | vs L16 | work vs opt16 | chain mean |
|---|---:|---:|---:|---:|
| 4096 segments + gap3 (reference) | 1.37241 | +0.048 % | 1.00 | 30.8 K |
| 1024 / 64 s | 1.37256 | +0.059 % | 1.12 | 9.5 K |
| 1024 / 32 s | 1.37246 | +0.052 % | 1.06 | 9.0 K |
| 512 / 128 | 1.37233 | +0.042 % | 1.49 | 6.6 K |
| 512 / 64 s | 1.37230 | +0.040 % | 1.24 | 5.6 K |
| 512 / 32 | 1.37190 | +0.011 % | 1.12 | 5.1 K |
| 512 / 16 s | 1.37162 | −0.009 % | 1.06 | 4.8 K |
| 256 / 64 s | 1.37182 | +0.005 % | 1.49 | 3.6 K |
| 256 / 128 s | 1.37205 | +0.022 % | 1.99 | 4.6 K |
| 128 / 64 s | 1.37109 | −0.048 % | 1.98 | 2.5 K |

For comparison, fixed segments with no overlap: 1024 gives −0.044 %, 512 −0.156 %, 256 −0.349 % and 128
−0.664 %.

**Full corpus, 512 / 64 s:** 1.37162 (+0.045 %) at 64 KiB and 1.32805 (+0.023 %) at 16 KiB.

**Sweet spot.**
- W ≥ 512 with O = 32–64 holds L16 **without** the in-progress wins. W = 256 needs O ≥ 64 (+0.005 %).
- But the overlap costs 1.1–1.5× the work, and on 8 GB cards K3 is throughput-bound. AWR achieves more with no
  overlap.
- Seeding reps from the previous pass is worth about +0.01–0.03 %.

**Oracle.** Deterministic (window DP plus integer splice rule). **Effort M. Not recommended over AWR.**

## 3. Static min-plus (tropical) parse (idea 2)

**Model.**
- Literal edge: `lit_price + δ`, with δ a per-block linearised LL slope.
- Match edges: A and B, using zstd's record lengths, at `match_price + ll_price(0)`.
- Optional rep edges with **reps frozen from the previous pass** (`r`).
- Band K = 32. Matches longer than 32 are committed greedily in a pre-pass, as `sufficient_len` does.
- This is an exact shortest path. Offsets are re-encoded with the true reps.
- Optional sequential fix-up (`f`): walk the path with the true reps and swap explicit offsets to a rep that
  covers the same length.

| variant (3 passes, Prior seed, sample) | ratio | vs L16 |
|---|---:|---:|
| no reps, δ = 0 / adaptive / 64 | 1.36379 / 1.36396 / 1.36408 | −0.58 / −0.57 / −0.56 % |
| frozen previous-pass reps | 1.36783 | −0.286 % |
| frozen reps + rep-swap fix-up | 1.36909 | −0.194 % (= L14) |
| same with K = 64 | 1.36912 | −0.192 % |
| as a start parse for 2–3 AWR passes | 1.37286 (4 passes) | +0.081 % (lazy start is as good and cheaper) |

**Ratio cost of the approximations.**
- Rep-free: −0.57 %.
- Frozen reps: −0.29 %.
- The cheap fix-up recovers only 0.09 %. It cannot add rep candidates the DP never saw.
- Exact rep handling is restored only by a real DP pass. One AWR pass does this, which makes the static model
  just another start parse.

**Coalescence.** This measures the merge distance from a fresh start at a chunk boundary to the true forward DP,
where the band differences become constant across 33 positions.
- p50 = 0, p75 = 3, p90 = 8, p99 = 14 bytes.
- 100 % of starts merge within 64 bytes (256 K starts).
- Edges per position: 1.73.

**Work and depth.**

| | work | depth |
|---|---|---|
| sequential | N·e ≈ 113 K relaxations per 64 KiB | N·e |
| min-plus prefix, K = 33, chunk C = 64 | chunk matrices K·N·e ≈ 3.8 M, plus Blelloch scan 2(N/C)·K³ ≈ 73.6 M min-plus ops, ≈ **700×** sequential | C·e + log₂(N/C)·K ≈ 110 + 330 ≈ 440 |
| windowed speculation, O = 64 left overlap | N·e·(W+64)/W, i.e. 1.06–1.25× | (W+64)·e |

- The windowed form is exact whenever the merge check passes: 100 % measured. If a merge fails, re-run from
  the left neighbour's band.
- Tropical rank collapses within 14 bytes at p99, so the min-plus scan solves a problem the data does not
  have.

**Verdict.** The parallelism is free, but the static ratio is −0.2…−0.6 %. Not recommended, except as a possible
AWR start parse.

## 4. Literature: what applies

- **Ferragina, Nitto, Venturini, "bit-optimal LZ77" (SODA 2009; TALG 2013); Farruggia, Ferragina, Frangioni,
  Venturini, "Bicriteria data compression" (SODA 2014).**
  - Parsing with static, codeword-length costs is a shortest path over a pruned DAG. One "maximal" edge per
    offset cost class suffices: the longest match within each class.
  - Our static model (§3) is exactly this setting. Its loss comes from zstd's path-dependent reps and LL codes,
    which their model lacks.
  - Applicable idea, a candidate-side lever for the a06/r3 angle: keep the longest match **per OF-code class**
    instead of only A and B. This is not measured here.
- **Matias, Rajpoot, Sahinalp, flexible parsing (1999–2001).**
  - One-step lookahead is optimal for phrase count with suffix-closed dictionaries.
  - Under bit costs, it is a heuristic similar to lazy2.
  - The applicable part: a path-independent rule `next[x]` can be computed per position in parallel and the
    path extracted by pointer jumping.
  - Our lazy start depends on reps, so it is segmented instead. It is cheap either way.
- **Shun & Zhao, "Practical Parallel Lempel-Ziv Factorization" (DCC 2013).**
  - Greedy LZ77 via longest-previous-factor arrays, then pointer jumping or list ranking for the parse: O(n)
    work, polylog depth.
  - It confirms the pointer-jumping pattern. Its suffix-array candidates are a dead end here (m6 C).
- **Ozsoy et al., CULZSS (2011/2014).**
  - One thread per chunk or position does the match search, with a greedy sequential selection.
  - Its ratio is below serial LZSS. That is the "chunks lose ratio" lesson our fixed small segments reproduce
    (§2 bottom row).
  - Nothing new beyond what K1/K2 already do.
- **nvCOMP (LZ4/Snappy/Deflate/zstd).**
  - Batched, independent chunks, with one warp or CTA per chunk and fast hash-based match finding. The public
    docs give ratio as the trade-off, not an optimal parser.
  - A recent arXiv note on GPU LZ77 discusses zstd's rep history as a parse-layer dependency and a per-warp
    "high-water mark" that costs up to 19 % of ratio.
  - Applicable lesson: nobody ships a GPU optimal parser. Our edge is the DP itself, and AWR keeps the exact rep
    DP while still parallelising.
- **zopfli, libdeflate's near-optimal parser, zstd btultra.** These are iterated re-parse with stats updates,
  which is what opt16 does. AWR differs by making each iteration a monotone local improvement, which is what
  lets more rounds keep paying off (+0.16 % at 6 rounds).

## 5. Dead ends checked

- Fixed small segments with no overlap: 1 KiB −0.044 %, 512 B −0.156 %, 256 B −0.349 %, 128 B −0.664 %
  (sample, 4 passes, gap3).
- Overlapping windows below W = 256 or O < 32: −0.01 to −0.05 %, at 1.1–2× the work.
- Min-plus prefix scan: about 700× the work. Static-model ratio −0.19 % at best.
- Greedy-longest start parse: −0.095 % after 3 AWR passes, against +0.06 % with lazy2.
- AWR without keep, or without the anchor/rep start state: −0.05 to −0.13 %.
- Level 2 on intermediate AWR passes: no gain.
- Tiny windows (W = 32) as "local moves": +0.10 % after 6 rounds. They work, but W = 256–512 is better per
  pass. The specific moves listed in the brief are all subsumed by a W = 32–64 window DP.

## 6. Top 3

1. **Build AWR as the opt16 replacement: lazy2 start + 2 AWR passes at W = 512 (or 256).**
   - Full corpus: +0.079 % over L16 at 64 KiB and 16 KiB. Today's opt16 is +0.032 % / +0.015 %.
   - It uses 2 DP passes instead of 4, with 8–16× the lanes and ~6× shorter chains per pass.
   - Estimated K3: about 0.55–0.6× opt16 on throughput-bound 8 GB cards, and 17–19 µs (possibly ~10 µs) instead
     of 33 µs on the 5090.
   - Exact oracle by construction. Effort M.
2. **Lean preset: lazy + 1 AWR pass plus split and/or h10.**
   - Alone it is −0.035 % (64 KiB) and −0.003 % (16 KiB).
   - With split's +0.16 % it should clear L16 at about 1.5 DP-pass-equivalents, about 0.3× opt16's K3 work.
     Measure the stack first.
3. **Use AWR rounds as the ratio knob instead of more full DP passes.** Each extra round adds +0.03–0.06 %, up to
   +0.16 % at 6 rounds. That surplus can pay for cheaper candidates (K1/K2 depth cuts) or `sufficient_len` 16.
   Also try per-OF-class maximal candidates (Ferragina et al.) in the candidate angle.
   - Do not build the min-plus scan or overlapping-window stitching.

Sources:
- [Shun & Zhao, Practical Parallel Lempel-Ziv Factorization (ACM)](https://dl.acm.org/citation.cfm?id=2495844)
- [nvCOMP algorithm selection](https://docs.nvidia.com/cuda/nvcomp/algorithm_selection.html)
- [What Actually Serializes GPU LZ77 Decode (arXiv 2608.10188)](https://arxiv.org/pdf/2608.10188)
- [GPULZ (arXiv 2304.07342)](https://arxiv.org/pdf/2304.07342)
