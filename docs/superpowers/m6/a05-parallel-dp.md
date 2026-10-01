# a05: parallel formulations of the optimal-parse DP

Agent a05-parallel-dp, 2026-09-30. Research only: no repository file was changed. Scratch is in
`/tmp/claude-1000/m6/a05-parallel-dp/`:
- `ws/` is a repo copy with `crates/gzc-core/src/opt/a05.rs`. It is a parameterised copy of `segment_ring`:
  custom start states, a recorded series-head trajectory with cost counters, rep modes, segment starts and the
  inner-segment `ilimit` gap. Its exact mode matches the oracle on 2016 / 2016 sample blocks.
- `ws/crates/gzc-core/examples/a05.rs` is the experiment driver: `base`, `rep`, `seed`, `seg`, `passes`,
  `full`, `full16`, `resync`, `resync2`, `resync3`, `series`, `stats`.
- `ws2/` is the gap3 prototype. It changes `Seg::new` in `opt.rs`, two lines of `k3_opt.wgsl`, and adds a
  `Prior x2` schedule to the tests.
- Raw outputs: `rep.txt`, `seed.txt`, `seg.txt`, `passes.txt`, `full.txt`, `full16.txt`, `resync*.txt`,
  `series.txt`, `timing1.txt`, `timing2.txt`.

**Samples.**
- "Sample" is the design's 1/50 sample: every 50th 64 KiB block, offset 0, 2016 blocks. On it, opt16 = 1.37211
  and libzstd L16 = 1.37175.
- "Full" is the full corpus at 64 KiB, 100 754 blocks, run as 4 interleaved quarters and summed by bytes. My
  opt16 is 1.37144, the same as the M5 result.
- "Full at 16 KiB" is the same 397 925 blocks at 16 KiB. My opt16 is 1.32794, also the same as M5.

## TL;DR

1. **None of the parallel DP formulations pays on the target.**
   - The exact form, speculative chunks with merge detection, does work. With the previous pass's state as the
     guess, 74 % of chunk boundaries merge at once, and the total work overhead is only +0.4–3 %.
   - It cuts the mean block's chain to 1/C. But the dispatch maximum only falls to 0.59–0.80, because a single
     series cannot be split (up to 35 % of the heaviest segment).
   - It also inherits the divergence wall. Its closest measured analog, 2 KiB segments, gave only 10–12 %.
   - On an 8 GB card K3 is throughput-bound, so extra work and extra lanes do not help there.
   - DAG shortest-path, prefix and wavefront forms cost 100–1000× the work, or are not exact.
   - Every rep approximation costs 0.1–0.4 % ratio, against +0.032 % of headroom.
2. **The by-product is the real result: segment boundaries waste ratio, and that buys a pass.**
   - The oracle stops match starts 8 bytes before every 4 KiB segment end, a rule that only matters at the block
     end. With **"gap3"**, inner segments allow match starts up to `iend − 3`.
   - Ratio gain: **+0.022 %** at 64 KiB (1.37174, +0.054 % over L16) and +0.016 % at 16 KiB.
   - It is a 2-line kernel change, verified byte-identical on the GPU.
3. **Recommendation: opt16 = Prior seed + 2 cheap passes + final (3 DP passes instead of 4).**
   - It already passes the L16 gate without gap3: +0.023 % at 64 KiB, +0.023 % at 16 KiB, where today's opt16
     has +0.015 %.
   - With gap3: +0.045 % / +0.039 %.
   - Measured K3 time: **−21.5 %** (34.4 → 27.0 µs/block on the 5090, interleaved, under contention).
   - Projected: opt16 about 1.43 → 1.7 GB/s on the 5090, and about 0.27 → 0.32 GB/s on a 4060.

## 1. Ideas evaluated

### 1.1 Shortest path on a DAG (frontier, delta-stepping, min-plus prefix): dead

**Mechanism.** Positions are nodes. Literal edges are x → x+1, and match edges are p → p+len with 3 ≤ len ≤ 32
(longer ones commit at once).

**Not the same problem as the oracle.** Out-of-order relaxation cannot be byte-identical, for four reasons:
- the optLevel-0 `+128` skip at `cur` reads `price[cur+1]` as it stands after relaxations from nodes before
  `cur` only;
- the optLevel-0 early abort is per record and depends on visit order;
- `sufficient_len` and the `cur + longest ≥ iend` path end a series early;
- most of all, a node's candidate set depends on the reps of the path that reached it.

A frontier or delta-stepping parse would need a new oracle. Its ratio would be about that of exact shortest
paths, which is not better in bits, since the prices are static estimates.

**Depth.** The optimal path has **2605 hops per 4 KiB segment** (377 sequences + 2228 literals, measured on the
sample), against 4096 sequential positions. So Bellman-Ford or delta-stepping can shorten the dependent chain by
at most about 1.6×, and each round touches every live edge: work goes up by 100×+.

**Min-plus prefix over the literal chain.**
- The state window is 33 prices (positions x−32..x), so a transfer is a 33×33 tropical matrix and a combine
  costs 33³ ≈ 36 K operations.
- A sequential step is about 50 operations, so the prefix needs about 700× the work for log depth.
- That is before two more problems:
  - `ll_price` depends on the literal-run length (the literal edge weight is not constant), which adds litlen to
    the state;
  - rep candidates depend on the path, so the matrices are not even static.

**Verdict.** No speedup on any GPU. **Effort L, risk high. Not recommended.**

### 1.2 Wavefront or software-pipelined DP within a lane

- Node x becomes final after the literal step from x−1 and all relaxations from positions up to x−3. So
  relaxation work from x has 2 positions of slack (MIN_MATCH = 3) against the literal chain.
- That is ILP inside one lane, not lane parallelism.
- The perf study already measured this direction (v7/v8: carry `ld(cur−1)`, early node load). The chain fell
  8–10 %, but registers rose to 112–117, so a batch of 2900 no longer fits one wave and the net was a loss.
- **Verdict.** Exact (same order of effects), but blocked by the register wall. Effort M. Not recommended.

### 1.3 Chunked speculative DP (exact): works, but the target gains little

**Mechanism.**
- Split each 4 KiB segment into C sub-chunks. Lane i starts at a guessed series-head state (ip, anchor, reps)
  and records its head states.
- The segment's DP between series is fully determined by (ip, anchor, rep). So once lane i−1's trajectory
  reaches a state lane i also visited, the two coincide from then on (`SegState.seq_reps == rep` is asserted).
- The fix-up splices sequences at the merge points, and the output is **byte-identical to today's oracle**. The
  fix-up already does literal carry and offset re-encoding.
- Each lane needs a private trace window of chunk + overlap.
- Measured with the CPU oracle on the sample, all 4 passes of opt16. Cost = DP trips + relaxation iterations.

**Guess quality.**

| guess | merge at once (L2 / L0) | never merges within the segment (L2 / L0, C=2..16) | spec cost before merge, p90 / p99 of a segment |
|---|---|---|---|
| fresh (reps 0, anchor = s) | 0 % | 0.8–1.4 % / 1.0–1.8 % | 4 % / 100 % |
| **previous pass's head state at the first head ≥ s** | **74 % / 41 %** | **0.12–0.27 % / 0.33–0.64 %** | **0.8 % / 7 %** |

**Lane model, previous-pass guess, window = one chunk, second exact round for failed chunks.** From
`resync3.txt` (L2 = the final pass):

| C | blocks needing round 2 | mean block chain (r1+r2)/sequential | dispatch: (max r1 + max r2) / max sequential | total work |
|---:|---:|---:|---:|---:|
| 2 | 0 % | 0.55 | 0.80 | 1.004 |
| 4 | 1.0 % | 0.32 | 0.66 | 1.012 |
| 8 | 4.3 % | 0.20 | 0.59 | 1.026 |
| 16 | 12.5 % | 0.14 | 0.61 | 1.051 |

Splitting at the previous pass's cost quantiles, per segment or with lanes allocated per block by cost, did not
improve the dispatch figure: 0.60–0.81.

**Why the dispatch number stalls.**
- Merge points exist only at series heads.
- In the heaviest segment of the sample (49 K cost units against a typical ~6.8 K), single series cost up to
  35 % of the segment and run up to 4096 bytes long (`series.txt`).
- Those series are indivisible in this formulation.

**What it would buy, measured analog plus estimate.**
- C = 2 has the same shape as the perf study's 2 KiB segments: 32 lanes per block in one full warp, with the
  chain halved. That measured **only 10–12 %** on the 5090 at one wave, because divergence grows with the
  active-lane count.
- C ≥ 4 needs 2+ warps per block, which halves the blocks per wave.
- On a 4060 (about 7 waves at a 6 GB batch) K3 is throughput-bound:
  - extra lanes give no gain;
  - the +0.4–5 % extra work is a loss;
  - the warp-filling benefit is available for free from the wg32 two-blocks-per-warp mapping (+16 % multi-wave,
    perf study).
- **Estimated:** ≤ 10 % on the 5090 at one wave, about 0 or slightly negative on 8 GB cards.
- **Effort M–L** (per-lane head logs, trace windows, merge and splice in the fix-up, a round-2 path). **Risk
  medium.** Not recommended.

### 1.4 Two-level DP: coarse boundaries first

**Variant measured (a new oracle).** Pass n ≥ 1 places inner segment boundaries at the previous pass's first
match end at or after k·2^log2. It starts each segment with that pass's decoder reps, instead of reps [0,0,0]
at a fixed position. This also lets an inner segment be shorter without a reset.

| segments | plain | seeded | gap3 | gap3 + seeded |
|---|---:|---:|---:|---:|
| 4 KiB | 1.37211 | 1.37208 | 1.37241 | 1.37253 |
| 2 KiB | 1.37135 | 1.37124 | 1.37196 | 1.37219 |
| 1 KiB | 1.36987 | 1.36962 | 1.37114 | – |
| 512 B | – | 1.36651 | – | – |
| (8 K / 16 K / whole block) | 1.37250 / 1.37271 / 1.37288 | | | |

All figures are on the sample.

- Seeding alone gives nothing, which matches the design's "closure points" +0.003 %.
- The boundary loss is mostly the 8-byte `ilimit` dead zone, not the rep reset. That observation is §1.6.
- 2 KiB with gap3 and seeding reaches today's 4 KiB ratio. So a 2 KiB-segment oracle would be ratio-neutral, but
  its measured speedup is only 10–12 %, and seeding needs per-segment start positions from the fix-up.
- **Verdict.** Effort M, small gain. Not recommended now.

### 1.5 Relaxing the rep dependency (new oracle): dead on ratio

Every pass of opt16 is run with the variant. Real offsets are kept per node, and the fix-up re-encodes them.

| variant | sample ratio | Δ vs oracle |
|---|---:|---:|
| exact per-node reps (oracle) | 1.37211 | – |
| every node probes the **series-start** reps | 1.37081 | −0.095 % |
| every node probes **static reps from the previous pass's parse** (decoder reps at p) | 1.36943 | −0.195 % |
| same, pass 0 exact | 1.36944 | −0.195 % |
| rep candidates only at the series start | 1.36660 | −0.40 % |

- Static reps would make the edge set path-independent. Rep probes could then be precomputed in parallel, as a
  K2-like kernel, and §1.1 would become well defined.
- But all these variants cost 3–15× the whole +0.032 % headroom, even after adding gap3's +0.022 %.
- An exact alternative is to cache rep-probe lengths for the previous pass's reps. It only saves the extension
  loop (516 `match_len` word iterations per lane against 4176 trips) and adds a load per probe, so it is not
  worth it.

### 1.6 gap3: inner segment ends allow match starts up to `iend − 3` (new oracle, recommended as headroom)

**Mechanism.**
- zstd's `ilimit = iend − 8` exists for the real buffer end.
- Inner 4 KiB segment ends are only a lane partition: matches are clamped to `iend`, and the bytes after
  `iend` are in the block. So `ilimit = iend − 3` is safe for every segment except the last.
- The GPU change is 2 lines:
  - `ilimit = select(iend − 8u, iend − 3u, iend < BLOCK_SIZE)`;
  - a `lim > 3u` guard in `rep_probe`'s 4-byte extension. Without it, `lim − 4` would underflow at `lim = 3`.
- The oracle change is one line in `opt::Seg::new`.

**Ratio.**

| block size | opt16 | opt16 gap3 | Δ |
|---|---:|---:|---:|
| sample | 1.37211 | 1.37241 | +0.022 % |
| full, 64 KiB | 1.37144 | 1.37174 | +0.022 % |
| full, 16 KiB | 1.32794 | 1.32815 | +0.016 % |

**Exactness, measured on the GPU** (`ws2`, `k3opt` tests):
- `k3opt_passes_corpus` on 1500 corpus blocks: every schedule (BlockInit and Prior × 0/1/2/3 cheap passes)
  matches the gap3 oracle, per-pass histograms included, and so do both later-pass Buffer-price configs.
- `k3opt_passes_opt_cases` and `k3opt_passes_synthetic` pass, including the private ring, wg32 and Buffer
  prices.
- All 130 `gzc-core` tests still pass with gap3. No scripted case covers an inner segment's last 8 bytes, so add
  one when this is implemented.

**Speed.** +1–2 % per pass, measured, but within the noise of the contention.

**Effort S, risk low.**

### 1.7 Pushing work to lanes (positions from several segments interleaved per lane)

- A lane that steps 2–4 segments round-robin keeps 2–4 DP states live. That is about +25–35 registers per
  extra state, on top of 96–102 now.
- The study measured that 112–117 registers drop occupancy from 20 to 16 warps/SM, and that 2900 blocks then
  stop fitting one wave.
- Each step would also execute the union of both segments' branch bodies, and the union of branch bodies is
  the main divergence cost.
- The latency-hiding goal is better served by full warps: wg32 with 2 blocks per warp, already measured at
  +16 % in multi-wave mode.
- No new subgroup-cooperative twist was found; the 1.7× slowdown stands.
- **Not recommended** (estimated).

## 2. What the angle produced: a cheaper opt16 schedule

Pass-count variants (gap3 is §1.6; L16 is libzstd at the same block size):

| schedule (DP passes) | sample | full 64 KiB | vs L16 | full 16 KiB | vs L16 |
|---|---:|---:|---:|---:|---:|
| opt16 today: BlockInit + 3 cheap + final (4) | 1.37211 | 1.37144 | +0.032 % | 1.32794 | +0.015 % |
| opt16 + gap3 (4) | 1.37241 | 1.37174 | +0.054 % | 1.32815 | +0.031 % |
| BlockInit + 2 cheap + final (3) | 1.37111 | – | – | – | – |
| BlockInit + 2 cheap + final, gap3 (3) | 1.37140 | – | – | – | – |
| **Prior + 2 cheap + final (3)** | 1.37185 | **1.37132** | **+0.023 %** | **1.32804** | **+0.023 %** |
| **Prior + 2 cheap + final, gap3 (3)** | 1.37215 | **1.37162** | **+0.045 %** | **1.32826** | **+0.039 %** |
| opt14 + gap3: Prior + 1 cheap + final (2) | 1.37149 | 1.37094 | −0.004 % | – | – |

- At 32 KiB, opt14 (Prior + 1 cheap) is already +0.059 % over L16, so Prior + 2 should pass there too. That is
  inferred, not run.
- The M5 design's row Q9 (Prior + 2 cheap, in-sample prior) gave +0.01 % on the sample, which looked too thin.
  The full corpus, with the shipped block-disjoint prior, shows a larger margin than today's opt16 at 16 KiB.
- `Prior x2` is already a valid `OptParams` (`passes: 2, seed: Prior`). The GPU path supports it with no kernel
  change, and it is byte-exact (1500 blocks above, and the `ws` baseline schedules).

**GPU timing.**
- Setup: `k3opt_passes_timing`, 2900 corpus blocks, one wave, wg16, 5 dispatches median.
- Runs were interleaved: baseline `ws`, gap3 `ws2`, 4 pairs. Contention: 2–4 other agents' k3opt/probe
  processes were resident, and utilisation between runs was 2–82 %.
- Each clean pair was taken where utilisation was ≤ 5 % both before and after. Figures are K3 µs/block (all DP
  passes plus fix-ups), and all are **under contention**.

| | baseline kernel | gap3 kernel |
|---|---:|---:|
| opt16 (4 passes) | 34.36 / 34.42 / 34.35 | 34.68 / 35.19 / 35.41 |
| Prior x2 (3 passes) | 27.03 / 26.97 / 26.99 | 27.64 / 27.49 |
| ratio Prior x2 / opt16 | **0.785** | 0.786 |

**What this means end to end (estimates).**
- 5090: opt16 goes from 44.7 to about 37.3 µs/block, so from 1426 MB/s to about 1.7 GB/s.
- 4060 class: K3 is about 2/3 of the per-block time there, so the total drops by about 15 %. opt16 goes from
  about 0.27 to about 0.32 GB/s using the perf study's v11 projection. This is still far below 1.25 GB/s.
- The headroom left by Prior x2 + gap3 (+0.045 % at 64 KiB, +0.039 % at 16 KiB) can absorb another agent's
  ratio-costing speedup of up to about 0.02 %.

## 3. Dead ends checked

- Frontier, delta-stepping and Bellman-Ford DP: not exact; depth ≥ 2605 hops per segment; 100×+ the work.
- Min-plus prefix: 33³ per combine, about 700× the work, and litlen and reps make the state larger still.
- Speculative chunks with a fresh guess (reps 0): 1–1.8 % of boundaries never merge, so p99 spec cost is the
  whole segment.
- Cost-balanced chunk boundaries: the dispatch maximum barely moves (indivisible series).
- Rep approximations: −0.095 % to −0.40 %.
- Seeded segment starts: +0.000 % at 4 KiB, and −0.01 to −0.02 % at 1–2 KiB without gap3.
- Interleaved multi-segment lanes: register wall (estimated from the perf study's measurements).
- 2 KiB / 1 KiB segments even with gap3: −0.011 % / −0.07 % on the sample, for a measured ≤ 10–12 % speedup.

## 4. Top-3 recommendation

1. **Switch opt16 to `Prior` + 2 cheap passes + final.**
   - K3 −21.5 % measured. Ratio full 64 KiB +0.023 %, 16 KiB +0.023 % over L16.
   - A preset change on an existing, GPU-exact code path (new oracle = the existing oracle with other
     `OptParams`).
   - Effort S, risk low. Rerun the full-corpus gate at 32 KiB.
2. **Adopt gap3 (`ilimit = iend − 3` for inner segments) in both the oracle and K3opt.**
   - +0.022 % ratio at about 1 % K3 cost.
   - GPU byte-identical on 1500 corpus blocks × 8 schedules.
   - Effort S. Add an opt case for a match starting in an inner segment's last 8 bytes.
   - Mainly a headroom bank for other speed trades. With it, opt14 + gap3 misses L16 by only 0.004 %, so one more
     small ratio gain would turn the 2-pass opt14 schedule into an L16-class preset (K3 about 18.8 instead of
     about 27 µs/block).
3. **Do not build a parallel or speculative DP.**
   - The exact speculative form is sound: 74 % of boundaries merge at once with previous-pass guesses, and work
     overhead is ≤ 3 %.
   - But dispatch time is set by indivisible series in heavy blocks, the C = 2 analog measured only 10–12 %, and
     8 GB cards are throughput-bound.
   - Put K3 effort into work per position and divergence (predication, wg32 two blocks per warp, fused passes),
     and into fewer passes.
