# r2-decoupled-dp: moving optimal-parse work out of the sequential DP

Agent r2-decoupled-dp, 2026-10-01. Research only: no repository file was changed.
- Sources, kernels and raw outputs: `.superpowers/m7-research/artifacts/r2-decoupled-dp/`.
- Scratch: `~/.cache/gzc-m7/r2-decoupled-dp/`.
- Following the coordinator's update, finalists were measured at 64 KiB only. The 16 KiB runs were dropped.

## TL;DR

**Where the per-position chain goes.** It is dominated by the rep probe, not by edge pricing.
- The explicit-edge prices (OF, ML, LL) are already static per pass, as shared-memory table lookups.
  Precomputing them gains nothing.
- The expensive part is path-dependent: a data load whose address depends on the node's rep history (itself an
  L2 load of the node payload), then a divergent extension loop.

**Validated precomputed reps cost −0.0115 % on the full corpus** (64 KiB, opt16, with top-4 pruning).
- A parallel pre-pass stores, for each position, the offsets that pass both of these tests:
  - they match at least 3 bytes at that position;
  - they come from a state-independent "likely rep" pool: A and B at p, the A/B offsets of the 32 previous
    positions, and the previous pass's 8 most recently used offsets.
- In the DP, a node rep is used only when it is in the list, and the list supplies its length. This removes
  every dependent data load and extension loop from the trip.
- The list is short: 1.04 entries on average, and 66 % of lists are empty.

**Relaxation pruning is free: −0.0025 % on the full corpus.**
- The DP relaxes only the top 4 lengths of each explicit record, which is exactly one 4-wide grid step.
- On the final path, 98.1 % of matches use their record's full length. Another 1.7 % are cut short by fewer than
  4 bytes.

**Cheaper LL models are a dead end.**
- zstd's DP already prices LL as a per-literal increment, so the "increment plus correction" form already
  exists.
- Every cheaper model costs −0.02 % to −1.0 %. They would save one shared-memory lookup per trip.

**GPU effect, measured with cost-mimic kernels** (RTX 5090, 2900 blocks, interleaved):

| comparison | L2 | L0 |
|---|---:|---:|
| decoupled DP pass vs today's kernel | 0.67× | 0.78× |
| decoupled DP pass vs a01's exact p5 kernel | 0.81× | 0.81× |

**The catch is the pre-pass.**
- Building the lists is estimated at about 0.4 µs/block per pass, plus about 0.7 µs/block once.
- Net K3 gain on top of p5: about −11 % for opt16 and about −9 % for a 2-pass preset, with a range of 0 to
  −12 % depending on what the pre-pass really costs.
- **Decoupling moves the work more than it removes it.** Pruning is the free part of the gain.

## 1. Method

**Oracle copy.**
- `r2/src/dp.rs` is a parametrised copy of `opt::Dp::segment_ring`, the GPU spec.
- With the default config it reproduces `opt::parse` byte for byte: 0 differing blocks out of 2016 on the
  sample (opt16), and 0 out of 1990 at 16 KiB.
- Every variant runs in every pass, with the real schedule:
  - opt16: BlockInit seed, 3 cheap passes at optLevel 0, then a final pass at L2;
  - opt14: Prior seed, 1 cheap pass, then the final pass.
- Prices are rebuilt from each pass's own histogram. The output goes through `encode_raw` and `write_frame`.
- **Ratio** is real bytes divided by frame bytes, measured on:
  - the 1/50 sample (2016 blocks);
  - the full corpus for finalists (100,754 blocks).

**Warp model** (`warp.rs`).
- K3's loop is flattened (one trip per position per lane), so a warp's trip t executes the union of its 16
  lanes' trip-t bodies. Loops run max-over-lanes iterations.
- Validation against a01's measured branch map (base kernel, L2, per block-pass):

  | quantity | this model | a01 |
  |---|---:|---:|
  | trips | 4381 | 4385 |
  | relaxation records | 6621 | 6621 |
  | 4-wide relaxation steps | 8402 | 8401 |
  | finishes | 3035 | 3058 |

  The model reproduces it.

**GPU cost mimics** (`gpu/*.wgsl`, run through a01's probe harness on a `git archive` copy).
- The kernels have the per-trip shape of the decoupled DP but not its exact parse:
  - rep lengths come from A/B at p plus a variable-length list loop of 1-word state-independent loads, with the
    measured list-length distribution;
  - relaxation is limited to one step.
- Times are interleaved, the median of 5 rounds, under contention (load 25–55, other GPU processes present).
  Same-kernel noise is about ±3 %.
- At batch 2900 the kernel is latency-bound per lane. At 1450 blocks base is 15.3 µs/block against 9.1 at 2900.
  a03 showed that one-wave time ≈ throughput-regime time, so these ratios stand in for the critical-path
  reduction.

## 2. Q1: edge-cost precomputation. What is state-independent, and what is left on the chain

**Per trip today** (the kernel's real chain, the final pass):

| step | depends on DP state? | latency class |
|---|---|---|
| loads x, xprev, candidate words w0/w1 | no (address = p) | global, issued at trip start |
| node payload ld(cur-1), ld(cur) | slot known; data written by earlier trips | LDS price + **L2 payload** |
| literal extension: lit price, ll(litlen) − ll(litlen−1), compare, store | yes (litlen) | 2–3 LDS |
| **rep sources: 3 × ld32(p − rep_i)** | **yes (address from node reps)** | **global (L1/L2), dependent on payload** |
| rep gate + extension loop (0.74 union iterations/trip) | yes | dependent global loads |
| candidate decode, cap-64 extension (rare) | no | – |
| per record: of_price, new_rep; per step: 4 × p_ml + 4 × ring_p, compare, stores | price tables static; ring is state | LDS; 1.51 records and 1.92 steps per trip (union) |

**What precomputation can and cannot move.**
- The explicit edges' `of(ob) + ml(len) + fee` are fully state-independent.
- But they are 1–2 shared-memory lookups whose addresses are known before the trip. Precomputing them into a
  global per-position edge list would turn an LDS lookup into a global load. a03 measured the payload-in-L2
  versus shared trade-off: that is not a win.
- So the useful "edge precomputation" is not pricing. It is:
  - the rep edges' existence and length;
  - the bound on how many lengths are relaxed.

**Operation counts per block-pass at L2** (warp-union model; lane means in brackets):

| | today (exact) | decoupled (finalist B) |
|---|---:|---:|
| trips | 4381 | 4381 |
| dependent global round trips per trip | ≈ 2 + 0.74 extension | **1** (payload only) |
| rep extension iterations (union) | 3262 (448) | 0 |
| list entries checked (union), state-independent loads | – | 19 716 (4240) |
| relaxation records (union) | 6621 | 6569 |
| relaxation 4-wide steps (union) | 8402 (1866) | 7526 (1754); about 6600 with reps pruned too |
| moved to the parallel pre-pass | – | about 10 distinct near offsets + 9 history offsets per position: gate test and length |

## 3. Q2: rep handling under precomputation

Rep mode, opt16, measured on the sample in every pass:

| variant | Δ ratio | notes |
|---|---:|---|
| no rep probes in the DP (`encode_raw` still finds repcodes) | −0.528 % | |
| rep0 only | −0.200 % | |
| validated, set = A/B at p | −0.409 % | |
| validated, set = previous pass's 3 reps (+ rep0−1) | −0.083 % | |
| validated, set = A/B + previous pass's 3 reps | −0.064 % | 14 % of probes miss |
| validated, the same with pass-0 reps from a greedy parse in every pass | −0.210 % | |
| validated, A/B + previous pass's last 8 offsets (hist8) | −0.041 % | |
| validated, A/B + hist6 + A/B offsets of the 32 previous positions | −0.010 % | |
| validated, the same with near64 | −0.004 % | |
| validated, A/B + K2 chain walk (every offset matching ≥ 3 within depth) | −0.173 % | rep offsets are mostly beyond chain depth |
| validated with miss → exact probe | ±0 | the misses are 14 % of probes, so on SIMT nearly every warp-trip has one: no gain |

**Gating changes nothing; capping hurts.**
- Gating the set (storing only offsets that match ≥ 3 bytes at p) leaves the ratio unchanged.
- Capping the list hurts, because the tail is where reps matter. With hist8 + near32/64:
  - cap 4: −0.045 %;
  - cap 8: −0.021…−0.026 %;
  - cap 12: −0.010 %;
  - cap 16: −0.005 %.
- Putting near offsets before the history (`nf`) is much worse: −0.17 % at cap 4.

**Combined with top-4 pruning (finalists):**

| variant | opt16 sample | opt14 sample | opt16 full 64 KiB | opt14 full 64 KiB |
|---|---:|---:|---:|---:|
| exact (today's oracle) | 1.37211 | 1.37118 | **1.37144** (+0.032 % vs L16) | **1.37064** (−0.026 % vs L16) |
| A: prune top4 | −0.0024 % | −0.0023 % | **1.37141 (−0.0025 %)** | 1.37060 (−0.0024 %) |
| **B: hist8 + near32, gated, uncapped + top4** | −0.0114 % | −0.0108 % | **1.37129 (−0.0115 %; +0.021 % vs L16)** | **1.37049 (−0.0108 %)** |
| C: B with the list capped at 8 | −0.0263 % | −0.0213 % | 1.37109 (−0.0259 %) | 1.37035 (−0.0207 %) |
| B with near64 | −0.0062 % | −0.0060 % | – | – |
| B without hist (static list, no per-pass rebuild) | −0.097 % | −0.079 % | – | – |
| B with refs (3 reps) instead of hist8 | −0.023 % | −0.030 % | – | – |
| gap3 alone / B + gap3 | +0.0216 % / +0.0106 % | +0.0228 % / +0.0117 % | 1.37174 (+0.0215 %) / **1.37158 (+0.0099 %)** | – |

- The losses are additive with gap3: B + gap3 = gap3 − 0.0115 %.
- Against libzstd at 64 KiB (L16 1.37100, L14 1.36827):
  - opt16 + B: +0.021 % over L16;
  - opt16 + B + gap3: +0.042 % over L16.
- The in-progress wins (split +0.16 %, h10 +0.10 %, refresh +0.08 %) are independent of the rep handling, so B
  leaves about 99 % of that margin intact.

## 4. Q3: literal-run pricing

| LL model (opt16 sample) | Δ ratio |
|---|---:|
| exact (zstd: per-literal increment ll(l) − ll(l−1), the match base adds ll(0)) | 0 |
| saturate the price at ll(min(l, 16)) | −0.024 % |
| saturate at 8 | −0.173 % |
| saturate at 4 | −0.700 % |
| saturate at 1 | −0.532 % |
| deferred: nodes exclude the open run's LL, charged as ll(litlen) at the next match start | −0.995 % |

- zstd's DP already is "a per-literal increment plus a correction at match time": the ll(0) in the match base.
  It costs one shared-memory lookup per trip, indexed by `min(litlen, 63)`.
- Any cheaper model changes which partial paths win at the literal-versus-match compare, and the deferred model
  biases every compare toward literals.
- **Dead end.** Keep the exact increment, optionally as a precomputed increment table: one lookup instead of
  today's two.

## 5. Q4: pruning the relaxed lengths

**Final-path statistics** (opt16 final pass, sample, 6026 matches per block):
- 98.1 % use their record's full length;
- 1.3 % are cut by 1;
- 1.7 % are cut by fewer than 4;
- 1.2 % are cut to a position where a candidate whose reach goes beyond the full match starts ("al2").

**On average the relaxation already does little work.** It visits 41,660 lengths over 20,733 records per
block-pass, about 2 per record. The cost is the union tail (8402 steps against 6621 records).

| prune (explicit records unless noted; opt16 sample) | Δ ratio |
|---|---:|
| full length only | −0.028 % |
| full length + "useful truncation" (al2) | −0.0010 % |
| full length + any candidate start (al1) | −0.0010 % |
| top 2 | −0.0089 % |
| top 2 + al2 | −0.0004 % |
| **top 4 (one grid step)** | **−0.0024 % (full corpus −0.0025 %)** |
| top 4 + reps too | −0.0058 % |
| top 4 + al2 + reps | −0.0002 % |
| top 8 | −0.0013 % |

**Recommendation: top 4.**
- The al2 variants are slightly better, but they make the step count variable again.
- top4 fixes the relaxation at exactly one 4-wide step per explicit record: a uniform trip shape.
- **Measured alone on the GPU, as an exact kernel against its own oracle:** L2 0.874–0.884×; L0 0.94× in one
  session and 1.04× in another. L0 is within noise: the optLevel-0 early abort already truncates there.
- This is the cheapest lever in this report:
  - a one-line oracle change: `keep()`;
  - a one-line kernel change: `stop = true`;
  - −0.0025 % ratio.

## 6. Q5: the combination, and the critical path per position

**Dependent latency stages per trip, in the final pass:**

| | today | p5 (a01, exact) | decoupled (B + top4) |
|---|---|---|---|
| global round trips on the chain | payload → rep source → extension (0.74/trip) | payload → rep source (memo hits skip extension) | **payload only** |
| LDS stages | lit/ll (2–3), 1.5 records × (of + ml + ring), steps 1.92 | the same, with the grid reaching about 1.35 steps | lit/ll, 1.5 records × 1 step |
| divergent loops | rep extension, records × steps | bounded lockstep extension, grid | list loop (ALU + independent loads only) |

**Measured, cost-mimic kernels** (µs/block per pass; 2900 blocks; interleaved; `time3.txt`, `time1.txt`,
`time2.txt`):

| kernel | L2 | L0 |
|---|---:|---:|
| today's kernel (base) | 9.09 | 7.69 |
| top4 only | 7.95–8.03 (0.874–0.884×) | 7.19–7.98 (0.94–1.04×) |
| A/B-only rep validation + top4, no list (lower bound) | 4.95 (0.54×) | 4.96 (0.65×) |
| **decoupled mimic: list (measured distribution, up to 16 entries) + top4** | **6.08–6.10 (0.67×)** | **5.99–6.03 (0.78×)** |
| the same, list capped at 8 | 6.02 (0.66×) | 5.99 (0.78×) |
| a01 p5 (exact; 0/100 differences from the oracle) | 7.75 (0.853×) | 6.63 (0.863×) |
| **p5 + decoupled rep list** (p5's grid relaxation, no top4) | **6.25 (0.69×; 0.81× vs p5)** | **5.34 (0.69×; 0.81× vs p5)** |

- A fixed 4- or 8-neighbour unrolled load loop measured 0.71 and 0.94–1.09×. Unconditional loads are
  expensive, so a variable list indexed from the spare high 16 bits of candidate word 1 is the right layout.

**Pre-pass cost (modelled).** The pre-pass has two parts.
- **Near part, static and built once:**
  - per position, about 10 distinct A/B offsets from the previous 32 positions (measured: 11.3 including the 3
    reference reps);
  - each needs a 4-byte gate compare, and the survivors a length.
  - These are independent loads, unlike K2opt's pointer-chasing 36-entry chain walk (2.6 µs/block).
  - Estimate: 0.5–1.0 µs/block, or 0.7 with an "episode" scheme that compares each offset once over its 32-byte
    window.
- **History part, per pass:**
  - the previous pass's last 8 offsets per position, rebuilt from its sequences;
  - fixup already walks them with true reps, so it can emit the move-to-front history at each sequence;
  - 9 gate tests per position.
  - Estimate: 0.3–0.5 µs/block per pass.
- Memory: the list averages 1.04 entries × 4 B per position, about 270 KB per block, plus a 16-bit index in the
  spare bits of candidate word 1.

**Net, on top of p5** (5090 µs/block; p5 per pass L0 6.63, L2 7.75):

| schedule | p5 K3 | DP savings | pre-pass | net K3 |
|---|---:|---:|---:|---:|
| opt16 (3 × L0 + L2) | 27.6 | −5.4 | +2.3 (range 1.7–3.5) | **−3.1 (−11 %; range −7…−13 %)** |
| 2-pass (L0 + L2) | 14.4 | −2.8 | +1.5 (range 1.1–2.5) | **−1.3 (−9 %; range −2…−12 %)** |

- Adding top4 to p5's grid relaxation would add perhaps −2…−4 % more. That is a guess based on top4 alone
  against base.
- **8 GB GPUs:** a03 found the per-SM regime is the same latency-bound one, so the ratios carry over:
  - on an RTX 4060, about −10 % of K3;
  - on Ampere, at 16 resident blocks per SM, the shorter chain matters more and fewer registers matter less.
    This is a projection.

## 7. How a byte-exact CPU/GPU oracle would work

**The new oracle is deterministic and integer-only:**
- `opt::Dp` plus a per-position `RepList` (offset, length ≤ 64 or "extend");
- A/B candidates relaxed only at lengths L..L−3;
- every pass's list built from:
  - `find_cands` (A/B at p and at the 32 previous positions);
  - `reps_hist8(prev_output)`, which is the greedy-from-candidates parse for pass 0.

**The list order must be canonical** (A, B, hist in move-to-front order, then near, nearest first; deduplicated
and gated). Membership is a set test, so the order only matters for caps.

**What the GPU matches:**
- the list contents, which a CPU unit test can check as a buffer;
- each DP pass, as today, through per-pass histograms.

**Two edge rules to pin down:**
- a rep whose listed length is ≥ 64 is extended exactly, as today's cap-64 path does;
- the "rep0 − 1" offset is included as hist[0] − 1.

The decoupled DP is simpler than today's (no extension loop in the trip), so the bit-exact risk is in the
pre-pass, which is embarrassingly parallel.

## 8. Effort and risk

| item | effort | risk |
|---|---|---|
| **top4 pruning** (oracle `keep`, kernel `stop = true`, and the p5 grid equivalent) | S | low; −0.0025 % |
| **rep lists** (oracle, K2-side near-list kernel, fixup-side history emission, per-pass list kernel, DP validation) | M–L | medium. The pre-pass cost decides whether it pays; build it as a fused K2 epilogue and measure the list kernel first |
| LL changes | – | rejected |
| edge-price precomputation | – | rejected (LDS lookups are already cheap) |

## 9. Dead ends checked

- **Precomputing OF/ML edge prices:** state-independent but already cheap LDS lookups. A global edge list would
  be slower.
- **Every cheaper LL model:** −0.02 % to −1.0 %.
- **Rep sets without the in-series "near" offsets:** at best −0.041 % (hist8). Paths take offsets from matches
  earlier in the series that the previous pass never used.
- **Rep sets from the K2 chain walk:** −0.17 %. Rep offsets lie beyond the chain depth.
- **Exact validation with fallback probes on a miss:** 14 % of probes miss, so most warp-trips still run the
  dependent probe.
- **Capped rep lists (cap ≤ 8):** −0.02…−0.24 %. Near-first ordering is worse.
- **Greedy-parse reps in every pass** instead of the previous pass's: −0.21 %.
- **Unrolled fixed-size neighbour loads on the GPU:** 8 neighbours cost more than they save (1.09× at L0).

## 10. Top 3

1. **top4 relaxation pruning.** −0.0025 % ratio on the full corpus; one line in the oracle and one in the kernel;
   L2 pass about −12 % on today's kernel. It makes every relaxation exactly one grid step, which also simplifies
   p5's grid.
2. **Validated rep lists (B), on top of p5.** −0.0115 % ratio (full corpus: opt16 −0.0115 %, opt14 −0.0108 %;
   both well under 0.03 %). The DP pass drops to 0.81× of p5 at both levels, measured with a mimic. Net K3 about −9…−11 %
   after the modelled pre-pass. Worth a prototype only after measuring a real list-building kernel.
3. **Keep LL and the edge prices as they are.** The "decoupled" gain is entirely rep-probe removal plus pruning.
   Spend no effort on precomputing prices or reshaping LL.
