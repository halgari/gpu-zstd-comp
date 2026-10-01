# a09: data-aware shortcuts that skip or cheapen DP work

Research only. No repository file was changed. Scratch work is in `/tmp/claude-1000/m6/a09-data-aware/`:
- `a09/`: an instrumented port of `opt::Dp::segment_ring` (`dp.rs`), a corpus sampler with per-position DDS regions (`corpus.rs`) and the experiments (`main.rs`);
- `gzc-core/`: an unmodified copy;
- `res/`: raw outputs and the per-block CSV;
- `dead-skip-fold.diff`: the scratch kernel and test diff. The worktree has been removed.

## TL;DR

1. **The DP visits about one position per byte: 67 K trips per block per pass. 61 % of all positions are provably "dead".**
   - A dead position has no earlier position in the block with the same first 3 bytes, so no candidate and no rep can match there.
   - 99 % of the positions with no candidate are dead.
   - Dead positions account for about 58 % of all trips: probe misses (43 %) plus in-series empty searches (15–18 %).
2. **Exact skip, measured on the GPU and byte-identical.**
   - The skip fast-forwards dead probe runs and skips the search at dead in-series positions.
   - K3 goes from 34.2 to 30.8 µs/block (−10 %).
   - "Fold" adds this: positions that need only the literal extension (dead, or the L0 `+128` skip) are processed inside the same trip. With it, the cheap passes go from 8.2 to 7.0 µs.
   - Best exact mix (fold in the cheap passes, skip in the final pass): about 29.6 µs (−13 %).
3. **One ratio-traded shortcut survives: final-pass `sufficient_len` 16 for DDS only.**
   - The final pass goes from 9.2 to 7.5 µs.
   - Full-corpus ratio 1.37133 at 64 KiB (L16 1.37100) and 1.32785 at 16 KiB (L16 1.32774).
   - NIF with 2 passes instead of 3 is free (NIF ratio slightly up).
   - Combined with 2: K3 about 28.5 µs (−17 %), so opt16 about 38.9 µs/block, about 1.64 GB/s on the 5090 (projection from measured K3).
4. **Dead ends**:
   - per-block routing to lvl9 (oracle bound 8 % of the time within budget);
   - BC phase-aligned search (−0.3 to −2.3 % per kind);
   - cheap-pass `target_length` (no time change);
   - region-based skipping (gain is spread proportionally to bytes);
   - exact price convergence (0.06–0.6 % of blocks).

## What was measured, and how

**CPU oracle port.** `a09/src/dp.rs` is a statement-for-statement port of `segment_ring` with knobs and trip logging.
- It is byte-identical to `gzc_core::opt::parse`: 908 of 908 blocks of a 1/128 sample, and full-corpus opt16 = 1.37144 (64 KiB) and 1.32794 (16 KiB), the published numbers.
- Every loop iteration of the GPU's flattened lane loop is logged as one trip, with a code: probe miss, series search with *n* relaxation steps of 4 lengths, literal-only, and so on.
- A lockstep model sums, per block, the maximum over its 16 lanes of each trip's cost (1, plus 0.1 per relaxation step).
- The model reproduces the perf study's 4385 warp-level trips per block (4397 here).

**Samples.**
- Sample A: the m5 ratio-driver stratified sample, every 32nd DDS block and every 8th NIF block (phase stride/2): 3633 blocks, weighted to the full corpus.
- Ratio finalists were run on the **full corpus** (100 754 blocks, dds + nif) at 64 KiB and 16 KiB.

**Region model.** Each parse is costed with its own empirical entropies:
- literals at −log2 p;
- LL/ML/OF codes at −log2 p plus their extra bits;
- each sequence's cost is spread over its match bytes.

The model is within 0.7 % of the real frame sizes (opt 4713 MB modelled vs 4744 MB real, sample-weighted). Regions:
- the DDS header (the first 128 bytes);
- BC fields by phase: DXT5 alpha endpoints 0–1, alpha indices 2–7, colour endpoints 8–11, colour indices 12–15; DXT1 endpoints 0–3, indices 4–7;
- mip level, from the header's width, height and mip count;
- NIF, and padding.

The corpus is DXT1 (30 % of blocks), DXT5 (66 %), 2 uncompressed DDS files, and NIF (5 %). There is no BC5/BC7.

**GPU.**
- Setup: RTX 5090, a scratch test (`a09_dead_skip`) in a worktree. 2900 corpus blocks (every 34th) in one batch, all 4 opt16 passes through `OptPasses`/`time_passes` with GPU-side prices, median of 3.
- Variants were interleaved with the unmodified kernel in every round (base, skip, fold, tl, tl+skip, tl+fold). Ratios between them are what count.
- **Absolute times are under contention.** About 11 agents shared the machine. `nvidia-smi` showed 1–11 % utilization and 8–25 GB of VRAM used by other agents' k3opt/probe processes (one run OOM'd and was retried). `uptime` load average was 30–125. Single outlier rounds (for example 14.65 µs) are visible in the raw logs; the medians below exclude them.
- Dead-run lengths were computed on the host and written into the dead positions' unused `w1` word (`0x8000_0000 | run`). There `w0 == 0`, and the unmodified kernel ignores `w1` when `lenB == 0`, so base and variants read the same buffers.

## Where the DP spends work (sample A, per 64 KiB block per pass)

| trip kind | pass 0 (L0) | final (L2) | share |
|---|---:|---:|---:|
| probe miss at a dead position | 28 510 | 28 102 | 42–43 % |
| probe miss, not dead (a rep check is needed) | 209 | 204 | 0.3 % |
| in-series search at a dead position (always empty) | 10 180 | 11 859 | 15–18 % |
| in-series literal-only (`+128` skip, `cur == last_pos`, past ilimit) | 10 559 | 5 463 | 8–16 % |
| search with matches (probe hit or series) | 17 448 | 21 022 | 26–31 % |
| non-dead empty search | 120 | 139 | 0.2 % |
| **total** | 67 027 | 66 789 | about 1.02 per byte |

- **Dead positions by kind:** DXT1 59 %, DXT5 63 %, NIF 45–53 %, raw DDS 44 %.
  - By field: DXT1 colour indices 78 %, DXT1 endpoints 41 %; DXT5 alpha endpoints 55 %, alpha indices 63 %, colour endpoints 59 %, colour indices 74 %; DDS header 28 %.
  - Positions with no candidate are only 0.4–0.9 points above the dead share.
- **Long literal runs.** Probe-miss runs have a mean length of about 6. Share of miss positions by run length: 1: 2 %, 2–3: 15 %, 4–7: 25 %, 8–15: 31 %, 16–31: 13 %, 32–63: 7 %, 64 and over: 6 %.
- **Lockstep vs work.** Warp-level trips are 4400 per block, against 4190 for the mean lane. Lanes are well balanced; the per-trip union cost is the issue, as the perf study says.
- **Kinds.** Trips per pass in the lockstep model (r = 0.1): DXT1 5161, DXT5 5045, NIF 5390. The modelled heavy tail is mild (max/mean 1.15). The GPU's 2–3× heavy blocks come from relaxation and `match_len` extension cost per trip, which this model does not capture.
- **DP vs greedy longest.**
  - 7.1 % of the final parse's sequences are shorter than the longest match available at their start.
  - 10 % of literal positions had a candidate of 3 bytes or more that the DP declined.
  - A greedy-longest parse over the same candidates (mm4) gets 1.33305, which is *below* lvl9 (1.33812) and 2.7 % below opt16 (1.37010).
  - So the DP's value lies in a minority of start/stop decisions that local structure does not predict.

### Ratio gain over lvl9 by region (sample A; share of opt16's total gain over lvl9)

| field | bytes | opt b/B | lvl9 b/B | share of gain |
|---|---:|---:|---:|---:|
| colour endpoints | 30.6 % | 5.21 | 5.41 | **44.9 %** |
| alpha indices (DXT5) | 24.5 % | 5.87 | 5.99 | 21.9 % |
| colour indices | 30.6 % | 6.65 | 6.74 | 19.4 % |
| NIF | 4.4 % | 4.40 | 4.87 | 15.4 % |
| alpha endpoints (DXT5) | 8.2 % | 5.36 | 5.33 | **−1.6 %** |
| DDS header / raw pixels / padding | 1.8 % | | | 0.0 / 0.1 / −0.1 % |

By mip level: L0 gives 64 % of the gain (70.5 % of bytes), L1 15 % (17.4 %), L2–3 4.9 % (5.8 %), L4+ 0.35 % (0.3 %).
- Trips are spread almost exactly in proportion to bytes: alpha endpoints 8 %, alpha indices 24 %, colour endpoints 30 %, colour indices 33 %, NIF 4.6 %, mip tails (L4+) 0.3 %.
- No region carries much DP work while giving no gain, apart from the DXT5 alpha endpoints (8 % of trips, slightly negative gain). Region-selective DP has nothing to win.

## Ideas

### 1. Dead-position skip (exact, byte-identical): MEASURED −10 % K3

**Mechanism.**
- `dead(p)` means: `p < HASHED_POSITIONS`, and the h3 chain walk from `p` reaches `NO_POS` within its 4 steps without finding 3 equal bytes.
- Every earlier position with the same 3 bytes has the same `hash3` key, so it would be on that chain. So nothing earlier shares the 3 bytes, and no A/B record and no rep (`rep[0]`, `rep[1]`, `rep[2]`, `rep[0] − 1`, all earlier positions) can reach `MIN_MATCH`. `get_all_matches(p)` is empty under every rep state.
- Depth 4 already catches 99.3 % of the dead probe misses; unlimited depth adds 0.7 %.

**K3 changes.**
- Outside a series: before the trip's loads, jump `st_ip` over the dead run (clamped by `ilimit`).
- In a series: at a dead position, `advance` without calling `get_all_matches`.

**Measured** (median µs/block, interleaved, under contention):

| | pass 0 | pass 1 | pass 2 | final | total K3 (with fix-up 0.27) |
|---|---:|---:|---:|---:|---:|
| base | 8.22 | 8.20 | 8.25 | 9.23 | 34.2 |
| skip | 7.34 | 7.40 | 7.43 | 8.38 | 30.8 (**−10 %, ×1.11**) |

- Exactness: 0 mismatches against the base kernel and against `opt::parse`, on 600 corpus blocks (all passes, final parse).
- The lockstep model predicted −15 %. The real lane loop has more fixed cost per trip than the model.

**Production form.** K2opt knows when a position is dead: no record, and its h3 walk ended in `NO_POS`. The open question is how K3 finds the run end.
- (a) Recommended: a 1-bit-per-position bitmask (8 KiB per block, 1.6 % of the candidate words), built in K2's workgroup memory. K3 jumps with `firstTrailingBit(~word >> bit)`, one load per 32 positions.
- (b) Or K2 writes the run length into the dead position's `w1`, which is free because `w0 == 0` there. That needs a suffix scan per segment.

The K2 side is not measured. It is estimated at under 0.3 µs/block, against the measured 3.4 µs saved. With (b) the VRAM is unchanged; with (a) it grows by 1.6 % of the candidate buffer, and `vram_bytes` must count it.

- Effort: S–M. Risk: low. The proof is short; the remaining risk is in getting the K2 flag right, including hash-collision chains, which the argument covers.

### 2. Fold literal-only positions into the trip (exact): MEASURED; use in cheap passes only

**Mechanism.** The in-series half of the trip becomes a loop. After the literal extension at `cur`, if the trip would only `advance` (dead position, L0 `+128` skip, or `inr > ilimit`) and `cur < last_pos`, then reload `p`, `x`, `xprev`, `w0`, `w1` for `cur + 1` and do the next literal extension in the same trip. The statements are identical, so the parse is identical.

**Measured:**

| | cheap passes | final pass | total |
|---|---|---|---|
| fold | 7.14, 6.97, 7.00 | 8.65 | 30.0 |

- Exactness: 0 mismatches against base (600 blocks).
- Cheap passes: −15 % against base, and −5 % on top of the skip, because L0 has 10.6 K `+128` skips per block.
- Final pass: worse than skip-only (8.65 vs 8.38), because it has fewer skips and more divergence.
- **Exact best mix (fold in L0 passes, skip in the L2 pass): about 3 × 7.0 + 8.38 + 0.27 ≈ 29.6 µs (−13 %, ×1.16).**

Effort: S (one loop wrap; `p`, `x`, `w0`, `w1` become vars). Risk: low.

### 3. Final-pass `sufficient_len` (target_length) 16 for DDS (new oracle and preset): MEASURED

**Ratio:**
- With cheap passes at T = 16 the ratio is −0.0015 %, and there is **no GPU time change** (8.2 µs; the L0 early abort already caps relaxation).
- The final L2 pass is where T matters.

Full-corpus ratio by final-pass `target_length`:

| final-pass `target_length` | 64 KiB ratio | 16 KiB ratio |
|---|---:|---:|
| 32 (today) | 1.37144 | 1.32794 |
| 24 | 1.37139 | |
| 20 | 1.37136 | |
| 16, all kinds | 1.37125 (NIF −0.16 %) | |
| **16 for DDS, 32 for NIF** | **1.37132** | **1.32785** |

The L16 gates are 1.37100 (64 KiB) and 1.32774 (16 KiB).

**Measured on the GPU** (`target_length` 16 in every pass), median µs/block:

| variant | cheap passes | final pass | total |
|---|---|---|---|
| tl | 8.2 (unchanged) | 8.45 | |
| tl + skip | | 7.5 | |
| tl + fold | 6.8–7.1 | 7.55 | 28.5 |

**Combined with 1 + 2: about 28.5 µs (−17 %, ×1.20).**

**Requirements and caveats.**
- A per-pass `target_length` field (ring size per kernel; scratch sized for the larger).
- A per-block kind tag: the downloader knows the file type, or it can be read from the DDS FourCC.
- The 16 KiB margin shrinks from +0.015 % to +0.008 %.

Effort: S–M. Risk: medium (ratio margin).

### 4. Per-kind pass count: NIF with 2 passes (new oracle): CPU-measured

- NIF ratio is 1.79839 with 2 passes against 1.79815 with 3, so 2 passes is better. Total ratio 1.37145 at 64 KiB (+0.0007 %) and 1.32794 at 16 KiB (unchanged).
- DDS needs all 3 cheap passes: with 2 passes DXT5 drops 0.10 %.
- Saving: one L0 pass on 5.1 % of blocks, about 1.3 % of K3 (estimated). It needs the batch split by pass count (NIF's final pass dispatched with DDS's third cheap pass), so it is only worth doing together with 3.

Effort: S–M. Risk: low.

### Dead ends (measured)

- **BC-aligned search** (search only at given BC phases).
  - Phase distribution of match starts: BC1 phase 0 73 %, phase 1 16 %, the rest spread; BC3 phase 0 22 %, 8 23 %, 9 11 %, the remaining 44 % spread across all phases.
  - Every restriction blows the budget (sample A):

| allowed phases | kind | ratio change |
|---|---|---:|
| BC1 {0, 1} | DXT1 | −0.86 % |
| BC1 {0, 1, 4, 5} | DXT1 | −0.30 % |
| BC3 {0, 1, 8, 9} | DXT5 | −2.3 % |
| BC3 all but phase 9 | DXT5 | −0.58 % |

  - Time saved is only 5–10 % (model), because the trips still happen.
- **Run the DP only where lvl9 and opt disagree** (per-block routing to lvl9s12seg).
  - The total budget is 0.032 % = 1.34 % of opt16's gain over lvl9.
  - An *oracle* router fits only 9.2 % of blocks (8.3 % of modelled DP time) under it.
  - Heavy blocks gain 2.9× the average (5.1 % vs 1.75 %), so they are the worst ones to route.
  - Per-segment routing does not help under lockstep: the block's time is set by its busiest lane.
- **Lazy parse for "no-gain" segments.** Same problem. Greedy-longest over the opt candidates is below lvl9. No region is gain-free (see the region table).
- **Smaller `target_length` in the cheap passes.** Ratio is free, but the GPU time is unchanged.
- **Exact price convergence** (skip the remaining cheap passes when the prices repeat). Prices repeat in 0.06 % (pass 2 vs 1) and 0.6 % (pass 3 vs 2) of blocks.
- **Mip-tail or header special-casing.** These are 0.3 % and 0.003 % of the bytes and trips.
- **Rep-only fast path for non-dead misses.** Only 0.3 % of trips.

## GPU speedup implied

All figures are opt16 at 64 KiB on the RTX 5090. K3 is measured; the end-to-end figures are projections that keep K1/K2/K4/K5 at the brief's 11.4 µs.

| config | K3 µs/block | opt16 µs/block | MB/s (from 1426) | ratio |
|---|---:|---:|---:|---|
| today | 34.2 | 44.7 | 1426 | 1.37144 |
| 1 (dead skip) | 30.8 | 41.3 | ≈ 1540 | identical |
| 1 + 2 (exact mix) | ≈ 29.6 | ≈ 40.0 | ≈ 1590 | identical |
| 1 + 2 + 3 (+ 4) | ≈ 28.5 (≈ 28.2) | ≈ 38.9 | ≈ 1640 | 1.37133 / 1.32785 (16 KiB) |

- **4060 class** (K3 is about 75 % of the time, throughput-bound): the same −13 to −17 % on K3 gives ×1.11–1.15, so opt16 goes from about 0.20–0.29 to about 0.23–0.33 GB/s. That is still far from line rate. These shortcuts stack with the kernel-level work (predication, fused passes) rather than replacing it.
- The dead bitmask also helps K2opt-free consumers: cover literals, and any future lazy/opt hybrid.

## Top-3 recommendation

1. **Dead-position skip (exact).** The K2 flag plus a bitmask or run lengths, a jump in the probe path, and no search at dead in-series positions. Measured −10 % K3, byte-identical, low risk.
2. **Fold literal-only positions in the L0 passes (exact).** Measured −15 % on each cheap pass (−5 % on top of 1). Keep the final L2 pass on plain skip.
3. **New preset variant: final-pass `sufficient_len` 16 for DDS blocks (NIF keeps 32 and drops to 2 passes).** Measured final pass −18 % on top of 1, K3 −17 % in total. Ratio 1.37133 at 64 KiB (+0.024 % over L16) and 1.32785 at 16 KiB (+0.008 %). Decide based on how much ratio margin other agents' proposals need.
