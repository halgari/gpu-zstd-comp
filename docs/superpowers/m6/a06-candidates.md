# a06: candidate generation for opt16 (K1 + K2opt)

Agent a06-candidates, 2026-09-30. Research only: no repository file was changed.

**Scratch location.** Prototypes are in `/tmp/claude-1000/m6/a06-candidates/`:
- `a06-scratch.diff`: the worktree diff (K1/K2opt shader and host hooks, and the K3 timing hook);
- `cand_var.rs`: the CPU ratio tool, a `gzc-core` example;
- `a06_bench.rs`: the interleaved GPU bench, a `gzc-gpu` example;
- `k1_fused.wgsl`: the fused K1;
- `full64_run1.txt`: raw output of the first full-corpus run.

The worktree was removed afterwards.

## TL;DR

1. **The biggest lever here is ratio, not K1/K2 speed. opt16's candidates miss long matches.** The h4 chain
   walked 32 deep often stops before the nearest 10–12-byte match, mostly in DXT5 blocks. Adding a third chain on a
   **10-byte hash** fixes this:
   - it is cheap, because it is only built on every 4th position;
   - it lifts the ratio by +0.1 to +0.2 %;
   - that buys enough headroom to run the **opt14 schedule**: Prior seed, 1 cheap pass and the final pass, so
     2 DP passes instead of 4;
   - the result still clears libzstd L16 by more than opt16 does today.

   **Full corpus at 64 KiB** (CPU oracle, every frame verified by libzstd):

   | config | ratio | vs L16 (1.37100) | DP passes |
   |---|---:|---:|---:|
   | opt16 today | 1.37144 | +0.032 % | 4 |
   | opt14 today | 1.37064 | −0.027 % | 2 |
   | **opt14 schedule + h3 d4 + h4 d8 + h10 (every 4th position) d16** | **1.37239** | **+0.101 %** | **2** |
   | opt14 schedule + h3 d4 + h4 d8 + h10 (every 2nd position) d32 | 1.37281 | +0.132 % | 2 |
   | opt14 schedule + h3 d4 + h4 d8 + h10 (all positions) d16 | 1.37263 | +0.119 % | 2 |

2. **GPU cost of the new candidates.** The prototype is byte-exact against a CPU model: 0 mismatches over 16 blocks
   for each configuration. Measured interleaved against today's K1 + K2opt:
   - stride 4 / d16: K1 ×1.07, K2opt ×0.75–0.79, **K1 + K2 ×0.97–0.99**, the same cost as today;
   - stride 2 / d32: K1 ×1.20, K2opt ×0.86–0.89, K1 + K2 ×1.09–1.10;
   - K3opt with the new candidates: opt14 schedule ×1.01–1.015 (18.97 / 18.99 vs 18.76 / 18.68 µs/block).

   Estimated end to end (not measured):
   - **about 29.5 µs/block against opt16's 44.7, about 1.5× faster (≈ 2.1 GB/s against 1.43 on the 5090)**;
   - the ratio is higher than opt16's;
   - opt14's time is almost unchanged (+0.2 µs).
3. The ideas in the brief do not pay off by themselves:
   - **Fused K1** (all chains of a block in one task) is **1.3× slower**.
   - **Chain depth cuts cost ratio.** Going 32 → 16 costs −0.026 % on the sample, which is the whole opt16 margin.
   - **Fusing K1 + K2opt in shared memory** cannot fit.
   - **A candidate format that the DP reads more cheaply** targets a load the perf study already found is not a
     bottleneck.
   - **An exact sorted finder** is possible (a 2-digit LSD radix sort over 16-bit keys), but it is an L-effort item.
     It is worth it only after item 1, when K1 + K2 grows to about a third of the time.

## Measurement setup

**CPU ratio**
- Tool: `cand_var` (a scratch `gzc-core` example). It builds candidate words with a generalised `find_cands`: the
  same nearest-first merged walk and record rules, with any number of chains, each walked to its own depth. It then
  runs `gzc_core::opt::parse` and `write_frame`.
- The base config reproduces the oracle exactly:
  - 1/50 design sample: opt16 1.37211, opt14 1.37118;
  - full corpus: opt16 1.37144, opt14 1.37064.
- Samples:
  - the design sample: every 50th 64 KiB block, offset 0, 2016 blocks, libzstd L16 = 1.37175;
  - the prior's training sample: offset 25 (L16 1.37179);
  - 16 KiB and 32 KiB builds: every 50th block;
  - the full corpus (100 754 blocks) with `VERIFY=1`, where every frame is decoded by libzstd.
- The CPU was heavily shared (load average 90–140). Ratios are deterministic, so this does not matter.

**GPU**
- Tool: `a06_bench`. It runs today's `OptCandKernel(OPT16)` and the variant K1 + K2opt on the same data, in the order
  base, variant, base, variant (5 reps each, median per batch).
- Workload: 2 batches of 2048 blocks (every 10th corpus block).
- The variant's candidates are checked against a CPU model of the recipe: 0 mismatching positions over 16 blocks,
  for every configuration.
- K3 was timed with `k3opt_passes_timing`: 2900 blocks, candidates from the host, runs interleaved base / new / base / new.
- **All GPU numbers were taken under contention.** About 10 other agents were active. `nvidia-smi` showed 1–85 %
  utilisation and 17–30 GB of VRAM in use at the start of the sets, and one run failed with a VRAM out-of-memory
  error from other processes and was repeated.
- Runs that overlapped another agent's burst were discarded and repeated. They show up as an outlier base or
  variant, for example base K1 9.29 µs or opt16-new 58.6 µs.
- Only ratios between interleaved runs are reported as results.

## 1. Chain depth and the h3 chain (brief ideas: "reduce depth", "drop or cheapen h3")

1/50 sample, opt16 schedule (the base is 1.37211 and L16 on this sample is 1.37175, so the headroom is +0.00036):

| variant | ratio | Δ | merged-walk steps / position |
|---|---:|---:|---:|
| h4 d32, h3 d4 (today) | 1.37211 | — | 3.88 |
| h4 d16 | 1.37176 | −0.00035 | 2.87 |
| h4 d8 | 1.37136 | −0.00075 | 2.26 |
| h4 d4 | 1.37075 | −0.00136 | 1.88 |
| h3 d3 / d2 | 1.37206 / 1.37180 | −0.00005 / −0.00031 | 3.76 / 3.59 |
| h4 d16, h3 d3 | 1.37171 | −0.00040 | 2.75 |
| h4 d64 / 128 / 256 | 1.37244 / 1.37279 / 1.37329 | +0.00033 / +0.00068 / +0.00118 | 5.6 / 8.4 / 13.4 |
| exact-key chains (no hash collisions), d32 | 1.37216 | +0.00005 | 3.17 (−18 %) |
| "ideal": exact chains h4 d4096, h3 d64 | 1.37492 | +0.00281 | 84 |

**Verdict.**
- With today's recipe, depth cannot be cut for free. h4 d16 uses the entire L16 margin, and at 16 KiB it would fail.
- This contradicts `m5-ratio-drivers.md`, which says D saturates at 8–16. That claim came from the prototype's float
  prices and its h4+h8 recipe.
- The h3 chain is not the problem: d4 → d3 is almost free (−0.00005), but it saves little.
- The ratio keeps climbing with depth. **The candidates are depth-starved, not over-provisioned.** That points to
  §2.

## 2. The main finding: a sparse 10-byte chain

The first test added a third chain with an exact w-byte key (depth 32) to today's h4 d32 + h3 d4 (1/50 sample, opt16
schedule):

| w | 5 | 6 | 7 | 8 | 9 | **10** | 11 | 12 | 13 | 14 | 16 | 20–32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Δ ratio (×1e-5) | +14 | +35 | +20 | +77 | +216 | **+244** | +218 | +204 | +27 | +16 | +12 | +6–9 |

- **Per kind** (w = 10): DXT5 +0.25 % (1.34728 → 1.35072), NIF +0.08 %, DXT1 +0.005 %.
- The cliff between 12 and 13 bytes is DXT5 structure. A 16-byte block is 8 bytes of alpha plus 4 bytes of color
  endpoints plus 4 bytes of color indices. Matches that cover the alpha block and the color endpoints (up to 12
  bytes) are common, but they sit behind dozens of 4-byte and 8-byte repeats on the h4 chain.
- The recipe is still generic: it is a hash chain, and nothing in it is specific to DDS.

A 16-bit **hashed** 10-byte key, as a GPU would use, behaves like the exact key: 1.37391 against 1.37392 for h4 d8 +
h10 d16. Sampling positions costs little:

| (1/50 sample) | opt16 schedule | **opt14 schedule** (Prior seed, 1 cheap + final) | walk steps / position |
|---|---:|---:|---:|
| today | 1.37211 | 1.37118 | 3.88 |
| h4 d8 + h10 d16 | 1.37391 | 1.37326 | 2.87 |
| h4 d16 + h10 d16 | 1.37410 | 1.37344 | 3.44 |
| h4 d8 + h10 on p ≡ 0 mod 2, d32 | | 1.37345 | 2.55 |
| **h4 d8 + h10 on p ≡ 0 mod 4, d16** | | **1.37302** | **2.35** |
| h4 d8 + h10 d16, **1 DP pass only** (opt14 without its cheap pass) | | 1.36920 | 2.87 |

**Held-out and block-size checks** (opt14 schedule, "s4" = stride 4, d16; "s2" = stride 2, d32):

| sample | libzstd L16 | opt16 today | new s4 | new s2 |
|---|---:|---:|---:|---:|
| 64 KiB, 1/50 offset 25 | 1.37179 | 1.37239 | 1.37329 (+0.110 %) | 1.37367 |
| 64 KiB, full corpus | 1.37100 | 1.37144 (+0.032 %) | **1.37239 (+0.101 %)** | **1.37281 (+0.132 %)** |
| 32 KiB, 1/50 | 1.35089 | 1.35226 | 1.35300 (+0.156 %) | 1.35326 |
| 16 KiB, 1/50 | 1.32513 | 1.32532 (+0.014 %) | 1.32588 (+0.057 %) | 1.32602 |

- Other full-corpus points: opt14 schedule with h4 d8 + full h10 d16 gives 1.37263, and with h4 d16 it gives 1.37281.
  The opt16 schedule with h4 d8 + h10 d16 gives 1.37319.
- One DP pass is not enough, even with better candidates: 1.3692–1.3698 against L16's 1.37175. **Two passes are
  the floor.**
- The priors (`OPT_PRIOR_*`) were not retrained for the new candidates. The Prior seed's cover literals do come from
  the new candidate words, and `opt::parse` computes them. Retraining can only help.

### GPU prototype (byte-exact against the CPU model)

**K1 changes**
- `k1_chains_sg.wgsl` builds a third Opt3 chain.
- Hash: `((lo·0x9E3779B1) ^ (hi·0x85EBCA77) ^ ((bytes 8..10)·0x27D4EB2F))·0xC2B2AE3D >> 16`, with bytes past the
  block read as zero.
- The words load grows from `vec3` to `vec4`.
- The chain is built over *slots* (position = slot × stride), so a stride-4 chain costs a quarter of a chain's tiles.
- Fingerprints are `pred_fp`.

**K2opt changes**
- The merged walk gets a third head, with `X_DEPTH`, and is skipped when `p % stride ≠ 0`.
- Upper bounds from the fingerprint: 3 if the lo field differs, 4 if the byte field differs.

Interleaved medians (µs/block, under contention; base = today's K1 + K2opt in the same runs):

| variant | K1 | K2opt | K1 + K2 | vs base (K1 / K2 / sum) |
|---|---:|---:|---:|---|
| h10 all positions, h4 d8, dx16 (3 runs) | 9.91–10.00 | 2.52–2.67 | 12.44–12.62 | 1.47 / 0.86–0.91 / **1.29** |
| **h10 stride 4, h4 d8, dx16** (2 clean runs) | 7.29 | 2.27–2.32 | 9.56–9.61 | 1.07 / 0.75–0.79 / **0.97–0.99** |
| h10 stride 2, h4 d8, dx32 (2 runs) | 8.12 | 2.58 | 10.70 | 1.20 / 0.86–0.89 / **1.09–1.10** |
| (base in those runs) | 6.78–6.81 | 2.90–3.02 | 9.68–9.83 | |

K3opt with the new candidates, `k3opt_passes_timing`, 2900 blocks, runs interleaved:

| candidates | opt14 schedule (cheap pass, final pass, fix-up) | opt16 schedule |
|---|---|---|
| today, run 1 | 18.76 (9.17, 9.32, 0.27) | 34.29 |
| new s4, run 1 | 18.97 (9.12, 9.58, 0.27) | 34.23 |
| today, run 2 | 18.68 (9.11, 9.30, 0.27) | 34.42 |
| new s4, run 2 | 18.99 (9.28, 9.43, 0.27) | (contended, discarded) |

**Projection** (not measured end to end):
- New preset: K1 7.35, K2 2.0, K3 18.2, K4 0.61, K5 1.31, for **≈ 29.5 µs/block**.
- Against opt16's 44.68 that is **≈ 1.5×**, or about 2.1 GB/s on the 5090. The ratio beats opt16 by +0.07 %.
- On an 8 GB card the saving is dominated by the 2 dropped DP passes, so it scales about like the K3 share: 1.4–1.5×.

**Costs and risks**
- **New oracle and preset.** The output changes. The oracle needs:
  - `Hashes::Opt3` plus a third chain;
  - a stride field;
  - `find_cands` with 3 chains;
  - retrained priors.
  The ratio gate passes with 3–7× today's margin at every block size tested.
- **VRAM.** The prototype stores the stride-4 chain at full size in `pred`, which adds 256 KiB per block. A compact
  layout (index p/4) adds 64 KiB per block, which is about −3 % on the maximum batch.
- **The 16 KiB margin** grows from +0.014 % to +0.057 % on the sample. That is still the thinnest margin; the full
  16 KiB corpus was not run.
- **Effort: M.** Oracle plus tests, K1/K2 changes (already prototyped), priors, and the gates.

## 3. Fused K1 (build all chains in one pass, the T2 suggestion)

- **Prototype:** `k1_fused.wgsl`. One task = one block, and every tile runs all 3 chains:
  - the words load is shared;
  - both barriers are shared;
  - each chain has its own head table (3 tables per workgroup);
  - each chain has its own ballot buffers.
- **Correctness:** 0 `pred` mismatches against the unfused K1.

| K1 for 3 chains (all positions) | µs/block |
|---|---:|
| unfused, 128 groups (1 table each, 32 MiB live) | 9.91–10.00 |
| fused, 128 groups (3 tables each, 96 MiB live) | 12.7–13.0 (**×1.28–1.30**) |
| fused, 85 groups | 14.4 |
| fused, 43 groups (same 32 MiB of live tables as unfused) | 25.9 (×2.6) |

- **What this shows about K1.** It is not bound by per-tile latency, which fusing would hide. Each workgroup's time
  is spent issuing work: the ballots and the matching. Fusing gave only about 14 % more work per workgroup.
- **The lever is more concurrent workgroups.** That number is capped by the live head tables that fit in L2. T2
  measured 1.5× at 256 groups on the 5090, and a contended run here gave base 4.53 and variant 6.64 µs.
- **Dead end** as proposed.

**A related open idea (not prototyped).**
- Halve each head table by using u16 entries. The one writer per half-word and tile can update it with `atomicXor`,
  using the old value it already loaded.
- Because a u16 entry has no room for a tag, the table would be cleared per block. That costs about 128 stores per
  lane.
- The result is twice the workgroups for the same L2 footprint, with output unchanged. Effort S–M.

**Target GPUs.** An RTX 3060 has 3 MB of L2 and a GTX 1660 Super 1.5 MB, so even 128 × 256 KiB of tables thrash
there. The persistent-table K1 is fragile on exactly the target class. §5 matters more there than on the 5090.

## 4. The other ideas in the brief

**Fuse K1 and K2opt per block in shared memory.**
- K2opt reads chain words at arbitrary earlier positions of the block. That is 2 or 3 chains × 256 KiB per block,
  against 16–48 KiB of portable workgroup memory (32 KiB on Apple).
- It works only with a partitioned or sorted K1 (§5), and even then K2's random reach across the block stays.
- **Dead end.**

**Make K2opt smarter with early exits.**
- The existing exits are the stop at `c == max` and the fingerprint skips. An exit at best ≥ 33, justified by
  `sufficient_len`, is not byte-identical: the stored B length decides the commit length.
- The walk is already short: 2.35 steps per position with the new recipe, against 3.88 today. The new recipe is the
  better K2 win (K2 ×0.75), so no separate work is needed.

**Suffix array or LCP within a 64 KiB block.**
- The ideal-candidate bound (exact chains, depth 4096) is 1.37492. That is only +0.0025 over the new s4 recipe on the
  sample and +0.0011 over h4 d8 + h10 d16 (opt16 schedule).
- A GPU suffix sort would need about 6 prefix-doubling rounds, each a sort of 64 K elements per block. That is
  estimated at 3–5× today's K1.
- A cheap chain gets most of the benefit. **Rejected at L16 as well**, unless a later angle wants L19-class ratio.

**Build candidates once in a form the DP reads more cheaply.**
- The K3opt perf study measured 0 gain from prefetching the candidate lines. K3 is bound by divergence and the serial
  chain, not by the 8 B/position loads.
- A compacted candidate list would change K3's control flow, which belongs to the K3 angles.
- **Low value.**
- Useful fact for the K3 angles: new candidates change K3 time by only +1–1.5 %.

**Exact-key chains (no hash collisions).** +0.00005 ratio and 18 % fewer walk steps. Too small to matter alone.

## 5. The sorted finder for opt (spec S5)

**Exactness.** The sorted finder can reproduce the chains exactly:
- A hash chain equals the positions of p's bucket that lie below p's slot, in a stable sort by key. `find_best_window`
  proves this for Single.
- A stable sort by the 16-bit key (the same key today's chains use) therefore gives today's chains exactly, including
  collisions. It stays byte-identical to today's oracle, or to the §2 oracle with a third, strided key.

**Why it is not the lvl9s12seg kernel.**
- That kernel counting-sorts 12-bit keys with a 4096-counter histogram in workgroup memory. opt needs 16-bit keys,
  and 64 K counters do not fit.
- The exact port is a 2-digit LSD radix sort, 8 + 8 bits:
  - each digit pass is a 256-bin histogram, a scan, and a ballot-ranked stable scatter;
  - the intermediate array lives in global memory (L2);
  - one kernel can sort the 2 or 3 chains' keys together, sharing the words it loads.

**Cost estimate (not measured).**
- One digit pass should cost about lvl9s12's sort (1.20 µs/block), since it needs 8 ballots instead of 12 and the same
  loads.
- So about 2.4 µs per full-width chain and about 0.6 µs for a stride-4 chain. In total about 5.4 µs for h4 + h3 +
  h10/4, against 7.3 µs for the persistent K1 of the new recipe.

**K2 on sorted arrays.**
- A chain's next d entries are contiguous: they are loaded in parallel from one or two cache lines, not as a
  dependent pointer chase. So K2opt should also drop, maybe to about lvl9's window K2 (2.0 µs in lvl9s12seg).
- Fingerprints need their own word, because `(key16 << 16) | pos16` already fills a u32.

**When it matters.**
- It does not need a large L2. That removes the head-table thrashing risk on cards with 3 MB or 1.5 MB of L2.
- With §2 adopted, K1 + K2 is about 9.3 of about 29.5 µs, 32 %. That is where this pays.
- Effort: L. Risk: medium; the ballot ranking and the tail padding need the self-test treatment K1 has today.

## 6. Top-3 recommendation

1. **Adopt the §2 candidate recipe as a new opt16 oracle: h3 d4 + h4 d8 + a stride-4 h10 chain d16, on the opt14
   schedule.**
   - Speed: ≈ 1.5× opt16 (estimate built from measured kernel ratios).
   - Ratio: +0.101 % over L16 on the full corpus, against +0.032 % today.
   - Effort M, risk low: the K1/K2 prototype is already byte-exact against the CPU model.
   - Alternatives:
     - the stride-2, d32 variant: +0.132 %, with K1 + K2 +10 %;
     - keep a 4-pass "opt16+" at 1.37319 if a higher-ratio preset is wanted.
   - The +0.07 % of extra margin over today's opt16 is also budget other angles can spend, for example 2 KiB K3
     segments (−0.058 %, −10–12 % K3).
2. **A sorted K1 (exact 8 + 8-bit LSD radix) with a window-style K2opt.**
   - After item 1, K1 + K2 is about a third of the time.
   - Estimated about −2 to −4 µs/block, and robust on small-L2 cards.
   - Effort L.
3. **Cheap K1 occupancy fix.**
   - u16 head tables with `atomicXor` updates double the workgroups at the same L2 footprint, with output unchanged.
     The 5090 gains up to 1.5× on K1 (256 groups in T2).
   - Pick the group count per device by L2 size, not a fixed 128.
   - Effort S–M.

**Not recommended:**
- fused multi-chain K1 (measured 1.3× slower);
- depth cuts on today's recipe (they cost the margin);
- K1 + K2 shared-memory fusion;
- suffix arrays;
- a new candidate format for K3.
