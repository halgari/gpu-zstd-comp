# K3opt performance study (design + measured prototypes)

Date: 2026-09-30. Study only: no repository file was changed. Prototypes, harness and tools are in
`/tmp/claude-1000/k3opt-perf/`. That directory holds a copy of the worktree (`ws/`) with scratch-only host hooks,
the kernel variants `v1..v11.wgsl`, `run.sh`, `stats.sh`, `vkstats.c` and `w2s/`.

**Measurement setup**
- GPU: RTX 5090, otherwise idle. A second agent's job was on the GPU for about 1 minute; everything measured in that
  window was re-run.
- Workload: 2900 corpus blocks at 64 KiB (the `k3opt_timing` sample).
- Prices: explicit per-block tables (`PriceSrc::Buffer`, block-init values).
- Timing: µs per block per DP pass, median of 5 dispatches. The fix-up is not included; it adds 0.15 µs at L2 and
  0.24 µs at L0.
- Correctness: every variant marked "exact" was compared parse for parse with `dp_pass_with(.., Engine::Ring)`.

## 1. Summary

| kernel | L2 pass | L0 pass | exactness evidence |
|---|---:|---:|---|
| current K3opt (wg16, workgroup ring) | 18.1 | 14.9 | T3 gates |
| **v6**: relaxation restructured + ring payload moved out of shared memory → one wave | **10.2** | **9.2** | 2900 / 2900 blocks at L2 and at L0 |
| **v11**: v6 + branch-free hot helpers | **8.8** | **7.8** | 500 blocks at L2, 300 at L0 (same transformations as v6) |
| v11 on uniform "typical" blocks (block 1000 × 3400) | 6.1 | – | throughput floor of this code at 20 warps/SM |

Three findings change the picture in the T3 report.

1. **The pass time is set by a heavy tail of blocks, not by the average block.**
   - One block alone on the GPU takes 12.0 ms on average (p50 10.5, p90 18.2, max 30 ms; 100 sampled blocks).
   - A dispatch lasts as long as its slowest block, and both waves end on heavy blocks: 25.8 + 25.8 ms.
   - Heavy blocks are highly repetitive data: many relaxations and long rep extensions.
   - Warp-level relaxation iterations per block: p50 10.4 K, p99 77 K, max 130 K.
2. **Divergence costs 2.5–3.3× and is the main latency term.**
   - One active lane per block: typical block 3.8 ms, heavy block 5.6 ms.
   - 16 lockstep lanes: 9.4 ms and 18.6 ms.
   - The chain grows about +3 ms per doubling of the active lane count, the signature of a max over lanes of a
     heavy-tailed per-trip cost.
   - Relaxation loops are one source. The larger part is the union of branch bodies: probe vs series, the rep-probe
     early exits, and the `load_u32_at` alignment branch taken per lane.
3. **Occupancy is set by shared memory, and registers are the next wall.**
   - `vkstats` (§6) reports 96 registers, no spills or local memory, and 11 352 B of shared memory per wg16.
   - That allows 9 workgroups per SM, or 1530 blocks per wave, so 2900 blocks take 2 waves (measured wave boundary:
     1530 → 25.8 ms, 2040 → 48 ms).
   - Shrinking shared memory to 3.5 KB puts 2900 blocks in one wave. The limit then becomes registers: about 20
     warps/SM at ≤ 100 registers, about 16 at 112–117, measured.

**Recommendation.** Adopt the v11 changes in this order: 3 and 6 first, then 1 (§3). That is about 2× now with low
bit-exactness risk. Then do 4 (predicated trip) and 5 (fused passes plus heavy-first order). That should bring an
L2 pass to about 6–7 µs on the 5090, the throughput floor measured with uniform blocks. Do not build the
subgroup-cooperative relaxation, since it measured 1.7× slower. Do not shrink segments either (−0.058 % ratio for
−10 %).

## 2. Where the time goes (measurements)

Per 4 KiB segment (lane), mean over 2900 blocks, current kernel at L2:

| counter | lane mean | warp-level (16 lanes, per block) |
|---|---:|---:|
| trips | 4176 | 4385 (max 4681) |
| relaxation inner iterations | 2614 | 16 760 (p50 10.4 K, p99 77 K, max 130 K) |
| seeding iterations | 681 | 7 467 |
| `match_len` word iterations | 516 | 4 084 (max 35 K) |
| series | 205 | |

Block times with the block alone on the GPU (8 copies, one per SM):

| block | kind | L2 time |
|---|---|---:|
| 1000 | typical | 8.9 ms |
| 500 | | 12.1 ms |
| 2230 | relaxation-heavy | 20.4 ms |
| 2089 | relaxation + extension heavy | 22.1 ms |
| 2626 | extension-heavy | 18.5 ms |

A warp relaxation iteration costs about 80 ns; for block 2089, 121 K extra iterations ≈ 9.7 ms.

Calibration (pointer chase, one thread):

| memory | ns per dependent load |
|---|---:|
| shared | 15 |
| L1 | 18.5 |
| L2 | 133 |

A lone lane's trip is 1.4 µs, so the trip is a long serial chain of about 25–40 memory operations plus ALU work.

Batch scaling on a fixed block set (current kernel, L2):

| blocks | ms |
|---:|---:|
| 16 | 14.1 |
| 170 | 17.6 |
| 680 | 22.5 |
| 1530 | 25.8 |
| 2040 | 48.0 |
| 2900 | 52.6 |

Contention from 1 to 9 warps/SM is only 1.47×, so the kernel is latency- and tail-bound.

Uniform typical blocks (block 1000 repeated), v6 / v11:

| blocks | ms |
|---:|---:|
| 170 | 10.5 |
| 1700 | 15.7 (12.6 with v11) |
| 3400 | 20.7 |

At 20 warps/SM the time rises about 0.5 ms per extra warp/SM, so the code is close to issue- or LSU-bound there.
That is the 6.1 µs/block floor.

## 3. Ranked plan

Speedups are for 2900 corpus blocks, measured on top of the previous step unless marked projected.

| # | change | measured gain | exactness risk | effort |
|---|---|---|---|---|
| 1 | **Relaxation restructure**: one flattened loop for seeding and relaxation (a series start is a relaxation from a virtual `n0` at cur 0, last_pos 0); records last first, lengths in steps of 4 with independent loads; the fill reduced to at most slots `c0+1`, `c0+2` | L2 18.1 → 15.0 (−17 %), L0 14.9 → 14.0 | low (argument in §4; gate suite + corpus pass) | S |
| 2 | u16 literal prices; no histogram arrays in buffer-price passes (block-init histogram aliased onto the ring or done in a pre-pass) | shared 11.4 → 9.8 KB; no time change alone (still 2 waves) | none (prices < 65536 are already enforced) | S |
| 3 | **Ring payload out of shared memory**: price stays in `var<workgroup>` (132 B/lane); reps / litlen / mlen / offBase (12 B/node) go to a per-workgroup global scratch (6.3 KB/block, L1/L2-resident) | **L2 15.0 → 10.2 (−32 %), L0 → 9.2**; shared 3.5 KB, 96 regs; 2900 blocks in one wave | none (same values, other memory); 2900 / 2900 exact at both levels | S–M |
| 4 | **Branch-free hot helpers** (predicated trip, first part): `load_u32_at` without its alignment branch (x, rep sources, `match_len`); `ll_price` and `rep_after` as selects | L2 10.2 → 8.8 (−14 %), L0 9.2 → 7.8 | none (arithmetic identities); 500 / 300 exact | S |
| 5 | Predicated trip, rest (projected): rep probes without early `return` (compute all three, choose with selects), the candidate loop without `continue`/`break`, `end_series`/finish as predicated stores | projected −10–20 % (continues the trend of 4) | low | M |
| 6 | **Register budget ≤ 96–100** as a hard gate (`vkstats`). v7/v8 (carry `ld(cur-1)` in registers, early node load) cut the per-block chain 8–10 % but raised registers to 112–117 → 16 warps/SM → 2900 blocks no longer fit one wave (14.5 µs) | protects the −32 % of 3 | none | S (tooling exists) |
| 7 | **Heavy-first block order** (LPT) for multi-wave batches: host sorts by a cost proxy (Σ min(max(lenA, lenB), 32) − 2 over the candidate words; later passes can use the previous pass's cost) | 4000-block batch: v6 55.9 → 44.8 ms (−20 %), v11 47.1 → 38.9 ms. No effect at one wave | none (only the order changes; outputs are per block) | S |
| 8 | Fused passes (e): one persistent dispatch per batch, each workgroup runs its block's passes back to back; in-workgroup fix-up + histogram + next price tables; intermediate passes write no sequences (candidate words stay intact) | projected −10–20 % per pass (the tail overlaps the next pass; no launch/fix-up gaps) | medium (the fix-up/histogram moves in-kernel; the per-pass histograms can be checked against `opt::passes`) | L |
| – | Subgroup-cooperative relaxation (a) | **1.7× slower** (L2 30.4): a round (owner search by shuffles + record fetch + compare) costs about 0.4–0.8 µs; the ×4 in-lane unroll (1) keeps the independent-items idea without cross-lane cost | exact (see §4) | M; rejected |
| – | 2 KiB segments (b), wg32 = full warps | L2 10.2 → 9.15, L0 → 8.05 (−10–12 %) | **changes output**: ratio 1.39999 → 1.39918 (−0.058 %; 8 KiB +0.029 %), 600 blocks, `opt::parse` + `write_frame` | S; rejected (worse than an extra pass trade) |
| – | wg32 (2 blocks/warp) | worse at one wave (12.3); better in multi-wave throughput mode (4000 blocks sorted: 32.6 vs 38.9 ms) | none | benchmark on 4060 |
| – | Software prefetch of the next candidate/data line; `shader_trusted` (no bounds checks); 16-byte `match_len`; ×8 unroll | 0 / 0 / +10 % / 0 (and 110 regs) | – | rejected |

**Target check.**
- With 1 + 2 + 3 + 4 the L2 pass is 8.8 µs and the cheap pass 7.8 µs, measured.
- The floor for this code on uniform blocks is 6.1 µs, so 5 + 8 are needed to reach about 6 µs.
- They work on the two remaining terms: divergence, and the heavy-block tail (the lone heavy chain is 13.8 ms
  against a 9.2 ms mean).
- The 3–4 µs projected in §3.3 of the design needs roughly one lane's chain per warp. That is not reachable without
  removing divergence entirely.

## 4. Relaxation parallelism and tie rules (answer to option (a))

At a relaxation from `cur`, the records `(ob_i, len_i)` have strictly increasing `len_i`, and record `i` covers
`mlen ∈ [len_{i-1}+1, len_i]` (`len_{-1}+1 = 3`).

- **The target sets are disjoint:** each `pos = cur + mlen` gets exactly one candidate price per step. So no
  min-reduction is needed and no "first/last writer" question arises within a step.
- **The compares are against the pre-step ring:**
  - A step writes only its own targets.
  - Every target `pos > last_pos` (the value on entry) "improves" in the oracle: either through the
    `pos > last_pos ||` short-circuit, or because the fill loop set it to `MAX_PRICE` before its compare.
- So `improves(pos) = pos > lp0 || price < ring_pre[pos]`, the fill leaves `last_pos = max(lp0, cur + longest)`, and
  the only fill-only slot is `cur + 2` (when `lp0 = cur + 1`).
- The literal extension of that slot reads only `price = MAX` and `litlen ≠ 0`. The stale fields are never observed.

**Order independence.** The oracle order (records ascending, lengths descending) is only needed for the optLevel-0
early abort, which is per record.

- Within record `i`, the writes are exactly the lengths above the first non-improving one, walking down.
- A parallel form: per record, `cut = max{mlen : ¬improves}` (an intra-step max per record, ties impossible); write
  every `mlen > cut`.
- Across rounds: carry an "aborted" flag for the current `(owner, record)`.

The `<` versus `≤` rules are local to each element: `<` in relaxation, `≤` in the literal extension. They are kept
verbatim, and no tie-breaking between candidates exists.

**Seeding.** Seeding a series equals relaxation from a virtual node at cur 0 with `lp0 = 0`. The same formula applies,
`n0.price + ll_price(0) + of_price + FEE + ml_price`, and `mrep = new_rep(st_rep, ob, ll0)`.

Slots 1 and 2 differ: the oracle writes litlen +1 and +2, the fill writes litlen 1. Both are overwritten at their
visit before any field is read, because `price ≤ MAX` holds and `pm.litlen ≠ 0` blocks the match + 1 literal check.

Verified: v1 and v3 are exact on corpus blocks at L0 and L2. v3 also passes `k3opt_matches_opt_cases` and
`k3opt_matches_oracle_synthetic`, all configurations.

## 5. Throughput estimates

Both tables are kernel-bound figures per 64 KiB block, and all figures are projections except the measured K3 pass
times.

Assumptions:
- K3: L0 passes plus fix-up at 0.24 µs, L2 pass plus fix-up at 0.15 µs.
  - opt16 = 3 L0 + 1 L2; opt14 = 1 L0 + 1 L2.
- Other kernels are the design's §3.3 projections, not measured: K1 (2 chains, persistent) 5.4, K2opt 2.1, K4 + K5
  1.6, copies 1.8, opt14 seed 0.3 µs.
- End to end is about 0.9× the kernel-bound figure.

**RTX 5090** (µs per block; GB/s):

| | K3 opt16 | opt16 total | opt16 | K3 opt14 | opt14 total | opt14 |
|---|---:|---:|---:|---:|---:|---:|
| current | 63.7 | 74.6 | 0.88 | 33.4 | 44.6 | 1.47 |
| v11 (measured K3) | 33.1 | 44.0 | 1.49 | 17.0 | 28.2 | 2.32 |
| plan (5 + 8 → about 6.3 µs/pass) | 26 | 37 | 1.8 (2.0 with a sorted K1) | 13.3 | 24.5 | 2.7 |

**4060-class, 24 SMs** (µs per block; GB/s). K3 is throughput-bound there (many waves), so per block ≈ 5090
full-occupancy per-block time × (170 · warps/SM) / (24 · warps/SM), plus the tail:
- current: 8 wg/SM (Ada shared) → about 73 µs (L0) and 89 µs (L2) per pass;
- v11: 20 warps/SM, corpus mean about 6.5 µs on the 5090 → about 42 µs (L0) and 48 µs (L2) with heavy-first order;
- other kernels ≈ 65 µs (×6).

| | opt16 | opt14 |
|---|---:|---:|
| current | 374 µs, **0.18 GB/s** | 229 µs, **0.29 GB/s** |
| v11 | 240 µs, **0.27 GB/s** | 157 µs, **0.42 GB/s** |
| plan (+ predication, fused, wg32 if it wins there) | ≈ 205 µs, ≈ 0.32 GB/s | ≈ 139 µs, ≈ 0.47 GB/s |

These are ±30 %. The design's 0.4–0.6 GB/s for opt16 on a 4060 is not reachable with a 4-pass DP. Removing passes
(prior seeding, §2.4 of the design) is worth more there than any kernel change.

- **Batch sizing matters on both GPUs.**
  - On the 5090 one wave holds about 3400 blocks (20 warps/SM at ≤ 100 regs); 2900 fits.
  - On a 4060 one wave holds about 480 blocks. Batches should be a multiple of that with heavy-first order, or the
    last partial wave costs a full heavy chain.

## 6. Tools and reproduction (all under `/tmp/claude-1000/k3opt-perf/`)

- `ws/`: copy of the worktree with scratch-only host hooks:
  - `GZC_K3OPT_SRC` overrides the kernel at run time;
  - `GZC_TRUSTED`; `GZC_NOFIX` (skip the fix-up while timing);
  - binding 6 is a scratch buffer;
  - opt `segment_log2` 11..=13 allowed;
  - `latbench`;
  - the `probe` test: `GZC_N`, `GZC_WG`, `GZC_LEVEL`, `GZC_CHECK`, `GZC_STATS`, `GZC_PICK`/`GZC_REPEAT`,
    `GZC_DUP`, `GZC_SORT`, `GZC_EACH`, `GZC_SEG`, `GZC_RATIO`, `GZC_DISPATCH`.
- `run.sh v.wgsl [ENV=..]` times a variant; `stats.sh v.wgsl` prints registers, local memory and shared memory.
  - `vkstats.c` is a Vulkan pipeline-executable-statistics tool; `w2s/` is naga 30 WGSL → SPIR-V.
  - This replaces Nsight for occupancy questions.
- Variants:
  - `v1` unified loop; `v3` ×4 relaxation; `v5a` u16 prices without histogram; `v6` payload in scratch;
    `v10` branch-free loads; `v11` branch-free `ll_price` and `rep_after`;
  - `v2*` cooperative relaxation (rejected); `v7`/`v8*` node carry (register wall); `v9` ×8; `v4` 16-byte `match_len`.
- `v11-vs-base.diff` is the kernel diff; `host-scratch.diff` holds the host hooks.
- v11 (like v6) dropped the block-init prologue and supports only the workgroup ring: the scratch index assumes
  `rix(s) = s·WG + lane`. A real port must:
  - keep PRICE_MODE 0 (alias the histogram onto the ring's shared memory; it is dead before the DP starts);
  - add the scratch buffer to `OptBuffers` (`16 × 33 × 12 B` per block);
  - rerun the T3 gates.
