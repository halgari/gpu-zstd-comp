# a01-divergence: branch and divergence reduction inside K3opt

Agent a01-divergence, 2026-09-30. Research only: no repository file changed. Prototypes, harness and data are in
`/tmp/claude-1000/m6/a01-divergence/`. The worktree is removed; its host-side hooks are saved as
`host-scratch.diff` and `probe.rs`. Base commit: 6fe7fa5. `k3_opt.wgsl` is unchanged at d5ebb1f.

## 1. Result in one table

These are measured kernels, all byte-identical to today's oracle (`Engine::Ring`; see §6). The table gives times as
ratios to the unmodified kernel, measured in the same session with the variants interleaved. RTX 5090, 2900 corpus
blocks, 64 KiB.

| variant (cumulative) | opt16 4 passes, sum | opt14 2 passes | L2 Buffer pass | L0 Buffer pass |
|---|---:|---:|---:|---:|
| base | 1.000 | 1.000 | 1.000 | 1.000 |
| D: register diet only | 0.995 | – | 1.005 | 0.997 |
| D + P2: merged, bounded rep extension | 0.928 | – | – | – |
| D + P2 + P1: rep-length memo | 0.895 | – | 0.878 | 0.869 |
| D + P2 + P1 + P3: grid relaxation | 0.891 | – | 0.857 | 0.880 |
| **D + P1 + P2 + P3 + P4 + P5: flat phase 2 = "p5"** | **0.880** | **0.892** | 0.852 | 0.870 |

Final opt16 per-pass figures, base → p5, in µs/block under contention:

| pass | base | p5 | ratio |
|---|---:|---:|---:|
| 0 | 8.23 | 7.05 | 0.857 |
| 1 | 8.21 | 7.30 | 0.889 |
| 2 | 8.24 | 7.32 | 0.888 |
| L2 | 9.24 | 8.14 | 0.881 |
| fix-up | 0.27 | 0.27 | 1.000 |

- At 4000 blocks, past one wave, p5 is 0.784 (L2) and 0.831 (L0). The heavy-tail chains shrink, so a partial
  second wave gains more.
- Projection, not measured: if the other kernels are unchanged, K3 on opt16 goes from 33.3 to about 29.3 µs/block.
  End to end that is about 1426 → 1560 MB/s on the 5090 (+10 %).
- Ratio impact: none. The output is byte-identical, so no new oracle and no new preset are needed.

**Main finding: the remaining divergence is no longer a control-flow problem.**
- On the same segment workload, "16 different segments per warp" vs "16 copies of one segment per warp" isolates
  control-flow divergence:
  - base: 1.37× at L2 and 1.31× at L0;
  - p5: 1.15× at L2 and 1.09× at L0.
- The rest of the perf study's "2.5–3.3× vs one lane" is the cost of a warp whose 16 lanes each touch their own cache
  lines, with 16× the work: replicated vs lone lane is still 2.2–2.7×.
- So pure control-flow work has at most about 9–15 % left. Most of it is in the relaxation (§4).

## 2. Method

- **Harness.** A copy of the repo had three scratch-only host hooks:
  - a kernel-source override (`SRC_OVERRIDE` / `GZC_K3OPT_SRC`);
  - a WGSL dump;
  - a readable scratch buffer.
- **`tests/probe.rs` modes:**
  - `check`: GPU vs `dp_pass_with(.., Engine::Ring)` at L2 and L0 with Buffer prices.
  - `checkpasses`: the 4-pass opt16 vs `opt::passes`, and vs the base GPU.
  - `time` / `passes`: `time_pass` / `time_passes`, variants interleaved per round, median of rounds, each round a
    median of 5.
  - `stats`: per-lane counters written past the batch's scratch.
  - `divsplit`: the mixed-vs-replicated experiment, described below.
- **Branch counters.** These use subgroup ops in a scratch kernel only. For every branch body or loop iteration:
  - "warp executions" E: the body runs for the warp when any lane takes it;
  - the lane count L;
  - Σ over trips of the per-trip max (`subgroupMax`).
- **Registers.** `vkstats` with naga 30 SPIR-V, using wgpu's Vulkan options: index Restrict, buffer Unchecked,
  robustness2, no loop bounding. It was run on every kernel of the 4 passes.
- **Corpus sample.**
  - Timing used the `corpus_blocks(2900)` sample: every 34th block of 100 754.
  - Exactness used that sample, plus a second sample of 6000 blocks taken as `corpus_blocks(7100)` minus the first
    1100, plus the repo's full K3opt gate suite including the ignored corpus tests (4000 blocks with frames, all
    pass configurations).
- **Contention.**
  - Up to 11 agents shared the GPU. Every timed set logged `uptime` and `nvidia-smi` before and after.
  - Sets where utilisation was high or rounds disagreed were re-run. One p3/base round set and one divsplit L2 set
    were discarded.
  - Absolute µs values are "under contention"; only the ratios matter.
- **divsplit.** Each 4 KiB segment's candidates are computed alone (zero padded), and two block sets are built:
  - "mixed": the real block, with segment-local candidates;
  - "rep": 16 copies of one of its segments (k = i mod 16).
  The same kernels run on both sets.
  - mixed / rep isolates control divergence, since the memory scatter is the same in both.
  - rep / lone lane gives the memory and occupancy side.

## 3. Branch map of K3opt (base kernel, L2 Buffer, mean per block, 2900 blocks)

E is the number of warp executions per block. "lanes/E" is the number of active lanes doing useful work per
execution, out of about 15.2. The heavy column is the heaviest 5 % of blocks; those blocks set the dispatch time.

| place | E | lanes/E | heavy 5 % E | remark |
|---|---:|---:|---:|---|
| trip (loop) | 4385 | 15.2 | 4336 | lane trips balanced: max/mean = 1.05 |
| series start (probe trip) | 4321 | 7.8 | – | body tiny, union negligible |
| series continue (lit-ext part) | 4335 | 7.7 | – | executed every trip |
| lit-ext taken (`price <= n.price`) | 4293 | 5.6 | – | the 4 stores run almost every trip anyway |
| search (`get_all_matches`) | 4381 | 14.0 | – | common path, little divergence |
| rep probe passes the 3-byte gate | 2978 | 2.5 | 11 738 | **divergent** |
| `match_len` word iterations | 4084 | 2.0 | 21 002 | **divergent, the heavy-block killer** (bounded part ≤ 32 B: Σt max 3178; long part only 573) |
| relaxation record iterations | 6621 | 4.1 | 9067 | **divergent** |
| relaxation steps (×4 lengths) | 8401 | 3.6 | 19 121 | **divergent** (Σt max 7718, lane mean 1868) |
| immediate encoding | 11 | 1.2 | 177 | rare |
| cap-64 candidate extension | 5 | 1.3 | 93 | rare |
| finish / `end_series` | 3058 | 1.8 | 2411 | divergent but small body |
| `emit_series` iterations (phase 2) | 470 | 8.0 | 1551 | 2× union over the lane mean |
| L2 "match + 1 literal" check | 0 with block-init prices; only in the final pass | – | – | |

L0 has the same structure:
- Relaxation steps fall to 7142 (lanes/E 3.2) because of the early abort.
- Searches fall to 12.8 lanes/E because of the `ld_price(cur+1) <= price + 128` skip.

The L0/L2 differences (abort chain, skip test, match+1 path) are all small predicated bodies. They only matter for
registers: the hist_out pass-0 kernel is the largest (§5).

## 4. Ideas, per place

**P2: merged, bounded rep extension.** Measured; S effort.
- **Mechanism.** The three rep probes each ran their own `match_len` loop, and each loop was a separate divergent
  union. Now:
  - all three gates are computed first;
  - the probes that pass extend in one lockstep loop up to RB = 36 bytes (> SUFF), loading the p-side word once
    per step;
  - lengths are packed into one register;
  - zstd's sequential probe order and early exits are then applied to the capped lengths with selects;
  - only the one probe that ends the search (a capped length means > SUFF) is extended exactly, from byte 36, at a
    single call site.
- **Exact because** a capped length ≥ 33 always ends the search at that probe, and the lengths below the cap are
  exact.
- **Gain.** opt16 sum 0.928 together with the register diet (passes 0.89 / 0.92 / 0.93 / 0.96).
- **Risk.** Low, but only with the diet: the unpacked first version was 1.5× slower because of the register wall
  (§5).

**P1: rep-length memo, "work you already did".** Measured; S effort.
- **Mechanism.** For the same offset, the capped common prefix at p + d is exactly L − d whenever L ≥ d + 3. That
  holds both when L < lim and when L = lim, since lim also falls by d.
- Each lane keeps its last search's three (offset, exact length) pairs and their position: 4 registers.
- A hit gives the length with no extension; an entry is stored only when exact.
- 68 % (L2) and 60 % (L0) of the gated rep probes hit.
- Lockstep extension iterations drop from 4084 to 1168 (heavy blocks: 21 002 to 5362).
- **Gain.** Measured on top of P2 + diet: L2 pass 0.900 → 0.878 (vs base); opt16 sum 0.928 → 0.895.
- **Risk.** Low. The positions a lane searches strictly increase, and a memo entry is used only for a valid offset.

**P3: relaxation as one length grid ("relax a fixed 4-wide grid").** Measured; M effort.
- **Mechanism.** The records × steps double loop becomes one loop over the match length, from the longest down, 4
  lengths per step.
- A step spans at most two records (the current one and the next lower one, both payloads in registers) and stops
  at the lower record's bottom.
- Targets are disjoint, so every compare is still against the pre-relaxation ring.
- The optLevel-0 per-record abort uses two flags plus a skip to the next record's top.
- **Effect.** Σt max steps 7718 → 5918 at L2 (heavy blocks 19 121 → 13 642), and the record loop is gone.
- **Gain.** Mostly at L2: L2 pass 0.878 → 0.857. At L0 it is neutral to slightly worse (0.869 → 0.880), because the
  abort already shortened records. Latency-bound runs (1450 blocks): L2 −7 % more than P1.
- **Risk.** Medium: an L0 abort bug would be subtle. It passes every gate and both samples.
- **Option.** It could be enabled at L2 only, since LEVEL is a compile-time constant.

**P5: flat phase 2.** Measured; S effort.
- **Mechanism.** The series × trace double loop becomes one loop that emits one sequence per iteration, reading a
  series header when the previous series is done. Reads and writes keep the same order, which keeps the HIST_OUT
  in-place log safe.
- Warp iterations go from Σ over series of the per-series max to the max over lanes of `n_seq`.
- **Gain.** opt16 sum 0.891 → 0.880, about −1.5 % per pass. Phase 2 plus its histogram was about 5.5 % of pass 0
  by ablation.

**D: register diet.** Measured; S effort; the enabler.
- **Mechanism.**
  - The 5 + 5 record arrays are packed into 5 words: `ob | len << 17`.
  - The 6 private lane constants (base, cbase, tbase, wbase, sbase, lbase) are derived from (b, k).
  - `pb` folds to 0 when `BPW == 1`.
- **Effect.** −4 / −2 / −1 registers on the base pass kernels, and 0 time on its own.
- Without it, P2 and P3 cross the wall in the hist_out kernels:
  - p1p pass 0: 98 regs, 11.6 vs 8.3 µs;
  - p3q pass 0: 99 regs.

**P4: `end_series(n)` instead of `end_series(ld(last_pos))`.** Measured; S effort.
- Every finish ends at this trip's own node, so the 4-load reload is dead.
- Gain: 0.99–1.00 (noise). It is kept in p5 because it is free.

**Estimated only, not built:**
- **FIFO-deferred ("smoothed") relaxation.** Each target has a deadline: target c + m is needed only before Part 1
  of trip c + m − 1 (the L0 skip and L2 match+1 read `cur + 1`). So per-source work could be spread over 2+ trips in
  source order, which preserves tie order.
  - The ablation "relaxation limited to one step" (a different parse, so this is an upper bound) saves 17 % at L2
    and 7.5 % at L0. Smoothing would recover maybe a third of that: about 3–5 %.
  - It needs about 6+ registers of queue state. Not worth it while at the register wall.
- **Lazy rep payload.** A relaxation stores only price and `mlen | ob`; the reps are materialised at the visit from
  node `cur − mlen`, which is final then. That cuts relaxation stores from 4 to 2 per target, but adds a dependent
  load to the trip chain. Estimated 0–5 %, uncertain.
- **Predicated finish and lit-ext stores, and the probe-vs-series union.** Bodies are tiny (§3). Estimated < 1 %.

**Rejected after measurement or analysis:**
- **Word-wise literal histogram** (4 bytes per iteration): 0 gain on top of P5.
- **Chunked / budgeted long extension or relaxation** ("work queue in a lane"): with SIMT union semantics, the
  warp still runs K iterations in every trip where any lane has pending work, so the total equals the unchunked
  total. It only gains when two lanes' chunks happen to overlap, which is small at 1.8 lanes/E.
- **A fixed 8×4 grid** (always 32 lengths): it would run 8 steps per trip against today's 1.35. Much worse.
- **An 8-wide grid step:** not tried, since there is no register room. The perf study's ×8 measured 0 at 110 regs.

## 5. Register wall (a hard gate for any K3 change)

- With 2900 blocks per wave, a kernel at ≤ 96 registers (vkstats count) runs in one wave. At 97 or more it spills
  into a second wave and costs +40–50 % (seen 4 times: p2, p3 at L2, p1p pass 0, p3q passes 0–1).
- The highest kernel is pass 0 (BlockInit + HIST_OUT): base 88, p5 95, so p5 has one register of headroom.
- `vkstats` must be checked on all three pass kernels (pass 0, passes 1–2, final), not just the L2 Buffer one.
- A future version of P1–P3 should keep the diet, or find more savings: for example the memo's 3 entries as 24-bit
  `(off, len ≤ 255)` saves one register.

## 6. Exactness evidence

- `check` (oracle, L2 and L0, Buffer prices): 0 differences on 2900 blocks and on a second 6000-block sample, for
  P2, P1, P3, P4, P5 and the intermediate variants.
- `checkpasses` (4-pass opt16 vs `opt::passes`, and vs the base GPU): 0 differences on 2900 and 6000 blocks.
- The repo's K3opt gate suite with p5 injected via `GZC_K3OPT_SRC` passes all 7 tests, including the ignored corpus
  ones. These cover opt cases and synthetic data, the private-ring fallback, wg8/32/64, BlockInit / Prior / Hist
  prices, and frames via K4/K5.
- p5 uses no subgroup operations, so the non-subgroup path is unchanged.

## 7. Dead ends checked

- P2 unpacked: 99 regs → 2 waves, 1.5× slower. At 1450 blocks it was −3 % (L2) and −14 % (L0).
- P3 with the next-record payload carried before the diet: 97 regs at L2 → 1.24× slower.
- P4 alone: noise.
- Word-wise `hist_lits`: 0.
- Chunked extension: no gain by union argument.
- Cooperative and subgroup relaxation was already rejected by the perf study at 1.7×. With control divergence now at
  ≤ 15 %, it could not pay back cross-lane costs anyway.

## 8. Top-3 recommendation

1. **Adopt the register diet + P2 + P1 together** (S, low risk): −10.5 % on the opt16 K3 sum and about −11 % on
   opt14. All of it is in `get_all_matches` and the private-variable declarations. Gate it with `vkstats` ≤ 96 on
   all pass kernels.
2. **Add P5 (flat phase 2) and P4** (S, low risk): another −1.5 %, giving −12 % in total (0.880).
3. **P3 grid relaxation, only for the L2 final pass** (M, medium risk, since the L0 abort logic is subtle):
   −1.5 to −2 % on the final pass, more in multi-wave and latency-bound runs. Then stop on control-flow work. About
   9–15 % of control divergence is left, mostly the relaxation union (§1). Further K3 gains must come from the
   memory side (per-lane cache-line scatter, 2.2–2.7×) and from the pass count, not from branches.

## 9. Reproduction

- Directory: `/tmp/claude-1000/m6/a01-divergence/`.
- To rebuild the harness, `git worktree add --detach wt master`, then apply `host-scratch.diff` and copy `probe.rs`
  into `crates/gzc-gpu/tests/`.
- `t.sh MODE variants [ENV..]` runs a mode; the modes are `check | checkpasses | time | passes | stats | divsplit`.
- `pstats.sh v.wgsl` gives registers per pass kernel; `stats.sh` gives the single-config count.
- Kernels:
  - `base.wgsl`;
  - `based.wgsl` (diet);
  - `p2d.wgsl`, `p1d.wgsl`, `p3d.wgsl`;
  - `p5.wgsl` (final), with its diff in `p5-vs-base.diff`;
  - `stats.wgsl` / `stats3.wgsl` (counters);
  - `abl_*.wgsl` (ablations, not exact);
  - `lone_*.wgsl` (one lane per block).
- `an.py` analyses the counters.
