# Speed phase: make GPU lvl9 fast on 8 GB-class GPUs — Experiment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Each task is an
> *experiment*: implement, prove byte-identical output, measure against the baseline, keep it if it is
> faster, otherwise revert it and record why.

**Goal:** raise GPU `lvl9` throughput as far as possible, with a design that scales down to ~8 GB gaming
GPUs (RTX 4060 / RX 7600 class). Ratio and output stay exactly what they are today.

**Inputs:** idea lists `.superpowers/speed/ideas-fable.md` and `ideas-sonnet.md` (read the relevant
section before each task), context `.superpowers/speed/context.md`, and the M4 spec and results
(`docs/superpowers/specs/2026-09-29-m4-lazy-lvl9-design.md`, `docs/results/2026-09-29-m4.md`).

## Global constraints (every task)

- GPU frames stay **byte-identical** to the CPU oracle for every preset (lvl3, rung1, rung2, lvl9). The CPU
  oracle (`gzc-core`) does not change in this phase. The lvl3 anchor test must pass. Every existing
  differential test must pass at both block sizes (default and
  `--no-default-features --features gzc-core/block-16k,gzc-gpu/block-16k,gzc-bench/block-16k`).
- VRAM budget 6144 MiB. `vram_bytes` must equal the real allocations, and `vram_matches_params` must keep
  passing.
- WGSL may use `enable subgroups;` only behind the wgpu `SUBGROUP` feature, and only with results that
  **do not depend on the subgroup size**, which ranges 8–64 across vendors. Either keep a non-subgroup
  fallback path, or document a hard requirement plus a clear error. Workgroup storage must stay within
  16 KiB unless a raised limit is requested *and* is ≤ 32 KiB, which 4060/7600-class devices support.
- No `shader-int64`. u32 only.
- **Benchmark protocol:**
  - Command: `gzc-bench gpu --input /home/tbaldrid/oss/gpu-zstd-comp/data/corpus --ext dds,nif --preset lvl9 --batch max --inflight 3 --out out/speed-<task>-<n>`, run **3 times**, reporting the median end-to-end MB/s and per-kernel ms/batch.
  - Also run one `--verify` of lvl9 per task, and one median-of-3 of `--preset rung1`.
  - Foreground, long timeouts. Other VMs may be running, so report `uptime` alongside the numbers.
  - Record the baseline (task S0) and each task's numbers in `docs/results/speed-log.md`, one section per
    task: what changed, before/after table, kept or reverted, and why.
- An experiment that does not improve lvl9 end-to-end or kernel-sum by more than noise (≈ 3 %) is **reverted**
  unless it enables a later task. Record it either way.
- Target-hardware reasoning: every task reports whether its gain should carry over to a 24–34 SM, ~280 GB/s
  card, following the scaling model in ideas-fable.md §0.

## Tasks (in order; S1 and S2 may run in parallel because they touch disjoint kernels)

### S0 — Baseline and harness (controller)
Re-measure M3 lvl3 (`c43eee2`) against M4 lvl3 and lvl9 back to back on the current machine (final-review
item 3). Create `docs/results/speed-log.md` with the baseline table. Add the RUNG2 mm6 case and the
`submit_from_best` checks if the M4 fix wave hasn't already.

### S1 — K2 rework (ideas-fable §2 + §3; ideas-sonnet 1, 2, 3)
- Cap early-out: once `best_len == SEARCH_CAP`, later chain entries cannot win (q only decreases, and a
  tie goes to the larger q), so break out of the whole chain loop. Byte-identical.
- Load the p-window into registers once, before the chain loop. Wider compare steps.
- **Pack `best[]` to one u32**: `offset` (17 bits) and the *capped* length (≤ SEARCH_CAP, fits in 7–9 bits).
  Semantics stay exactly as today. **Do NOT compute uncapped lengths in K2**: every position of a flat run
  would extend to the block end, which is O(BLOCK_SIZE²) per block. That is the reason `search_cap`
  exists. Extension stays in K3 (and becomes cooperative in S3). The buffer halves, so update
  `vram_bytes` and batch sizes.
- Files: k2_best.wgsl, k3_parse.wgsl / k3_lazy.wgsl (read side of best), compressor.rs buffer sizes,
  submit_from_best harness.

### S2 — K1 rework (ideas-fable §1; ideas-sonnet 4, 6)
One subgroup (or a small workgroup) per block. Equal-hash groups are found with subgroup ballot/shuffle
instead of the bitonic sort, and there are no per-tile workgroup barriers. An epoch-tagged persistent
`head` table avoids the per-batch clear. `pred` must stay identical. The implementation must not depend
on subgroup size, with a fallback when subgroups are unavailable (keep the current kernel as the
fallback). Files: k1_chains.wgsl (+ new file), chains.rs, context.rs (feature request).

### S3 — Warp-cooperative K3 lazy2 (ideas-fable §4)
One subgroup per block, with parse state replicated on all lanes:
- Lanes prefill a lookahead window of `best[ip..ip+W]` and rep probes.
- The control loop runs uniformly using shuffles/broadcasts.
- `match_len` for catch-up and repeat checks is cooperative, W×4 bytes per step.
- Literal-run skipping evaluates many positions per step.

Output identical. The greedy K3 path gets the same treatment if it is cheap. Depends on S1 (best format).

### S4 — Literals gathered from seqs (ideas-fable §5)
K3 stops emitting literals. K5 and K4 compute literal positions from `(data, seqs)` with a prefix sum
over lit_len. This removes the `lits` buffer (VRAM) and the byte-wise `push_lits`. Depends on S3.

### S5 — Cross-batch overlap (ideas-fable §8)
Per-slot scratch (now affordable after S1/S4 memory cuts), with two batches' kernels recorded so that
K3/K4/K5 of batch i overlap K1/K2 of batch i+1. Verify the actual overlap with timestamps. Keep it only
if it helps within 6 GiB.

### S6 — Transfer path (ideas-fable §9)
GPU-side frame compaction (prefix sum over frame_len), so readback copies only the real bytes, plus
overlapped copies. Matters on PCIe ×8 cards. Measure the host-overhead share before and after.

### S7 — Speed results doc (controller)
`docs/results/2026-09-30-speed.md`, containing:
- the final numbers and per-kernel table;
- the cumulative gains from the log;
- a 4060-class projection using the §0 scaling model, labelled as a projection;
- remaining bottlenecks.

Then the whole-branch review.

## Added during execution

- **S5 dropped** (ledger ruling): S6 measured zero copy/compute or dispatch overlap on wgpu's single queue on this
  driver, and per-slot scratch would undo S6's VRAM savings.
- **S8 — K1/K2 second pass** (added after S3): per-kernel profile at b4959f8, lvl9 b2431 i3 = K3 32.7, K1 15.5,
  K2 13.0, K4 12.3, K5 4.7 ms/batch. Targets:
  - K2's 32-deep dependent chain walk (memory-level parallelism, e.g. interleaving the walks of several
    positions per thread, or prefetching pred[q] one step ahead);
  - K1's table-group count and tile loop.
  - Output is byte-identical.
- **S9 — K4 sequence-entropy pass** (added after S4): K4 ≈ 10 ms/batch, dominated by thread 0's sequential
  work (histogram → normalize → table builds → backward FSE bitstream).
  - Split the bitstream encode: first a sequential pass of FSE state transitions only (cheap table lookups),
    recording per-step (value, nbits) for states and extra bits.
  - Then a prefix sum over bit counts and a parallel bit placement (atomicOr), as K5 already does.
  - Also parallelize the normalize/cost loops where the oracle order allows it.
  - Output is byte-identical.
