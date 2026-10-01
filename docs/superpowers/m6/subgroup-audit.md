# Subgroup-uniformity audit (M6, GTX 1660 Super wrong-tail frames)

Base: master 4fd0090. Line numbers are master's. Scope: every subgroup builtin in
`crates/gzc-gpu/src/shaders/*.wgsl`. Only three files have any: `k1_chains_sg.wgsl`, `k1_sort_sg.wgsl`
and `k3_coop.wgsl`. No quad ops, no `subgroupElect`, no Inclusive scans. That makes 31 calls on 29 lines.

## Criterion

A call is **uniform** when every active lane of its subgroup reaches the same dynamic instance under
the same conditions. Each enclosing `if`/loop condition must be equal in all lanes, and no earlier
`return`/`break`/`continue` taken by only some lanes may have left lanes elsewhere. That is the most
core SPIR-V guarantees: control flow that is uniform at a structured header reconverges at the
merge block, provided every lane leaves through that merge. Anything beyond it needs
`VK_KHR_shader_maximal_reconvergence`, which we do not get (see below).

There is a second class of hazard, **"operand from a divergent branch"**. The op itself is uniform,
but its operand was computed inside a lane-dependent `if` that closed just before it. Correctness
then rests on merge reconvergence right at that op. If a driver ever runs the op before lanes
leave the branch, the result is wrong. That is the brief's hypothesis.

**Key finding:** naga 30 lowers every runtime `a && b` / `a || b` into
`var t; if a { t = b } else { t = false }` (`naga/src/front/wgsl/lower/mod.rs`, `logical`). So each
per-lane `&&` / `||` that feeds a ballot or shuffle is a lane-divergent branch right before that
collective. The WGSL source does not show it, but naga's output and the SPIR-V do. The non-short-circuit
`&` / `|` on `bool` become `OpLogicalAnd` / `OpLogicalOr` with no branch, and give the same value
whenever the right operand has no side effects and is safe to evaluate.

The WGSL static uniformity analysis (workgroup scope, with private/read_write values treated as
non-uniform) would reject most of `k3_coop` by construction, because the whole parse is data-driven.
naga does not enforce it for subgroup ops. The criterion here is dynamic subgroup uniformity.

## Table

| file:line | op | uniform? | why |
|---|---|---|---|
| k1_chains_sg.wgsl:112 | subgroupBallot (publish_bit, x16 per tile) | yes, **operand fixed** | Called at the top of the tile loop. The task loop breaks on `t >= n_tasks`, which depends only on the workgroup, and the tile loop bound is constant. The preceding `if (cl == i) bal[..] = m` stores only. The operand `h` came from three lane-divergent branches: `if (live) { h = .. }` (184), `chain_hash_words`' `if (sh != 0u)` (59, `p & 3` per lane) and `load_words`' `if (p < HASHED_POSITIONS)` (97). **Fix:** the hash is now computed one tile ahead, just before the tile's closing `storageBarrier`; the first tile's is computed before the loop and gets its own `workgroupBarrier`. So the ballots follow a barrier, not the branches. A branch-free rewrite (select or split shifts, clamped loads) was tried first and cost K1 6–16 % (register/scheduling sensitivity). The pipelined form is 11 % *faster* (14.85 → 13.18 ms/batch on lvl9seg) because it hides the hash latency. A dead lane's `h` is masked out by the live ballot. |
| k1_chains_sg.wgsl:121 | subgroupBallot(live) | yes, fixed with 112 | Same position. It now follows the barrier. |
| k1_chains_sg.wgsl:197 | subgroupAny(chunk_first) | yes, **operand fixed** | After `workgroupBarrier`. The operand was `live && lower == 0u`, a naga `if` on per-lane `live`. It is now `live & (lower == 0u)`. The `if (chunk_first) old = atomicLoad(..)` (192) does not feed it. |
| k1_sort_sg.wgsl:17 | subgroupShuffle x2 (word_at, x3 per key) | yes | The histogram and ranking loops have constant trip counts (`CHUNKS`, 4). `w`/`wn` come from `chunk_words`, which is branch-free (min-clamped). |
| k1_sort_sg.wgsl:45 | subgroupBallot (match_bit, x KEY_BITS) | yes | Inside `if (!subgroupAll(..))`, and a subgroup result is uniform. The `if (KEY_BITS > n)` guards are constants. |
| k1_sort_sg.wgsl:93 | subgroupBroadcastFirst | yes | Top of the histogram tile. The previous tile's `cnt_add` branches only perform atomics. |
| k1_sort_sg.wgsl:94 | subgroupBallot(live) | yes | Same as 93. |
| k1_sort_sg.wgsl:95 | subgroupAll(k == k0 \|\| !live) | yes, **operand fixed** | naga turned `\|\|` into an `if` on per-lane `k == k0`. It is now `(k == k0) \| !live`. |
| k1_sort_sg.wgsl:112, 114 | subgroupExclusiveAdd, subgroupAdd (CNT16) | yes | The scan loop has a constant trip count (`NWORDS / 32`) and no lane branches. `CNT16` is a constant. It comes after a barrier. |
| k1_sort_sg.wgsl:116, 117 | subgroupExclusiveAdd, subgroupAdd | yes | Same as 112. |
| k1_sort_sg.wgsl:132 | subgroupBallot(live) | yes | Top of the ranking tile, after the previous tile's barrier. |
| k1_sort_sg.wgsl:133 | subgroupBroadcastFirst | yes | Same as 132. |
| k1_sort_sg.wgsl:135 | subgroupAll(k == k0 \|\| !live) | yes, **operand fixed** | Same `\|\|` lowering as 95. Now `\|`. |
| k1_sort_sg.wgsl:139 | subgroupShuffle(old, leader) | only by merge reconvergence, **fixed (removed)** | Not *inside* the leader-only `if` as the brief said, but right after it. Its operand `old` is written inside `if (live && lane == leader)` (itself two nested naga `if`s). This is the brief's hazard: a shuffle run by lanes that are not reconverged reads a stale `old`. Now the leader writes its first slot to the workgroup array `first_slot[parity*32 + lane]`, the tile's existing `workgroupBarrier` moves to sit between that write and the read, and the lanes read `first_slot[leader]`. This uses barrier semantics only, with no subgroup op and no extra barrier. The buffer is double-buffered by tile parity, so the next tile's leaders cannot overwrite a slot before every lane has read it. The barrier still orders the tiles' `cnt_add`s. |
| k3_coop.wgsl:77 | subgroupMin (coop_match_len) | yes | `mode`, `max`, `n` and `x0` (same address in every lane) are uniform. The early returns at 61 and 80 are taken by all lanes, because `m` is a subgroup result. The operand is a `select`. |
| k3_coop.wgsl:134 | subgroupBallot(!ok) (catch_up) | yes, **operand fixed** | The loop exits on `cnt` (a ballot), so it is uniform. `bound_ok = a && b && c` and `ok = bound_ok && (load == load)` were nested per-lane naga `if`s, with loads inside, feeding the ballot. They now use `&`. The loads were already clamped by `select`, so they are safe for every lane. |
| k3_coop.wgsl:180 | subgroupShuffle x2 (win_at) | yes, **operand fixed** | `i` is uniform (private state that is identical in all lanes). The preceding `if (..) win_fill` has a uniform condition, but inside `win_fill` the expressions `usable = off1 != 0u && off1 <= rp` and `usable && (load == load)` were per-lane `if`s feeding `win_rep4`. They now use `&`. |
| k3_coop.wgsl:213 | subgroupBallot(hit) (lazy scan) | yes, **operand fixed** | `valid`, `usable`, `rep4` and `hit` were all `&&`/`\|\|` chains on per-lane values, which became per-lane `if`s, some with loads. They now use `&`/`\|`. |
| k3_coop.wgsl:214 | subgroupBallot(!valid) | yes, fixed with 213 | Same as 213. |
| k3_coop.wgsl:225, 226 | subgroupShuffle(bw, h), (rep4, h) | yes, fixed with 213 | `h` comes from a ballot. The `continue` at 217 is uniform. The operand `rep4` is now branch-free. |
| k3_coop.wgsl:358 | subgroupBallot(hit) (greedy scan) | yes, **operand fixed** | `valid`, `usable`, `rep_ok` and `hit` were `&&`/`\|\|` chains. They now use `&`/`\|`. |
| k3_coop.wgsl:360 | subgroupBallot(!valid) | yes, fixed with 358 | Inside `if (h == W)`, and `h` comes from a ballot. |
| k3_coop.wgsl:364, 365 | subgroupShuffle(bw, h), (rep_ok, h) | yes, fixed with 358 | Same as 358. |
| k3_coop.wgsl:416 | subgroupBallot(true) (layout guard) | **only by host invariant, fixed** | It came after `if (b >= n_blocks) { return; }` (404), which is subgroup-uniform only while the host keeps `BPW > 1` limited to exactly-W-lane subgroups. The guard is there to catch layouts the host did not expect, so it must not depend on that rule. Now there is no early return: `in_range` is folded into the guard and into both branches. |
| k3_coop.wgsl:420 | subgroupAll(lanes_ok) | same, **fixed** | Same as 416. Also, `lanes_ok` was a `&&` chain over per-lane `sid == li`, and is now `&`. It is now `subgroupAll(lanes_ok & in_range)`. The lane-0-only `return` at 428 has no subgroup op after it, and is gone anyway. |

**Totals:** 31 calls on 29 lines. None sat inside lane-divergent control flow. One (k1_sort_sg:139)
consumed a value from the leader-only branch just before it. One pair (k3_coop:416/420) came
after an early return that was uniform only by host convention. 13 more had operands built in
divergent branches, almost all because of naga's `&&`/`||` lowering. All of these are fixed. The
output is byte-identical (see Verification).

What remains are side-effect-only lane branches followed by a collective, with no data flowing
from the branch into it:
- k1_chains_sg `if (cl == i) bal[..] = m` between ballots.
- k1_sort_sg `cnt_add` branches and `if (live) rankw[..] = ..`.
- k3_coop `if (k == 0u) seqs[..] = ..`.

These rely only on core SPIR-V merge reconvergence. Removing them would mean redundant
atomics or stores.

## Maximal reconvergence in wgpu 30

- Neither wgpu-hal 30.0.1 nor naga 30.0.1 contains `maximal_reconvergence` or
  `MaximallyReconvergesKHR`. wgpu never requests the extension, and naga's SPIR-V backend never emits the
  execution mode or `OpExtension "SPV_KHR_maximal_reconvergence"`.
- A hal hook does exist. `wgpu_hal::vulkan::Adapter::open_with_callback(features, limits, hints,
  Some(cb))` lets the callback push `VK_KHR_shader_maximal_reconvergence` onto `args.extensions` and
  chain `vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR { shader_maximal_reconvergence: 1 }`
  (ash 0.38 has it) into `args.create_info`'s pNext. `wgpu::Adapter::create_device_from_hal` then
  wraps the result. The device feature alone changes nothing, though: the
  driver only applies maximal reconvergence to modules that declare the execution mode.
- The modules would have to go in as SPIR-V passthrough: `Features::PASSTHROUGH_SHADERS`, then
  `Device::create_shader_module_passthrough` with naga run by us (`naga::back::spv`). The words
  need patching to add `OpExtension "SPV_KHR_maximal_reconvergence"` and `OpExecutionMode %ep
  MaximallyReconvergesKHR` (6023). That costs wgpu's validation and its injected bounds checks
  (we would set naga's own policies), adds explicit pipeline layouts, and makes the path
  Vulkan-only. It is feasible without changing wgpu, but the plumbing is significant.
- The local NVIDIA 610.57 driver advertises the extension (the string is in libnvidia-glcore). Turing
  on 595.97 is likely to support it but this is unverified. Recommendation: do not pursue it now. With the fixes above, the
  kernels no longer need it. Keep it as a diagnostic option if the 1660 still fails.

## GZC_EMULATE_SKEW

Before this change, skew stalled at entry, after barriers and `workgroupUniformLoad`, and *before* each
subgroup statement. A stall in front of a collective does not separate lanes that took different
branches. It now also stalls at the start of every `if` / `else` / `case` / `default` body in naga's
output, which includes the `if`s that naga makes from `&&`/`||`. Lanes that took a branch then fall up to
512 dependent ALU steps behind the ones that did not, right before the following collective. The
new unit test `skew_stalls_inside_divergent_branches` covers this.

## Verification (worktree, RTX 5090, driver 610.57)

- `cargo test --release -p gzc-core -p gzc-gpu -p gzc-bench`: all pass.
- gzc-gpu with `GZC_NO_SUBGROUPS=1`, `GZC_POISON=1` and `GZC_EMULATE_SKEW=1` (the new
  branch stalls included): 150/150 in each, no Xid.
- `gzc-bench gpu --synthetic --verify` over all 10 presets: OK.
- Clippy: no new warnings.

Skew note: the first attempt put a full `gzc_skew` in every branch body of every module. That ran
K2 and K3 dispatches past NVIDIA's preemption timeout (Xid 109 CTX SWITCH TIMEOUT), and the test
then waited forever in `vkWaitSemaphores`. Branch stalls now go only into modules that call subgroup
builtins, and use the lighter `gzc_skew_branch` (1 call in 4, at most 64 steps).

## Perf: interleaved A/B against master 4fd0090

Setup: `--input data/corpus --ext dds,nif --batch max --inflight 3`, 3 alternating pairs. The GPU
was shared with another session's long-running test.

| preset | ratios new/master | |
|---|---|---|
| lvl3 | 1.147, 1.149, 1.146 | K1 chains pipelining |
| lvl9seg | 1.053, 1.050, 1.048 | K1 chains pipelining |
| lvl9s12seg | 0.986, 0.981, 0.989 | k1_sort +0.35 ms/batch (+5.7 % of that kernel), from the workgroup-memory handoff |
| opt16 | 1.004, 1.020, 1.006 | |
