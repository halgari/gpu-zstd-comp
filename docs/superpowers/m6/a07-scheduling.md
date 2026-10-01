# a07-scheduling: work scheduling, load balancing, pass fusion and overlap for opt16

Date: 2026-09-30. Research only: no repository file was changed. Scratch is in `/tmp/claude-1000/m6/a07-scheduling/`
(the worktree has been removed; the scripts, kernels and dumps listed in §7 remain).

**Contention.** 3 to 5 other agents' GPU processes and a load average of 26–140 were present throughout. All
absolute times are **under contention**. Every comparison was run interleaved (A B C D, repeated 2–3 rounds), and
`ab.sh` logs the load average and the GPU process count for each round. Rounds where a pass jumped by 1.5–2× (another
agent's kernel) are ignored. The tables below give the best clean round.

## 1. Headline findings

1. **On this NVIDIA Vulkan driver, multi-wave K3opt dispatches run as synchronous waves.** The hardware does not backfill.
   - Measurement: 4000 blocks with a per-workgroup clock (§6). The first 3400 workgroups start at t = 0. All 600
     remaining workgroups start at **27.21 ms**, the moment the *last* first-wave block ends. By then 2597 first-wave
     blocks had already finished before 20 ms, and SM 0 had been completely empty since 25.1 ms.
   - The 4-blocks/SM emulation shows the same pattern: waves start at 0, 18.8, 38.0, 56.7 and 75.2 ms.
   - Controls:
     - CUDA kernels on the same GPU backfill dynamically at 22, 92 and 121 registers (second-wave starts 0.2–2 ms).
     - A WGSL spin kernel at 37 registers backfills dynamically.
     - The same spin kernel with K3's code kept alive (93 registers) is wave-synchronous again.
   - So the cause is in the Vulkan driver path and correlates with these heavy kernels. I did not isolate it.
   - This is why "a partial second wave costs 20–24 %" (in fact more: 3586 blocks give 51 against 34.9 µs/block).
     Each wave pays its slowest block. It is probably also why speed-1 S6 saw no dispatch overlap on the single queue.
2. **A persistent-workgroup K3opt fixes it in software, byte-identical, with no occupancy query.**
   - Each workgroup loops: `b = perm[atomicAdd(ctr, 1)]`, then prologue, DP and epilogue for block b.
   - Dispatched with grid = n, as today. Non-resident workgroups start after the queue is drained and exit at once.
     Grid = n measured the same as grid = capacity.
   - With heavy-first order:
     - **opt16 at 4000 blocks: 47.3 → 31.1 µs/block (−34 %)**;
     - **at 3586 (`--batch max`): 51.1 → 31.1 (−39 %)**;
     - both are **11 % below today's best one-wave figure (b2900, 34.8)**.
3. **The block cost is predictable from K2opt's output.**
   - The proxy is the number of positions whose longest candidate has length 3..32, called `mid`. Its Spearman
     correlation with the measured block time is 0.925 at pass 0 and 0.89 at the final pass.
   - The study's proxy (Σ min(len,32) − 2) correlates at only 0.48, and its Pearson correlation is *negative* (−0.43).
     Long matches (> 32) are encoded immediately and are cheap.
   - Block times correlate 0.97–0.99 across passes, so one order serves all 4 passes.
   - 64 linear buckets (`min(mid >> 10, 63)`) are as good as the exact order in the model.
4. **On an 8 GB card (24 SMs × 20 slots, about 480 blocks per wave), the model gives persistent + heavy-first −32 %
   for K3opt at b2900 and −27 % at b3586**, within 2 % of the perfect-packing floor.
   - If the 4060 driver backfills dynamically (unknown; same driver family), only the LPT part counts: about −7 %.
   - Projection for opt16 on a 4060: about 0.22 → about 0.29 GB/s (K3 235 → 160 µs/block; other kernels about 65 µs
     from the perf study). It helps, but it is not line rate.

## 2. Measured cost distribution (RTX 5090, opt16, 2900 corpus blocks at 64 KiB, one wave)

Tool: device-clock stamps (VK_KHR_shader_clock) and SM id (VK_NV_shader_sm_builtins), patched into naga's SPIR-V of the
real kernel (§6). Timing overhead is ≤ 1 % (passes 8.17 / 8.18 / 8.16 / 9.44 µs/block clocked against 8.29 / 8.66 /
8.86 / 9.31 unclocked in the same session, under contention).

| pass | makespan ms | block latency under load (ms) p10 / p50 / p90 / p99 / max | slot utilisation | active < 90 % from | perfect-packing floor |
|---|---:|---|---:|---:|---:|
| 0 (L0) | 23.5 | 13.1 / 15.7 / 19.4 / 22.3 / 23.5 | 0.67 | 0.56 of makespan | 0.67× |
| 1 (L0) | 23.6 | 13.4 / 16.1 / 19.7 / 22.4 / 23.6 | 0.69 | 0.57 | 0.69× |
| 2 (L0) | 23.5 | 13.4 / 16.2 / 19.8 / 22.3 / 23.5 | 0.69 | 0.57 | 0.69× |
| 3 (L2) | 26.7 | 12.9 / 15.6 / 20.6 / 23.5 / 26.7 | 0.60 | 0.48 | 0.61× |

- **Heavy blocks come in runs.** The worst are neighbours from the same files: 1296/1297, 1500/1501 and 2688/2689. So
  the natural order is close to adversarial for wave-synchronous dispatch.
- **Per segment (lane), within a block** (per-lane counters):
  - trips per lane are balanced: max/mean in a block is p50 1.015;
  - relaxation steps per lane have max/mean 1.37 (p90 1.74–1.94);
  - so all 16 chains have the same length, and a heavy block is heavy because each trip costs more (relaxation and
    divergence), not because it has more trips;
  - the sum of lane relaxations correlates with block time at Spearman 0.87–0.93.
  - **Splitting or re-assigning segments does not shorten the critical chain.**
- **Uniform-block latency against residency** (ms, block replicated k per SM):

  | block | k = 1 | 2 | 4 | 6 | 10 | 14 | 20 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | typical (1563) | 10.6 | 11.9 | 12.7 | 13.4 | 14.4 | 15.9 | 22.4 |
  | heavy (1297) | 17.6 | 18.7 | 19.8 | 21.2 | 22.6 | 24.7 | 28.9 |
  | light (2602) | 8.8 | 9.3 | 10.0 | 10.5 | 11.1 | 12.4 | 15.1 |

  - SM throughput is almost flat from 14 to 20 warps per SM.
  - Under full load a heavy block is only 1.3× a typical one; alone it is 1.7×.
- **Model.** I used a per-SM processor-sharing model with f(k) from this table, inverted the clock dumps into
  per-block work, and checked it against the 4-blocks-per-SM emulation (§3):

  | | model | measured | difference |
  |---|---:|---:|---:|
  | wave-synchronous, natural order | 31.0 | 32.0 | −3 % |
  | wave-synchronous, heavy-first | 23.9 | 25.3 | −6 % |
  | dynamic (persistent), natural order | 21.3 | 22.6 | −6 % |
  | dynamic (persistent), heavy-first | 19.1 | 20.7 | −8 % |

  The model is consistently 3–8 % optimistic, but it preserves the ratios.

## 3. Prototype results (interleaved, µs/block for the 4 DP passes plus fix-up, RTX 5090, under contention)

The persistent kernel is `k3_persist_v2.wgsl`, at 93 / 92 / 92 registers. LPT order uses the `mid` proxy, with the
blocks physically reordered on the host.

| batch | base | base + LPT | persistent | **persistent + LPT** |
|---|---:|---:|---:|---:|
| 2900 (0.85 wave) | 34.8 | 34.8 | 34.9 | 34.2 (neutral) |
| 3586 (`--batch max`) | 51.1 | 41.6 | 37.7 | **31.1** |
| 4000 (1.18 waves) | 47.3 | 40.4 | 37.2 | **31.0–31.3** |

The emulated small GPU uses shared-memory ballast so that 4 blocks fit per SM: 680 slots, 2900 blocks, 4.3 waves.
Pass-0 times in µs/block:

| base | LPT | persistent | persistent + LPT |
|---:|---:|---:|---:|
| 32.0 | 25.3 | 22.6 | **20.7 (−35 %)** |

**Model projections** (K3 µs/block over 4 passes; ratio to the base kernel):

| GPU (SMs × slots), batch | base, wave-sync | + LPT | persistent | **persistent + LPT** | fused persistent + LPT | floor |
|---|---:|---:|---:|---:|---:|---:|
| 5090, b2900 | 33.6 | 1.00 | 1.00 | 1.00 | 0.99 | 0.66 |
| 5090, b3586 | 47.6 | 0.81 | 0.76 | **0.63** | 0.62 | 0.47 |
| 4060-class (24 × 20), b2900 | 235 | 0.79 | 0.73 | **0.68** | 0.68 | 0.67 |
| 4060-class, b3586 | 221 | 0.85 | 0.76 | **0.73** | 0.73 | 0.71 |
| 4060-class, b960 | 209 | 0.92 | 0.91 | **0.80** | 0.80 | 0.75 |
| 4060-class, b480 (1 wave) | 205 | 0.98 | 1.00 | 0.98 | 0.98 | 0.74 |

5090 end-to-end estimate at `--batch max` with persistent + LPT: K3 31.1 + 11.4 µs for the other kernels = 42.5 µs/block,
**about 1.50 GB/s against 1.43 today (b2900) and 1.09 today (`--batch max`)**.

Correctness:
- The persistent kernels (v1 and v2) are byte-identical to the oracle under `k3opt_passes_corpus`: 1000 corpus blocks,
  all 6 schedules with histograms, and the later-pass Buffer configs.
- Those runs used grid 37 for 256-block chunks, so each workgroup loops about 7 times.

## 4. Ideas evaluated

| # | idea | mechanism | expected speedup | ratio | exactness | effort | risk |
|---|---|---|---|---|---|---|---|
| 1 | **Persistent-workgroup K3opt** (queue of blocks) | a loop around prologue / DP / epilogue; `b = perm[atomicAdd]`; grid = n | measured −21 % at 4000 and −26 % at 3586 alone; neutral at one wave; emulated small GPU −29 % | none | byte-identical (only the order changes) | S | **registers:** the naive loop went from 92 to 138 / 114 / 107 registers (12 warps/SM). Recomputing `lane`/`pb` from `lid + (b >> 30)` inside the loop restores 93 / 92 / 92. Needs a `vkstats` gate. Unknown on AMD/Intel, but it is safe everywhere: no workgroup waits on another |
| 2 | **Heavy-first order from a K2opt proxy** | K2opt counts positions with max(lenA, lenB) in 3..32 per block (one subgroup reduction plus atomicAdd); a 1-workgroup counting sort into 64 buckets gives `perm`; the same order for all passes | with 1: measured −34 to −39 % (5090 ≥ 1.05 waves), model −32 % on a 4060 at b2900; alone (wave-sync): −15 to −19 %; alone with dynamic hardware: about −7 % | none | identical | S | low; proxy only Spearman 0.925 (exact order adds ≤ 1 % in the model) |
| 3 | Batch sizing | base kernel: clamp `--batch max` to one wave (b3400 gave +4 % in T5); with 1 + 2: larger is better up to VRAM and the 2 GiB `cands` binding (≤ 4096 blocks) | +4 % (base) / with 1 + 2, b3586 is 11 % faster per block than b2900 | none | identical | S | wgpu cannot query occupancy; 1 makes that unnecessary |
| 4 | Fused passes per block (one dispatch) | a workgroup runs all passes for its block, so tails overlap and the tables stay in shared memory | model ≤ 1–2 % over 1 + 2. T4 measured the runtime prologue mode at +3–4 % and the in-kernel loop at +0.7 %. After 1 + 2 a 5090 pass is bounded by its heaviest block under load (27 ms of 29.5); fusion leaves 4 × that chain | none | identical | L | **not worth it** |
| 5 | Segment work queue / finer units | rebalance segments across lanes | ≈ 0: chains are balanced (trip max/mean 1.015); cost per trip is the imbalance | – | – | M | dead end |
| 6 | Async compute: K1/K2 of n+1 alongside K3 of n | second queue (family 2) | measured with `a07_async`: concurrent/serial 0.95–1.42 (K1/K2 first), 1.00–1.42 (K3 first); no reliable overlap, matching speed-2 E3. Persistent grids would occupy the GPU even more | none | identical | M–L | dead end on NVIDIA Vulkan; AMD's hardware compute queues are untested |
| 7 | Smaller persistent grid (14–16 warps/SM) to cut contention | throughput is flat from 14 to 20 warps/SM | measured at 2900: grid 2720 neutral, 2380 +7 %, 1700 +17 % | – | – | – | dead end (except to leave room for 6, which does not overlap) |
| 8 | De-densify predicted-heavy blocks (fewer active lanes per warp) | cut the divergence on the heaviest chain (the study: 18.6 ms with 16 lanes against 5.6 ms with 1 lane, alone) | untested; only helps the 5090 one-wave / 1.2-wave case, where the heaviest block bounds a pass (p99 22.3 against max 23.5–27 ms): ≤ 10–15 % per pass; ≈ 0 on multi-wave cards | none | identical | M | moderate (new lane mapping in prologue / epilogue) |

## 5. Top-3 recommendation

1. **Persistent-workgroup K3opt plus an LPT `perm` from a K2opt `mid` counter (ideas 1 + 2 together).**
   - Measured −34 to −39 % for K3 at 3.6–4K blocks on the 5090, which opens `--batch max` (about 1.50 GB/s
     end to end).
   - Modelled −27 to −32 % on a 4060-class card. It is byte-identical, effort S–M.
   - Gate registers at ≤ 96 with `vkstats` (the loop-variant `lane` trick).
   - Add `perm` + counter (4 B/block + 16 B) to `vram_bytes`.
   - Reset the counter per pass, with `clear_buffer` or one counter per pass.
2. **Measure the dispatch behaviour on a real 8 GB card (4060, RX 7600, A750) before relying on the 4060 numbers.**
   - Run `a07_clock` at about 2–3 waves and check whether the second-wave start times are synchronous.
   - If the card backfills, idea 1 is worth only about 0 % there and the gain is LPT's 5–7 %.
3. **Use batch sizing as a fallback.**
   - Until 1 lands, clamp `--batch max` to one wave on the 5090 (b3400, +4 %).
   - Also add the LPT order to the base kernel (−15 to −19 % at > 1 wave).
   - Fusion (4), segment queues (5) and async compute (6) are not worth pursuing.

## 6. How it was measured

- **Clock harness** (scratch `k3opt.rs`, `GZC_CLOCK=1`):
  - naga 30 compiles the real composed module to SPIR-V with wgpu's Vulkan options (SPIR-V 1.6, Restrict / Unchecked
    bounds, unbounded loops);
  - `patch_clock.py` rewrites stores of a marker constant into `OpReadClockKHR` (device scope, ns) and `SMIDNV`;
  - the module is loaded with `create_shader_module_passthrough`, on a `RawDevice` with VK_KHR_shader_clock and
    VK_NV_shader_sm_builtins enabled;
  - stamps are taken at block start, after the prologue, after the DP, and at the end; per-lane trip and relaxation
    counters are taken at loop exit.
- **Harness** `a07_clock`: per-pass GPU timestamps, one submission per pass, median of 3.
  - Corpus blocks are every 34th (2900) or 25th (4000) block of 100 754, the `k3opt_timing` sample, cached with their
    CPU candidates.
  - Options: `GZC_PERM` (order), `GZC_PICK`/`GZC_COPIES` (uniform blocks), `GZC_PERSIST`, and `GZC_K3OPT_SRC`
    (kernel override, also in the clock path).
- **Low-occupancy emulation:** a 20 KB workgroup ballast gives exactly 4 blocks/SM (680 slots, from the clock's peak
  concurrency). Durations were 1.21× the k = 1 work, which matches f(4).
- **Model:** `sim.py` and `sim4060.py` (processor sharing per SM; wave-synchronous mode and dynamic mode).
- **Wave-sync controls:** `cuda/bf.cu` and `bf2.cu` (CUDA, 22/92/121 registers); `k3_spin.wgsl` (37 registers, dynamic);
  `k3_spin2.wgsl` (93 registers, wave-synchronous).
- **Async:** `a07_async` (MultiQueue with the async-compute family; K3opt on main, K1 + K2opt of another set on family 2).

## 7. Scratch files (`/tmp/claude-1000/m6/a07-scheduling/`)

- **Kernels:**
  - `k3_persist_v2.wgsl` (the one to port);
  - `k3_persist.wgsl` (naive, 138 registers);
  - `k3_persist_wt.wgsl` (tables in shared memory: no register change, so the tables were not the cause);
  - `k3_ballast_*.wgsl` and `k3_persist_ballast.wgsl`;
  - `k3_spin*.wgsl`.
- **Host:** `host.diff` (gzc-gpu `Cargo.toml`, `context.rs`, `multiqueue.rs`, `k3opt.rs`, `tests/k3opt.rs`), together
  with `patch_clock.py`, `vkstats` (16 bindings) and `regs.sh`.
- **Data:** `clk*/` (clock dumps), `proxies.py`, `perm_*.bin`, `mid*.npy`, `ab.sh` (the interleaved runner with load
  and GPU-process logging).
