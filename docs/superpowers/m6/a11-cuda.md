# a11-cuda: does a CUDA K3opt give us anything?

Date: 2026-09-30. Research only: no repository file was changed and nothing was committed. RTX 5090 (sm_120), driver
610.57, CUDA 13.3. The scratch worktree and build outputs have been removed. The small artifacts are in
`.superpowers/m6-research/artifacts/a11-cuda/`:
- `k3.cu`: the port, with every variant;
- `harness.diff`: the scratch Rust harness, plus three host hooks (trace buffer readable, `GZC_K3SRC`, `GZC_DUMP`);
- `ab.sh`, `pad.wgsl`, `ctx.cu`, `w2s.rs` (WGSL to SPIR-V for `vkstats`);
- `runs/*.txt`: every timing log, with `uptime` and `nvidia-smi` at the start and end.

## Short answer

- A CUDA K3opt is not faster than a WGSL K3opt with the same register diet. The one thing CUDA really adds is
  backfill of multi-wave dispatches, and a07's persistent kernel already gets that in WGSL.
- A straight port is 11–14 % slower than today's WGSL. nvcc leaves the private `m_ob[5]`/`m_len[5]` arrays in local
  memory (a 184 B stack frame); the Vulkan driver promotes them to registers.
- The tuned port is 2–4 % faster than today's WGSL at b2900. Its tuning is the same transformations a02 found
  (pb folded, records packed into registers, u16 tables).
- a02's portable WGSL (`v_pk16`) is just as fast in the same session: 0.94/0.97 against CUDA's 0.98/0.96. At b4080 it
  beats CUDA (0.69 vs 0.70–0.81), because CUDA reserves 1 KB of shared memory per 16-thread block and that caps its
  residency at 20 blocks/SM for the cheap pass.

## 1. The port, and how exactness was checked

`k3.cu` ports `main_opt` (PRICE_MODE 3: the price tables built from the previous pass's Hist) statement by
statement:
- at LEVEL 0 with HIST_OUT, the cheap pass, including `hist_epilogue`;
- at LEVEL 2 without HIST_OUT, the final pass;
- plus `main_fixup`.

Everything is fixed for opt16 at 64 KiB: 16 segments, wg16, SUFF 32. One template, `Cfg<...>`, switches the tuned
changes on one at a time. A C API (`k3c_*`) is linked into a scratch `#[ignore]` test (`a11_cuda`), so the CUDA and
Vulkan contexts run in one process on the same uploaded inputs.

**Inputs:**
- 2900 corpus blocks (every 34th of 100 754; 4080 for the batch-size runs);
- `find_cands` candidate words;
- the oracle's `opt::passes` histograms: Hist of pass 0 as the cheap pass's input, Hist of pass 2 as the final
  pass's input.

The harness asserts `Prices::from_hist(Hist::of_output(pass k)) == pass k+1 prices` for every block.

**Check:** both implementations start from zeroed trace/seqs/counts/scratch, and whole buffers are compared word by
word:
- cheap pass: the `prices` (output Hist), `seqs` and `trace` buffers;
- final pass + fix-up: `best`, `seqs`, `trace` and `counts`.

In the same run, the WGSL cheap-pass Hist was checked against `Hist::of_output(oracle pass 1)` (0 words differ),
and the WGSL final parse against the oracle's pass-3 output (0/2900 blocks differ).

**Result:** all 23 CUDA variants (2 passes each) have 0 differing words on 2900 blocks. The first port was exact,
which is good evidence that the WGSL kernel's semantics map 1:1 onto CUDA C++:
- `firstLeadingBit` maps to `31 - __clz`, which gives -1 for 0, as in WGSL;
- the WGSL shifts are already masked;
- naga's index clamps never trigger.

## 2. µs per block per pass (CUDA straight, CUDA tuned, WGSL), same blocks, interleaved

Method:
- In each round: the WGSL pass via `time_pass` (GPU timestamps), then each CUDA variant (`cudaEvent`).
- Inputs are restored before every run, outside the timed region.
- I report the median of the per-round ratios and the min/min ratio. Under contention the min is the more robust
  number.

**Batch 2900, the production one-wave case (`runs/final2900.txt`):** 15 rounds. This was the quietest run: load
2.4–4.0, and the only other GPU process was an idle 5.5 GB test holding memory.

| implementation | cheap L0 µs/blk | ratio | final L2 µs/blk | ratio |
|---|---:|---:|---:|---:|
| WGSL repo kernel | 8.23 | 1.000 | 9.27 | 1.000 |
| CUDA straight port (v0) | 9.34 | 1.137 (min 1.123) | 10.38 | 1.121 (1.104) |
| CUDA tuned (v6: pb0, packed records, u16 tables, `ld.global.nc`) | 8.11 | 0.985 (0.969) | 8.87 | 0.959 (0.953) |
| CUDA tuned + trimmed final-pass smem (v16) | 8.05 | 0.978 (0.971) | 8.89 | 0.961 (0.953) |
| WGSL a02 `v_pk16` (portable diet) | **7.75** | **0.941** (0.937) | 9.03 | 0.975 (0.973) |

The rounds are tight: for example, v6 cheap stays within 0.966–0.994 and `v_pk16` within 0.934–0.947.

**Batch 1360, the latency regime: one block per resident slot at about 8/SM, so the heaviest block's chain
dominates (`runs/lat1360.txt`).** 8 rounds, contended (load 5–9). Min/min ratios:

| implementation | cheap | final |
|---|---:|---:|
| CUDA straight | 0.988 | 1.020 |
| CUDA tuned v6 | 0.859 | 0.899 |
| WGSL `v_pk16` | **0.842** | 0.903 |

The serial per-lane chain is about 14 % shorter with the diet in either language.

**Batch 4080, `--batch max` territory (`runs/big_4080.txt`, contended):**
- The repo WGSL kernel spills into a second wave at 4080 and is slower per block: 11.47 / 12.69 µs, against 8.23
  / 9.27 at b2900. One wave holds at least 3400 and fewer than 3600 blocks.
- Median ratios:

| implementation | cheap | final |
|---|---:|---:|
| CUDA straight | 0.94 | 0.88 |
| CUDA v6 | 0.84 | 0.77 |
| CUDA v16 | 0.82 | 0.73 |
| WGSL `v_pk16` | **0.69** | **0.68** |

- `v_pk16` fits 24/SM, which is one wave.
- CUDA's cheap pass is capped at 20/SM (see §3) and runs 1.2 waves. It is still well ahead of the repo kernel thanks
  to backfill, but behind `v_pk16`.

**The whole opt16 schedule (3 cheap + final + fix-up) in CUDA v6:** 97.9 ms for 2900 blocks, or 33.8 µs/block. The
WGSL figure in context.md is 33.3.

## 3. What each CUDA-only capability gives (each measured against the same-session baseline)

### Register control (`__launch_bounds__`, `-maxrregcount`, ptxas `-v`)

| kernel (cheap / final) | registers | stack/local | static smem | blocks/SM |
|---|---|---|---|---|
| CUDA straight | 72 / 66 | **184 B stack** | 4332 | 19 / 19 |
| CUDA tuned v6 | 88 / 80 | 0 | 4004 | 20 / 20 |
| CUDA v16 (final pass without `hist`) | 88 / 80 | 0 | 4004 / **2984** | 20 / **24** |
| CUDA `__launch_bounds__(16,24)` | 80 / 80 | 0 | 4004 | 20 / 20 |
| Vulkan repo kernel (`vkstats`, driver 610, naga Restrict / unchecked) | 85–89 / 82–87 | 0 | **4520 / 4520** | ~20 measured |
| Vulkan `v_pk16` | 75 / 72 | 0 | 4192 | 24 (fits b4080) |

The CUDA blocks/SM come from the occupancy API. The Vulkan "~20 measured" means b3400 runs as one wave and b3600
does not.

- **nvcc does not beat the Vulkan driver on registers** once the code is the same. The driver already does what
  `__launch_bounds__` would force.
  - Capping registers to 80 (min 24 blocks/SM) changes nothing, because shared memory binds first.
  - `minBlocks` 28 or 32 is rejected: sm_120 allows at most 24 blocks/SM.
- These Vulkan register counts are lower than a02's 101/99 for the same source, on today's driver. Gate on `vkstats`
  on the driver actually shipped, not on old numbers.
- **The CUDA shared-memory reserve is a structural loss for this kernel shape.** `reservedSharedMemPerBlock` is
  1024 B on sm_80+, and the per-SM limit is 102 400 B.
  - 16-thread blocks pay 1 KB each, so 4004 B of tables and ring give 20 blocks/SM in CUDA.
  - Vulkan fits 24 × 4192 B (`v_pk16` runs b4080 in one wave). The driver evidently does not reserve the extra KB
    there.
  - To reach 24/SM in CUDA, the cheap pass would need at most 3242 B of static shared memory. Only the final pass gets
    there, by dropping `hist`.
- **Backportable finding:** the Vulkan pipeline reserves 4520 B of workgroup memory for both passes. That includes
  the 1024 B `hist` the final pass never uses, and `k3_fixup`'s 192 B of segment arrays, which `main_opt` never
  touches. The driver does not strip unused module-scope `var<workgroup>`.
  - Declaring `hist` only when HIST_OUT, and keeping the fix-up in its own module, takes the final pass to about
    3300 B (about 2980 with `v_pk16`).
  - Estimated, not timed in WGSL: this matters only above one wave. CUDA v16 shows the effect: final pass 0.73 vs
    0.77 at b4080.

### 16-bit and 8-bit types, `__byte_perm`, SIMD-in-word, `__clz`/`__ffs`, `__funnelshift`

- **u16 shared tables:** 0.98–1.01 at one wave. They cut shared memory by 328 B and add 1 block/SM.
- **`__funnelshift_r` for `ld32`:** 0.960 against 0.985 for v16 on cheap (min/min), and 0.949 against 0.950 on final;
  exact. That is within noise, ≤ 1 %.
- **`__clz`/`__ffs`:** the same SASS as WGSL's `firstLeadingBit`/`countTrailingZeros`.
- **`__vadd4`/`__vminu4`/`__byte_perm`:** nothing in the DP is 8-bit SIMD. `match_len` is XOR + ctz, and the prices
  are 16–30-bit table sums.
- **8-bit types:** no use.

### Warp intrinsics legal in divergent code (`__shfl_sync`, `__match_any_sync`, `__reduce_*_sync`, `__activemask`)

- **`__match_any_sync` aggregation of the hist code atomics (v10):** 0.975 / 0.960, against v6's 0.96–0.97: noise.
  a02 already bounded the histogram atomics at 2.7 % of a pass.
- **`__reduce_add_sync` for the prologue sums (v9):** noise. It is sm_80+ only: the sm_75 build needed a shuffle
  fallback, and the GTX 1660 Super is sm_75.
- **Profiling, the most useful divergent-legal use.** The PROF variant reads `clock64()` at the trip's reconvergence
  points, uses `__reduce_max/add_sync(__activemask())` for relax work, and has 0 differing words
  (`runs/prof.txt`). It splits the DP loop's warp cycles as:

  | pass | head + part 1 | get_all_matches | relax | tail |
  |---|---:|---:|---:|---:|
  | cheap | 28.9 % | 29.5 % | 32.9 % | 8.8 % |
  | final | 27.3 % | 29.9 % | 34.3 % | 8.5 % |

  - Active lanes per trip: 15.2 of 16.
  - Searching lanes per trip: 12.7 (cheap) and 14.0 (final).
  - Relax steps per trip: the warp runs 1.57–1.77 steps (its max), and the lanes together do 5.3–6.8. Lane
    utilization inside relax is therefore 0.22–0.25.
  - Phase 2 (emit) costs 8.6 % (cheap) and 4.0 % (final) of the DP-loop time.
  - Heaviest block: 1.53 / 1.76 × the median.
- **What the profile rules out:** cooperative relaxation over all 32 lanes, including the 16 idle ones of a wg16
  warp, which CUDA makes legal.
  - Even at perfect balance it removes at most about 0.6 of a relax step per trip, about 12 % of the loop.
  - It would cost a per-trip prefix sum, an owner search and a parameter fetch of about 6–10 shuffles. At level 0 it
    also needs a segmented scan to keep the early abort exact.
  - This is consistent with the perf study's measured 1.7× slowdown. Estimated, not built.

### Shared-memory layout, carveout, L2 persistence

- **Carveout 100 %:** 0.970 / 0.955–0.963, the same as the default; the driver already picks the maximum.
- **Carveout 50 %:** cuts residency, with median ratios of 1.9–2.4 under contention (min/min 1.12–1.14 for v6 on both
  passes).
- **L2 access-policy window (persisting, hitRatio 1) on the 18 MB node scratch (`scr`):** **slower**, 1.136–1.142
  cheap and 1.064–1.080 final, consistent over 8 rounds.
  - The set-aside L2 (60 MB) is taken from the streaming candidate and data traffic.
  - The candidate buffer itself (1.5 GB per wave) cannot be windowed.
- **Dynamic shared-memory sizing:** used only as an occupancy limiter (below).

### Occupancy APIs, persistent kernels, backfill: the one real CUDA difference

The test pads both implementations to 8 blocks/SM: dynamic smem in CUDA, a dummy `var<workgroup>` in WGSL, both 11.6
KB per block. One wave is then 1360 blocks. Cheap pass, total ms (`runs/bf_*.txt`):

| blocks | waves | WGSL (pad) | CUDA (pad) |
|---:|---:|---:|---:|
| 1360 | 1.00 | 21.2 (1.00×) | 18.6 (1.00×) |
| 1500 | 1.10 | **37.8 (1.78×)** | 24.5 (1.32×) |
| 2040 | 1.50 | 41.3 (1.95×) | 31.5 (1.70×) |
| 2720 | 2.00 | 41.7 (1.97×) | 33.7 (1.81×) |
| 2900 | 2.13 | **57.5 (2.71×)** | 35.3 (1.90×) |

- Vulkan's time steps with ceil(waves): 1, 2, 2, 2, 3 × about 20 ms, the heavy-block chain per wave. This
  independently confirms a07's synchronous-waves finding.
- CUDA backfills. Its time grows roughly with the work: at 2.13 waves it is 0.61× the Vulkan time.
- This is the only measured lever that is genuinely CUDA-only. a07's persistent WGSL kernel (atomic block counter,
  heavy-first) recovers the same thing portably: 51.1 → 31.1 µs/blk at b3586–4000.
- CUDA needs no persistent kernel for backfill. Heavy-first ordering would still help both: the tail is 1.5–1.8× the
  median block.

### cp.async / TMA staging; CUDA graphs and streams

- **cp.async / TMA:** not applicable by capacity.
  - A segment's data is 4 KiB and its candidate words 32 KiB. A block needs 64 KiB + 512 KiB, against a budget of
    about 4 KB per block.
  - A small look-ahead window is the same as prefetching, and that was measured: `prefetch.global.L1`/`.L2` two lines
    ahead of the candidate and data streams gave 0.982 / 0.970 (cheap) and 0.953 / 0.960 (final), against v16's
    0.976 / 0.971. That is noise, and exact.
  - K3 is not load-latency-bound; this agrees with a03.
- **CUDA graph vs stream launches** of the 5-kernel schedule: 97.919 vs 97.904 ms (+0.015 ms, noise). WGSL's dispatch
  gaps (span − Σ passes) are 0.024 ms. Nothing to win.

### Independent thread scheduling, `__syncwarp`

- **`__syncwarp(__activemask())` at the end of each trip (v7):** 0.977 / 0.947 against 0.960 / 0.951 for v6. No gain,
  exact. The compiler already reconverges at the trip's structured join.
- **Lane utilization:** 15.2/16 lanes are active per trip, so the trip-level lockstep is already intact. The
  divergence that remains is inside relax and get_all_matches, and is data-dependent.

## 4. Cost of a CUDA path in the product

| item | cost |
|---|---|
| Code | A second byte-identical implementation of K3opt (main_opt × 4 price modes, `hist_epilogue`, fix-up), about 900 lines that track every WGSL change. For a full CUDA pipeline, also K1, K2opt, K4 and K5. A CI gate would need an NVIDIA runner. |
| Build | nvcc on the build machine (CUDA 13 still targets sm_75; checked). A fatbin for sm_75/86/89/120 plus compute_75 PTX takes 37 s to build and is about 0.45 MB per kernel set (9.4 MB for my 23 variants × 2 passes × 4 archs). NVRTC at runtime would add a ~100 MB redistributable; PTX JIT through the driver API (e.g. the `cudarc` crate, loaded dynamically) avoids it. |
| Driver matrix | A CUDA 13.x fatbin or PTX needs an R580-era or newer driver. Older drivers on GTX 1660 / RTX 30 machines need PTX from an older toolkit, or the WGSL fallback. Windows and Linux both work through the display driver (`nvcuda.dll` / `libcuda.so`); there is no extra install with `cudart` linked statically. (Driver-version claims are from NVIDIA's support matrix as I know it, not tested here.) |
| VRAM | **The CUDA context costs about 500 MiB of device memory, measured with `cudaFree(0)` and with the module loaded.** That is 8 % of the 6144 MiB budget on an 8 GB card, and `vram_bytes` would have to count it. |
| Interop (K3 only in CUDA) | Possible: create the K3 buffers as exportable VkBuffers (`VkExternalMemoryBufferCreateInfo`, OPAQUE_FD / OPAQUE_WIN32) through ash and wgpu-hal (the repo already uses both for ReBAR and multi-queue), wrap them with `create_buffer_from_hal`, and import them with `cudaImportExternalMemory`. wgpu 30 exposes no external semaphores, so synchronization would be a CPU wait per hand-off (submit → poll → CUDA → stream sync → submit). Separate CUDA and Vulkan contexts time-slice and never overlap. Effort M–L, with no DX12 or Metal story. |
| Coverage | NVIDIA only. Arc, Apple and AMD still need the WGSL path, so the portable path keeps every maintenance cost. |

**HIP:** not testable here.
- `k3.cu`'s tuned path uses only `__ldg`, `__clz`, `__ffs`, `__syncthreads` and shared/global atomics. hipify would be
  nearly mechanical.
- The `__reduce_*_sync` and `__match_any_sync` variants, `prefetch` asm and access-policy windows don't port, but none
  of them helped anyway.
- HIP compiles gfx10/gfx11 as wave32 by default. That would settle a02's open wave32/64 question for wg16 on RDNA,
  which is the one plausible HIP benefit.
- The costs: a third byte-identical implementation, and a runtime that ROCm officially supports on few consumer
  cards on Linux. RX 6600/7600 are not on the official list; the Windows HIP runtime ships with the Adrenalin driver.
- A cheaper first check is RADV/AMDVLK pipeline statistics for the wave size of the wg16 pipeline on an RX card. If
  that shows wave64, force `requiredSubgroupSize` through wgpu-hal (VK_EXT_subgroup_size_control) rather than add
  HIP.

## 5. Ideas, ranked

| # | idea | mechanism | speedup | ratio | exactness | effort | risk |
|---|---|---|---|---|---|---|---|
| 1 | **Don't build a CUDA K3.** Ship a02's diet (`v_pk16` or I1+I2) in WGSL | the same register and shared-memory diet CUDA needed to break even | measured here: WGSL `v_pk16` 0.94 / 0.97 at b2900, 0.84 / 0.90 at b1360, 0.69 / 0.68 at b4080; CUDA tuned is at best equal | none | exact (checked: 0/2900 parses, 0 hist words) | S | low |
| 2 | **Persistent WGSL K3 + heavy-first (a07)** as the portable replacement for CUDA's backfill | the Vulkan driver runs whole waves; CUDA (measured) backfills | CUDA-measured upper bound: 1.90× instead of 2.71× of one-wave time at 2.13 waves (0.70× time); a07 measured 0.61× at b3586 | none | exact | M | low |
| 3 | **Trim workgroup memory per entry point** | final pass: no `hist` (1024 B); `main_opt`: no fix-up arrays (192 B); the driver does not strip unused `var<workgroup>` | above one wave only: final pass 22 → 24/SM (CUDA analogue: 0.77 → 0.73 at b4080); 0 at b2900 | none | exact | S | low |
| 4 | Keep `k3.cu` as a **dev-only profiler and cross-check** (not shipped) | `clock64` + `__reduce_*_sync` section attribution; a second implementation that has to agree word for word | no runtime gain | — | — | S | none |
| — | CUDA-only knobs: L2 persistence (1.08–1.14× **slower**), carveout, graphs, prefetch, funnelshift, `__syncwarp`, redux/match, `__launch_bounds__` | — | 0 ± 1 % or worse | — | — | — | dead ends |

## 6. Dead ends checked (all measured unless marked)

- **Straight CUDA port:** 1.10–1.14× slower, because of local-memory record arrays.
- **`__launch_bounds__`/`maxrregcount`:** no gain; shared memory binds at 20/SM.
- **L2 persisting window on the scratch:** 1.06–1.14× slower.
- **Carveout:** default = 100 %; 50 % is worse.
- **CUDA graphs:** 0.015 ms on 97.9 ms.
- **Prefetch L1/L2:** ±1 %.
- **`__funnelshift_r`:** ±1 %.
- **`__syncwarp` per trip:** 0.
- **`__match_any_sync` hist aggregation, `__reduce_add_sync` prologue:** 0, and sm_80+ only.
- **ll_code without the constant table:** 0.
- **u16 tables at one wave:** 0. They pay only through occupancy.
- **cp.async/TMA staging:** no shared-memory room; it would be equivalent to prefetch, which measured nothing.
- **Warp-cooperative relaxation using the idle half-warp:** estimated ≤ 12 % before overhead (relax is about 1.6
  warp-steps per trip); the perf study measured 1.7× slower.
- **CUDA as a residency lever:** worse than Vulkan, because of the 1 KB per-block reserve.

## 7. Measurement conditions

- Shared GPU. Other agents' GPU tests ran during most runs (the `gzc_gpu`, `differential` and `k3opt` processes in
  the logs), with load averages of 2.4–9.3.
- All absolute µs are under contention unless the row says otherwise. b2900 in §2 is the quietest (load 2.4–4.0,
  rounds within ±1.5 %).
- Every comparison is interleaved in one process on the same uploaded blocks and reported as a ratio. Rounds
  overlapped by other agents show up as outliers in the per-round lists, which is why min/min is given too.

## Top-3 recommendation

1. **Do not build a CUDA (or HIP) K3 path.** Measured on the same blocks:
   - the tuned CUDA port only matches a portable WGSL with a02's diet, and loses above one wave because of CUDA's 1
     KB per-block shared-memory reserve;
   - the product would pay a second byte-identical implementation, a driver/toolkit matrix, an interop layer
     without external semaphores, and about 500 MiB of VRAM for the CUDA context on 6 GB-budget cards.
2. **Backport the lessons to WGSL:**
   - a02's I1+I2 (`v_pk16`), with the register and shared-memory gate measured by `vkstats` on the shipped driver
     (85–89 regs and 4520 B today for the repo kernel; 75/72 and 4192 for `v_pk16`);
   - per-entry-point workgroup trimming (no `hist` in the final pass, fix-up arrays in their own module);
   - a07's persistent + heavy-first K3, which delivers in WGSL the backfill CUDA gets for free (CUDA measured 0.70×
     the Vulkan time at 2.13 waves).
3. **Keep the CUDA prototype as a dev tool, not a product path.** Its `clock64` and warp-reduction profile is the
   only per-section attribution available on this machine. It shows the remaining K3 time is about 30 % trip head,
   30 % match search and 33 % relax, at 15.2/16 active lanes. That points future work at fewer passes (synthesis B)
   and fewer instructions per trip (a01/a09), not at hardware features.
