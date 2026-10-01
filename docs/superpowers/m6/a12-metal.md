# a12-metal: does dropping to Metal give us anything?

Agent a12-metal, 2026-09-30. Linux box (RTX 5090); no Mac access. M4 Pro figures come from the peer session's ledger entry (1.36 GB macOS-files corpus, batch 5118).

## Summary (≤ 300 words)

**A native Metal backend would buy almost nothing. A few Metal-specific fixes inside the wgpu path are worth landing, and the biggest one is a gating bug, not a codegen issue.**

1. **On Metal, every subgroup kernel falls back.**
   - wgpu reports Apple subgroups as 4..=64 (hard-coded in `wgpu-hal/src/metal/mod.rs:445`). Our gates want a minimum of ≥ 32.
   - So on Apple, K1 runs the bitonic fallback: 38 barriers per 256-position tile, a 256 KiB head-table clear per chain, and 256 live tables (64 MiB).
   - The sorted K1 runs its O(32)-per-position fallback.
   - K3coop runs at W = 8 instead of 32.
   - Apple compute SIMD width is always 32.
   - Measured on the 5090, the fallbacks cost: K1 ×1.44 (opt16), ×1.45 (lvl3), ×1.80 (lvl9seg); k1_sort ×2.2.
   - This explains "K1 = 70 % of lvl3" and "lvl9s12seg slower than lvl9seg" on the M4.
   - Fix: trust width 32 on Metal; the existing self-tests guard it. Patch included (`GZC_METAL_SG32`).
2. **Apple runs shifts, multiplies, ctz/clz and popcount at quarter rate** (Philip Turner: 32 per core-cycle vs 128 for add; RSHIFT about 8 cycles). NVIDIA runs them at half rate.
   - K2's funnel-shift match loop and K1's hashing are full of these ops. That plausibly explains why K2 is about 31–41× slower on the M4 than on the 5090, against about 14× raw FP32.
   - Native MSL can't fix this. A portable fix can: a word-per-position data layout that needs no shifts. A microbenchmark is included.
3. **Codegen is mostly clean.**
   - `simd_ballot` is native, `max_total_threads_per_threadgroup` is emitted, and there's no div/mod in K1 or K2.
   - Costs found:
     - (a) The "checked"/"unbounded" modules clamp every storage access on Metal (no robust buffer access): 76 sites in K3opt. Vulkan has no such clamps.
     - (b) Workgroup memory is zero-initialised by one thread, serially: 2048 atomic stores per block in the sorted fallback.
     - (c) The K3 hot loop does 4 × `% 33` per step. Patch included; byte-identical; neutral on the 5090.
4. **Native-Metal extras are irrelevant here:** unified-memory zero-copy (wgpu can already do it with `GZC_DIRECT_UPLOAD=1`), ICBs, heaps, imageblocks, simdgroup_matrix, fast resource loading and Metal 4. Of everything native Metal offers, only `ushort` might matter.
5. **Estimate: opt16 on the M4 goes from 141 to about 155–175 MB/s, which is still CPU parity.**
   - Apple GPUs aren't a sensible acceleration target for this tool.
   - On a Mac, either default to CPU libzstd or run CPU L16 and GPU opt16 side by side on disjoint blocks (about 2×; measure it).

---

## 1. What runs on Metal today (source inspection)

All of this comes from reading `chains.rs`, `sorted.rs`, `compressor.rs::k3_mode` and wgpu-hal 30.0.1.

| Kernel | Gate | Metal (adapter reports 4..=64) | NVIDIA (32..=32) |
|---|---|---|---|
| K1 chains | `subgroup_min_size >= 32 && max <= 128` (`chains.rs:180`; the comment even says "Apple GPUs take the fallback") | **k1_chains.wgsl fallback**: bitonic sort, 36 + 2 barriers per 256-position tile | k1_chains_sg: 2 barriers per tile, ballots |
| K1 sorted (lvl9s12seg) | `subgroup_min_size >= 32` (`sorted.rs:70`) | **`main` fallback**: each lane scans 32 workgroup keys per tile | `main_sg`: ballots, simd prefix sums |
| K3 lazy/coop (lvl3/lvl9) | `w = min.clamp(8, 64)`; `GZC_K3_W` must be ≤ min (4), so it can't even be overridden | **W = 8** | W = 32 |
| K3opt | no subgroups | same kernel | same |

Apple GPUs (Apple 7–9, M1–M4) execute compute at `threadExecutionWidth` = 32. The 4..=64 range is a static, conservative constant in wgpu-hal (`metal/mod.rs:445-446`), not a property of the pipeline.

The sg kernels already self-test, and the K3coop path probes lanes with `probe_lanes`. So trusting 32 on Metal is safe in the sense that a wrong guess falls back.

**Measured on the 5090: what the fallbacks cost.** `GZC_NO_SUBGROUPS=1` vs default, 800 MB corpus subset, batch 2900, 2 interleaved rounds. Load was 3–99 % from other agents; the K1/K2 rows were stable across rounds.

| preset | K1 sg (ms/batch) | K1 fallback | ratio | whole-pipeline sum, sg → fallback |
|---|---:|---:|---:|---|
| lvl3 | 17.3 | 25.1 | 1.45× | 25.5 → 44.2 ms (K3coop off as well) |
| lvl9seg | 7.2 | 13.0 | 1.80× | 20.0 → 25.9 |
| lvl9s12seg (k1_sort) | 3.4 | 7.5 | 2.2× | 17.3 → 21.4 |
| opt16 | 17.0 | 24.6 | 1.44× | 119.2 → 127.3 |

On NVIDIA, K1 is 68 % of lvl3 even with the sg kernel. So the M4's "K1 ≈ 70 % of lvl3" isn't Apple-specific in itself, but on Apple that K1 is the slow fallback. On Apple I'd expect the fallback's penalty to be at least the NVIDIA one, for three reasons:
- its 9,700 barriers per chain per block;
- a 256 KiB head-table clear per chain per block, written to device memory;
- 256 live head tables (64 MiB), far beyond Apple's small L2 (M2 Pro: 3 MB; Chips and Cheese) and beyond the SLC.

That penalty is an estimate, not a measurement. Use `GZC_K1_GROUPS` to size the sg grid for Apple's caches (see the plan).

## 2. The MSL naga 30 generates (inspected)

Method:
- A scratch tool (`naga = "=30.0.1"`, `wgsl-in` + `msl-out`) with exactly the option set from `wgpu-hal-30.0.1/src/metal/device.rs::load_shader`: MSL 3.2, `zero_initialize_workgroup_memory: true` (wgpu's default; we pass `Default::default()` compilation options), and `emit_int_div_checks: true`.
- Bounds policy: `Restrict` for index and buffer when `runtime_checks.bounds_checks`, otherwise `Unchecked`. `force_loop_bounding` per module.
- The exact WGSL came from a dump hook in `GpuContext::wgsl_module`, captured during a short bench run. The Metal-only variants (K1 fallback, sorted fallback, sg-wide) were assembled from the same prefix.

| Module (crate policy) | storage clamps `min(i, (size-4)/4)` | index clamps | 64-bit loop-bound counters | `naga_div/mod` | simd ops |
|---|---:|---:|---:|---:|---:|
| K1 fallback (checked) | 10 | 18 | 6 loops | 5 (setup only) | 0 |
| K1 sg, wide masks (checked) | 13 | 22 | 4 | 5 | `simd_ballot`, `simd_any` |
| sorted fallback (checked) | 5 | 18 | 8 | 0 | 0 |
| sorted sg (checked) | 6 | 12 | 7 | 0 | ballot, shuffle, broadcast_first, all, prefix_exclusive_sum, sum |
| K2 best / K2opt / K2 window (trusted) | 0 | 0 | 0 | 0 | 0 |
| K3opt cheap/final (unbounded) | **76** | **169** | 0 | 9 (`% RING_N` in `slot()`) | 0 |

### Findings

1. **Storage clamps exist only on Metal.**
   - On Vulkan, wgpu sets `buffer: Unchecked` when the device has robust buffer access (`vulkan/adapter.rs:2838-2846`). Metal has no such feature, so every checked or unbounded module clamps every device load and store against `_buffer_sizes.sizeN`.
   - K2 is "trusted" and is clean. K1 and K3opt pay for the clamps.
   - Each clamp is one `IMIN32` (full rate on Apple) plus a loop-invariant size expression. That's cheap but not free.
   - I estimate 1–4 % on K3opt (unmeasured). The patch has `GZC_K3_TRUSTED=1`: on the 5090, `k3opt` tests pass and output stays byte-identical. There's no written safety argument yet.
2. **Workgroup zero-init runs on a single thread.**
   - naga emits `if (lid == 0) { var = {}; ... for (i < N) atomic_store(...) }` followed by a barrier. Vulkan instead delegates zero-init to the driver (`shaderZeroInitializeWorkgroupMemory`).
   - K3opt: about 4.6 KiB of struct zeroing plus 256 + 5 serial atomic stores per workgroup per pass. That's under 1 % of a pass (estimate).
   - Sorted fallback: 2048 serial atomic stores per block, and the kernel clears `cnt` again itself, so the work is redundant. Estimate 3–5 % of k1_sort.
   - K4 and K5 also declare several KiB of workgroup memory (`k4_seq_entropy.wgsl:49-95`, `k5_huffman.wgsl:54-66`).
   - Every one of our kernels explicitly initialises what it reads (K3 clears `hist`/`hsum`; K1 writes `keys` before reading). So `zero_initialize_workgroup_memory: false` should be safe. That needs an audit before it lands; the M4 tests are the gate.
   - On the 5090, with zero-init off, the oracle tests pass (k3opt, opt_pipeline, differential). The timing was too contended to call; a quiet round showed no measurable change.
3. **Force-loop-bounding in the checked modules (K1, sorted).**
   - Every loop gets a `uint2` 64-bit down-counter and an `all(==0)` test. That's 2–3 extra ALU ops per iteration, and it obscures trip counts from the AGX compiler (the bitonic k/j loops).
   - It only matters for the fallbacks, which shouldn't run anyway.
4. **Ballots map natively**: `metal::uint4((uint64_t)metal::simd_ballot(x), 0, 0, 0)`.
   - With "wide" masks (`K1_BALLOT_WORD = [word]`, chosen because max = 64), the kernel indexes the vec4 dynamically with `word = sg_lane >> 5`, which is always 0 on Apple. That's minor.
   - The sg-sort's `subgroupShuffle(w, i & 31)` with per-lane indices is a "random" shuffle on Apple: about 28–32 cycles, vs 2 for broadcast or rotate (Turner). That's acceptable; it's still far cheaper than the fallback.
5. **K3opt's `slot(pos) = pos % 33` is called 4× per relaxation step.**
   - That's a modulo by a constant (multiply-high, multiply, subtract). On Apple, `IMULHI32` takes 8 cycles and `IMUL32` 4, all on the quarter-rate "integer and complex" pipe.
   - Fix (byte-identical): one modulo, then `select(s-1, RING_N-1, s==0)` three times. The patch is `a12-k3slot.patch`.
   - On the 5090: opt16 pipeline sum 119.25 vs 119.38 ms/batch (quiet round), identical output size, and oracle tests pass.
   - On Apple I estimate 2–5 % of K3. Needs the M4.
6. **Nothing else stood out.**
   - Workgroup memory is passed as `threadgroup T&` kernel arguments, and wgpu sets the lengths at encode time; that's standard.
   - `[[max_total_threads_per_threadgroup(N)]]` is emitted from `@workgroup_size`.
   - Constants are injected as WGSL text, so everything is already fully specialised; function constants would add nothing.
   - Divisions occur only in kernel prologues.
   - `preserveInvariance` only affects vertex shaders.

## 3. Apple GPU architecture vs our kernels

| Property | Apple (M1/M2 measured; M3/M4 = Apple 9) | Effect on us |
|---|---|---|
| SIMD width | 32 lanes, 4 schedulers × 1 instruction per cycle; 128 ALUs per core ([metal-benchmarks]) | Same as NVIDIA warps, so the sg kernels fit as written |
| Integer pipe | IADD/bitwise/select: 128 per core-cycle. **IMUL32, shifts, BITEXTRACT, POPCOUNT, CLZ: 32 per core-cycle; RSHIFT32 about 8 cycles; IMULHI 8; dynamic-amount shifts look like multi-instruction sequences** ([metal-benchmarks], integer table). Ampere/Ada: 64 per SM-cycle for IMUL and shifts | Our hot code is shift/mul/ctz heavy: funnel loads (3 shifts each), hash `mix` (3 IMUL), `ctz` (CLZ ∘ BITREV ≈ 8 cycles), `% 33`. **That pipe is 2× slower relative to add than on NVIDIA.** |
| Register file / occupancy | ~208 KB per core, 384–3072 threads per core; ALU saturates at 24 SIMDs per core. **M3/M4 Dynamic Caching:** registers, threadgroup, tile, stack and buffer data share one on-chip cache; registers are allocated as used, not at the program-wide maximum, and the hardware lowers occupancy to avoid spilling ([Apple tech talk 111375]) | K3opt's 96–102 registers cost less occupancy than on NVIDIA, because peak-register regions are short. So A1 (register diet) will gain less on M4 than on NVIDIA. Apple 9 also issues FP32, FP16 and INT in parallel "to a greater degree", but only across SIMD groups, so occupancy still matters |
| Threadgroup memory | 32 KiB per threadgroup (adapter), ~60 KB per core; on Apple 9 it's a cache over the same SRAM | K3opt (4.6 KiB) is fine |
| L1 / L2 / SLC / DRAM | L1D 8 KB per core (M1/M2). L2: M2 Pro 3 MB. SLC latency ~234 ns, DRAM > 342 ns, rising beyond 400 ns with TLB misses (M2 Pro) ([Chips and Cheese M2 Pro]). DRAM latency is notably higher than discrete GPUs | K2's dependent `pred[q]` chain walk is latency-bound. The 5090 has ~5× more warps in flight (170 SM × 48 vs 16 cores × ≤ 96 SIMDs) to hide a similar or smaller latency |
| Global atomics | ~58.6 ns latency (M2 Pro, Chips and Cheese), comparable to RDNA2; no special fast path | Only K1 sg (relaxed head-table atomics, one per hash run per tile) and K3 persistent queues (A4) use them. Not a bottleneck |
| Unified memory | No PCIe; `StorageModeShared` buffers are GPU-visible with no copy | Upload is a CPU memcpy: about 1–3 ms per 335 MB batch against a batch time of 0.7–2.4 s. **< 1 %** |

### Why each kernel behaves as measured

- **Hash-chain K1 (fallback)**
  - It's barrier-bound: 36 sort stages per 256 positions, each doing about 2 threadgroup loads, a compare and 2 stores, then a barrier.
  - It also does DRAM-level random RMW on 256 head tables (64 MiB), plus 256 KiB of clears per chain.
  - The arithmetic is 3 IMUL per hash at quarter rate, but the barriers dominate.
  - The M4 K1 time (≈ 92 µs/block for lvl3, derived from 70 % of 670 ms / 5118 blocks) is about 15× the 5090's 6 µs with the sg kernel. The M4 is running the slower kernel.
- **K2 chain walk**
  - Per candidate, it does:
    - one dependent random 4-byte load (`pred[pb+q]`);
    - 3 data loads;
    - 2 funnels (6 shifts, half of them `>>`, which take about 8 cycles);
    - a ctz.
  - Its 32 lanes diverge over a variable chain depth.
  - The M4 takes 350–460 ms per 5118 blocks (68–90 µs/block). The 5090's K2opt takes 2.2 µs/block.
  - That's a **31–41× gap, vs about 14× in FP32 peak** (5090 ~105 TFLOPS; M4 Pro 16-core ~7 TFLOPS).
  - The extra 2–3× matches two effects together:
    - the quarter-rate shift/ctz pipe, which costs 2× relative to NVIDIA's half rate;
    - weaker latency hiding.
  - The trusted K2 MSL is clean, so this is the hardware, not the translation.
- **Sorted K1**
  - Its fallback ranking does 32 compares per lane per tile, plus 2048 serial zero-init atomics per block. NVIDIA runs this fallback 2.2× slower than the sg kernel.
  - The K2 window it feeds does the same shift-heavy `match_len_capped` as K2.
  - Hence lvl9s12seg < lvl9seg on Metal, while on NVIDIA (sg kernels) it's the reverse.
- **K3opt**
  - It's mostly adds, compares, selects and threadgroup loads, which run full rate on Apple.
  - Its M4/5090 ratio (≈ 465/41 ≈ 11×) is *better* than the FP32 ratio. That's consistent with the integer-pipe explanation.

## 4. What native Metal offers that wgpu/WGSL doesn't

| Feature | Relevance and estimated gain | Reachable from wgpu? |
|---|---|---|
| SIMD functions legal in divergent code (MSL ops act on active lanes) | The subgroup-cooperative ideas were already measured as dead ends on NVIDIA (a01/a02). **≈ 0** | naga emits `simd_*` without uniformity enforcement in practice; not needed |
| `simdgroup_matrix` | Float 8×8 MMA. We have no matrix math. **0** | No |
| Native `ushort`/`uchar` arithmetic (16-bit register halves, "zero additional cost" operands) | K3opt node fields are 16-bit packed (`a >> 16`, `& 0xFFFF`: 18 sites). With `ushort`, register halves remove the BITEXTRACTs and roughly halve register footprint for those fields. **Estimate 3–8 % K3 on Apple**, unmeasured; under Dynamic Caching register savings matter less | **No.** WGSL has `f16` but no `u16`. The only option is the patch-the-MSL hack (not viable) |
| Threadgroup imageblocks | Tile-shading feature for render passes. **0** | No |
| `maxTotalThreadsPerThreadgroup`, function constants | Already emitted and already specialised as text. **0** | Already |
| Unified memory, zero-copy input (`StorageModeShared` / `newBufferWithBytesNoCopy`) | Removes one memcpy per batch. **< 1 %** | **Yes**: wgpu-hal maps `MAP_WRITE` buffers to `StorageModeShared` (`metal/device.rs:469-476`), and Metal exposes `MAPPABLE_PRIMARY_BUFFERS`. Our `rebar()` heuristic is Vulkan-only, so direct upload is off; `GZC_DIRECT_UPLOAD=1` forces it |
| Indirect command buffers, dispatch overhead | About 15 dispatches per 0.7–2.4 s batch. **0** | n/a |
| MTLHeap aliasing | VRAM budget, not speed. Unified 24 GB. **0** | n/a |
| Metal 3 fast resource loading (MTLIO) | Loads files into buffers with built-in LZ4/LZFSE/zlib *decompression*. Our input comes from the network. **0** | n/a |
| Metal 4 (MTL4 encoders, argument tables, residency sets, MTLTensor) | Lower CPU encode cost and ML tensors. **0** for 15 big dispatches | n/a |
| GPU counters (Xcode "ALU / Integer and Complex limiter") | Diagnostic only, but the best way to confirm finding 2 | Xcode / Instruments on a capture |

**Net:** native Metal's only real lever is 16-bit integer types. That's worth maybe 3–8 % of K3 on Apple, against an L-size second backend that would need its own oracle-equality CI. Not worth it.

## 5. Ideas, ranked

| # | Idea | Mechanism | Expected effect on M4 (estimated unless stated) | Ratio / exactness | Effort | Risk |
|---|---|---|---|---|---|---|
| M1 | **Trust SIMD width 32 on Metal** (`GZC_METAL_SG32` in the patch; make it the default if it validates) | sg K1, sg sort and K3coop W=32 instead of the fallbacks | K1 ÷1.4–2 and k1_sort ÷2+ (5090-measured ratios); **lvl3 +30–45 %, lvl9seg +20–35 %, lvl9s12seg +25–40 %, opt16 +5–10 %** | none / byte-identical (self-tested) | S | Low: self-tests fall back |
| M2 | **sg-K1 grid sized for Apple caches** (`GZC_K1_GROUPS`) | 128 × 256 KiB = 32 MiB of live tables vs M4 Pro's L2 and SLC | K1 ±10–30 % | none | S (env sweep) | Low |
| M3 | **Shift-free "expanded" byte view**: `data4[b*BS + p]` = bytes p..p+3, built once per batch (K1 can write it from the words it loads) | K2/K2opt/K2window `match_len_capped`, K1 hashing and K3 rep checks load one aligned word per 4-byte compare, with no funnel shifts. Costs 4× data footprint (+192 KiB per block; free on unified memory, budget-relevant on 8 GB NVIDIA, so Metal-only at first) | K2 maybe ×0.6–0.8 if microbench B confirms; **opt16 +4–8 %, lvl9 +10–20 %** | byte-identical (pure speed) | M | Medium: cache footprint |
| M4 | **K3 slot: one `% 33` per step** (`a12-k3slot.patch`) | removes 3 IMULHI+IMUL sequences per relaxation step | K3 2–5 % (estimate); **5090 measured neutral** (119.25 vs 119.38 ms/batch), exact | byte-identical (oracle tests pass) | S | Very low; also fine on NVIDIA |
| M5 | **No workgroup zero-init** (`GZC_NO_WG_ZERO`) | drop naga's single-lane zeroing (Metal) / the driver's (Vulkan) | sorted fallback 3–5 %, K3opt < 1 %, K4/K5 1–3 % | exact if every kernel initialises what it reads (oracle tests pass on 5090); needs an audit | S | Low–medium: UB if a kernel reads uninitialised memory |
| M6 | **Trusted K3opt on Metal** (`GZC_K3_TRUSTED`) | drop 76 storage and 169 index clamps | 1–4 % K3 | exact; needs the E9-style index argument written down | S–M | Medium (UB if wrong) |
| M7 | Direct upload on Metal (`GZC_DIRECT_UPLOAD=1`; make `rebar()` say yes for Metal) | zero-copy input | < 1–2 % | exact | S | Low |
| M8 | Native Metal backend (MSL with `ushort` in K3) | 16-bit registers | 3–8 % K3 on top of M1–M7 | needs a separate oracle-equality CI | L | High maintenance |

Combined M1+M3–M6 for opt16 on M4: about **+10–25 %, from 141 to 155–175 MB/s** (estimate). lvl3/lvl9 gain more, but CPU libzstd on 10 threads is 2.7–10× faster there anyway.

## 6. Experiment plan for the M4 Pro peer

Prerequisites:
- Repo at master (patches apply cleanly on 1f825ec).
- `C` = your corpus dir.
- Run serially on a quiet machine. Plug in the charger and close other apps.
- Interleave variants; report ratios.

```sh
cd gpu-zstd-comp
A=.superpowers/m6-research/artifacts/a12-metal
git switch -c a12-metal-exp
git apply $A/a12-env.patch                       # env knobs only; defaults unchanged
cargo build --release -p gzc-bench && cp target/release/gzc-bench /tmp/gzc-env
git apply $A/a12-k3slot.patch                    # + K3 slot change
cargo build --release -p gzc-bench && cp target/release/gzc-bench /tmp/gzc-slot

# 0. Correctness gates (each must pass; look for "using the fallback" lines: there must be none with SG32)
GZC_METAL_SG32=1 cargo test --release -p gzc-gpu --test chains --test differential --test k3opt --test opt_pipeline 2>&1 | grep -E "test result|fallback|probe failed"
GZC_METAL_SG32=1 GZC_NO_WG_ZERO=1 cargo test --release -p gzc-gpu --test differential --test k3opt --test opt_pipeline 2>&1 | grep -E "test result|FAIL"
GZC_K3_TRUSTED=1 cargo test --release -p gzc-gpu --test k3opt --test opt_pipeline 2>&1 | grep "test result"

# 1. Main A/B: two interleaved rounds. --verify also catches anything the tests missed.
P="gpu --input $C --preset lvl3,lvl9seg,lvl9s12seg,opt16 --batch 5118 --verify --out /tmp/a12"
for r in 1 2; do
  /tmp/gzc-env $P                                   2>&1 | tee /tmp/a12-base-$r.txt
  GZC_METAL_SG32=1 /tmp/gzc-env $P                  2>&1 | tee /tmp/a12-sg32-$r.txt
  GZC_METAL_SG32=1 /tmp/gzc-slot $P                 2>&1 | tee /tmp/a12-slot-$r.txt
  GZC_METAL_SG32=1 GZC_NO_WG_ZERO=1 /tmp/gzc-slot $P 2>&1 | tee /tmp/a12-nz-$r.txt
  GZC_METAL_SG32=1 GZC_NO_WG_ZERO=1 GZC_K3_TRUSTED=1 GZC_DIRECT_UPLOAD=1 /tmp/gzc-slot $P 2>&1 | tee /tmp/a12-all-$r.txt
done
grep -hE "^gpu |k1_|k2_|k3|sum|k3 mode" /tmp/a12-*-?.txt
# Expect: "k3 mode: Coop { w: 32" with SG32 (w: 8 without), and k1_chains/k1_sort time down. Ratios must be identical across variants.

# 2. sg-K1 grid size for Apple's caches (lvl9seg + opt16)
for g in 16 32 64 128 256; do GZC_METAL_SG32=1 GZC_K1_GROUPS=$g /tmp/gzc-slot gpu --input $C --preset lvl9seg,opt16 --batch 5118 --out /tmp/a12 2>&1 | grep -E "^gpu |k1_chains" | sed "s/^/g=$g /"; done

# 3. Integer-pipe and layout microbenchmark (inline below, also at $A/applebench.swift)
swiftc -O $A/applebench.swift -o /tmp/applebench && /tmp/applebench

# 4. Hybrid ceiling: GPU opt16 and CPU libzstd-16 at the same time (sum the two MB/s; compare with each alone)
( GZC_METAL_SG32=1 /tmp/gzc-slot gpu --input $C --preset opt16 --batch 5118 --out /tmp/a12g & \
  /tmp/gzc-slot cpu --input $C --levels 16 --threads 8 --out /tmp/a12c ; wait ) 2>&1 | grep -E "^(gpu|cpu)"

# 5. (optional) Xcode/Instruments: capture one opt16 batch; check the "Integer and Complex" vs "F32" limiter for k2_*/k1_*.
```

### How to read the results

- **SG32 vs base:** K1 per-batch time should drop by ×1.4 or more and k1_sort by ×2 or more; otherwise check stderr for the self-test message.
- **slot vs sg32:** a k3_opt change of 2 % or more makes M4 worth landing. It's neutral on NVIDIA, so land it regardless if the gain is ≥ 0.
- **applebench Part A:** if `shr_dyn`, `funnel_naga`, `ctz`, `imul` and `mod33` come out ≥ 3× `iadd` on the M4 (M1 data says 4–8×), finding 2 holds on Apple 9 too.
  - If `funnel_c` or `funnel_64` is much cheaper than `funnel_naga`, a cheaper spelling of `funnel` in WGSL is an S-sized win (`funnel_c`'s `s == 0` select matches today's semantics).
- **applebench Part B:** a ratio of ≤ 0.8 with identical outputs justifies implementing M3 behind a Metal-only flag.

### `applebench.swift` (save and run as above)

```swift
// a12-metal microbenchmark for Apple GPUs. Build and run on the Mac:
//   swiftc -O applebench.swift -o applebench && ./applebench
// Part A: integer op throughput (ops per lane-cycle relative to IADD) for the ops our kernels lean on.
// Part B: K2-style chain walk + match_len over 64 KiB blocks, packed words (today's layout, funnel
//         shifts) vs a word-per-position "expanded" layout (no shifts); outputs must match.
import Foundation
import Metal

let src = """
#include <metal_stdlib>
using namespace metal;

// ---------- Part A ----------
#define ITERS 2048u
#define ALU(NAME, OP) \\
kernel void NAME(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]], \\
                 uint tid [[thread_position_in_grid]]) { \\
  uint a = in[tid & 1023u], b = in[(tid + 1u) & 1023u], c = in[(tid + 2u) & 1023u], d = in[(tid + 3u) & 1023u]; \\
  uint y = in[(tid + 5u) & 1023u] | 1u; uint s = in[(tid + 7u) & 1023u] & 31u; \\
  for (uint i = 0u; i < ITERS; i++) { \\
    a = OP(a, y, s); b = OP(b, y, s); c = OP(c, y, s); d = OP(d, y, s); \\
    a = OP(a, y, s); b = OP(b, y, s); c = OP(c, y, s); d = OP(d, y, s); \\
    s = (s + 1u) & 31u; \\
  } \\
  out[tid] = a ^ b ^ c ^ d; }

inline uint op_iadd(uint x, uint y, uint s)        { return (x + s) ^ y; }
inline uint op_shl_dyn(uint x, uint y, uint s)     { return (x << s) ^ y; }
inline uint op_shr_dyn(uint x, uint y, uint s)     { return (x >> s) ^ y; }
inline uint op_shr_const(uint x, uint y, uint s)   { return (x >> 16u) ^ y ^ s; }
inline uint op_and16(uint x, uint y, uint s)       { return (x & 0xFFFFu) ^ y ^ s; }
// naga's spelling of K2's funnel (two shifts on the high word)
inline uint op_funnel_naga(uint x, uint y, uint s) { return (x >> s) | ((y << (31u - s)) << 1u); }
// a C spelling the compiler might map to one extract instruction
inline uint op_funnel_c(uint x, uint y, uint s)    { return s == 0u ? x : ((x >> s) | (y << (32u - s))); }
inline uint op_funnel_64(uint x, uint y, uint s)   { return uint(((ulong(y) << 32) | ulong(x)) >> s); }
inline uint op_extract(uint x, uint y, uint s)     { return extract_bits(x, s & 15u, 8u) ^ y; }
inline uint op_ctz(uint x, uint y, uint s)         { return (x ^ y) + ctz(x | 1u); }
inline uint op_clz(uint x, uint y, uint s)         { return (x ^ y) + clz(x | 1u); }
inline uint op_popc(uint x, uint y, uint s)        { return (x ^ y) + popcount(x); }
inline uint op_imul(uint x, uint y, uint s)        { return (x * y) ^ s; }
inline uint op_mulhi(uint x, uint y, uint s)       { return mulhi(x, y) ^ s; }
inline uint op_mod33(uint x, uint y, uint s)       { return (x % 33u) + y + s; }
inline uint op_dec33(uint x, uint y, uint s)       { uint t = x + y + s; return t == 0u ? 32u : t - 1u; }
inline uint op_min(uint x, uint y, uint s)         { return min(x, y) + s; }
inline uint op_u16(uint x, uint y, uint s)         { return uint(ushort(x) + ushort(y)) ^ s; }

ALU(k_iadd, op_iadd)
ALU(k_shl_dyn, op_shl_dyn)
ALU(k_shr_dyn, op_shr_dyn)
ALU(k_shr_const, op_shr_const)
ALU(k_and16, op_and16)
ALU(k_funnel_naga, op_funnel_naga)
ALU(k_funnel_c, op_funnel_c)
ALU(k_funnel_64, op_funnel_64)
ALU(k_extract, op_extract)
ALU(k_ctz, op_ctz)
ALU(k_clz, op_clz)
ALU(k_popc, op_popc)
ALU(k_imul, op_imul)
ALU(k_mulhi, op_mulhi)
ALU(k_mod33, op_mod33)
ALU(k_dec33, op_dec33)
ALU(k_min, op_min)
ALU(k_u16, op_u16)

// ---------- Part B ----------
constant uint BS = 65536u;
constant uint CAP = 64u;
constant uint DEPTH = 32u;
constant uint NONE = 0xFFFFFFFFu;

inline uint funnel(uint lo, uint hi, uint sh) { return (lo >> sh) | ((hi << (31u - sh)) << 1u); }
inline uint ld_packed(device const uint* d, uint b, uint off) {
  uint w = b * (BS / 4u) + (off >> 2u); return funnel(d[w], d[w + 1u], (off & 3u) * 8u);
}
inline uint ld_exp(device const uint* d, uint b, uint off) { return d[b * BS + off]; }

#define K2(NAME, LD) \\
kernel void NAME(device const uint* data [[buffer(0)]], device const uint* pred [[buffer(1)]], \\
                 device uint* best [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) { \\
  uint p = gid.x, b = gid.y, o = b * BS + p; \\
  if (p >= BS - 8u) { best[o] = 0u; return; } \\
  uint mx = min(BS - p, CAP); \\
  uint p0 = LD(data, b, p), p1 = LD(data, b, p + 4u); \\
  uint bl = 0u, bq = 0u, q = pred[o]; \\
  for (uint dd = 0u; dd < DEPTH && q != NONE; dd++) { \\
    uint qn = pred[b * BS + q]; \\
    uint n; uint x = p0 ^ LD(data, b, q); \\
    if (x != 0u) { n = ctz(x) >> 3u; } \\
    else { x = p1 ^ LD(data, b, q + 4u); \\
      if (x != 0u) { n = 4u + (ctz(x) >> 3u); } \\
      else { n = 8u; \\
        while (n < mx) { x = LD(data, b, p + n) ^ LD(data, b, q + n); uint left = mx - n; \\
          if (left < 4u) { x &= (1u << (left * 8u)) - 1u; } \\
          if (x != 0u) { n += ctz(x) >> 3u; break; } n += 4u; } \\
        n = min(n, mx); } } \\
    if (n > bl) { bl = n; bq = q; if (n == mx) { break; } } \\
    q = qn; \\
  } \\
  best[o] = (bl << 16u) | (p - bq); }

K2(k2_packed, ld_packed)
K2(k2_expanded, ld_exp)
"""

guard let dev = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
print("device: \(dev.name)")
let queue = dev.makeCommandQueue()!
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: nil) } catch { fatalError("compile: \(error)") }
func pipe(_ n: String) -> MTLComputePipelineState {
  let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!)
  return p
}
func run(_ p: MTLComputePipelineState, _ bufs: [MTLBuffer], _ grid: MTLSize, _ tg: Int) -> Double {
  let cb = queue.makeCommandBuffer()!
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(p)
  for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
  e.dispatchThreads(grid, threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
  e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  return cb.gpuEndTime - cb.gpuStartTime
}

// ---------- Part A ----------
let nThreads = 1 << 20
var seed: UInt32 = 12345
func rnd() -> UInt32 { seed = seed &* 1664525 &+ 1013904223; return seed }
let inA = dev.makeBuffer(bytes: (0..<1024).map { _ in rnd() }, length: 4096, options: .storageModeShared)!
let outA = dev.makeBuffer(length: nThreads * 4, options: .storageModeShared)!
let ops = ["iadd", "shl_dyn", "shr_dyn", "shr_const", "and16", "funnel_naga", "funnel_c", "funnel_64", "extract",
           "ctz", "clz", "popc", "imul", "mulhi", "mod33", "dec33", "min", "u16"]
let pipesA = ops.map { pipe("k_" + $0) }
for p in pipesA { _ = run(p, [inA, outA], MTLSize(width: nThreads, height: 1, depth: 1), 256) } // warm-up
var bestA = [Double](repeating: 1e9, count: ops.count)
for _ in 0..<5 { for (i, p) in pipesA.enumerated() {
  bestA[i] = min(bestA[i], run(p, [inA, outA], MTLSize(width: nThreads, height: 1, depth: 1), 256)) } }
let opsPerRun = Double(nThreads) * 2048.0 * 8.0
print("\nPart A: lane-ops/s (each op incl. one xor/add); ratio = time / time(iadd)")
for (i, n) in ops.enumerated() {
  print(String(format: "  %-12@ %8.1f Gop/s  x%5.2f", n as NSString, opsPerRun / bestA[i] / 1e9, bestA[i] / bestA[0]))
}

// ---------- Part B ----------
let nBlk = 512, BS = 65536
var bytes = [UInt8](repeating: 0, count: nBlk * BS + 64)
for b in 0..<nBlk {
  var pos = 0
  let base = b * BS
  while pos < BS {
    let r = rnd()
    if pos < 64 || r % 4 != 0 {
      let n = Int(r >> 8) % 8 + 1
      for _ in 0..<n where pos < BS { bytes[base + pos] = UInt8(truncatingIfNeeded: (rnd() >> 16) % 32 + 65); pos += 1 }
    } else {
      let len = Int(r >> 8) % 37 + 4
      let off = Int(r >> 16) % min(pos, 4096) + 1
      for _ in 0..<len where pos < BS { bytes[base + pos] = bytes[base + pos - off]; pos += 1 }
    }
  }
}
func le32(_ i: Int) -> UInt32 {
  let b0 = UInt32(bytes[i]), b1 = UInt32(bytes[i + 1]) << 8
  let b2 = UInt32(bytes[i + 2]) << 16, b3 = UInt32(bytes[i + 3]) << 24
  return b0 | b1 | b2 | b3
}
let nWords = nBlk * BS / 4 + 4
var packed = [UInt32](repeating: 0, count: nWords)
for w in 0..<(nBlk * BS / 4) { packed[w] = le32(4 * w) }
var expanded = [UInt32](repeating: 0, count: nBlk * BS + 8)
for i in 0..<(nBlk * BS) { expanded[i] = le32(i) }
var pred = [UInt32](repeating: 0xFFFF_FFFF, count: nBlk * BS)
var head = [UInt32](repeating: 0xFFFF_FFFF, count: 1 << 16)
for b in 0..<nBlk {
  for i in 0..<head.count { head[i] = 0xFFFF_FFFF }
  for p in 0..<(BS - 8) {
    let h = Int((le32(b * BS + p) &* 2654435761) >> 16)
    pred[b * BS + p] = head[h]; head[h] = UInt32(p)
  }
}
let bPacked = dev.makeBuffer(bytes: packed, length: packed.count * 4, options: .storageModeShared)!
let bExp = dev.makeBuffer(bytes: expanded, length: expanded.count * 4, options: .storageModeShared)!
let bPred = dev.makeBuffer(bytes: pred, length: pred.count * 4, options: .storageModeShared)!
let out1 = dev.makeBuffer(length: nBlk * BS * 4, options: .storageModeShared)!
let out2 = dev.makeBuffer(length: nBlk * BS * 4, options: .storageModeShared)!
let kp = pipe("k2_packed"), ke = pipe("k2_expanded")
let grid = MTLSize(width: BS, height: nBlk, depth: 1)
_ = run(kp, [bPacked, bPred, out1], grid, 256); _ = run(ke, [bExp, bPred, out2], grid, 256)
var tp = 1e9, te = 1e9
for _ in 0..<5 {
  tp = min(tp, run(kp, [bPacked, bPred, out1], grid, 256))
  te = min(te, run(ke, [bExp, bPred, out2], grid, 256))
}
let same = memcmp(out1.contents(), out2.contents(), nBlk * BS * 4) == 0
print("\nPart B (\(nBlk) blocks of 64 KiB, depth 32, cap 64): outputs identical: \(same)")
print(String(format: "  packed   %7.2f us/block", tp / Double(nBlk) * 1e6))
print(String(format: "  expanded %7.2f us/block  (x%.3f of packed)", te / Double(nBlk) * 1e6, te / tp))
```

The MSL part was syntax-checked with clang++ against C++ stubs. The Swift part has **not been compiled** (there's no Swift toolchain here). Fix any trivial syntax errors on the Mac.

### `a12-k3slot.patch` (byte-identical; neutral on the 5090)

```diff
--- i/crates/gzc-gpu/src/shaders/k3_opt.wgsl
+++ w/crates/gzc-gpu/src/shaders/k3_opt.wgsl
@@ -837,9 +837,10 @@ fn dp(b: u32, k: u32) {
                             let v2 = mlen - 2u >= lo;
                             let v3 = mlen - 3u >= lo;
                             let s0 = slot(c0 + mlen);
-                            let s1 = slot(c0 + mlen - 1u);
-                            let s2 = slot(c0 + mlen - 2u);
-                            let s3 = slot(c0 + mlen - 3u);
+                            // One modulo per step (a12-metal): slot(x - 1) from slot(x).
+                            let s1 = select(s0 - 1u, RING_N - 1u, s0 == 0u);
+                            let s2 = select(s1 - 1u, RING_N - 1u, s1 == 0u);
+                            let s3 = select(s2 - 1u, RING_N - 1u, s2 == 0u);
```

### `a12-env.patch`: what it adds (full file at `artifacts/a12-metal/a12-env.patch`, 120 lines)

- `context.rs`:
  - `compile_opts()`: `zero_initialize_workgroup_memory = !GZC_NO_WG_ZERO`. It's used by all 5 `create_compute_pipeline` sites (chains, sorted, compressor, pipeline).
  - In `Prepared::context()`: if `GZC_METAL_SG32` is set, the backend is Metal and subgroups are on, set `adapter_info.subgroup_min_size = subgroup_max_size = 32`. That routes K1, sorted K1 and K3coop through their existing subgroup paths, self-tests and lane probe. K1 also gets `.x` ballots (narrow masks).
  - `GZC_DUMP_WGSL=<dir>`: dumps every module's final WGSL with its check policy, for naga→MSL inspection.
- `k3opt.rs`: `GZC_K3_TRUSTED=1` builds K3opt with `shader_trusted` (no clamps; no safety argument).

Defaults are unchanged. On the 5090 (Vulkan), with the patch built: `k3opt`, `opt_pipeline` and `differential` pass, also with `GZC_NO_WG_ZERO=1` and `GZC_K3_TRUSTED=1`.

## 7. Recommendation

1. **Don't build a native Metal backend.**
   - The only native-only lever with a plausible gain is 16-bit integer registers in K3 (3–8 %, estimated).
   - Unified-memory zero-copy is already reachable through wgpu (`MAPPABLE_PRIMARY_BUFFERS`, which maps to `StorageModeShared`) and is worth < 1–2 %.
   - Everything else (ICBs, heaps, imageblocks, simdgroup_matrix, MTLIO, Metal 4) doesn't apply to 15 large integer dispatches.
   - A second shader language would also double the oracle-equality surface.
2. **Do the cheap Metal-specific fixes in the wgpu path, in this order:**
   - **M1:** trust SIMD 32 on Metal. This is really a bug fix; the self-tests are the guard.
   - **M4:** the K3 slot change. It's portable and exact.
   - **M2:** a K1 groups default for Apple.
   - **M5:** no zero-init, after an audit.
   - **M3:** the shift-free byte view, Metal-only, only if applebench Part B shows ≤ 0.8.
   - All of this is S–M effort, and all of it keeps output byte-identical.
3. **Apple Silicon as a GPU target.**
   - On the M4 Pro, 10 CPU threads beat the GPU 10× at lvl3 and 2.7× at lvl9. At opt16 they're at parity (137 vs 141 MB/s).
   - Even +25 % only lifts the GPU to about 175 MB/s.
   - The GPU is about 1/14 of a 5090, has quarter-rate shift/mul units and high memory latency, while the CPU is strong.
   - **Treat Apple as a correctness target, not a speed target.**
   - On a Mac, the sensible product choice is one of:
     - CPU libzstd for every preset;
     - for opt16 only, CPU L16 on N−2 cores *alongside* GPU opt16 on a deterministic block split. That's an estimated 1.7–2× either one alone; plan step 4 measures it.
   - The split satisfies the ratio gate block by block, since both encoders are ≥ L16. It breaks "whole stream byte-identical to the GPU oracle", so it needs a product decision.

## 8. Dead ends checked

- The naga MSL for K2 is clean: no clamps, no div/mod, no loop bounds. Its slowness is hardware, not translation.
- `simd_ballot` is native.
- Function constants and `maxTotalThreadsPerThreadgroup`: already covered.
- Dispatch overhead, ICBs, heaps, imageblocks, `simdgroup_matrix`, MTLIO, Metal 4: no lever.
- Divergent-code SIMD ops: the cooperative variants were already measured slower on NVIDIA (a01/a02).
- The K3 slot change on NVIDIA: neutral (no gain, no loss).

## Sources

- Philip Turner, metal-benchmarks (Apple GPU microarchitecture; instruction throughput tables): https://github.com/philipturner/metal-benchmarks
- Apple, "Explore GPU advancements in M3 and A17 Pro" (Dynamic Caching, flexible on-chip memory, parallel FP32/FP16/INT): https://developer.apple.com/videos/play/tech-talks/111375/
- Chips and Cheese, "A Brief Look at Apple's M2 Pro iGPU" (8 KB L1, 3 MB L2, SLC 234 ns, DRAM > 342 ns, global atomics 58.6 ns): https://chipsandcheese.com/p/a-brief-look-at-apples-m2-pro-igpu
- Chips and Cheese, "iGPU Cache Setups Compared, Including M1": https://chipsandcheese.com/p/igpu-cache-setups-compared-including-m1
- Apple Metal docs: `MTLComputePipelineState.threadExecutionWidth`; Metal Shading Language Specification (SIMD-group functions, 16-bit integer types).
- wgpu-hal 30.0.1:
  - `src/metal/device.rs` (naga options 185–245, buffer storage modes 465–478);
  - `src/metal/mod.rs:445-446` (subgroup 4..=64);
  - `src/vulkan/adapter.rs:2838-2846` (buffer bounds unchecked with robust access).

Scratch (`/tmp/claude-1000/m6/a12-metal`, worktree, `~/.cache/gzc-a12`) has been removed. The artifacts kept are in `.superpowers/m6-research/artifacts/a12-metal/`: `a12-env.patch`, `a12-k3slot.patch` and `applebench.swift`.
