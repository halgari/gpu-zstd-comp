# a02: GPU hardware features and intrinsics for opt16 (K3opt first)

Date: 2026-09-30. Research only: no repository file was changed. Scratch is in `/tmp/claude-1000/m6/a02-gpu-features/`:
- kernel variants `v_*.wgsl`, with diffs against `base.wgsl` in `v_*.diff`;
- the host hooks in `host-hooks.diff`:
  - `GZC_K3SRC` / `GZC_PRE` override the K3 source and prepend an `enable` line;
  - `GZC_I16` requests `Features::SHADER_I16`;
  - `GZC_DUMP` dumps the composed WGSL;
  - `GZC_WG` sets the workgroup size;
  - test `a02_ab` is an interleaved A/B harness with a parse/hist exactness check;
- `ab.sh` (timing, with contention logging) and `stats.sh` (registers and shared memory from the NVIDIA
  driver, through `vkstats`);
- `cuda/props` (CUDA occupancy limits).

The worktree has been removed.

## 1. What the measurements say about the bottleneck (it decides which features can matter)

All runs below were on the RTX 5090 with opt16 on 2900 corpus blocks (every 34th block), unless noted. The GPU
was under contention: 5–8 GPU processes, load average 45–140. Each A/B is interleaved (A B A B …, 3 timed reps
per round, 3–5 rounds). Every ratio is the median of the per-round totals. I discarded rounds that another
agent's run overlapped (visible as one round at 1.4–1.8×).

| probe | result (ratio to baseline in the same session) |
|---|---|
| A/A (repo kernel vs a copy) | 0.990–1.000: noise is about ±2 % |
| literal-histogram atomics removed (`e1`; pass 1 has identical input) | pass 1 −2.7 %. Shared-atomic histograms are about 0.2 µs per pass |
| no matches found (`e2_nomatch`: prologue + epilogue + 4088 trips of loads) | 0.73 µs per block for the four passes, so about 98 % of the time is in the match/relax path |
| batch scaling (packed variant): 1450 / 2900 / 4000 blocks | 17.4 / 22.5 / 30.4 ms per pass |

- Above about 17 blocks per SM the pass is throughput-bound: 2900 → 4000 blocks is +38 % blocks for +35 % time.
- Below that it is bound by the latency tail. The heaviest block's chain is about 13 ms, a fixed floor.
- The CUDA query on the 5090 gives:
  - max **24 blocks/SM**;
  - 48 warps/SM;
  - 100 KB shared/SM;
  - 64 K registers/SM.
- Ampere GA10x and Turing allow 16 blocks/SM; Ada (4060) allows 24.
- With wg16, every block is one warp with only 16 of its 32 lanes active. That wastes half of the issue slots and half of
  the register file, and the per-SM block cap binds early.

The consequence: features that shorten individual ALU operations or memory latency give little. What pays off is
(a) fewer instructions and registers per trip, and (b) more blocks per wave. Wave quantization is large, because a
second partial wave costs a whole heavy-block chain.

## 2. Ideas, ranked

### I1. Register diet by packing and rematerialization (portable u32 WGSL). Measured; exact.

- **Mechanism**
  - The 10 record registers `m_ob[5]` + `m_len[5]` become `m_rec[5] = ob | len << 17`. Lengths are at most 4096, and
    ob is at most 65538.
  - The six lane bases (`base, cbase, tbase, wbase, sbase, lbase`) and `iend`/`ilimit` are derived from one
    register, `gsg = b*NSEG + k`.
  - `pb` is constant-folded when BPW = 1.
  - `ll_inc1` and `ll_p0` are re-read from shared memory at use.
  - `st_rep` is packed into two words, and the `get_all_matches` reps are passed packed.
  - The epilogue's `b, k, valid` are recomputed from `gsg`, so they are not live across the DP. That one change
    alone was worth 3 registers.
- **Registers** (NVIDIA pipeline statistics): L0 kernels 101 → **86**, L2 kernel 99 → **81**.
  - Packing the Node struct alone (`v_pack`) did nothing (102/96): the compiler already does that.
  - The m_rec + lane-constant step is what moved the count.
- **Speed** (`v_regs5`, under contention): **0.922–0.937 at 2900 blocks** (one wave in both; L0 passes −8 %, L2 −3.5 %),
  and 0.916–0.925 at 4000 blocks.
- **Exactness:** byte-identical. 1000/1000 corpus blocks gave equal final parses and equal cheap-pass histograms against
  the repo kernel.
- **Effort and risk:** S, low.
- **Why it helps at one wave:** fewer instructions and register moves per trip in a throughput-bound loop. Residency
  is not the reason at 2900 blocks, since both versions fit.

### I2. u16 price tables (shared 4520 → 4192 B), giving 23–24 blocks/SM. Measured; exact.

With I1's 86 registers, occupancy is set by shared memory: 102400 / 4520 = 22 workgroups per SM.

Packing the LL, LL-by-litlen, ML and OF tables as u16 brings it to 4192 B, which allows 24 by shared memory and 23 by
registers. One wave then holds about 3900 blocks, against about 3230 for the repo kernel (19 by registers).

Two implementations, both measured:
- **`v_i16`:** `enable wgpu_int16;` with `var<workgroup> p_ml: array<u16, …>`.
  - **wgpu 30 does expose i16/u16:** `Features::SHADER_I16`, native only, on Vulkan with shaderInt16, on Metal always,
    and on DX12 with SM 6.2.
  - The 5090 has it.
- **`v_pk16`:** portable u32 pairs.
  - The prologue stages the four tables in the then-dead `hist` words, then packs them into `p_tab[83]`.
  - Each lookup is a shift and a mask.

| batch | v_regs5 | v_i16 | v_pk16 |
|---|---:|---:|---:|
| 2900 (one wave for all) | 0.929 | 0.930 | 0.941 |
| 3586 (`--batch max`; the repo kernel spills into 2 waves) | – | – | **0.634** (4/5 rounds 0.63–0.66) |
| 4000 | 0.916 | **0.667** | **0.675** |

- **Exactness:** byte-identical.
  - `v_pk16` with the repo's gate tests through `GZC_K3SRC` (`k3opt_matches_opt_cases`,
    `k3opt_matches_oracle_synthetic`, `k3opt_passes_*`, `k3opt_ring_choice`): all pass.
  - `k3opt_passes_corpus`: 4000 corpus blocks equal to the CPU oracle, with the Prior seed ×3 + final, histograms
    included, plus the Buffer-price configurations at L0 and L2.
  - `v_i16`: 1000/1000 equal to the repo kernel.
- **What it buys:**
  - On the 5090, `--batch max` (b3586) runs as one wave: about −37 % K3 time at that batch, against the 1090 MB/s it
    measures today. A one-wave batch of about 3900 is then possible.
  - On an 8 GB card the batch is many waves anyway. There the gain is I1's instruction saving (about 6–8 %) plus
    finer wave quantization:
    - 4060: 24 SMs × 23 = 552 blocks/wave instead of 456, assuming that one Ada SM matches one 5090 SM;
    - Turing/Ampere: the 16 blocks/SM cap binds first (§3), so there is no residency gain.
- **Effort and risk:** S, low.
  - Prices are already < 65536, and `p_lit` is already packed this way.
  - The portable form costs about 1 % against native u16 at one wave. A hybrid that keeps `p_ml` (the hot one in the
    relax loop) as i32 and packs the other three saves 264 B, which still reaches 24 by shared memory.

### I3. Choosing the workgroup shape per architecture (lane utilization). Estimated; one 5090 data point.

- wg16 = one block per 32-wide warp or wave.
- On Turing/Ampere (16 blocks/SM) the cap is 16 resident blocks whatever the register or shared-memory diet.
  - wg32 lifts the cap to 32 blocks.
  - Shared memory then binds at about 22 blocks with I2, about +40 % residency.
- On the 5090, wg32 is worse:
  - measured 53.3 vs 47.1 µs/blk at 4000 unsorted;
  - the perf study's 4000 sorted run favoured wg32 by 16 %.
- **Recommendation:** benchmark wg16 vs wg32 on a 3060 or 1660 before choosing. It is a K3OptConfig knob, not a code
  change.

### I4. Modulo-free ring slots (integer throughput). Measured; dead end.

- `v_slot` replaces the `% 33` in the relax loop and the trip with an add and conditional wrap.
- Result: 0.997 (4/4 rounds 0.996–0.998), exact on 600 blocks.
- IMAD.HI-based modulo is not on the bottleneck.

## 3. Feature survey: exposure in wgpu 30 and fit to this DP

| feature | wgpu 30 / naga 30 | fit for K3opt |
|---|---|---|
| subgroup shuffle / broadcast / ballot / min / max / add (`Features::SUBGROUP`) | yes, compute; `SUBGROUP_BARRIER` separate | **No for the DP**: the relax loop and get_all_matches are lane-divergent, and shuffles from inactive lanes are undefined. Cooperative relaxation measured 1.7× slower (perf study). A `subgroupMin` + index tie-break is not needed: relaxation targets are disjoint per step. Possible only in the uniform prologue sums (`hsum` atomics → `subgroupAdd`), which are under 1 % |
| quad ops | yes | no use |
| `dot4U8Packed` / `dot4I8Packed` | yes; SPIR-V `OpUDot` 4x8-packed when `VK_KHR_shader_integer_dot_product` is on (wgpu-hal enables it), else emulated | none: `match_len` is equality (XOR + ctz is already minimal), and prices are table lookups. The 16-byte `match_len` measured +10 % (perf study) |
| `pack4xU8`, `unpack4xU8`, `extractBits`, `insertBits`, `countLeadingZeros`, `firstTrailingBit` | yes (core WGSL) | already used where it matters (ctz in `match_len`, `firstLeadingBit` in prices). `extractBits` is an alternative spelling of I1/I2 unpacking, with no extra speed |
| i16/u16 (`SHADER_I16`, `enable wgpu_int16;`) | **yes** (native only) | I2: u16 workgroup tables work. NVIDIA has no 16-bit integer registers, so u16 private variables do not cut registers; RDNA3 may differ (packed 16-bit VGPR halves), but that is untested |
| f16 (`SHADER_F16`) | yes | none (integer DP) |
| `workgroupUniformLoad` | yes | none (no workgroup-uniform values gate control flow) |
| shared-memory atomics | yes | histogram atomics are 2.7 % of pass 1 (measured upper bound for any subgroup-aggregated histogram) |
| subgroup size control / required size | **not exposed**. wgpu-hal sets `ALLOW_VARYING_SUBGROUP_SIZE` on every compute stage once `SUBGROUP` is on; there is no `requiredSubgroupSize` and no full-subgroups flag | NVIDIA is fixed at 32. **Risk on AMD and Intel:** the driver may pick wave64 (RDNA) or SIMD8/32 (Arc) for a wg16 kernel. Wave64 would idle 3/4 of each wave. Check on an RX 6600/7600 and an A750; only raw Vulkan can force it |
| `VK_KHR_shader_clock` | not exposed (no naga builtin; wgpu-hal does not enable the extension) | profiling only. Passthrough SPIR-V cannot use it legally either, because the extension is not enabled |
| `PASSTHROUGH_SHADERS` (SPIR-V/MSL/DXIL) | yes on Vulkan/DX12/Metal | escape hatch for hand-written SPIR-V, but it does not add pipeline flags (subgroup size) or extensions |
| cooperative matrix | yes, 8×8 f32 only | no fit (see the RT/NPU no-go) |
| int64 / int64 atomics | yes | none |

**Architectures:**
- **NVIDIA Turing → Blackwell:** 32-wide warps.
  - Blocks/SM cap: 16 on Turing and Ampere, 24 on Ada and Blackwell consumer.
  - 64 K registers/SM.
  - Shared memory: 64 KB on Turing, 100 KB on Ampere, Ada and sm_120.
  - Independent thread scheduling (Volta+) does not remove the union-of-branches cost here. The compiler reconverges
    at structured merges (BSSY/BSYNC), and SPIR-V is structured.
- **AMD RDNA2/3:** wave32 is native.
  - LDS is 64 KB per CU, which is about 15 K3 workgroups at 4.2 KB, so LDS binds before VGPRs (86 → about 11
    waves/SIMD).
  - I2 helps (14 → 15).
  - The wave32/64 choice is the main unknown.
- **Intel Arc:** with 86 live 32-bit values, SIMD16 would need 5.5 KB of GRF per thread, against the 4 KB default.
  The IGC will likely pick SIMD8 or large-GRF mode, so I1 matters more there. Unmeasured.
- **Apple M:** 32-wide SIMD-groups, so wg16 is half empty as on NVIDIA. u16 is always available. M3+ Dynamic Caching
  softens register cliffs. The SIMD-group matrix and permute ops do not fit, for the same divergence reasons as the
  subgroups.

**CUDA/HIP for this DP:** a port would not pay off. The relevant levers are all available in WGSL:
- **Register cap:** I1 does by hand what `__launch_bounds__` / `maxrregcount` would force, and without spills.
- **Warp-synchronous code:** no use, because the hot path is divergent.
- **PTX `vmin`/`vadd` SIMD-in-word:** emulated since Maxwell.
- **DPX:** hardware on sm_90 only, emulated on sm_120.
- **`__nanosleep`:** irrelevant.
- **L2 persistence:** the 5090 allows 60 MB, but the per-wave working set (candidate words 512 KB/block plus trace)
  is gigabytes, streamed once.
- **What CUDA alone would add:** a required warp size on AMD (HIP wave32) and `ld.global.nc` for the candidate words,
  which are bound `read_write` because the final pass overwrites them. Both are small next to the cost of a second
  code path.

## 4. Dead ends checked

- Modulo-free slots: −0.3 %.
- Node-struct packing alone: 0 %, and +1 register.
- Subgroup shuffles or broadcasts of tables in divergent code: invalid.
- `dot4U8Packed` for compares: not applicable.
- `workgroupUniformLoad`: nothing to apply it to.
- Shader clock and required subgroup size: not reachable in wgpu 30.
- CUDA-only intrinsics: see above.
- wg32 on the 5090 without sorting: +13 % time.
- A hang worth knowing about: `e2_nodp`, which forces every segment to skip its DP (`st_ip = ilimit`), hangs the GPU.
  I did not investigate. It never occurs in real parses.

## 5. Top-3 recommendation

1. **Adopt I1 + I2 together (S, byte-identical).**
   - Expected gains:
     - about −6–8 % K3 time at today's batch 2900 on the 5090;
     - −33–37 % at batch 3586–4000, where the repo kernel spills into a second wave;
     - about −6–8 % per block in steady state on 8 GB cards (I1), plus finer wave quantization.
   - Use the portable `v_pk16` form, or `v_i16` behind `SHADER_I16` with `v_pk16` as the fallback.
   - Gate: `vkstats` registers ≤ 88 and shared ≤ 4266 B.
   - Then size `--batch max` to registers and shared memory (about 3900 on the 5090).
2. **Measure wave shape on the real targets before choosing defaults:**
   - wg16 vs wg32 on Turing/Ampere, where the 16 blocks/SM cap binds;
   - the wave32/64 choice on RDNA under `ALLOW_VARYING_SUBGROUP_SIZE` (driver statistics or timing);
   - SIMD width on Arc.
   - Size batches to multiples of the resulting wave capacity.
3. **Skip subgroup, dot-product, CUDA/HIP and clock work for K3opt.** Spend the effort on the instruction count of
   the relax path and on the heavy-block tail (a01/a07 angles). That is where the remaining time is.
