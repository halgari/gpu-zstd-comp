# r6-exotic-hardware: other GPU units, and different methods, for the L16 pipeline

Date: 2026-10-01. Research only; the repository was not changed. Scratch is in `~/.cache/gzc-m7/r6-exotic-hardware/`
and the important files are copied to `.superpowers/m7-research/artifacts/r6-exotic-hardware/`:
- `r6.rs`: the CPU experiment (a gzc-core example run in a copied workspace);
- `probe/`: `features.rs` (wgpu 30 adapter features and cooperative-matrix configurations), `coop.rs` (WGSL tensor-core
  correctness and throughput), `rt.rs` (wgpu ray-query BLAS build and lookup);
- `cuda/intr.cu` with `runs/sass_intrinsics.txt` (SASS for DPX and SIMD intrinsics on sm_89, sm_90 and sm_120);
- `runs/*.txt`: every ratio and probe log.

**Conditions.** The ratios come from the CPU oracle (`opt::parse` + `write_frame`) on the 1/50 sample (2016 blocks,
`every 50 offset 0`). The 16 KiB runs use a `block-16k` build on 1990 blocks (`every 200`). libzstd ran on the same
blocks. Sample baselines at 64 KiB: **L16 1.37175, L14 1.36911**. At 16 KiB: **L16 1.32969, L14 1.32809**. libzstd decoded
every frame from the diagonal-candidate variant: 0 failures in 4032 frames. The GPU probes ran on the shared 5090, with
load averages of 35–47 and another agent's GPU jobs running, so I give their absolute numbers as ranges.

## Short answer to the user's question

- **Tensor cores, RT cores and DPX don't give a faster way to produce the same zstd output.** For matching, the problem is
  byte equality. It costs one XOR and a zero-byte test per 4 bytes on the ALUs. As a dot product it costs at least 8 MACs per
  byte. The tensor core's rate advantage, which I measured at 4.5× from WGSL on the 5090, does not cover that 32× increase in
  work. The DP is a sequential, path-dependent min-plus recurrence. MMA computes multiply-add, and none of the known min-plus
  tricks keep the 1/256-bit price resolution the parse depends on.
- **Correlation analysis did point at a new candidate source, and it is cheap.** It finds candidate offsets for the
  whole block, then scans those offsets' diagonals for match runs. It produces standard zstd and adds +0.02…+0.09 % ratio,
  depending on the base recipe. But the tensor-core or FFT part adds nothing: a histogram of the offsets K2 already found
  gives the same offset set for free. What is worth prototyping is an ordinary integer-ALU kernel.
- **Two corrections to M5:**
  - **wgpu 30 does reach NVIDIA tensor cores from WGSL** (f16 16×16×16; measured on the 5090).
  - The RT-core BLAS build cost is now **measured on our stack**: 109–231 µs per 64 KiB block on the 5090, which confirms
    M5's estimate from the literature.

## 0. What the stack exposes (checked in source and on the 5090, driver 610.57)

| unit | wgpu 30 / WGSL | measured on the 5090 | raw Vulkan / CUDA needed for |
|---|---|---|---|
| Tensor cores | `Features::EXPERIMENTAL_COOPERATIVE_MATRIX`: **true**. wgpu-hal keeps the Vulkan `VK_KHR_cooperative_matrix` configurations with subgroup scope, sizes 8/16 and A/B/C types f16/f32/i32/u32. The 5090 reports **16×16×16 f16·f16→f16 and →f32**, plus 16×8×16 and 16×8×8. naga 30 only has `coop_mat8x8` / `coop_mat16x16` with **f16 or f32 elements** (`valid/type.rs`: `MatrixElementNotFloat` otherwise). M5's "8×8 f32 only, unreachable" was the feature doc comment, not what the code does. | `coop.rs`: a 16×16 product is exact for both accumulator types (0/256 wrong). Throughput: **~500 TFLOPS with f16 accumulation and ~230–250 with f32**, against **111 TFLOPS of f32 FMA on the ALUs** in the same session (5 rounds each). | **int8/int4 MMA** (naga and wgpu-hal reject the 8-bit component types); the 16×8 shapes; element access to a fragment (WGSL can only `coopStore` it to memory) |
| RT cores | `EXPERIMENTAL_RAY_QUERY`: true, and `EXPERIMENTAL_RAY_TRACING_PIPELINES`: true. WGSL `ray_query` works in compute shaders, and AABB candidates can be iterated with `rayQueryProceed` / `rayQueryGenerateIntersection`. `BlasAabbGeometry.primitive_offset` is a **byte** offset. | `rt.rs` (§2) | Shader Execution Reordering; Mega Geometry cluster acceleration structures |
| DP4A | `dot4U8Packed` / `dot4I8Packed` | SASS `IDP.4A` | – |
| Subgroups | `SUBGROUP` true, `subgroupMin` etc. | `__reduce_min_sync` compiles to `REDUX.MIN` (sm_89/sm_120). The WGSL→SPIR-V→SASS lowering was not inspected. | `__match_any_sync` (`MATCH.ANY`): there is no WGSL partition op, and wgpu does not expose `VK_NV_shader_subgroup_partitioned` |
| Optical flow / NVENC / decompression engine / L2 compression | none | – | VK_NV_optical_flow, the Video Codec SDK, nvCOMP, or CUDA `cuMemCreate` compressible memory |
| Thread-block clusters, DSMEM, TMA | none | – | CUDA, and sm_90/sm_120 only (not Ada) |

## 1. Tensor cores for match finding

### 1a. Correlation of the whole block to get an offset set (FFT or MMA), then verify on the diagonals. **Measured.**

**Mechanism.**
- Map each position's hashed 4-byte key to a random unit phasor. One complex 128K-point FFT, |F|², and an inverse FFT
  give score(d), approximately the number of positions p whose 4 bytes equal those at p−d, plus noise with σ≈180.
- Take the top-K offsets D.
- For each d in D, compute the run length of `block[p..] == block[p−d..]`. That is a per-diagonal suffix scan, fully
  coalesced, and needs no chains.
- Merge into the (A = nearest ≥3, B = longest, ties nearer) candidate words that the DP already consumes.

I compared three ways of choosing D:
- **fft**: the correlation above;
- **cand**: top-K of a length-weighted histogram of the B offsets K2 already found;
- **oracle**: top-K of the offsets the final opt16 parse actually used, as an upper bound.

**Coverage of opt16's matched bytes by the top-K offsets** (46.4 % of input bytes are matched; 23.8 % of matched bytes reuse
one of the previous 3 offsets):

| K | fft | cand (free) | oracle |
|---:|---:|---:|---:|
| 8 | 16.8 % | 23.7 % | 25.7 % |
| 32 | 23.2 % | 33.5 % | 35.7 % |
| 128 | 30.8 % | 48.3 % | 50.7 % |

Offsets are not concentrated. Even the oracle's best 128 offsets cover only half of the matched bytes, so a correlation can't
**replace** hash chains.

**Ratios (64 KiB sample; L16 = 1.37175):**

| variant | ratio | vs L16 | vs its base |
|---|---:|---:|---:|
| opt16 base (h4 d32 + h3 d4) | 1.37211 | +0.026 % | – |
| opt16 base + fft64 diagonals | 1.37247 | +0.052 % | +0.026 % |
| opt16 base + cand64 diagonals | 1.37245 | +0.051 % | +0.025 % |
| opt16 base + oracle64 (upper bound) | 1.37249 | +0.054 % | +0.028 % |
| opt16, h4 d4 only (shallow chains) | 1.37075 | −0.073 % | −0.099 % |
| opt16, h4 d4 + fft32 diagonals | 1.37131 | −0.032 % | +0.041 % |
| opt16, h4 d4 + cand32 diagonals | 1.37129 | −0.034 % | +0.039 % |
| opt16, **fft64 diagonals only (no chains)** | **1.27073** | **−7.36 %** | – |
| opt14 base | 1.37118 | −0.042 % | – |
| opt14 base + cand128 | 1.37189 | +0.010 % | +0.052 % |
| opt14 base + cand256 | 1.37243 | +0.050 % | +0.091 % |
| opt14 a06 recipe (h4 d8 + h3 d4 + h10/4 d16) | 1.37302 | +0.093 % | – |
| opt14 a06 + cand64 / 128 / 256 / 512 | 1.37319 / 1.37331 / 1.37351 / **1.37382** | +0.105 / +0.114 / +0.128 / **+0.151 %** | +0.012 / +0.021 / +0.036 / **+0.058 %** |
| opt16 a06 → + cand128 | 1.37369 → 1.37398 | +0.141 → +0.163 % | +0.021 % |
| 1-pass a06 → + cand512 | 1.36898 → 1.36979 | −0.202 → −0.143 % | +0.059 % |
| opt14 a06 + offsets taken from the previous pass's parse, K=128 | 1.37334 | +0.116 % | +0.023 % |

At 16 KiB (L16 = 1.32969):
- opt14 base: 1.32951 (−0.014 %);
- opt14 base + cand128: 1.33024 (+0.041 %);
- opt14 base + cand256: 1.33072 (+0.077 %);
- a06: 1.33044 (+0.056 %);
- a06 + cand128: 1.33074 (+0.079 %);
- a06 + cand512: 1.33101 (+0.099 %).

**What this says:**
1. **The FFT/MMA correlation is never better than the free histogram:** 1.37247 against 1.37245. So the exotic unit has no
   job here.
2. Diagonal candidates are a **real, additive candidate source**:
   - on today's chains, +0.05–0.09 % (comparable to h10);
   - on top of the a06 h10 recipe, +0.02–0.06 %, so they mostly overlap with h10.
3. They do not remove a DP pass: a 1-pass schedule stays 0.14 % short.
4. Shallow chains plus diagonals do not match deep chains (−0.058 % against base), so this does not replace K1/K2.

**GPU cost (estimated).**
- **Correlation on tensor cores or FFT:**
  - A direct Toeplitz product over all offsets is n² = 4.3 G MAC per block, about 34 µs on the 5090 at the measured f16
    rate and about 0.3 ms on a 4060, for no gain.
  - An FFT is bandwidth-bound: about 8 MB of global traffic per block with f32 complex (4.4 µs on the 5090, 30 µs on a
    4060).
  - Either way it costs more than the free histogram it would replace.
- **Diagonal stage (integer ALU):**
  - One thread owns 32 positions and loops over the K offsets. For each offset it XORs 8 words with funnel-shifted words,
    does the exact zero-byte test, packs a 32-bit equality mask, forms run≥3 masks from neighbour words, and does a sparse
    ctz-based update of A/B for the set bits that beat the stored B.
  - That is about 60 ops per (32 positions × offset), or 2048 × K × 60 per block: K=128 ≈ 16 M ops, K=512 ≈ 63 M ops.
  - 5090 (~50 T int ops/s effective): **0.3 µs (K128) / 1.3 µs (K512) per block**, against K1+K2 at about 9.5 µs.
  - 4060 (24 SMs, ~7.5 T int ops/s): **2.1 / 8.4 µs**, against an estimated ~60 µs for K1+K2.
  - Plus the histogram and top-K: a 4096-entry shared hash table of B offsets and a bitonic top-K, about 1 µs on one SM and
    negligible when amortized.
  - Memory: it reads the 64 KiB block (from L1/shared) and rewrites the 512 KiB of candidate words once.
  - It adds no dependent chain and no DP work: the DP still sees 2 candidates per position.
- **Overall:** about +3 % (K128) to +13 % (K512) on K1+K2, roughly +1–5 % of the end-to-end opt time.

**Oracle.** Everything is deterministic integer work:
- The histogram counts are independent of insertion order, and top-K is a total order on (count desc, offset asc).
- Run lengths are capped at 64, the same "≥64" semantics as K2.
- The A/B merge rule is the one `find_cands` uses.

A `gzc-core::reference::diag_cands(block, cands, K)` would be about 60 lines (`merge_diag` in `r6.rs` is the model).

**Effort and risk.** Effort M: one new kernel between K2 and K3, an oracle function, and a K parameter. Risk is low for
correctness. On ratio, the risk is that the gain shrinks once h10, split blocks and the in-pass price refresh are stacked.
The gain is +0.02 % on the h10 recipe at K128 and needs K512 for +0.06 %.

**Verdict:**
- **Worth prototyping**, as a portable integer kernel and a cheap ratio lever: it buys +0.02–0.06 % for about +1–5 % of
  time, which can be traded for fewer DP work elsewhere.
- **No-go** for tensor cores or the FFT in it: no gain over the free histogram.
- **No-go** as a chain replacement: −7.4 % alone, −0.06 % with shallow chains.
- The zstd format is unchanged.

### 1b. All-pairs byte equality as an MMA (one-hot or bit-sliced). **No-go (arithmetic, plus readback).**

- **Making it exact.** A one-hot encoding over 256 symbols does not fit 16×16×16 tiles. Random phasors are not exact:
  with 256 symbol values, two phases can be 1e-4 rad apart, which f16 cannot resolve. The exact encoding is ±1 bit
  slices, 8 dims per byte, where dot = 8 − 2·hamming, so a byte is equal iff dot = 8. That is exact in f16.
- **Cost of a 4-byte equality.** It takes K=32 MACs per pair:
  - the full 64 KiB window is 2.1 G pairs × 32 = **69 G MAC per block**: 275 µs on the 5090 at the measured ~500 TFLOPS
    (29× K1+K2) and about 2.3 ms on a 4060 (~60 TFLOPS f16, from the spec);
  - a window of W=256 offsets is 537 M MAC, 2.1 µs on the 5090. The diagonal scan in 1a covers those same 256 offsets
    with about 4 M integer ops.
- **Readback.** WGSL cannot read fragment elements. Each 16×16 result must go through `coopStore` to workgroup memory,
  which is 2 B of shared traffic per pair. That is more than comparing the bytes directly.
- It proves equality of a fixed width only. Lengths still need the byte compare.
- It needs uniform control flow per subgroup. That is fine for a match finder, but irrelevant given the cost.

### 1c. Could correlation make better candidates so the DP needs fewer passes?

No:
- The 1-pass schedule gains +0.059 % at most (K512) and stays −0.14 % below L16.
- The pass count is set by price convergence, not by candidate quality (see the M6 synthesis and r2's angle).

## 2. RT cores for match finding. **Measured; no-go.**

**What I built (`rt.rs`, wgpu 30 ray queries in a compute shader).**
- Geometry: one BLAS per 64 KiB block, with 64K AABBs at (key(q) = 16-bit hash of 4 bytes, y = q), and a TLAS of 64
  blocks offset in z.
- Query: one ray per position from (key(p), p−½) in −y. A candidate loop checks the key exactly, because AABB candidates
  are conservative, and generates the hit at t = p−q.
- Answer: the closest hit is the nearest earlier same-key position, i.e. the hash-chain head (a depth-1 chain).

| quantity (5090, contended) | measured |
|---|---|
| BLAS build, PREFER_FAST_BUILD | **109–231 µs per 64 KiB block** (0.28–0.60 G prims/s) over 5 runs |
| BLAS build, PREFER_FAST_TRACE | 167–190 µs per block |
| TLAS build (64 instances) | 0.24–2.0 ms |
| Query, one closest-hit ray per position | **65–74 µs per block (0.9–1.0 G rays/s)**. Without the exact key check it runs at 7 G rays/s but returns wrong-key hits. |
| Exactness against the CPU chain head | 94 % equal. In 6 % of positions it returned a farther same-key position. The cause was not found and may be in the probe, so these quality numbers are not reliable. |

**Reading.**
- The build alone is 11–24× all of K1+K2 (9.5 µs). One depth-1 lookup is another 7–8×.
- The opt16 recipe walks up to 32+4 candidates with byte compares.
- A 4060 has 24 3rd-gen RT cores against 170 4th-gen, and 272 GB/s against 1.8 TB/s, so expect 5–7× slower.

This matches M5's literature estimate (~98 µs per block on a 4090). M5's further points still stand:
- LCP is not expressible as a ray.
- The suffix-rank encoding needs a suffix sort, which makes the BVH unnecessary.

**Verdict.** No-go. Ray queries are reachable in WGSL, but each BLAS rebuild per block costs 10–25× the whole current match
finder.

## 3. Tensor cores for the DP. **No-go (analysis).**

- **Log-semiring softmin.** Prices are integers in 1/256 bit.
  - Getting the soft-min within one unit of the hard min over m≈5–50 terms needs β ≥ ln m ≈ 4 per unit.
  - Competing path prices at one node differ by hundreds to thousands of units, so the terms span e^-4000. f16 (min
    normal e^-9.7) and f32 (e^-87) cannot hold that.
  - Rescaling per row needs the row minimum first, which is circular.
  - Measured parse margins are 0.03–0.15 %, i.e. 2–10 bits per 64 KiB. Any error that flips decisions costs more than that;
    the m6 rep-history approximations cost 0.1–0.4 %.
- **Bit-decomposed exact min-plus.** Comparing a sum against a threshold as a dot product needs unary (thermometer)
  encodings of size 2^bits. Path prices need about 23 bits per 4 KiB segment, so this cannot be done. I found no verified
  method for exact min-plus on tensor cores at this precision.
- **Structure.** Each node has about 5–15 in-edges, and the rep-offset state makes edge prices path-dependent. This is not a
  dense semiring product unless you approximate rep history, which is a measured dead end.
- **Batched price evaluation.** MMA can form outer sums exactly: [a_i, 1]·[1; b_j] = a_i + b_j, with K=2, exact in f16 up to
  2048 and in f32 up to 2^24. So 16×16 (length × candidate) price grids are computable. But:
  - the relax loop is lane-divergent (wg16, 15.2/16 lanes active, divergent relax), and `coopMultiplyAdd` needs uniform
    control flow per subgroup;
  - results must go through `coopStore` to shared memory before the min, which costs more than the 256 IADDs it replaces;
  - relax is 33 % of K3 and is limited by latency and divergence, not by arithmetic (a11's profile).

**Verdict.** No-go. Expected speedup ≤ 0; the format would be unchanged.

## 4. NVIDIA intrinsics (SASS checked with nvcc 13.3 for sm_89, sm_90 and sm_120)

| intrinsic | sm_89 (Ada, 4060) | sm_90 (Hopper) | sm_120 (5090) | WGSL equivalent |
|---|---|---|---|---|
| `__viaddmin_s32` (DPX) | IADD3 + IMNMX | **VIADDMNMX** (1 op) | IADD + VIMNMX | `min(a + b, c)`. **The same SASS as the intrinsic on sm_120 and sm_89** |
| `__vimax3_s32` (DPX) | 2× IMNMX | **VIMNMX3** | 2× VIMNMX | `max(max(a,b),c)`, with the same SASS |
| `__vimin3_u16x2` (DPX) | 11 ops (emulated) | VIMNMX3.U16 | 2× VIMNMX.U16x2 | none (packed u16 min). Not useful: DP prices are 16–30 bits and the relax is divergent |
| `__vcmpeq4`, `__vsub4` | 5–7 LOP3/IADD/PRMT | 5 | 5 | the XOR + zero-byte bit trick gives the same op count; no SIMD-in-word hardware since Maxwell |
| `__byte_perm` | PRMT | PRMT | PRMT | none directly (shifts / `extractBits`); `match_len` uses XOR + ctz, which needs no permute |
| `__funnelshift_r` | SHF | SHF | SHF | a shift/or pattern; a11 measured ≤ 1 % difference |
| `__popc`, `__clz`, `__ffs` | POPC, FLO | – | – | `countOneBits`, `countLeadingZeros`, `firstTrailingBit` |
| `__dp4a` | IDP.4A | – | IDP.4A | `dot4I8Packed` / `dot4U8Packed`. Not an equality op |
| `__reduce_min_sync` | REDUX.MIN | – | REDUX.MIN | `subgroupMin` |
| `__match_any_sync` | MATCH.ANY | – | MATCH.ANY | **none** |

**Answer.**
- DPX is hardware only on sm_90/sm_100. **Consumer Blackwell (sm_120) has no fused 3-input min/max or add-min:** it emits the
  same two instructions WGSL gets from `min`/`max`/`+`.
- For 4- and 8-byte match extension, XOR + `countTrailingZeros` is already minimal in WGSL. a11 measured the CUDA-only
  extras (funnelshift, `__syncwarp`, match/redux) at 0 ± 1 %.
- The only intrinsic with no WGSL equivalent that could matter is `__match_any_sync`. a11 measured it on histogram
  aggregation at 0.

**Verdict.** No-go. Speedup 0; format unchanged.

## 5. Other fixed-function units

| unit | could it help? | verdict |
|---|---|---|
| **NVOFA / NVENC motion estimation** | A 64 KiB block can be viewed as a 256×256 8-bit image, and BC1/BC3 rows repeat at offsets of dy·stride + dx. But the OFA/ME searches **between two frames** for an approximate-cost, sub-pixel 2D vector in a bounded window, giving one vector per 1×1 to 4×4 grid cell. We need **exact byte equality** at arbitrary 1D offsets up to 65535, with lengths. §1a also shows that a small offset set covers only 30–50 % of matched bytes, and the free histogram already supplies it. Throughput: one OFA per GPU, so at roughly 2–3 Gpix/s about 25 µs per block, more than K1+K2. It is reachable only through VK_NV_optical_flow (Ada+, not in wgpu) or the Optical Flow SDK. | **No-go**: byte-level mismatch, slower, not portable |
| **Texture units / read-only path** | Gathers with no interpolation give the same bytes as `ld.global.nc`. a11 measured `ld.global.nc` and prefetch on K3 as noise. No equality or min in the texture filter path is usable for bytes. | **No-go** |
| **L2 "compute data compression"** (Ampere+) | Transparent compression of sparse or zero-heavy buffers through CUDA `cuMemCreate` compressible allocations. Our hot buffers are texture data and dense candidate words. It is not exposed in Vulkan/wgpu, and it compresses nothing we emit. | **No-go** |
| **Decompression engine** (Blackwell datacenter B200: LZ4/Snappy/Deflate through nvCOMP) | Decompression only, not on GeForce parts (not verified for GB20x), and not zstd. | **No-go** |
| **nvJPEG hardware** (A100/H100) | Decode only, and JPEG. | **No-go** |
| **Raster/ROP depth test as a scatter-min** ("Raster is faster", CIDR'26) | The depth test computes min-per-pixel, i.e. min-plus scatter in fixed function. But the DP needs node i's final value before relaxing from i, and Bellman-Ford rounds would equal the sequence count (thousands). Shared-memory `atomicMin` already gives the same thing. | **No-go** |

## 6. Ada/Blackwell features

| feature | availability | what it would buy | verdict |
|---|---|---|---|
| Thread-block clusters + distributed shared memory | sm_90 and sm_120 (5090) only; **not Ada (4060)**; CUDA only | A cluster of about 6–8 SMs could hold a whole block's data (64 KiB), the h4/h3 head tables and the pred arrays (about 0.5 MB) on chip, so K1+K2 never touch L2. That attacks the "data or candidates in shared memory don't fit" dead end. But it is 5090-only, needs CUDA (a11: ~500 MiB context, a second byte-exact implementation), and K1+K2 is about 25 % of the projected 2-pass time. | Interesting, not now |
| TMA / `cp.async.bulk` | sm_90/sm_120, CUDA only | Bulk staging. a11 measured the equivalent prefetch as noise; there is no shared-memory room for K3. | No-go |
| Shader Execution Reordering | Ada+, ray-tracing pipelines only (wgpu exposes RT pipelines but not SER) | Would regroup threads by a key, but only in raygen; the DP's divergence is inside a lane's own chain. | No-go |
| Cooperative vectors (VK_NV_cooperative_vector, Blackwell) | not in wgpu | Per-thread matrix-vector products in divergent code; the DP has no matrix-vector product. | No-go |
| WGSL f16 cooperative matrices (new finding) | wgpu 30 on NVIDIA (5090 verified), probably RDNA3 WMMA (not checked), Apple simdgroup 8×8; not Turing GTX 16xx or RDNA2 | Available if some stage ever becomes a real GEMM. In this pipeline none does (§1, §3). | Note for the record |

## 7. Dead ends checked

- **Correlation peaks (FFT/MMA) as the offset source:** never better than the free K2 histogram (1.37247 vs 1.37245), and
  coverage is lower (31 % vs 48 % at K128).
- **Correlation-only match finding (no chains):** −7.4 %.
- **Shallow chains (h4 d4) plus diagonals instead of deep chains:** −0.058 %.
- **Diagonals to enable a 1-pass DP:** still −0.143 % below L16.
- **Exact all-pairs equality as an MMA:** 69 G MAC per block, 29× K1+K2 on the 5090 at peak.
- **RT-core BLAS per block:** 109–231 µs per block measured; depth-1 lookups 65–74 µs more.
- **Softmin or bit-decomposed min-plus on tensor cores:** dynamic range and precision make it impossible at 1/256-bit
  prices.
- **Tensor-core outer-sum price grids:** possible in arithmetic, but blocked by divergence and the readback cost.
- **DPX on sm_120:** no fused instructions; the SASS is identical to plain WGSL `min`/`max`/`+`.
- **`__vcmpeq4` and friends:** emulated everywhere.
- **OFA/NVENC, texture path, L2 compression, decompression and JPEG engines, ROP min, SER, cooperative vectors:** all
  covered in §5–6.

## Top 3

1. **Prototype a "diagonal candidate" kernel, K2.5 (integer ALU; the useful idea that the correlation analysis
   surfaced).**
   - Mechanism: top-K offsets from a histogram of K2's B offsets, then a per-diagonal run-length scan, merged into A/B.
   - Ratio, opt14 schedule:
     - on today's chains: +0.052 % (K128) / +0.091 % (K256);
     - on the a06 h10 recipe: +0.021 % (K128) / +0.058 % (K512), reaching +0.151 % over L16 at 64 KiB and +0.099 % at
       16 KiB.
   - Cost: an estimated 0.3–1.3 µs per block on the 5090 and 2–8 µs on a 4060 (about +3–13 % on K1+K2). No change to
     the DP. Exact oracle; standard zstd.
   - Use it as margin to buy away other work.
2. **Correct the M5 record.**
   - wgpu 30 reaches tensor cores from WGSL: f16 16×16×16, f16 or f32 accumulation, measured at ~500 / ~240 TFLOPS on the
     5090. There is no int8.
   - RT BLAS builds are measured at 109–231 µs per block.
   - DPX is not hardware on sm_120.

   None of them is a path to the same output faster. Future "use the tensor cores" proposals have to beat 32 MACs per
   4-byte equality against one XOR.
3. **Interesting, not now: CUDA clusters/DSMEM to keep a whole block's chains on chip (5090 only).** Revisit only if K1+K2
   becomes the bottleneck after the DP goes to 2 passes, and only if a CUDA path is accepted for other reasons.
