# M5 feasibility: RT cores, tensor cores and NPUs for LZ77 match finding / optimal parsing

Date: 2026-09-30. Read-only research; no GPU runs. Numbers taken from papers are read from
their text or charts and are marked as such. Citation status: **[V]** = title/authors/venue/year
verified against the paper PDF or the Semantic Scholar record; **[P]** = partially verified (see
note); **[U]** = not verified here.

## 1. Verdicts

| Technology | Verdict | One-line reason |
|---|---|---|
| RT cores (VK_KHR_ray_query / DXR / OptiX) for match finding | **No-go** | Acceleration-structure build alone costs ~100–400 µs per 64 KiB block on 4090/4060-class hardware vs ~5 µs for K1+K2 today; hardware "closest hit" gives one candidate per ray at ~1 G rays/s (4090), our K2 already resolves ~30 G positions/s. LCP is not expressible without a suffix sort, and once you have a suffix sort you do not need the BVH. |
| RT cores for the optimal parser's candidate set | **No-go** (research-only at best) | Same build-cost wall; candidate enumeration is "any-hit" iteration, which RTIndeX shows is the expensive part; ordering by LCP would need the suffix-rank encoding in §4.3. |
| Tensor cores for all-pairs / windowed match finding | **No-go** | Exact-equality GEMM over a 64 KiB block needs ~1,800 TOPS at 7 GB/s (4060 int8 peak ~120 TOPS), only proves 4-byte equality (no length), and produces an n×n result no one can afford to read back. |
| Tensor cores for a learned price model (optimal parser) | **Research-only** | A price table for a 64 KiB block is ~120 numbers; estimating it is not on the critical path (K3 is latency-bound, not price-bound). Also unreachable from wgpu 30 on NVIDIA/AMD (8×8 f32 only). |
| NPUs (Intel NPU 3720, AMD XDNA) | **No-go** | Absent from Ryzen 9000 desktops; present only in APUs (Ryzen 8000G) and Arrow Lake-S. Intel exposes compiled graphs only (Level Zero + NPU compiler); AMD IRON allows custom kernels but on Ubuntu/XRT, not from Vulkan/wgpu. 13–16 TOPS over system memory cannot host a 7 GB/s random-access match finder. |

**Single most promising experiment** (salvaging the "binary tree over suffixes" intuition on
compute cores, not RT cores): a K2 variant that, per block, radix-sorts (8-byte prefix key,
position) pairs and takes, for every position, its nearest earlier-position neighbours in sorted
order (previous/next-smaller-position over the rank array). This yields the two LCP-maximal
candidates per position with no chain walk and extends naturally to k candidates for the optimal
parser. Details in §4.4. Go/no-go threshold: sort + neighbour scan ≤ K1+K2 today (≈ 26 ms per
2,559-block batch on the 5090) at equal or better lvl9 ratio.

## 2. What the literature actually says (verified)

### RT cores repurposed for search/index workloads

| Work | Status | What it shows that matters to us |
|---|---|---|
| Henneberg & Schuhknecht, *RTIndeX: Exploiting Hardware-Accelerated GPU Raytracing for Database Indexing*, PVLDB 16(13):4268–4281, 2023 (arXiv:2303.01139) | **[P]** authors/title/year from the PDF; PVLDB volume/pages from the vldb.org listing (`vol16/p4268-schuhknecht.pdf`); DOI 10.14778/3625054.3625063 did not resolve via Semantic Scholar | Keys become triangles/AABBs on a line, lookups are rays. float32 forces splitting 64-bit keys into 3 coordinates (23-bit exact chunks). RTX 4090: building a BVH over 2^26 keys takes ~100 ms (Fig. 7b/10c, chart read) and "the BVH creation is significantly more expensive than the build phase for the other indexes"; 2^27 point lookups over 2^26 keys take ~100–140 ms (Fig. 12, chart read) ≈ 1 G lookups/s. Hash table (WarpCore) wins point lookups; RX only wins on high-miss / high-skew workloads. Least-squares split of 2^27 rays: 103 ms traversal vs 36 ms intersection. Updates: "full rebuild". Conclusion: use read-only. |
| Doraiswamy & Haritsa, *Raster is Faster: Rethinking Ray Tracing in Database Indexing*, CIDR 2026 | **[V]** from the PDF header (CIDR'26, Jan 18–21 2026, Chaminade) | Rasterisation-based RasterScan builds its index ~50× faster than RTScan and searches >10× faster at low selectivity on an RTX 4090; attributes the RT loss to BVH build and traversal overhead. |
| Lv, Zhang, Wang, Zhang, Lee, He, Jing, Wang, *RTScan: Efficient Scan with Ray Tracing Cores*, PVLDB 17, 2024 (DOI 10.14778/3648160.3648183) | **[V]** Semantic Scholar | Conjunctive predicates as 3-D ray jobs; 1.2–4.7× over GPU scans (review's Table 3). |
| Shi, Zhang, Wang, Zhang, Lee, *RayDB: Building Databases with Ray Tracing Cores*, PVLDB 2025 (DOI 10.14778/3772181.3772185) | **[V]** Semantic Scholar (ACM page returned 403; abstract not read) | Follow-on to RTScan. |
| Zhu, *RTNN: Accelerating Neighbor Search Using Hardware Ray Tracing*, PPoPP 2022 (DOI 10.1145/3503221.3508409) | **[V]** | Fixed-radius/kNN as sphere-vs-ray; 2.2–65× over GPU NN libraries. Query scheduling and partitioning needed to keep RT cores busy. |
| Nagarajan, Mandarapu, Kulkarni, *RT-kNNS Unbound*, ICS 2023 (DOI 10.1145/3577193.3593738) | **[V]** | Unrestricted kNN on RT cores. |
| Nagarajan & Kulkarni, *RT-DBSCAN*, IPDPS 2023 | **[V]** | Clustering via radius search. |
| Wald, Usher, Morrical, Lediaev, Pascucci, *RTX Beyond Ray Tracing: Exploring the Use of Hardware Ray Tracing Cores for Tet-Mesh Point Location*, HPG 2019 (DOI 10.2312/hpg.20191189) | **[V]** | The original "RT cores as a spatial index" paper; point location. |
| Meneses, Navarro, Ferrada, Quezada, *Accelerating Range Minimum Queries with Ray Tracing Cores* (RTXRMQ), arXiv:2306.03282, journal version in Future Generation Computer Systems 2024 (ScienceDirect S0167739X24001110) | **[P]** preprint verified from PDF; journal volume/pages not verified | Elements become triangles shaped by value/position; closest hit = RMQ answer. Only wins for small ranges: up to 5× vs CPU HRMQ, 2.3× vs GPU LCA; loses to LCA at medium/large ranges. Needs a blocked two-level geometry and float mantissa tricks. RTX 6000 Ada. |
| Xiao, Xiao, Yuan, Yu, Lee, Zhang, *A Case Study for Ray Tracing Cores: Performance Insights with BFS and Triangle Counting in Graphs*, Proc. ACM Meas. Anal. Comput. Syst. (POMACS) 9(2), 2025 (DOI 10.1145/3727108) | **[V]** | The most relevant negative result: RT BFS is 3.7–3.9× slower than CUDA; RT set intersection wins only at high skew; "RT cores are more efficient at searching for elements, but this comes with a constant and non-trivial overhead of the execution pipeline"; "the overhead of BVH construction is smaller than sorting on CUDA cores only in the small-n range"; "binary search on CUDA cores is a more efficient alternative to BVH traversal on RT cores". Primitives cost an int32 + nine float32 each. |
| Meneses, Navarro, Ferrada, Verichev, Salazar-Concha, *Ray Tracing Cores for General-Purpose Computing: A Literature Review*, arXiv:2603.28771 (Jan 2026), FGCS per Semantic Scholar | **[V]** | 35 papers / 32 problems. Winners: nearest-neighbour variants and problems where geometry prunes work. Table 3 worst-case speedups: Binary Search 0.5×, BFS 0.4×, RMQ 0.2×, Point Queries (RTIndeX) 0.2×. **No paper on string matching, suffix structures or compression** (grep of the review text for string/suffix/compress: none). |
| Kim, Lee, Kim, *RT-HDIST*, Computer Graphics Forum 2025; Bai, Chen, Wahib, *RT-RkNN*, PVLDB 2026; Yang et al., *StreamingRT*, CIKM 2025 | **[V]** | Further geometry/NN work; not applicable. |

I searched specifically for RT cores applied to string matching, LZ77, suffix arrays or
compression and found nothing (two searches, plus the review's full problem list). That matches
the review's characterisation: RT cores pay off when a low-dimensional geometric predicate prunes
a large candidate set. Byte-string equality beyond the first ~3 bytes (24-bit float exactness) is
not such a predicate.

### Tensor cores for non-GEMM work

| Work | Status | Relevance |
|---|---|---|
| Dakkak, Li, Xiong, Gelado, Hwu, *Accelerating Reduction and Scan Using Tensor Core Units*, ICS 2019 (DOI 10.1145/3330345.3331057) | **[V]** | Reduction/scan as small GEMMs; 89–98 % of memcpy bandwidth. Shows tensor cores can do non-ML work, but only bandwidth-bound linear algebra. Not our bottleneck (K1/K2/K3 are latency/sector bound). |
| Tensor-core Smith-Waterman / edit distance / string matching | **[U]** — four searches found no such paper | GPU alignment work (WFA-GPU, QuickEd, eWFA-GPU, CUDASW++) uses CUDA/SIMD cores, not tensor cores. One phylogenetics paper uses tensor cores for likelihood matrices, which is genuinely a matrix product. I found no verified tensor-core formulation of exact substring matching. |

### Learned cost models for optimal parsing

Search found no verified work on learned/neural price models for LZ-style optimal parsing;
the literature is either classical bit-optimal parsing (Ferragina, Nitto, Venturini, *Bit-Optimal
Lempel-Ziv compression*, arXiv:0802.0835 **[U]** venue) or fully neural compressors (DeepZip,
L3TC, cmix/nncp) that are orders of magnitude too slow for us. Treat this as an open, low-value
question for our pipeline: zstd's optimal parser already estimates prices from statistics of the
previous block / a first pass, which is cheap.

## 3. What our stack exposes

Checked in `~/.cargo/registry/src/*/wgpu-30.0.1`, `wgpu-types-30.0.1`, `wgpu-hal-30.0.1`,
`naga-30.0.1`:

- `Features::EXPERIMENTAL_RAY_QUERY` (bit 32) and `EXPERIMENTAL_RAY_HIT_VERTEX_RETURN` exist.
  The feature doc says Vulkan-only, but `wgpu-hal/src/dx12/adapter.rs:617` also sets it on DXR
  tier 1.1 + SM 6.5 and `metal/adapter.rs:1326` on Metal ray-tracing hardware. Vulkan requires
  `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_deferred_host_operations`,
  `VK_KHR_buffer_device_address`.
- API: `Device::create_blas` / `create_tlas`, `CommandEncoder::build_acceleration_structures`
  (takes an iterator of BLAS builds, so 2.5K builds can go in one command). BLAS geometry can be
  triangles or **AABBs** (`BlasAabbGeometry`, 24-byte min stride). TLAS instances carry a 3×4
  transform and a custom index.
- WGSL (naga 30 lowering): `rayQueryInitialize`, `rayQueryProceed`, `rayQueryGenerateIntersection`,
  `rayQueryConfirmIntersection`, `rayQueryTerminate`, `rayQueryGetCommittedIntersection`,
  `rayQueryGetCandidateIntersection`. So AABB (procedural) candidates are iterable from a compute
  shader, which is the any-hit-style enumeration we would need for multiple candidates.
- `Features::EXPERIMENTAL_COOPERATIVE_MATRIX` (bit 57): **8×8 f32 only**, and the doc itself notes
  "Most Vulkan implementations (NVIDIA, AMD) primarily support f16 inputs at larger sizes (e.g.
  16x16), so Vulkan support may be limited." In practice tensor cores are not reachable from wgpu
  30 on our target GPUs; it would take raw Vulkan `VK_KHR_cooperative_matrix` (f16/int8 16×16×16)
  or CUDA/HIP.
- Hardware caveat: on RDNA3 (RX 7600) the ray accelerators do box/triangle intersection but BVH
  traversal runs in shader code (dedicated traversal hardware arrived with RDNA4) — widely
  reported, **[U]** not verified in this session. Any RT-core gain would be NVIDIA-only in the
  target class.
- Relevant repo facts: `HASH_BITS == 16` in `k1_chains_sg.wgsl` (64K-entry head table per work
  group, tag-validated); K2 walks `DEPTH ≤ 64` predecessors per chain with a 64-byte `SEARCH_CAP`;
  Dfast uses 8-byte and 5-byte hashes; positions are 17-bit. Floats can represent all of these
  exactly (< 2^24), so precision would not be the blocker, cost would be.

## 4. RT cores: concrete analysis for our workload

### 4.1 Rates we must hit

- Target: 7 GB/s at 64 KiB blocks = **~107K blocks/s = 7.0 G positions/s** to index and to query.
- Today on the 5090 (lvl9, b2559 = 2,559 blocks of 128 KiB = 335 MB per batch):
  K1 16.0 ms + K2 10.4 ms = 26.4 ms per batch → K1+K2 stream **12.7 GB/s** of input, i.e.
  **~5.2 µs per 64 KiB of input** (≈ 3.1 µs K1, 2.0 µs K2), with K2 resolving ~32 G positions/s
  at depth 32 thanks to fingerprint skips.
- A 4060 has 24 3rd-gen RT cores vs 128 on a 4090 and 170 4th-gen on the 5090 (RTIndeX Table 8,
  NVIDIA specs), and 272 GB/s vs 1,008 / 1,792 GB/s. Scale 4090 numbers by ~1/5 for the 4060.

### 4.2 Encoding A — hash-key chains as geometry (what the user sketched)

Primitive per position p: an AABB (or degenerate triangle) at x = key(p) (16-bit hash or the
first 3 bytes exactly), y = p, in one BLAS per block (64K primitives). Query for position p: a ray
from (key(p), p − ε) in −y with t_max = p (window); **closest hit** = most recent position with the
same key = the chain head; **iterating candidates** (`rayQueryProceed` loop over AABB candidates in
any order, or a sorted-by-t triangle closest-hit chain re-cast from each hit) = the chain walk.
LCP beyond the key is not expressible: the hardware answers "which primitives does this ray
touch", and the ray is defined by ≤ 3 float coordinates; equality of bytes 4..N has to be checked
on the shader cores exactly as K2 does now.

Cost:
- **Build.** RTIndeX on a 4090: 2^26 primitives in ~100 ms (chart) ≈ 0.67 G prims/s in a single
  large build, and AABBs are the fastest primitive type. That is **~98 µs per 64 KiB block** on a
  4090 and **~400–500 µs on a 4060**, i.e. 20–100× our K1 (3 µs per 64 KiB on the 5090), before
  counting the per-build fixed overhead that the POMACS case study calls "constant and
  non-trivial" (2,559 separate builds per batch), the scratch memory (RTIndeX Table 6: the BVH
  needs far more space during build than a hash table) and the fact that a BVH over 64K
  primitives is ~2.5 MB uncompacted (10 int32/float32 per primitive per the case study) vs our
  256 KiB pred array. At 0.67 G prims/s a 4090 would cap the whole compressor at ~0.67 GB/s from
  build alone; a 4060 at ~0.15 GB/s.
- **Query.** RTIndeX: ~1 G closest-hit lookups/s on a 4090 (2^27 lookups in 100–140 ms). One
  closest hit per position = a depth-1 chain (lvl3 quality). To match lvl9 we need up to 32
  candidates per position with byte compares in between; RTIndeX's decomposition (103 ms
  traversal + 36 ms intersection for 2^27 rays) says every additional candidate costs about a
  third of a traversal, so ~32 candidates ≈ 10 ray-equivalents. Expected: ≤ 0.1 G positions/s on a
  4090 → **~0.1 GB/s**, vs 32 G positions/s for K2 on the 5090 today.
- **Net.** Two to three orders of magnitude slower than K1+K2. The POMACS paper's direct
  measurement (RT binary search beats CUDA binary search only at large n, and BVH build costs more
  than a sort except at small n) matches: our n is 64K, and per block we do ~n queries against ~n
  keys, exactly the regime where a sorted array / hash table on shader cores wins.

### 4.3 Encoding B — suffix ranks (the only way to get LCP-ordered candidates out of rays)

If suffixes of the block are sorted (rank(p) for every p), the LCP-maximal earlier match of p is
one of its two nearest neighbours in rank order that have position < p ("previous/next smaller
value" over the rank→position array). That *is* a geometric query: primitive q = AABB spanning
x ∈ [rank(q) − ½, rank(q) + ½], y ∈ [pos(q), +∞); ray from (rank(p), p − ε) in +x hits, as its
closest hit, the smallest rank > rank(p) whose position is < p, and the −x ray gives the other
side. Two rays per position, closest-hit only, no candidate iteration. k candidates for the
optimal parser = the k nearest such hits per side (`rayQueryProceed` loop).

But: it requires the suffix sort first, which is the expensive part (a 64K-key radix sort of
(8-byte prefix, position) pairs per block, plus tie handling for prefixes longer than 8 bytes),
and once ranks exist the "previous smaller value" scan is an O(n) compute pass with a small
stack, or an O(n log n) segmented scan — far cheaper than building a BVH over the same 64K items
and shooting 128K rays. RTXRMQ, the closest published analogue (an RMQ-shaped query), only beats
the compute-core LCA approach for small query ranges and is 0.2× at worst (review Table 3). So
Encoding B is a correct reformulation but strictly dominated by doing the same thing on shader
cores.

### 4.4 The compute-core version of the user's intuition (recommended experiment)

Per block (64K positions): (1) radix-sort keys (64-bit prefix, 17-bit position) — for 64K
elements this is a workgroup-local sort, no global passes; (2) walk the sorted array: for each
rank r, the nearest r' < r and r'' > r with pos < pos(r) (segmented previous-smaller-value; a
32-lane subgroup scan with a per-lane stack handles it in a few passes) give the two candidates
with the longest common prefix among all earlier positions (exact for LCP ≤ 8 bytes, best-effort
beyond); (3) K2's capped compare extends them. This is the suffix-array LZ77 idea (Crochemore,
Ilie, Smyth; the `Parallel-LZ77` GPU repo) specialised to a 64K window, and it needs no chain
depth parameter: lvl9-class quality at depth-2 work. Widening to k neighbours per side gives the
optimal parser its candidate list in LCP order. Its CPU oracle is easy to write in `gzc-core`.
Risks: sort cost at 2,559 blocks per batch; ratio parity with the depth-32 chain walk on DDS data
(where many equal 8-byte prefixes make ties common). This is the only item here I would spend
GPU time on.

## 5. Tensor cores: concrete analysis

### 5.1 All-pairs / windowed GEMM for match finding

Equality of k bytes between positions p and q can be turned into a dot product only through
sum-of-squared-differences (a² + b² − 2ab, k MACs per pair) or one-hot bytes (256k MACs per pair).
With k = 4 and the cheaper trick:
- Full block: n²/2 ≈ 2.1 G pairs × 4 MACs = 8.6 G MAC per 64 KiB; at 107K blocks/s that is
  9.2e14 MAC/s ≈ **1,840 TOPS**. RTX 4060 int8 dense tensor peak is ~120 TOPS (NVIDIA lists 242
  with sparsity; **[U]** marketing figure). 15× short at peak, and the result is a 64K×64K matrix
  (16 GB of int32 per block) from which the sparse matches would still have to be extracted.
- Windowed (W = 4,096): 64K × 4K × 4 = 1.07 G MAC per block → 230 TOPS at target rate, still
  above peak, and the 268M-entry result per block (1 GB) cannot be written, let alone reduced;
  and it caps offsets at 4 KiB, which would destroy the ratio on a 64 KiB window.
- Either way it proves 4-byte equality only; length, tie-breaks and LCP still need the byte
  compare that K2 does now. Hash chains obtain the same candidate set in O(n·depth).

Verdict: no-go, by two orders of magnitude on arithmetic and by more on output bandwidth.

### 5.2 Tensor cores for the optimal parser's prices

Zstd's optimal parser prices literals, literal lengths, match lengths and offset codes from
symbol statistics (≈ 256 + 36 + 53 + 32 numbers per block). Estimating or "learning" that table is
microseconds of work; K3's cost is the latency-bound sequential parse (31 ms per batch, ~2 µs per
sequence), not price evaluation. A learned model would only matter if it changed the parse
structure (e.g. predicting where to cut), and nothing verified in the literature does that for
LZ77. Research-only, and not reachable from wgpu 30 anyway (8×8 f32 cooperative matrices only,
"support may be limited" on NVIDIA/AMD Vulkan).

## 6. NPUs

Availability on the target class (gaming desktop with a discrete GPU):
- AMD Ryzen 9000 desktop CPUs have **no NPU** (AEC Magazine / Phoronix Zen 5 coverage). Only the
  Ryzen 8000G APUs (Phoenix, XDNA 1, ~16 TOPS per AMD marketing, **[U]** page fetch timed out)
  and mobile/mini-PC Ryzen AI parts (XDNA 2, 50 TOPS) have one.
- Intel Arrow Lake-S desktop (Core Ultra 200S) has NPU 3720, 13 TOPS (Wikipedia; Linux `ivpu`
  driver adds PCI IDs, Phoronix).
- So a typical "RTX 4060 / RX 7600" buyer has no NPU at all, or a 13–16 TOPS one.

Programmability:
- Intel: `intel/linux-npu-driver` docs: user API is oneAPI Level Zero with the graph extension;
  models must be compiled through the NPU compiler from OpenVINO IR / ONNX. **No custom
  kernels.** Supported: Meteor Lake, Arrow Lake, Lunar Lake, Panther Lake, Wildcat Lake.
- AMD: `amd/IRON` (mlir-aie) is a close-to-metal Python/MLIR toolchain with C++ AIE compute
  kernels loaded through XRT; custom kernels **are** possible, but the supported path is Ubuntu
  24.04/24.10 plus the `amdxdna` kernel driver (Linux 6.14+), Windows via Ryzen AI SW. Nothing
  reaches it from Vulkan/wgpu; Rust would go through XRT's C API.
- Performance shape: AIE tiles stream data through DMA object FIFOs with tens of KB of local
  memory and system-memory bandwidth; a hash-chain match finder is random access over a 64 KiB
  block with 64K dependent lookups. Even the 50-TOPS parts are the wrong shape, and at 7 GB/s
  the NPU would need to ingest data faster than DDR5 dual-channel can feed it alongside the GPU.

Verdict: no-go (not present on the target, not reachable from our stack, wrong execution
model).

## 7. Sources

Verified papers (see status column above): arXiv:2303.01139 (PDF read); CIDR 2026 p18 PDF (read);
arXiv:2306.03282 (PDF read); arXiv:2603.28771 (PDF read); POMACS 9(2) art. 16 preprint TR-25-2
(PDF read); Semantic Scholar records for DOIs 10.14778/3648160.3648183, 10.14778/3772181.3772185,
10.1145/3503221.3508409, 10.1145/3577193.3593738, 10.1145/3727108, 10.1109/IPDPS54959.2023.00100,
10.1111/cgf.70229, 10.1145/3330345.3331057, 10.2312/hpg.20191189, 10.1145/3746252.3761409.
Vendor/driver docs: `intel/linux-npu-driver/docs/overview.md`; `amd/IRON` README; NVIDIA OptiX
forum thread 251639 (qualitative: AABB builds are the fastest/smallest primitive type, no
numbers); Wikipedia Arrow Lake (NPU 3720, 13 TOPS). Local: wgpu/naga 30.0.1 sources listed in §3;
`crates/gzc-gpu/src/shaders/k1_chains_sg.wgsl`, `k2_best.wgsl`, `crates/gzc-core/src/params.rs`,
`config.rs`, `docs/results/2026-09-30-speed.md`.
