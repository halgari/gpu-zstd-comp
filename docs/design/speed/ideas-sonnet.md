# GPU zstd compressor: speed ideas (lvl9, targeting 4060/7600-class GPUs)

Grounded in `k1_chains.wgsl`, `k2_best.wgsl`, `k3_parse.wgsl`, `k3_lazy.wgsl`, `k4_seq_entropy.wgsl`,
`k5_huffman.wgsl`, `common.wgsl`, and the host code in `pipeline.rs` / `compressor.rs` / `chains.rs` /
`context.rs`. Baseline (RTX 5090, lvl9): K1 23.3 ms, K2 16.6 ms, K3 61.3 ms, K4 7.1 ms, K5 3.2 ms per
batch (108.5 ms of kernel time), 1611 MB/s end-to-end, ratio 1.3549.

---

## 1. K2: early-exit the depth loop once a match hits SEARCH_CAP

**Mechanism.** `k2_best.wgsl`'s depth loop walks `pred[pb+q]` for up to `DEPTH` (32 for lvl9) hops,
keeping the longest match (ties broken by larger `q`). `pred[p]` is defined (K1's own doc comment) as
"the most recent `q < p`" — i.e. every hop is to a strictly smaller position, so `q` decreases
monotonically with depth. Once a hop yields `len == SEARCH_CAP` (64), no later hop can score higher
(length is capped) and every later hop has a strictly smaller `q`, so it always loses the tie-break.
It is therefore safe to `break` out of the depth loop the moment `len == SEARCH_CAP`, with byte-identical
`best[]`. One-line change: `if (len == SEARCH_CAP) { break; }` right after the length compare, inside
both the `chain` and `d` loops in `k2_best.wgsl`.

**Expected gain.** K2 is 16.6 ms/batch, dominated by `DEPTH` dependent global loads (`pred[pb+q]`) each
followed by a capped `match_len`. The corpus is 95% DDS textures — highly repetitive (padding, repeated
mip rows/solid blocks) — so a 64-byte match is plausible within the first few chain hops for a large
fraction of positions. If the average realized depth drops from 32 to, say, 3-6 on this corpus, K2's
per-thread work drops 5-10x; conservatively expect K2 to fall from 16.6 ms to roughly 3-6 ms/batch
(saving ~10-13 ms/batch, ~10-12% of total kernel time). Worst case (no position ever reaches the cap)
this is a no-op — the change cannot regress correctness or runtime.

**Oracle.** Byte-identical; `best[]` selection is unchanged, just computed with less wasted work.

**Risk/complexity.** S. One-line change per loop, in a kernel already covered by
`stream_matches_reference_every_index_once` and `vram_matches_params`-style tests. Add a synth test case
with a long repeated run to specifically exercise depth > 1 with an early cap hit, and assert `best[]`
matches the CPU oracle bit-for-bit (already implied by the existing byte-identical pipeline tests).

**Dependencies/conflicts.** None; composes cleanly with idea 5 (K1/K2 fusion) — carry the same early
break into the fused search loop.

---

## 2. K3: skip re-comparing the first SEARCH_CAP bytes when extending a capped match

**Mechanism.** `search_max` (`k3_lazy.wgsl`) and the greedy extension in `k3_parse.wgsl` both call
`match_len(base, p, q, 0xFFFFFFFFu)` to extend a match whose K2-reported length is `SEARCH_CAP` (64).
`match_len` (`common.wgsl`) always restarts at `n = 0`, redundantly re-comparing the 64 bytes K2 already
verified equal. Add a `match_len_from(base, p, q, cap, start)` variant that begins its word loop at
`n = start` and add `start` to the returned length; call it with `start = SEARCH_CAP` from both
extension sites. Byte-identical result (same final length), fewer u32 loads/xors per extension.

**Expected gain.** Each extension call currently wastes 16 redundant u32-pair loads + xor + ctz
(64 bytes / 4). K3 is 61.3 ms/batch (55%) and the lazy2 parse calls `search_max` up to 3 times per
position considered (depth 0/1/2 lookahead), many of which hit the cap on this repetitive corpus. Fixed
per-call savings (~16 loads) across a very large call count plausibly recovers low-to-mid single-digit
milliseconds per batch (rough estimate: 5-10% of K3, i.e. 3-6 ms/batch) — smaller than idea 1's gain
because the *rest* of the extension (the actual long tail beyond 64 bytes) still dominates for very long
matches, but the win is real and unconditional.

**Oracle.** Byte-identical.

**Risk/complexity.** S. Pure refactor of `match_len` into a shared helper; test via the existing
byte-identical GPU-vs-CPU-oracle tests plus a synth case with matches > 64 bytes.

**Dependencies/conflicts.** Independent of idea 1 (different kernel, same SEARCH_CAP boundary
insight). Should be re-applied if idea 10 (parse restructuring) is ever pursued.

---

## 3. Pack `best[]` into one u32 per position instead of two

**Mechanism.** `best[p] = (offset, len)` is currently 8 bytes/position (`best_bytes`:
`n_blocks * BLOCK_SIZE * 8`, ~1 MiB/block). `offset` needs at most `log2(BLOCK_SIZE) = 17` bits
(128 KiB block) and `len` at most `log2(SEARCH_CAP) = 7` bits (64) — 24 bits total, fits one u32 with
room to spare (e.g. `len << 17 | offset`, with `len == 0` still meaning "no match" so the all-zero
sentinel is preserved). Change K2's write and K3/`k3_lazy`'s reads (`best[bbase+p*2]`/`+1`) to
pack/unpack through two small helper functions in `common.wgsl`.

**Expected gain.** Halves `best[]` VRAM (1 MiB → 0.5 MiB/block) and halves the memory traffic of K2's
write and K3's read of this buffer. K2 (16.6 ms) and K3 (61.3 ms) are latency/compute-heavy more than
raw-bandwidth-bound on the 5090's 1.8 TB/s, so the 5090 win is modest, but the 4060/7600 target has
6-7x less bandwidth (270-290 GB/s) and a much smaller L2 (32 MB vs 96 MB), so `best[]` is less likely to
stay L2-resident there — halving it measurably improves L2 hit rate for K3's repeated re-reads across a
batch. Also halves scratch VRAM (`scratch_bytes` currently ~2.4-2.9 MiB/block), which either frees
headroom for a larger batch/more inflight slots (idea 9) or a bigger `head[]`/subgroup budget (idea 6).

**Oracle.** Byte-identical (lossless repacking of the same two integers).

**Risk/complexity.** S/M. Touches three kernels' bind data but each read/write site is small and
mechanical; test with the existing pipeline byte-identical tests plus boundary values (offset at
`BLOCK_SIZE - 1`, len at `SEARCH_CAP`).

**Dependencies/conflicts.** Changes the `best[]` layout that idea 5 (K1/K2 fusion) would write and idea
2/10 would read — do this before or together with idea 5 so the fused kernel is written against the
packed layout from the start.

---

## 4. K1: avoid the per-batch `head[]` clear with a generation tag

**Mechanism.** `ChainsKernel::record_timed` does `enc.clear_buffer(head, 0, head_bytes(...))` every
batch (256 KiB/block for lvl9's single hash chain) so `head[h] == 0` means "empty". Instead, store
`head[h] = (generation << 18) | (pos + 1)` and pass the current batch's generation as a small uniform;
K1 treats any entry whose generation field doesn't match the current batch as empty (no different from
a fresh 0). This removes the `clear_buffer` call entirely (pure write bandwidth with zero compute); on
wraparound (generation field width e.g. 12-13 bits within a u32 alongside `HASH_BITS`-independent pos
bits) reset to 0 and clear once every ~4096+ batches, or simply clear only that rare fallback case.

**Expected gain.** `clear_buffer` for a 1638-block batch at lvl9 moves ~1638 * 256 KiB ≈ 410 MiB. At the
5090's 1.8 TB/s this is ~0.23 ms (negligible, consistent with K1's cost being dominated by the sort/tile
loop) — but at the 4060/7600's 270-290 GB/s the same clear costs ~1.4-1.5 ms, roughly 6% of K1's
proportional budget on that GPU specifically. Small in isolation, but "free" (no downside) and stacks
with every other K1 change.

**Oracle.** Byte-identical; this only changes how "empty" is represented internally. `pred[]` output is
unaffected.

**Risk/complexity.** S. Needs a small uniform/push-constant for the batch generation (or bake it as a
dynamic value the host passes per submission) and a compare in K1's head read; test that two consecutive
batches sharing the same `head`/`pred` scratch buffers (as slots already do via `new_sharing`) produce
identical `pred[]` to a version that clears every time.

**Dependencies/conflicts.** Independent; combine with idea 5/6 since all three touch K1.

---

## 5. Fuse K1 and K2 for single-hash presets (lvl9 is `Hashes::Single`)

**Mechanism.** K1 processes a block in 512 sequential 256-position tiles; by the time tile `t`
finishes, `pred[]` is fully resolved for every position < `t*256` (chain hops only ever point backward,
per K1's own invariant). That means K2's depth-32 search for position `p` needs nothing K1 hasn't
already finalized by the time `p`'s tile completes. For `N_HASHES == 1` presets (lvl9), fold K2's
per-position search into the same per-tile loop in `k1_chains.wgsl`, right after `pred_out[pbase+pos]`
is written for that tile: each thread walks the chain it just helped build and writes `best[]` directly,
in the same kernel/dispatch. `pred[]` still has to be materialized to global memory (later tiles' deeper
chain hops need it), so this isn't a full memory-traffic elimination, but it removes: (a) one kernel
launch + bind-group setup between K1 and K2, and (b) the cold read of `pred`/`data` in a freshly launched
K2 workgroup that had no chance to inherit anything from K1's caches. Combine with idea 1's early break —
most searches terminate within a couple of hops on this corpus, meaning most of K2's reads land on
positions from the current or immediately preceding tile, which are still warm in L1/L2 right after K1
wrote them.

**Expected gain.** K1 + K2 = 39.9 ms/batch combined (36%). Removing one kernel-launch/bind-group boundary
and getting L1/L2 warm-cache reuse for the (now typically short, thanks to idea 1) chain walk is hard to
bound precisely without profiling, but a 10-20% reduction of the *combined* K1+K2 time (roughly 4-8
ms/batch) is a reasonable expectation, larger on the 4060/7600's smaller L2 where the warm-cache effect
matters more.

**Oracle.** Byte-identical — same computation, same order of writes, just one dispatch instead of two.

**Risk/complexity.** M/L. Needs careful rewrite of `k1_chains.wgsl` to interleave the existing sort/tile
logic with K2's search loop and its own bind group (adds a `best` binding to K1's layout). `Hashes::Dfast`
presets (LVL3) need both chains fully built before a position's combined search, so this fusion should be
gated to `N_HASHES == 1` and K1/K2 kept as separate kernels for Dfast presets (or dispatched as two K1
passes then one K2 pass, unchanged). Test: byte-identical `best[]` against the unfused path across the
existing preset matrix, plus a timing comparison.

**Dependencies/conflicts.** Should land after idea 3 (packed `best[]` layout) so the fused kernel is
written once against the final layout, and after idea 1 (early break) so the fused loop already contains
it. Does not conflict with idea 4 or 6 (both are K1-internal, orthogonal phases: sort vs. per-tile search).

---

## 6. K1: subgroup-accelerated bitonic sort to cut barrier count

**Mechanism.** Each of K1's 512 tiles runs a full 256-wide bitonic sort: `log2(256)*(log2(256)+1)/2 = 36`
compare-exchange stages, each gated by a `workgroupBarrier()` — roughly 18,000+ barriers per workgroup
per block. Split the sort in two: first sort each 32-lane subgroup independently using
`subgroupShuffle`-based compare-exchange (the low `log2(32)*(log2(32)+1)/2 = 15` stages), which needs no
`workgroupBarrier()` at all since a subgroup is implicitly lockstep in hardware; then merge the 8
subgroup-sorted runs of 32 using the existing shared-memory bitonic network for the remaining ~21 stages
(only those touching `k > 32` in the `for (var k = 2u; k <= WG; k <<= 1u)` loop). This roughly halves the
barrier count of the sort phase (15 of 36 stages become barrier-free) without changing the sort's result
(same total order, same tie-break by lane index already baked into the key).

**Expected gain.** Barriers are pure synchronization latency, not throughput; their relative cost is
higher on a GPU with fewer resident warps to hide them behind (24-34 SMs vs. 170). Hard to bound tightly
without profiling, but cutting ~40% of the sort's barrier count in a kernel that's 21% of total time (23.3
ms) is plausibly worth low-single-digit milliseconds per batch, likely more pronounced on the 4060/7600.

**Oracle.** Byte-identical if the subgroup sort produces the same total order (it does — it's the same
comparator, just batched by hardware lockstep instead of explicit barriers).

**Risk/complexity.** M. Requires the `SUBGROUP` wgpu feature (gated per the hard constraints — the
context notes subgroup ops are available on Vulkan for this GPU class) and a non-subgroup fallback path
(keep the current full barrier-based sort) selected at pipeline-build time based on adapter support.
Test: compare sorted `keys[]` output (or downstream `pred[]`) bit-for-bit between the subgroup and
fallback paths on the same input.

**Dependencies/conflicts.** Orthogonal to idea 4 (clear avoidance) and idea 5 (fusion) — all three touch
K1 but different phases (sort vs. head/pred write vs. added search); safe to combine, higher total
engineering cost if done all at once, so consider landing sequentially.

---

## 7. K3: vectorize the literal-copy loop (`push_lits`)

**Mechanism.** `push_lits` in `k3_parse.wgsl` appends literal bytes one at a time via a private
`acc`/`acc_n` shift-and-flush accumulator, even for long literal runs that are word-aligned in `data`
relative to `lits`' packing. When `start` and `lit_w`'s implied bit offset align (both multiples of 4),
copy whole words directly via `load_u32_at`/`data[]` reads instead of four separate byte-shift-flush
steps; fall back to the byte loop only for the unaligned head/tail. This cuts per-literal-byte instruction
count roughly 4x for the aligned bulk (one store per 4 bytes instead of load+shift+or+compare+branch per
byte).

**Expected gain.** Depends on the literal fraction of the corpus; DDS textures compress well so literals
are likely a minority of bytes, but literal runs after incompressible regions (noise, high-frequency
detail) still occur. A conservative estimate: if literals are ~10-20% of block bytes and this cuts their
copy cost ~3-4x, that's roughly 2-5% of K3's 61.3 ms, i.e. 1-3 ms/batch — modest but nearly free to add
alongside idea 2 since both touch the same function/file.

**Oracle.** Byte-identical (same bytes, same final `lits[]`/`n_lit`, just written in bulk when aligned).

**Risk/complexity.** S/M. Care needed with the partial-word tail and the `acc_n`-carrying state across
calls (multiple `push_lits` calls per block, mid-word states must still compose correctly). Test with
literal runs of varying length and alignment against the CPU oracle.

**Dependencies/conflicts.** None; independent of every other idea, safe to land any time.

---

## 8. K3: reorder independent loads in the lazy2 lookahead for memory-level parallelism

**Mechanism.** In `lazy_parse`'s lookahead loop (`k3_lazy.wgsl`), `rep_len(base, ip, offset_1)` and
`search_max(base, bbase, ip)` are computed back-to-back but are data-independent (different memory
regions: `data[]` for the former, `best[]` then possibly `data[]` for the latter). Restructure the code
to issue both loads (the `rep_len` byte comparisons and the `best[]` read) before branching on either
result, giving the compiler/hardware more opportunity to overlap the two independent memory latencies
instead of a strictly serial load-use-load-use chain. This is a source-level reordering only, not an
algorithm change.

**Expected gain.** Speculative and likely small — WGSL compilers can already reorder independent loads
within a basic block in many cases, so this may be a no-op in practice. Worth a cheap, low-effort try
(rough estimate: 0-3% of K3, i.e. 0-2 ms/batch) but should be validated with a profiler (Nsight Compute)
rather than assumed; keep the change only if it measurably helps.

**Oracle.** Byte-identical (pure reordering of independent expressions).

**Risk/complexity.** S. Trivial to write and revert; test is the existing byte-identical suite plus a
before/after timing comparison — don't keep it without a measured win.

**Dependencies/conflicts.** None.

---

## 9. Host: tune batch size / inflight slot count for the 4060's VRAM and SM count

**Mechanism.** `PipelineConfig::{batch, inflight}` and the 11% measured host overhead (1611 vs. 1819
MB/s) suggest upload/readback/submission overhead isn't fully hidden behind kernel time. The 6144 MiB
VRAM budget is generous relative to the ~2.4-2.9 MiB/block scratch plus ~0.5 MiB/block/slot — there is
headroom to raise `inflight` from 3 toward 4-5 (more batches' uploads/readbacks in flight to overlap with
kernel execution) and/or tune `batch` size specifically for a 24-34 SM device (fewer SMs may saturate at
a smaller batch than the 5090 needs, freeing VRAM for more inflight slots instead). This is pure
host-side scheduling, no WGSL changes.

**Expected gain.** Directly closes some fraction of the 11% host-overhead gap; if fully hidden, that's a
~11% throughput gain on top of every kernel-side win above. This must be measured on real 4060/7600
hardware since it's a scheduling/overlap question, not a fixed compute cost.

**Oracle.** No effect on output whatsoever — pure scheduling.

**Risk/complexity.** S. Config-only change (`PipelineConfig` fields), test via existing pipeline tests
(batching/`inflight` combinations already covered) plus a wall-clock measurement sweep.

**Dependencies/conflicts.** None; independent, cheap, do any time — good first thing to sweep once other
changes land, since kernel-time reductions from ideas 1-8 shift the balance further toward host overhead
being the limiter.

---

## 10. (Higher risk) Restructure K3's parse for intra-block parallelism

**Mechanism.** K3's lazy2 parse is single-thread-per-block by necessity (rep-offset state and lazy
deferral are sequential). A more invasive idea: split each block into a handful of segments (e.g. 4-8),
run a speculative parse of each segment in parallel (different subgroup lanes or separate dispatches),
each assuming a plausible rep-offset seed, then a cheap serial reconciliation pass fixes up sequence
boundaries and rep-offset history where segments disagree with what a true single-pass parse would have
produced. This is the only idea here that fundamentally breaks K3's serial-per-block structure instead of
speeding up what each serial thread does.

**Expected gain.** Potentially large (K3 is 55% of total time) if it works, but the reconciliation pass's
cost and correctness are unproven without prototyping, and a naive version very likely changes which
sequences get emitted (different local optima near segment boundaries), i.e. **not** byte-identical to
the current oracle.

**Oracle.** Requires a new CPU oracle and a new preset (per the hard constraints: "if an idea changes the
algorithm... the CPU oracle changes in lockstep and a new preset is defined"), plus revalidation that
ratio stays ≥ libzstd L9 (1.3532) on the full corpus — not guaranteed, since segment-boundary artifacts
in LZ parsing typically cost a small amount of ratio.

**Risk/complexity.** L. This is a research spike, not a drop-in change: new oracle in `gzc-core`, new
WGSL, new correctness tests, and a ratio re-run against the corpus before it can be trusted. Only pursue
after ideas 1-9 are exhausted and still leave K3 as the dominant cost.

**Dependencies/conflicts.** Supersedes/conflicts with ideas 2, 7, 8 (all K3-internal tweaks to the
*current* serial algorithm) — if this is pursued, those three would need to be reapplied to the new
structure rather than assumed to still apply.

---

## Recommended implementation order

1. **Idea 1** (K2 early-exit on SEARCH_CAP) — S, largest single expected win, zero risk, do first.
2. **Idea 4** (head[] generation tag) — S, independent, easy parallel-track win.
3. **Idea 2** (skip redundant 64-byte recompare in K3 extension) — S, independent of 1, do alongside.
4. **Idea 7** (vectorize K3 literal copy) — S/M, same file as idea 2, bundle together.
5. **Idea 3** (pack best[] to one u32) — S/M, do before idea 5 so the fused kernel targets the final
   layout; also shrinks scratch VRAM, helping idea 9's batch/inflight tuning.
6. **Idea 9** (batch/inflight tuning) — S, cheap, sweep on real 4060/7600 hardware once 1-5 land and
   shift the kernel/host time balance.
7. **Idea 8** (K3 load reordering) — S, try cheaply, keep only if profiling shows a real win.
8. **Idea 5** (fuse K1+K2 for single-hash presets) — M/L, bigger engineering lift; do after the cheap
   wins are in and measured, so its incremental value is clear against a faster baseline.
9. **Idea 6** (subgroup bitonic sort in K1) — M, similar effort/uncertainty to idea 5, can be done
   before or after it since they touch different phases of K1.
10. **Idea 10** (K3 parse restructuring) — L, last resort / research spike, only if K3 is still the
    dominant cost after everything else.

## Traps to flag

- **Re-widening K3's workgroup** (dispatching multiple blocks per workgroup so a warp/wavefront is
  fuller) was already tried at 64 threads/workgroup and found **6x slower** than the current
  workgroup-size-1 design (per `k3_parse.wgsl`'s own comment), because independent blocks diverge almost
  immediately in a data-dependent parse. Don't re-attempt this verbatim; if revisited at all, it would
  need blocks pre-grouped by similar compressibility to bound divergence, which is itself unproven and
  adds host-side complexity for uncertain payoff.
- **Caching the full 128 KiB block in workgroup shared memory** for K2/K3 to avoid repeated global loads
  doesn't fit the hard constraint's shared-memory budget (default 16 KiB, "typically 32-48 KiB on
  consumer GPUs" — nowhere near 128 KiB). A trap driven by wishful thinking about cache locality that the
  hardware constraint rules out directly.
- **Doubling scratch buffers to let batch i+1's K1 overlap batch i's K3** across submissions is likely a
  trap: wgpu executes submissions on a single queue in program order, and the current design already
  shares scratch deliberately ("since the queue runs one batch's kernels after another anyway" — see
  `pipeline.rs`'s module doc), implying this was already considered and rejected. Cheap to sanity-check
  (un-share scratch, measure), but don't expect a win without genuine multi-queue/concurrent-kernel
  support, which wgpu's single-queue model likely doesn't give you here.
- **Bit-packing `pred[]` tighter than u32** (it only needs ~18 bits) looks like a memory-saving idea but
  is probably a trap: K2's depth walk is latency-bound on the dependent pointer chase, not
  bandwidth-bound, so adding unpack arithmetic to every hop likely costs more than the bandwidth it
  saves — and idea 1 makes most chains short anyway, shrinking exactly the population of hops that would
  have benefited from smaller `pred[]` while leaving the unpack tax on every hop.
