# Speed ideas (fable) — GPU zstd lvl9

Grounded in `crates/gzc-gpu/src/shaders/*.wgsl`, `compressor.rs`, `pipeline.rs`, `chains.rs` and the oracle in
`crates/gzc-core/src/{reference.rs,lazy.rs}`. Numbers are back-of-envelope; every "4060" figure is an
estimate from the 5090 measurements, not a measurement.

## 0. Where the time goes, and how it scales down

Per batch (1638 blocks, 215 MB) on the 5090: K1 23.3, K2 16.6, K3 61.3, K4 7.1, K5 3.2 ms.

What each kernel is actually bound by (this decides what works on a 24-SM card):

- **K1** (`k1_chains.wgsl`): one 256-thread workgroup per block runs 512 tiles *sequentially*; each tile
  is a 36-stage bitonic sort with a `workgroupBarrier` per stage plus two `storageBarrier`s, i.e. ~20 000
  barriers per block. On top of that every tile does 256 random 4-byte reads and up to 256 random writes into
  a 256 KiB `head` table; with ~1360 blocks resident on the 5090 the live head set is ~340 MB, far above the
  96 MB L2, so a good part of K1 is DRAM sector traffic (1638 × 128 K × 2 × 32 B ≈ 14 GB → 7–15 ms at 1.8 TB/s).
  Both costs scale with SM count / bandwidth: **~7× slower on a 4060 (≈160 ms)**.
- **K2** (`k2_best.wgsl`): one thread per position, 32 *dependent* `pred[q]` loads (L2 latency, ~250 ns each)
  and, per candidate, `match_len` with `load_u32_at` = 2 loads per 4 bytes for *both* p and q, re-loading
  p's bytes for every candidate (up to 64 loads per candidate, 2048 per position). On DDS with long runs this
  is LSU/issue bound → **~7× slower on a 4060 (≈115 ms)**.
- **K3** (`k3_parse.wgsl` + `k3_lazy.wgsl`): one lane per block, ~465 ns per visited position, pure
  dependent-load latency (best[], rep check, byte-wise `push_lits`). It runs as one wave of 1638 warps that
  fill 10 % of the 5090's warp slots. A 4060 has ~1150 warp slots → 1.4 waves → **~1.5–2× slower (≈100 ms)**,
  the *least* bad scaler, but still the largest single item.
- **K4/K5**: thread-0-serial phases (FSE table build + backward bitstream; Huffman build). 1 wave on the
  5090; **~5–7 waves on a 4060 (≈45 ms combined)**.

So a 4060 lands around 400–500 ms/batch ≈ 450 MB/s before PCIe: about libzstd L9 on a typical 8-core, i.e.
no win. The ideas below target ~4–8× on the 5090 and ≥5× on the 4060, all byte-identical unless stated.

## 1. K1 → warp-tile chains with subgroup ballot (no sort, no workgroup barriers)

**Mechanism.** Replace the 256-key bitonic sort with tiles of one subgroup (32 positions on NVIDIA, 32/64 on
AMD via `subgroup_size`). Each lane hashes its position, then a uniform loop finds equal-hash groups:
`h0 = subgroupShuffle(h, lowest unprocessed lane); m = subgroupBallot(h == h0)`; a lane's in-tile
predecessor is the highest set bit of `m` below its lane (`firstLeadingBit(m & lane_mask_lt)`), the first of
the group reads `head[h0]`, the last writes `head[h0] = pos+1`. The loop runs once per *distinct* hash in the
tile (≤ 32 iterations, 1 for a run of equal bytes). Workgroup = one subgroup, so the two barriers between
tiles are warp-cheap; 4096 tiles per block sit in one warp's program order, so the head read/write ordering
is exactly the current one (positions processed in increasing order, ties inside a tile by lane). Data loads
per tile are 32 consecutive bytes + 7 → two coalesced lines.

Second half, needed to fix the DRAM-random-head problem on both cards: make K1 **persistent with
epoch-tagged head tables**. Dispatch G workgroups (G ≈ 256 on the 5090, ≈ 96 on a 4060; a tunable), WG i
processes blocks i, i+G, …; head entries store `(epoch << 17) | (pos+1)` where epoch = block ordinal within
the WG, so an entry is valid iff its epoch matches — no `clear_buffer`, no per-block clearing, and `head`
shrinks from 1638 to G tables (G × 256 KiB = 24–64 MB) which now fits L2.

**Expected gain.** Removes ~20 000 barriers per block and the sort's shared-memory traffic; per tile the cost
is ~50–180 warp-instructions + one L2-hit head load. 5090: 23 → ~3–5 ms. 4060: ~160 → ~15–25 ms (issue-bound
around 5 G warp-instr per batch plus L2-resident head traffic). Also drops head VRAM by ~350 MB.

**Oracle.** Byte-identical: `pred[p]` is still the largest q < p with the same hash.

**Risk.** M. Needs `Features::SUBGROUP` (Vulkan on NVIDIA/AMD/Intel all expose it); keep the current K1 as
fallback when absent. Subgroup size must be read at runtime (`@builtin(subgroup_size)`), ballots are
`vec4<u32>`. Test: `chains::gpu_preds` vs `reference::chains` on the corpus samples (already how K1 is
checked), then K1 timestamps; also probe the DRAM hypothesis first with a throwaway `HASH_BITS=12` build
(if K1 drops a lot, head locality is the lever; if not, the barriers are).

**Deps.** Independent. Provides the persistent/epoch pattern reused by idea 7. Conflicts with a block-local
radix-sort K1 (see traps).

## 2. K2 compare diet: cap early-out, register-hoisted p-window, wide loads

**Mechanism.** Three local changes in `k2_best.wgsl`:
(a) `if (best_len == SEARCH_CAP) { break; }` after a candidate wins — candidates are visited in decreasing q
and ties keep the larger q, so no later candidate can replace a cap-length match. On DDS flat regions this
ends most walks at the first candidate.
(b) Load p's 64-byte window once per thread into 16 registers (already pre-shifted to alignment) instead of
re-loading `load_u32_at(base, p+n)` for every candidate: halves loads and removes the funnel shift on the p
side.
(c) For q, compute `q & 3` once, then load aligned words (or `vec4<u32>` from a second binding view of `data`)
and funnel-shift by that constant; compare 16 bytes per step with `x = a ^ b`, exit on first nonzero with
`countTrailingZeros`. That is ≤ 5 vec4 loads (or 17 scalar) per candidate instead of up to 64, and most
candidates exit after the first 16-byte step.

**Expected gain.** K2 is compare-bound on DDS (≈ 440 G lane-loads/batch at 32 candidates × 64 loads). Cutting
loads 3–6× and truncating walks at the cap: 5090 16.6 → ~5–8 ms; 4060 ~115 → ~30–40 ms. Register cost
~+20/thread, still fine at 256-thread workgroups.

**Oracle.** Byte-identical (same candidate order, same lengths).

**Risk.** S. Test: existing K2-vs-`find_best` comparison; then timestamps. Watch `load_u32_at` preconditions
near block end (`match_len` bounds `n + 4 <= max`; keep that for the last words).

**Deps.** Independent; (b)/(c) reuse the same match primitive idea 4 wants.

## 3. best[] as one packed u32 with extension moved into K2

**Mechanism.** K2 currently writes `(offset, capped len)` = 8 B/position (1 MiB/block, 1.7 GB/batch) and K3
re-extends every capped match it *looks at* (three `search_max` per deferral cycle → up to three full
extensions per sequence, sequentially, on one lane). Instead K2, for the winning candidate only, runs
`match_len` uncapped when `best_len == SEARCH_CAP` (parallel, once per position) and stores
`(offset << 15) | min(len, 0x7FFF)` in a single u32 (offset < 2^17, len 15 bits; 0x7FFF is a "still capped,
K3 extends" sentinel for the rare ≥ 32 KiB match). K3's `search_max` becomes one load and a couple of shifts.

**Expected gain.** Halves best[] VRAM (−840 MB/batch, which pays for idea 8's second scratch set) and
halves K2's write + K3's read traffic (on a 4060, 3.4 GB → 1.7 GB of streaming per batch ≈ −6 ms). Removes
the K3 extension work (a few ms of the 61, more on flat DDS). K2 grows slightly (one extension per position
that hits the cap, in parallel lanes).

**Oracle.** Byte-identical: the parse sees exactly `match_len(ip, ip - off)` either way.

**Risk.** S. Test: K2 output decode vs `find_best` (compare the unpacked pair, extended len vs
`match_len`), `--verify`.

**Deps.** Do before idea 4 (it removes the divergent per-lane extension from the window refill).

## 4. K3 → warp-cooperative lazy2 parse (one subgroup per block, lookahead windows)

**Mechanism.** Keep the sequential control flow of `lazy_parse` — but run it *uniformly on all 32 lanes*
(identical state in every lane), and feed it from a **window** the lanes fill in parallel:

- Refill at window base `w0`: lane k loads `best[w0+k]` (one coalesced u32 per lane after idea 3) and runs
  the rep probe `rep_len(w0+k, offset_1)` **capped at 32 bytes** (8 loads per lane; consecutive lanes hit
  consecutive bytes so both sides are coalesced). Result per lane: `best_len/off`, `rep_len_capped`.
- The control loop reads window values with `subgroupShuffle(v, ip - w0)` instead of dependent global
  loads. `offset_1` only changes at a store, so the rep column is valid for the whole deferral loop; on a
  store, or when `ip` leaves the window, refill. A rep probe that hit the 32-byte cap is extended with a
  **cooperative match_len**: lane k compares word k of the two sides, `subgroupBallot(x != 0)` → first
  differing lane → 128 bytes per step instead of 4.
- Catch-up (`while start > anchor && byte[start-1] == byte[start-1-off]`) becomes one cooperative step:
  32 lanes compare 32 bytes backwards, ballot, count trailing matches, bounded by `anchor`.
- `push_lits` (byte-wise on one lane today) becomes lane-parallel word stores: lane k builds output word k
  from 4 unaligned source bytes; only the first/last partial words go through the accumulator. (Or drop it
  entirely: idea 5.)
- The immediate-repcode loop after a store uses the cooperative match_len directly.

Everything stays data-dependent and uniform, so no divergence problem; the only per-lane divergent work is
the capped rep probe (≤ 8 iterations).

**Expected gain.** The sequential parse today pays ~2 dependent L1/L2 round trips per visited position
(~465 ns). With windows the steady-state cost per visited position is the control loop's ALU + shuffles
(~30–60 ns) plus one cooperative step for positions whose rep probe passes; refills are one coalesced round
trip per ≤ 32 positions or per store. Estimate 3–5×: 5090 K3 61 → ~12–20 ms; 4060 ~100 → ~20–30 ms
(it is still one warp per block, so the 4060's 1.4 waves stay; the per-warp latency drops).

**Oracle.** Byte-identical: every branch of `lazy_parse` is unchanged, only where its inputs come from.
`lazy.rs`'s deviation notes apply verbatim.

**Risk.** M/L. WGSL subgroup ops need uniform control flow — satisfied because state is replicated; a bug
here shows as non-determinism, so test with the `lazy_test_cases()` corpus run through the GPU and with
`--verify` on the full corpus. Register pressure: window (2 regs) + replicated state (~20) — fine. AMD
wave64: window = 64 positions, same code. Fallback without subgroups: `workgroup_size(32)` with the window
in `var<workgroup>` and `workgroupBarrier` (single-warp barriers are cheap) — still most of the gain.

**Deps.** Wants idea 3 (no per-lane extension in refill), idea 5 (simpler literal path). Enables idea 6/7.

## 5. Drop literal emission from K3; K5/K4 gather literals from (data, seqs)

**Mechanism.** Literals are fully determined by the sequences: literal byte j lies at block offset
`Σ_{s<i}(ll_s + ml_s) + (j − Σ_{s<i} ll_s)` for the sequence i whose literal run contains j. K3 writes only
`seqs` and `n_seq`; K5 (256 threads) does a workgroup prefix sum over `ll` and `ll+ml` (MAX_SEQS ≤ 32769 →
128 per thread, two passes), then every thread gathers its literal words straight from `data`, so its
histogram and stream encode read the same bytes they read from `lits` today. K4's raw-literals copy does the
same gather. `n_lit = BLOCK_SIZE − Σ ml`.

**Expected gain.** Removes `push_lits` (the one byte-at-a-time loop left on the critical lane; ~5–8 % of K3
today, more on literal-heavy DDS) and the `lits` buffer (−210 MB/batch VRAM, −0.4 GB/batch of traffic, and
the lits staging copy on the parse path). After idea 4 the gain is smaller but it simplifies the warp code
(no accumulator state to keep uniform).

**Oracle.** Byte-identical (same literal bytes in the same order).

**Risk.** M — K5 changes in two places (histogram, encode) and K4 in one; the parse-path `run` needs a CPU
gather in `decode_output`. Test: K5/K4 frames vs `write_frame` on the corpus; a unit test that the gathered
stream equals `BlockOutput::literals`.

**Deps.** Pairs with idea 4; independent otherwise.

## 6. Speculative segmented parse with convergence merge (5090-only, see trap note)

**Mechanism.** Split a block into S segments (e.g. 4 × 32 KiB). S warps each run the (idea-4) parse from
their segment start with a guessed state (`anchor = ip = seg_start`, reps = INITIAL). Afterwards a fix-up
pass re-runs the parse from segment s−1's *true* exit state into segment s until its `(ip, anchor, r0, r1,
r2, offset_1, offset_2)` coincides with a state the speculative run passed through (log the speculative
state at every store; ~5K entries per segment) — from there on the outputs are identical by determinism, so
the speculative sequences are kept and renumbered (prefix sum of per-segment counts). Literals come from
idea 5 so nothing has to be re-emitted.

**Expected gain.** On the 5090 K3 runs one wave at 10 warps/SM; S× more warps of 1/S the length is close to
an S× cut of K3 wall time (61 → ~15 ms even without idea 4, ~5 ms with it), if convergence is quick.
On a 4060 the warp slots are already full after one block per warp, so total warp-time — which the fix-up
*increases* — is what matters: **no gain, possibly a loss**.

**Oracle.** Byte-identical *if* convergence is reached; if it is not by the segment end the fix-up simply
parses the whole segment sequentially (correct, slow). Convergence is data dependent: rep-offset state can
keep two parses apart for a long time on stride-structured DDS.

**Risk.** L (state logging, merge, worst-case fallback, testing the fallback). Test: force non-convergence
(S segments with random reps) and check identity; measure the convergence distance histogram on the corpus
before building the merge.

**Deps.** Needs 4 and 5. Alternative to 7 for filling the 5090; pointless on the target card.

## 7. Per-block megakernel with per-resident-WG scratch (K1+K2+K3 in one persistent warp)

**Mechanism.** One `workgroup_size(32)` WG loops: take the next block from an atomic counter, run idea-1 K1
into *its own* head/pred slice (indexed by WG id, epoch-tagged), run K2 over the block with 32 lanes × 4096
positions each (idea 2's compare), then idea-4 K3, writing `seqs` for that block; repeat. Scratch (head 256 K
+ pred 512 K + best 0.5 M after idea 3) is allocated per resident WG (say 512) instead of per block, and the
batch size stops mattering for occupancy — the host can stream 1638 blocks or 400.

**Expected gain.** Three things at once: (i) no inter-kernel bubbles and no separate dispatch tails; (ii) the
SM always holds WGs in *different* phases, so K3's latency-bound stretch is hidden behind other WGs' K2
compares — the occupancy fix for a 24-SM card without needing subgroup tricks in every phase; (iii) scratch
VRAM 4.6 GB → ~0.7 GB and each WG's 1.9 MiB working set is reused block-to-block (L2 friendly). Expect
1.3–1.8× over ideas 1+2+4 done as separate kernels on a 4060, less on the 5090 (which has slots to spare).
K2 in-warp is the long pole: ~26 ms of warp-time per block, but 1150 warps run it concurrently.

**Oracle.** Byte-identical (same three algorithms, per block).

**Risk.** L. Register pressure = max of the three phases; a very long dispatch (whole batch) can trip GPU
watchdogs on Windows — bound each dispatch to a batch and keep an eye on TDR; WGSL uniformity is easy
(one subgroup). Test: outputs identical to the split kernels on the corpus, then measure; try WG counts
G = 2×, 4×, 8× SM count.

**Deps.** Builds on 1, 2, 4 (and 3, 5). Supersedes idea 8's overlap for K1–K3; K4/K5 stay separate and can
still overlap via idea 8's pass trick.

## 8. Per-slot scratch and cross-batch overlap inside one compute pass

**Mechanism.** wgpu inserts barriers at *pass* boundaries, not between dispatches inside a pass. Today each
kernel is its own pass and all slots share scratch, so batch i+1's K1 can only start after batch i's K4.
Give each in-flight slot its own scratch (affordable once ideas 3/5 free ~1 GB, or with batch ≈ 800) and
software-pipeline the recording: pass A = { K1(i+1), K3(i) }, pass B = { K2(i+1), K5(i)+K4(i) } with the
two dispatches of a pass touching disjoint buffers. K3/K4/K5 are latency-bound warps that leave the SM's
issue slots idle; K1/K2 fill them.

**Expected gain.** Today: hides most of K3 (61 ms) behind K1+K2 (40 ms) → batch ≈ max(...) ≈ 70–75 ms,
~1.5×. After ideas 1–4 the absolute gain shrinks but it still hides K4/K5 (which become the largest item on
a 4060, ~45 ms) behind the next batch's K1/K2. Also removes the 11 % host gap partially: the staging copies
of batch i land in the same submission as batch i+1's kernels.

**Oracle.** Byte-identical.

**Risk.** M. It must be verified with timestamps that the driver really overlaps dispatches in one pass
(NVIDIA does when no barrier separates them; AMD usually too). If it does not, fall back to idea 7, which
gets the overlap within a dispatch. VRAM: 2 scratch sets at batch 1638 does not fit the 6144 MiB budget
without 3 and 5.

**Deps.** Needs 3 and/or 5 for memory; partially superseded by 7.

## 9. Transfer path: GPU-side frame compaction and copy overlap (matters on PCIe x8 cards)

**Mechanism.** Readback copies `frames` at a fixed stride (1638 × 128 KiB + 64 = 215 MB/batch) although the
frames total ~160 MB. Add a tiny kernel after K4: exclusive prefix sum over `frame_len` (1638 values, one
WG), then a copy kernel that packs frames contiguously; read back `prefix[n]` bytes. Similarly on the host
side, keep the upload as is but check with timestamps whether the upload copy of batch i+1 overlaps batch
i's kernels; if not, move the copy into the *first* dispatch of the batch by having K1 read from the upload
copy target only after the copy pass (i.e. record the copy at the end of the previous submission).

**Expected gain.** A 4060 is PCIe 4.0 ×8 (~12 GB/s): 215 MB up + 215 MB down ≈ 35 ms per batch, i.e.
comparable to the whole optimized kernel time. Compaction cuts the down copy by ~25 %; overlap hides the
rest. On the 5090 (PCIe 5 ×16) this is ~5 %.

**Oracle.** Byte-identical.

**Risk.** S (compaction) / M (overlap experiments). Test: frame bytes unchanged, end-to-end MB/s.

**Deps.** Independent.

## Smaller items (S each, byte-identical)

- **K4/K5 warp budget on a 4060.** K5 uses 8 warps per block, K4 2; on a 24-SM card that is 5–7 waves. Give
  K5 64 threads for its serial-heavy phases (or run K5's histogram/encode with 4 warps) and shrink K4's
  `st[S_ALL]`/`spread` so ≥ 16 WGs per SM are resident; check shared-memory-limited occupancy with the
  adapter's `max_compute_workgroup_storage_size`.
- **K2 dispatch order.** Dispatch `(n_blocks, BLOCK_SIZE/256)` → keep it; but consider two positions per
  thread (p, p+65536) to double memory-level parallelism if the profile says latency after idea 2.
- **K1 `clear_buffer` of head** disappears with the epoch tag (idea 1); the K1 `pred_out` tail writes
  (`NO_POS` for the last 7 positions) can be folded into the last tile.
- **Batch sizing per card.** After idea 1/4 the per-warp kernels want batch ≈ warp slots (5090: any; 4060:
  ~1150). Expose it and let `inflight` grow to 4 with the smaller per-slot memory.

## Recommended order (value per effort)

1. **Idea 2** (K2 compare diet) — a day, S, likely 2–3× on K2 on both cards, no structural change.
2. **Idea 3** (packed best[] + extension in K2) — S, frees 840 MB, simplifies everything after it.
3. **Idea 1** (subgroup K1 + persistent epoch-tagged head) — M, ~5× on K1, first subgroup kernel so it
   settles the feature-detection/fallback plumbing.
4. **Idea 4** (warp-cooperative K3) — M/L, the largest single win (55 % of GPU time today) and the one whose
   gain carries over to the 4060 undiminished.
5. **Idea 5** (literals from seqs) — M, do it as part of 4 or right after.
6. **Idea 8** (per-slot scratch + in-pass overlap) — M, cheap to try once memory allows; measure first.
7. **Idea 9** (compaction/copy overlap) — S/M, when the target card is a ×8 part it stops being optional.
8. **Idea 7** (megakernel) — L, only if the 4060 profile after 1–5 still shows idle SMs during K3/K4/K5.
9. **Idea 6** (segmented parse) — only for filling a huge GPU; skip for the target hardware.

Rough end state (5090, per batch): K1 ~4, K2 ~6, K3 ~15, K4+K5 ~10 → ~35 ms ≈ 6 GB/s kernel-only, at which
point PCIe and host copies dominate. 4060 estimate: ~100–130 ms/batch ≈ 1.7–2.1 GB/s kernel-only, i.e.
~3× the 8-thread libzstd L9 figure on the same class of machine, with ratio unchanged.

## Traps

- **Segmented/speculative K3 (idea 6) on the target card**: adds fix-up work to a machine whose warp slots are
  already full; the 5090 numbers will look great and mean nothing for a 4060.
- **Block-local radix/counting sort as the K1 replacement**: stable 16-bit sort of 128 K keys per block on
  wgpu is 4 passes × 1 MiB of traffic per block (6.5 GB/batch → 25 ms of DRAM time on a 4060) plus a big
  implementation; the ballot-chain K1 (idea 1) gets the same `pred[]` with one pass. Only worth revisiting
  if K2 turns out to be dependent-load bound after idea 2 (a sorted (hash,pos) array would let K2 read its 32
  candidates as one contiguous window).
- **Shrinking `HASH_BITS` to 15/14 to make head tables L2-resident**: changes the chains, so it is a new
  preset with a ratio risk against a 0.13 % margin over libzstd L9. Use the persistent epoch-tagged head
  instead (same L2 effect, identical output).
- **Relying on cross-submission overlap**: wgpu serialises passes with barriers; overlap must be inside one
  pass (idea 8) or one dispatch (idea 7). Measure with timestamps before building on it.
- **On-demand K2 inside K3** (compute best[] for the window in K3's lanes, skipping positions the parse
  never visits, ~35 % on DDS): tempting (deletes K2 and best[] entirely) but it puts a 32-deep chain walk
  (~8 µs) into every window refill, so K3 wall time per block goes *up* ~5× and it only pays on a card with
  idle warp slots. Keep as a later experiment, after 7.
