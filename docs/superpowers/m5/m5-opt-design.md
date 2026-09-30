# M5: a GPU optimal parser for libzstd L14 / L16 ratio (design)

Date: 2026-09-30. Research and design only; nothing in the repo was changed. Ground truth: libzstd 1.5.7
`lib/compress/zstd_opt.c` (the `zstd-sys-2.1.0+zstd.1.5.7` sources in the cargo registry). Every ratio here was
measured with a CPU prototype in `/tmp/claude-1000/m5d/` (path dependency on `gzc-core`, `block-64k`) on
**every 50th 64 KiB block of `data/corpus` (2016 blocks, 123.9 MiB real, DDS + NIF in file order)**; every
prototype frame was decoded by libzstd (`CHECK=1`). libzstd numbers use `zstd::bulk::Compressor` on the padded
block exactly as `gzc-bench cpu` does. GPU times are projections from the measured `lvl9seg` kernels and are
marked as such. The parallel report `m5-ratio-drivers.md` (different sample, independent prototype) agrees on
every factor both measured; where its numbers are used they are cited.

## 0. Summary

- **What L14/L16 are at 64 KiB.** `ZSTD_adjustCParams_internal` turns the table row into windowLog 16,
  chainLog 17, hashLog 17, hashLog3 16; the post-block splitter is disabled (it needs windowLog >= 17), so at
  64 KiB the extra ratio is *parse only*. L14 = btopt (optLevel 0), searchLog 4, minMatch 3, targetLength 32.
  L16 = btultra (optLevel 2), searchLog 6, minMatch 3, targetLength 128. L16 is one pass; the two-pass
  statistics seeding (`ZSTD_initStats_ultra`) is btultra2 = L19+.
- **Where the ratio comes from** (this sample; libzstd L9 = 1.33856, our lvl9 = 1.34011, lvl9seg = 1.34009):

  | libzstd | ratio | vs L9 | note |
  |---|---:|---:|---|
  | L14 | **1.36911** | +2.28 % | |
  | L16 | **1.37175** | +2.48 % | searchLog 10 gives the same: the tree is saturated at 64 compares |
  | L19 | 1.37441 | +2.68 % | btultra2 |
  | L14 with minMatch 4 | 1.35150 | +0.97 % | **3-byte matches are 58 % of L14's gain** |
  | L14 as btultra (optLevel 2) | 1.37083 | | fractional prices, no early aborts: +0.13 % |
  | L14 with searchLog 6 | 1.36993 | | finder depth: +0.06 % |
  | L16 with targetLength 32 | 1.37165 | | targetLength is irrelevant (−0.007 %) |

- **A faithful port of `ZSTD_compressBlock_opt_generic` is not the hard part.** With an ideal longest-match
  candidate set the port lands at 1.37152 (L16 is 1.37175). What decides the GPU ratio is (1) 3-byte
  candidates and how many, (2) how close the candidate set gets to the true longest match, (3) the price model.
- **Recommended GPU design, measured on the CPU prototype:**
  - **`opt16`: 1.37211 = above libzstd L16 (+0.03 %)**, with per position the K = 2 records (nearest match
    >= 3 bytes, longest match) of a nearest-first walk over **two K1 chains: the 4-byte hash 32 deep and a
    3-byte hash 4 deep**, both with K2's 64-byte compare cap; the DP run **one lane per 4 KiB segment**
    exactly like `lvl9seg` (cost −0.06 % vs the whole block), rep history **exact per DP node**, zstd's series
    structure with `targetLength 32` (live price table = a **33-entry ring** per lane); **static prices per
    block per pass**, iterated: pass 0 priced from zstd's block init, then 2 cheap (btopt-arithmetic) passes and
    one btultra final pass, each priced from the previous pass's own histogram. Converged static prices beat
    zstd's sequential adaptive statistics (+0.17 %).
  - **`opt14`: 1.37127 = above L14 (+0.16 %), −0.03 % under L16, with two DP passes**: the same candidates,
    pass 0 priced from a corpus prior for LL/ML/OF codes and a parse-free literal estimate (bytes no candidate
    covers), then the final pass. A single pass gives 1.36732 (−0.13 % under L14).
  - Adding an 8-byte-hash chain (Dfast-style) gives +0.03 % (1.37255); K = 3 another +0.015 %. With an ideal
    longest-match finder the recipe reaches **1.37487 (> L19)**; the finder is the lever left after this design.
- **Cost (projected, 5090, per 64 KiB block):** K3opt ≈ 10–14 µs for `opt16`'s 4 DP passes (5–7 µs for
  `opt14`'s 2) against `lvl9seg`'s 1.25 µs for K3; with a second K1 chain and the extra K2 walk the kernel
  sums are ≈ 21–25 µs (`opt16`) and 16–18 µs (`opt14`) with today's K1, i.e. **≈ 2.7–3.1 GB/s and 3.6–4.0 GB/s
  on the 5090, ≈ 0.4–0.6 and 0.55–0.8 GB/s on a 4060** (÷5–7). Neither preset holds 1.25 GB/s on a 4060; they
  are quality presets (links ≤ 3–5 Gbit/s, or repacks).
- **VRAM:** ≈ 1.35 MiB per block (today 0.77): 2 pred chains (512 KiB, reused as the DP trace), 2 words of
  candidates per position (512 KiB), `MAX_SEQS` for min match 3 (256 KiB). Batch max drops to ≈ 2900 blocks.

## 1. How `ZSTD_compressBlock_opt_generic` works

Line numbers refer to `zstd_opt.c` (1.5.7). The function is shared by btopt (`optLevel 0`), btultra and
btultra2 (`optLevel 2`); `ZSTD_compressBlock_btultra2` only adds a first pass to seed statistics.

### 1.1 Match collection: `ZSTD_insertBtAndGetAllMatches` (l. 590–818)

Called at every position the DP examines (`getAllMatches`), it fills `matches[]` with **(offBase, len)
records in strictly increasing length**, each longer than all previous ones (`bestLength` starts at
`lengthToBeat − 1 = minMatch − 1`):

1. **Rep candidates** (l. 646–688). With `ll0 = (litlen == 0)` the candidates are `rep[ll0 .. ll0+3)` where
   index 3 means `rep[0] − 1`: right after a match (ll0 = 1) they are rep[1], rep[2], rep[0]−1 (zstd's repcode
   numbering), otherwise rep[0], rep[1], rep[2]. Each is validated (`repOffset − 1 < curr − dictLimit`, i.e.
   1 <= off <= curr), tested with a `minMatch`-byte compare (`ZSTD_readMINMATCH`, 3 or 4 bytes) and extended
   with `ZSTD_count`. A rep longer than `bestLength` is recorded as `REPCODE_TO_OFFBASE(repCode − ll0 + 1)`
   (offBase 1..3). If `repLen > sufficient_len` or the rep reaches `iLimit` the function returns at once.
2. **hash3** (l. 691–720), only when `mls == 3` and **no rep of length >= 3 was found**:
   `ZSTD_insertAndFindFirstIndexHash3` inserts every position below `ip` into `hashTable3` (2^hashLog3
   entries, hashLog3 = min(17, windowLog) = 16 at 64 KiB; `ZSTD_hash3 = ((u32 << 8) * 506832829) >> 16`) and
   returns the **last position with the same hash**; the candidate is verified by `ZSTD_count` (`mlen >= 3`)
   and recorded as an explicit offset. One candidate, nearest by hash bucket, any length. (Our design
   generalises this into a chain walked 4 deep, §2.1.)
3. **Binary tree** (l. 722–774): `hashTable[h]` (hashLog 17 over `mls` bytes) gives the root for the current
   hash; the walk descends comparing `ip` against `match` with the `commonLengthSmaller/Larger` trick, at most
   `nbCompares = 1 << searchLog` nodes (16 for L14, 64 for L16), recording every node whose `matchLength >
   bestLength` as `OFFSET_TO_OFFBASE(curr − matchIndex)` (offset + 3). A match reaching `iLimit` or longer
   than `ZSTD_OPT_NUM` stops the walk. The tree is *re-linked* on the way (`smallerPtr/largerPtr`): insertion
   order matters, so it is inherently sequential. `ZSTD_updateTree_internal` (l. 562–581) first inserts every
   position from `nextToUpdate` to `ip` (`ZSTD_insertBt1`, l. 442–558, same walk without recording).
4. `ms->nextToUpdate = matchEndIdx − 8` (l. 816): after a match whose *source* reaches beyond `curr + 8`
   (overlapping runs) the next positions are a "skipped area" returning 0 matches (l. 846). Measured effect
   on our data: none (1.37068 vs 1.37067); the GPU design drops it.

The result is a short list (typically 1–4 records) sorted by length. The tree finds long matches far away that
a nearest-first chain of equal depth misses (§2.1); `searchLog` beyond 6 changes nothing at 64 KiB.

### 1.2 Prices (l. 25–387)

Integers in 1/256 bit (`BITCOST_MULTIPLIER = 256`). `WEIGHT(stat, optLevel)` is `ZSTD_bitWeight` (whole bits,
`highbit32(stat+1) * 256`) for optLevel 0 and `ZSTD_fracWeight` (linear interpolation between powers of two,
l. 53–65) for optLevel 2. A symbol's price is `WEIGHT(sum) − WEIGHT(freq[sym])` plus its extra bits:

- `ZSTD_rawLiteralsCost` (l. 266–291): per literal `litSumBasePrice − min(WEIGHT(litFreq[b]), litSumBasePrice
  − 256)`, i.e. at least one bit per literal.
- `ZSTD_litLengthPrice` (l. 295–315): `LL_bits[llCode] * 256 + llSumBasePrice − WEIGHT(llFreq[llCode])`.
- `ZSTD_getMatchPrice` (l. 324–352): offset code `highbit32(offBase)` (repcodes 1..3 cost as codes 0/1) with
  `offCode` extra bits, the ML code price with its extra bits, plus a constant **`256/5` (0.2 bit) per match**.
  optLevel < 2 adds `(offCode − 19) * 2` bits for offCode >= 20, impossible at 64 KiB (max offCode 16).
- **Statistics** (`ZSTD_rescaleFreqs`, l. 141–261). Our frames are one block each, so the "first block, no
  dictionary" branch always applies: `litFreq` = histogram of the raw block downscaled to `(c > 0) + (c >> 8)`;
  `litLengthFreq` = `{4, 2, 1, 1, ...}`; `matchLengthFreq` = all 1; `offCodeFreq` = `{6, 2, 1, 1, 2, 3, 4, 4, 4,
  3, 2, 1, ...}`. `ZSTD_setBasePrices` caches `WEIGHT(sum)` per table.
- **Updates** (`ZSTD_updateStats`, l. 356–387): after every committed series (§1.4) each stored sequence adds
  `ZSTD_LITFREQ_ADD = 2` per literal byte and 1 per LL/ML/OF code, then `ZSTD_setBasePrices`. Prices are
  constant *within* a series and adapt *between* series. **btultra has the same statistics as btopt**; it
  differs only in `optLevel` (fractional weights, no early aborts, the match+1-literal check). btultra2 (L19+)
  runs the block once to fill the tables, forgets the sequences, and runs again.

### 1.3 The forward pass (l. 1119–1338)

The parse is a sequence of **series**. `ip` advances by 1 while `getAllMatches(ip)` finds nothing (l. 1129).
At the first position with a match a series starts: `opt[0]` = {mlen 0, litlen = ip − anchor, price =
LL_PRICE(litlen), rep}. `opt[pos]` (`ZSTD_optimal_t`: price, off, mlen, litlen, rep[3]) stores **stretches**
("a match followed by `litlen` literals ending at pos"), not sequences, so different literal runs can follow
the same match.

- **Large match, immediate encoding** (l. 1155–1169): if the longest match at the series start exceeds
  `sufficient_len = min(targetLength, ZSTD_OPT_NUM − 1)` it is taken greedily (`goto _shortestPath`). This is
  the only role of `targetLength`; at 64 KiB it is worth −0.007 % (32 vs 128).
- Initial prices for `pos` in [minMatch, maxML] from the series-start records (l. 1173–1196): for each record
  and each length up to its `len`, `opt[pos] = opt[0].price + matchPrice(offBase, pos) + LL_PRICE(0)`.
- **Main loop** `for cur = 1 ..= last_pos` (l. 1200): at each position
  1. *literal extension* (l. 1206–1249): `price = opt[cur−1].price + LIT_PRICE(ip+cur−1) + LL_INCPRICE(litlen)`
     replaces `opt[cur]` when `<=`; optLevel >= 1 also checks whether "the replaced match + 1 literal" is
     cheaper at `cur + 1` than "more literals" and, if so, writes `opt[cur+1]` (`last_pos` may grow by one);
  2. *rep update* (l. 1256–1261): if `opt[cur]` ends with a match (litlen 0), `opt[cur].rep = ZSTD_newRep(
     opt[cur − mlen].rep, off, opt[cur − mlen].litlen == 0)` — **exact per node**, from the predecessor node;
  3. `if (inr > ilimit) continue; if (cur == last_pos) break;` — the series closes at the frontier;
  4. optLevel 0 only (l. 1268–1272): skip the position if `opt[cur+1].price <= opt[cur].price + 128`;
  5. `getAllMatches(inr, opt[cur].rep, ll0)`; if the longest match `> sufficient_len`, or `cur + longestML >=
     ZSTD_OPT_NUM`, or it reaches `iend` → immediate encoding of that match;
  6. *relaxation* (l. 1305–1336): for each record `(off, lastML)` and `mlen` from `lastML` **down** to
     `startML` (previous record's len + 1, or minMatch): `price = opt[cur].price + LL_PRICE(0) +
     matchPrice(off, mlen)`; if `pos = cur + mlen` is beyond `last_pos` or the price is lower, write
     `opt[pos] = {mlen, off, litlen 0, price}` (filling skipped positions with MAX_PRICE); optLevel 0 breaks
     out of the length loop at the first non-improvement.

  Two consequences matter for the GPU: **every relaxation targets `cur + mlen <= cur + sufficient_len`**, so
  the live part of `opt[]` is a window of `sufficient_len + 1` entries behind the frontier (33 for
  targetLength 32), and **every position is visited exactly once** over the block, as a literal skip or inside
  a series (measured: 33.0 K in-series + 28.3 K skips = 61.3 K + positions inside immediately-committed
  matches; relaxations 52 K per block at optLevel 2 / targetLength 32, 32 K at optLevel 0; rep probes 183 K =
  2.8 per position).
- **`ZSTD_OPT_NUM = 4096`** bounds a series (and `opt[]`); with 4 KiB segments it coincides with the segment.

### 1.4 The backward trace (l. 1340–1437)

`lastStretch = opt[last_pos]` (or the immediately encoded match); `cur = last_pos − mlen [− litlen]`. Reps for
the next series are `ZSTD_newRep(opt[cur].rep, off, ...)` for a series ending in a match, or the stretch's own
reps. The path is walked backwards through `opt[stretchPos]` with `stretchPos −= litlen + mlen`, converting
stretches into sequences (each match takes the *previous* stretch's trailing literals as its `litlen`), stored
in reverse into the top of `opt[]`, then emitted in order with `ZSTD_updateStats` + `ZSTD_storeSeq`. Trailing
literals of the last stretch are not consumed (`ip = anchor + llen`): the next series may start inside them.

## 2. GPU mapping

### 2.1 Candidate generation (K1 / K2)

The binary tree cannot be built in parallel (R5 §2, §1.1 above); the DP needs only its *output*: a short
increasing-length list per position. Measured on the prototype (whole block, zstd-adaptive prices, optLevel 2,
zstd's hash3 slot; libzstd L16 = 1.37175), only the explicit candidate set varies:

| explicit candidates per position | ratio | vs L16 |
|---|---:|---:|
| ideal: longest match among the 4096 nearest same-hash positions, K = 8 | 1.37152 | −0.02 % |
| ideal with K2's 64-byte compare cap, K = 2 | 1.37132 | −0.03 % |
| 4-byte-hash chain depth 1024 / 256 / 128, cap 64, K = 2 | 1.37106 / 1.36992 / 1.36935 | −0.05 / −0.13 / −0.17 % |
| **depth 32 (= lvl9's K2 walk), K = 2** | **1.36855** | −0.23 % |
| depth 32, K = 8 / 4 / 2 as the two longest | 1.36856 / 1.36856 / 1.36856 | K = 2 loses nothing |
| depth 32, K = 1 (longest only) | 1.36495 | −0.50 % |
| depth 16 / 64, K = 2 | 1.36815 / 1.36888 | |
| 4-byte 32 + 8-byte-hash 32 (Dfast-style), K = 2 | 1.36882 | −0.21 % (+0.06 % under converged prices) |
| no hash3, minMatch 4 (depth 64, K = 8) | 1.35019 | −1.57 % |

And with the recommended price model (block init + 3 cheap passes + optLevel 2 final, 4 KiB segments,
targetLength 32), varying the 3-byte source and the finder:

| 3-byte source | 4-byte chain | 8-byte chain | K | ratio | vs L16 |
|---|---|---|---|---:|---:|
| zstd hash3 slot (last position, used only when no rep) | 32 | – | 2 | 1.37018 (tl128) | −0.11 % |
| zstd hash3 slot | 32 | 32 | 2 | 1.37057 | −0.09 % |
| **3-byte chain depth 2 / 4 / 8, records from length 3** | 32 | – | 2 | 1.37180 / **1.37211** / 1.37215 | +0.004 / **+0.03** / +0.03 % |
| 3-byte chain depth 4 | 16 | – | 2 | 1.37176 | 0.00 % |
| 3-byte chain depth 2 / 4 | 32 | 32 | 2 | 1.37223 / 1.37255 | +0.04 / +0.06 % |
| 3-byte chain depth 4 | 32 | 32 | 3 | 1.37275 | +0.07 % |
| 3-byte chain depth 4 | ideal (4096) | – | 2 | **1.37487** | +0.23 % (> L19 1.37441) |

`m5-ratio-drivers.md` measured the same h3 depth effect independently (+0.13 % for depth 2, +0.15 % for 4).
The 3-byte chain is the better second key: it costs a 4-deep walk where the 8-byte chain costs a 32-deep one,
and buys 3× more. The remaining +0.23 % to the ideal finder is what a sort-based LCP-neighbour finder
(`m5-rt-npu.md` §4.4) could recover; it should be evaluated against the last row once the bucket-sort K1
(track B) exists, because it reuses that machinery. Deep single-chain walks are not the way: W = 1024 is 4 KiB
of window per position.

**K1.** Two keys per block: `h4` (today's 16-bit hash of 4 bytes) and `h3` (zstd's `ZSTD_hash3` of 3 bytes,
16 bits). Today's K1 already builds `N_HASHES` chains (Dfast = 2), so `pred` becomes `[block][2][pos]` with
`h3` in the slot Dfast uses for the short hash. Under track B's bucket-sorted array each key is one counting
sort. (An optional third `h8` chain buys +0.03 %.)

**K2opt.** One thread per position `p < PARSE_END`, as today:

1. walk the `h4` chain 32 deep and the `h3` chain 4 deep (fingerprint filter, 64-byte capped compare; the h3
   pred word's fingerprint carries the 3-byte hash and byte 3 so a 3-byte miss costs no data load); merge the
   two walks by position (both are ordered nearest-first) and keep the records where the capped length
   strictly beats the best so far, starting at length 3 — K2's current loop with the tie rule "longer, then
   nearer". Only two registers survive: `A` = the first record (nearest match of >= 3 bytes) and `B` = the
   last (longest, nearest on ties). The prototype merged the visited positions by offset; running the h3 walk
   first and the h4 walk after with `A`/`B` carried across is equivalent if the merge order is kept.
2. output two words per position: `word0 = offA:16 | lenA:8 | lenB:8`, `word1 = offB:16 | 0` (16 spare bits;
   `lenA = 0` = no candidate; lengths are capped values, the DP extends a length of 64 exactly as `search_max`
   does today). 8 B per position = 512 KiB per block.

zstd's rule "hash3 only when no rep >= 3 exists" is replaced by treating 3-byte records as ordinary explicit
candidates; the DP prices them against their literals like any other, which is why depth helps (+0.12 % over
the slot at depth 2).

### 2.2 The DP kernel (K3opt)

**Parallel structure: exactly `lvl9seg`'s.** One lane per 4 KiB segment, 16 lanes per block, workgroups of
32 lanes (2 blocks), no subgroup operations, a per-block fix-up epilogue. Segment `k > 0` starts with `ip =
anchor = k·4096`, reps `[0, 0, 0]` (no rep candidate validates until the segment's first match), `iend =
(k+1)·4096`, `ilimit = iend − 8`; explicit candidates are clamped to `iend − p` and dropped below 3; segment 0
starts at `ip = 1` with reps `[1, 4, 8]`. Trailing literals are carried into the next segment's first
sequence and `off_base` is re-encoded against the true decoder reps by the same fix-up as `k3_seg.wgsl`.

Why nothing more parallel inside a segment: the DP's dependency is 1-D and every position is visited once,
so a wavefront over positions gains nothing over more segments per block, and smaller segments cost ratio
(2 KiB −0.06 %, 8 KiB +0.03 %, whole block +0.06 %, measured on the recipe; `m5-ratio-drivers.md` measured
−0.015 % for 4 KiB on its sample). Rep probes at a position depend on the reps of the best path to it, which
rules out precomputing them.

**Lane state.** The per-position work is small and uniform: 2.8 rep probes, one 8-byte candidate load,
~0.85 relaxations (0.5 at optLevel 0), one literal step. The whole per-lane state fits a ring of
`sufficient_len + 1 = 33` nodes (targetLength 32; 128 would need 129):

```
node (20 B): price: i32; rep0, rep1, rep2: u16; litlen: u16; mlen: u16; off: u16 (offset, 0 = none); rc: u8 (repcode 0..3 | explicit)
ring[33] per lane → 660 B; 32 lanes → 21 KiB workgroup memory, + 2 × 754 B u16 price tables ≈ 23 KiB
```

`var<workgroup>` at 23 KiB needs `max_compute_workgroup_storage_size` >= 32 KiB (NVIDIA 48 KiB, AMD 64 KiB;
wgpu's default is 16 KiB, so the pipeline requests the adapter limit). Fallback: the ring in `var<private>`
(local memory, L1-cached, per-lane interleaved). Occupancy: 5090 128 KB shared per SM → 5 workgroups = 5 warps
per SM (`lvl9seg` runs 15). If latency shows, 16-lane workgroups (one block, 11.5 KiB) double the count at
the cost of half-empty warps on NVIDIA, or the node shrinks to 16 B (derive `rc` from `off`).

**The lane loop** (one trip per position, all lanes in lockstep; `pos` is relative to the segment; `ip`,
`anchor`, `last_pos`, `cur` are zstd's):

```
loop (pos = ip .. ilimit):
  if not in_series:                                       # zstd l.1122-1133
     cands = getAllMatches(pos, rep, ll0 = pos == anchor)
     if none: pos += 1; continue                            # literal skip (~45 % of positions)
     open series: ring[0] = {litlen = pos-anchor, price = LL_PRICE(litlen), rep}
     if maxML > 32: commit(maxML) (immediate encoding); continue
     seed ring[3..maxML] from the records; last_pos = maxML; cur = 1
  else:
     n = ring[cur]; p = ring[cur-1]
     literal extension (and optLevel 2's match+1-literal check into ring[cur+1])
     if n.litlen == 0: n.rep = newRep(ring[cur - n.mlen].rep, n.off, ...)   # exact per node
     trace[pos] = (n.mlen, n.litlen, n.off, n.rc)          # 8 B to global memory; n is final now
     if cur == last_pos: commit(); continue
     [optLevel 0: skip if ring[cur+1].price <= n.price + 128]
     cands = getAllMatches(pos, n.rep, n.litlen == 0)
     if longest > 32 or cur + longest >= 4096: commit(longest); continue
     for each record (off, lastML), mlen = lastML down to startML: relax ring[cur + mlen]   # <= 30, typically 3-6
     cur += 1; pos += 1
commit(): backward trace over trace[] from last_pos (dependent 8-byte loads, one per sequence, ~330 per
  segment and pass); sequences appended to the lane's slot range with the lane's own off_base (as k3_seg);
  rep = reps at the series end; ip = anchor (+ trailing literals); in_series = false
```

`getAllMatches` on the GPU: up to three rep probes (a 4-byte load at `pos − rep_i` and a compare, all three
independent → one memory round trip), each hit extended with the existing `match_len` (bounded by `iend`);
then the candidate words (one 8-byte load streamed per lane; `A`/`B` clamped to `iend`). Records are ordered
as zstd orders them (reps by increasing length, then `A`, then `B`) so the `startML` rule is the same.

**Trace and outputs.** `trace[segment][4096]` × 8 B = 512 KiB per block, placed in the dead `pred` buffer
(512 KiB after K2). The final pass writes sequences into the lane's slot of `best`/`seqs` like `main_seg` and
runs `main_fixup` unchanged (prefix sum of counts, literal carry, `off_base` re-encode until the true reps meet
the lane's). Intermediate passes (§2.4) never write sequences: their backward trace only feeds the histogram.

**Instruction count per trip:** the prototype does ~120 scalar operations per in-series position and ~40 per
literal skip; with 2 records the relaxation loop is bounded by 30 and averages ~5 → an estimated 150–250
instructions per trip against `lvl9seg`'s ~200 per state-machine iteration, of which it runs ~2.6 K per lane
instead of 4096.

### 2.3 Rep-offset state

Exact, as zstd: the three reps live in every ring node (u16 each: offsets < 65536 in a 64 KiB block, 0 =
unset), updated with `ZSTD_newRep` from the predecessor node when a node ends in a match, copied through
literal extensions. Rep candidates are probed from `ring[cur].rep` with zstd's `ll0` numbering. The only
approximation is the segment start (reps zero), already included in the segmentation cost; carrying the
previous pass's segment-end reps into the next pass's segment start measured +0.006 % (noise) and is not worth
the coupling. What exact rep tracking is worth: no rep candidates in the DP (repcodes recovered by the encoder
only) −0.45 % (1.36454 vs 1.37067), one rep candidate −0.19 %, two −0.08 %.

### 2.4 Price model: static per block, iterated

zstd's prices adapt sequentially through the block; a lane cannot see other segments' statistics. Measured
(K = 2, depth 32 + zstd's hash3 slot, optLevel 2 unless stated; "whole" = one sequential DP, "seg4k" = 16
independent segments):

| Prices | whole block | seg4k |
|---|---:|---:|
| zstd adaptive (block init + updates per series) | 1.36855 | 1.36220 (each lane adapting from the block init) |
| frozen block init (literal histogram >> 8, baseline LL/ML/OF) | 1.33537 | |
| frozen lvl9seg-parse histogram | 1.35170 | |
| block init → 1 static pass | 1.36482 | 1.36420 |
| block init → 2 static passes | 1.37004 | 1.36933 |
| **block init → 3 static passes** | **1.37092** | **1.37018** |
| block init → 4 static passes | | +0.007 % over 3 |
| lvl9seg histogram → 1 / 2 passes | 1.35780 / 1.36363 | |
| "oracle": frozen histogram of the adaptive parse (+1 pass) | 1.37149 (1.37153) | 1.37082 |
| block init → 3 *cheap* (optLevel 0) passes, optLevel 2 final (dual-chain finder) | | 1.37067 vs 1.37077 with full passes |

Each pass prices every symbol as `WEIGHT(sum) − WEIGHT(count)` (fractional weights, 1/256 bit) from the
previous pass's own output: literal byte histogram, LL/ML/OF code histograms with `off_base` under the decoder
reps. This is the fixed-point iteration btultra2 does once; it converges in three passes and lands **above**
the sequential adaptive statistics, because for a single-block frame the final histogram *is* the cost.
Seeding from the lazy parse converges slower (its histograms have no 3-byte matches and different LL codes),
so the seed is zstd's own block init — or better, a prior (next paragraph). Intermediate passes can use the
btopt arithmetic (whole-bit weights, early abort, the `+128` skip: 32 K relaxations instead of 52 K per block)
at −0.01 %; an optLevel 0 *final* pass costs −0.05 %.

**Seeding with a prior** (recipe finder: h4 32 + h3 chain 4, K 2, seg4k, targetLength 32, cheap intermediate
passes; the prior is the summed LL/ML/OF histogram of the recipe's own output over the sample — in-sample, but
`m5-ratio-drivers.md` found held-out per-kind tables within 0.0002 of in-sample ones):

| seed | passes total | ratio | vs L14 / L16 |
|---|---:|---:|---|
| block init, + 3 cheap | 4 | 1.37211 | +0.22 % / +0.03 % |
| block init, + 2 cheap (h4+h8 finder) | 3 | 1.36969 | +0.04 % / −0.15 % |
| block init, + 1 cheap (h4+h8 finder) | 2 | 1.36468 | −0.32 % |
| prior codes + raw-block literals, + 1 cheap | 2 | 1.37072 | +0.12 % / −0.08 % |
| prior codes + raw-block literals, no extra pass | 1 | 1.35976 (h4+h8) | −0.68 % |
| **prior codes + cover literals, + 1 cheap** | **2** | **1.37127** | **+0.16 % / −0.03 %** |
| prior codes + cover literals, + 2 cheap | 3 | 1.37189 | +0.20 % / +0.01 % |
| prior codes + cover literals, no extra pass | 1 | 1.36732 | −0.13 % / −0.32 % |
| same, optLevel 0 | 1 | 1.36689 | −0.16 % |

"Cover literals" = the histogram of the bytes no candidate match covers (a prefix max of `p + len` over the
candidate words, then one 256-bin count per block: parse-free, one light pass over the candidate buffer).
`m5-ratio-drivers.md` reaches L14 (+0.03 %) in one pass with *per-kind* (DXT1 / DXT5 / NIF) tables; the single
global prior here does not, so the 1-pass variant is a stretch goal that needs per-kind tables selected from
the DDS header (R9's mechanism).

GPU form: pass `n` histograms its own sequences in the workgroup epilogue (shared-memory atomics into
256 + 36 + 53 + 32 counters per block, written to a 1.5 KiB per-block table); pass `n+1`'s prologue converts
them to `u16` prices in workgroup memory (one `WEIGHT` per lane per entry). Pass 0's prologue computes the
cover-literal histogram from the candidate words (or the raw-block histogram) and loads the constant prior
tables. No extra dispatches; the same counts are what K4/K5 build anyway.

### 2.5 3-byte matches

Essential: −1.57 % without them (the whole gap from L14 to L9 is 2.28 %). They are cheap: one more K1 key
(16-bit hash of 3 bytes), a 4-deep walk in K2, records from length 3 in the same two candidate words.
Downstream: `MatchParams::min_seq_len` 3 → `MAX_SEQS = BLOCK_SIZE / 3 + 1` (the `seqs` buffer grows 33 %),
`ml_code` already accepts 3, the frame format allows it, K4's cost tables are unchanged (ML code 0). Sequence
counts rise ~55 % against lvl9 (≈ 5.3 K per 64 KiB block), so K4 and K5 scale accordingly (§3.3).

### 2.6 What was dropped from zstd and why (all measured)

- `nextToUpdate` skipped area (l. 816/846): 1.37068 vs 1.37067.
- targetLength 128 → 32: −0.007 %; the ring shrinks 4×.
- Snapping segment boundaries to "closure points" (positions no explicit candidate crosses): +0.003 %; not
  worth variable segment bounds.
- Per-lane adaptive statistics: −0.6 % against static passes at equal pass count.
- The "hash3 only when no rep" rule: replaced by ordinary 3-byte records (+0.12 % at depth 2).

## 3. Oracle, memory, time, risks

### 3.1 CPU oracle: preset family `opt14` / `opt16`

`gzc-core` gains `opt.rs`: the integer-only port in `/tmp/claude-1000/m5d/src/opt.rs` cleaned up (~450
lines; the DP body follows `zstd_opt.c` statement by statement, and with an ideal finder its ratio matches
libzstd's — its bytes cannot, since libzstd's tree is not reproducible without the tree), plus
`reference::find_cands` (the K2opt semantics of §2.1: the current `find_best` loop over two chains, the
"record when strictly longer, from length 3" rule and the `A`/`B` selection). `params.rs`:

```rust
pub struct OptParams { pub level: u8 /* 0 | 2, final pass */, pub target_length: u32 /* 32 */,
                       pub passes: u8 /* cheap intermediate passes */, pub seed: Seed /* BlockInit | Prior */, pub k: u8 /* 2 */ }
pub enum Hashes { Dfast, Single, Opt3 /* h4 depth `depth` + h3 depth 4 */ }
// MatchParams gains `opt: Option<OptParams>`; validate(): min_match 3 only with `opt`, `lazy == 0`,
// segment_log2 required (12), search_cap 64, hashes Opt3.
pub const OPT16: MatchParams = { hashes: Opt3, min_match: 3, depth: 32, lazy: 0, search_cap: 64, segment_log2: 12,
                                 opt: Some(OptParams { level: 2, target_length: 32, passes: 3, seed: BlockInit, k: 2 }) };
pub const OPT14: MatchParams = { ..OPT16, opt: Some(OptParams { level: 2, target_length: 32, passes: 1, seed: Prior, k: 2 }) };
```

`reference::parse` dispatches on `opt`. The parse is deterministic integer arithmetic (prices in 1/256 bit,
`<` / `<=` exactly as zstd), the candidate order is defined (chains nearest-first merged by position, A then
B), histograms are exact counts, the prior tables are constants in `codes.rs`, so the GPU can be byte-identical;
every tie rule becomes a `lazy::cases`-style test. Existing presets are untouched (`opt: None`).

Oracle targets on this sample (the ratio gates for S0):

| preset | recipe | sample ratio | libzstd on the sample |
|---|---|---:|---|
| `opt16` | h4 32 + h3 4, K 2, block init + 3 cheap passes + optLevel 2 final, tl 32, seg 4 K | **1.37211** | L14 1.36911, L16 1.37175, L19 1.37441 |
| `opt14` | same candidates, prior + cover literals, 1 cheap pass + optLevel 2 final | **1.37127** | |
| `opt14` 1-pass (stretch, needs per-kind priors) | prior + cover literals, final pass only | 1.36732 (global prior) | `m5-ratio-drivers`: 1.3673 vs L14 1.3669 with per-kind tables |

`m5-ratio-drivers.md` is the corpus-side reference for L14/L16 (its stratified sample A: L14 1.3669, L16
1.3695 vs L9 1.3366; the full-corpus L9 is 1.3379, so the corpus L14/L16 are ≈ 1.3684 / 1.3710). Our two
samples differ in absolute level (its L9 1.3366, mine 1.33856) but every paired delta agrees: mm3 −1.3 to
−1.6 %, K 2 saturates, three reps, ~3 passes from the block init, 4 KiB segments cheap, h3 depth +0.13 %.

### 3.2 VRAM per 64 KiB block

| buffer | today (lvl9seg) | opt16 / opt14 | note |
|---|---:|---:|---|
| data | 64 KiB | 64 KiB | |
| pred (K1) | 256 KiB | 512 KiB | 2 chains (`h4`, `h3`); dead after K2 → the DP trace (512 KiB) |
| best / candidates | 256 KiB | 512 KiB | 2 words per position |
| seqs | 192 KiB | 256 KiB | `MAX_SEQS` for min match 3 |
| histograms + prices | – | 1.5 KiB | per block, between passes |
| counts, frame_len, frames stride | ~4 KiB + stride | same | |
| **sum** | **≈ 0.77 MiB** | **≈ 1.35 MiB** | batch max at 6144 MiB, i3: ≈ 5118 → ≈ 2900 blocks |

Workgroup memory: 23 KiB per 32-lane workgroup (§2.2). K1's head tables are unchanged (or gone with track B).
No per-lane global scratch beyond the trace.

### 3.3 Projected kernel time

Calibration: `lvl9seg` at 64 KiB, b5118 i3 on the 5090 (`docs/results/speed2-log.md`): K1 13.84, K2 8.83,
K3 6.41, K4 1.97, K5 4.52 ms per 320 MiB batch = 2.70 / 1.73 / 1.25 / 0.38 / 0.88 µs per block, copies
1.76 µs per block, e2e 7068 MB/s = 9.3 µs per block. R4's model put K3's lane loop at ~2.6 K iterations per
lane; the DP does 4096 per lane per pass with 1–1.5× the instructions per trip (§2.2) at a third of the
occupancy. Per 64 KiB block:

| kernel | opt16 (5090) | opt14 (5090) | basis |
|---|---:|---:|---|
| K1, 2 chains | 5.4 (persistent grid) or ~2 (track B sort) | same | 2.70 µs per chain today; R5 option C 2–4 ms per 2559 128K blocks per key |
| K2opt, 32-deep h4 walk + 4-deep h3 walk | 2.1 (sorted array ~1.5) | same | 1.73 µs today + a short second walk |
| seed prologue (cover histogram) | – | 0.3 | one pass over the candidate words |
| K3opt, per pass | 3.0–4.0 (cheap 2.5–3.2) | same | 1.25 × 4096/2600 × 1.5 instr × 1.0–1.3 occupancy |
| K3opt, all passes | **10.5–13.6** (3 cheap + final) | **5.5–7.2** (1 cheap + final) | |
| K4 + K5 | 1.6 | 1.6 | 1.55× sequences, fewer literals |
| copies | 1.8 | 1.8 | unchanged |
| **sum per block** | **21–25 µs (persistent K1) / 18–21 (sorted)** | **16–18 / 13–15** | |
| **throughput** | **2.7–3.1 GB/s / 3.1–3.6** | **3.6–4.0 / 4.3–5.0** | 5090, kernel-bound; e2e ≈ 0.9× |
| 4060 (÷5–7, R7) | 0.4–0.6 GB/s | 0.55–0.8 GB/s | the DP's per-lane traffic (~1.5 MB per block per pass: candidates in, trace out and in) is 5–6 µs per block per pass at 272 GB/s, the same order as its issue time |

These are ±2× projections in R4's sense; the one measured anchor is that the DP is ~1.6× the positions and
~1.5× the work of the segmented lazy lane, per pass. The 4060 does not reach 1.25 GB/s with either preset;
both are "quality" presets for links below ~3–5 Gbit/s or for offline repacks. Each DP pass removed saves
~3 µs per block for the ratio listed in §2.4.

### 3.4 Risks

1. **K3opt occupancy and divergence.** 5 warps per SM with a 23 KiB ring may not hide the probe latency; lanes
   in different phases (skip vs series) diverge every trip. Mitigations: 16-lane workgroups, the ring in local
   memory, a 16 B node, or targetLength 16 (`m5-ratio-drivers`: −0.02 %). Measure with the naive port first
   (stage S2) before anything else.
2. **Pass count.** The DP passes are 50–60 % of the kernel time. The prior seed cuts `opt16` from 4 to 3
   passes at +0.01 % over L16 (§2.4, "prior + cover, + 2 cheap"); per-kind priors might cut `opt14` to one
   pass. The pass count is a preset constant, so it is a ratio/speed dial, not a design risk.
3. **Finder gap.** +0.23 % to the ideal finder remains (1.37487, > L19). The sort-based LCP finder is the only
   candidate and its cost is unmeasured (R5 rejected a full suffix array at ~50 ms per 128K batch; the 8-byte
   prefix sort of `m5-rt-npu.md` §4.4 is cheaper but unproven).
4. **Byte identity.** The port must fix every tie rule (`<=` in the literal extension, `<` in relaxation,
   descending length order, record order, `newRep` numbering) and the K2opt selection rule (nearest + longest
   with "longer, then nearer" ties across two merged chains). The cap-64 rule interacts with A/B: a capped
   record stops the walk, so a farther, truly longer match is never seen — this is in the oracle too.
5. **VRAM.** 1.35 MiB per block cuts the batch by 43 %; `resolve_max_batch` handles it, but the 4060's 8 GB
   sets the practical batch (≈ 1700 blocks at 6 GiB with i2).
6. **Entropy coder headroom is small.** Our writer is +0.12 % over libzstd at L9 with the same parse class,
   and `m5-ratio-drivers` found it costs nothing; the numbers above transfer.
7. **Sample vs corpus.** All numbers are every-50th-block (124 MiB). The ranking was identical on a 40-block
   pilot and matches the independent sample of `m5-ratio-drivers.md`; the full-corpus run is stage S0's gate.
8. **Prior tables in-sample.** The prior was summed from the sample's own recipe output; `m5-ratio-drivers`
   validated held-out tables (≤ 0.0002 change), but S0 must train them on a disjoint block set.

## 4. Staged plan with checkpoints

Each stage has a measurable gate on the 1/50 sample (`cd /tmp/claude-1000/m5d && cargo build --release
--offline && ./target/release/m5d 50 100000 <set>`), then the full corpus with `gzc-bench ref`.

- **S0 — oracle (CPU only).** `opt.rs` + `find_cands` + presets + prior tables (trained on blocks not in the
  1/50 sample), differential tests (tie cases, segment cases, libzstd round trip of every synthetic block and
  4000 corpus blocks). Gate: sample ratio `opt16` = 1.37211 ± 0.0003 (row Q1), `opt14` = 1.37127 ± 0.0005
  (Q6; the prior differs), and full corpus `opt16` >= libzstd L16 at 64 KiB, `opt14` >= L14 (corpus L14/L16
  from `m5-ratio-drivers`' stratified estimate ≈ 1.3684 / 1.3710, to be confirmed with `gzc-bench cpu`). Also
  dump the per-pass histograms so the GPU passes can be checked one at a time.
- **S1 — K1 with the `h3` key + K2opt (GPU).** `N_HASHES = 2` with `hash3` in `common.wgsl`; K2opt writes the
  two candidate words. Gate: byte-identical candidate words against `find_cands` on the corpus sample; K1 + K2
  per block ≤ 1.3× today's 4.4 µs (persistent K1); re-measure after track B lands.
- **S2 — K3opt naive port, one pass, block-init prices, no histogram.** Ratio gate: = oracle with `passes: 0,
  seed: BlockInit` (1.3354-class: a correctness gate, not a quality one). Time gate: the per-pass cost; above
  ~6 µs per block on the 5090, iterate on occupancy (§3.4.1) before adding passes.
- **S3 — histogram epilogue + price prologue, N passes, prior seed.** Gate: byte-identical to the oracle for
  `passes 0, 1, 3` and both seeds; ratio = the S0 numbers; per-pass time × passes.
- **S4 — fix-up and integration.** `main_fixup` reuse, `MAX_SEQS`, batch sizing, `--verify` on the corpus,
  `speed2-log.md` rows (median of 3). Gate: full corpus `opt16` ≥ L16, `opt14` ≥ L14 + 0.1 %, throughput in the
  §3.3 band.
- **S5 — optional finder upgrade.** Evaluate the LCP-neighbour finder (`m5-rt-npu.md` §4.4) as a K2opt
  alternative; target the QC row (1.37487, > L19). Cheaper first step: the `h8` chain (+0.03 %, Q3) if K1/K2
  time allows.
- **S6 — speed dials.** Passes (3 vs 4 for `opt16`, 1 vs 2 for `opt14`), per-kind priors from the DDS header
  (R9's mechanism), targetLength 16, optLevel 0 final (−0.05 %), per-format K1/K2 presets (R9's DXT1 grid rule
  applies unchanged).

## Appendix A: method and reproduction

- Prototype: `/tmp/claude-1000/m5d/src/{cands.rs, opt.rs, main.rs}`. `m5d <every> <max_blocks> <set>` with
  set ∈ `zstd`, `dp`, `all`, `dump` (writes `prior.txt`), or a variant-name prefix (`A` … `Q`); `CHECK=1`
  decodes every frame with libzstd. 8 rayon threads; `all` takes 192 s, `Q` 59 s, `P` 18 s.
- Candidates: `hash_width(4)` chains from `gzc_core::hash::compute_preds` (16-bit hash, so a chain also holds
  collisions, filtered by the compare), `hash_long` for the 8-byte key, `ZSTD_hash3` (16 bits) for the 3-byte
  chain or slot; "ideal" = the 4-byte chain walked 4096 deep without the 64-byte cap.
- The DP port keeps zstd's series structure, stretch/sequence conversion, `ZSTD_OPT_NUM`, `sufficient_len`,
  price arithmetic (`bitWeight`/`fracWeight`, `litPriceMax`, `+256/5`), `ZSTD_updateStats`/`setBasePrices`
  for the adaptive mode, the optLevel 0 skip and early abort, the optLevel >= 1 match+1-literal rule, and
  `ZSTD_newRep`. Sequences go through `gzc_core::lazy::encode_raw` (decoder reps) and `write_frame`.
- Sample ratios are real bytes / frame bytes over all blocks; DDS and NIF are reported separately in the
  result files.

## Appendix B: full variant tables (1/50 sample, 2016 blocks)

`/tmp/claude-1000/m5d/full50.txt` (libzstd rows, `ours`, sets A–H), `p50.txt` (P: preset candidates and the
raw-literal prior), `q50.txt` (Q: 3-byte chain depth, cover seed, ceilings). Per-block work counters: `iters` =
in-series positions, `litskips` = positions with no candidate, `relax` = price relaxations, `probes` =
rep/hash3 compares, `series` = series per block. Selected rows:

| id | variant | ratio |
|---|---|---:|
| A1 | h4 depth 64, K 8, zstd hash3 slot, adaptive, whole block, optLevel 2 tl128 | 1.36893 |
| A2 | same with the ideal finder | 1.37152 |
| A0 | h4 depth 64, K 8, adaptive, whole, optLevel 0 tl32 (btopt) | 1.36743 |
| B2 | h4 depth 32 cap 64, K 2, adaptive, whole | 1.36855 |
| C4 | B2 candidates, block init + 3 static passes, whole | 1.37092 |
| D3 | same, seg4k | 1.37018 |
| G2 | h4 32 + h8 32, K 2, init + 3 cheap, seg4k, tl128 final | 1.37067 |
| P2 | G2 with tl32 final | 1.37057 |
| P0 | h4 32, K 2, init + 2 cheap, optLevel 0 final (btopt) | 1.36866 |
| **Q1** | **h4 32 + h3 chain 4, K 2, init + 3 cheap, optLevel 2 tl32 final, seg4k** | **1.37211** |
| Q3 | Q1 + h8 32 | 1.37255 |
| Q5 | Q3 with K 3 | 1.37275 |
| **Q6** | **Q1 candidates, prior + cover literals, 1 cheap + final** | **1.37127** |
| Q9 | Q1 candidates, prior + cover literals, 2 cheap + final | 1.37189 |
| Q7 | Q1 candidates, prior + cover literals, final only | 1.36732 |
| QC | ideal 4-byte + h3 chain 4, init + 3 cheap | 1.37487 |
| H0 / H1 / H2 | G2 with 0 / 1 / 2 rep candidates | 1.36454 / 1.36803 / 1.36955 |
| H5 / H6 | G2 with 8 KiB / 2 KiB segments | 1.37107 / 1.36991 |
