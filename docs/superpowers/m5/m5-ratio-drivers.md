# M5: what drives the libzstd L14/L16 ratio over L9 (64 KiB independent blocks)

CPU-only measurements. Scratch code is in `/tmp/claude-1000/m5r/` (a Rust crate using `zstd` 0.14 / libzstd 1.5.7
plus `gzc-core` by path, `src/main.rs`; raw result lines are in `res_*.txt`). The repo was not modified. Runs used 8 rayon threads.

## TL;DR

1. **The L9 to L14 gain is two things multiplied together: minMatch 3 and a priced (optimal) parse.** Neither one gets
   there alone. On total (weighted) ratio, sample A:
   - L9: 1.3366
   - opt parse, mm4 (libzstd `L14,mm=4`): 1.3495 (+0.97 %)
   - mm3 with a heuristic lazy2 parse (our prototype): 1.3471 (+0.79 %)
   - both (L14): 1.3669 (+2.27 %)

   On NIF, minMatch 3 alone already gives about 85 % of the gain (lazy2 mm3 NIF +9.6 % vs L14 +11.2 %). On DDS, which is
   95 % of the bytes, mm3 without prices gives only +0.1 to +0.5 %. DXT1 needs both.
2. **Everything else in btopt/btultra hardly matters on this data:**
   - Candidates per position: K=2 saturates, K=1 costs 0.1 %.
   - 4-byte chain depth: D=4 to 16 is enough (the chains are short on DXT data).
   - targetLength: 16 or more saturates.
   - searchLog: 4 gives almost all of it (sl=1 still gets +1.95 %).
   - The zstd "match+1 literal" trick, dynamic in-block statistics and the 0.2-bit match penalty are not needed.
3. **Things that do matter:**
   - A 3-byte candidate source, walked 2 deep across the whole 64 KiB block. Removing it gives back most of the mm3 gain,
     and capping its distance at 1 KiB costs 1 %.
   - Exact rep codes per DP node. No reps costs 0.5 to 0.6 % (NIF −2.5 %). Two reps cost 0.06 to 0.1 %.
   - **Literal prices.** The first-block zstd defaults (raw-block literal histogram, flat code tables) need about 3
     passes. With good starting prices one pass reaches L14 and two passes reach L16.
4. **Splitting the optimal parse into 4 KiB independent segments costs about 0.015 %** (1 KiB costs 0.07 %). It is free,
   so the DP can run one lane per segment, as lvl9seg does.
5. **Minimum sufficient configurations (held-out price tables, both samples):**
   - **L14 class (1 DP pass):** 1.3673 / 1.3702 vs L14 1.3669 / 1.3696.
   - **L16 class (same candidates, 2 DP passes):** 1.3696 / 1.3724 vs L16 1.3695 / 1.3723.

   Full recipes are in the recommendation section.

## Sample

The sample is a stratified every-Nth-block selection over `data/corpus`, taking only `.dds` and `.nif` files, cut into
64 KiB blocks (zero-padded like `chunk_file`). DDS files are classified by the FourCC at offset 84.

| kind | corpus blocks | stride | sample A blocks |
|---|---:|---:|---:|
| DDS DXT1 | 29,175 | 32 | 912 |
| DDS DXT5 | 66,206 | 32 | 2,069 |
| DDS other (raw) | 215 | 32 | 7 |
| NIF | 5,158 | 8 | 645 |

- "TOTAL" is weighted by stride, so it estimates the full-corpus ratio: real bytes over compressed bytes, with
  compressed = the whole padded 64 KiB frame, as `gzc-bench cpu` computes it.
- Sample A (phase = stride/2) gives L9 = 1.3366. The full-corpus 64 KiB L9 is 1.3379.
- Sample B is a disjoint phase-5 sample (3,633 blocks). It gives L9 = 1.3393, L14 = 1.3696 and L16 = 1.3723, so the
  two samples differ by about 0.2 to 0.4 % in absolute terms. **Only paired comparisons (same sample) are meaningful.**
- The global price tables below were trained on B and evaluated on A (and the reverse) to rule out overfitting.
- Sanity check: gzc `lvl9` on sample A gives 1.3381 (`lvl9seg` also 1.3381), matching the known +0.1 % over libzstd L9.
  So gzc's frame writer (used for all our prototypes) costs nothing against libzstd's.

## 1. libzstd baseline curve (sample A)

| config | DXT1 | DXT5 | NIF | DDS | TOTAL | Δ TOTAL vs L9 |
|---|---:|---:|---:|---:|---:|---:|
| L9 (lazy2, mm4, sl5, tl8) | 1.3569 | 1.3128 | 1.6122 | 1.3261 | 1.3366 | — |
| L12 (btlazy2) | 1.3586 | 1.3148 | 1.6142 | 1.3281 | 1.3385 | +0.14 % |
| L13 (btopt, mm4, sl3, tl12) | 1.3751 | 1.3194 | 1.6449 | 1.3361 | 1.3473 | +0.80 % |
| **L14** (btopt, mm3, sl4, tl32) | 1.3940 | 1.3342 | 1.7927 | 1.3520 | **1.3669** | **+2.27 %** |
| L15 (btopt, sl6, tl256) | 1.3941 | 1.3353 | 1.7929 | 1.3528 | 1.3677 | +2.33 % |
| **L16** (btultra, sl6, tl128) | 1.3944 | 1.3373 | 1.8066 | 1.3543 | **1.3695** | **+2.46 %** |
| L17 | 1.3944 | 1.3373 | 1.8066 | 1.3543 | 1.3695 | +2.46 % |
| L19 (btultra2) | 1.3969 | 1.3399 | 1.8098 | 1.3570 | 1.3721 | +2.66 % |

Per kind, L14 over L9 is DDS +1.95 % (DXT1 +2.7 %, DXT5 +1.6 %) and NIF +11.2 %. That matches the earlier
1 GB-sample figures.

## 2. libzstd ablations (sample A, TOTAL; per-kind values in `res_ablate.txt`)

| ablation | TOTAL | Δ vs its base | notes |
|---|---:|---:|---|
| **L14, mm=4** | 1.3495 | −1.27 % | DXT1 1.3753, NIF 1.6519: **mm3 is the main L14 driver** |
| L14, mm=5 | 1.2852 | −5.98 % | DXT1 collapses (1.2284): 4-byte matches are DXT1's main currency |
| L16, mm=4 | 1.3508 | −1.36 % | |
| L13, mm=3 (sl3, tl12) | 1.3645 | +1.28 % vs L13 | L13 to L14 is almost entirely mm3 |
| L14, strat=btultra | 1.3686 | +0.12 % | accurate fractional prices |
| L14, strat=btultra2 | 1.3713 | +0.32 % | btultra2 = a stats pre-pass (a second iteration) |
| L16, strat=btopt | 1.3676 | −0.14 % vs L16 | L16 − L14 = btultra (+0.12) + sl6 (+0.05) |
| L16, strat=btultra2 | 1.3721 | +0.19 % vs L16 | = L19 |
| L14, sl = 1 / 2 / 3 / 6 / 8 | 1.3626 / 1.3639 / 1.3650 / 1.3676 / 1.3676 | | saturates at sl6; sl1 (2 bt compares) still +1.95 % over L9 |
| L16, sl = 4 / 8 / 10 | 1.3687 / 1.3695 / 1.3695 | | |
| L14, tl = 4 / 8 / 16 / 64 / 128 / 999 | 1.3503 / 1.3596 / 1.3667 / 1.3669 / 1.3669 / 1.3670 | | tl ≥ 16 saturates; tl 8 −0.5 % |
| L16, tl = 32 / 999 | 1.3694 / 1.3695 | | |
| L9 with mm=3; greedy with mm=3; L12 with mm=3 | identical to their mm4 runs | | **libzstd cannot test "lazy with mm3"**: the greedy/lazy/btlazy2 match finders clamp mls to at least 4 (`BOUNDED(4, minMatch, 6)` in `zstd_lazy.c`). Our own lazy prototype covers this (section 3). |

## 3. Our prototypes

The prototype is a forward-DP optimal parse (`opt_block` / `dp` in `main.rs`). It uses gzc-core's `encode_raw` and
`write_frame` for exact sizes, and every frame round-trips through libzstd (`VERIFY=1`).

**Match finder.** Candidates at each position are:
- the rep candidates;
- a 3-byte-hash chain ("h3", depth `h3d`, distance below `2^h3log`, like zstd's hashTable3);
- a 4-byte-hash chain (17-bit hash, depth `D`). The walk keeps strictly increasing lengths, as zstd's `getAllMatches`
  does, and keeps the longest `K` explicit candidates.

**DP.**
- Each node holds price, litlen and reps; reps are exact per node.
- Every match length `startML..len` of each candidate is relaxed, as zstd does.
- A literal step uses the incremental LL price.
- A candidate longer than `T` commits immediately (zstd's `sufficient_len`).
- Prices are static per pass: `log2(sum+1) − log2(count+1)` plus extra bits, floats, with a 1-bit minimum per literal.

**Price initialisation (`init`):**
- 0: zstd first-block defaults (raw-block literal histogram, zstd base LL/OF frequencies, flat ML).
- 1: stats of the gzc lvl9 parse.
- 2: global per-kind LL/ML/OF tables (121 numbers per kind), with literals from the raw-block histogram.
- 3: global code tables, with literals from the lvl9 parse.
- 5: global code tables, with the literal histogram estimated **without a parse**: bytes not covered by any candidate
  match (a prefix-max of `p + len` over the candidates).

`it` = the number of DP passes; each pass after the first re-prices from the previous pass's output.

**Segmentation (`seg`).** Each `2^seg`-byte segment is parsed on its own: matches are clamped to the segment and a
segment starts with empty reps (segment 0 with the initial reps). Offsets are then re-encoded with the block's true
rep history, as `lazy_parse_segmented` does.

**Lazy prototype (`lz`).** A zstd-lazy-gain-rule lazy/lazy2 over the same candidates, with no prices. Calibration:
`lz=2, mm=4, D=32` gives 1.3380, which matches gzc lvl9 (1.3381).

In the tables below, `init=2` rows used in-sample tables. Held-out tables change TOTAL by at most 0.0002 (section 5).

### 3a. What mm3 alone buys (lazy2, no prices; sample A)

| config | DXT1 | DXT5 | NIF | DDS | TOTAL |
|---|---:|---:|---:|---:|---:|
| lazy2 mm4 D32 (≈ L9) | 1.3586 | 1.3132 | 1.6345 | 1.3269 | 1.3380 |
| lazy2 mm3, h3 unlimited | 1.3546 | 1.3211 | 1.7635 | 1.3313 | 1.3459 |
| lazy2 mm3, h3 distance < 16 K | 1.3576 | 1.3215 | 1.7663 | 1.3325 | 1.3471 |
| lazy2 mm3, h3 distance < 4 K | 1.3564 | 1.3164 | 1.7689 | 1.3285 | 1.3433 |
| (L14 for reference) | 1.3940 | 1.3342 | 1.7927 | 1.3520 | 1.3669 |

**The answer to "does minMatch 3 alone give most of the NIF gain":** yes for NIF (+9.6 to 9.7 % of the +11.2 %), no for
DDS. Without prices, DXT1 even loses a little with 3-byte matches: the lazy gain rule takes every short match, and many
of them cost more than their literals.

### 3b. Optimal-parse ablations (sample A; base = `mm3, D16, K∞, T32, h3d1, 3 reps, whole block`)

| factor | setting | TOTAL | Δ | notes |
|---|---|---:|---:|---|
| **passes / init** | init=0 it=1 / 2 / 3 / 4 | 1.3290 / 1.3625 / 1.3677 / 1.3684 | | zstd default prices are very poor for DXT1 in pass 1 |
| | init=1 (lvl9 stats) it=1 / 2 | 1.3493 / 1.3551 | | **bad**: lvl9 has no ML=3, so 3-byte matches get priced out and stay out |
| | init=2 it=1 / 2 / 3 | 1.3604 / 1.3678 / 1.3685 | | |
| | init=3 it=1 (h3d2) | 1.3673 | | lvl9 literals + global codes in one pass reaches L14 |
| | **init=5 it=1 (h3d2)** | **1.3677** | | parse-free literal estimate is as good as the lvl9 one |
| | init=4 it=1 (bytes with no candidate) | 1.3664 | | weaker estimator |
| which prices need pass 2 | pass 2 with global codes but own-literal stats | 1.3678 | = full pass 2 | **literal prices are what pass 2 fixes** |
| | pass 2 with own codes but raw-block literals | 1.3645 | −0.33 % | |
| **minMatch** | mm4 (init=2 it=2) | 1.3469 | −1.53 % | |
| **3-byte source** | none (`h3=0`) | 1.3488 | −1.39 % | most of mm3's value comes from this one extra candidate |
| | h3 distance < 1 K / 4 K / 16 K / 64 K | 1.3549 / 1.3593 / 1.3660 / 1.3678 | | **needs the full 64 KiB window** |
| | **h3 depth 1 / 2 / 4** | 1.3678 / **1.3696** / 1.3699 | | depth 2 is worth +0.13 % |
| **reps** | 0 / 1 / 2 / 3 rep codes per node | 1.3617 / 1.3656 / 1.3669 / 1.3678 | | NIF 1.7633 / 1.7879 / 1.8020 / 1.8100 |
| **K** (explicit candidates kept) | 1 / 2 / 4 / 8 / all | 1.3667 / 1.3677 / 1.3678 / – / 1.3678 | | K=2 saturates |
| **D** (4-byte chain depth) | 4 / 8 / 16 / 32 / 64 | 1.3667 / 1.3673 / 1.3677 / 1.3680 / 1.3633 (it=2) | | chains are short: 1.5 steps per byte at D=16 |
| | D4, K1 | 1.3662 | | |
| **T** (commit length) | 16 / 32 / 128 / 999 | 1.3676 / 1.3678 / 1.3678 / 1.3679 | | T=999 multiplies the relax work by 4.5 for nothing |
| match penalty | 0.2 bit vs 0 | 1.3678 vs 1.3677 | 0 | |
| pick the best pass instead of the last | | +0.0001 | 0 | |

In the D row, 1.3633 is an it=2 run and 1.3680 (D32) is it=3, so the row is only roughly paired. The it=3 D-sweep
shows D=4 1.3667, D=8 1.3673, D=16 1.3677 and D=32 1.3680.

### 3c. Segmenting the optimal parse (init=2, it=2, h3d2; sample A)

| segment | whole block | 32 K | 16 K | 8 K | **4 K** | 2 K | 1 K |
|---|---:|---:|---:|---:|---:|---:|---:|
| TOTAL | 1.3696 | 1.3696 | 1.3696 | 1.3695 | **1.3694** | 1.3692 | 1.3687 |
| NIF | 1.8100 | 1.8100 | 1.8097 | 1.8095 | 1.8084 | 1.8068 | 1.8035 |

- The loss is mostly NIF, from the empty rep history at segment starts and from matches cut at segment ends.
- Seeding each segment's reps from the previous pass changes nothing (1.3694).
- With one pass (init=2, it=1), 4 K costs 0.02 % (1.3620 to 1.3617).

## 4. Why these factors matter (code statistics of the converged parse)

| kind | ML=3 share | ML=4 share | rep codes (OF 0/1) | offset ≥ 1 K |
|---|---:|---:|---:|---:|
| DXT1 | 22 % | 71 % | 6 % | 64 % |
| DXT5 | 46 % | 29 % | 13 % | 69 % |
| NIF | 49 % | 18 % | 66 % | 11 % |

- **DDS.** The payload is 3- and 4-byte matches at long offsets, found anywhere in the 64 KiB block (color endpoints,
  alpha endpoints, index words). Each one saves only a few bits, and many would cost more than the literals they
  replace. The ratio therefore comes from (a) finding the 3-byte ones anywhere in the block and (b) pricing each one
  against its literals. The literal price in particular needs the block's own post-parse literal histogram. Candidate
  quality beyond the nearest one or two per length (K, D) is worth nothing.
- **NIF.** Rep-heavy, with short matches. mm3 plus exact per-node rep handling carries it, and a heuristic parse already
  gets most of it.

## 5. Recommended minimum sufficient configurations

Global per-kind LL/ML/OF tables were trained on the other sample (held out). Results as A / B:

| config | TOTAL A | TOTAL B | vs L14 (A / B) | vs L16 (A / B) |
|---|---:|---:|---:|---:|
| libzstd L14 | 1.3669 | 1.3696 | — | |
| libzstd L16 | 1.3695 | 1.3723 | | — |
| **L14 class:** mm3, D16, K2, h3d2, 3 reps, T32, seg 4 K, init=5, **1 pass** | **1.3673** | **1.3702** | +0.03 % / +0.04 % | |
| same, D8 | 1.3667 | 1.3696 | −0.01 % / 0.00 % | |
| same, D8, h3d4 | 1.3670 | 1.3699 | +0.01 % / +0.02 % | |
| same, D4, T16 | 1.3659 | 1.3688 | −0.07 % / −0.06 % | |
| same, D8, h3d1, T16 | 1.3647 | 1.3676 | −0.16 % | |
| **L14 with margin / L16 class:** mm3, D8, K2, h3d2, 3 reps, T32, seg 4 K, init=2, **2 passes** | **1.3688** | **1.3716** | +0.14 % / +0.15 % | −0.05 % / −0.05 % |
| **L16 class:** mm3, D16, K2, h3d4, 3 reps, T32, seg 4 K, init=2, **2 passes** | **1.3696** | **1.3724** | | +0.01 % / +0.01 % |
| upper end found: init=5, 3 passes, D16, K4, h3d4, seg 4 K | 1.3704 | 1.3732 | | +0.07 % / +0.07 % (≈ L19) |

The one-pass L14-class result clears L14 by only +0.03 to 0.04 %. **If L14 is a hard floor, use the 2-pass
variant.** Its second pass reruns only the DP (the candidates don't depend on the parse), and it clears L14 by about
0.15 %.

### What the design can drop (measured ≤ 0.02 % each)
- the bt match finder: a plain 4-byte hash chain with D=8 to 16 is enough;
- more than 2 explicit candidates per position;
- targetLength above 32 (use 16 to 32; commit longer matches immediately);
- the 4096-position series limit (`ZSTD_OPT_NUM`) and whole-block DP: 4 KiB independent segments are fine;
- zstd's dynamic in-block stat updates;
- the match+1-literal re-check;
- the 0.2-bit match penalty;
- rep seeding at segment starts;
- btultra's fractional-vs-integer weight distinction (our prices are simply accurate).

### What it must keep
| feature | cost of dropping it |
|---|---|
| minMatch 3 | −1.5 % |
| a 3-byte candidate source covering the full 64 KiB window, depth ≥ 2 | −1.4 % without it; −0.3 % at a 4 K distance cap; −0.13 % at depth 1 |
| exact rep codes per DP node, all three | −0.6 % with none; −0.1 % with 2 |
| a real price-driven DP rather than lazy gain rules | −1.5 % |
| decent literal prices before the first DP pass (candidate-coverage histogram or a previous pass) | −0.5 to −3 % |
| per-kind global LL/ML/OF tables (DXT1 / DXT5 / NIF) as starting code prices | −0.5 % with zstd defaults |

### Expected work per input byte (measured by the prototype, recommended config)

**K1:** today's 4-byte chain plus a second 3-byte chain (head 64 K entries plus `prev3[64 K]` per block).

**K2 (candidates):**
- about 1.4 to 1.8 chain steps per byte in total (D8 to D16 on the 4-byte chain plus depth 2 on the 3-byte chain). This
  is less than lvl9's 32-deep walk, because the real chains on DXT data are short.
- output: up to 2 (len, off) pairs per position (0.44 candidates per position on average, including reps, so most
  positions have none). Lengths can stay capped (for example at 64) as today: the DP extends only a match above T, when it commits.

**Literal estimate (init=5):** one prefix-max scan of `p + len` and a 256-bin histogram per block. Plus 3 × 121 constant
table entries per kind.

**K3 (DP), one lane per 4 KiB segment (16 per block, like lvl9seg):**
- per position, 3 rep compares (at 92 % of positions: positions inside a long commit are skipped);
- about 0.9 relax updates per byte at T=32 (a relax is a table-lookup add, a compare and a store);
- 1 literal update;
- state: a sliding forward window of about T+1 nodes (price, litlen, 3 reps), plus a backtrack record per position
  (mlen, offset or candidate index; about 4 KiB × 6 to 8 B per segment).
- Unlike lazy, which jumps over matched bytes (0.73 gathers per byte), the DP visits every position. Expect K3 work of
  roughly 3 to 5 times lvl9seg's per byte, and twice that for the 2-pass variant: pass 2 reruns the DP only, after a
  histogram of pass 1's literals and codes.

**K4/K5:** unchanged. There are about 6,300 sequences per block vs about 3,800 for lvl9 (1.65 times more), so K4 grows.

## Caveats
- The prototype prices use f32 `log2`. A GPU version would use fixed-point tables, which should be within noise.
  Frames come from gzc's writer (auto FSE mode choice, Huffman literals) and decode with libzstd.
- The lazy/mm3 prototype uses zstd's gain rules over our candidates. zstd itself cannot run lazy with mm3, so this is
  the closest available answer, not zstd's.
- The two samples differ by 0.2 to 0.4 % in absolute ratio. All conclusions rest on paired deltas, which agree across
  A and B to within 0.01 %.
- Reproduce with `cd /tmp/claude-1000/m5r && cargo build --release --offline`, then:
  - `./target/release/m5r sample 32 8` (to rebuild sample A)
  - `./target/release/m5r zstd L14 L14,mm=4 …`
  - `GSTATS=gstats_train.txt ./target/release/m5r opt init=5,it=1,D=16,K=2,h3d=2,T=32,seg=12`
  - `SAMPLE=… PHASE=5 m5r sample 32 8` and `DUMP=path` regenerate sample B and the tables.
