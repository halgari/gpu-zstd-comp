# r3-match-finding: GPU-built candidates that make a cheaper parse reach L16

Agent r3-match-finding, 2026-10-01. Research only: nothing in the repository was changed.

- Scratch workspace: `~/.cache/gzc-m7/r3-match-finding/ws/`, a copy of `gzc-core`. The build outputs have been deleted.
- Kept in `.superpowers/m7-research/artifacts/r3-match-finding/`:
  - `r3.rs`, the CPU tool;
  - `opt-ext.diff`, which lets the oracle DP take arbitrary per-position candidate lists and a runtime prior override;
  - `full64.txt`, the raw full-corpus results.
- The tool also uses a10's split writer (`split.rs` with the matching `fse.rs`, `huffman.rs` and `seqenc.rs`), copied verbatim; see `NOTE.txt`.

## TL;DR

1. **Exact candidates (a suffix array) gain +0.24 % over today's, at every pass count. Almost all of that gain is in B, the longest match.**
   - A full Pareto frontier adds only +0.014 % over A/B.
   - Taking A exactly (instead of from the chains) adds nothing.
2. **Three cheap stride-4 hash chains get within 0.01 % of the exact suffix array.** I call this recipe **S3**: keys of 6, 10 and 12 bytes, each 16 deep, hashed only at positions p ≡ 0 (mod 4).
   - It costs about +1 µs/block of K1 over a06's recipe.
   - A full GPU suffix array is not worth it: about 20 radix digit passes, 2–3× today's K1+K2, for +0.03 %.
3. **Headline: with S3 plus block splitting (B1), ONE DP pass clears L16 at 64 KiB and 32 KiB.**
   - Full corpus, 64 KiB: **1.37204, +0.076 % over L16**. Today's 4-pass opt16 gives 1.37144.
   - That turns K3 from 4 passes into 1. Estimated **≈ 21–26 µs/block on the 5090, against 44.7** (1.7–2.1×).
   - At 16 KiB a single pass is 0.02–0.06 % short. 2 passes there give +0.14 %.

## Setup

**Tool.** `r3`, a `gzc-core` example.
- Candidate sources:
  - any mix of hash chains, walked like `find_cands` (nearest first, merged), with per-chain key width, insert/search stride and depth;
  - the exact SA frontier: a suffix array plus LCP and a rank segment tree, giving for L = 3, then last length + 1, the exact nearest q < p with LCP ≥ L.
- Selection policies cut a frontier down to 2–3 records.
- Lists go into the real `opt` DP (`get_all_matches` generalised to a list), then the real frame writer, optionally a10's split writer.
- Checks:
  - `CHECK=1` asserts that chain + A/B reproduces `find_cands` exactly;
  - opt16 = 1.37211 and opt14 = 1.37118 on the sample, and pf + split = 1.36979, all as reported before;
  - `VERIFY=1` decodes every frame with libzstd (sample runs).

**Data.**
- The 1/50 sample at offset 0: L16 1.37175, L14 1.36911.
- Held out: offset 25.
- 32 KiB and 16 KiB samples: every 50th block.
- Full 64 KiB corpus: 100,754 blocks.

**Notation.**
- Schedules: **1p** is the Prior seed plus the final pass only. **2p** is the Prior seed, one cheap pass and the final pass (opt14's schedule). **4p** is opt16's.
- **+split** means a10's per-frame multi-block writer, with the smaller of split and unsplit kept.
- **S3** = h3 d4 + h4 d8 + three stride-4 hashed chains (6-, 10- and 12-byte keys), each d16. The 10-byte chain is a06's h10.

## Q2: candidate quality against quantity (64 KiB sample, 2p, no split)

| candidates | ratio | vs L16 |
|---|---:|---:|
| today (h4 d32 + h3 d4), A/B | 1.37118 | −0.042 % |
| a06 (h4 d8 + h3 d4 + h10 s4 d16), A/B | 1.37302 | +0.093 % |
| a06, full walk frontier | 1.37310 | +0.098 % |
| exact SA, B only | 1.37319 | +0.105 % |
| exact SA, A/B | 1.37433 | +0.188 % |
| exact SA, A/B, lengths capped at 64 (= exact chains h3 d4 + h4 d4096) | 1.37422 | +0.180 % |
| exact SA, **top2** (the two longest) | 1.37447 | +0.198 % |
| exact SA, **A + 2 longest** | **1.37451** | +0.201 % |
| exact SA, full Pareto frontier | 1.37452 | +0.202 % |
| exact SA, longest per offset code | 1.37452 | identical to the full frontier |
| mix: A from a06's chains, frontier from SA | 1.37452 | |
| mix: A from SA, B from a06's chains | 1.37314 | |
| exact SA, Prior retrained on SA output (held-out offset 25) | 1.37455 | +0.003 % from retraining |

**Findings.**
- **What matters is B (the longest match), not more records.**
  - The full frontier is worth +0.014 % over A/B.
  - Three records (A + 2 longest) capture all of that, and top2 captures 70 % of it. That is the same 8-byte format: just keep the second-longest instead of the nearest.
  - Swapping in an exact A gains nothing.
- **Offset-code collapse is exact.** `match_price` depends only on the offset code, so within one code only the longest record matters. Keeping the longest per offset code gave a result identical to the full frontier, so the reps effect is nil. The "ideal" set is therefore at most about 16 records, and in practice 1–2.
- **The other candidate ideas from the brief:**
  - **Rep-aligned candidates** are redundant, because the DP already probes the exact reps at every node.
  - **Continuation candidates** (B at p−k shifted to p): +0.00002 on the sample. The DP already gets them through rep0.
  - **Nearest per length class** (lc6.12) is +0.00006 over A/B, below top2.
- **The 3-byte matches are essential.** Removing them (SA min length 4, or h4 only) costs −1.3 %.
- **DXT1 caveat.** Exact candidates are *worse* on DXT1 under 2p: 1.3821 against 1.3836 for a06. They are better under 4p: 1.38448 against 1.38394. The gain comes from DXT5 (+0.2 %), NIF and uncompressed DDS. Retraining the prior does not fix DXT1. It looks like price staleness, because more long far B's trigger the `sufficient_len` immediate commit.

## Q3: cheaper chains (64 KiB sample)

| recipe | 2p | 1p + split | walk steps/pos |
|---|---:|---:|---:|
| today | 1.37118 | 1.36979 (−0.143 %) | 3.88 |
| a06 (h10 s4) | 1.37302 | 1.37148 (−0.020 %) | 2.35 |
| a06 + h6 s4 | 1.37369 | 1.37219 | 2.49 |
| h6 s4 + h12 s4 (no h10) | 1.37335 | 1.37166 | 2.46 |
| **S3** (h6, h10, h12, all s4 d16) | **1.37423** | **1.37269 (+0.069 %)** | 2.53 |
| S3 + h24 s4 | 1.37425 | | 2.56 |
| S3 with the sparse chains at d8 | | 1.37255 | 2.42 |
| S3 with h4 at d4 | | 1.37252 | 2.19 |
| **S3-lite** (h4 d4, sparse chains d8, 14-bit sparse tables) | | **1.37238 (+0.046 %)** | 2.32 |
| S3, sparse tables at 14 or 12 bits | | 1.37269 (no change) | 2.77 / 3.76 (more collisions) |
| exact SA (upper bound) | 1.37452 | 1.37316 (+0.103 %) | – |
| a06 h10 s2 d32 | 1.37345 | | 2.55 |
| h10 at all positions d64 | 1.37375 | | 3.22 |
| exact chains h3 + h4 d4096 + h10 d4096 | 1.37422 | | 84 |

- The widths are complementary: 6 bytes fills the gap between h4 d8 and h10, and 12 bytes covers DXT5's alpha + colour-endpoint run. They do not substitute for each other: 8- or 16-byte keys alone *hurt* (1.37141).
- Stride 4 works because DDS offsets are aligned. The mechanism is generic, but the gain is data-shaped. It is neutral on DXT1 and NIF and loses nothing there.
- The S3 sparse chains tolerate 12–14-bit head tables at no ratio cost. That matters for K1, because the live table size is what limits K1's concurrency (a06 §3).
- **Not worth it:**
  - **Continuation propagation** (0.00).
  - **"Insert everywhere, search at stride"**: same as the a06 recipe.
  - **LSH.** It was not built. LZ needs exact prefix lengths, and multi-width exact hashing is the LSH-like idea that works.

## The headline: 1 DP pass + split + S3

| | 1p + split | 2p + split | L16 | L14 |
|---|---:|---:|---:|---:|
| 64 KiB sample | 1.37269 (+0.069 %) | 1.37693 | 1.37175 | 1.36911 |
| 64 KiB held-out (offset 25) | 1.37281 (+0.074 %) | 1.37698 | 1.37179 | |
| **64 KiB full corpus** | **1.37204 (+0.076 %)** | 1.37625 (+0.383 %) | 1.37100 | |
| 64 KiB full corpus, a06 recipe | 1.37085 (−0.011 %) | | | |
| 32 KiB sample | 1.35153 (+0.047 %) | 1.35479 | 1.35089 | 1.34862 |
| 16 KiB sample | 1.32438 (−0.057 %); retrained prior 1.32490 (−0.017 %); SA 1.32456 | **1.32700 (+0.141 %)** | 1.32513 | 1.32345 |

- For comparison, opt16 today on the full corpus is 1.37144. **1p + split + S3 beats today's 4-pass opt16.**
- Cheaper single-pass variants, on the 64 KiB sample:
  - final pass at optLevel 0 control flow (a cheap pass): 1.37232 (+0.042 %);
  - `sufficient_len` 16: 1.37256;
  - S3-lite with optLevel 0: 1.37199 (+0.017 %), which is too thin.
- **At 16 KiB the candidates are saturated** (the SA gives the same result), so the shortfall comes from prices. The preset should therefore be **1 pass at 32 and 64 KiB, and 2 passes at 16 KiB**. The other option at 16 KiB is to test a04's in-pass refresh or gap3, worth +0.016 % there, which would close the remaining −0.017 % only with the retrained prior.
- **Not stacked yet:** gap3, in-pass refresh and retrained priors. A retrained prior is worth +0.018 % on 64 KiB 1p.

## Q1: the suffix array on the GPU, with a concrete design

**Design.** Prefix doubling, capped at depth 64, which is all K2 needs (`search_cap`):
- an initial 32-bit key sort (4 × 8-bit LSD digit passes);
- 4 doubling rounds (h = 4 → 64). Each round is a stable sort by rank[i+h] within groups, a 16-bit key, so 2–4 digit passes plus a rank scan;
- LCP by direct compare, capped at 64;
- a nearest-previous-per-threshold query. Exactly, this is a range-max over rank intervals; the sparse table for that is 4.3 MB/block, which rules it out. So the practical version is a ±W window walk with running-min LCP, like `k2_window`.

**Cost.**
- About 20 digit passes over 64K elements. a06's calibration is about 1.2 µs per 8-bit digit pass, the same as lvl9s12's 12-bit counting sort.
- That is **≈ 24 µs** plus scans, gathers and the window (≈ 3 µs), against K1+K2 = 9.5 µs today and ≈ 10.8 µs for S3. Even an ideal device-wide onesweep (5 full 32-bit sorts) is about 16 µs.
- Memory: SA, rank, LCP and temp come to 1 MB live per block, against 0.5 MB today.
- Parallel width is good: 64K per block. But it needs about 25 dependent passes, against 1 for the chains.
- **a06's "3–5× K1" stands.** Against S3 the SA buys +0.03 % (1p + split: 1.37316 against 1.37269).

**Sparse SA.** This sorts only stride-4 positions, where doubling with h ∈ {4, 8, 16, 32} stays inside the sparse set.
- It is 20 passes over 16K elements, about 5 equivalent passes, ≈ 6 µs.
- It needs the window step on top, and it only sees offsets ≡ 0 (mod 4), which is the same coverage as S3's chains.
- The chains cost ≈ 1.5 µs, so the sparse SA loses as well.

**Passes needed with exact candidates.**
- 2p clears L16 easily without split (+0.20 %).
- 1p without split does not (1.37054–1.37075, −0.07…−0.09 %).
- **1p + split does clear it** (+0.10 % with the SA, +0.069 % with S3).
- Exact candidates do **not** make a 1-pass parse work at 16 KiB.

## Q4: sorting across the whole batch

**Mechanism.** One global radix sort with key (block, hash) and value pos.

**Cost.**
- The block id adds 12 or more key bits. That means 28-bit keys, so 4 digit passes instead of 2 for a segmented per-block sort of the 16-bit hash, with no output benefit: per-block order is already implied by the layout.
- Traffic is about 64 B per element per chain, 4 MB per block. That is ≈ 3 µs/chain on the 5090, against 3.4 µs/chain for today's K1, but ≈ 10–15 µs/chain on 270–450 GB/s cards.
- The one real advantage is no head tables, so no dependence on L2 size. A *segmented* per-block sort has that advantage too, with half the passes (a06 §5).

**Verdict.** **Dead end.** If K1 moves to sorting, use a per-block 2 × 8-bit LSD sort. For S3's stride-4 chains that is 16K keys, so ≈ 0.6 µs per chain. The no-cross-block constraint is automatically met in either form.

## GPU cost against opt16 (5090, calibrated to a06's and the synthesis's measured kernel times)

| stage | opt16 today | 1p + split + S3 (today's kernels) | with A1–A4 K3 wins |
|---|---:|---:|---:|
| K1 | 6.9 | ≈ 8.3 (a06: +0.48 µs per stride-4 chain, × 3) | 8.3, or ≈ 7.5 with 14-bit sparse tables and fused sparse chains |
| K2opt | 2.6 | ≈ 2.5 (2.53 steps/pos, 5 heads) | 2.5 |
| K3 | 33.3 (4 passes) | ≈ 9.6 (one final pass + fix-up) | ≈ 6–7 |
| K4 + K5 + split | 1.9 | 3–6 | 3–6 |
| **total (µs/block)** | **44.7** | **≈ 23–26 (1.7–1.9×, ≈ 2.5–2.8 GB/s)** | **≈ 20–23 (2.0–2.2×)** |

- Dependent-chain length: K3 drops from 4 serial DP passes to 1. K1 and K2 keep their per-tile and per-position chains.
- Memory: the sparse chains add 3 × 64 KiB per block in the compact p/4 layout, or 3 × 256 KiB as prototyped, plus 64 KiB per sparse head table.
- Small GPUs: K3 is an even larger share on 8 GB cards, so going from 4 passes to 1 is worth about 2–2.5× there (a 4060 at roughly 0.3–0.45 GB/s, from a08's 0.15–0.2).
- **After this change, K1+K2 is about 45 % of the time.** That is where a sorted K1 (a06 §5) or a K1 occupancy fix pays next.

## Byte-exact oracle

- `find_cands` grows to N chains: (key width, stride, depth). Records, fingerprint skips and the stop rule are unchanged.
- The stride-4 chains are built over slots p/4.
- a06 already prototyped one strided chain byte-exactly in `k1_chains_sg` and `k2_opt` (h10 s4). S3 adds two more chains of the same kind, and K2's merged walk takes 5 heads.
- The DP is unchanged, with `passes = 0` (1 at 16 KiB: preset by block size).
- The split writer comes from a10 (B1, already in progress).
- Retrain `OPT_PRIOR_*` on the new candidates.
- Effort: **M** (an incremental change on top of a06's h10 work). Risk: low to medium.

**Risks.**
- The gain is concentrated in DXT5 and depends on alignment.
- Margin is +0.076 % on the full corpus at 64 KiB and +0.047 % at 32 KiB (sample). The 32 KiB full corpus has not been run.
- 16 KiB needs the 2-pass fallback.

## Dead ends checked

- **Full or sparse GPU suffix array:** 2–3× K1+K2 for +0.03 %.
- **Batch-wide (block, hash) radix sort:** more passes than a segmented sort.
- **Larger candidate sets:** a full frontier is +0.014 %.
- **Exact A:** 0.
- **Continuation candidates:** 0.
- **Rep-aligned candidates:** already covered by the DP's rep probes.
- **8- or 16-byte sparse keys:** they hurt.
- **More depth on a06's chains:** s2 d32 or all-position h10 d64 are both worse than S3 at higher cost.
- **Exact candidates for a 1-pass parse at 16 KiB:** the shortfall is prices, not candidates.
- **1p with the final pass at optLevel 0 and S3-lite:** the margin is too thin (+0.017 %).

## Top 3

1. **Preset "opt1s": S3 candidates + B1 split + a single DP pass at 32 and 64 KiB (2 passes at 16 KiB).**
   - Ratio: +0.076 % over L16 on the full 64 KiB corpus, above today's opt16.
   - Speed: ≈ 1.7–2.2× opt16 on the 5090 and more on small GPUs.
   - Effort M.
2. **Keep the A/B 8-byte format, but consider top2** (the two longest instead of nearest + longest): +0.006–0.010 %, free. Do not widen K2's output.
3. **Next, attack K1+K2.**
   - Use 14-bit head tables for the sparse chains (ratio-neutral) and build the three sparse chains in one task.
   - Or move to a06's segmented 8 + 8-bit sorted K1 (0.6 µs per sparse chain).
   - Not a suffix array, and not a batch-wide sort.
