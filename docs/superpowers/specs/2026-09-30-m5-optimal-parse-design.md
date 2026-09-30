# M5: libzstd L14 / L16 ratio on the GPU (optimal parse) — design

Date: 2026-09-30 · Status: design approved in conversation ("looks good … continue with the next algorithm")
Builds on: master `b356ee4` (speed2: 64 KiB default, `lvl9seg` segmented parse, sorted finder, transfer queue).
Detailed design and evidence: `docs/superpowers/m5/m5-opt-design.md` (normative detail for kernels and tie
rules), `docs/superpowers/m5/m5-ratio-drivers.md` (independent ratio measurements), `m5-rt-npu.md`
(RT cores / NPUs / tensor cores: no-go).

## 1. Goal

Two new quality presets that reach the compression ratio of libzstd **level 14** and **level 16** on our
blocks, bit-exact with a CPU oracle and decodable by standard zstd:

- `opt14` ≥ libzstd L14 and `opt16` ≥ libzstd L16, measured on the full corpus at **64 KiB**, and also
  validated at 32 KiB and 16 KiB (block size is ≤ 64 KiB; blocks are independent).
- They are **quality presets**, not replacements for `lvl9s12seg`. Projected throughput on the 5090 is
  ≈ 3.6–5.0 GB/s (`opt14`) and ≈ 2.7–3.6 GB/s (`opt16`). On an 8 GB-class card it is ≈ 0.4–0.8 GB/s, below
  10 Gbit line rate, which suits slower links, 7z-extraction-bound installs and repacks.
- Existing presets stay byte-identical (the lvl3 anchor and all differential tests).

## 2. What L14 / L16 are at 64 KiB (established by research)

- libzstd for sources ≤ 128 KiB: L14 = btopt (minMatch 3, searchLog 4, targetLength 32); L16 = btultra
  (minMatch 3, searchLog 6, targetLength 128).
- At 64 KiB the gain over L9 is **parse only**: windowLog clamps to 16 and the block splitter is off.
- The ratio comes from **3-byte matches together with a priced optimal parse**. Neither alone gets half of it.
  Without 3-byte matches the result is −1.3 to −1.6 %; with no rep candidates it is −0.45 %; with K = 1
  candidates it is −0.5 %.
- These can be dropped for ≤ 0.02 % each (all measured): the binary tree, more than 2 candidates per
  position, targetLength above 32, whole-block DP (4 KiB segments cost ≈ 0.015–0.06 %), and zstd's in-block
  adaptive statistics. Static prices iterated to a fixed point beat the adaptive ones by +0.17 %.

## 3. Algorithm (normative; the GPU mirrors the CPU oracle byte for byte)

### 3.1 Parameters

- `MatchParams` gains `opt: Option<OptParams>` and a hash mode `Opt3`: a 4-byte hash chain of `depth` plus a
  3-byte hash chain of depth 4.
- `min_match = 3` is allowed only with `opt`. `segment_log2 = 12` is required, and `lazy = 0`.
- `OptParams { level: 0|2 (final pass), target_length: 32, passes: u8 (cheap intermediate passes),
  seed: BlockInit | Prior, k: 2 }`.
- Presets:
  - `OPT16` = `Opt3`, depth 32, block-init seed, 3 cheap passes then an optLevel-2 final pass.
  - `OPT14` = the same candidates, seeded from a prior plus cover literals, 1 cheap pass then the final pass.
- Existing presets have `opt: None` and are unchanged.

### 3.2 Candidates (K1 + K2opt)

- K1 builds two chains per block: `h4` (today's 16-bit hash of 4 bytes) and `h3` (a 16-bit hash of 3 bytes),
  reusing the `N_HASHES = 2` layout.
- K2opt, one thread per position, walks the `h3` chain (depth 4) and the `h4` chain (depth 32) nearest-first,
  merged by position. It uses the fingerprint filter and the 64-byte capped compare. It records a candidate
  whenever the capped length strictly beats the best so far, starting at length 3, and keeps
  `A` = the first record (the nearest ≥ 3) and `B` = the last record (the longest; ties go to the nearer).
- **Fingerprint caveat.** Entries on the `h4` chain whose first 4 bytes differ (16-bit hash collisions) can
  still share 3 bytes with `p`, and such an entry is a valid 3-byte record. A 4-byte fingerprint mismatch may
  therefore skip the compare only once `best >= 3`, or when the first 3 bytes also differ.
- Output is two u32 words per position (8 B): `offA:16|lenA:8|lenB:8`, `offB:16|0`.
- The CPU oracle is `reference::find_cands`, with identical order and tie rules.
- The sorted finder (speed2 E2) is optional follow-up work (S5). S1 uses the chain K1.

### 3.3 Parse (K3opt)

- The structure is exactly `lvl9seg`'s: 16 independent 4 KiB segments per block, one lane per segment,
  workgroups of 32 lanes (2 blocks), then the same per-block fix-up (`main_fixup`). The fix-up handles
  literal carry across segments and re-encodes `off_base` against the true decoder reps.
- Per segment: a port of `ZSTD_compressBlock_opt_generic`, integer-only, statement by statement. That covers
  series, `sufficient_len = target_length`, forward relaxation, the backward trace, and exact per-node rep
  history (3 × u16 per node, `ZSTD_newRep`).
- It uses a ring of `target_length + 1 = 33` nodes per lane, following the oracle's `opt::Engine::Ring`
  exactly: match-node reps are computed at relaxation, the `last_pos + 1` sentinel is virtual, and the backward
  trace reads a per-position trace, never the ring. The ring lives in workgroup memory (≈ 23 KiB per
  32-lane workgroup), which needs the adapter limit (≥ 32 KiB; 48 KiB on NVIDIA and 64 KiB on AMD). The
  fallback is `var<private>` when the limit is below that. The trace is 8 B per position in the dead `pred`
  buffer.
- Every tie rule is fixed and tested, following `m5-opt-design.md` §3.4 item 4: `<=` in the literal
  extension, `<` in relaxation, lengths in descending order, record order (reps, then A, then B), and
  `newRep` numbering.

### 3.4 Prices

- Prices are static per block per pass, in 1/256-bit units: `WEIGHT(sum) − WEIGHT(count)` from the previous
  pass's own literal histogram and LL/ML/OF histograms, with `off_base` taken under the decoder reps.
- Pass 0 is seeded either from zstd's block initialisation (`BlockInit`) or from a prior (`Prior`): constant
  LL/ML/OF tables in `codes.rs`, **trained on blocks disjoint from the 1/50 evaluation sample**, plus a
  "cover-literal" histogram of the bytes no candidate covers.
- Every pass prices with the same fractional `ZSTD_fracWeight` arithmetic (`opt::frac_weight`); `ZSTD_bitWeight`
  is used nowhere. Intermediate ("cheap") passes differ from the final pass only in their control flow,
  optLevel 0: the relaxation's early abort, the `+128` skip, and no match + 1 literal check. The final pass
  runs optLevel 2. (The measured recipe, 1.37211, was built this way.)
- On the GPU, pass n histograms its own output in the workgroup epilogue (about 1.5 KiB per block), and pass
  n+1's prologue turns that into u16 price tables. This needs no extra dispatch.

### 3.5 Downstream

- `min_seq_len` becomes 3 for `opt`, so `MAX_SEQS = BLOCK_SIZE/3 + 1`.
- K4's ML code 0 is already supported. K4/K5 handle ~55 % more sequences.
- VRAM rises to ≈ 1.35 MiB per block, so `--batch max` shrinks (≈ 2900 blocks at 6 GiB, i3), and all of it
  is counted in `vram_bytes`.
- Transfer queue, direct upload and the other speed2 paths are unchanged.

## 4. Testing and gates

- **CPU oracle:**
  - A `lazy::cases`-style hand-built suite for every DP tie rule, segment boundaries and rep numbering.
  - Round trips through libzstd for all synthetic cases and 4000 corpus blocks, at 64/32/16 KiB.
- **GPU:**
  - Byte-identical candidate words (K2opt against `find_cands`).
  - Byte-identical frames for `opt14`/`opt16`, and for oracle variants (`passes 0, 1, 3`, both seeds), on
    all synthetic blocks and 4000 corpus blocks at 64K and 16K.
  - Subgroups on and off, all transfer modes.
- **Ratio gates** (full corpus via `gzc-bench ref`, which is byte-identical to the GPU):
  - `opt16` ≥ libzstd L16 and `opt14` ≥ libzstd L14, at 64 KiB, 32 KiB and 16 KiB.
  - libzstd L14/L16 are measured with `gzc-bench cpu --levels 14,16` on the same blocks. The corpus
    estimates are ≈ 1.3684 / 1.3710 at 64 KiB.
- **Performance** is recorded with the usual protocol (median of 3, `--verify`) and compared with the §3.3
  projections of `m5-opt-design.md`. There is no hard throughput gate; the stages (below) have time checks.

## 5. Stages

- **S0** CPU oracle, presets and prior tables; full-corpus ratio gates.
- **S1** K1 `h3` key + K2opt, byte-identical candidates.
- **S2** Naive K3opt with one pass (correctness gate); measure the per-pass cost and fix occupancy if it is
  above ~6 µs/block.
- **S3** Histogram epilogue + price prologue, N passes, prior seed; byte identity for every variant.
- **S4** Integration (`MAX_SEQS`, batch sizing, fix-up), corpus `--verify`, results.
- **S5** Optional: sorted-finder / LCP-neighbour candidate source (`m5-rt-npu.md` §4.4), and the `h8` chain.
- **S6** Speed dials: pass counts, per-kind priors from the DDS header (R9), targetLength 16.

## 6. Out of scope

- libzstd byte-compatibility of the parse (the tree is not reproducible, and it isn't needed).
- Levels above 16.
- Cross-block history.
- Replacing the fast presets.
