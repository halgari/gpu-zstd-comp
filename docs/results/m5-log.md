# M5 log: optimal-parse presets `opt14` / `opt16`

Plan: `docs/superpowers/plans/2026-09-30-m5-optimal-parse.md`. Design: `docs/superpowers/m5/m5-opt-design.md`.
Corpus: `data/corpus --ext dds,nif` (3172 files, 6.49 GB). Ratios are real bytes / frame bytes and are
deterministic (CPU oracle), so one run per row. CPU runs used 16 threads, with another agent's GPU benchmarks
running at the same time; the MB/s columns are informational only.

## T1: CPU oracle (stage S0), branch `m5`, 2026-09-30

`gzc-core`:

- `opt.rs`: the integer port of `ZSTD_compressBlock_opt_generic`, the price tables, seeds and the pass driver.
- `reference::find_cands`: K2opt.
- `hash::hash3`.
- `params`: `Hashes::Opt3`, `OptParams`, `Seed`, `OPT14`/`OPT16`.
- `codes::OPT_PRIOR_*`.

### Reproducing the design sample (every 50th 64 KiB block, offset 0: 2016 blocks)

`cargo run --release -p gzc-core --example opt_sample -- eval data/corpus 50 0 zstd`

| | design (prototype) | oracle | delta |
|---|---:|---:|---:|
| opt16 | 1.37211 (Q1) | **1.37211** | 0 |
| opt14, prototype's in-sample prior | 1.37127 (Q6) | 1.37127 | 0 |
| **opt14, block-disjoint trained prior (shipped)** | | **1.37118** | −0.00009 (gate ± 0.0005) |
| libzstd L14 / L16 | 1.36911 / 1.37175 | 1.36911 / 1.37175 | |

- The prototype harness was re-run first: `m5d 50 100000 Q1` gave 1.37211 and `Q6` gave 1.37127.
- Two prototype details were dropped without changing the result at 5 digits:
  - zstd's `nextToUpdate` skip, which the prototype still emulated;
  - the cover-literal histogram now uses the stored (64-capped) `lenB`, as the GPU will.
- Prior tables: summed LL/ML/OF histograms of `opt16`'s output over every 50th block at **offset 25** (2015
  blocks, block-disjoint from the 1/50 evaluation sample), each table scaled to 65536. Produced by `opt_sample -- train
  data/corpus 50 25`.
- The prototype's prior was in-sample and came from a different candidate recipe (h4+h8, hash3 slot). The
  block-disjoint prior costs 0.00009 on the sample.

### Full corpus: `gzc-bench ref --preset opt14,opt16 --verify` vs `gzc-bench cpu --levels 14,16`

Each block size used its own build: `CARGO_TARGET_DIR=target/b16|b32`, `--no-default-features --features
block-16k|block-32k`. Every frame decoded with libzstd (`--verify`).

| block | blocks | libzstd L14 | libzstd L16 | opt14 | opt14 vs L14 | opt16 | opt16 vs L16 | opt14 vs L16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 KiB | 100754 | 1.36827 | 1.37100 | **1.37064** | **+0.173 %** | **1.37144** | **+0.032 %** | −0.027 % |
| 32 KiB | 199740 | 1.34797 | 1.35025 | **1.35105** | **+0.228 %** | **1.35158** | **+0.098 %** | +0.059 % |
| 16 KiB | 397925 | 1.32606 | 1.32774 | **1.32765** | **+0.120 %** | **1.32794** | **+0.015 %** | −0.007 % |

- All six gates pass: `opt14` ≥ L14 and `opt16` ≥ L16 at every block size.
- The design's corpus estimates were L14 ≈ 1.3684 and L16 ≈ 1.3710. Measured: 1.36827 / 1.37100.
- The 16 KiB margin for `opt16` is thin (+0.015 %).

Oracle speed on 16 threads, shared machine (informational): opt14 211 / 227 / 269 MB/s and opt16 146 / 161 / 181
MB/s at 64 / 32 / 16 KiB. libzstd L14/L16 ran at 423 / 381 MB/s at 64 KiB.

### Correctness

- Hand-built DP cases (`opt::cases`, 15 cases, 22 runs): literal `<=`, relaxation `<`, `startML`, the rep `ll0`
  numbering in-series, the optLevel-0 abort and `+128` skip, match + 1 literal, `sufficient_len`, the rep early
  return, `ilimit`, segment clamps, capped extension, literal carry, true-rep re-encode, and the presets end to end.
- Each rule was mutation-checked: 14 of 15 single-rule mutations fail at least one case.
- The 15th mutation is not observable with K = 2 records, and neither is its rep-only twin, so no case can catch
  them:
  - `>` vs `>=` for an explicit record equal in length to a rep record;
  - `>` vs `>=` for a rep record equal in length to an earlier rep.
- Debug builds check that the DP never reads an `opt[]` entry not written in the current series. This held on all
  synthetic blocks and on 4000 corpus blocks × 9 variants. So a GPU ring never needs to reset entries left over
  from an earlier series.
- Note for T3: a 33-node ring only works if a match node's reps are computed when it is relaxed, from the source
  node, which is final by then. zstd (and the `Linear` engine) computes them when the node is visited, from `opt[cur
  - mlen]`. By the visit of `cur`, relaxations from `cur - 1` have written positions up to `cur + 31`, whose ring
  slots are those of `cur - 32 ..= cur - 2`: the whole window `cur - mlen` reads. The fix round adds
  `opt::Engine::Ring`, which is that spec.
- libzstd round trips:
  - all synthetic blocks × 9 variants (opt14, opt16, passes 0/1/3 × both seeds, optLevel-0 final);
  - 4000 corpus blocks × the same 9 variants (`opt_corpus_roundtrip`, ignored test) at 64/32/16 KiB;
  - the full corpus via `--verify` above.
- Existing presets are byte-identical: all 129 gzc-core tests pass at 64, 32 and 16 KiB, including the lvl3 anchor.

### T1 fix round 1 (review), 2026-09-30

- **`opt::Engine::Ring`**: the GPU K3opt spec, documented on `Engine`.
  - A ring of `target_length + 1` (33) nodes.
  - Match-node reps are computed at relaxation or seeding, and the match + 1 literal node reuses `prevMatch.rep`.
  - The immediate encoding computes its reps itself, and the commit uses `lastStretch.rep`.
  - The sentinel is virtual: `price = MAX` past `last_pos`, with no `opt[last_pos + 1]` write.
  - The backward trace reads a per-position `trace` written when each node becomes final.
  - Debug builds check that every ring read finds its own position in its slot. A 32-slot ring trips the check at once.
- **Ring vs linear:** outputs are asserted identical on:
  - every `opt::cases` run;
  - every synthetic block × 9 variants (opt14, opt16, passes 0/1/3 × both seeds, optLevel-0 final);
  - 2000 corpus blocks (every 50th at 64 KiB) × 9 variants, at 64, 32 and 16 KiB;
  - the corpus runs at 16 and 32 KiB used debug assertions.
- **Trace bound:** a node can sit at `iend` itself, so `trace` holds `BLOCK_SIZE + 1` entries. That entry is never read, so the GPU may drop its write. The new case `last_segment_match_to_block_end` pins this.
- **New cases:**
  - `tail_literals_are_not_reparsed` pins `ip = anchor + litlen`. It is observable at level 0 through the `+128` skip, and restarting at the anchor fails it.
  - `rep_candidates_at_series_start` covers `ll0` reps from `opt[0]` after a series that ends in a match.
  - `opt::cases` now has 18 cases.
- **Docs:** the spec (§3.2, §3.3, §3.4) and m5-opt-design (§0, §2.1, §2.4) were amended.
  - Cheap passes use the same `frac_weight` prices; `ZSTD_bitWeight` is used nowhere.
  - A 4-byte fingerprint mismatch may skip the compare only once `best >= 3`, because h4-chain collisions can be valid 3-byte records.
- **gzc-gpu:** `ChainsKernel::with_options` rejects `opt` params until T2.
- **Tests:** 130 gzc-core tests pass at 64, 32 and 16 KiB, and the workspace builds with `--all-targets`. The ratios are unchanged: the `Linear` engine was not touched.

## T2: GPU candidates, K1 Opt3 chains + K2opt (stage S1), 2026-09-30

Branch: T2 worktree based on `m5` at d4d05e2. Machine: RTX 5090, subgroups on unless stated. `uptime` load about 1.7.

### What changed

- `common.wgsl`: new `hash3` (== `hash::hash3`) and `pred_fp3`.
- `pred_fp3` is the fingerprint of the h3 chain: bits 17..24 hold a 7-bit hash of bytes 0..3, and bits 24..32 hold byte 3.
- K1 (both kernels) builds the Opt3 chains: `hash_width(4)` in chain 0 and `hash3` in chain 1 (`OPT3` constant from `finder_wgsl`). The T1 guard is gone. The K1 self-test checks the h3 fingerprints too (`chains::chain_fp`).
- New `k2_opt.wgsl` (`main_opt`, appended to `k2_best.wgsl`) implements the merged nearest-first walk and writes 2 words per position. `OptCandKernel` in compressor.rs holds K1 + K2opt, and `cands_from_blocks` is the test harness.
- The `best` buffer is 8 B per position for `opt` params (`best_words`, `best_bytes_for`), counted by `max_batch_blocks`, `BatchBuffers` and `scratch_bytes`, so it reaches `vram_bytes`.
- `gpu_supports` still rejects `opt` presets.
- **The fingerprint caveat never applies to our h4 hash.** `hash_width(.., 4) = (lo * 0x9E3779B1 * 0xC2B2AE3D) >> 16` is injective in byte 3 when bytes 0..3 are fixed: byte 3 only enters the top 8 bits, through a bijection. So two h4-chain entries whose first 4 bytes differ also differ in their first 3 bytes. K2opt therefore skips an h4 fingerprint-hash mismatch outright (c <= 2), not only once best >= 3.
  - This is still byte-identical; the spec's "or when the first 3 bytes differ as well" clause covers it.
  - `tests/cands.rs::h4_hash_is_injective_in_byte_3` checks the property exhaustively over byte 3.
  - A mutation of the skip bounds (h4 byte-4 bound 4 → 3) fails the differential test at once.

### Correctness

| check | 64 KiB | 16 KiB |
|---|---|---|
| `tests/cands.rs`: synthetic + small-alphabet blocks; opt16, opt14, depth 1/4/64; subgroup and fallback K1 | pass | pass |
| `tests/chains.rs`: Opt3 preds + raw pred words (fingerprints), both K1s | pass | pass |
| corpus, 4000 blocks (`GZC_CORPUS=… cargo test --release -p gzc-gpu --test cands corpus -- --ignored`), subgroups | 4000 equal | 4000 equal |
| same, `GZC_NO_SUBGROUPS=1` | 4000 equal | 4000 equal |
| whole gzc-gpu suite (existing presets byte-identical) | pass | pass |
| whole gzc-gpu suite with `GZC_NO_SUBGROUPS=1` | pass | cands + chains pass |

### Time: K1 + K2opt vs lvl9 K1 + K2, 64 KiB, full corpus (100754 blocks, batch 2048)

`cargo run --release -p gzc-gpu --example k2opt_bench -- data/corpus 2048`. For each batch, the per-kernel timestamps are the median of 3 reps; the table sums them over the corpus. Three full runs (µs per block):

| run | lvl9 K1 | lvl9 K2 | lvl9 sum | opt16 K1 | opt16 K2opt | opt16 sum | lvl3 K1 (2 chains) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.756 | 1.545 | 4.301 | 6.665 | 2.325 | 8.989 | – |
| 2 | 2.765 | 1.553 | 4.317 | 6.675 | 2.346 | 9.021 | – |
| 3 | 2.776 | 1.562 | 4.338 | 6.677 | 2.350 | 9.027 | 6.747 |
| **median** | 2.765 | 1.553 | **4.317** | 6.675 | 2.346 | **9.021** | |

- **K2opt costs 1.5× lvl9's K2.** It walks h4 32 deep plus h3 4 deep and writes 8 B per position instead of 4 B.
- **K1 costs 2.4× lvl9's K1.** Two chains cost the same in Dfast (lvl3 K1 6.75 µs), so this is the existing two-chain K1 cost, not something Opt3 adds.
- `GZC_K1_GROUPS=256` (the 5090's best setting): lvl9 1.95 + 1.56 = 3.51 µs, opt16 4.25 + 2.35 = 6.59 µs.
- `GZC_NO_SUBGROUPS=1` (fallback K1): lvl9 6.40 µs, opt16 9.76 + 2.36 = 12.12 µs.
- **Total: K1 + K2opt = 9.0 µs per block against 4.3 µs for lvl9 (+4.7 µs per block).**
- **Possible follow-up:** build both chains in one K1 task, sharing the data words and the tile loop. Today each chain is a separate task that re-reads the block.
