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
