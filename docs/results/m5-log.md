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
| **opt14, disjoint trained prior (shipped)** | | **1.37118** | −0.00009 (gate ± 0.0005) |
| libzstd L14 / L16 | 1.36911 / 1.37175 | 1.36911 / 1.37175 | |

- The prototype harness was re-run first: `m5d 50 100000 Q1` gave 1.37211 and `Q6` gave 1.37127.
- Two prototype details were dropped without changing the result at 5 digits:
  - zstd's `nextToUpdate` skip, which the prototype still emulated;
  - the cover-literal histogram now uses the stored (64-capped) `lenB`, as the GPU will.
- Prior tables: summed LL/ML/OF histograms of `opt16`'s output over every 50th block at **offset 25** (2015
  blocks, disjoint from the evaluation sample), each table scaled to 65536. Produced by `opt_sample -- train
  data/corpus 50 25`.
- The prototype's prior was in-sample and came from a different candidate recipe (h4+h8, hash3 slot). The
  disjoint prior costs 0.00009 on the sample.

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
  node, which is final by then. The oracle, like zstd, computes them when the node is visited, from `opt[cur -
  mlen]`. By then the ring slots of `cur - 1` and `cur - 2` may already hold `cur + 32` and `cur + 31`. Both
  forms give the same result.
- libzstd round trips:
  - all synthetic blocks × 9 variants (opt14, opt16, passes 0/1/3 × both seeds, optLevel-0 final);
  - 4000 corpus blocks × the same 9 variants (`opt_corpus_roundtrip`, ignored test) at 64/32/16 KiB;
  - the full corpus via `--verify` above.
- Existing presets are byte-identical: all 129 gzc-core tests pass at 64, 32 and 16 KiB, including the lvl3 anchor.
