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

## T3: K3opt single pass (stage S2), branch `worktree-agent-adc20584ad0360e20` (from `m5` d4d05e2), 2026-09-30

`gzc-gpu`:

- `shaders/k3_opt.wgsl`: one DP pass of `opt::dp_pass_with(.., Engine::Ring)`.
  - One lane per 4 KiB segment; each trip of the flattened lane loop is a series-start probe or a series position.
  - A 33-slot ring of 16 B nodes (price; rep0|rep1; rep2|litlen; mlen|offBase<<8) in workgroup memory, or in
    `var<private>` as the fallback.
  - The per-position trace goes in its own buffer (K1's `pred` in the pipeline, 8 B per position). The write at
    `iend` is dropped.
  - The end of a series only updates `ip`/`anchor`/reps and logs the series in `seqs`. The backward traces run
    after the DP (phase 2), last series first, so each segment's raw sequences come out reversed.
  - Prices come from the block-init rule (a literal histogram in a prologue; LL/ML/OF tables from the host) or
    from explicit per-block tables (the `opt::cases` hook).
- `shaders/k3_fixup.wgsl`: `main_fixup` and the rep helpers, moved out of `k3_seg.wgsl` unchanged except for
  `SEG_WORDS` (words per segment in `best`) and `RAW_REVERSED`. lvl9seg is byte-identical (differential suite,
  with and without subgroups).
- `k3opt.rs`: `K3Opt` (config: level 0/2, `wg`, ring memory, price source), `OptBuffers`, `parses_from_cands`
  (the `parses_from_best` counterpart, fed `find_cands` words from the host) and `time_pass`.
- `context.rs`: requests the adapter's `max_compute_workgroup_storage_size` (48 KiB on the 5090).
- Stored offBases are the DP's own. They are `off_base_for` under the DP's history, as `main_fixup` needs; the
  argument is in `emit_series`. A test on every synthetic and corpus block confirms it.

### Correctness (all byte-identical to the oracle, parses compared)

| gate | 64 KiB | 16 KiB |
|---|---|---|
| every `opt::cases` run (27: 20 at level 2, 7 at level 0; explicit tables; the preset runs with the oracle's final-pass prices), workgroup + private ring | pass | pass |
| synthetic blocks (15) × {wg16 L2, wg16 L0, private L2, private L0, buffer prices, wg32, wg8, wg8 private L0, checked loops} | pass | pass |
| 4000 corpus blocks × {wg16 L2 block-init (+ K4/K5 frames == `write_frame`), wg16 L0, private ring L2, wg32 buffer prices} | pass | pass |

Commands:

- `cargo test --release -p gzc-gpu --test k3opt`;
- `GZC_CORPUS=… cargo test --release -p gzc-gpu --test k3opt k3opt_corpus -- --ignored`;
- the same with `--no-default-features --features block-16k`.

The full `gzc-gpu` suite passes at 64K, and `differential` also passes with `GZC_NO_SUBGROUPS=1` and at 16K.

### K3opt time per pass (RTX 5090, 64 KiB, 2900 corpus blocks ≈ the design's 6 GiB batch)

Protocol:

- `k3opt_timing` (ignored test), timestamp queries, median of 5 dispatches per run, three runs.
- Candidate words re-uploaded before each dispatch.
- The GPU was otherwise idle (gated on no other test or bench process; load average 1.9).
- The fix-up adds 0.15 µs/block (0.24 at level 0).

| variant | runs (µs/block) | median | ms/batch |
|---|---|---:|---:|
| **wg16, workgroup ring, level 2, block-init prologue** | 18.18 / 18.16 / 18.04 | **18.16** | 52.7 |
| wg16, workgroup ring, level 2, table prices | 18.06 / 18.00 / 18.07 | 18.06 | 52.4 |
| wg16, workgroup ring, level 0 (cheap pass) | 14.88 / 14.74 / 14.86 | 14.86 | 43.1 |
| wg32, workgroup ring, level 2 | 30.21 / 30.17 / 30.16 | 30.17 | 87.5 |
| wg32, workgroup ring, level 0 | 24.31 / 24.32 / 24.42 | 24.32 | 70.5 |
| wg64, workgroup ring, level 2 | 30.43 / 30.44 / 30.43 | 30.43 | 88.2 |
| wg8, workgroup ring, level 2 | 21.32 / 21.34 / 21.31 | 21.32 | 61.8 |
| wg8, private ring | 25.84 / 25.76 / 25.86 | 25.84 | 74.9 |
| wg16, private ring | 21.23 / 21.22 / 21.24 | 21.23 | 61.6 |
| wg32, private ring | 19.85 / 20.06 / 20.11 | 20.06 | 58.2 |
| wg64, private ring | 19.60 / 19.53 / 19.86 | 19.60 | 56.8 |
| wg16, workgroup ring, naga loop bounding on | 19.24 / 19.21 / 19.22 | 19.22 | 55.7 |

- 16 KiB (one run, 11600 blocks, the same bytes): wg16 L2 7.61 µs/block, L0 6.07, wg32 11.03, wg32 private 9.93.
  That is 30 µs per 64 KiB of data: 4 blocks per workgroup of 16 carry 4 sets of price tables.
- **Target ≤ 6 µs/block: missed by 3×.** The design projected 3.0–4.0 µs.

### Where the time goes

Batch scaling (wg16, workgroup ring, level 2):

| blocks | 16 | 680 | 1360 | 2900 | 4000 |
|---|---:|---:|---:|---:|---:|
| ms | 18.0 | 24.7 | 25.9 | 52.4 | 81.7 |

- The kernel is bound by per-lane latency. One lone warp per SM needs 18 ms for its 4096 trips (~4.4 µs per trip).
  Up to one resident wave (~1400 blocks) costs little more; each further wave adds its full latency.
- Residency is capped by the ring: 33 × 16 B = 528 B per lane, plus ~1.7 KiB of price tables per block.
  - wg16: 8 workgroups per SM, about 1360 blocks per wave, so 2900 blocks take 2 waves.
  - 16 B is the smallest node: price 30 bits, 3 reps × 16, litlen 13, mlen 6, offBase 17 is about 114 bits.
- Per-lane latency grows with the active lanes per warp, from intra-warp divergence in the variable-length loops
  (relaxation, series seeding, candidate extension):
  - 16 blocks: wg8 14.9 ms, wg16 18.0 ms, wg32 22.9 ms;
  - wg16 beats wg32 at the same lanes in flight.
- Changes kept, in order (wg32 unless noted; the early figures were partly contended):
  1. Deferring the backward traces out of the DP loop (series log + phase 2).
  2. Storing the DP's offBases as they are: no per-sequence canonicalisation, one reversed write pass.
     - 33.9 → 29.9 µs; phase 2 fell from ~15 to ~4 ms per 1360-block batch.
  3. Removing `continue` from the trip loop, so every lane reaches the end of each trip: 29.9 → 21.4 µs.
     This wg32 number did not survive later changes; it is 30.2 in the final table.
  4. Issuing each trip's global loads first and the three rep-source loads together: no measurable change,
     kept because it is simpler.
- Ablations (test-only knob, since removed; 16 blocks, wg32; outputs wrong, time only):
  - full: 22.9 ms;
  - no relaxation: 17.8;
  - no phase 2: 21.4;
  - no trace writes: 22.3;
  - no relaxation + no phase 2 + no trace: 15.4;
  - reps only (no candidate words): 7.0;
  - no rep probes: 68.6 (long matches then stay inside series);
  - `best` bound read-only: −2 %.
- Rejected:
  - wg32/wg64 (divergence);
  - the private ring (latency; better occupancy only above ~3000 blocks);
  - wg8 (occupancy: 8-lane warps);
  - naga loop bounding (+6 %).

### Next steps (T4/T5 or a follow-up)

- Cut per-lane latency:
  - a subgroup-cooperative relaxation (lanes of a warp share one lane's up-to-30 relaxations), with the sequential
    path as the non-subgroup fallback;
  - fewer dependent shared-memory round trips per trip (the literal extension and relaxation read the ring
    serially).
- Raise residency:
  - u16 price tables (−0.8 KiB per block);
  - at 16 KiB, a smaller workgroup (4 blocks' tables per wg16 today; wg8 would halve that).
- A profiler (Nsight Compute is not installed on this machine) would settle how the trip time splits.
- Concern: during one of the experiments a kernel was killed with Xid 109 (CTX SWITCH TIMEOUT). The cause was an
  experiment build that skipped the trailer, so `main_fixup` ran on garbage counts; the shipped kernel always writes
  the trailer. No hang otherwise.
