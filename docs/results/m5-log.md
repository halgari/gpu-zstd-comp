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

`cargo run --release -p gzc-gpu --example k2opt_bench -- data/corpus 2048` (the example has since been removed). For each batch, the per-kernel timestamps are the median of 3 reps; the table sums them over the corpus. Three full runs (µs per block):

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

## T4: K3opt passes (stage S3), branch `worktree-agent-ae29833b5f305bfaa` (from `m5` af45a3d), 2026-09-30

`gzc-gpu`:

- `k3opt.rs`: `OptPasses` runs a preset's whole `opt::passes` schedule with one dispatch per pass.
  - Pass 0's kernel is priced by the seed: `PriceSrc::BlockInit`, or `PriceSrc::Prior` (the prior tables
    plus the cover literals, taken from the resident candidate words).
  - Each cheap pass (optLevel 0, `hist_out`) writes its histogram. The next pass's prologue reads it
    (`PriceSrc::Hist`).
  - Only the final pass writes its parse into `best` and runs the fix-up.
  - Host harnesses: `parses_from_passes` (it can read every cheap pass's histogram back), `read_hists` and
    `time_passes`.
- `shaders/k3_opt.wgsl`: the new prologue modes are `PRICE_MODE` 2 (prior) and 3 (from the histogram), plus the
  `HIST_OUT` epilogue.
  - The DP body moved into `fn dp` unchanged.
  - The `prices` binding is now read_write. It holds `Prices` (mode 1) or a `Hist` (mode 3 / `HIST_OUT`),
    377 words per block either way.
- **What is histogrammed:** exactly what the oracle's `Hist::of_output(&dp_pass(..))` counts, i.e. the
  fixed-up block parse (`lazy::encode_raw` output), not each segment's raw DP output. Concretely:
  - every literal byte of the block, including the block's last literals;
  - each sequence's LL code, including the literal carry on a segment's first sequence;
  - its ML code;
  - its OF code, taken from `off_base` under the block's true decoder reps.

  The GPU gets this in three steps:
  - Each lane counts its own raw sequences with its own `off_base`s, plus its literal bytes (literal ranges and
    the tail after its last match).
  - The block's segment-0 lane then walks the segments exactly as `main_fixup` does. It moves the first
    sequence's LL count by the carry, and moves the OF code of every sequence that the re-encode changes.
  - The block's lanes write the 377 words.
- **Deviations from the design text (§2.4, §2.2):**
  - A `HIST_OUT` pass does write its raw sequences. They go to the segment's series-log words of `seqs`,
    written last-first from the end of the region, which never overwrites an unread log entry (see `LOG_SEQS`).
    The OF fix needs them in order. The candidate words stay intact.
  - The design's separate 1.5 KiB shared-memory counters are not used; see the rejected variants below.
  - The price tables stay i32, as in T3 (u16 tables belong to the K3opt performance study).
- Prior-seed and `hist_out` kernels need a block's segments in one workgroup (`wg % 16 == 0` at 64 KiB). Both
  wg16 and wg32 qualify.
- T3 review minors:
  - later-pass Buffer-price configs: opt16's pass-1 tables at L2 and L0 on synthetic and corpus blocks;
  - the `context.rs` limit comment now names `SortKernel::new`;
  - a forced workgroup ring over the adapter limit is skipped with a message (`runs_here`);
  - the k3opt corpus sampler is now uniform over (file, block) and skips when the corpus is missing.

### Correctness (byte-identical to `opt::passes`: the final parse and every cheap pass's histogram)

Schedules: cheap passes {0, 1, 3} × seeds {BlockInit, Prior}, final pass at optLevel 2. BlockInit×3 is opt16 and
Prior×1 is opt14.

| gate | 64 KiB | 16 KiB |
|---|---|---|
| every `opt::cases` block (18, scripted candidates) × 6 schedules | pass | pass |
| synthetic blocks (15) × 6 schedules; opt16 / opt14 on the private ring and at wg32 | pass | pass |
| 4000 corpus blocks (every 25th / 99th (file, block)) × 6 schedules | pass | pass |
| later-pass Buffer tables at L2 and L0: synthetic (15) and 4000 corpus | pass | pass |
| the same k3opt tests with `GZC_NO_SUBGROUPS=1` | pass | pass |
| full `gzc-gpu` suite; `differential` with `GZC_NO_SUBGROUPS=1` and at 16K | pass | pass (differential) |

The optLevel-2 match + 1 literal path (`ll_inc1 < 0`) is exercised:

- final passes with `ll[1] < ll[0]`: 12 synthetic, 1779 corpus (64K), 2316 (16K), summed over the schedules;
- later-pass tables: 3 of 15 synthetic blocks, 787 of 4000 corpus blocks (837 at 16K).

The synthetic test asserts that the count is above zero.

Commands:

- `cargo test --release -p gzc-gpu --test k3opt`;
- `GZC_CORPUS=… cargo test --release -p gzc-gpu --test k3opt -- --include-ignored --skip timing`;
- the same with `GZC_NO_SUBGROUPS=1`, and with `--no-default-features --features block-16k`
  (`CARGO_TARGET_DIR=target/b16`).

### Time (RTX 5090, 2900 corpus blocks at 64 KiB / 11600 at 16 KiB, one batch, wg16, workgroup ring)

`k3opt_passes_timing` reports the timestamp medians of 5 reps. The tables give the median of 3 runs, all
GPU-idle-gated: runs where another agent's GPU job appeared were discarded and repeated. Load average was
2–20 (CPU only).

| 64 KiB, µs/block | pass 0 | pass 1 | pass 2 | pass 3 (final) | fix-up | **total** | span (with dispatch gaps) |
|---|---:|---:|---:|---:|---:|---:|---:|
| opt14 (Prior, 1 cheap) runs | 16.41 / 16.35 / 16.41 | 18.27 / 18.26 / 18.27 (final) | | | 0.27 | 34.94 / 34.88 / 34.95 | 34.94 |
| opt14 median | 16.41 | 18.27 | | | 0.27 | **34.94** | 34.94 |
| opt16 (BlockInit, 3 cheap) runs | 15.24 / 15.15 / 15.18 | 15.28 / 15.25 / 15.22 | 15.34 / 15.34 / 15.28 | 18.31 / 18.33 / 18.34 | 0.26 | 64.43 / 64.33 / 64.28 | 64.34 |
| opt16 median | 15.18 | 15.25 | 15.34 | 18.33 | 0.26 | **64.33** | 64.34 |

The same runs gave these T3 single-pass references:

- wg16 L2 block-init: 18.08 / 18.16 / 18.17;
- wg16 L0 buffer: 14.88 / 14.85 / 14.92.

Pass costs against those references:

- A cheap pass with its histogram costs 15.2 µs, against 14.85 without it: +0.35 µs (+2.4 %).
- The final pass priced from the histogram costs 18.33, against 18.16.
- The Prior seed's cover-literal prologue costs 16.41 − 15.2 ≈ +1.2 µs, where the design projected 0.3. It is a
  serial 4096-position scan per lane over the candidate words.

| 16 KiB, µs/block (3 runs) | passes (DP…) | fix-up | total |
|---|---|---:|---:|
| opt14 | 6.86, 7.78 | 0.04 | 14.67 / 14.67 / 14.69 |
| opt16 | 6.24, 6.49, 6.48, 7.81 | 0.04 | 27.05 / 27.08 / 27.06 |

Per 64 KiB of data, that is opt14 58.7 µs and opt16 108.2 µs, against 34.9 and 64.3 µs at 64 KiB.

### Decisions and rejected variants (64 KiB, clean runs)

- **Workgroup memory is the cliff.** The first epilogue kept LL/ML/OF counters and per-lane segment summaries in
  their own workgroup arrays, about 870 B more per wg16 workgroup. Cheap passes took 20.7–22.2 µs.
  - Ablations did not explain the gap: skipping the literal counts cut about 0.5 µs from pass 0; skipping all histogram work
    and the epilogue still left pass 0 at 20.1 µs (with the same block-init prices as T3's 14.9 µs pass).
  - The cause: T3's unchanged kernels with those arrays merely allocated ran at L2 24.26 / 24.29 (vs 18.1)
    and L0 20.14 / 20.17 (vs 14.9) µs, i.e. +33 %, in 2 clean runs.
  - **Kept:** the code counts share the prologue's `hist` words (literal count in the low 17 bits, since it is at
    most 65536; code count in the high 15 bits, since it is below 32768). The segment summary goes in the
    segment's dead trace words. There is zero extra workgroup memory: cheap passes 15.2 µs.
- **One dispatch per pass (kept) vs an in-kernel pass loop (rejected).** Tried with `LOOP_PASSES` and a runtime
  prologue mode.
  - Fused, opt16's 3 cheap passes took 45.85 / 46.02 / 46.01 µs in one dispatch. One dispatch per pass on the
    kept kernel took 45.7 µs (15.18 + 15.25 + 15.34). So the loop is 0.7 % slower.
  - The runtime prologue mode also slowed the unfused kernels by 3–4 % (cheap 15.8–15.95, final 18.7).
  - The span (first DP start to fix-up end) equals the sum of the passes within 0.03 µs/block, so dispatch gaps
    leave nothing for a loop to recover.
  - Under 3 %, so the simpler form stays; the experiment diff is not committed.
- Global atomics for the code counts were not tried. The packed form already costs only 0.35 µs.

### Next steps

- K3opt per-pass latency remains the whole cost (4 passes × 15–18 µs). The T3 performance study owns the
  relaxation loop. The prologue/epilogue code is modular and independent of the DP body:
  - `prologue(valid, b)`, with `count_block` and `cover_chunk`;
  - `hist_seq` / `hist_lits` / `hist_move`, called only from `emit_series` and the end of `dp`;
  - `hist_epilogue`.
- Any rewrite must keep the workgroup footprint in mind: +870 B per workgroup costs 33 % at wg16.
- The cover-literal prologue (+1.2 µs, opt14 only) could be cut:
  - unroll or prefetch the candidate-word loads;
  - or compute it in K2opt, which writes those words.

## T3b: K3opt speed port, branch `worktree-agent-ac3bc416894d39a90` (from `m5` d6a6700), 2026-09-30

This task ports the K3opt performance study (`docs/superpowers/m5/k3opt-perf.md`) into the T4 kernel. T4's
prologue and epilogue, packed histogram, both seeds, cover literals and all pass schedules are unchanged.
Output is byte-identical to the oracle for every schedule. `gzc-core` was not changed.

`gzc-gpu`:

- `shaders/k3_opt.wgsl`:
  - **Node payload in global scratch.** Only each DP node's price stays in the ring (`ring_p`, workgroup or
    private memory). Reps, litlen, mlen and offBase move to a new binding 6 (`scr`): 3 consecutive words per
    node, 33 nodes per segment lane.
  - **One loop for seeding and relaxation**, 4 lengths per step. A series start is handled as a relaxation from
    a virtual node at cur 0 with last_pos 0. The argument is in the kernel comment and in study §4.
  - **Branch-free helpers:** `ld32` / `match_len_nb` (an unaligned load without the alignment branch), and
    `ll_price` and `rep_after` as selects.
  - **u16 literal prices** in workgroup memory (study #2). The histogram is not aliased, because a `HIST_OUT`
    pass keeps `hist` live through the DP.
- `k3opt.rs`:
  - `OptBuffers::new(ctx, m, capacity)` now takes the opt params and allocates `scratch`.
  - `scratch_bytes_per_block(m)` is 6336 B at 64 KiB and 1584 B at 16 KiB. T5 must add it to `vram_bytes`.
  - `ring_for(m, cfg, limit)` replaces the inline choice. It returns an error when a forced workgroup ring
    does not fit, and also when the price tables alone exceed the limit (T4 review Minor 2). Both errors come
    before any pipeline is created.
  - `ring_bytes` now counts 4 B per node.
- `tests/k3opt.rs`:
  - `k3opt_ring_choice` covers the Workgroup/Private choice at small limits, the clean error, the scratch size,
    and building the Private fallback.
  - `k3opt_passes_timing` takes `GZC_K3OPT_SORT` (heavy-first order, measurement only).

### Correctness

All gates are byte-identical to the oracle:

- tests: `opt::cases`, synthetic, 4000-block corpus × the 6 schedules (histograms included), later-pass tables,
  and single-pass L2/L0 on workgroup and private rings, wg8/16/32;
- block sizes: 64 KiB and 16 KiB;
- subgroups: on, and off with `GZC_NO_SUBGROUPS=1`.

The full `gzc-gpu` suite passes. `differential` also passes at 16 KiB and with `GZC_NO_SUBGROUPS=1`. The
optLevel-2 match + 1 literal path is exercised on 1779 corpus final passes (64K) and 2316 (16K), the same counts
as T4.

### Time (RTX 5090, 2900 corpus blocks at 64 KiB, wg16, workgroup ring, µs/block, median of 3, GPU idle-gated)

Single-pass columns come from `k3opt_timing` (L2 block-init, L0 buffer). Registers and shared memory come from
`vkstats` (L2 buffer kernel). Waves are for 2900 blocks. The wave boundary was measured with the final kernel:
3400 blocks give 8.91 µs, 3570 give 13.27, so one wave holds 20 warps/SM.

| variant | L2 pass | L0 pass | regs | shared B | waves | opt14 | opt16 | kept |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| T4 baseline (d6a6700) | 18.18 | 14.86 | 97 | 11368 | 2 | 34.95 | 64.39 | – |
| 1. payload → scratch, lanes interleaved per slot | 11.72 (11.70 / 11.72 / 11.73) | 9.84 | 99 | 5032 | 1 | 22.97 (1 run) | 42.50 (1 run) | no (see 1b) |
| 1b. payload → scratch, 3 consecutive words per node | 11.26 (11.21 / 11.26 / 11.28) | 9.42 | 102 | 5032 | 1 | 22.25 | 40.86 | **yes** |
| 1c. 16-byte nodes (4 words) | 11.30 (11.30 / 11.30) | 9.52 | – | 5032 | 1 | – | – | no (+0.4 %) |
| 2. one seeding/relaxation loop, ×4 lengths | 10.86 (10.85 / 10.86 / 10.88) | 9.33 | 96 | 5032 | 1 | 21.52 | 39.98 | **yes** (−3.3 % opt14, −3 regs) |
| 3. branch-free `ld32`, `ll_price`, `rep_after` | 9.20 (9.21 / 9.20 / 9.20) | 7.73 | 97 (98 in the pass kernels) | 5032 | 1 | 18.50 | 33.94 | **yes** |
| 4. u16 literal prices | 9.21 (9.24 / 9.21 / 9.18) | 7.74 | 97 | 4520 | 1 | **18.38** | **33.80** | **yes** (neutral at 64K, −27 % at 16K) |

Pass totals, runs (µs/block, DP passes then fix-up):

| | pass 0 | pass 1 | pass 2 | pass 3 | fix-up | total (3 runs) |
|---|---:|---:|---:|---:|---:|---|
| opt14 T4 | 16.41 | 18.27 (final) | | | 0.27 | 34.95 |
| opt14 T3b | 8.96 | 9.14 (final) | | | 0.27 | 18.42 / 18.38 / 18.37 → **18.38 (−47 %)** |
| opt16 T4 | 15.25 | 15.23 | 15.30 | 18.34 | 0.27 | 64.39 |
| opt16 T3b | 8.12 | 8.10 | 8.15 | 9.17 | 0.26 | 34.09 / 33.80 / 33.71 → **33.80 (−47 %)** |

The study's targets were L2 8.8 and L0 7.8 µs, measured on the T3 kernel without the T4 prologue/epilogue. T3b
reaches 9.2 / 7.7 with them.

**16 KiB** (11600 blocks, 3 runs):

| | T4 | after step 3 | T3b final (u16 lit) |
|---|---:|---:|---:|
| opt14 | 14.67 | 10.40 / 10.39 / 10.41 | 7.57 / 7.55 / 7.57 |
| opt16 | 27.05 | 19.68 / 19.66 / 19.68 | 14.07 / 14.08 / 14.06 |

At 16 KiB a wg16 workgroup holds 4 blocks' tables, so it is bound by shared memory: about 13.0 KB before
the u16 change and 11.0 KB after, which is 7 → 9 workgroups per SM. The first set of 16K runs of step 3 was
bimodal (3.4–5.2 µs for the same pass, with no other GPU process), so only the stable repeats are listed. 16K
runs stay noisier than 64K ones.

### Other variants (not kept)

- wg32 L2 buffer: 10.98. wg8: 13.41. wg16 L2 buffer: 9.18. wg16 stays the default.
- Private-ring fallback (only the prices private): wg16 13.66, wg32 14.32 at L2. It works and is tested;
  it is slower than the workgroup ring, as expected.
- Checked loops (`unbounded: false`): 9.64, +5 %.
- **Heavy-first block order (study #5), measured but not ported.** Blocks were host-sorted by the study's
  cost proxy in `k3opt_passes_timing`:
  - 2900 blocks (one wave): no effect (opt14 18.52 → 18.52, opt16 33.81 → 34.06).
  - 4000 blocks (2 waves): opt14 25.19 → 21.05 (−16 %), opt16 45.07 → 39.30 (−13 %). Unsorted opt16 is
    bimodal per pass (9.1–11.4).
  - 16 KiB, 11600 blocks: opt14 7.6–8.0 → 5.6–6.1, opt16 13.6–14.1 → 10.0–11.6. These runs are noisy.
  - It is not cheap here: in the pipeline the candidate words are GPU-resident, so the order needs a GPU cost
    pass plus a sort, or a previous pass's cost. That belongs to T5 or later, and matters for multi-wave batches
    (5090 batches above 3400 blocks, the 16 KiB path, 8 GB cards).
- Rest-of-loop predication (study #6) and pass fusion (#7) were not attempted (out of scope).
- A GPU game started during one measurement window (15:10–15:13). The runs in that window were discarded and
  repeated after it exited.

### Notes for T5

- VRAM: add `scratch_bytes_per_block(m) × capacity` (6336 B/block at 64K, 18.4 MB at 2900 blocks).
- Keep one K3opt batch at 3400 blocks or fewer on the 5090 (20 warps/SM at ≤ 100 registers, about 4.5 KB
  shared). Past that, a second wave costs a full heavy-block chain unless the blocks are ordered heavy-first.

## T5: integration and measurement (stage S4), branch `worktree-agent-acef2efd020aefb65` (from `m5` 31d5d72), 2026-09-30

Summary and tables: `docs/results/2026-09-30-m5.md`. RTX 5090, 64 KiB, full corpus, `--inflight 3`, GPU idle apart
from an idle `parsecd` (sampled every 0.5 s during each run). The host ran two QEMU VMs and a `haskill` process
(load average 5–30).

### Throughput (three runs per row, MB/s end to end; median in bold)

| config | run 1 | run 2 | run 3 | median | K1 / K2 / K3 / K4 / K5 ms/batch (median) |
|---|---:|---:|---:|---:|---|
| opt14 b2900 | 2176.7 | 2168.5 | 2165.6 | **2168.5** | 19.89 / 7.49 / 51.96 / 1.77 / 3.81 |
| opt16 b2900 | 1428.3 | 1425.7 | 1419.9 | **1425.7** | 19.90 / 7.56 / 96.57 / 1.74 / 3.80 |
| opt14 `--batch max` (b3586) | 1753.3 | 1745.8 | 1743.7 | **1745.8** | 23.90 / 9.02 / 88.26 / 2.12 / 4.47 |
| opt16 `--batch max` (b3586) | 1096.1 | 1089.7 | 1089.1 | **1089.7** | 23.88 / 9.00 / 165.57 / 2.02 / 4.46 |
| lvl9s12seg `--batch max` (b5403) | 10472.0 | 10423.9 | 10442.8 | **10442.8** | 6.46 / 10.96 / 6.84 / 1.79 / 5.16 |
| libzstd L14, 32 threads | 573.5 | 579.4 | 577.6 | **577.6** | – |
| libzstd L16, 32 threads | 496.6 | 495.2 | 498.0 | **496.6** | – |

- Single runs: b3200 gave opt14 2237.0 and opt16 1474.9; b3400 gave 2248.2 and 1487.3.
- The one-wave boundary (≈ 3400 blocks, T3b) holds in the pipeline. K3 per block:
  - `opt14`: 17.9 µs at b2900 against 24.6 µs at b3586;
  - `opt16`: 33.3 µs against 46.2 µs.
- `k3opt_passes_timing` in the same session (2900 blocks):
  - `opt14`: 9.09 + 9.28 + 0.27 = 18.64 µs;
  - `opt16`: 8.23 + 8.17 + 8.24 + 9.20 + 0.26 = 34.10 µs.

### Correctness and ratios

- `gpu --verify` passed on the full corpus at 64, 32 and 16 KiB, and the GPU's compressed bytes equal `ref`'s at
  each size. Ratios are unchanged from T1:
  - 64 KiB: 1.37064 / 1.37144 against L14/L16 1.36827 / 1.37100;
  - 32 KiB: 1.35105 / 1.35158 against 1.34797 / 1.35025;
  - 16 KiB: 1.32765 / 1.32794 against 1.32606 / 1.32774.
- `opt_pipeline` covers 4000 corpus blocks, opt14 and opt16, all four transfer modes, byte-identical. It passed
  at 64 KiB and 16 KiB, with subgroups on and off.
- Full `gzc-gpu` suites passed at 64 KiB, at 64 KiB with no subgroups, and at 16 KiB. `gzc-core` and `gzc-bench`
  pass too.
