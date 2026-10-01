# a10-contrarian: question the opt16 architecture

Date: 2026-09-30. CPU-only research; no GPU runs, and no repository files were changed. The scratch code is in
`/tmp/claude-1000/m6/a10-contrarian/`: a copied `gzc-core` plus the harness `a10/`; raw results are in
`full_runs.txt` there.

## Headline

**libzstd L16 at 64 KiB writes one zstd block per frame and leaves about 0.2 % ratio unused.** We can split a
64 KiB frame into 2–4 zstd blocks at sequence boundaries. Each block gets its own Huffman and FSE tables, with
treeless literals and FSE Repeat mode allowed. The output is still one standard frame per 64 KiB block, with no
cross-frame references, and libzstd decodes it: every frame was round-tripped.

On the full corpus at 64 KiB, this gives:
- **today's `opt14` parse, unchanged (2 DP passes): 1.37064 → 1.37327, +0.166 % over L16.** opt16 today is
  +0.032 % over L16.
- opt16 + split: 1.37389, +0.211 % over L16.

So the entropy stage can buy back two of opt16's four DP passes, with about 5× the current ratio margin.
On the 5090 that is roughly 44.7 → 30–33 µs/block, about **+40 % throughput**. That number is projected, not
measured on the GPU.

## Method

- **Samples.** The design's 1/50 sample is every 50th 64 KiB block at offset 0: 2016 blocks, real 129.9 MB.
  On this sample libzstd L14 = 1.36911 and L16 = 1.37175.
- **Full corpus.** `data/corpus --ext dds,nif`, 100754 blocks, run as halves or quarters by block index and
  summed. The base byte totals reproduce the documented ones exactly: opt16 4735028925 and opt14 4737822257.
- **Harness.** Parse variants run through the oracle's own `opt::dp_pass` / `seed_prices` / `find_cands`.
  The schedule DSL is:
  - first letter, the seed: `p` = Prior, `b` = BlockInit;
  - then one letter per DP pass: `c` = level-0 pass, `f` = level-2 pass, `g`/`h` = level-2/level-0 pass with
    per-part prices (below).
  - So `bcccf` is opt16 and `pcf` is opt14; both are byte-identical to the oracle.
- **Split frames.** `split.rs` writes them exactly: Repeat mode was added to a scratch `SeqMode`, with treeless
  literal sections and per-block mode choice. Every split frame was checked with `zstd::bulk::decompress` against
  the source block.
- **Timings.** Thread CPU time comes from `/proc/thread-self/schedstat`: min of 3 runs, 250 blocks, one thread.
  **All timings were measured under contention**: load average 50–125 on 32 hardware threads, with the other
  agents' a04/a05/a09 jobs at 340–850 % CPU and SMT siblings busy. Relative numbers were measured interleaved
  per block in the same loop.

## (c) Entropy side — measured

1/50 sample. The ratio columns are whole frames.

| parse | gzc today | + best Huffman depth/description | + best FSE log & normalizer | order-0 bound, no headers | **split frame (exact DP cuts, 4 KiB grid)** |
|---|---:|---:|---:|---:|---:|
| lvl9s12seg | 1.34004 | 1.34005 | 1.34015 | 1.34909 | **1.34267** (+0.196 %) |
| opt14 (`pcf`) | 1.37118 | 1.37119 | 1.37131 | 1.37985 | **1.37383** (+0.194 %) |
| opt16 | 1.37211 | 1.37212 | 1.37224 | 1.38094 | **1.37460** (+0.181 %) |

- **Per-table choices** inside one zstd block are worth almost nothing:
  - Huffman max-depth 5..11 search plus the smaller weight description: +0.001 %;
  - FSE table log 5..max with a cost-optimal normalization: +0.009 %.
  - The 0.64 % gap to the order-0 bound is Huffman's whole-bit code lengths plus headers. The format cannot
    close it.
- **"Accurate" prices** use real Huffman lengths and FSE −log2(norm) instead of `frac_weight`:
  - bcccF (accurate final pass) is +0.012 % over opt16;
  - with 2 passes it hurts: pcF −0.004 %.
  - Not worth a new oracle.
- **Treeless/Repeat modes only exist between blocks of one frame,** so they matter only together with splitting.
- **Splitting.** libzstd L16 on 64 KiB inputs writes **one block per frame** (checked on all 2016 sample
  frames: `{1: 2016}`). zstd turns its block splitter on only when windowLog ≥ 17, and a 64 KiB source caps it
  at 16.
- **How the gain is spread.** It is broad rather than a file-boundary artifact:
  - 81 % of blocks gain;
  - 9 % of blocks hold half of the gain;
  - the mean is 88 B per block.
- **A GPU-friendly cut choice loses little:**

  | cut choice (sample, `pcf`) | ratio | gain |
  |---|---:|---:|
  | exact DP over 4 KiB cells | 1.37383 | +0.194 % |
  | **estimate DP** | **1.37376** | **+0.188 %** |
  | at most 3 parts | 1.37359 | +0.176 % |
  | at most 2 parts | 1.37332 | +0.156 % |
  | 2 parts on an 8 KiB grid | 1.37324 | +0.150 % |

  - The estimate DP works from per-4 KiB-cell histograms: the order-0 entropy of literals + LL/ML/OF codes, plus
    the extra bits, plus a header model of 40 B + 0.3 B per literal symbol + 0.4 B per code. It is robust to the
    header constants (the sweep gave +0.17 to +0.19 %).
  - The cell histograms line up with K3's 4 KiB segments (one lane each).
- **Split-aware prices.** The final pass is priced per part from the previous pass's part histograms (`g`). It
  adds another +0.03 %: `pcg` + split is 1.37418 on the sample.

### Full corpus, 64 KiB

L16 = 1.37100. The opt16 and `pcf` rows use exact cuts; the `pcf (est)`, `pch` and `pcg` rows use estimated cuts.

| preset | ratio | vs L16 |
|---|---:|---:|
| lvl9s12seg | 1.33926 | |
| lvl9s12seg + split (est) | 1.34234 | (+0.230 % over its base) |
| opt14 = `pcf` | 1.37064 | −0.027 % |
| opt16 | 1.37144 | +0.032 % |
| **`pcf` + split (est)** | **1.37320** | **+0.161 %** |
| `pcf` + split (exact cuts) | 1.37327 | +0.166 % |
| `pch` (2 level-0 passes, per-part prices in pass 2) + split | 1.37332 | +0.169 % |
| `pcg` + split | 1.37364 | +0.193 % |
| opt16 + split | 1.37389 | +0.211 % |

**16 KiB check** (1/50 sample of 16 KiB blocks: 7959 blocks, L16 = 1.32513):

| preset | ratio | vs L16 |
|---|---:|---:|
| opt16 | 1.32532 | +0.014 % (full corpus: +0.015 %) |
| `pcf` | 1.32504 | −0.007 % |
| `pcf` + split (2 KiB grid, est) | 1.32577 | **+0.048 %** |

The gain is smaller at 16 KiB, but `pcf` + split still holds about 3× opt16's current 16 KiB margin.

**Costs.**
- **CPU oracle** (same session, interleaved):
  - `pcf` DP: 3.66 ms/block;
  - estimate cuts: 0.35 ms, naive with no prefix sums;
  - split writer: 0.74 ms against 0.36 ms for the 1-block writer, though it uses an exhaustive table search
    that is not needed.
- **GPU (projected, not measured).** The CPU proportions transfer: frame/pass is 0.22 on the CPU and
  (K4+K5)/pass is 0.23 on the GPU.
  - Split is roughly +1 to +4 µs/block on top of K4+K5's 1.9 µs, against 16.4 µs saved by dropping two cheap
    passes.
  - Projected 5090: 30–33 µs/block, or 2.0–2.2 GB/s, against 1.43 for opt16.
  - Projected 4060 class (opt14's ÷5–7 scaling): ≈ 0.27–0.42 GB/s.
- **Decode.** libzstd single-thread decode of split frames is 4–6 % slower: 636–656 against 662–695 MB/s,
  interleaved, under contention.

## (b) Cheaper parse algorithms — measured on the sample (L16 = 1.37175)

| parse | ratio | + split (est) | verdict |
|---|---:|---:|---|
| `pf` (1 pass, Prior) | 1.36727 | 1.36979 | −0.14 % vs L16: dead |
| `q` (1 pass, cover literals per 8/16 KiB region) | 1.36641–1.36677 | 1.36929–1.36958 | dead |
| `pcc` (2 passes, both level 0) | 1.37077 | 1.37349 | +0.13 % |
| `pcf` (= opt14) | 1.37118 | 1.37376 | +0.15 % |
| `pccf` (3 passes) | 1.37185 | 1.37449 | |
| `pcfcf` (4 passes, Prior seed) | 1.37212 | | ≈ opt16 |
| priced lazy, lookahead 1–2, 1–3 re-pricings (`plazy`) | 1.3582–1.3616 | 1.3614–1.3646 | −0.55 to −0.8 % vs `pcf`: dead |

- **Spending the split margin** (all clear L16 on the sample by about 0.10 %):
  - `pcf` with 2 KiB segments: 1.37063, then 1.37320 with split;
  - `pcf` with targetLength 16 and h4 depth 16: 1.37066, then 1.37322 with split.
- **Segment-size sweep** (`pcf` / opt16):

  | segment | `pcf` | opt16 |
  |---|---:|---:|
  | 2 KiB | 1.37063 | 1.37135 |
  | 4 KiB | 1.37118 | 1.37211 |
  | 8 KiB | 1.37146 | 1.37250 |
  | 16 KiB | 1.37161 | 1.37271 |
  | whole block | 1.37172 | 1.37288 |

  - 4 KiB segmentation costs 0.04 / 0.056 % against the whole block, which is more than the prototype's 0.015 %.
  - Cross-segment rep seeding was measured at ±0 in m5-ratio-drivers §3c. Most of the loss is matches cut at
    segment ends, so I did not re-test it.
- **A\* / beam search:** not built.
  - The DP is already exact for its static price model.
  - A beam keeps more rep states per position, so it can only add work.
  - The only ratio lever left on the parse side is the price model, measured above.

## (a) Hybrid CPU+GPU — measured CPU costs, projected throughput

**Measured** (one thread, thread CPU time, min of 3, under contention: load 50–120, SMT siblings busy).
Per 64 KiB block:

| stage | time |
|---|---:|
| chains | 0.24 ms |
| `find_cands` | 2.73 ms |
| opt16 DP, 4 passes (Linear) | 6.79 ms (Ring 7.85) |
| `pcf` DP | 3.66 ms |
| frame | 0.36 ms |
| **libzstd L16, the whole compression** | **4.43 ms** |
| libzstd L9 | 1.15 ms |

The oracle's DP alone costs 1.5× libzstd L16's whole compression.

**Candidate density:**
- 0.357 positions per position carry a record;
- there are 0.452 records per position;
- a compacted list would be about 2.5 B/pos against 8 B/pos.

**PCIe.** 512 KiB of candidates per block:
- at 10 Gbit (19.1 k blocks/s) that is 10 GB/s, which does not fit PCIe 3.0 x8 (about 7 GB/s) and saturates
  4.0 x8 (the 4060 is x8);
- at the CPU DP's own rate it is only about 1.2 GB/s.

**8-core/16-thread gaming CPU, projected.** This assumes every thread is free for compression and per-thread
speed like this machine's SMT-shared threads. Both depend on an idle machine.
- **CPU DP on GPU candidates:**
  - opt16 16 / 6.79 ms → 154 MB/s;
  - `pcf` → 287 MB/s.
  - That is no faster than the GPU alone on a 4060 (0.20–0.29 / 0.31–0.43 GB/s projected), and it needs the
    PCIe round trip. **Dead.**
- **Block-level split, GPU opt + CPU libzstd L16:**
  - the CPU adds about 0.24 GB/s (16 / 4.43 ms; consistent with the documented 497 MB/s at 32 threads);
  - 4060 + CPU ≈ 0.44–0.53 GB/s with opt16, or 0.51–0.66 GB/s with `pcf` + split.
  - It is still far below 1.25 GB/s.
  - Output depends on scheduling unless the assignment is fixed by block index, which breaks reproducibility.
- **The CPU is probably busy anyway.** LZMA (7z) decode measured 50 MB/s per core (xz -T1, under contention).
  Unpacking 7z sources at 1.25 GB/s alone would need many cores. If the source archives are 7z, a 10 Gbit
  install is CPU-bound before compression starts. Check the real archive mix before optimizing the GPU for
  10 Gbit.

## (d) Questioning the target

Per 300 GB of input (full-corpus ratios):

| preset | GB out | saved vs lvl9s12seg |
|---|---:|---:|
| lvl9s12seg | 224.00 | – |
| lvl9s12seg + split | 223.49 | 0.51 |
| opt14 | 218.88 | 5.13 |
| L16 | 218.82 | 5.19 |
| opt16 | 218.75 | 5.26 |
| **`pcf` + split** | **218.47** | **5.54** |
| opt16 + split | 218.36 | 5.65 |

**Time (4060 class, projected).** Downloading 300 GB at 10 Gbit takes 240 s.

| preset | projected speed | time for 300 GB | extra over the 240 s download |
|---|---:|---:|---:|
| lvl9s12seg | 1.5–2.1 GB/s | ≤ 240 s | 0 (keeps line rate) |
| opt16 | 0.20–0.29 GB/s | 1034–1500 s | +13 to 21 min, to save 5.3 GB (≈ 4–7 MB per extra second) |
| `pcf` + split | 0.27–0.42 GB/s | about 710–1110 s | +8 to 15 min, to save 5.5 GB |

- **The opt14 → opt16 step is worth only 0.13 GB per 300 GB (0.058 %)** for +52 % GPU time.
- At ≤ 2 Gbit, which covers most users, every opt preset on a 4060 keeps up with the line. The 10 Gbit line-rate
  goal only matters for a minority of users.
- For them, an adaptive per-block policy loses almost all of the opt gain at 10 Gbit on a 4060 (only about 7 %
  of blocks could be opt). A background "upgrade later" recompression is the honest way to get both.

## Ideas, rated

| # | idea | speed | ratio | exactness | effort | risk |
|---|---|---|---|---|---|---|
| 1 | **In-frame block split + `pcf` (opt14 parse) as the L16 preset** | ≈ +40 % on the 5090 against opt16 (projected; K3 halves, K4/K5 +1–4 µs) | +0.161 % over L16 (full), 16 KiB +0.048 % (sample) | parse byte-identical to today's opt14 oracle; new frame-writer oracle (multi-block, treeless, Repeat, integer cut estimator) | M | consumers that assume one zstd block per frame; decode −5 %; the f64 estimator needs an integer port (log2_x256) |
| 2 | Split-aware prices (`pcg`/`pch`) | ±0 | +0.03 % on top of #1 | new oracle | S, after #1 | low |
| 3 | Spend the split margin: 2 KiB segments (+10–12 % K3 per the perf study), or TL16 + depth 16 | 10–20 % K3/K1 | still ≈ +0.10 % over L16 (sample) | new oracle | S | margin bookkeeping |
| 4 | Split for lvl9s12seg | ±0 | +0.23 % (full) | new oracle | S, shares #1 | none |
| – | CPU DP on GPU candidates | worse than the GPU alone | – | – | – | dead |
| – | CPU libzstd L16 as a co-worker | +0.24 GB/s only if the CPU is idle | = L16 | nondeterministic unless fixed | S | CPU busy unpacking |
| – | Huffman depth / FSE table-log / accurate prices | – | ≤ +0.012 % | – | – | dead |
| – | 1-pass + split, priced lazy | – | −0.14 % / −0.8 % | – | – | dead |

## Top 3

1. **Redefine the L16-class preset as "opt14 parse + in-frame block split".**
   - Full corpus: +0.161 % over L16, against +0.032 % for opt16 today.
   - It drops two of the four DP passes. K1/K2/K3 for opt14 already exist and match the oracle, so the work is
     mostly K4/K5 plus a new frame-writer oracle.
2. **Add split-aware final-pass prices** (+0.03 %), and spend part of the margin on 2 KiB segments or a
   shallower K1 for small GPUs.
3. **Before chasing 10 Gbit on 8 GB cards:**
   - measure the real archive-unpack CPU cost;
   - adopt a per-throughput policy: opt when the GPU keeps up (it does at ≤ 2 Gbit), lvl9s12seg + split
     otherwise, with an optional background upgrade.

## Reproduce

```sh
cd /tmp/claude-1000/m6/a10-contrarian && cargo build --release --offline
./target/release/a10 ent lvl9s12seg pcf opt16            # entropy-choice table
./target/release/a10 xsplit opt16 pcf lvl9s12seg         # exact split frames, libzstd-verified
EST=40,0.3,0.4 ./target/release/a10 dpsplit pcf pcg pch  # GPU-style estimated cuts
EST=1 EVERY=2 OFFSET=0 ./target/release/a10 full pcf     # full corpus (halves)
./target/release/a10 time 250; ./target/release/a10 nblocks; ./target/release/a10 dec
```
