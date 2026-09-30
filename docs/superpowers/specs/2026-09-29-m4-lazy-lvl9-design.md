# M4: libzstd level-9 ratio on the GPU — design

Date: 2026-09-29
Status: approved in conversation
Builds on: `2026-09-29-gpu-zstd-compression-design.md` (M0–M3) and `docs/results/2026-09-29-m3.md`

## 1. Purpose

M3 reached libzstd L3 parity on the GPU (ratio 1.271 at 1571 MB/s within 6 GiB VRAM).
M4 raises the GPU's compression ratio to **libzstd level 9** on 128 KiB blocks, with every
frame still bit-exact to a CPU oracle and decodable by libzstd. **Speed is not a goal of
M4**: a later phase iterates on making the level-9 pipeline much faster than the CPU.

Success criterion: on the dev corpus (`data/corpus`, `--ext dds,nif`), the GPU `lvl9`
preset's ratio is **within 1 % of libzstd L9** (measured on the same padded blocks), with
all frames verified by libzstd. If it falls short, the results doc explains the gap and
what would close it.

## 2. What level 9 is for 128 KiB blocks

libzstd uses a separate parameter table for `srcSize <= 128 KB`
(`lib/compress/clevels.h`, vendored under `~/.cargo/registry/src/*/zstd-sys-*/`):

| Level | Strategy | minMatch | searchLog (depth) | hashLog / chainLog |
|---|---|---|---|---|
| L3 | dfast | 5 | – | 16 / 15 |
| L4 | dfast | 4 | – | 17 / 17 |
| L5 | greedy | 4 | 2^3 = 8 | 17 / 16 |
| L6 | lazy | 4 | 8 | 17 / 16 |
| L7 | lazy2 | 4 | 8 | 17 / 16 |
| L8 | lazy2 | 4 | 16 | 17 / 16 |
| **L9** | **lazy2** | **4** | **32** | 17 / 16 |

The L3→L4 ratio jump measured in M3 (1.267 → 1.334) comes from minMatch 5→4. L9 is a
lazy2 parse over a 4-byte hash chain searched 32 deep.

## 3. Normative algorithm (CPU oracle; the GPU mirrors it bit-exactly)

### 3.1 Parameters (runtime, not compile-time)

```rust
pub enum Hashes { Dfast /* long 8 B + short 5 B */, Single /* min_match bytes */ }
pub struct MatchParams { pub hashes: Hashes, pub min_match: u32, pub depth: u32, pub lazy: u32 /*0,1,2*/, pub search_cap: u32 }
```

Presets, defined once in `gzc-core` and used by CPU, GPU and CLI:

| Preset | hashes | min_match | depth | lazy | search_cap | Compare against |
|---|---|---|---|---|---|---|
| `lvl3` | Dfast | 5 | 1 | 0 | 64 | M3 output (byte-identical) / L3 |
| `rung1` | Single | 4 | 8 | 0 | 64 | L5 |
| `rung2` | Single | 4 | 8 | 1 | 64 | L6 |
| `lvl9` | Single | 4 | 32 | 2 | 64 | L9 |

`MatchParams` replaces the compile-time `MIN_MATCH` in the match-finding and parse code.
`HASH_BITS`, `BLOCK_SIZE` and `MATCH_SEARCH_CAP`'s default stay as they are. `min_match` is
limited to 4..=8. Out-of-range params are rejected when the preset table is built and by a
constructor check.

### 3.2 Chains (K1)

- `Dfast`: exactly today's two chains (`hash_long` 8 B, `hash_short` 5 B).
- `Single`: one chain over `hash_single(block, p)`, which hashes `min_match` bytes. For
  4 bytes that is `mix(read_u32(block, p), 0)`. Other widths mask the second word to
  `min_match - 4` bytes. `pred[p]` has the same definition as today: the most recent `q < p`
  with an equal hash, for `p < HASHED_POSITIONS`.

### 3.3 Best match per position (K2)

For each `p < PARSE_END`, walk up to `depth` predecessors along each chain in turn. For
Dfast that is long then short, as today. Compare each candidate with the capped compare
(`search_cap`), keep the longest, and on a tie keep the larger `q`. Store
`best[p] = (offset, capped_len)` if `capped_len >= min_match`. This is unchanged from M3
apart from the parameters.

### 3.4 Parse (K3)

- **`lazy == 0` with `Dfast`**: today's greedy parse, unchanged. The `lvl3` preset must
  reproduce M3's output byte for byte.
- **`lazy == 0` with `Single`** (rung1): today's greedy parse with `min_match` from params.
- **`lazy >= 1`**: a port of libzstd's `ZSTD_compressBlock_lazy_generic`
  (`lib/compress/zstd_lazy.c`, vendored 1.5.7) for `dictMode = noDict`, with
  `depth = lazy`. The only change is that `searchMax(ip)` returns `best[ip]`, extended to
  its true length when `len == search_cap` (as the M3 greedy parse does). The following
  are normative and taken from the vendored source:
  1. The repcode-1 check at `ip+1` before the main search (and `goto store` when `lazy == 0`,
     which cannot happen here).
  2. The literal-run skip `step = ((ip - anchor) >> 8) + 1` when no match of at least
     `min_match` is found. Keep only the `lazySkipping` behaviour that affects which
     positions are searched. Our K2 searches every position, so hash-table insertion
     skipping does not apply; document this.
  3. The deferral loop at `ip+1` (and `ip+2` when `lazy == 2`), using libzstd's integer gain
     formulas exactly: repcode ×3 at step 1 and ×4 at step 2; match gains
     `ml*4 - highbit(offBase)` against `+4` / `+7`; `continue` when a later position wins.
  4. Catch-up: extend the chosen explicit-offset match backwards while
     `start > anchor && start - offset > 0 && block[start-1] == block[start-1-offset]`.
  5. Store the sequence (the M3 rep bookkeeping, `off_base_for` / `apply_off_base`), then the
     immediate `offset_2` repcode loop.
  6. Stop at `PARSE_END`. Trailing literals are handled as in M3.

  Where libzstd reads `MEM_read32` near the end of the input, use the M3 bounded
  compares; nothing may read past `BLOCK_SIZE`. Any intentional difference from libzstd
  is listed in a comment next to the code.

### 3.5 Frames

These are unchanged: the Huffman and FSE entropy stages and the frame layout are the same
on CPU and GPU.

The CPU oracle's `lvl9` is therefore essentially libzstd's L9 algorithm. Its ratio should be
close to libzstd L9 but not bit-exact, for three reasons: a hash chain instead of zstd's
row-hash matcher, a 16-bit hash instead of 17-bit, and capped candidate compares.

## 4. GPU and host

- `Kernels::new(ctx, GpuParams { matching: MatchParams, huffman })` compiles pipelines per
  params, injecting them as WGSL constants alongside the existing ones.
- K1: dispatch y is the number of hashes, and a new `hash_single` goes in `common.wgsl`.
  With a single hash, the `pred` and `head` buffers are allocated for one chain, which saves
  about 0.5 MiB per block.
- K2: a `DEPTH` loop per chain.
- K3: the greedy paths are unchanged. The new lazy path is a WGSL port of §3.4 and is still
  one block per workgroup of size 1. It writes the same `seqs`, `lits` and `counts` layout,
  so K5, K4 and the pipeline are unchanged.
- `vram_bytes` and `max_batch_blocks` take the params into account. The 6144 MiB budget
  still applies.
- CLI: `gzc-bench ref|gpu|all --preset <list>` (default `lvl3`) runs one engine per preset.
  Configs are labelled by preset. `cpu --levels` can go up to 9 (the hard cap stays at 16),
  and the default list stays 1–6.

## 5. Testing

- CPU:
  - All presets round-trip every synthetic case through `reconstruct` and libzstd, at both
    block sizes.
  - Gain-rule boundary tests: hand-built `best[]` or blocks where the deferred match wins by
    exactly 1, and where it ties (the current match is kept).
  - Catch-up and immediate-repcode tests.
  - `lvl3` regression anchor: a test pins a hash of all synthetic `lvl3` frames, captured
    from M3 (commit `c43eee2`) before the refactor.
- GPU:
  - Differential byte equality against the CPU for every preset on all synthetic blocks,
    single and batched (300 mixed, adjacent ends), at both block sizes.
  - A scripted-`best[]` harness that injects K2 output into K3 to reach every lazy branch.
  - Mutation checks on the K3 gain comparisons, catch-up and the `ip+2` path, recorded in
    the report.
- Dev only (`#[ignore]`): differential checks on a few hundred real corpus blocks per preset.

## 6. Measurement

- CPU libzstd: levels 1–9 at 8 threads on the full corpus, plus 1 thread on a 500 MB subset.
- `ref --threads 8` and `gpu` (6 GiB budget, one `--verify` run) for each preset.
- Results doc `docs/results/<date>-m4.md`, containing:
  - a per-rung table comparing our ratio with the matching libzstd level;
  - per-kernel times;
  - the L9 gap analysis.

## 7. Milestones

1. **Params refactor**: `MatchParams` and presets throughout CPU, GPU and CLI. `lvl3`
   stays byte-identical, with the anchor test.
2. **Rung 1**: single 4-byte hash, depth 8, greedy, on CPU and GPU.
3. **Rung 2**: lazy on CPU (zstd port), then K3 lazy.
4. **Rung 3**: lazy2, depth 32 (`lvl9`).
5. **Results doc, then the whole-branch review.**

## 8. Out of scope

Throughput targets and speed optimization (next phase), optimal or price-based parsing,
row-hash matching, dictionaries, and entropy-stage changes.
