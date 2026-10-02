# M5: optimal-parse presets `opt14` / `opt16` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** new presets `opt14` / `opt16` whose ratio is ≥ libzstd L14 / L16 on independent ≤ 64 KiB blocks. The
GPU is byte-identical to a CPU oracle.

**Architecture:**
- **K1:** gains a 3-byte-hash chain.
- **K2opt:** emits two candidate records per position.
- **K3opt:** a segmented (4 KiB, one lane each) port of zstd's optimal parser with exact per-node reps and
  static prices iterated over passes.
- **K4/K5 and the pipeline:** unchanged apart from sizes.

**Spec:** `docs/design/specs/2026-09-30-m5-optimal-parse-design.md` (binding).
**Normative kernel/tie-rule detail:** `docs/design/m5/m5-opt-design.md`.
**Prototypes:** `/tmp/claude-1000/m5d/` (DP port `src/opt.rs`) and `/tmp/claude-1000/m5r/`.

## Global Constraints

- Standard zstd, one frame per block. Blocks are ≤ 64 KiB and independent (64 KiB is the default).
- Existing presets stay byte-identical (lvl3 anchor, all differential tests).
- The GPU is byte-identical to the CPU oracle for the new presets.
- Ratio gates are on the full corpus with `gzc-bench ref`: `opt14` ≥ libzstd L14 and `opt16` ≥ libzstd L16 at
  64, 32 and 16 KiB. The L14/L16 references are measured with `gzc-bench cpu --levels 14,16` on the same
  build.
- u32 WGSL only. VRAM budget 6144 MiB, with `vram_bytes` equal to the real allocations.
- Workgroup storage above 16 KiB only via an explicitly requested adapter limit, with a fallback path.
- Tests at 64K and 16K, plus `GZC_NO_SUBGROUPS=1` and the forced transfer modes where relevant.
- Commits end with `Claude-Session: https://claude.ai/code/session_01C7yssTUPsjdQxYsDW3tGsM`.
- Long commands run in the foreground; no subagents; libzstd levels ≤ 16.

## Review Focus

1. **DP tie rules** (`<=` vs `<`, descending lengths, record order, `newRep` numbering with `ll0`). A wrong rule
   gives valid but different frames. Pinned by hand-built DP cases (T1) replayed on the GPU (T3).
2. **Segment boundaries in the DP.** Candidates are clamped to the segment end, dropped below 3, and the
   series is forced to commit at `iend`. Pinned by T1 segment cases.
3. **Price-table determinism across passes.** Histograms must be exact counts with `off_base` under the decoder
   reps, and `WEIGHT` must be integer. Pinned by the per-pass histogram dumps (T1) compared on the GPU (T4).
4. **`MAX_SEQS` with min match 3** (`BLOCK_SIZE/3 + 1`) and the 3-byte nbSeq header path. Pinned in T5.
5. **Workgroup-memory fallback for K3opt** (a ring of ≈ 23 KiB). Pinned by a test that forces the private-memory
   path (T3).

## Tasks

- **T1: CPU oracle (S0).** Owns `gzc-core`:
  - `opt.rs`: the integer DP port from `/tmp/claude-1000/m5d/src/opt.rs`, cleaned up.
  - `reference::find_cands`: K2opt semantics, spec §3.2.
  - `params.rs`: `Opt3`, `OptParams`, `OPT14`, `OPT16`.
  - `codes.rs`: prior LL/ML/OF tables trained on blocks disjoint from the evaluation sample. Document the
    training set and command.
  - `reference::parse` dispatch; `min_seq_len` 3 for opt.

  Tests:
  - hand-built tie and segment cases (`opt::cases`, like `lazy::cases`);
  - libzstd round trips for synthetic cases;
  - per-pass histogram dumps.

  Gates:
  - sample ratios match the design (`opt16` 1.37211 ± 0.0003, `opt14` 1.37127 ± 0.0005 on the 1/50 sample);
  - full corpus via `gzc-bench ref` against `gzc-bench cpu --levels 14,16` at 64/32/16 KiB.

  Record the results in `docs/results/m5-log.md`.

- **T2: GPU candidates (S1)**, after T1:
  - `common.wgsl` gets `hash3`;
  - K1 builds the `h3` chain (`N_HASHES` = 2 layout);
  - a new `k2_opt.wgsl` writes two words per position;
  - `vram_bytes`/batch sizing for 8 B per position.

  Gate: candidate words byte-identical to `find_cands` on synthetic blocks and 4000 corpus blocks at 64K and
  16K; K1+K2 time recorded.

- **T3: K3opt single pass (S2)**, after T1, in parallel with T2 (feed `find_cands` output through a harness
  like `frames_from_best`):
  - `k3_opt.wgsl`, one lane per 4 KiB segment;
  - the ring in workgroup memory with the adapter limit requested, plus a `var<private>` fallback;
  - trace in `pred`;
  - reuse `main_fixup`.

  Gate: byte-identical to the oracle with `passes: 0, seed: BlockInit` (one final pass), including every
  `opt::cases` entry; per-pass time recorded (target ≤ 6 µs/block on the 5090, occupancy variants if above).

- **T4: passes (S3).** A histogram epilogue plus a price prologue, N passes, and the prior seed plus cover
  literals. Gate: byte identity for `passes 0, 1, 3` × both seeds; per-pass time × passes.

- **T5: integration and measurement (S4).**
  - Full pipeline: K1+K2opt+K3opt+K5+K4, `MAX_SEQS` for min match 3, the batch-max check, the 3-byte nbSeq
    form, and the transfer/direct-upload paths.
  - `--verify` on the corpus.
  - Throughput (median of 3) for `opt14`/`opt16` at 64K.
  - Results doc `docs/results/<date>-m5.md` with ratio tables at 64/32/16 KiB against libzstd L14/L16.

- **After T5:** a whole-branch review, then S5/S6 as a follow-up plan.
