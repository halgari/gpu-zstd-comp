# Research brief: make `opt16` (libzstd-L16 ratio) faster on the GPU

## Use case
- A Skyrim mod downloader compresses about 300 GB of mostly DDS textures (about 95 %) and NIF meshes while downloading at up to 10 Gbit/s (1250 MB/s).
- Target machine: a typical gaming PC with an 8-core CPU and an **8 GB GPU**. Examples: RTX 3060/4060, GTX 1660 Super, RX 6600/7600, Arc A750. Apple M-series is a secondary target.
- Dev machine: RTX 5090 (170 SMs), Ryzen 9 9950X3D.
- One question matters most: can a typical 8 GB card run opt16 at or near line rate?

## Hard constraints
- Output is standard zstd, one frame per block, and libzstd must decode it.
- Blocks are at most 64 KiB (64 KiB is the default) and fully independent: no cross-block references.
- The ratio must stay at or above libzstd L16 on the full corpus at 64 KiB (1.37100). opt16 today gets 1.37144, so the headroom is only +0.032 %. opt14 gets 1.37064, which is 0.027 % below L16. At 16 KiB, opt16's margin is +0.015 %.
- The GPU must be byte-identical to a CPU oracle in `crates/gzc-core`.
  - A different algorithm is allowed, but it needs a new oracle and a new preset, and the ratio gate still applies.
  - A pure speed change must keep output byte-identical to today's oracle.
- VRAM budget is 6144 MiB, and `vram_bytes` must equal the real allocations.
- Code is Rust with wgpu 30 and WGSL: u32 only, subgroups optional, and a non-subgroup fallback must stay.
- Other stacks (CUDA, HIP, raw Vulkan/SPIR-V) may be evaluated. A portable path must remain.
- libzstd levels above 16 are out of scope.

## Current opt16 (RTX 5090, 64 KiB, full corpus, batch 2900, one wave): 1426 MB/s, 44.68 µs/block

| Kernel | µs/block |
|---|---:|
| K1: two hash chains (h4 depth 32, h3 depth 4) | 6.86 |
| K2opt: 2 candidate records per position (A = nearest ≥3, B = longest), 8 B/pos | 2.61 |
| **K3opt, 4 passes + fix-up** | **33.30** (8.23 + 8.17 + 8.24 cheap optLevel-0 passes, 9.20 final optLevel-2 pass, 0.26 fix-up) |
| K4 (sequence FSE and frame) | 0.60 |
| K5 (Huffman literals) | 1.31 |

- `--batch max` (b3586) spills into a second wave on the 5090 and drops to 1090 MB/s.
- For comparison, libzstd L16 on 32 threads does 497 MB/s. lvl9s12seg does 10.4 GB/s at 5.8 µs/block.
- 4060-class projections (not measured): opt16 about 0.20–0.29 GB/s, opt14 about 0.31–0.43 GB/s. These sit below 10 Gbit and near 1–3 Gbit.

## How K3opt works now (after the T3b speed port)
- Each 64 KiB block is split into 16 independent 4 KiB segments, with one lane (thread) per segment and workgroups of 16 lanes (one block per workgroup).
- Each lane runs an integer port of zstd's `ZSTD_compressBlock_opt_generic`:
  - forward price DP over a series, with `sufficient_len = target_length = 32`;
  - a 33-node ring: only the price lives in workgroup memory, and the node payload (3 words) sits in a global scratch buffer;
  - exact per-node rep history;
  - candidates are 3 reps plus the 2 records, with relaxation over match lengths descending;
  - a backward trace (8 B/pos) in the `pred` buffer.
- Prices are static per block per pass (1/256 bit). Pass n's epilogue histograms the fixed-up block parse, and pass n+1's prologue builds the price tables.
- About 4.6 KiB of workgroup memory; 96–102 registers; about 3400 blocks per wave on the 5090.
- A fix-up kernel stitches segments together: literal carry, and offset re-encoding against the true reps.
- Diagnosis from the perf study:
  - A dispatch's time is set by its heaviest blocks: a repetitive block takes 2–3× a typical one.
  - Divergence costs 2.5–3.3×: one active lane per block runs 3.8 ms, 16 lanes in lockstep run 9.4 ms. The cost comes mostly from the union of branch bodies.
  - Occupancy is register- and shared-memory-bound.
  - Subgroup-cooperative relaxation (lanes split match lengths) measured 1.7× slower.
  - 2 KiB segments: only 10–12 % faster, and ratio −0.058 %.
- Design says 3–4 µs/pass; actual is about 8–9.

## Documents to read (paths relative to the repo root /home/tbaldrid/oss/gpu-zstd-comp)
- `docs/results/2026-09-30-m5.md`: results and follow-ups.
- `docs/results/m5-log.md`: per-task logs with every measured variant.
- `docs/design/m5/k3opt-perf.md`: the perf study (ranked, measured).
- `docs/design/m5/m5-opt-design.md`: normative design and projections.
- `docs/design/m5/m5-ratio-drivers.md`: what drives the L14/L16 ratio (measured ablations).
- `docs/design/m5/m5-rt-npu.md`: RT cores, NPUs and tensor cores (no-go).
- `docs/design/specs/2026-09-30-m5-optimal-parse-design.md`: spec, with as-built notes.
- `docs/design/speed/speed2-synthesis.md`: earlier speed research, including dead ends.
- Code:
  - `crates/gzc-gpu/src/shaders/{k1_chains*.wgsl, k2_opt.wgsl, k3_opt.wgsl, k3_fixup.wgsl, common.wgsl}`
  - `crates/gzc-gpu/src/{k3opt.rs, compressor.rs, pipeline.rs, context.rs}`
  - `crates/gzc-core/src/{opt.rs, reference.rs, params.rs, codes.rs}` (the oracle)
  - benches: `crates/gzc-bench`, `crates/gzc-gpu/examples/*`, `crates/gzc-gpu/tests/k3opt.rs` (`k3opt_passes_timing`)
- The corpus is at `data/corpus` (read-only).

## Rules for research agents
- This is research, not implementation. Do NOT modify repository files and do NOT commit.
- Write scratch prototypes only under `/tmp/claude-1000/m6/<your-angle-id>/`.
  - You may copy the repo there, or use `git worktree add /tmp/claude-1000/m6/<id>/wt master`. Remove your worktree when done with `git worktree remove`.
- Run CPU oracle ratio experiments freely, for example with a modified copy of `gzc-core` and `gzc-bench ref` on a corpus sample. Use a `--sample`-style subset or the 1/50 sample used in the design docs, and state the sample.
- GPU microbenchmarks are allowed, but up to 9 other agents may use the GPU at the same time.
  - Keep runs short (under 2 minutes).
  - Always measure your variant against the unmodified baseline in the same session, interleaved, and report ratios rather than absolute times.
  - Check `nvidia-smi` and state the contention.
- Run commands in the foreground with long timeouts. No background runs, no monitors, no subagents.
- Don't ask questions. If blocked, say so in the report.
- Write your report to `/home/tbaldrid/oss/gpu-zstd-comp/.superpowers/m6-research/<angle-id>.md`. If the sandbox blocks that path, use `/tmp/claude-1000/m6/<angle-id>/report.md` and say so.
- The report should cover:
  - the ideas, each with the mechanism, the expected speedup (measured or estimated, and say which), the ratio impact, the exactness impact (byte-identical to today's oracle, or a new oracle), the effort (S/M/L), and the risk;
  - what you measured, and how;
  - the dead ends you checked;
  - a top-3 recommendation.
- Return a summary of at most 300 words.
