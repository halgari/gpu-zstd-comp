# M7 research brief: GPU-friendly ways to reach libzstd-L16 ratio

## Goal from the user
"Research alternative ways of getting lvl16 compression in a way that is more friendly to the GPU. Byte-perfect output compared to libzstd has never been a requirement. We want maximum performance from the GPU, with ratios equal to that level of compression or better."

## Hard constraints
- The output must be standard zstd: one frame per block, blocks of at most 64 KiB (64 KiB is the default), fully independent of each other, and decodable by stock libzstd. No custom decoder, no dictionaries, no references across blocks, and no data transforms that need an inverse step.
- Ratio: at least libzstd L16 on the full corpus at 64 KiB (L16 is 1.37100). It should also hold at 32 and 16 KiB (L16 is 1.35025 and 1.32774).
- Any algorithm we ship needs a CPU oracle in `gzc-core` that the GPU matches byte for byte. That is an engineering rule for us, not a link to libzstd. A research prototype only needs a CPU model that measures ratio and estimates GPU cost.
- Target hardware: 8 GB gaming GPUs (RTX 3060/4060, RX 6600/7600, GTX 1660 Super) and Apple M-series. The dev GPU is an RTX 5090.
- Code is Rust + wgpu 30 + WGSL: u32 only, subgroups optional, no CUDA.

## Where we are
- opt16 is a port of zstd's btultra optimal parser: 4 DP passes, 4 KiB segments with one lane each, and exact per-node rep history.
  - RTX 5090: 1.1–1.5 GB/s, ratio 1.37144.
  - About 75 % of the time is the DP, which is a long sequential dependency chain per lane.
- opt14 (2 passes) reaches 1.37064.
- lvl9s12seg (a lazy2 parse) runs at 10.4 GB/s with ratio 1.33926.
- Projected opt16 on an RTX 4060: about 0.15–0.2 GB/s.
- Measured and documented, so don't redo these, though you may build on them:
  - `docs/superpowers/m6/synthesis.md`, with reports a01–a12 in the same folder. Read the synthesis first, then the reports relevant to your angle.
  - Dead ends: rep-history approximations cost 0.1–0.4 %; a 1-pass L16 misses by 0.14 %; a lazy parse with price guidance loses 0.55–0.8 % against opt14; BC-aligned search; and speculative DP chunks, which keep the ratio but only help the slowest tail.
  - Wins that are in progress, so assume they will exist:
    - splitting a frame into several zstd blocks (+0.16 %);
    - an h10 candidate chain (+0.10 %);
    - in-pass price refresh (+0.08 %);
    - "gap3" (+0.02 %);
    - kernel-level speedups to the existing DP.
- Corpus: `data/corpus` with `--ext dds,nif`, 6.49 GB, 100,754 blocks of 64 KiB. About 95 % is DDS (BC1/BC3/BC5 and some uncompressed) and the rest is NIF meshes.
- Use the 1/50 sample (2016 blocks) for exploration, and the full corpus only for finalists.

## Code
- CPU oracle and helpers in `crates/gzc-core/src/`:
  - `opt.rs` — the DP, prices, passes;
  - `reference.rs` — `find_cands`, parse dispatch;
  - `frame.rs` and `seqenc.rs` — the frame writer;
  - `huffman.rs` and `fse.rs`;
  - `params.rs`;
  - `codes.rs` — the priors.
- `gzc-bench ref` (CPU oracle ratio) and `gzc-bench cpu --levels 14,16` (libzstd).
- GPU kernels in `crates/gzc-gpu/src/shaders/`.
- Scratch code from the earlier round is in `.superpowers/m6-research/artifacts/<agent>/`.

## Rules
- Research only. Do not modify the repository.
- Work in `~/.cache/gzc-m7/<your-id>/` (a disk path; /tmp is a RAM disk). Delete build outputs when you finish, but keep source files and diffs, and copy the important ones to `.superpowers/m7-research/artifacts/<your-id>/`.
- **The machine is shared.** Another agent is timing the GPU, another uses 16 CPU threads, and four other research agents are running.
  - Cap yourself at **6 CPU threads**.
  - GPU microbenchmarks are allowed only if they are short (under 1 minute), interleaved against a baseline, and reported as ratios. Prefer cost models: op counts, dependent-chain length, parallel width.
- Run commands in the foreground with long timeouts. No subagents, no questions.
- Report to `/home/tbaldrid/oss/gpu-zstd-comp/.superpowers/m7-research/<your-id>.md`. For each idea, give:
  - the mechanism;
  - the ratio, measured, against L16 (and L14) at 64 KiB, plus 16 KiB for finalists;
  - the GPU cost estimate against today's opt16: dependent-chain length, parallel width, memory, passes;
  - how a byte-exact GPU/CPU oracle would work;
  - effort and risk.

  Include the dead ends you checked and a top-3. Return a summary of at most 300 words.
