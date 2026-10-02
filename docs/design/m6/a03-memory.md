# a03-memory: K3opt memory layout, memory hierarchy and occupancy

Agent a03-memory, 2026-09-30. This is research only: no repository file was changed. Everything lives in
`/tmp/claude-1000/m6/a03-memory/`:
- the kernel variants `v_*.wgsl`;
- the throughput-emulation copies `q_*.wgsl`;
- `host-hooks.diff`, the scratch-only host and test hooks;
- `v_mlc2.diff` and `v_mlc3.diff`, the two recommended kernels as diffs against today's `k3_opt.wgsl`;
- the raw logs `r_*.txt`, and `counts.txt`.

## TL;DR

- **K3opt is not memory-bound, on either GPU.**
  - DRAM traffic is about 1.25 MB per block per pass, or about 5 MB per block for opt16's K3. That is about 8 % of
    the 5090's bandwidth, and about 10 % of a 4060's at the speeds a 4060 can reach.
  - Halving the global load/store instruction count changed nothing (0 ± 0.5 %).
  - Shrinking the shared carve-out changed nothing either.
  - The kernel is bound by the latency and issue of a long divergent dependency chain. It still gains from more
    resident warps.
- **The one memory/occupancy lever that pays is registers.** The changes:
  - pack the 5 match records into one word (`ob | len << 17`);
  - recompute the 6 per-lane base constants from one lane id;
  - drop `pb`, `ll_inc1` and `ll_p0`;
  - optionally, u16 LL/ML/OF tables.

  Together they take K3opt from 98 to 86–88 registers (L0 passes) and 80–81 (L2 pass), as compiled by w2s and
  vkstats. Residency on the 5090 goes from **20 to 23–24 blocks per SM**, measured. All of this is exact.
  - **Batch 2900, one wave:** K3 −3.3 % to −4.3 %.
  - **Throughput regime, the 4060 proxy:** −11 % per cheap pass.
  - **One wave of up to 4080 blocks instead of 3400:** at batch 3586 (today's `--batch max`), K3 is −36 %.
    Today's kernel spills into a second wave there; v_mlc keeps one wave. At 4080 blocks the gain is −31 %.
- The ideas that sounded like memory wins are dead ends:
  - trace compression;
  - candidates or segment data in shared memory;
  - vectorized node and trace accesses;
  - moving the payload back to shared memory;
  - wg32.

  Each one is either impossible within the shared-memory budget or measured at about 0 %.

## 1. Where K3opt's memory traffic goes (measured)

**How it was measured.** `cnt.wgsl` is today's kernel with per-lane event counters. The counters are written to a
region appended to the scratch buffer and read back by the `m6_counts` test. Workload: 2900 corpus blocks at 64 KiB
(the `k3opt_timing` sample, every 34th block), full opt16 schedule. Figures are per input position, per pass. The L0
figures are the mean of the three cheap passes.

| access | L2 (final) pass | L0 pass | memory | notes |
|---|---:|---:|---|---|
| trips | 1.02 | 1.02 | – | |
| candidate words (`best`) | 2.04 words = 8.2 B | 2.04 words | global, streaming | 1 new 128 B line per 16 positions per lane |
| `data` words (`ld32` = 2 words) | 9.2 words = 36.6 B | 9.2 words | global | x + xprev about 3 words, the 3 rep sources about 5.6 words, `match_len` about 0.5 words (0.12 iterations per position) |
| payload scratch, loads | 3.48 words = 13.9 B | 3.45 words | global (L1/L2) | `ld(cur-1)`, `ld(cur)`, `fill` |
| payload scratch, stores | 2.43 words = 9.7 B | 2.44 words | global | |
| ring prices, loads / stores | 1.91 / 0.93 | 2.09 / 0.94 | shared | heavy blocks: up to 4× the mean |
| price-table lookups | 5.9 | 5.3 | shared | lit, ll (2 per `ll_price`), ml, of |
| trace writes | 1.18 words = 4.7 B | 1.17 words | global, fire-and-forget | only positions inside a series, about 59 % |
| trace reads (phase 2) | 0.11 words | 0.08 words | global, scattered | |
| series log + sequence writes | 0.51 words | 0.42 words | global | |
| histogram atomics | 0 | 0.82 | shared | |
| `get_all_matches` calls / rep probes | 0.94 / 2.8 | 0.86 / 2.6 | – | |

**Which level serves each access.** This is an analytic model, checked by the experiments in §3.

| access | RTX 5090 (170 SMs, 96 MB L2) | RTX 4060 (24 SMs, 24 MB L2, 272 GB/s) / RX 7600 (32 MB IC) |
|---|---|---|
| candidates | L1 within a line, otherwise L2 → **DRAM**. No reuse across passes: the batch's candidates are 1.45 GB against a 96 MB L2. | the same, DRAM (480–576 resident blocks × 512 KiB ≫ 24 MB) |
| data (x, rep sources) | L1/L2 (rep sources sit close to p) | the same |
| data (`match_len` sources) | L2, sometimes DRAM (3400 resident blocks × 64 KiB = 218 MB) | L2/IC, sometimes DRAM (about 31–37 MB resident) |
| payload scratch | **L2**: the lanes' hot set is about 127 KB per SM, more than the 20–40 KB of L1 left after shared memory. L2 holds it (21 MB). | **L2**: 3–3.6 MB, fits |
| trace writes | L2, then DRAM write-back | the same |
| trace reads (phase 2) | L2 or DRAM; about 440 dependent reads per lane, about 1 % of a lane's chain | DRAM |
| ring prices, tables, histogram | shared | shared (LDS on RDNA) |

**DRAM per block per pass:**

| traffic | bytes |
|---|---:|
| candidates | 512 KiB |
| data | about 64–128 KiB |
| trace write-back | ≤ 512 KiB |
| trace reads | about 0.1–0.2 MB |
| sequences and log | about 0.13 MB |
| **total** | **≈ 1.25 MB** |

- opt16's K3 is therefore about 5 MB per block. K1 and K2 add about 1.5–2 MB.
- **5090:** at 34 µs per block, K3 moves about 147 GB/s, about 8 % of 1792 GB/s.
- **4060:** at the projected 0.2–0.3 GB/s, about 20–30 GB/s, about 10 % of 272 GB/s.
- **At 10 Gbit line rate** (19.1 k blocks/s), the whole opt16 pipeline would need about 130 GB/s. That is about
  48 % of a 4060's bandwidth. It would become a co-limit only if compute got about 4× faster.

## 2. Memory-bound or latency-bound? A model for the 5090 and a 4060

**Evidence** (5090; interleaved; min of 3–8 rounds; contention noted in §5):

1. **LSU and L1-tag pressure is not the limit.** Two changes measured 0.997–1.004×:
   - vec4 payload nodes plus vec2 trace entries (1 instruction instead of 3 + 3 + 2);
   - vec4 scratch alone.
2. **The L1 carve-out does not matter.** Shared memory in the final pass went 4520 → 3372 B (crossing the 64 KB
   carve-out at about 20 blocks per SM) and measured 1.00×.
3. **Payload latency is a small term.** At equal waves (1360 blocks), payload in shared memory is 5–7 % faster per
   pass. At normal batches its occupancy cost (10.9 KB per workgroup, 9 per SM) loses: −56 % in the throughput
   regime.
4. **Occupancy matters, sublinearly.**
   - **How the throughput regime was emulated.** The `q_*` kernels dispatch 4× the workgroups for the HIST passes
     (block = workgroup id mod n). Scratch and price/histogram regions are indexed per copy. 2900 distinct blocks run
     in about 4 waves, which is the per-SM regime a 4060 (480 blocks per wave) runs in.
   - **How occupancy was varied.** Padding shared memory set the resident workgroups per SM.

   | resident workgroups per SM | µs per block-execution, cheap pass |
   |---:|---:|
   | 8 | 15.6 |
   | ≈ 12 | 11.1 |
   | ≈ 16 | 9.65 |
   | 20 (today) | 8.60 |
   | 23–24 (v_mlc*) | 7.56–7.65 |

   - Measured residency (the knee of one-wave time on uniform repeated blocks): today 20 per SM (3400 per wave,
     98 registers); v_mlc 23; v_mlc2 23; v_mlc3 24.
5. **The one-wave time at 2900 is close to the throughput time** (8.2 against 8.6 µs). On the 5090 the kernel is
   already near its per-SM throughput floor at batch 2900. The heavy-block tail adds only about 5 % there.

**Model.** A trip is a serial chain of about 25–40 dependent operations.
- Most operations are shared memory or L1 (15–20 ns).
- About 3–6 per trip are L2 round trips: payload, candidate-line misses, rep sources.
- One DRAM miss happens per 16 positions per lane.
- 20 warps per SM hide part of this. Each further warp adds throughput at about 0.5× proportionality.

The per-SM behaviour is the same on Ada: same SM organisation, 64 K registers, about 100 KB of shared memory, and
L2 latency in the same range. DRAM bandwidth per SM is nearly identical: 4060 11.3 GB/s per SM against 5090
10.5 GB/s per SM. So:

> **4060 K3 time per block ≈ 5090 throughput-regime time × 170 / 24 × (5090 clock / 4060 clock ≈ 1.05)**

| | opt16 K3 per block (4 passes) | opt16 total (other kernels ≈ 11.4 µs × 7.4) | throughput |
|---|---:|---:|---:|
| today: 3 × 8.6 + 9.6 = 35.4 µs (5090-equivalent) | ≈ 263 µs | ≈ 347 µs | **≈ 0.19 GB/s** |
| v_mlc2/3: 3 × 7.6 + 8.9 = 31.7 µs | ≈ 236 µs | ≈ 320 µs | **≈ 0.20–0.21 GB/s** |

- This sits at the bottom of the brief's 0.20–0.29 GB/s range: no heavy-first order, no fused passes.
- The 4060 is **latency- and issue-bound per SM, exactly like the 5090, and about 7× smaller.** DRAM is about 10 %
  used. Memory engineering cannot close a 5–6× gap to line rate. Removing passes and divergence can.
- **Per-architecture occupancy caveats** (projected, not measured):
  - **RTX 3060 (Ampere, CC 8.6)** allows at most 16 resident blocks per SM. With wg16 that is 16 warps per SM
    whatever the register count, so register cuts do nothing there. wg32 needs ≤ 6.4 KB of shared memory per
    workgroup to reach 16 workgroups (32 blocks); today it uses 8848 B.
  - **RX 7600 (RDNA3)** is LDS-bound. With 64 KB per CU, today's 4520 B gives about 14 workgroups per CU. The
    u16-table variant (4192 B) gives 15. The global-histogram variant (3368 B) gives 19.
  - **Arc** uses SIMD16, so wg16 fills it.

## 3. Ideas evaluated

Speedups are measured on the 5090 under contention unless marked as estimates. "Exact" means byte-identical to
today's oracle: per-pass histograms and the final parse equal `opt::passes`.

| # | idea | mechanism | result | ratio | exactness | effort | risk |
|---|---|---|---|---|---|---|---|
| **R1** | **Register diet** (v_mlc → v_mlc2) | Match records packed into one word (ob 17 bits, len 15 bits). `base`/`cbase`/`tbase`/`wbase`/`sbase`/`lbase` recomputed from one lane id `gl`. `pb` folded to 0 when BPW = 1. `ll_inc1`/`ll_p0` re-read from the tables. Registers 98 → 86/86/81 (w2s + vkstats), residency 20 → 23 blocks per SM. | **2900 batch: K3 −4.3 %** (all passes 0.95–0.96). **Throughput regime: −11 to −12 % per cheap pass.** 4080 batch: −4 % (spills at 23 per SM). | none | **exact**: 1500 corpus blocks (opt16 + opt14), the full `k3opt_passes_synthetic` suite (private ring, wg32, later-pass tables), `opt_cases` | S | low. The register count is the driver's, so a vkstats gate is needed. |
| **R2** | **+ u16 LL/ML/OF tables** (v_mlc3) | ll-by-length, ll-by-code, ml and of packed as u16 pairs (staged as i32 in `ring_p` during the prologue, then packed). Shared memory 4520 → 4192 B, so shared no longer caps residency at 23. Residency 24 per SM (4080 per wave). | 2900: −3.0 %. Throughput: −10.5 %. **Batch 3586: −36 % (measured with v_mlc; v_mlc3 holds at least as many blocks per SM). Batch 4080: one wave, −31 % against today at 4080; 32.1 against 34.2 µs per block at today's best batch (−6 %).** | none | exact on 1500 corpus blocks, wg16 and wg32, `opt_cases`. **The private-ring fallback is broken** (staging writes beyond its 33-entry `ring_p`); a port must stage elsewhere for that path. | S | low |
| R3 | Bigger one-wave batch (follows from R1/R2) | `--batch max` = 3586 today spills into a second wave (1090 MB/s). At 23–24 per SM, 3586 fits one wave. 4080 fits too, but needs ≤ 1.5 MiB per block (today 1.78) and stays below the 2 GiB binding cap (4095 blocks at 8 B/position). | 5090 only: about −5 % per block at 3586 against 2900 (estimated from the 4080 run) | none | exact | S | none |
| R4 | Histogram in global memory (v_ghist) | Counts atomically into the block's `prices` region; shared memory 4520 → 3368 B (wg32: 8848 → 6544). | +1.6 % on cheap passes, **+8 % on pass 0** (block-init literal count via global atomics). Useful only where LDS caps occupancy (RDNA, wg32 on Ampere). | none | exact (300 blocks, wg16 and wg32) | S | low |
| R5 | wg32 (2 blocks per warp), with or without R4 | Doubles the useful lanes per register file | one wave +20 %; throughput regime +14–17 % (slower), even at 30 blocks per SM | none | exact | – | rejected on NVIDIA Ada/Blackwell. Benchmark on Ampere (16-block cap). |
| R6 | Payload ring back in shared memory | No L2 trips for `ld`/`st` | −5 to −7 % at equal waves; −56 % in the throughput regime (9 per SM) | none | exact | – | rejected |
| R7 | vec4 nodes + vec2 trace entries (v_vec, v_vscr) | 1 memory instruction per node or trace entry | 0.997–1.004× | none | exact | – | rejected: not LSU-bound |
| R8 | Smaller shared → bigger L1 carve-out (v_nohist16, final pass) | 3372 B, so L1 grows from about 28 to 64 KB | 1.00× | none | exact | – | rejected |
| R9 | Packed `Node` struct (v_pk) | Keep payload words packed in registers | 0 registers saved on the L0 kernel (89) | – | – | – | rejected |
| R10 | 4 B trace (estimated) | (mlen 6 \| litlen 13 \| ob-class 3) = 22 bits. Explicit offsets are re-read from the candidate words at the match start in phase 2. | **No speed effect expected:** halving trace store instructions measured 0 (R7), and trace writes are fire-and-forget. **VRAM:** the trace is K1's dead `pred`, so it saves 256 KiB per block only if K1's pred also goes to u16 (positions < 65536). That gives 1.78 → 1.53 MiB per block, which is what a 4080-block batch needs on the 5090 (R3). DRAM −0.25 MB per block per pass. | none | exact only with care. The final pass overwrites candidate words with sequences from the segment start, so the sequence layout must move to the segment end, which changes `main_fixup`. | M | medium |
| R11 | Trace only at series ends | – | Not possible: the backtrace needs every node's (mlen, litlen, ob), since any position can be a match start | – | – | – | dead end |
| R12 | Segment data (4 KiB) or the whole chunk (64 KiB) in shared memory | Cheaper `match_len` and rep reads | A lane needs its segment plus arbitrary earlier block bytes, so the block's 64 KiB per workgroup → 1 workgroup per SM (100 KB). Per segment only: 64 KiB per wg16, the same. `match_len` is just 0.12 iterations per position, and the payload-in-shared test bounds the gain at < 7 %. | – | – | – | dead end |
| R13 | Compact candidates in shared memory | – | 512 KiB per block (32 KiB per lane); even 4 B per position is 256 KiB. Does not fit. | – | – | – | dead end |
| R14 | Coalesce candidates by interleaving segments (position-major) | Lanes at the same q share a line | Lanes drift apart (1.02 trips per position, series boundaries), and instruction count is not the limit (R7) | – | – | M | not worth it (estimate) |
| R15 | L2 persistence for candidates | – | Vulkan/WGSL has no access-policy window. Candidates are streamed once per pass and a batch is ≫ L2. | – | – | – | dead end |
| R16 | Scratch per resident slot, not per block | – | 6.3 KiB of 1.78 MiB per block | – | – | – | negligible |

**VRAM footprint.**
- **8 GB cards:** a batch is many waves (a 4060 has about 480–576 blocks per wave against a budget of about 3450
  blocks), so a smaller footprint changes only the partial-last-wave tail (at most 1 in about 7 waves). It does not
  matter much there.
- **5090:** it matters. The budget (3586 blocks) against the wave size (3400 today, 4080 with R2) decides whether
  `--batch max` falls off the 2-wave cliff. With R2, 3586 already fits.
- The 2 GiB `max_storage_buffer_binding_size` caps `best` at 4095 blocks.

## 4. Top-3 recommendation

1. **R1 + R2 register and shared-memory diet** (S, exact, low risk).
   - **5090:** −3 to −4 % at batch 2900. −36 % at `--batch max` 3586 (one wave instead of two). One-wave batches up
     to 4080.
   - **4060-class (throughput regime):** −11 % per cheap pass, measured in emulation.
   - **Gate it:**
     - K3opt residency ≥ 23 blocks per SM: registers ≤ about 88 for the L0 kernels and shared memory ≤ 4.4 KB, as
       a vkstats gate;
     - plus a knee test.
   - Keep the private-ring fallback by staging the u16 tables without `ring_p` on that path.
2. **Re-pick `--batch max` against the wave size, not only VRAM** (S). Choose a multiple of
   (resident blocks per SM × SMs), taken from the knee measurement. Optionally shrink `pred`/trace to 4 B per
   position (R10, M) to reach 4080 blocks within 6144 MiB on the 5090.
3. **Stop spending on memory layout for K3opt.** Spend on pass count and divergence instead.
   - DRAM is about 10 % used on a 4060 at reachable rates.
   - L1/L2 placement changes are measured at ≤ 7 % at best.
   - Memory work cannot close the about 5–6× gap to line rate on an 8 GB card.
   - For RDNA/Ampere ports, keep R4 (global histogram) and wg32 as per-architecture options, to be benchmarked on
     that hardware.

## 5. Method and contention

- **Harness.** A scratch worktree carries host hooks (`host-hooks.diff`):
  - a kernel source override and a grid multiplier;
  - `GZC_SCR_BYTES` and `GZC_PRICES_MUL`;
  - a WGSL dump for vkstats;
  - the tests `m6_interleave` (variants interleaved per round; reports per-pass µs per block), `m6_check` (pass and
    histogram equality with `opt::passes`, opt16 + opt14) and `m6_counts`.
- **Statistics.** Each table entry is the min over 3–8 interleaved rounds of the median of 3 dispatches, reported as
  ratios against `base` (today's kernel) in the same process. A same-kernel A/A run was within 0.3 %.
- **Contention.**
  - Every timed set ran with 4–8 other GPU processes from the other m6 agents (memory used 7–25 GB, sporadic
    60–100 % utilisation) and a load average of 20–140.
  - Outlier rounds (for example r2 in `r_nohist`) were rerun or dropped by the min statistic.
  - Absolute µs figures are under contention.
  - The `PRE`/`POST` lines in each `r_*.txt` record `nvidia-smi` and `uptime`.
- **Knee tests.** 10 corpus blocks were repeated to 3060–4080 blocks. Residency is the block count at which one-wave
  time jumps about 1.5×.
- **Not done.** No Nsight Compute: it is not installed and cannot profile wgpu Vulkan compute. L1 and L2 hit rates
  are therefore modelled, not counted.
