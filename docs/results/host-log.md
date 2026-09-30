# Host-side speed track: log

Date 2026-09-30 · branch based on `master` b356ee4 · RTX 5090, Ryzen 9 9950X3D (32 threads), 64 KiB
blocks · full corpus (`--input data/corpus --ext dds,nif`, 6.49 GB, 100754 blocks) · `--batch max`
(b5403 at `--inflight 3`, b6078 at `--inflight 2`, direct upload + transfer-queue readback active).
Two idle QEMU VMs and a desktop session were running (load average 2–4 at rest); other agents ran
GPU and CPU jobs during the session, and runs were gated on no other GPU test/bench process and a
1-minute load average below 4.

**How to read the numbers.** GPU kernel time drifted within the session for the *same* binary
(lvl9 kernel sum 48 → 58 ms/batch, lvl9s12seg 28.6 → 31 ms/batch), so MB/s from different
times of day are not comparable. Every comparison below is **interleaved** (base, new, new w4,
base, …) and each row gives three runs. The column **overhead** is end-to-end wall time minus the
summed kernel time of the run: the part of the wall time the host adds on top of the GPU (fill,
drain, and any time the GPU waited for the host). It is insensitive to the kernel drift.

Host timers are ms per batch (`PipelineStats::transfer_ms`). Before: all on the one pipeline
thread. After: `upload_*` on the producer (calling) thread, `wait`/`deliver` on the completion
thread (see `TRANSFER_NAMES`); `upload_wait` is now the producer's slack (it waits for a free slot).

## Step 1: where the pipeline thread's time went (baseline code, lvl9s12seg i3)

| Measurement | Result |
|---|---|
| Page faults during the upload write (minor, `/proc/self/stat`, per batch) | 110–790 (the writer threads' stacks; ~0.1 ms) — **not a factor** |
| Page faults during delivery, per batch | 0–4 — mapped staging is prefaulted; no prefault/huge-page work needed |
| Upload write vs writer threads (`GZC_UPLOAD_THREADS` 1/2/4/8/16) | 17.7 / 14.2 / 13.5 / 13.5 / 12.7 ms/batch: **bus-bound** at ~26 GB/s (ReBAR write-combined stores over PCIe 5) from 2 threads on |
| Delivery, sink records only the length (no copy) | 0.24 ms/batch: per-frame sink overhead is negligible (e2e 11085 / 11070 MB/s, 2 runs) |
| Delivery, sink reads one byte per cache line of every frame | 8.7 ms/batch: reading the freshly DMA'd staging from DRAM, ~29 GB/s on one thread |
| Delivery, sink copies each frame into a reused 128 KiB buffer | 10.3–10.8 ms/batch |
| Delivery, sink `to_vec`s each frame (the bench's stand-in for writing out) | 15–20 ms/batch: malloc + copy into fresh memory |

So the 15 ms upload is a PCIe transfer the CPU drives (it cannot shrink, only overlap), and the
15 ms delivery is a DRAM read of the staging buffer plus the sink's own copy. The fix is to stop
running them one after the other on one thread.

## Results (interleaved, three runs per row)

| Config | Variant | MB/s (3 runs) | median | overhead ms (3 runs) | host ms/batch |
|---|---|---|---:|---|---|
| lvl9s12seg i3 | base (b356ee4) | 9404 / 10430 / 10362 | 10362 | 135 / 32 / 39 | write 14.7 + deliver 15.5 on one thread, wait 2.2, gpu_idle 0.37 |
| lvl9s12seg i3 | new, w0 | 11036 / 10509 / 10587 | 10587 (+2.2 %) | 30 / 30 / 27 | producer: write 14.3, slot wait 13.6; completion: deliver 10.4; gpu_idle 0.04 |
| lvl9s12seg i3 | new, w4 | 10705 / 10536 / 10604 | 10604 (+2.3 %) | 24 / 26 / 25 | producer: write 14.0, slot wait 14.5; completion: deliver 5.4 |
| lvl9s12seg i2 | base | 9732 / 9065 / 9370 | 9370 | 87 / 124 / 119 | write 16.6 + deliver 19.2, gpu_idle 4.5 |
| lvl9s12seg i2 | new, w0 | 10500 / 10622 / 10535 | 10535 (+12.4 %) | 35 / 33 / 52 | write 15.5, slot wait 18.1; deliver 11.4; gpu_idle 0.42 |
| lvl9s12seg i2 | new, w4 | 10761 / 10804 / 11070 | **10804 (+15.3 %)** | 26 / 28 / 31 | write 15.4, slot wait 17.3; deliver 6.1; gpu_idle 0.05 |
| lvl9 i3 | base | 5817 / 5712 / 5633 | 5712 | 32 / 32 / 31 | write 13.5 + deliver 16.5, wait 29.9 (GPU-bound) |
| lvl9 i3 | new, w0 | 5650 / 5697 / 5506 | 5650 (−1.1 %, noise) | 27 / 28 / 34 | slot wait 40.3; deliver 11.0 |
| lvl9 i3 | new, w4 | 5710 / 5796 / 5688 | 5710 (0 %) | 25 / 25 / 25 | slot wait 39.0; deliver 5.5 |
| lvl9 i2 | base | 5856 / 5789 / 5762 | 5789 | 35 / 34 / 34 | write 15.0 + deliver 19.4, wait 31.6 |
| lvl9 i2 | new, w0 | 5820 / 5873 / 5779 | 5820 (+0.5 %) | 30 / 30 / 31 | slot wait 46.1; deliver 11.9 |
| lvl9 i2 | new, w4 | 5862 / 5754 / 5778 | 5778 (0 %) | 26 / 27 / 29 | slot wait 44.8; deliver 6.4 |

Under CPU load (16 busy-spinning threads on the 32-thread CPU, load average 5–11), lvl9s12seg i3:

| Variant | MB/s (3 runs) | median | overhead ms |
|---|---|---:|---|
| base | 10390 / 9784 / 10191 | 10191 | 65 / 89 / 62 |
| new, w0 | 10802 / 10782 / 10802 | 10802 (+6.0 %) | 28 / 30 / 28 |
| new, w4 | 10338 / 11034 / 10966 | 10966 (+7.6 %) | 51 / 26 / 23 |

Earlier in the session, with faster kernels (28.6–28.9 ms/batch; sequential, not interleaved):
base 10858 / 10934 / 10835, new w0 11306 / 11260 / 11261, new w4 11301 / 11279 / 11272 MB/s
(lvl9s12seg i3). The kernel-only rate in those runs was 11.3–11.5 GB/s.

`--verify` (lvl9s12seg i3 w4, new code): passes, ratio 1.33926 (unchanged: output is byte-identical).

**Reading:** the host is off the critical path: the producer now *waits* 13–18 ms per batch for a
free slot (lvl9s12seg) instead of the GPU waiting for it, `gpu_idle` fell to ~0.04 ms/batch, and
the host's share of the wall time is ~25 ms per run (the first batch's 13.5 ms upload before the
GPU can start, plus the last batch's delivery) against 32–135 ms before. At `--inflight 3` on an
idle box the old pipeline was only just host-bound, hence +2 %; at `--inflight 2` (less slack)
and under CPU load it was clearly host-bound, hence +12–15 % and +6–8 %. lvl9 is GPU-bound and
unchanged. What remains of the gap to the kernel-only rate is the fill (one batch's upload), which
only a smaller first batch could hide, and that loses (below).

## Kept

- **Completion thread + staging leases** (`Pipeline::stream_frames`, `FrameBatch`): delivery
  overlaps the next upload; the slot is recycled only after the sink drops the batch.
- **Parallel delivery** (`run_frames_par`, `FrameBatch::deliver_par`, bench `--writer-threads N`):
  +0–3 % at i3, +3 % at i2, halves drain; the bench default stays 0 (comparability).
- **Zero-copy intake** (`FrameStream::next_upload_slot`, `UploadSlot::{blocks_mut, block_mut,
  regions_mut, submit}`, `pad_payload`, `payload_real_lens`): the caller writes blocks or
  multi-block payloads straight into the mapped upload buffer; `run*` copy in as before. The bench
  still copies (its corpus lives in RAM), so this is an API for real producers, not a bench gain.
- **Zero-copy output** (`FrameBatch` frames borrow the staging buffer; a sink may hand the batch
  to writer threads and release it later).

## Rejected (numbers)

| Variant | Runs (MB/s) | Verdict |
|---|---|---|
| Ramp-up: first submission `batch / k` to start the GPU sooner (lvl9s12seg i3 w4) | k=2: 10859 / 11033 / 10673; k=4: 10913 / 9427 / 10741; k=8: 10653 / 10662 / 10805; vs 11049 / 11016 / 11045 without | the extra (partial) batch costs more GPU time than the ~7–12 ms of fill it hides |
| `GZC_PACK` as the default output (packed contiguous frames, lvl9s12seg i3 w4, interleaved) | 8139 / 7839 / 7771 vs 10249 / 9551 / 11304 | −24 %: packing reads back on the main queue (no transfer queue) and the pack kernel is serial with the kernels; host copy no longer dominates, so no reason to pack on this card |
| Page-fault prefaulting / huge pages for mapped buffers | not built | faults are ~0 (step 1) |
| More upload threads | 8 / 16 threads: 13.5 / 12.7 ms vs 13.5 at 4 | bus-bound; default stays 4 |

## Tests

`cargo test --workspace --release` at 64 KiB and at 16 KiB (`--features …/block-16k`),
`GZC_NO_SUBGROUPS=1` (workspace), and gzc-gpu + gzc-bench with `GZC_TRANSFER_QUEUE=0`,
`GZC_DIRECT_UPLOAD=0`, `GZC_PACK=1`: all pass. New tests in `pipeline.rs`:
`stream_frames_zero_copy_every_mode` (producer writes in place, partial submits, batches released
on a writer thread, 12 batches over 3 slots, every upload/readback mode, ordering and
exactly-once), `stream_frames_held_batches_keep_their_bytes` (held batches stay intact while slots
recycle; an upload slot dropped without submit), `stream_frames_errors_mid_stream` (sink error,
producer error, bad submit sizes, sink and producer panics; the pipeline stays usable, every
mode), `run_frames_par_every_index_once`, `stream_frames_multi_block_payloads` (payloads spanning
blocks, written from threads into a slot holding stale bytes, per-block real lengths).
