use super::*;
use crate::sizing::{BufferSizes, chain_pred_bytes, head_bytes, pred_bytes_for};
use gzc_core::block::chunk_file;
use gzc_core::fixtures::RUNG1;
use gzc_core::params::{LVL3, LVL9SEG, LVL9S12SEG, MatchParams, OPT14, OPT16, OPT16P1};
use gzc_core::reference::compress_block;
use gzc_core::synth::test_cases;

std::thread_local! {
    /// Test hook: bytes `Pipeline::new` adds to each upload buffer it creates on this thread.
    pub(super) static EXTRA_UPLOAD_BYTES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// An upload buffer far beyond any device (and above wgpu's `max_buffer_size` where that is
/// smaller): `Pipeline::new` fails cleanly at once, naming the allocation, rather than
/// handing out a pipeline whose upload buffer is invalid ("Buffer with 'pipeline.upload'
/// label is invalid" at the first `next_upload_slot`), and the context stays usable.
#[test]
fn absurd_upload_buffer_fails_cleanly_in_new() {
    let _gpu = crate::testing::gpu_test_slot();
    let ctx = crate::testing::gpu();
    let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
    let cfg = PipelineConfig { batch: 4, inflight: 2, params };
    // 1 PiB: no allocator can place it, and it fails before any memory is committed.
    EXTRA_UPLOAD_BYTES.set(1 << 50);
    let r = Pipeline::new(&ctx, &cfg);
    EXTRA_UPLOAD_BYTES.set(0);
    let msg = match r {
        Ok(_) => panic!("a 1 PiB upload buffer was created"),
        Err(e) => format!("{e:#}"),
    };
    eprintln!("Pipeline::new error: {msg}");
    assert!(msg.starts_with("GPU allocation of ") || msg.starts_with("GPU device lost while allocating"), "{msg}");
    assert!(msg.contains("batch 4, inflight 2"), "{msg}");
    if msg.starts_with("GPU allocation") {
        assert!(msg.contains("pipeline.upload") && msg.contains("batch_blocks"), "{msg}");
    }
    // Out of memory and validation errors leave the device usable (a backend that loses the
    // device instead is reported as such above).
    if ctx.device_lost().is_none() {
        let blocks: Vec<Vec<u8>> = distinct_blocks().into_iter().take(6).collect();
        let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
        let mut got = CollectFrames(vec![None; refs.len()]);
        Pipeline::new(&ctx, &cfg).unwrap().run_frames(&refs, &mut got).unwrap();
        for (i, b) in refs.iter().enumerate() {
            assert!(got.0[i].as_deref() == Some(cpu_frame(b, params).as_slice()), "block {i}");
        }
    }
}

/// On a lost device every allocation fails with an error wgpu hands to no error scope (the
/// M4 Pro's parallel-test failure: a lost device left 'pipeline.upload' invalid, surfacing at
/// the first `next_upload_slot`). `Pipeline::new` must report the loss itself.
#[test]
fn pipeline_new_on_lost_device_errors() {
    let _gpu = crate::testing::gpu_test_slot();
    let ctx = crate::testing::gpu();
    let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
    ctx.device.destroy();
    // A hal error loses the device at once (wgpu-core calls the lost callback then); after
    // `destroy` the callback comes with the next poll.
    let _ = ctx.device.poll(wgpu::PollType::Poll);
    assert!(ctx.device_lost().is_some(), "the device-lost callback did not fire");
    let msg = match Pipeline::new(&ctx, &PipelineConfig { batch: 4, inflight: 2, params }) {
        Ok(_) => panic!("Pipeline::new succeeded on a destroyed device"),
        Err(e) => format!("{e:#}"),
    };
    eprintln!("Pipeline::new error: {msg}");
    assert!(msg.starts_with("GPU device lost"), "{msg}");
    // The allocation check itself, for a loss after the constructor's first check.
    let e = BatchBuffers::new(&ctx, 2, true, &LVL3).err().expect("batch buffers on a lost device");
    assert!(format!("{e:#}").starts_with("GPU device lost while allocating"), "{e:#}");
}

struct Collect(Vec<Option<BlockOutput>>);

impl BlockSink for Collect {
    fn put(&mut self, index: usize, out: BlockOutput) {
        assert!(self.0[index].is_none(), "index {index} delivered twice");
        self.0[index] = Some(out);
    }
}

/// The frame path's kernel timers (Huffman literals): all five, each positive. The one
/// exception: on an adapter that samples timestamps at pass boundaries only (no
/// `TIMESTAMP_QUERY_INSIDE_ENCODERS`: Metal on Apple GPUs), the device may leave the last
/// kernel's (K4's) pair unwritten, so `kernel_ms` leaves K4 out (`PipelineStats::kernel_ms`);
/// every other timer must still be there.
fn assert_frame_timers(ctx: &GpuContext, stats: &PipelineStats, what: &str) {
    if !ctx.timestamps {
        assert!(stats.kernel_ms.is_empty(), "{what}: {:?}", stats.kernel_ms);
        return;
    }
    let all = ["k1_chains", "k2_best", "k3_parse", "k4_entropy", "k5_huffman"];
    let names: Vec<&str> = stats.kernel_ms.iter().map(|(n, _)| n.as_str()).collect();
    if ctx.timestamps_inside_encoders || names != ["k1_chains", "k2_best", "k3_parse", "k5_huffman"] {
        assert_eq!(names, all, "{what}");
    }
    assert!(stats.kernel_ms.iter().all(|&(_, ms)| ms > 0.0), "{what}: {:?}", stats.kernel_ms);
}

fn cfg(batch: u32, inflight: u32) -> PipelineConfig {
    PipelineConfig { batch, inflight, params: GpuParams { matching: LVL3, emit_frames: false, huffman: true } }
}

#[test]
fn stream_matches_reference_every_index_once() {
    let _gpu = crate::testing::gpu_test_slot();
    let ctx = crate::testing::gpu();
    let distinct: Vec<Vec<u8>> =
        test_cases().into_iter().flat_map(|(_, bytes)| chunk_file(&bytes)).map(|b| b.data).collect();
    let want: Vec<BlockOutput> = distinct.iter().map(|b| compress_block(b, LVL3)).collect();
    // 1000 = 15 * 64 + 40: the last batch is partial and reuses a slot holding stale blocks.
    let blocks: Vec<&[u8]> = (0..1000).map(|i| distinct[i % distinct.len()].as_slice()).collect();

    let mut sink = Collect(vec![None; blocks.len()]);
    let stats = Pipeline::new(&ctx, &cfg(64, 3)).and_then(|mut p| p.run(&blocks, &mut sink)).expect("run");

    for (i, got) in sink.0.iter().enumerate() {
        let got = got.as_ref().unwrap_or_else(|| panic!("index {i} never delivered"));
        assert!(*got == want[i % distinct.len()], "index {i}: GPU != reference");
    }
    assert_eq!(stats.batches, 16);
    assert!(stats.wall_s > 0.0);
    if ctx.timestamps {
        let names: Vec<&str> = stats.kernel_ms.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["k1_chains", "k2_best", "k3_parse"]);
        assert!(stats.kernel_ms.iter().all(|&(_, ms)| ms > 0.0), "{:?}", stats.kernel_ms);
    }
}

#[test]
fn stream_odd_batch_single_slot_and_reuse() {
    let _gpu = crate::testing::gpu_test_slot();
    // Odd batch: the staging timestamp region is only 4-byte aligned. One slot: every batch
    // waits for the previous one. The pipeline is reused for a second run.
    let ctx = crate::testing::gpu();
    let distinct: Vec<Vec<u8>> =
        test_cases().into_iter().flat_map(|(_, bytes)| chunk_file(&bytes)).map(|b| b.data).collect();
    let blocks: Vec<&[u8]> = distinct.iter().map(|b| b.as_slice()).collect();
    let mut pipe = Pipeline::new(&ctx, &cfg(7, 1)).unwrap();
    for _ in 0..2 {
        let mut sink = Collect(vec![None; blocks.len()]);
        let stats = pipe.run(&blocks, &mut sink).unwrap();
        assert_eq!(stats.batches as usize, blocks.len().div_ceil(7));
        for (i, got) in sink.0.into_iter().enumerate() {
            assert!(got.unwrap() == compress_block(blocks[i], LVL3), "index {i}");
        }
    }
    let mut sink = Collect(vec![None; 1]);
    assert!(pipe.run(&[&[0u8; 3]], &mut sink).is_err(), "short block rejected");
}

#[test]
fn stream_handles_empty_input_and_rejects_bad_config() {
    let _gpu = crate::testing::gpu_test_slot();
    let ctx = crate::testing::gpu();
    let mut sink = Collect(Vec::new());
    let mut run = |c: &PipelineConfig| Pipeline::new(&ctx, c).and_then(|mut p| p.run(&[], &mut sink));
    let stats = run(&cfg(8, 2)).unwrap();
    assert_eq!(stats.batches, 0);
    assert!(run(&cfg(0, 2)).is_err());
    assert!(run(&cfg(8, 0)).is_err());
    assert!(run(&cfg(u32::MAX, 1)).is_err());
    let bad = MatchParams { lazy: 3, ..LVL9SEG };
    let bad = PipelineConfig { params: GpuParams { matching: bad, ..cfg(8, 2).params }, ..cfg(8, 2) };
    let e = Pipeline::new(&ctx, &bad).err().expect("lazy 3 is invalid");
    assert!(e.to_string().contains("lazy 3"), "{e}");
}

struct CollectFrames(Vec<Option<Vec<u8>>>);

impl FrameSink for CollectFrames {
    fn put(&mut self, index: usize, frame: &[u8]) {
        assert!(self.0[index].is_none(), "index {index} delivered twice");
        self.0[index] = Some(frame.to_vec());
    }
}

fn distinct_blocks() -> Vec<Vec<u8>> {
    test_cases().into_iter().flat_map(|(_, bytes)| chunk_file(&bytes)).map(|b| b.data).collect()
}

/// The CPU oracle's frame of a block given as its real bytes (1..=BLOCK_SIZE).
fn cpu_frame(block: &[u8], params: GpuParams) -> Vec<u8> {
    gzc_core::reference::compress_block_to_frame(block, params.matching, params.frame_options())
}

/// Files of k BLOCK_SIZE blocks plus r bytes, each block given as its real bytes: every
/// preset's frames equal the CPU oracle's (`compress_block_to_frame` of the real bytes, which
/// pins full blocks to their unchanged padded-parse frames) and decode with libzstd to exactly
/// the files. Covers batches mixing full and partial blocks, an all-full batch, RLE and
/// 1-byte-FCS (< 256 bytes) blocks, lengths 2 and 3 (around zstd's minimum match), a segment
/// boundary (4095, 4096, 4097) and zero-tailed blocks (real bytes that run on into the
/// padding, so K3t cuts a match that crosses the real length); every upload/readback mode
/// for three presets, Raw literals for lvl3. Every frame declares its block's real length
/// (K4 takes it from the cut parse, the host from `lens`), and K3t is timed exactly when a
/// batch holds a partial block.
#[test]
fn partial_blocks_frames_match_cpu_every_preset() {
    let _gpu = crate::testing::gpu_test_slot();
    let source: Vec<u8> = test_cases().into_iter().flat_map(|(_, bytes)| bytes).collect();
    let bs = BLOCK_SIZE;
    let lens = [1, 2, 3, 100, 255, 256, 257, 4095, 4096, 4097, bs - 1, bs, bs + 31_000, 2 * bs + 3];
    let mut files: Vec<Vec<u8>> =
        lens.iter().enumerate().map(|(i, &n)| source[i * 7777..][..n].to_vec()).collect();
    files.push(vec![7u8; bs + 300]);
    files.push(vec![7u8; 200]);
    // Zero-tailed: text whose last bytes are zeros, and all-zero partial blocks.
    for n in [3000, bs + 4096, bs / 2] {
        let mut f = source[50_000..][..n].to_vec();
        let cut = n - (n % bs) / 2;
        f[cut..].fill(0);
        files.push(f);
    }
    files.push(vec![0u8; 5000]);
    files.push(vec![0u8; 3]);
    // Eight full blocks: one batch of 8 with no partial block.
    files.push(source[3 * bs..11 * bs].to_vec());
    let blocks: Vec<gzc_core::block::Block> = files.iter().flat_map(|f| chunk_file(f)).collect();
    let real: Vec<&[u8]> = blocks.iter().map(|b| b.real()).collect();
    let default_ctx = [("default".to_string(), crate::testing::gpu())];
    let all_modes = mode_contexts();
    for (name, m) in gzc_core::params::PRESETS {
        let modes = if [LVL3, LVL9S12SEG, OPT16P1].contains(&m) { &all_modes[..] } else { &default_ctx[..] };
        let raw_lits = if m == LVL3 { &[true, false][..] } else { &[true][..] };
        for &huffman in raw_lits {
            let params = GpuParams { matching: m, emit_frames: true, huffman };
            let want: Vec<Vec<u8>> = real.iter().map(|b| cpu_frame(b, params)).collect();
            let mut at = 0;
            for f in &files {
                let n = payload_blocks(f.len());
                let dec: Vec<u8> = want[at..at + n].iter().flat_map(|w| zstd::bulk::decompress(w, bs).unwrap()).collect();
                assert!(dec == *f, "{name}: a {}-byte file did not round-trip", f.len());
                at += n;
            }
            for (mode, ctx) in modes {
                let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 8, inflight: 2, params }).unwrap();
                let mut sink = CollectFrames(vec![None; real.len()]);
                let stats = pipe.run_frames(&real, &mut sink).unwrap();
                for (i, got) in sink.0.iter().enumerate() {
                    let got = got.as_ref().unwrap_or_else(|| panic!("{name} {mode}: block {i} never delivered"));
                    assert!(
                        *got == want[i],
                        "{name} huffman={huffman} {mode}: block {i} ({} bytes): GPU frame != CPU frame",
                        real[i].len()
                    );
                    assert_eq!(
                        zstd::zstd_safe::get_frame_content_size(got).ok().flatten(),
                        Some(real[i].len() as u64),
                        "{name} {mode}: block {i}: K4's declared content size is not the block's length"
                    );
                }
                // K3t ran (some batch held a partial block): timed last, after the kernels.
                if ctx.timestamps {
                    let (last, ms) = stats.kernel_ms.last().expect("kernel timers");
                    assert_eq!(last, TRUNC_KERNEL_NAME, "{name} {mode}: {:?}", stats.kernel_ms);
                    assert!(*ms > 0.0, "{name} {mode}: {:?}", stats.kernel_ms);
                    assert_eq!(stats.kernel_ms.iter().filter(|(n, _)| n == TRUNC_KERNEL_NAME).count(), 1);
                }
                // Full blocks only: no K3t, no timer for it.
                let full: Vec<&[u8]> = real.iter().copied().filter(|b| b.len() == bs).collect();
                let mut sink = CollectFrames(vec![None; full.len()]);
                let stats = pipe.run_frames(&full, &mut sink).unwrap();
                assert!(stats.kernel_ms.iter().all(|(n, _)| n != TRUNC_KERNEL_NAME), "{name} {mode}: {:?}", stats.kernel_ms);
                assert!(sink.0.iter().all(|f| f.is_some()));
                if huffman && m == LVL3 {
                    assert_frame_timers(ctx, &stats, name);
                }
            }
        }
    }
    let ctx = &default_ctx[0].1;
    let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
    let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 8, inflight: 2, params }).unwrap();
    let mut sink = CollectFrames(vec![None; 2]);
    let e = pipe.run_frames(&[&[1u8; 10][..], &[][..]], &mut sink).unwrap_err().to_string();
    assert!(e.contains("block 1"), "{e}");
    assert!(pipe.run_frames(&[&vec![1u8; bs + 1][..]], &mut sink).is_err());
}

#[test]
fn staging_regions_are_aligned() {
    for cap in [1, 7, 1365, 1535, 1890] {
        for frames in [true, false] {
            let l = StagingLayout::new(cap, frames, &LVL3);
            assert!([l.a, l.ts].iter().all(|x| x % STAGING_ALIGN == 0), "cap {cap} frames {frames}");
            let end = l.a + if frames { frames_bytes(cap) } else { seqs_bytes_for(cap, &LVL3) };
            assert!(l.a >= if frames { frame_len_bytes(cap) } else { counts_bytes(cap) } && end <= l.ts);
        }
    }
}

/// M6 `opt16p1`'s sparse chains cost 3 * BLOCK_SIZE / 4 pred words (192 KiB at 64 KiB) per
/// block over opt16 once the head tables are capped, which `vram_bytes` counts, so a VRAM
/// budget's largest batch (gzc-bench `--batch max`) is smaller.
#[test]
fn vram_counts_opt16p1_sparse_chains() {
    use gzc_core::params::{OPT16, OPT16P1};
    let frames = |m, batch, inflight| PipelineConfig {
        params: GpuParams { matching: m, emit_frames: true, huffman: true },
        ..cfg(batch, inflight)
    };
    // At batches whose head tables are capped (5 * n and 2 * n >= HEAD_TABLES).
    for n in [200u32, 2900] {
        let extra = vram_bytes(&frames(OPT16P1, n, 2)) - vram_bytes(&frames(OPT16, n, 2));
        assert_eq!(extra, n as u64 * 3 * BLOCK_SIZE as u64);
    }
    let max_at = |m, inflight, budget_mib: u64| {
        (1..=20_000u32).take_while(|&n| vram_bytes(&frames(m, n, inflight)).div_ceil(1 << 20) <= budget_mib).last().unwrap()
    };
    for inflight in [1, 2, 3] {
        let (a, b) = (max_at(OPT16, inflight, 6144), max_at(OPT16P1, inflight, 6144));
        assert!(b < a, "inflight {inflight}: opt16 {a} opt16p1 {b}");
        assert!(vram_bytes(&frames(OPT16P1, b, inflight)) <= 6144 << 20);
        assert!(vram_bytes(&frames(OPT16P1, b + 1, inflight)).div_ceil(1 << 20) > 6144);
        eprintln!(
            "6144 MiB, inflight {inflight}: opt16 batch max {a} ({} B/block), opt16p1 {b} ({} B/block)",
            vram_bytes(&frames(OPT16, a, inflight)) / a as u64,
            vram_bytes(&frames(OPT16P1, b, inflight)) / b as u64
        );
    }
}

#[test]
fn vram_counts_scratch_once_and_slots_per_inflight() {
    let frames = |batch, inflight| PipelineConfig {
        params: GpuParams { matching: LVL3, emit_frames: true, huffman: true },
        ..cfg(batch, inflight)
    };
    let one = vram_bytes(&frames(100, 1));
    let per_slot = vram_bytes(&frames(100, 2)) - one;
    assert_eq!(vram_bytes(&frames(100, 4)), one + 3 * per_slot);
    // A slot owns only its upload and staging buffers; data, frames and frame_len are shared.
    assert_eq!(per_slot, data_bytes(100) + StagingLayout::new(100, true, &LVL3).size);
    assert_eq!(one, scratch_bytes(100, &LVL3) + slot_bytes(100, true) + per_slot + k4_tables_bytes());
    // The parse path reads back the fixed-stride seqs instead of the frames.
    assert!(vram_bytes(&cfg(100, 2)) > vram_bytes(&frames(100, 2)));
    // The direct upload drops the shared `data` buffer only.
    assert_eq!(vram_bytes_with(&frames(100, 2), true), vram_bytes(&frames(100, 2)) - data_bytes(100));
    // K5 (Huffman literals) needs no buffers of its own.
    let raw_lits =
        PipelineConfig { params: GpuParams { huffman: false, ..frames(100, 2).params }, ..frames(100, 2) };
    assert_eq!(vram_bytes(&raw_lits), vram_bytes(&frames(100, 2)));
    // ~1.44 MiB of scratch per block, ~0.125 MiB per block per slot on the frame path.
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64 / 100.0;
    assert!((1.4..1.5).contains(&mib(scratch_bytes(100, &LVL3))), "{}", mib(scratch_bytes(100, &LVL3)));
    assert!((0.12..0.13).contains(&mib(per_slot)), "{}", mib(per_slot));
}

/// `vram_bytes` equals the bytes of every buffer a `Pipeline` actually creates (shared scratch
/// once), per preset: single-hash presets allocate one chain's head/pred; the optimal parse
/// adds its second candidate word, the 8 B-per-position trace (in `pred`), the larger `seqs`
/// and K3opt's prices and scratch.
#[test]
fn vram_matches_params() {
    let _gpu = crate::testing::gpu_test_slot();
    let ctx = crate::testing::gpu();
    for matching in [LVL3, RUNG1, LVL9SEG, OPT16, OPT14, OPT16P1] {
        for (emit_frames, batch, inflight) in [(true, 7, 1), (true, 16, 3), (false, 5, 2)] {
            let cfg =
                PipelineConfig { batch, inflight, params: GpuParams { matching, emit_frames, huffman: true } };
            let pipe = Pipeline::new(&ctx, &cfg).unwrap();
            assert_eq!(
                pipe.allocated_bytes(),
                vram_bytes_with(&cfg, ctx.direct_upload),
                "{matching:?} {emit_frames} b{batch} i{inflight}"
            );
        }
    }
    let scratch = |m: MatchParams| {
        vram_bytes(&PipelineConfig {
            batch: 10,
            inflight: 1,
            params: GpuParams { matching: m, emit_frames: true, huffman: true },
        })
    };
    // One chain instead of two: head and pred halve.
    assert_eq!(scratch(LVL3) - scratch(RUNG1), head_bytes(10, 1) + chain_pred_bytes(10, &RUNG1));
    // The optimal parse over lvl3 (two chains too): candidates, seqs (in `slots` staging
    // only on the parse path), K3opt's prices and scratch.
    let o = BufferSizes::new(10, &OPT16);
    let opt_extra = crate::sizing::best_bytes(10) + o.seqs - seqs_bytes_for(10, &LVL3) + o.opt_prices + o.opt_scratch + o.opt_sched;
    assert_eq!(scratch(OPT16) - scratch(LVL3), opt_extra);
    assert_eq!(scratch(OPT14), scratch(OPT16));
    // opt16p1 (M6): three more head tables and sparse pred words; K3opt's buffers (and the
    // drop pass, which has none) are the same.
    let p1_extra = pred_bytes_for(10, &OPT16P1) - pred_bytes_for(10, &OPT16)
        + head_bytes(10, 5)
        - head_bytes(10, 2);
    assert_eq!(scratch(OPT16P1) - scratch(OPT16), p1_extra);
}

/// M5 T5: the optimal parse (K1 Opt3 → K2opt → K3opt passes → K5 → K4; M6 B4: opt16p1 with
/// its sparse chains and drop pass) through the streaming pipeline in every upload/readback
/// mode, over partial batches and reuse, equals the CPU oracle's frames (which libzstd
/// decodes); the parse path too. `allocated_bytes` equals `vram_bytes` in every mode.
#[test]
fn opt_stream_every_mode_matches_cpu() {
    let _gpu = crate::testing::gpu_test_slot();
    let distinct = distinct_blocks();
    let blocks: Vec<&[u8]> = (0..50).map(|i| distinct[i % distinct.len()].as_slice()).collect();
    let modes = mode_contexts();
    for matching in [OPT14, OPT16, OPT16P1] {
        let params = GpuParams { matching, emit_frames: true, huffman: true };
        let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
        for (w, b) in want.iter().zip(&distinct) {
            assert_eq!(zstd::bulk::decompress(w, BLOCK_SIZE).unwrap(), *b, "libzstd decodes the oracle's frame");
        }
        for (name, ctx) in &modes {
            let pcfg = PipelineConfig { batch: 16, inflight: 3, params };
            let mut pipe = Pipeline::new(ctx, &pcfg).unwrap();
            assert_eq!(pipe.transfer_readback(), ctx.transfer.is_some(), "{name}");
            assert_eq!(pipe.allocated_bytes(), vram_bytes_with(&pcfg, ctx.direct_upload), "{name}");
            for _ in 0..2 {
                let mut sink = CollectFrames(vec![None; blocks.len()]);
                let stats = pipe.run_frames(&blocks, &mut sink).unwrap();
                assert_eq!(stats.batches, 4, "{name}");
                for (i, got) in sink.0.into_iter().enumerate() {
                    assert!(got.unwrap() == want[i % distinct.len()], "{name} {matching:?}: index {i}");
                }
                assert_frame_timers(ctx, &stats, name);
            }
        }
        // The parse path: the `seqs` readback at MAX_SEQS_OPT per block.
        let ctx = &modes[0].1;
        let pcfg = PipelineConfig { batch: 7, inflight: 2, params: GpuParams { emit_frames: false, ..params } };
        let mut pipe = Pipeline::new(ctx, &pcfg).unwrap();
        assert_eq!(pipe.allocated_bytes(), vram_bytes_with(&pcfg, ctx.direct_upload));
        let mut sink = Collect(vec![None; blocks.len()]);
        pipe.run(&blocks, &mut sink).unwrap();
        for (i, got) in sink.0.into_iter().enumerate() {
            assert!(got.unwrap() == compress_block(blocks[i], matching), "parse path {matching:?}: index {i}");
        }
    }
}

#[test]
fn stream_frames_match_cpu_every_index_once() {
    let _gpu = crate::testing::gpu_test_slot();
    let ctx = crate::testing::gpu();
    let distinct = distinct_blocks();
    // Huffman literals (cfg's default).
    let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, cfg(1, 1).params)).collect();
    // 1000 = 15 * 64 + 40: the last batch is partial and reuses a slot holding stale blocks.
    let blocks: Vec<&[u8]> = (0..1000).map(|i| distinct[i % distinct.len()].as_slice()).collect();

    let mut sink = CollectFrames(vec![None; blocks.len()]);
    let c = PipelineConfig { params: GpuParams { emit_frames: true, ..cfg(64, 3).params }, ..cfg(64, 3) };
    let stats = Pipeline::new(&ctx, &c).and_then(|mut p| p.run_frames(&blocks, &mut sink)).expect("run_frames");

    for (i, got) in sink.0.iter().enumerate() {
        let got = got.as_ref().unwrap_or_else(|| panic!("index {i} never delivered"));
        assert!(*got == want[i % distinct.len()], "index {i}: GPU frame != CPU frame");
    }
    assert_eq!(stats.batches, 16);
    assert_frame_timers(&ctx, &stats, "lvl3");
}

/// Contexts for every upload/readback mode the adapter supports, whatever the environment:
/// (copy upload, main-queue readback), (direct, main), (copy, transfer), (direct, transfer).
fn mode_contexts() -> Vec<(String, std::sync::Arc<GpuContext>)> {
    use crate::context::GpuOptions;
    let mut out = Vec::new();
    for transfer_queue in [false, true] {
        for direct in [false, true] {
            let opts = GpuOptions { direct_upload: Some(direct), transfer_queue, ..GpuOptions::from_env() };
            let ctx = crate::testing::gpu_with(opts);
            if ctx.direct_upload != direct || ctx.transfer.is_some() != transfer_queue {
                eprintln!("mode direct={direct} transfer={transfer_queue} unsupported here: skipped");
                continue;
            }
            out.push((format!("direct={direct} transfer={transfer_queue}"), ctx));
        }
    }
    out
}

/// Every upload/readback mode delivers the CPU's frames, over partial batches and reuse, for
/// the chain finder (lvl9seg) and the bucket-sorted finder with the segmented parse (lvl9s12seg).
#[test]
fn stream_frames_every_mode_matches_cpu() {
    let _gpu = crate::testing::gpu_test_slot();
    let distinct = distinct_blocks();
    let blocks: Vec<&[u8]> = (0..200).map(|i| distinct[i % distinct.len()].as_slice()).collect();
    let modes = mode_contexts();
    assert!(modes.iter().any(|(_, c)| c.transfer.is_none() && !c.direct_upload), "the fallback mode always exists");
    for matching in [LVL9SEG, LVL9S12SEG] {
        let params = GpuParams { matching, emit_frames: true, huffman: true };
        let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
        for (name, ctx) in &modes {
            let pcfg = PipelineConfig { batch: 23, inflight: 3, params };
            let mut pipe = Pipeline::new(ctx, &pcfg).unwrap();
            assert_eq!(pipe.transfer_readback(), ctx.transfer.is_some(), "{name}");
            assert_eq!(pipe.allocated_bytes(), vram_bytes_with(&pcfg, ctx.direct_upload), "{name}");
            for _ in 0..2 {
                let mut sink = CollectFrames(vec![None; blocks.len()]);
                pipe.run_frames(&blocks, &mut sink).unwrap();
                for (i, got) in sink.0.into_iter().enumerate() {
                    assert!(got.unwrap() == want[i % distinct.len()], "{name} {matching:?}: index {i}");
                }
            }
        }
    }
}

/// One transfer-readback pipeline per context (`GpuContext::transfer`): a second one errors
/// while the first is alive, and works once it is dropped. Pipelines that do not use the
/// transfer queue (the parse path) are not affected.
#[test]
fn second_transfer_pipeline_errors() {
    let _gpu = crate::testing::gpu_test_slot();
    let Some((name, ctx)) = mode_contexts().into_iter().find(|(_, c)| c.transfer.is_some()) else {
        eprintln!("no transfer queue on this adapter: skipped");
        return;
    };
    let distinct = distinct_blocks();
    let blocks: Vec<&[u8]> = distinct.iter().map(|b| b.as_slice()).collect();
    let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
    let pcfg = PipelineConfig { params, ..cfg(7, 2) };
    let mut first = Pipeline::new(&ctx, &pcfg).unwrap();
    assert!(first.transfer_readback(), "{name}");
    let e = Pipeline::new(&ctx, &pcfg).err().expect("a second transfer pipeline must error");
    assert!(e.to_string().contains("one Compressor (one frame Pipeline) at a time"), "{name}: {e}");
    assert!(matches!(crate::Error::from_anyhow(e), crate::Error::InvalidInput(_)), "{name}");
    // The parse path does not touch the transfer queue.
    let mut parse_pipe = Pipeline::new(&ctx, &cfg(7, 1)).unwrap();
    parse_pipe.run(&blocks, &mut Collect(vec![None; blocks.len()])).unwrap();
    drop(parse_pipe);
    // The refused attempt left the first pipeline intact.
    let mut sink = CollectFrames(vec![None; blocks.len()]);
    first.run_frames(&blocks, &mut sink).unwrap();
    for (i, got) in sink.0.into_iter().enumerate() {
        assert!(got.unwrap() == cpu_frame(blocks[i], params), "{name}: index {i}");
    }
    drop(first);
    let mut second = Pipeline::new(&ctx, &pcfg).expect("the queue is free again once the first pipeline dropped");
    assert!(second.transfer_readback());
    let mut sink = CollectFrames(vec![None; blocks.len()]);
    second.run_frames(&blocks, &mut sink).unwrap();
    assert!(sink.0.iter().all(|f| f.is_some()));
}

/// A delivery that fails mid-stream, with batches still in flight (and, where supported, on the
/// transfer readback), leaves the pipeline usable: the next run delivers every frame right.
#[test]
fn stream_frames_recovers_from_failed_delivery() {
    let _gpu = crate::testing::gpu_test_slot();
    let distinct = distinct_blocks();
    let blocks: Vec<&[u8]> = (0..300).map(|i| distinct[i % distinct.len()].as_slice()).collect();
    let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
    let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
    for (name, ctx) in &mode_contexts() {
        let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 16, inflight: 3, params }).unwrap();
        for fail_at in [0u32, 3] {
            pipe.fail_deliveries_after = Some(fail_at);
            let mut sink = CollectFrames(vec![None; blocks.len()]);
            let e = pipe.run_frames(&blocks, &mut sink).expect_err("delivery failure must surface");
            assert!(e.to_string().contains("injected"), "{name}: {e}");
            assert!(pipe.fail_deliveries_after.is_none());
            let mut sink = CollectFrames(vec![None; blocks.len()]);
            pipe.run_frames(&blocks, &mut sink).unwrap();
            for (i, got) in sink.0.into_iter().enumerate() {
                assert!(got.unwrap() == want[i % distinct.len()], "{name} fail_at {fail_at}: index {i}");
            }
        }
    }
}

#[test]
fn stream_frames_odd_batch_single_slot_and_reuse() {
    let _gpu = crate::testing::gpu_test_slot();
    // Raw literals (K5 writes Raw sections only): the huffman: false frame path stays covered
    // end to end.
    let ctx = crate::testing::gpu();
    let distinct = distinct_blocks();
    let blocks: Vec<&[u8]> = distinct.iter().map(|b| b.as_slice()).collect();
    let params = GpuParams { matching: LVL3, emit_frames: true, huffman: false };
    let frames_cfg = PipelineConfig { params, ..cfg(7, 1) };
    let mut pipe = Pipeline::new(&ctx, &frames_cfg).unwrap();
    // The frame path reads back through the transfer queue whenever the context has one.
    assert_eq!(pipe.transfer_readback(), ctx.transfer.is_some());
    for _ in 0..2 {
        let mut sink = CollectFrames(vec![None; blocks.len()]);
        let stats = pipe.run_frames(&blocks, &mut sink).unwrap();
        assert_eq!(stats.batches as usize, blocks.len().div_ceil(7));
        for (i, got) in sink.0.into_iter().enumerate() {
            assert!(got.unwrap() == cpu_frame(blocks[i], params), "index {i}");
        }
    }
    // The frame pipeline has no parse output, and the parse pipeline no frames.
    assert!(pipe.run(&blocks, &mut Collect(vec![None; blocks.len()])).is_err());
    let mut parse_pipe = Pipeline::new(&ctx, &cfg(7, 1)).unwrap();
    assert!(parse_pipe.run_frames(&blocks, &mut CollectFrames(vec![None; blocks.len()])).is_err());
}

/// Frames of one `stream_frames` run, keyed by block index; checks exactly-once and batch order.
#[derive(Default)]
struct Batches {
    frames: Vec<Option<Vec<u8>>>,
    /// `(first index, len)` per batch, in delivery order.
    order: Vec<(usize, usize)>,
}

impl Batches {
    fn take(&mut self, batch: &FrameBatch) {
        self.order.push((batch.first_index(), batch.len()));
        for (i, frame) in batch.frames() {
            if self.frames.len() <= i {
                self.frames.resize(i + 1, None);
            }
            assert!(self.frames[i].is_none(), "index {i} delivered twice");
            self.frames[i] = Some(frame.to_vec());
        }
    }

    /// Every index below `n` once, equal to `want(i)`; batches in submission order, contiguous.
    fn check(&self, n: usize, want: impl Fn(usize) -> Vec<u8>, what: &str) {
        assert_eq!(self.frames.len(), n, "{what}");
        for (i, f) in self.frames.iter().enumerate() {
            assert!(*f.as_ref().unwrap_or_else(|| panic!("{what}: index {i} never delivered")) == want(i), "{what}: index {i}");
        }
        let mut next = 0;
        for &(first, len) in &self.order {
            assert_eq!(first, next, "{what}: batches out of order: {:?}", self.order);
            next += len;
        }
    }
}

/// The zero-copy API in every upload/readback mode: the producer writes blocks straight into
/// the slots (block by block, and some partial batches as a flush timer would submit), the
/// sink hands each `FrameBatch` to a writer thread that releases it later, and slots are
/// recycled many times over (far more batches than slots). Frames match the CPU's, each index
/// once, batches in submission order; the pipeline then runs again.
#[test]
fn stream_frames_zero_copy_every_mode() {
    let _gpu = crate::testing::gpu_test_slot();
    let distinct = distinct_blocks();
    // M5 T5: the optimal parse too (its K3opt reads one word past each block: the slot's
    // trailing zero word after partial batches of stale blocks). M6 B4: opt16p1 (sparse
    // chains, drop pass) as well.
    for matching in [LVL9S12SEG, OPT14, OPT16P1] {
        zero_copy_every_mode(&distinct, GpuParams { matching, emit_frames: true, huffman: true });
    }
}

fn zero_copy_every_mode(distinct: &[Vec<u8>], params: GpuParams) {
    let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
    // Batch sizes per submission: full (16), and partial ones as a flush timer would send.
    let sizes = [16usize, 5, 16, 1, 16, 16, 9, 16, 16, 16, 3, 16];
    let total: usize = sizes.iter().sum();
    for (name, ctx) in &mode_contexts() {
        let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 16, inflight: 3, params }).unwrap();
        for round in 0..2 {
            let (tx, rx) = mpsc::channel::<FrameBatch>();
            let collected = std::thread::scope(|s| {
                // The writer: takes each batch, copies its frames, drops it (releasing the slot).
                let writer = s.spawn(move || {
                    let mut got = Batches::default();
                    for batch in rx {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        got.take(&batch);
                    }
                    got
                });
                let stats = pipe
                    .stream_frames(
                        move |batch| {
                            tx.send(batch).map_err(|_| anyhow!("writer gone"))?;
                            Ok(())
                        },
                        |stream| {
                            assert_eq!(stream.slot_capacity(), 16);
                            for &n in &sizes {
                                let first = stream.submitted_blocks();
                                let mut slot = stream.next_upload_slot()?;
                                for k in 0..n {
                                    // SAFETY: native wgpu-core backend (`blocks_mut`'s
                                    // contract; pinned by `upload_slot_bytes_are_initialized`).
                                    let block = unsafe { slot.block_mut(k) };
                                    block.copy_from_slice(&distinct[(first + k) % distinct.len()]);
                                }
                                assert_eq!(slot.submit(n)?, first);
                            }
                            Ok(())
                        },
                    )
                    .unwrap_or_else(|e| panic!("{name}: {e:#}"));
                assert_eq!(stats.batches as usize, sizes.len(), "{name}");
                writer.join().unwrap()
            });
            assert_eq!(collected.order.len(), sizes.len(), "{name}");
            collected.check(total, |i| want[i % distinct.len()].clone(), &format!("{name} round {round}"));
        }
    }
}

/// Lease lifetimes under stress, in every mode and with both completion waits (`poll_only`,
/// Metal's): the sink hands every `FrameBatch` to a holder thread that keeps up to
/// `inflight - 1` of them and drops them out of order after random delays (a batch whenever
/// none arrives for up to a millisecond: the producer may be waiting for it), and still holds
/// the last ones when the producer finishes and the completion thread closes the channel
/// (the stream's end waits for them, `wait_released`); some streams fail mid-way
/// (`fail_deliveries_after`), so `abandon` runs while batches are held. Every frame read
/// right before its batch drops equals the CPU's, every stream's errors are its own, and the
/// pipeline is reused and dropped right after a stream whose last batches another thread
/// released.
#[test]
fn stream_frames_leases_held_across_stream_end() {
    let _gpu = crate::testing::gpu_test_slot();
    let distinct = distinct_blocks();
    let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
    let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
    let blocks: Vec<&[u8]> = (0..120).map(|i| distinct[i % distinct.len()].as_slice()).collect();
    let mut seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64 | 1;
    eprintln!("seed {seed}");
    for (name, ctx) in &mode_contexts() {
        let poll_modes: &[bool] = if ctx.transfer.is_some() { &[false] } else { &[false, true] };
        for &poll_only in poll_modes {
            let inflight = 3;
            let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 4, inflight, params }).unwrap();
            pipe.poll_only = poll_only;
            for round in 0..6 {
                let fail_at = (round % 3 == 2).then_some(round as u32 * 2);
                pipe.fail_deliveries_after = fail_at;
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let mut rng = seed;
                let (tx, rx) = mpsc::channel::<FrameBatch>();
                let what = format!("{name} poll_only={poll_only} round {round}");
                let (r, seen) = std::thread::scope(|s| {
                    let want = &want;
                    let what = &what;
                    let holder = s.spawn(move || {
                        let mut next = || {
                            rng ^= rng << 13;
                            rng ^= rng >> 7;
                            rng ^= rng << 17;
                            rng
                        };
                        let mut held: Vec<FrameBatch> = Vec::new();
                        let mut seen = 0;
                        let release = |b: FrameBatch, seen: &mut usize| {
                            for (i, f) in b.frames() {
                                assert!(f == want[i % want.len()], "{what}: index {i}");
                            }
                            *seen += b.len();
                            drop(b);
                        };
                        // The producer reuses the slots in turn, so holding the oldest
                        // batch stalls it: release a random held batch whenever none
                        // arrives for a while, and whenever `inflight - 1` are held.
                        loop {
                            match rx.recv_timeout(std::time::Duration::from_micros(300 + next() % 700)) {
                                Ok(b) => held.push(b),
                                Err(mpsc::RecvTimeoutError::Timeout) if !held.is_empty() => {
                                    let k = next() as usize % held.len();
                                    release(held.swap_remove(k), &mut seen);
                                }
                                Err(mpsc::RecvTimeoutError::Timeout) => {}
                                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                            }
                            while held.len() >= inflight as usize {
                                let k = next() as usize % held.len();
                                std::thread::sleep(std::time::Duration::from_micros(next() % 500));
                                release(held.swap_remove(k), &mut seen);
                            }
                        }
                        // The channel closed: the producer is done and the stream waits for
                        // these.
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        while !held.is_empty() {
                            let k = next() as usize % held.len();
                            std::thread::sleep(std::time::Duration::from_micros(next() % 500));
                            release(held.swap_remove(k), &mut seen);
                        }
                        seen
                    });
                    let r = pipe.stream_frames(
                        move |b| tx.send(b).map_err(|_| anyhow!("holder gone")),
                        |stream| stream.upload_blocks(&blocks),
                    );
                    (r, holder.join().unwrap())
                });
                match fail_at {
                    Some(_) => {
                        let e = format!("{:#}", r.expect_err("injected delivery failure"));
                        assert!(e.contains("injected delivery failure"), "{what}: {e}");
                    }
                    None => {
                        let stats = r.unwrap_or_else(|e| panic!("{what}: {e:#}"));
                        assert_eq!(stats.batches as usize, blocks.len().div_ceil(4), "{what}");
                        assert_eq!(seen, blocks.len(), "{what}");
                    }
                }
            }
            // Dropped right after a stream whose last batches the holder thread released.
            drop(pipe);
        }
    }
}

/// A device lost mid-stream (here `Device::destroy`; on Metal, wgpu-hal's fence race in
/// `Device::wait` loses the device from a `poll`): wgpu-core then destroys every buffer, so
/// `abandon`'s unmap of the in-flight staging buffers raises "Buffer with 'pipeline.staging'
/// label has been destroyed" in the stream's error scope. The stream must report the failure
/// itself (the lost device), with that validation error only as context.
#[test]
fn stream_frames_reports_device_loss_not_the_cleanup_error() {
    let _gpu = crate::testing::gpu_test_slot();
    use crate::context::GpuOptions;
    let opts = GpuOptions { direct_upload: Some(false), transfer_queue: false, ..GpuOptions::from_env() };
    let ctx = crate::testing::gpu_with(opts);
    let distinct = distinct_blocks();
    let blocks: Vec<&[u8]> = (0..400).map(|i| distinct[i % distinct.len()].as_slice()).collect();
    let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
    let mut pipe = Pipeline::new(&ctx, &PipelineConfig { batch: 8, inflight: 3, params }).unwrap();
    let mut delivered = 0;
    let e = pipe
        .stream_frames(
            |_batch| {
                delivered += 1;
                if delivered == 2 {
                    ctx.device.destroy();
                }
                Ok(())
            },
            |stream| stream.upload_blocks(&blocks),
        )
        .expect_err("a stream on a lost device fails");
    let msg = format!("{e:#}");
    eprintln!("stream error: {msg}");
    assert!(!msg.starts_with("wgpu validation error"), "the cleanup's validation error masks the cause: {msg}");
}

/// Payloads of arbitrary size spanning several blocks (and one of 0 bytes), written from
/// several threads into disjoint regions of a slot that holds stale bytes, then padded: the
/// frames equal the CPU's for `chunk_file` of each payload, and `payload_real_lens` gives
/// `chunk_file`'s real lengths.
#[test]
fn stream_frames_multi_block_payloads() {
    let _gpu = crate::testing::gpu_test_slot();
    let ctx = crate::testing::gpu();
    let params = GpuParams { matching: LVL9S12SEG, emit_frames: true, huffman: true };
    let source: Vec<u8> = test_cases().into_iter().flat_map(|(_, bytes)| bytes).collect();
    let bs = BLOCK_SIZE;
    let sizes = [3 * bs + 100, 2 * bs, 1, 0, 5 * bs - 1, bs + 7];
    let mut at = 0;
    let payloads: Vec<&[u8]> = sizes
        .iter()
        .map(|&n| {
            let p = &source[at % (source.len() - 6 * bs)..][..n];
            at += n + 12345;
            p
        })
        .collect();
    let (mut want, mut want_lens) = (Vec::new(), Vec::new());
    for p in &payloads {
        let blocks = chunk_file(p);
        want_lens.extend(blocks.iter().map(|b| b.real_len));
        want.extend(blocks.iter().map(|b| cpu_frame(b.real(), params)));
        assert_eq!(payload_blocks(p.len()), blocks.len());
    }
    let lens: Vec<usize> = payloads.iter().flat_map(|p| payload_real_lens(p.len())).collect();
    assert_eq!(lens, want_lens, "per-block real lengths");
    let blocks_each: Vec<usize> = payloads.iter().map(|p| payload_blocks(p.len())).collect();
    let total: usize = blocks_each.iter().sum();

    let mut pipe = Pipeline::new(&ctx, &PipelineConfig { batch: 16, inflight: 2, params }).unwrap();
    // Dirty both slots.
    pipe.stream_frames(
        |_batch| Ok(()),
        |stream| {
            for _ in 0..2 {
                let mut slot = stream.next_upload_slot()?;
                slot.regions_mut(&[16])?[0].write_only().fill(0xAB);
                slot.submit(16)?;
            }
            Ok(())
        },
    )
    .unwrap();
    let mut got = Batches::default();
    pipe.stream_frames(
        |batch| {
            got.take(&batch);
            Ok(())
        },
        |stream| {
            for _ in 0..2 {
                let mut slot = stream.next_upload_slot()?;
                assert!(slot.regions_mut(&[slot.capacity() + 1]).is_err());
                let regions = slot.regions_mut(&blocks_each)?;
                std::thread::scope(|s| {
                    for (mut region, p) in regions.into_iter().zip(&payloads) {
                        s.spawn(move || {
                            region.write(0, p);
                            assert_eq!(region.pad(p.len()).unwrap(), payload_blocks(p.len()));
                            assert!(region.pad(region.len() + 1).is_err());
                        });
                    }
                });
                slot.submit(total)?;
            }
            Ok(())
        },
    )
    .unwrap();
    got.check(2 * total, |i| want[i % total].clone(), "payloads");
}

/// Pins what `UploadSlot::blocks_mut`'s safety relies on, in every upload/readback mode: the
/// mapping is real host memory whose bytes are initialized: a new slot reads back all zeros,
/// and a reused one reads back exactly what was written into it before (and zeros where
/// nothing was), not garbage. Recheck on every wgpu upgrade.
#[test]
fn upload_slot_bytes_are_initialized() {
    let _gpu = crate::testing::gpu_test_slot();
    if crate::context::GpuOptions::from_env().poison {
        eprintln!("skipped: poisoning fills the slots past the trailing zero word");
        return;
    }
    let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
    let bs = BLOCK_SIZE;
    for (name, ctx) in &mode_contexts() {
        let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 4, inflight: 2, params }).unwrap();
        pipe.stream_frames(
            |_batch| Ok(()),
            |stream| {
                // First use of both slots: zeros. Write block 0 only, submit it.
                for s in 0..2u8 {
                    let mut slot = stream.next_upload_slot()?;
                    // SAFETY: native wgpu-core backend; this test pins the contract.
                    let bytes = unsafe { slot.blocks_mut() };
                    assert!(bytes.iter().all(|&b| b == 0), "{name}: new slot {s} not zeroed");
                    bytes[..bs].fill(0x50 + s);
                    slot.submit(1)?;
                }
                // Reused: block 0 as written, the trailing zero word `submit` writes at
                // block 1's start, the rest still zero.
                for s in 0..2u8 {
                    let mut slot = stream.next_upload_slot()?;
                    // SAFETY: as above.
                    let bytes = unsafe { slot.blocks_mut() };
                    assert!(bytes[..bs].iter().all(|&b| b == 0x50 + s), "{name}: slot {s} lost its bytes");
                    assert!(bytes[bs..].iter().all(|&b| b == 0), "{name}: slot {s} holds garbage");
                    // Submitted, not dropped: a dropped slot would be handed out again.
                    slot.submit(1)?;
                }
                Ok(())
            },
        )
        .unwrap_or_else(|e| panic!("{name}: {e:#}"));
    }
}

/// A submission that fails after its slot was marked in flight aborts the stream: the
/// producer's later calls error instead of waiting for a slot that never comes free, even if
/// it swallowed the first error; `stream_frames` returns the error; the pipeline is reusable.
#[test]
fn failed_submit_is_sticky() {
    let _gpu = crate::testing::gpu_test_slot();
    let distinct = distinct_blocks();
    let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
    let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
    let blocks: Vec<&[u8]> = (0..64).map(|i| distinct[i % distinct.len()].as_slice()).collect();
    for (name, ctx) in &mode_contexts() {
        let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 16, inflight: 3, params }).unwrap();
        pipe.fail_submits_after = Some(1);
        let e = pipe
            .stream_frames(
                |_batch| Ok(()),
                |stream| {
                    stream.upload_blocks(&blocks[..16])?;
                    let e = stream.upload_blocks(&blocks[16..32]).expect_err("second submit fails");
                    assert!(e.to_string().contains("injected submit"), "{name}: {e:#}");
                    // Swallowed: every later call errors at once.
                    for _ in 0..4 {
                        let e = stream.next_upload_slot().err().expect("aborted stream hands out no slot");
                        assert!(e.to_string().contains("aborted"), "{name}: {e:#}");
                    }
                    Ok(())
                },
            )
            .expect_err("a swallowed submit failure still fails the stream");
        assert!(format!("{e:#}").contains("injected submit"), "{name}: {e:#}");
        let mut sink = CollectFrames(vec![None; blocks.len()]);
        pipe.run_frames(&blocks, &mut sink).unwrap();
        for (i, got) in sink.0.into_iter().enumerate() {
            assert!(got.unwrap() == want[i % distinct.len()], "{name}: index {i}");
        }
    }
}

/// `submit_with` tags reach `FrameBatch::tag` with their batch; `submit` tags 0.
#[test]
fn frame_batches_carry_their_tags() {
    let _gpu = crate::testing::gpu_test_slot();
    let ctx = crate::testing::gpu();
    let distinct = distinct_blocks();
    let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
    let mut pipe = Pipeline::new(&ctx, &PipelineConfig { batch: 8, inflight: 3, params }).unwrap();
    let mut seen = Vec::new();
    pipe.stream_frames(
        |batch| {
            seen.push((batch.first_index(), batch.len(), batch.tag()));
            Ok(())
        },
        |stream| {
            for (k, n) in [8usize, 3, 8, 5, 1, 8].into_iter().enumerate() {
                let mut slot = stream.next_upload_slot()?;
                for (b, mut region) in slot.regions_mut(&vec![1; n])?.into_iter().enumerate() {
                    region.write(0, &distinct[(k + b) % distinct.len()]);
                }
                if k == 4 {
                    slot.submit(n)?;
                } else {
                    slot.submit_with(n, 1000 + k as u64)?;
                }
            }
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(seen, [(0, 8, 1000), (8, 3, 1001), (11, 8, 1002), (19, 5, 1003), (24, 1, 0), (25, 8, 1005)]);
}

/// A batch kept alive holds its slot: its frames stay intact while later batches reuse the
/// other slots, and the stream only waits for it once it needs that slot again. Holding
/// `inflight - 1` batches does not stall. An upload slot dropped without a submit is handed
/// out again.
#[test]
fn stream_frames_held_batches_keep_their_bytes() {
    let _gpu = crate::testing::gpu_test_slot();
    let ctx = crate::testing::gpu();
    let distinct = distinct_blocks();
    let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
    let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
    let blocks: Vec<&[u8]> = (0..8 * 10).map(|i| distinct[i % distinct.len()].as_slice()).collect();
    let mut pipe = Pipeline::new(&ctx, &PipelineConfig { batch: 8, inflight: 3, params }).unwrap();
    let (tx, rx) = mpsc::channel::<FrameBatch>();
    let got = std::thread::scope(|s| {
        // Keeps the last two batches; releases the rest once the channel closes, which happens
        // when the pipeline drops `on_batch` (before `stream_frames` waits for the releases).
        let writer = s.spawn(move || {
            let mut held = std::collections::VecDeque::new();
            let mut got = Batches::default();
            let mut check = |(batch, snap): (FrameBatch, Vec<Vec<u8>>)| {
                for (k, f) in snap.iter().enumerate() {
                    assert!(batch.frame(k) == f.as_slice(), "held batch {} changed", batch.first_index());
                }
                got.take(&batch);
            };
            for batch in rx {
                // Snapshot at delivery; compare again once two more batches went by.
                let snap: Vec<Vec<u8>> = (0..batch.len()).map(|k| batch.frame(k).to_vec()).collect();
                held.push_back((batch, snap));
                if held.len() > 2 {
                    check(held.pop_front().unwrap());
                }
            }
            held.into_iter().for_each(&mut check);
            got
        });
        let stats = pipe
            .stream_frames(
                move |batch| tx.send(batch).map_err(|_| anyhow!("writer gone")),
                |stream| {
                    // Taken and dropped without a submit: the same slot comes back.
                    drop(stream.next_upload_slot()?);
                    stream.upload_blocks(&blocks)
                },
            )
            .unwrap();
        assert_eq!(stats.batches, 10);
        writer.join().unwrap()
    });
    got.check(blocks.len(), |i| want[i % distinct.len()].clone(), "held");
}

/// Errors mid-stream, from either side, surface from `stream_frames`, stop the other side and
/// leave the pipeline usable (every upload/readback mode): the sink failing on its third
/// batch; the producer failing after four submissions; a bad submit size; a panicking sink.
#[test]
fn stream_frames_errors_mid_stream() {
    let _gpu = crate::testing::gpu_test_slot();
    let distinct = distinct_blocks();
    let params = GpuParams { matching: LVL3, emit_frames: true, huffman: true };
    let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
    let blocks: Vec<&[u8]> = (0..200).map(|i| distinct[i % distinct.len()].as_slice()).collect();
    for (name, ctx) in &mode_contexts() {
        // Both completion waits (`poll_only`: Metal's), which `abandon` and the producer's
        // upload-map wait follow too.
        let poll_modes: &[bool] = if ctx.transfer.is_some() { &[false] } else { &[false, true] };
        for &poll_only in poll_modes {
            let name = &format!("{name} poll_only={poll_only}");
            let mut pipe = Pipeline::new(ctx, &PipelineConfig { batch: 16, inflight: 3, params }).unwrap();
            pipe.poll_only = poll_only;
            // The sink fails on its third batch: the producer, blocked or not, gets an error.
            let mut seen = 0;
            let e = pipe
                .stream_frames(
                    |_batch| {
                        seen += 1;
                        anyhow::ensure!(seen < 3, "sink failed");
                        Ok(())
                    },
                    |stream| stream.upload_blocks(&blocks),
                )
                .expect_err("sink error must surface");
            assert!(e.to_string().contains("sink failed"), "{name}: {e:#}");
            assert_eq!(seen, 3, "{name}: no delivery after the failing one");
            // The producer fails after four submissions; batches may still be in flight.
            let e = pipe
                .stream_frames(
                    |_batch| Ok(()),
                    |stream| {
                        stream.upload_blocks(&blocks[..64])?;
                        anyhow::bail!("producer failed")
                    },
                )
                .expect_err("producer error must surface");
            assert!(e.to_string().contains("producer failed"), "{name}: {e:#}");
            // Bad submit sizes are refused, and the slot stays usable.
            let e = pipe
                .stream_frames(
                    |_batch| Ok(()),
                    |stream| {
                        let slot = stream.next_upload_slot()?;
                        assert!(slot.capacity() == 16);
                        slot.submit(17)
                    }
                    .map(|_| ()),
                )
                .expect_err("17 > capacity");
            assert!(e.to_string().contains("not in 1..=16"), "{name}: {e:#}");
            let e = pipe
                .stream_frames(|_batch| Ok(()), |stream| stream.next_upload_slot()?.submit(0).map(|_| ()))
                .expect_err("0 blocks");
            assert!(e.to_string().contains("not in 1..=16"), "{name}: {e:#}");
            // A panic on either side propagates without leaving the other side waiting.
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pipe.stream_frames(|_batch| panic!("sink panicked"), |stream| stream.upload_blocks(&blocks))
            }));
            assert!(r.is_err(), "{name}: the sink's panic propagates");
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pipe.stream_frames(
                    |_batch| Ok(()),
                    |stream| {
                        stream.upload_blocks(&blocks[..48])?;
                        panic!("producer panicked")
                    },
                )
            }));
            assert!(r.is_err(), "{name}: the producer's panic propagates");
            // After all of that the same pipeline delivers every frame right.
            let mut sink = CollectFrames(vec![None; blocks.len()]);
            pipe.run_frames(&blocks, &mut sink).unwrap();
            for (i, got) in sink.0.into_iter().enumerate() {
                assert!(got.unwrap() == want[i % distinct.len()], "{name}: index {i}");
            }
        }
    }
}

/// Parallel delivery (`run_frames_par`): every index exactly once and right, whatever the
/// thread count (more threads than frames in a batch included), and actually from several
/// threads.
#[test]
fn run_frames_par_every_index_once() {
    let _gpu = crate::testing::gpu_test_slot();
    use std::sync::atomic::AtomicU32;
    struct Par {
        hits: Vec<AtomicU32>,
        frames: Vec<Mutex<Vec<u8>>>,
        threads: Mutex<std::collections::HashSet<std::thread::ThreadId>>,
    }
    impl ParFrameSink for Par {
        fn put(&self, index: usize, frame: &[u8]) {
            self.hits[index].fetch_add(1, Ordering::Relaxed);
            *self.frames[index].lock().unwrap() = frame.to_vec();
            self.threads.lock().unwrap().insert(std::thread::current().id());
        }
    }
    let ctx = crate::testing::gpu();
    let distinct = distinct_blocks();
    let params = GpuParams { matching: LVL9SEG, emit_frames: true, huffman: true };
    let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
    let blocks: Vec<&[u8]> = (0..150).map(|i| distinct[i % distinct.len()].as_slice()).collect();
    let mut pipe = Pipeline::new(&ctx, &PipelineConfig { batch: 32, inflight: 2, params }).unwrap();
    for threads in [1, 3, 8, 64] {
        let sink = Par {
            hits: (0..blocks.len()).map(|_| AtomicU32::new(0)).collect(),
            frames: (0..blocks.len()).map(|_| Mutex::new(Vec::new())).collect(),
            threads: Default::default(),
        };
        let stats = pipe.run_frames_par(&blocks, &sink, threads).unwrap();
        assert_eq!(stats.batches, 5);
        for i in 0..blocks.len() {
            assert_eq!(sink.hits[i].load(Ordering::Relaxed), 1, "threads {threads}: index {i}");
            assert!(*sink.frames[i].lock().unwrap() == want[i % distinct.len()], "threads {threads}: index {i}");
        }
        let used = sink.threads.lock().unwrap().len();
        assert!(if threads == 1 { used == 1 } else { used > 1 }, "threads {threads}: {used} delivery threads");
    }
}
