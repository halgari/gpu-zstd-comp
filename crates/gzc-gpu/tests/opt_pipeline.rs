//! M5 T5: the optimal-parse presets (`opt14`, `opt16`, and since M6 B4 `opt16p1`) through the
//! full GPU pipeline (K1 Opt3 chains, plus opt16p1's three sparse chains → K2opt → the K3opt
//! passes, then opt16p1's drop pass → K5 → K4), byte-identical to the CPU oracle
//! (`reference::compress_block` + `write_frame`) on synthetic blocks and on real corpus blocks, in
//! every upload/readback mode the adapter has. Run once more with `GZC_NO_SUBGROUPS=1` for the
//! subgroup-less kernels.
use gzc_core::block::chunk_file;
use gzc_core::config::BLOCK_SIZE;
use gzc_core::frame::write_frame;
use gzc_core::params::{MatchParams, OPT14, OPT16, OPT16P1};
use gzc_core::reference::compress_block;
use gzc_core::seq::BlockOutput;
use gzc_gpu::testing::{GpuParams, Kernels, compress_batch, compress_frames, max_seqs};
use gzc_gpu::{GpuContext, GpuOptions};
use gzc_gpu::pipeline::{FrameSink, Pipeline, PipelineConfig, vram_bytes_with};

const OPT_PRESETS: [(&str, MatchParams); 3] = [("opt14", OPT14), ("opt16", OPT16), ("opt16p1", OPT16P1)];

fn synthetic_blocks() -> Vec<Vec<u8>> {
    gzc_core::synth::test_cases().into_iter().flat_map(|(_, bytes)| chunk_file(&bytes)).map(|b| b.data).collect()
}

/// `f` over `items` on every core, results in order.
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per = items.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = items.chunks(per).map(|c| s.spawn(|| c.iter().map(&f).collect::<Vec<R>>())).collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

/// The oracle's parse and frame of every block.
fn oracle(blocks: &[Vec<u8>], m: MatchParams) -> Vec<(BlockOutput, Vec<u8>)> {
    let opts = GpuParams { matching: m, emit_frames: true, huffman: true }.frame_options();
    par_map(blocks, |b| {
        let parse = compress_block(b, m);
        let frame = write_frame(b, &parse, opts);
        (parse, frame)
    })
}

/// Contexts for every upload/readback mode the adapter supports (copy / direct upload × main-queue
/// / transfer-queue readback), with the environment's other knobs (`GZC_NO_SUBGROUPS`).
fn mode_contexts() -> Vec<(String, std::sync::Arc<GpuContext>)> {
    let mut out = Vec::new();
    for transfer_queue in [false, true] {
        for direct in [false, true] {
            let opts = GpuOptions { direct_upload: Some(direct), transfer_queue, ..GpuOptions::from_env() };
            let ctx = gzc_gpu::testing::gpu_with(opts);
            if ctx.direct_upload() != direct || ctx.transfer_readback() != transfer_queue {
                eprintln!("mode direct={direct} transfer={transfer_queue} unsupported here: skipped");
                continue;
            }
            out.push((format!("direct={direct} transfer={transfer_queue} subgroups={}", ctx.subgroups()), ctx));
        }
    }
    out
}

struct CollectFrames(Vec<Option<Vec<u8>>>);

impl FrameSink for CollectFrames {
    fn put(&mut self, index: usize, frame: &[u8]) {
        assert!(self.0[index].is_none(), "index {index} delivered twice");
        self.0[index] = Some(frame.to_vec());
    }
}

/// Every mode's pipeline frames (two runs each, `batch` blocks per submission, 3 in flight)
/// against `want`; `allocated == vram_bytes` is checked by the pipeline's own unit tests.
fn check_modes(blocks: &[Vec<u8>], m: MatchParams, name: &str, want: &[(BlockOutput, Vec<u8>)], batch: u32) {
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let params = GpuParams { matching: m, emit_frames: true, huffman: true };
    for (mode, ctx) in &mode_contexts() {
        let cfg = PipelineConfig { batch, inflight: 3, params };
        let mut pipe = Pipeline::new(ctx, &cfg).expect("Pipeline::new");
        let mib = vram_bytes_with(&cfg, ctx.direct_upload()).div_ceil(1 << 20);
        for round in 0..2 {
            let mut sink = CollectFrames(vec![None; refs.len()]);
            pipe.run_frames(&refs, &mut sink).unwrap_or_else(|e| panic!("{name} {mode}: {e:#}"));
            let mut bytes = 0usize;
            for (i, got) in sink.0.into_iter().enumerate() {
                let got = got.unwrap_or_else(|| panic!("{name} {mode}: block {i} never delivered"));
                assert!(got == want[i].1, "{name} {mode} round {round}: block {i}: GPU frame != oracle frame");
                bytes += got.len();
            }
            if round == 0 {
                eprintln!("{name} {mode} b{batch} ({mib} MiB): {} blocks equal, {bytes} frame bytes", refs.len());
            }
        }
    }
}

/// Synthetic blocks: the parse (`compress_batch`) and the frames (`compress_frames`, then the
/// streaming pipeline in every mode) equal the oracle's; libzstd decodes every frame.
#[test]
fn opt_matches_oracle_synthetic() {
    let _gpu = gzc_gpu::testing::gpu_test_slot();
    let blocks = synthetic_blocks();
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    for (name, m) in OPT_PRESETS {
        let want = oracle(&blocks, m);
        let ctx = gzc_gpu::testing::gpu();
        let kp = Kernels::new(&ctx, GpuParams { matching: m, emit_frames: false, huffman: true }).unwrap();
        assert!(kp.is_opt());
        let parses = compress_batch(&ctx, &kp, &refs).unwrap();
        for (i, (got, (w, _))) in parses.iter().zip(&want).enumerate() {
            assert!(got == w, "{name}: block {i}: GPU parse != oracle");
            assert!(got.sequences.len() <= max_seqs(&m) as usize);
        }
        let kf = Kernels::new(&ctx, GpuParams { matching: m, emit_frames: true, huffman: true }).unwrap();
        let frames = compress_frames(&ctx, &kf, &refs).unwrap();
        for (i, (got, (_, w))) in frames.iter().zip(&want).enumerate() {
            assert!(got == w, "{name}: block {i}: GPU frame != oracle frame");
            assert_eq!(zstd::bulk::decompress(got, BLOCK_SIZE).unwrap(), blocks[i], "{name}: block {i}");
        }
        check_modes(&blocks, m, name, &want, 16);
    }
}

/// M6 B4: `gpu_supports` accepts the M6 options at every valid value (sparse chains of stride 4 or
/// 8), not just opt16p1's. A few such combinations through `compress_frames` on synthetic blocks:
/// frames equal the oracle's.
#[test]
fn option_variants_match_oracle_synthetic() {
    use gzc_core::params::{OptParams, PriorTables, Seed, SparseChain};
    use gzc_gpu::testing::gpu_supports;
    let _gpu = gzc_gpu::testing::gpu_test_slot();
    let blocks = synthetic_blocks();
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let p1 = OPT16P1.opt.unwrap();
    let o16 = OPT16.opt.unwrap();
    let variants = [
        // opt16's schedule and candidates with the longest drop and the tightest pruning.
        ("opt16 drop32 top1", MatchParams { opt: Some(OptParams { drop_max_len: 32, relax_lengths: Some(1), inner_gap: 3, ..o16 }), ..OPT16 }),
        // Two sparse chains at stride 8, a cheap pass, the M5 prior, drop 3.
        (
            "s8 x1 base-prior drop3",
            MatchParams {
                opt: Some(OptParams {
                    passes: 1,
                    seed: Seed::Prior,
                    prior: PriorTables::Base,
                    sparse_chains: [Some(SparseChain { width: 5, stride: 8, depth: 3 }), Some(SparseChain { width: 9, stride: 8, depth: 64 }), None],
                    drop_max_len: 3,
                    relax_lengths: Some(32),
                    ..p1
                }),
                ..OPT16P1
            },
        ),
    ];
    let ctx = gzc_gpu::testing::gpu();
    for (name, m) in variants {
        assert!(gpu_supports(&m), "{name}");
        let want = oracle(&blocks, m);
        let kf = Kernels::new(&ctx, GpuParams { matching: m, emit_frames: true, huffman: true }).unwrap();
        let frames = compress_frames(&ctx, &kf, &refs).unwrap();
        for (i, (got, (_, w))) in frames.iter().zip(&want).enumerate() {
            assert!(got == w, "{name}: block {i}: GPU frame != oracle frame");
        }
    }
}

/// Informal (reads the real corpus): `GZC_CORPUS_BLOCKS` (default 4000) uniform-stride corpus
/// blocks, opt14, opt16 and opt16p1, through the pipeline in every mode (batch 256, a partial last batch):
/// frames byte-identical to the oracle's.
/// `GZC_CORPUS=/path/to/data/corpus cargo test --release -p gzc-gpu --test opt_pipeline -- --ignored --nocapture`
#[test]
#[ignore]
fn opt_corpus_matches_oracle() {
    let _gpu = gzc_gpu::testing::gpu_test_slot();
    let n: usize = std::env::var("GZC_CORPUS_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(4000);
    let Some(blocks) = gzc_core::testdata::corpus_sample(n) else { return };
    for (name, m) in OPT_PRESETS {
        let t = std::time::Instant::now();
        let want = oracle(&blocks, m);
        eprintln!("{name}: oracle {:.1}s", t.elapsed().as_secs_f64());
        check_modes(&blocks, m, name, &want, 256);
    }
}

/// `gpu_supports` accepts the whole valid M6 option space (kernels take the options as injected
/// constants, not per-value code paths). This sweep backs that up: a fixed pseudo-random spread
/// of option combinations, each one byte-identical to the oracle end to end on the synthetic
/// blocks. `GZC_SWEEP_N` sets the number of combinations (default 16).
#[test]
fn option_sweep_matches_oracle_synthetic() {
    use gzc_core::params::{OptParams, PriorTables, Seed, SparseChain};
    use gzc_gpu::testing::gpu_supports;
    let _gpu = gzc_gpu::testing::gpu_test_slot();
    let blocks = synthetic_blocks();
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let n: usize = std::env::var("GZC_SWEEP_N").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = |k: u64| {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (x >> 33) % k
    };
    let ctx = gzc_gpu::testing::gpu();
    let base = OPT16P1.opt.unwrap();
    let mut tested = 0;
    while tested < n {
        let n_sparse = rnd(4) as usize;
        let mut sparse = [None; 3];
        for c in sparse.iter_mut().take(n_sparse) {
            *c = Some(SparseChain {
                width: 5 + rnd(8) as u32,
                stride: [4u32, 8][rnd(2) as usize],
                depth: [1u32, 3, 16, 64][rnd(4) as usize],
            });
        }
        let prior = [PriorTables::Base, PriorTables::Sparse][rnd(2) as usize];
        let seed = if prior == PriorTables::Sparse { Seed::Prior } else { [Seed::BlockInit, Seed::Prior][rnd(2) as usize] };
        let o = OptParams {
            level: [0u8, 2][rnd(2) as usize],
            target_length: [8u32, 16, 32][rnd(3) as usize],
            passes: rnd(4) as u8,
            seed,
            prior,
            sparse_chains: sparse,
            inner_gap: [8u8, 3][rnd(2) as usize],
            relax_lengths: [None, Some(1), Some(4), Some(32)][rnd(4) as usize],
            drop_max_len: [0u8, 3, 6, 32][rnd(4) as usize],
            ..base
        };
        let m = MatchParams { depth: [1u32, 8, 32, 64][rnd(4) as usize], opt: Some(o), ..OPT16P1 };
        if m.validate().is_err() || !gpu_supports(&m) {
            continue;
        }
        let want = oracle(&blocks, m);
        let kf = Kernels::new(&ctx, GpuParams { matching: m, emit_frames: true, huffman: true }).unwrap();
        let frames = compress_frames(&ctx, &kf, &refs).unwrap();
        for (i, (got, (_, w))) in frames.iter().zip(&want).enumerate() {
            assert!(got == w, "sweep {tested} {m:?}: block {i}: GPU frame != oracle frame");
        }
        tested += 1;
    }
}
