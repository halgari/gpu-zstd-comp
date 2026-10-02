//! A differential subset under memory poisoning (`gzc_gpu::poison`): before every batch, garbage
//! fills every scratch and output buffer, the input buffer past the batch's trailing zero word, the
//! padding past every buffer's logical end and the SMs' workgroup memory (whose zero-init is off).
//! Frames must still equal the CPU oracle byte for byte: no kernel may depend on memory it did not
//! write in the same batch. The full suite runs poisoned with `GZC_POISON=1`.
use gzc_core::block::chunk_file;
use gzc_core::frame::write_frame;
use gzc_core::params::{LVL3, LVL9S12SEG, LVL9SEG, MatchParams, OPT14, OPT16P1};
use gzc_core::reference::compress_block;
use gzc_gpu::compressor::{GpuParams, Kernels, compress_frames};
use gzc_gpu::context::{GpuContext, GpuOptions};
use gzc_gpu::pipeline::{FrameSink, Pipeline, PipelineConfig};

fn blocks() -> Vec<Vec<u8>> {
    gzc_core::synth::test_cases().into_iter().flat_map(|(_, bytes)| chunk_file(&bytes)).map(|b| b.data).collect()
}

fn presets() -> Vec<(&'static str, MatchParams)> {
    vec![
        ("lvl3", LVL3),
        ("lvl9seg", LVL9SEG),
        ("lvl9s12seg", LVL9S12SEG),
        ("opt14", OPT14),
        // M6 B4: sparse chains, gap3, top-4 and the drop pass.
        ("opt16p1", OPT16P1),
    ]
}

fn poisoned(opts: GpuOptions) -> std::sync::Arc<GpuContext> {
    let ctx = gzc_gpu::testing::gpu_with(GpuOptions { poison: true, ..opts });
    assert!(ctx.poisoning());
    ctx
}

fn cpu_frame(block: &[u8], params: GpuParams) -> Vec<u8> {
    write_frame(block, &compress_block(block, params.matching), params.frame_options())
}

/// The one-shot path (`compress_frames`), with and without subgroups, all blocks in one batch and
/// then in batches of 3 (partial batches leave stale blocks past the trailing word).
#[test]
fn poisoned_frames_match_cpu() {
    let _gpu = gzc_gpu::testing::gpu_test_slot();
    let blocks = blocks();
    for subgroups in [true, false] {
        let ctx = poisoned(GpuOptions { subgroups, ..GpuOptions::from_env() });
        for (name, matching) in presets() {
            let params = GpuParams { matching, emit_frames: true, huffman: true };
            let kernels = Kernels::new(&ctx, params).expect("Kernels::new");
            let want: Vec<Vec<u8>> = blocks.iter().map(|b| cpu_frame(b, params)).collect();
            let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
            let mut got = compress_frames(&ctx, &kernels, &refs).expect("compress_frames");
            for chunk in refs.chunks(3) {
                got.extend(compress_frames(&ctx, &kernels, chunk).expect("compress_frames"));
            }
            for (i, g) in got.iter().enumerate() {
                let k = i % blocks.len();
                assert!(*g == want[k], "{name} subgroups={subgroups}: block {k} (output {i}) differs from the CPU frame");
            }
        }
    }
}

/// Poisoned and with every invocation's timing skewed (`Emulation::skew`: pseudo-random stalls at
/// entry, after barriers and before subgroup operations), for races a slow, preempted GPU would
/// expose. A few blocks only: the stalls make the kernels much slower.
#[test]
fn poisoned_skewed_frames_match_cpu() {
    let _gpu = gzc_gpu::testing::gpu_test_slot();
    let blocks: Vec<Vec<u8>> = blocks().into_iter().step_by(4).collect();
    let emulate = gzc_gpu::Emulation { skew: true, ..gzc_gpu::Emulation::NONE };
    let env = GpuOptions::from_env();
    let ctx = poisoned(GpuOptions { emulate: emulate.or(env.emulate), ..env });
    for (name, matching) in [("lvl3", LVL3), ("lvl9seg", LVL9SEG), ("lvl9s12seg", LVL9S12SEG)] {
        let params = GpuParams { matching, emit_frames: true, huffman: true };
        let kernels = Kernels::new(&ctx, params).expect("Kernels::new");
        let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
        let got = compress_frames(&ctx, &kernels, &refs).expect("compress_frames");
        for (k, (g, b)) in got.iter().zip(&blocks).enumerate() {
            assert!(*g == cpu_frame(b, params), "{name} skewed: block {k} differs from the CPU frame");
        }
    }
}

struct CollectFrames(Vec<Option<Vec<u8>>>);

impl FrameSink for CollectFrames {
    fn put(&mut self, index: usize, frame: &[u8]) {
        assert!(self.0[index].is_none(), "index {index} delivered twice");
        self.0[index] = Some(frame.to_vec());
    }
}

/// The streaming pipeline in every upload/readback mode the adapter has, over partial batches and
/// slot reuse.
#[test]
fn poisoned_pipeline_every_mode_matches_cpu() {
    let _gpu = gzc_gpu::testing::gpu_test_slot();
    let distinct = blocks();
    let blocks: Vec<&[u8]> = (0..61).map(|i| distinct[i % distinct.len()].as_slice()).collect();
    for transfer_queue in [false, true] {
        for direct in [false, true] {
            let ctx = poisoned(GpuOptions { direct_upload: Some(direct), transfer_queue, ..GpuOptions::from_env() });
            if ctx.direct_upload() != direct || ctx.transfer_readback() != transfer_queue {
                eprintln!("mode direct={direct} transfer={transfer_queue} unsupported here: skipped");
                continue;
            }
            for (name, matching) in [("lvl3", LVL3), ("lvl9s12seg", LVL9S12SEG)] {
                let params = GpuParams { matching, emit_frames: true, huffman: true };
                let want: Vec<Vec<u8>> = distinct.iter().map(|b| cpu_frame(b, params)).collect();
                let mut pipe = Pipeline::new(&ctx, &PipelineConfig { batch: 13, inflight: 3, params }).unwrap();
                for run in 0..2 {
                    let mut sink = CollectFrames(vec![None; blocks.len()]);
                    pipe.run_frames(&blocks, &mut sink).unwrap();
                    for (i, got) in sink.0.into_iter().enumerate() {
                        let ok = got.unwrap() == want[i % distinct.len()];
                        assert!(ok, "{name} direct={direct} transfer={transfer_queue} run {run}: index {i}");
                    }
                }
            }
        }
    }
}
