//! M5 T5: the optimal-parse presets (`opt14`, `opt16`) through the full GPU pipeline (K1 Opt3
//! chains → K2opt → the K3opt passes → K5 → K4), byte-identical to the CPU oracle
//! (`reference::compress_block` + `write_frame`) on synthetic blocks and on real corpus blocks, in
//! every upload/readback mode the adapter has. Run once more with `GZC_NO_SUBGROUPS=1` for the
//! subgroup-less kernels, and in a `block-16k` build for 16 KiB blocks.
use gzc_core::block::chunk_file;
use gzc_core::config::BLOCK_SIZE;
use gzc_core::frame::write_frame;
use gzc_core::params::{MatchParams, OPT14, OPT16};
use gzc_core::reference::compress_block;
use gzc_core::seq::BlockOutput;
use gzc_gpu::compressor::{GpuParams, Kernels, compress_batch, compress_frames, max_seqs};
use gzc_gpu::context::{GpuContext, GpuOptions};
use gzc_gpu::pipeline::{FrameSink, Pipeline, PipelineConfig, vram_bytes_with};

const OPT_PRESETS: [(&str, MatchParams); 2] = [("opt14", OPT14), ("opt16", OPT16)];

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
fn mode_contexts() -> Vec<(String, GpuContext)> {
    let mut out = Vec::new();
    for transfer_queue in [false, true] {
        for direct in [false, true] {
            let opts = GpuOptions { direct_upload: Some(direct), transfer_queue, ..GpuOptions::from_env() };
            let ctx = GpuContext::with_gpu_options(opts).expect("GPU required for gzc-gpu tests");
            if ctx.direct_upload != direct || ctx.transfer.is_some() != transfer_queue {
                eprintln!("mode direct={direct} transfer={transfer_queue} unsupported here: skipped");
                continue;
            }
            out.push((format!("direct={direct} transfer={transfer_queue} subgroups={}", ctx.subgroups), ctx));
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
        let mib = vram_bytes_with(&cfg, ctx.direct_upload).div_ceil(1 << 20);
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
    let blocks = synthetic_blocks();
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    for (name, m) in OPT_PRESETS {
        let want = oracle(&blocks, m);
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
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

/// Up to `n` corpus blocks sampled uniformly over the whole corpus (`GZC_CORPUS`, default
/// `data/corpus`, `.dds`/`.nif`): every k-th of all (file, block) pairs, files in sorted path
/// order, k = total blocks / n (as `tests/k3opt.rs`). `None` (after a message) when the corpus
/// directory does not exist.
fn corpus_blocks(n: usize) -> Option<Vec<Vec<u8>>> {
    use std::path::{Path, PathBuf};
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            let ext = p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
            if p.is_dir() {
                walk(&p, out);
            } else if matches!(ext.as_str(), "dds" | "nif") {
                out.push(p);
            }
        }
    }
    let root = std::env::var("GZC_CORPUS")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/corpus").to_string());
    if !Path::new(&root).is_dir() {
        eprintln!("corpus directory {root} not found (set GZC_CORPUS): skipping");
        return None;
    }
    let mut files = Vec::new();
    walk(Path::new(&root), &mut files);
    files.sort();
    let counts: Vec<usize> =
        files.iter().map(|f| (std::fs::metadata(f).unwrap().len() as usize).div_ceil(BLOCK_SIZE)).collect();
    let total: usize = counts.iter().sum();
    let step = (total / n.max(1)).max(1);
    let mut blocks = Vec::new();
    let mut first = 0usize;
    for (f, &c) in files.iter().zip(&counts) {
        if blocks.len() >= n {
            break;
        }
        let picked: Vec<usize> = (first.div_ceil(step) * step..first + c).step_by(step).map(|g| g - first).collect();
        if !picked.is_empty() {
            let chunks = chunk_file(&std::fs::read(f).unwrap());
            assert_eq!(chunks.len(), c, "{}", f.display());
            for i in picked {
                blocks.push(chunks[i].data.clone());
            }
        }
        first += c;
    }
    blocks.truncate(n);
    eprintln!("{} corpus blocks (every {step}th of {total}) from {} files", blocks.len(), files.len());
    Some(blocks)
}

/// Informal (reads the real corpus): `GZC_CORPUS_BLOCKS` (default 4000) uniform-stride corpus
/// blocks, opt14 and opt16, through the pipeline in every mode (batch 256, a partial last batch):
/// frames byte-identical to the oracle's.
/// `GZC_CORPUS=/path/to/data/corpus cargo test --release -p gzc-gpu --test opt_pipeline -- --ignored --nocapture`
#[test]
#[ignore]
fn opt_corpus_matches_oracle() {
    let n: usize = std::env::var("GZC_CORPUS_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(4000);
    let Some(blocks) = corpus_blocks(n) else { return };
    for (name, m) in OPT_PRESETS {
        let t = std::time::Instant::now();
        let want = oracle(&blocks, m);
        eprintln!("{name}: oracle {:.1}s", t.elapsed().as_secs_f64());
        check_modes(&blocks, m, name, &want, 256);
    }
}
