//! Differential tests: the GPU parse (K1→K2→K3) must equal `reference::compress_block` exactly,
//! and the GPU frames (K5 Huffman literals + K4) must equal `write_frame` on that parse byte for
//! byte, with Huffman literals (`FrameOptions::default()`) and without (`huffman: false`).
use gzc_core::block::chunk_file;
use gzc_core::config::BLOCK_SIZE;
use gzc_core::codes::{LL_DEFAULT_LOG, LL_DEFAULT_NORM, ML_BASE};
use gzc_core::frame::{FrameOptions, frame_header, write_frame, write_literals_raw};
use gzc_core::fse::{choose_table_log, cost_x256, normalize, write_ncount};
use gzc_core::huffman::HufTable;
use gzc_core::huffman::{HUF_MAX_BITS, MIN_HUF_LITERALS, build_table, compressed_section, table_description};
use gzc_core::lazy::cases::{LazyCase, lazy_test_cases, segment_test_cases};
use gzc_core::lazy::lazy_parse;
use gzc_core::params::{LVL3, LVL9, LVL9SEG, MatchParams, RUNG1, RUNG2};
use gzc_core::reference::{Match, chains, compress_block, find_best, match_len_capped};
use gzc_core::seq::{BlockOutput, INITIAL_REPS, Sequence, apply_off_base, off_base_for, reconstruct};
use gzc_core::seqenc::{SeqMode, StreamKind, StreamTable, histograms, write_sequences_section_auto};
use gzc_core::synth::{random, test_cases, text, zeros};
use gzc_gpu::compressor::{
    GpuParams, K3Mode, Kernels, best_from_blocks, compress_batch, compress_frames, frames_from_best, frames_from_parses,
    parses_from_best,
};
use gzc_gpu::context::GpuContext;
use gzc_gpu::pipeline::{FrameSink, Pipeline, PipelineConfig};

/// Every test case chunked into padded blocks, labelled "name[i]".
fn all_blocks() -> Vec<(String, Vec<u8>)> {
    test_cases()
        .into_iter()
        .flat_map(|(name, bytes)| {
            chunk_file(&bytes)
                .into_iter()
                .enumerate()
                .map(move |(i, b)| (format!("{name}[{i}]"), b.data))
        })
        .collect()
}

fn case(name: &str) -> Vec<u8> {
    test_cases().into_iter().find(|(n, _)| *n == name).unwrap().1
}

fn setup(matching: MatchParams) -> (GpuContext, Kernels) {
    let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
    let kernels = Kernels::new(&ctx, GpuParams { matching, emit_frames: false, huffman: false }).expect("Kernels::new");
    (ctx, kernels)
}

/// The presets the GPU implements, each checked by the differential tests below.
const GPU_PRESETS: [(&str, MatchParams); 5] =
    [("lvl3", LVL3), ("rung1", RUNG1), ("rung2", RUNG2), ("lvl9", LVL9), ("lvl9seg", LVL9SEG)];

/// LVL3 with a deeper chain walk.
const DEPTH4: MatchParams = MatchParams { depth: 4, ..LVL3 };

fn first_diff(got: &BlockOutput, want: &BlockOutput) -> String {
    let i = got.sequences.iter().zip(&want.sequences).position(|(a, b)| a != b);
    match i {
        Some(i) => format!("first differing sequence {i}: gpu {:?} cpu {:?}", got.sequences[i], want.sequences[i]),
        None if got.sequences.len() != want.sequences.len() => format!(
            "sequence count gpu {} cpu {} (common prefix equal)",
            got.sequences.len(),
            want.sequences.len()
        ),
        None => {
            let j = got.literals.iter().zip(&want.literals).position(|(a, b)| a != b);
            format!(
                "sequences equal; literals len gpu {} cpu {}, first differing literal {j:?}",
                got.literals.len(),
                want.literals.len()
            )
        }
    }
}

/// Runs one batch and compares every block against the CPU reference.
fn check_batch(ctx: &GpuContext, kernels: &Kernels, blocks: &[(String, Vec<u8>)], params: MatchParams) -> Vec<BlockOutput> {
    assert_eq!(kernels.matching(), params, "kernels built for other params");
    let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
    let outs = compress_batch(ctx, kernels, &refs).expect("compress_batch");
    assert_eq!(outs.len(), blocks.len());
    for (i, ((name, block), got)) in blocks.iter().zip(&outs).enumerate() {
        let want = compress_block(block, params);
        if *got != want {
            panic!("batch index {i} ({name}): GPU != reference; {}", first_diff(got, &want));
        }
    }
    outs
}

#[test]
fn gpu_matches_reference() {
    for (name, params) in GPU_PRESETS {
        eprintln!("preset {name}");
        let (ctx, kernels) = setup(params);
        // All blocks in one batch (every block has neighbours on both sides)...
        check_batch(&ctx, &kernels, &all_blocks(), params);
        // ...and each block alone, so its end is followed only by the trailing zero word.
        for b in all_blocks() {
            check_batch(&ctx, &kernels, std::slice::from_ref(&b), params);
        }
    }
}

#[test]
fn gpu_matches_reference_depth4() {
    let (ctx, kernels) = setup(DEPTH4);
    let blocks: Vec<_> = all_blocks().into_iter().step_by(2).collect();
    check_batch(&ctx, &kernels, &blocks, DEPTH4);
}

/// Depth 4, where search_cap changes the choice: at 2000 the long chain offers q=1500
/// (80-byte match) then q=0 (200 bytes). Both cap to 64, so the tie goes to the closer q=1500,
/// and the parse extends that match to its full 80 bytes.
#[test]
fn capped_search_prefers_closer_candidate() {
    let (ctx, kernels) = setup(DEPTH4);
    let mut block = random(20, BLOCK_SIZE);
    let s = random(21, 200);
    block[..200].copy_from_slice(&s);
    block[1500..1580].copy_from_slice(&s[..80]);
    block[2000..2200].copy_from_slice(&s);
    let blocks = [("capped".to_string(), block)];
    let out = check_batch(&ctx, &kernels, &blocks, DEPTH4);
    assert!(
        out[0].sequences.iter().any(|s| s.match_len == 80 && s.off_base == 500 + 3),
        "expected the extended capped match (offset 500, len 80): {:?}",
        out[0].sequences
    );
}

/// Blocks whose last bytes matter, each followed by a block that continues its pattern (an
/// unbounded match would run on into it) and by one that breaks it (a 4-byte compare reading
/// past the end would see a mismatch).
#[test]
fn block_ends_with_adjacent_neighbours() {
    let period3 = case("period3");
    // period3 continued across the boundary: its phase at BLOCK_SIZE.
    let mut period3_cont = period3.clone();
    period3_cont.rotate_left(BLOCK_SIZE % 3);
    let exact = case("exact_block");
    let named = |n: &str, v: &Vec<u8>| (n.to_string(), v.clone());
    let blocks = vec![
        named("zeros", &zeros(BLOCK_SIZE)),
        named("zeros", &zeros(BLOCK_SIZE)),
        named("random", &random(9, BLOCK_SIZE)),
        named("period3", &period3),
        named("period3_cont", &period3_cont),
        named("period3", &period3),
        named("zeros", &zeros(BLOCK_SIZE)),
        named("exact_block", &exact),
        named("exact_block", &exact),
        named("text", &text(10, BLOCK_SIZE)),
        named("period3", &period3),
        named("random", &random(11, BLOCK_SIZE)),
        named("zeros", &zeros(BLOCK_SIZE)),
    ];
    for (name, params) in GPU_PRESETS {
        eprintln!("preset {name}");
        let (ctx, kernels) = setup(params);
        check_batch(&ctx, &kernels, &blocks, params);
    }
}

#[test]
fn gpu_frames_roundtrip() {
    for (preset, params) in GPU_PRESETS {
        let (ctx, kernels) = setup(params);
        let blocks = all_blocks();
        let outs = check_batch(&ctx, &kernels, &blocks, params);
        for ((name, block), out) in blocks.iter().zip(&outs) {
            let frame = write_frame(block, out, FrameOptions::default());
            let back =
                zstd::bulk::decompress(&frame, BLOCK_SIZE).unwrap_or_else(|e| panic!("{preset} {name}: libzstd: {e}"));
            assert!(back == *block, "{preset} {name}: libzstd output differs from input block");
        }
    }
}

#[test]
fn batch_of_300_mixed() {
    for (name, params) in GPU_PRESETS {
        eprintln!("preset {name}");
        let (ctx, kernels) = setup(params);
        let blocks: Vec<_> = all_blocks().into_iter().cycle().take(300).collect();
        check_batch(&ctx, &kernels, &blocks, params);
    }
}

/// Kernels for different match params coexist in one process, and unsupported or invalid
/// params are errors, not panics.
#[test]
fn kernels_per_params_coexist_and_reject_unsupported() {
    let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
    let gp = |matching| GpuParams { matching, emit_frames: false, huffman: false };
    let lvl3 = Kernels::new(&ctx, gp(LVL3)).unwrap();
    let depth4 = Kernels::new(&ctx, gp(DEPTH4)).unwrap();
    let rung1 = Kernels::new(&ctx, gp(RUNG1)).unwrap();
    let rung2 = Kernels::new(&ctx, gp(RUNG2)).unwrap();
    let lvl9 = Kernels::new(&ctx, gp(LVL9)).unwrap();
    let blocks = [("text".to_string(), text(12, BLOCK_SIZE))];
    check_batch(&ctx, &lvl3, &blocks, LVL3);
    check_batch(&ctx, &depth4, &blocks, DEPTH4);
    check_batch(&ctx, &rung1, &blocks, RUNG1);
    check_batch(&ctx, &lvl9, &blocks, LVL9);
    check_batch(&ctx, &rung2, &blocks, RUNG2);
    check_batch(&ctx, &lvl3, &blocks, LVL3);
    let e = Kernels::new(&ctx, gp(MatchParams { lazy: 3, ..LVL9 })).err().expect("lazy 3 rejected");
    assert!(e.to_string().contains("lazy 3"), "{e}");
    let e = Kernels::new(&ctx, gp(MatchParams { depth: 0, ..LVL3 })).err().expect("depth 0 rejected");
    assert!(e.to_string().contains("depth"), "{e}");
}

#[test]
fn empty_batch_is_empty() {
    let (ctx, kernels) = setup(LVL3);
    assert!(compress_batch(&ctx, &kernels, &[]).unwrap().is_empty());
}

// ---- K4 (+ K5): complete frames (literals, sequence entropy, block assembly) ----

/// The CPU frame options of the raw-literals GPU path (`GpuParams { huffman: false }`).
const NO_HUFFMAN: FrameOptions = FrameOptions { checksum: false, huffman: false };

/// LVL3 frame kernels; `huffman` adds K5 (Huffman literals), matching `FrameOptions::default()`.
fn setup_frames(huffman: bool) -> (GpuContext, Kernels) {
    setup_frames_for(LVL3, huffman)
}

/// `setup_frames` for any match params.
fn setup_frames_for(matching: MatchParams, huffman: bool) -> (GpuContext, Kernels) {
    let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
    let kernels = Kernels::new(&ctx, GpuParams { matching, emit_frames: true, huffman }).expect("Kernels::new");
    assert_eq!(kernels.frame_options().huffman, huffman);
    (ctx, kernels)
}

/// Small deterministic PRNG.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// Random bytes with copies of earlier stretches: a wide spread of literal lengths, match lengths
/// and offsets, so the LL/OF/ML streams mix predefined, computed and RLE modes across shapes.
fn copies_block(seed: u64, max_lit: u32, max_match: u32, near: u32) -> Vec<u8> {
    let mut r = Lcg(seed);
    let mut v: Vec<u8> = (0..16).map(|_| r.next() as u8).collect();
    while v.len() < BLOCK_SIZE {
        for _ in 0..r.below(max_lit + 1) {
            v.push(r.next() as u8);
        }
        let len = 5 + r.below(max_match) as usize;
        let dist = 1 + r.below(near.min(v.len() as u32)) as usize;
        let start = v.len() - dist;
        for k in 0..len {
            v.push(v[start + k]);
        }
    }
    v.truncate(BLOCK_SIZE);
    v
}

/// Literal runs of 64..=120 fresh bytes, each followed by a copy of its own start: every sequence
/// has LL code 25, offset = run length (off_base 67..=123, OF code 6; never a repeat offset, as
/// each run length differs from the previous three) and match length 20: all three streams are RLE.
fn rle_codes_block(seed: u64) -> Vec<u8> {
    let mut r = Lcg(seed);
    let mut v: Vec<u8> = Vec::new();
    let mut recent = [0u32; 3];
    let mut avoid = None;
    // Stop while a whole unit (<= 120 literals + 20) still fits; fresh literals fill the rest.
    while v.len() + 140 <= BLOCK_SIZE {
        let mut n = 64 + r.below(57);
        while recent.contains(&n) {
            n = 64 + r.below(57);
        }
        recent = [n, recent[0], recent[1]];
        let start = v.len();
        // The run's last byte differs from the byte before the run, so the copy cannot start early.
        let before = start.checked_sub(1).map(|i| v[i]);
        for i in 0..n {
            let mut b = r.next() as u8;
            while (i == 0 && Some(b) == avoid) || (i == n - 1 && Some(b) == before) {
                b = r.next() as u8;
            }
            v.push(b);
        }
        let len = 20;
        for k in 0..len {
            v.push(v[start + k]);
        }
        avoid = Some(v[start + len]);
    }
    while v.len() < BLOCK_SIZE {
        let mut b = r.next() as u8;
        while Some(b) == avoid {
            b = r.next() as u8;
        }
        avoid = None;
        v.push(b);
    }
    v
}

/// 50 short matches, one match longer than 2^16 at 128K (ML code 52; 49 at 16K), 50 short matches:
/// the ML histogram has a gap of more than 24 unused codes, so the computed ML table's NCount
/// takes `write_ncount`'s 24-zero-run branch (checked by `zero_run_block_exercises_long_ncount_zero_runs`).
fn zero_run_block() -> Vec<u8> {
    let mut r = Lcg(0x24);
    let pattern = [0x50u8, 0x51, 0x52, 0x53, 0x54];
    // Separators are all distinct (0x80.., 0xB8..), so each match is exactly one 5-byte pattern.
    let short = |v: &mut Vec<u8>, sep: u8| {
        for i in 0..50u8 {
            v.push(sep + i);
            v.extend_from_slice(&pattern);
        }
    };
    let mut v = pattern.to_vec();
    short(&mut v, 0x80);
    v.push(0x77);
    v.extend(std::iter::repeat_n(0u8, BLOCK_SIZE - 2 * 50 * 6 - 64));
    v.push(0x78);
    v.extend_from_slice(&pattern);
    short(&mut v, 0xB8);
    while v.len() < BLOCK_SIZE {
        v.push(0xC0 + r.below(64) as u8);
    }
    v.truncate(BLOCK_SIZE);
    v
}

/// Every synthetic block plus the stress blocks above.
fn frame_blocks() -> Vec<(String, Vec<u8>)> {
    let mut blocks = all_blocks();
    blocks.push(("zero_run".to_string(), zero_run_block()));
    blocks.push(("all_rle".to_string(), rle_codes_block(8)));
    let shapes = [(0, 3, 8), (2, 40, 64), (40, 200, 4096), (300, 3000, 1 << 20), (3, 12, 1 << 20), (1000, 10, 200)];
    for (i, &(lit, mat, near)) in shapes.iter().enumerate() {
        blocks.push((format!("copies{i}"), copies_block(100 + i as u64, lit, mat, near)));
    }
    blocks
}

fn first_byte_diff(got: &[u8], want: &[u8]) -> String {
    let i = got.iter().zip(want).position(|(a, b)| a != b);
    format!("len gpu {} cpu {}, first differing byte {i:?}", got.len(), want.len())
}

/// GPU frames for `blocks` must equal `write_frame` on the CPU reference parse (with the
/// kernels' match params and frame options), and decode.
fn check_frames(ctx: &GpuContext, kernels: &Kernels, blocks: &[(String, Vec<u8>)]) {
    let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
    let frames = compress_frames(ctx, kernels, &refs).expect("compress_frames");
    assert_eq!(frames.len(), blocks.len());
    for (i, ((name, block), got)) in blocks.iter().zip(&frames).enumerate() {
        let want = write_frame(block, &compress_block(block, kernels.matching()), kernels.frame_options());
        assert!(*got == want, "batch index {i} ({name}): GPU frame != CPU frame; {}", first_byte_diff(got, &want));
        let back = zstd::bulk::decompress(got, BLOCK_SIZE).unwrap_or_else(|e| panic!("{name}: libzstd: {e}"));
        assert!(back == *block, "{name}: libzstd output differs from input block");
    }
}

#[test]
fn zero_run_block_exercises_long_ncount_zero_runs() {
    let block = zero_run_block();
    let out = compress_block(&block, LVL3);
    let h = histograms(&out.sequences);
    let used: Vec<usize> = (0..h[2].len()).filter(|&c| h[2][c] > 0).collect();
    assert!(used.windows(2).any(|w| w[1] - w[0] > 24), "ML codes {used:?} have no gap of 24");
    let ml = StreamTable::choose(StreamKind::MatchLength, &h[2], out.sequences.len());
    assert_eq!(ml.mode, SeqMode::Compressed, "ML table must be computed");
}

/// The frame set exercises every block type and every per-stream mode on the CPU side, so byte
/// equality in `gpu_frames_identical_no_huffman` covers each GPU path.
#[test]
fn frame_blocks_cover_every_mode() {
    let mut block_types = [0usize; 3];
    let mut modes = [[0usize; 3]; 3]; // [LL, OF, ML][predefined, RLE, computed]
    for (_, block) in frame_blocks() {
        let frame = write_frame(&block, &compress_block(&block, LVL3), NO_HUFFMAN);
        let h = frame_header(NO_HUFFMAN).len();
        let bt = ((frame[h] >> 1) & 3) as usize;
        block_types[bt] += 1;
        if bt != 2 {
            continue;
        }
        let c = &frame[h + 3..];
        let (lit_hdr, n_lit) = match (c[0] >> 2) & 3 {
            0 | 2 => (1, (c[0] >> 3) as usize),
            1 => (2, (c[0] >> 4) as usize | (c[1] as usize) << 4),
            _ => (3, (c[0] >> 4) as usize | (c[1] as usize) << 4 | (c[2] as usize) << 12),
        };
        let s = &c[lit_hdr + n_lit..];
        let nb_hdr = match s[0] {
            0 => continue,
            1..128 => 1,
            255 => 3,
            _ => 2,
        };
        let m = s[nb_hdr];
        for (k, shift) in [6, 4, 2].into_iter().enumerate() {
            modes[k][((m >> shift) & 3) as usize] += 1;
        }
    }
    assert!(block_types.iter().all(|&n| n > 0), "block types raw/RLE/compressed: {block_types:?}");
    for (k, m) in modes.iter().enumerate() {
        assert!(m.iter().all(|&n| n > 0), "stream {k}: predefined/RLE/computed counts {m:?}");
    }
}

#[test]
fn gpu_frames_identical_no_huffman() {
    for (name, params) in GPU_PRESETS {
        eprintln!("preset {name}");
        let (ctx, kernels) = setup_frames_for(params, false);
        let blocks = frame_blocks();
        // All blocks in one batch, then each block alone.
        check_frames(&ctx, &kernels, &blocks);
        for b in &blocks {
            check_frames(&ctx, &kernels, std::slice::from_ref(b));
        }
    }
}

#[test]
fn gpu_frames_identical_huffman() {
    for (name, params) in GPU_PRESETS {
        eprintln!("preset {name}");
        let (ctx, kernels) = setup_frames_for(params, true);
        let blocks = frame_blocks();
        // All blocks in one batch, then each block alone.
        check_frames(&ctx, &kernels, &blocks);
        for b in &blocks {
            check_frames(&ctx, &kernels, std::slice::from_ref(b));
        }
    }
}

#[test]
fn gpu_frames_batch_of_300_mixed() {
    for (name, params) in GPU_PRESETS {
        eprintln!("preset {name}");
        let (ctx, kernels) = setup_frames_for(params, true);
        let blocks: Vec<_> = frame_blocks().into_iter().cycle().take(300).collect();
        check_frames(&ctx, &kernels, &blocks);
    }
}

/// With subgroups off (`GpuContext::with_subgroups(false)`), K1 builds only its fallback kernel
/// (`ChainsKernel::with_options` only attempts the subgroup kernel when `ctx.subgroups`) and
/// `compressor::k3_mode` (which also checks `ctx.subgroups`) picks the sequential K3 instead of
/// the cooperative one — the code path a GPU without subgroup support runs everywhere. Runs the
/// full frame pipeline for every preset on a few synthetic blocks, so plain `cargo test` covers
/// fallback K1 and sequential K3 together, without needing such hardware.
#[test]
fn gpu_frames_identical_without_subgroups() {
    let blocks: Vec<_> = frame_blocks().into_iter().step_by(4).collect();
    assert!(blocks.len() >= 3, "expected a handful of synthetic blocks, got {}", blocks.len());
    for (name, params) in GPU_PRESETS {
        eprintln!("preset {name}");
        let ctx = GpuContext::with_subgroups(false).expect("GPU required for gzc-gpu tests");
        assert!(!ctx.subgroups, "with_subgroups(false) must disable subgroups");
        let kernels =
            Kernels::new(&ctx, GpuParams { matching: params, emit_frames: true, huffman: true }).expect("Kernels::new");
        assert_eq!(kernels.k3_mode(), K3Mode::Seq, "{name}: with_subgroups(false) must force the sequential K3");
        check_frames(&ctx, &kernels, &blocks);
    }
}

#[test]
fn compress_frames_needs_emit_frames() {
    let (ctx, kernels) = setup(LVL3);
    let block = zeros(BLOCK_SIZE);
    assert!(compress_frames(&ctx, &kernels, &[&block]).is_err());
    assert!(frames_from_parses(&ctx, &kernels, &[&block], &[BlockOutput::default()]).is_err());
}

// ---- K4 alone on scripted parses: exact ties and boundaries the match finder rarely produces ----

/// (block, parse) from (lit_len, offset, match_len) steps, literals drawn from `seed`; the rest of
/// the block is literals.
fn scripted(script: &[(u32, u32, u32)], seed: u64) -> (Vec<u8>, BlockOutput) {
    let mut r = Lcg(seed);
    scripted_with(script, || r.next() as u8)
}

/// `scripted` with literal bytes from `next_lit`.
fn scripted_with(script: &[(u32, u32, u32)], mut next_lit: impl FnMut() -> u8) -> (Vec<u8>, BlockOutput) {
    let mut out = BlockOutput::default();
    let mut reps = INITIAL_REPS;
    let mut data: Vec<u8> = Vec::new();
    let mut lit = |out: &mut BlockOutput, data: &mut Vec<u8>| {
        let b = next_lit();
        out.literals.push(b);
        data.push(b);
    };
    for &(ll, off, ml) in script {
        for _ in 0..ll {
            lit(&mut out, &mut data);
        }
        let ob = off_base_for(off, ll, &reps);
        apply_off_base(&mut reps, ob, ll);
        let start = data.len() - off as usize;
        for k in 0..ml as usize {
            data.push(data[start + k]);
        }
        out.sequences.push(Sequence { lit_len: ll, match_len: ml, off_base: ob });
    }
    while data.len() < BLOCK_SIZE {
        lit(&mut out, &mut data);
    }
    assert_eq!(data.len(), BLOCK_SIZE, "script overran the block");
    (data, out)
}

/// Prefix of cases whose block deliberately differs from its parse's reconstruction (see
/// `literal_cases`); their frames must decode to the reconstruction.
const MISMATCHED: &str = "mismatched_";

/// GPU frames for scripted (block, parse) pairs must equal `write_frame` on the same parse (with
/// the kernels' frame options), and decode.
fn check_scripted(ctx: &GpuContext, kernels: &Kernels, cases: &[(String, Vec<u8>, BlockOutput)]) {
    let blocks: Vec<&[u8]> = cases.iter().map(|c| c.1.as_slice()).collect();
    let parses: Vec<BlockOutput> = cases.iter().map(|c| c.2.clone()).collect();
    let frames = frames_from_parses(ctx, kernels, &blocks, &parses).expect("frames_from_parses");
    assert_eq!(frames.len(), cases.len());
    for ((name, block, parse), got) in cases.iter().zip(&frames) {
        let content = reconstruct(parse).unwrap();
        assert_eq!(content == *block, !name.starts_with(MISMATCHED), "{name}: bad script");
        let want = write_frame(block, parse, kernels.frame_options());
        assert!(*got == want, "{name}: GPU frame != CPU frame; {}", first_byte_diff(got, &want));
        let back = zstd::bulk::decompress(got, BLOCK_SIZE).unwrap_or_else(|e| panic!("{name}: libzstd: {e}"));
        assert!(back == content, "{name}: libzstd output differs from the parse's content");
    }
}

/// Random short scripts over small code palettes: frequent ties in the normalization and in the
/// predefined-vs-computed cost, one- and two-sequence streams, RLE streams, repeat offsets.
#[test]
fn k4_random_scripts_match_cpu() {
    let (ctx, kernels) = setup_frames(false);
    let mut r = Lcg(0x4b4);
    let mut cases = Vec::new();
    for i in 0..400 {
        let n_seq = [1, 2, 3, 4, 5, 8, 13, 30, 60, 200][r.below(10) as usize];
        let pick = |r: &mut Lcg, palette: &[u32]| palette[r.below(palette.len() as u32) as usize];
        let size = |r: &mut Lcg| 1 + r.below(4) as usize;
        let lls: Vec<u32> = (0..size(&mut r)).map(|_| [r.below(4), r.below(20), r.below(300)][r.below(3) as usize]).collect();
        let mls: Vec<u32> = (0..size(&mut r)).map(|_| 3 + [r.below(4), r.below(40), r.below(500)][r.below(3) as usize]).collect();
        let offs: Vec<u32> = (0..size(&mut r)).map(|_| 1 + [r.below(8), r.below(64), r.below(4000)][r.below(3) as usize]).collect();
        let mut script = Vec::new();
        let mut pos = 0u32;
        for _ in 0..n_seq {
            let ll = pick(&mut r, &lls).max((pos == 0) as u32);
            let ml = pick(&mut r, &mls);
            let off = pick(&mut r, &offs).min(pos + ll);
            if (pos + ll + ml) as usize > BLOCK_SIZE {
                break;
            }
            script.push((ll, off, ml));
            pos += ll + ml;
        }
        let (block, parse) = scripted(&script, i);
        cases.push((format!("script{i}"), block, parse));
    }
    check_scripted(&ctx, &kernels, &cases);
}

/// K4 encodes the sequences in chunks of 256, last chunk first: sequence counts on both sides of
/// chunk multiples (so the chunk holding the initializing last sequence is full or partial), wide
/// code palettes (bit counts per sequence varying across the staging words), and sequences with
/// 16-bit literal-length, match-length and offset extra fields next to each other.
#[test]
fn k4_chunk_boundaries_match_cpu() {
    let mut r = Lcg(0xc4c);
    let mut cases = Vec::new();
    for (i, &n_seq) in [1usize, 2, 4, 5, 255, 256, 257, 259, 511, 512, 513, 1000, 1024, 1025].iter().enumerate() {
        let mut script = Vec::new();
        let mut pos = 0u32;
        let budget = BLOCK_SIZE as u32 - 1024;
        for _ in 0..n_seq {
            let ll = [0, r.below(4), r.below(12)][r.below(3) as usize].max((pos == 0) as u32);
            let ml = 3 + [r.below(2), r.below(8), r.below(30)][r.below(3) as usize];
            let off = (1 + [r.below(4), r.below(200), r.below(20000)][r.below(3) as usize]).min(pos + ll);
            if pos + ll + ml > budget {
                break;
            }
            script.push((ll, off, ml));
            pos += ll + ml;
        }
        assert_eq!(script.len(), n_seq, "script {i} did not fit");
        let (block, parse) = scripted(&script, 0x100 + i as u64);
        cases.push((format!("chunks{n_seq}"), block, parse));
    }
    // Wide fields (128K blocks): 16-bit literal-length and offset and 15-bit match-length extra
    // bits in one sequence, a 16-bit offset in another, short sequences between them.
    if BLOCK_SIZE >= 1 << 17 {
        let mut script = vec![(65600, 65550, 33000)];
        script.extend((0..300).map(|i| (i % 5, 1 + (i * 37) % 900, 4 + i % 11)));
        script.push((0, 90000, 20000));
        let (block, parse) = scripted(&script, 0x16);
        cases.push(("wide_fields".to_string(), block, parse));
    }
    for huffman in [false, true] {
        let (ctx, kernels) = setup_frames(huffman);
        check_scripted(&ctx, &kernels, &cases);
    }
}

/// With `min_match = 4` a 128K block can hold more than 0x7F00 sequences (`MAX_SEQS` is
/// `BLOCK_SIZE / 4 + 1`), so K4's 3-byte nbSeq header is reachable: 4 literals then 32767
/// length-4 matches at offset 4 fill the block exactly. Unreachable below 128K.
#[cfg(feature = "block-128k")]
#[test]
fn k4_nbseq_three_byte_form_matches_cpu() {
    let n = BLOCK_SIZE / 4 - 1;
    let mut script = vec![(4, 4, 4)];
    script.extend(std::iter::repeat_n((0, 4, 4), n - 1));
    let (block, parse) = scripted(&script, 0x7f00);
    assert!(parse.sequences.len() >= 0x7F00);
    // Also next to an ordinary block, so the two frames' seqs regions sit side by side.
    let (block2, parse2) = scripted(&[(10, 5, 20), (3, 30, 9)], 1);
    let cases = vec![
        ("nbseq_3byte".to_string(), block, parse),
        ("small".to_string(), block2, parse2),
    ];
    for huffman in [false, true] {
        let (ctx, kernels) = setup_frames_for(RUNG1, huffman);
        check_scripted(&ctx, &kernels, &cases);
        check_scripted(&ctx, &kernels, &cases[..1]);
        // Byte equality alone would also hold for a Raw block: pin that the GPU frame really is
        // a Compressed block whose sequences section starts with the 3-byte nbSeq form.
        let frame = &frames_from_parses(&ctx, &kernels, &[&cases[0].1], &[cases[0].2.clone()]).unwrap()[0];
        let n = cases[0].2.sequences.len();
        assert_eq!(nb_seq_header(frame, kernels.frame_options()), [0xFF, (n - 0x7F00) as u8, ((n - 0x7F00) >> 8) as u8]);
    }
}

/// Composes `perm` with itself `k` times (`perm^k`).
#[cfg(feature = "block-128k")]
fn compose_pow(perm: &[usize], k: usize) -> Vec<usize> {
    let l = perm.len();
    let mut cur: Vec<usize> = (0..l).collect();
    for _ in 0..k {
        cur = cur.iter().map(|&i| perm[i]).collect();
    }
    cur
}

/// A permutation of `0..l` whose powers `perm^1..=perm^max_k` all avoid "successions"
/// (`perm^k[i+1] == perm^k[i] + 1`): a hash-chain match found up to `max_k` generations back (the
/// deepest a `depth`-limited chain walk can reach) always caps at exactly 4 bytes, whichever
/// generation it lands on, rather than accidentally continuing into the next chunk. Fisher-Yates,
/// then fix up violations (any power, any `i`) with random swaps until none remain.
#[cfg(feature = "block-128k")]
fn no_succession_at_any_depth_perm(seed: u64, l: usize, max_k: usize) -> Vec<usize> {
    let mut r = Lcg(seed);
    let mut perm: Vec<usize> = (0..l).collect();
    for i in (1..l).rev() {
        let j = r.below((i + 1) as u32) as usize;
        perm.swap(i, j);
    }
    for _pass in 0..10_000 {
        let mut fixed_any = false;
        for k in 1..=max_k {
            let pk = compose_pow(&perm, k);
            if let Some(i) = (0..l - 1).find(|&i| pk[i + 1] == pk[i] + 1) {
                let j = r.below(l as u32) as usize;
                perm.swap(i + 1, j);
                fixed_any = true;
            }
        }
        if !fixed_any {
            return perm;
        }
    }
    panic!("no_succession_at_any_depth_perm: did not converge for l={l} max_k={max_k}");
}

/// A block that `RUNG1` parses into more than `0x7F00` sequences, found by the *real* hash-chain
/// match finder (not scripted): `l` unique 4-byte seed values (byte 0 = the value's own index, so
/// two different values can never share a first byte — whenever a candidate match's successor is
/// a *different* value, the byte right after the match differs immediately, capping it at exactly
/// 4 bytes instead of drifting a few bytes into the next chunk), then repeated "generations" each
/// a fixed permutation (`perm`, applied again each generation) of the previous generation's 4-byte
/// values. Every generation's chunk is a genuine offset match against the previous generation
/// (found by K1's real hash chains), and `perm`'s powers avoid successions up to RUNG1's `depth`
/// (8), so a chain walk landing on any of the last 8 generations still caps at 4 bytes: every
/// generation is `l` separate `len == 4` sequences, not one long one. `(l, seed) = (72, 3)` was
/// found by a small search over both and confirmed with the CPU oracle (`reference::compress_block`).
#[cfg(feature = "block-128k")]
fn over_0x7f00_sequences_block() -> Vec<u8> {
    let l = 72usize;
    let gens = BLOCK_SIZE / 4 / l - 1;
    let seed = 3u64;
    let mut r = Lcg(seed);
    let seed_vals: Vec<[u8; 4]> =
        (0..l).map(|k| [k as u8, r.next() as u8, r.next() as u8, r.next() as u8]).collect();
    let perm = no_succession_at_any_depth_perm(seed ^ 0x5EED, l, 8);

    let mut block = Vec::with_capacity(BLOCK_SIZE);
    for v in &seed_vals {
        block.extend_from_slice(v);
    }
    let mut cur = seed_vals;
    for _ in 0..gens {
        let next: Vec<[u8; 4]> = (0..l).map(|i| cur[perm[i]]).collect();
        for v in &next {
            block.extend_from_slice(v);
        }
        cur = next;
    }
    // Fill the remainder (the block doesn't divide evenly by `l`) with incompressible bytes.
    while block.len() < BLOCK_SIZE {
        block.push(r.next() as u8);
    }
    assert_eq!(block.len(), BLOCK_SIZE);
    block
}

/// A block whose RUNG1 parse (found by the real K1/K2/K3 pipeline, not scripted) has more than
/// `0x7F00` sequences: the GPU frame (K1→K2→K3→K5→K4, the full real pipeline via
/// `compress_frames`) must still equal the CPU reference frame byte for byte and decode. Only
/// reachable at 128K (`over_0x7f00_sequences_block`'s doc, and `k4_nbseq_three_byte_form_matches_cpu`).
#[cfg(feature = "block-128k")]
#[test]
fn gpu_frames_match_cpu_over_0x7f00_sequences() {
    let block = over_0x7f00_sequences_block();
    let cpu_parse = compress_block(&block, RUNG1);
    assert!(cpu_parse.sequences.len() >= 0x7F00, "only {} sequences", cpu_parse.sequences.len());

    for huffman in [false, true] {
        let (ctx, kernels) = setup_frames_for(RUNG1, huffman);
        let want = write_frame(&block, &cpu_parse, kernels.frame_options());
        let got = &compress_frames(&ctx, &kernels, &[&block]).expect("compress_frames")[0];
        assert!(*got == want, "GPU frame != CPU frame; {}", first_byte_diff(got, &want));
        let back = zstd::bulk::decompress(got, BLOCK_SIZE).expect("libzstd decode");
        assert_eq!(back, block, "libzstd output differs from the input block");
    }
}

/// The 3 bytes starting the sequences section of a one-block frame whose block is Compressed
/// (asserted) with a Raw or RLE literals section (asserted).
#[cfg(feature = "block-128k")]
fn nb_seq_header(frame: &[u8], opts: FrameOptions) -> [u8; 3] {
    let h = frame_header(opts).len();
    assert_eq!((frame[h] >> 1) & 3, 2, "block type must be Compressed");
    let c = &frame[h + 3..];
    let lit_type = c[0] & 3;
    assert!(lit_type <= 1, "literals section must be Raw or RLE, got type {lit_type}");
    let (hdr, size) = match (c[0] >> 2) & 3 {
        0 | 2 => (1, (c[0] >> 3) as usize),
        1 => (2, (c[0] >> 4) as usize | (c[1] as usize) << 4),
        _ => (3, (c[0] >> 4) as usize | (c[1] as usize) << 4 | (c[2] as usize) << 12),
    };
    let body = if lit_type == 0 { size } else { 1 };
    c[hdr + body..hdr + body + 3].try_into().unwrap()
}

/// Hand-built cases pinning each decision boundary of the mode choice and the block type.
#[test]
fn k4_decision_boundaries_match_cpu() {
    let (ctx, kernels) = setup_frames(false);
    let mut cases = Vec::new();
    let mut push = |name: &str, script: &[(u32, u32, u32)]| {
        let (block, parse) = scripted(script, cases.len() as u64);
        cases.push((name.to_string(), block, parse));
    };

    // LL histogram [0, 5, 1, 1]: computed cost + NCount bits == predefined cost exactly, so the
    // tie goes to predefined.
    let h = [0u32, 5, 1, 1];
    let log = choose_table_log(7, 3, 9);
    let norm = normalize(&h, 7, log);
    let mut ncount = Vec::new();
    write_ncount(&norm, log, &mut ncount);
    assert_eq!(
        cost_x256(&h, &norm, log) + 2048 * ncount.len() as u64,
        cost_x256(&h, &LL_DEFAULT_NORM, LL_DEFAULT_LOG),
        "not a tie"
    );
    push("cost_tie", &[(1, 1, 20), (1, 7, 20), (1, 9, 20), (1, 7, 20), (1, 9, 20), (2, 7, 20), (3, 9, 20)]);
    // Two sequences with one code each: too few for RLE.
    push("two_same", &[(4, 3, 9), (4, 3, 9)]);
    push("three_same", &[(4, 3, 9), (4, 3, 9), (4, 3, 9)]);
    // Two ML codes with equal largest counts: the normalization surplus goes to the lower code.
    let mut s = vec![(1, 1, 90); 20];
    s.extend(vec![(1, 1, 100); 20]);
    s.extend(vec![(1, 1, 140); 5]);
    push("normalize_tie", &s);
    // 135 sequences over 36 ML codes, skewed so the computed table wins: the table log (6) comes
    // from the symbol count, not the total (5).
    let mut s: Vec<_> = (1..36u32).map(|c| (1, 1, ML_BASE[c as usize])).collect();
    s.extend(vec![(1, 1, 3); 100]);
    push("symbol_bound_log", &s);

    // One sequence whose match length puts the content at exactly BLOCK_SIZE (Raw) and at
    // BLOCK_SIZE - 1 (Compressed).
    let content = |ml: u32| {
        let (_, parse) = scripted(&[(1, 1, ml)], 0);
        let mut c = Vec::new();
        write_literals_raw(&parse.literals, &mut c);
        write_sequences_section_auto(&parse.sequences, &mut c);
        c.len()
    };
    for target in [BLOCK_SIZE, BLOCK_SIZE - 1] {
        let ml = (3..200).find(|&ml| content(ml) == target).expect("no match length hits the boundary");
        push(&format!("content_{}", BLOCK_SIZE as i64 - target as i64), &[(1, 1, ml)]);
    }
    assert_eq!(histograms(&cases[0].2.sequences)[0][..4], h, "cost_tie script");
    let bound = &cases.iter().find(|c| c.0 == "symbol_bound_log").unwrap().2.sequences;
    assert_eq!(choose_table_log(bound.len() as u32, 36, 9), 6);
    let ml = StreamTable::choose(StreamKind::MatchLength, &histograms(bound)[2], bound.len());
    assert_eq!(ml.mode, SeqMode::Compressed, "symbol_bound_log: ML table must be computed");
    let n = cases.len();
    let blocks: Vec<&[u8]> = cases[n - 2..].iter().map(|c| c.1.as_slice()).collect();
    for (b, raw) in blocks.iter().zip([true, false]) {
        let parse = &cases[n - 2 + (!raw) as usize].2;
        let frame = write_frame(b, parse, NO_HUFFMAN);
        let bt = (frame[frame_header(NO_HUFFMAN).len()] >> 1) & 3;
        assert_eq!(bt, if raw { 0 } else { 2 }, "boundary case block type");
    }

    check_scripted(&ctx, &kernels, &cases);
}

// ---- K5: Huffman literals on scripted parses ----

/// (name, block, parse) whose literals are exactly `lits`: one sequence (`lits.len()` literals,
/// offset 1, match to the block end), or no sequences at all for a full block of literals.
fn lits_case(name: &str, lits: Vec<u8>) -> (String, Vec<u8>, BlockOutput) {
    let n = lits.len();
    let script: Vec<(u32, u32, u32)> = if n == BLOCK_SIZE {
        Vec::new()
    } else {
        assert!(n >= 1 && n + 3 <= BLOCK_SIZE, "{name}: {n} literals do not fit");
        vec![(n as u32, 1, (BLOCK_SIZE - n) as u32)]
    };
    let mut it = lits.into_iter();
    let (block, parse) = scripted_with(&script, || it.next().expect("literal source exhausted"));
    (name.to_string(), block, parse)
}

/// Deterministic Fisher-Yates shuffle.
fn shuffle(v: &mut [u8], r: &mut Lcg) {
    for i in (1..v.len()).rev() {
        v.swap(i, r.below(i as u32 + 1) as usize);
    }
}

/// `counts[s]` copies of each byte `s`, shuffled.
fn from_counts(counts: &[u32], r: &mut Lcg) -> Vec<u8> {
    let mut v: Vec<u8> = (0..counts.len()).flat_map(|s| std::iter::repeat_n(s as u8, counts[s] as usize)).collect();
    shuffle(&mut v, r);
    v
}

/// Low-entropy literals over a small alphabet ('a' + products of two dice).
fn low_entropy(r: &mut Lcg, n: usize) -> Vec<u8> {
    (0..n).map(|_| b'a' + (r.below(6) * r.below(6)) as u8).collect()
}

fn raw_section_len(n: usize) -> usize {
    n + 1 + (n >= 32) as usize + (n >= 4096) as usize
}

fn histogram(lits: &[u8]) -> [u32; 256] {
    let mut counts = [0u32; 256];
    for &b in lits {
        counts[b as usize] += 1;
    }
    counts
}

/// Depth of the deepest leaf of an unlimited Huffman tree for `counts` (test-side, heap-based).
fn unlimited_depth(counts: &[u32]) -> u32 {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> =
        counts.iter().filter(|&&c| c > 0).map(|&c| Reverse((c as u64, 0))).collect();
    while heap.len() > 1 {
        let Reverse((a, da)) = heap.pop().unwrap();
        let Reverse((b, db)) = heap.pop().unwrap();
        heap.push(Reverse((a + b, da.max(db) + 1)));
    }
    heap.pop().map_or(0, |Reverse((_, d))| d)
}

/// First literal set of `n` bytes, uniform over `k` symbols with `n`, `k` searched, whose
/// Compressed section is exactly `raw_section_len(n) - delta` bytes long.
fn size_boundary_lits(delta: usize, seed: u64) -> Vec<u8> {
    let mut r = Lcg(seed);
    for _ in 0..20000 {
        let n = 64 + r.below(1500) as usize;
        let k = 120 + r.below(137);
        let lits: Vec<u8> = (0..n).map(|_| r.below(k) as u8).collect();
        if compressed_section(&lits).is_some_and(|c| c.len() + delta == raw_section_len(n)) {
            return lits;
        }
    }
    panic!("no literal set with a Compressed section {delta} bytes below Raw");
}

/// Weight histogram (12 values) of the symbols below `t.max_symbol`: what the FSE description codes.
fn weight_counts(t: &HufTable) -> [u32; 12] {
    let mut wc = [0u32; 12];
    for s in 0..t.max_symbol {
        wc[t.weight(s) as usize] += 1;
    }
    wc
}

/// True when the FSE weights description normalizes with a surplus that two weight values tie
/// for (it goes to the lower one).
fn weights_normalize_tie(t: &HufTable) -> bool {
    let wc = weight_counts(t);
    let n = t.max_symbol as u32;
    let max = *wc.iter().max().unwrap();
    let used = wc.iter().filter(|&&c| c > 0).count();
    if t.max_symbol <= 128 || wc.iter().filter(|&&c| c == max).count() < 2 || max == n || max == 1 {
        return false;
    }
    let size = 1u32 << choose_table_log(n, used, 6);
    let sum: u32 = wc.iter().filter(|&&c| c > 0).map(|&c| (c * size / n).max(1)).sum();
    sum < size
}

/// Literals over symbols up to 255 (FSE weights) whose weights normalization hits the surplus tie,
/// compressed.
fn weights_tie_lits() -> Vec<u8> {
    let mut r = Lcg(0x77);
    for _ in 0..5000 {
        let mut counts = [0u32; 256];
        for c in counts.iter_mut() {
            let bits = r.below(8);
            *c = r.below(1 << bits);
        }
        counts[255] = counts[255].max(1);
        let t = build_table(&counts).unwrap();
        let lits = from_counts(&counts, &mut r);
        if lits.len() + 3 <= BLOCK_SIZE
            && weights_normalize_tie(&t)
            && compressed_section(&lits).is_some_and(|c| c.len() < raw_section_len(lits.len()))
        {
            return lits;
        }
    }
    panic!("no literal set hits the weights normalization tie");
}

/// Scripted literal sets pinning every literals-section path: Raw / RLE / Compressed, 1 and 4
/// streams, all four size formats, direct and FSE weights (incl. max_symbol 128 and 129),
/// the undescribable table, length limiting, nbSeq == 0, and the Compressed-vs-Raw size boundary.
fn literal_cases() -> Vec<(String, Vec<u8>, BlockOutput)> {
    let mut r = Lcg(0x6b35);
    let r = &mut r;
    let mut cases = Vec::new();
    let big = 20000.min(BLOCK_SIZE - 3);
    for n in [64, 65, 100, 255, 256, 257, 259, 300, 1000, 1023, 1024, 5000, 16383, 16384, big] {
        if n + 3 <= BLOCK_SIZE {
            cases.push(lits_case(&format!("low_{n}"), low_entropy(r, n)));
        }
    }
    cases.push(lits_case("nbseq0", low_entropy(r, BLOCK_SIZE)));
    cases.push(lits_case("raw_2", vec![7, 8]));
    // Small symbols: Huffman would win, but 63 literals are below MIN_HUF_LITERALS.
    cases.push(lits_case("raw_63", (0..MIN_HUF_LITERALS - 1).map(|_| r.below(3) as u8).collect()));
    cases.push(lits_case("huf_64", (0..MIN_HUF_LITERALS).map(|_| r.below(3) as u8).collect()));
    cases.push(lits_case("raw_random", (0..3000).map(|_| r.next() as u8).collect()));
    // RLE literals: all literals equal means every byte of a valid parse's block is equal, so the
    // frame is an RLE block and never shows its literals section. Pair the parse with a block
    // that differs in its last byte instead: the block-level RLE check fails, and the frame
    // (Compressed) holds the RLE literals section and decodes to the parse's content.
    // n covers each RLE header size (1, 2 and 3 bytes).
    // n = 1 (also only valid in a uniform block) is Raw, not RLE.
    for n in [1, 2, 31, 32, 4095, 4096] {
        let (name, mut block, parse) = lits_case(&format!("{MISMATCHED}equal_{n}"), vec![0x61; n]);
        block[BLOCK_SIZE - 1] ^= 1;
        cases.push((name, block, parse));
    }
    cases.push(lits_case("two_symbols", (0..500).map(|_| [3u8, 250][r.below(2) as usize]).collect()));
    // Symbols up to 255, skewed: FSE-compressed weights.
    let mut v: Vec<u8> = (0..3000).map(|_| (r.next() & r.next()) as u8).collect();
    v.push(255);
    cases.push(lits_case("fse_weights", v));
    cases.push(lits_case("weights_normalize_tie", weights_tie_lits()));
    // max_symbol 128 (the largest direct description) and 129 (the smallest FSE one).
    for top in [128u32, 129] {
        let counts: Vec<u32> = (0..=top).map(|s| 1 + s * s % 37 + if s < 8 { 300 } else { 0 }).collect();
        cases.push(lits_case(&format!("max_symbol_{top}"), from_counts(&counts, r)));
    }
    // All 256 symbols equally often: every weight equal, so no description exists -> Raw.
    cases.push(lits_case("undescribable", from_counts(&[10; 256], r)));
    // Fibonacci counts: an unlimited tree 17 deep, limited to 11.
    let mut fib = vec![1u32, 1];
    while fib.len() < 18 {
        fib.push(fib[fib.len() - 1] + fib[fib.len() - 2]);
    }
    cases.push(lits_case("fibonacci", from_counts(&fib, r)));
    // Steep, noisy geometric histograms over many symbols (most need limiting).
    for i in 0..24 {
        let nsym = 20 + r.below(237) as usize;
        let decay = 1 + r.below(12);
        let mut c = [0u32; 256];
        for s in 0..nsym {
            let base = (1u32 << 13) >> ((s as u32 * 8 / decay).min(13));
            c[(s * 7 + i) % 256] = base.max(1) + r.below(3);
        }
        // Keep the literals within half a block (the sum can exceed a 16K block).
        let total: u32 = c.iter().sum();
        let limit = (BLOCK_SIZE as u32 - 3) / 2;
        if total > limit {
            for x in c.iter_mut() {
                *x = (*x as u64 * limit as u64 / total as u64).max((*x > 0) as u64) as u32;
            }
        }
        cases.push(lits_case(&format!("steep{i}"), from_counts(&c, r)));
    }
    // Compressed exactly as long as Raw (-> Raw) and one byte shorter (-> Compressed).
    cases.push(lits_case("size_tie", size_boundary_lits(0, 0x71e)));
    cases.push(lits_case("size_minus_1", size_boundary_lits(1, 0x71f)));
    cases
}

/// Literals-section features of a CPU frame.
fn literal_features(frame: &[u8], parse: &BlockOutput) -> Vec<&'static str> {
    let h = frame_header(FrameOptions::default()).len();
    if (frame[h] >> 1) & 3 != 2 {
        return vec!["block_not_compressed"];
    }
    let c = &frame[h + 3..];
    let (ty, sf) = (c[0] & 3, (c[0] >> 2) & 3);
    let n = parse.literals.len();
    let counts = histogram(&parse.literals);
    let mut f = vec![["lit_raw", "lit_rle", "lit_huf", "lit_treeless"][ty as usize]];
    if parse.sequences.is_empty() {
        f.push("nbseq0");
    }
    if ty == 2 {
        f.push(["sf0", "sf1", "sf2", "sf3"][sf as usize]);
        f.push(if sf == 0 { "1stream" } else { "4streams" });
        let d = c[[3, 3, 4, 5][sf as usize]];
        f.push(if d < 128 { "fse_weights" } else { "direct_weights" });
        if d == 255 {
            f.push("direct_128");
        }
        if unlimited_depth(&counts) > HUF_MAX_BITS {
            f.push("limited");
        }
        if compressed_section(&parse.literals).unwrap().len() + 1 == raw_section_len(n) {
            f.push("huf_size_minus_1");
        }
        if n == 16384 {
            f.push("size_16384");
        }
        if weights_normalize_tie(&build_table(&counts).unwrap()) {
            f.push("weights_normalize_tie");
        }
    }
    if ty == 0 && n < MIN_HUF_LITERALS && compressed_section(&parse.literals).is_some_and(|c| c.len() < raw_section_len(n)) {
        f.push("raw_below_min_but_compressible");
    }
    if ty == 0 && n >= MIN_HUF_LITERALS {
        match build_table(&counts).map(|t| table_description(&t)) {
            Some(None) => f.push("raw_undescribable"),
            _ => match compressed_section(&parse.literals) {
                Some(s) if s.len() == raw_section_len(n) => f.push("raw_size_tie"),
                Some(_) => f.push("raw_not_smaller"),
                None => {}
            },
        }
    }
    f
}

/// The literal cases reach every literals-section path on the CPU, so byte equality in
/// `k5_literal_cases_match_cpu` covers each GPU path.
#[test]
fn literal_cases_cover_every_path() {
    let mut seen = std::collections::BTreeSet::new();
    for (name, block, parse) in literal_cases() {
        let frame = write_frame(&block, &parse, FrameOptions::default());
        let f = literal_features(&frame, &parse);
        assert!(!f.contains(&"block_not_compressed"), "{name}: block not Compressed");
        seen.extend(f);
    }
    for want in [
        "lit_raw", "lit_rle", "lit_huf", "nbseq0", "sf0", "sf1", "sf2", "sf3", "1stream", "4streams",
        "fse_weights", "direct_weights", "direct_128", "limited", "raw_undescribable", "raw_size_tie",
        "raw_not_smaller", "huf_size_minus_1", "size_16384", "weights_normalize_tie",
        "raw_below_min_but_compressible",
    ] {
        assert!(seen.contains(want), "no literal case reaches {want}: {seen:?}");
    }
}

#[test]
fn k5_literal_cases_match_cpu() {
    let (ctx, kernels) = setup_frames(true);
    let cases = literal_cases();
    check_scripted(&ctx, &kernels, &cases);
    // A few alone too: nothing leaks between neighbouring blocks' frames.
    for c in cases.iter().step_by(5) {
        check_scripted(&ctx, &kernels, std::slice::from_ref(c));
    }
}

/// Random scripts with random literal styles (bytes, low entropy, constant, a few symbols):
/// literals sections of every type and many lengths, spliced by K4 at every byte alignment.
#[test]
fn k5_random_scripts_match_cpu() {
    let (ctx, kernels) = setup_frames(true);
    let mut r = Lcg(0x4b5);
    let mut cases = Vec::new();
    for i in 0..300 {
        let n_seq = [0, 1, 2, 3, 5, 13, 60, 200][r.below(8) as usize];
        let mut script = Vec::new();
        let mut pos = 0u32;
        for _ in 0..n_seq {
            let ll = [r.below(4), r.below(40), r.below(700)][r.below(3) as usize].max((pos == 0) as u32);
            let ml = 3 + [r.below(4), r.below(60), r.below(3000)][r.below(3) as usize];
            let off = (1 + r.below(2000)).min(pos + ll);
            if (pos + ll + ml) as usize + 64 > BLOCK_SIZE {
                break;
            }
            script.push((ll, off, ml));
            pos += ll + ml;
        }
        // Most scripts end in one long match that leaves a random number of trailing literals.
        let tail = [0u32, 1, 7, 100, 5000, BLOCK_SIZE as u32][r.below(6) as usize];
        let room = BLOCK_SIZE as u32 - pos;
        if tail + 8 <= room {
            let ll = 1 + r.below(4);
            script.push((ll, 1 + r.below(pos + ll), room - tail - ll));
        }
        let style = r.below(4);
        let mut lr = Lcg(1000 + i);
        let (block, parse) = scripted_with(&script, || match style {
            0 => lr.next() as u8,
            1 => (lr.below(6) * lr.below(6)) as u8,
            2 => 0x42,
            _ => [1u8, 2, 3, 200][lr.below(4) as usize],
        });
        cases.push((format!("script{i}"), block, parse));
    }
    check_scripted(&ctx, &kernels, &cases);
}

/// K5 gathers the literals from the block through a per-thread index over groups of
/// G = ceil(n_seq / 256) sequences (speed phase S4). Scripts with hundreds to thousands of
/// sequences (G > 1), mostly empty literal runs (so group starts and seeks land on runs of
/// length 0) with a few long ones (runs across group and thread boundaries), trailing literals of
/// every size including none; Raw and Huffman (1 and 4 streams) sections; K5 with and without
/// Huffman. Only these cases reach a seek that walks more than one run.
#[test]
fn k5_gather_many_sequences_match_cpu() {
    let mut r = Lcg(0x5a4);
    let mut cases = Vec::new();
    let most = BLOCK_SIZE as u32 / 8;
    for (i, &target) in [255u32, 256, 257, 511, 700, 2000, most, most].iter().cycle().take(32).enumerate() {
        let mut script = Vec::new();
        let mut pos = 0u32;
        let zero_share = [2u32, 6, 9][r.below(3) as usize];
        for _ in 0..target {
            let ll = if r.below(10) < zero_share {
                0
            } else {
                [1 + r.below(3), 1 + r.below(40), 50 + r.below(BLOCK_SIZE as u32 / 320)][r.below(3) as usize]
            }
            .max((pos == 0) as u32);
            let ml = 4 + r.below(4);
            if (pos + ll + ml) as usize + 16 > BLOCK_SIZE {
                break;
            }
            script.push((ll, (1 + r.below(64)).min(pos + ll), ml));
            pos += ll + ml;
        }
        // Every fourth script ends with a match up to the block's last byte (no trailing literals).
        if i % 4 == 0 {
            let room = BLOCK_SIZE as u32 - pos;
            script.push((1, 1 + r.below(pos + 1), room - 1));
        }
        let style = r.below(4);
        let mut lr = Lcg(2000 + i as u64);
        let (block, parse) = scripted_with(&script, || match style {
            0 => lr.next() as u8,
            1 => (lr.below(6) * lr.below(6)) as u8,
            2 => 0x42,
            _ => [1u8, 2, 3, 200][lr.below(4) as usize],
        });
        assert!(parse.sequences.len() >= 255, "script {i}: {} sequences", parse.sequences.len());
        cases.push((format!("many{i}"), block, parse));
    }
    for huffman in [true, false] {
        let (ctx, kernels) = setup_frames(huffman);
        check_scripted(&ctx, &kernels, &cases);
    }
}

/// The literal stream is the block bytes the sequences leave uncovered (what K5 and the parse
/// path gather instead of reading K3 output): true of every CPU parse, every preset.
#[test]
fn literals_are_the_uncovered_block_bytes() {
    for (name, block) in all_blocks() {
        for (preset, params) in GPU_PRESETS {
            let parse = compress_block(&block, params);
            let got = gzc_gpu::compressor::gather_literals(&block, &parse.sequences);
            assert!(got == parse.literals, "{name} {preset}: gathered literals differ");
        }
    }
}

// ---- K2 ----

/// K2 params beyond the presets: deeper Dfast walks and the smallest search_cap (8 = the long
/// hash width, the edge of the cross-chain early-out argument in k2_best.wgsl), and a Single
/// chain with a small cap.
const K2_VARIANTS: [MatchParams; 6] = [
    MatchParams { depth: 4, ..LVL3 },
    MatchParams { depth: 16, search_cap: 8, ..LVL3 },
    MatchParams { depth: 8, search_cap: 16, ..LVL3 },
    MatchParams { hashes: gzc_core::params::Hashes::Single, min_match: 5, depth: 4, lazy: 2, search_cap: 8, segment_log2: 0 },
    // Deep Single walks over the fingerprint skips (S8), with min_match 4 (a byte-4 mismatch
    // skips only against a best of >= 4) and 6.
    MatchParams { hashes: gzc_core::params::Hashes::Single, min_match: 4, depth: 64, lazy: 0, search_cap: 16, segment_log2: 0 },
    MatchParams { hashes: gzc_core::params::Hashes::Single, min_match: 6, depth: 64, lazy: 2, search_cap: 64, segment_log2: 0 },
];

/// Blocks for K2's fingerprint skips (S8): candidates that share the hash but not the first 4
/// bytes (hash collisions), candidates equal in 4 bytes that differ at byte 4 (the byte field),
/// and ties of length 4 and 5 between candidates. `words-tail1`: 4-byte words from 12 values, each
/// followed by one of 3 bytes (so a word recurs with a different fifth byte); `words-tail2`: the
/// same with two trailing bytes; `alphabet2`/`alphabet3`: random bytes of 2 or 3 values (dense
/// chains with every length 0..=cap); `collide`: 4-byte random words drawn from 4096 values, which
/// the 16-bit hash maps onto few buckets with many collisions.
fn k2_fp_blocks() -> Vec<(String, Vec<u8>)> {
    let mut r = Lcg(0x58f);
    let words: Vec<u32> = (0..12).map(|_| r.next()).collect();
    let pools: Vec<u32> = (0..4096).map(|_| r.next()).collect();
    let mut out = Vec::new();
    for tail in [1usize, 2] {
        let mut b = Vec::with_capacity(BLOCK_SIZE + 8);
        while b.len() < BLOCK_SIZE {
            b.extend_from_slice(&words[r.below(12) as usize].to_le_bytes());
            for _ in 0..tail {
                b.push([7u8, 9, 200][r.below(3) as usize]);
            }
        }
        b.truncate(BLOCK_SIZE);
        out.push((format!("words-tail{tail}"), b));
    }
    for alphabet in [2u32, 3] {
        out.push((format!("alphabet{alphabet}"), (0..BLOCK_SIZE).map(|_| r.below(alphabet) as u8).collect()));
    }
    let mut b = Vec::with_capacity(BLOCK_SIZE);
    while b.len() < BLOCK_SIZE {
        b.extend_from_slice(&pools[r.below(4096) as usize].to_le_bytes());
    }
    out.push(("collide".to_string(), b));
    out
}

/// CPU mirror of K2's walk with the cap early-out (stop the whole walk, all chains, at the first
/// candidate reaching search_cap).
fn find_best_early_out(block: &[u8], chains: &[Vec<u32>], params: &MatchParams) -> Vec<Match> {
    let mut best = vec![Match::default(); BLOCK_SIZE];
    let (depth, cap) = (params.depth, params.search_cap as usize);
    for (p, slot) in best.iter_mut().enumerate().take(gzc_core::config::PARSE_END) {
        let (mut best_len, mut best_q) = (0usize, 0usize);
        'walk: for preds in chains {
            let mut q = preds[p];
            for _ in 0..depth {
                if q == gzc_core::config::NO_POS {
                    break;
                }
                let len = match_len_capped(block, p, q as usize, cap);
                if len > best_len || (len == best_len && q as usize > best_q) {
                    (best_len, best_q) = (len, q as usize);
                    if len == cap {
                        break 'walk;
                    }
                }
                q = preds[q as usize];
            }
        }
        if best_len >= params.min_match as usize {
            *slot = Match { offset: (p - best_q) as u32, len: best_len as u32 };
        }
    }
    best
}

/// Dfast blocks where both chains offer candidates at the same p = 5000: there a 200-byte string
/// `s`; further back (q = 1000) its full copy (on the long chain, caps); nearer (q = 3000) a
/// prefix of `s` of 5..=9 bytes and then a break (on the short chain, and on the long chain too
/// once it is >= 8 bytes). The background seed is picked so no hash collision gets in the way:
/// the first long-chain candidate at 5000 is 3000 (near >= 8) or 1000, the first short-chain one
/// is 3000.
fn k2_chain_blocks() -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    for near in [5usize, 6, 7, 8, 9] {
        let block = (0u64..)
            .map(|seed| {
                let mut block = random(40 + 100 * seed + near as u64, BLOCK_SIZE);
                let s = random(50, 200);
                block[1000..1200].copy_from_slice(&s);
                block[3000..3000 + near].copy_from_slice(&s[..near]);
                block[3000 + near] = s[near] ^ 0x55;
                block[5000..5200].copy_from_slice(&s);
                block
            })
            .find(|block| {
                let ch = chains(block, &LVL3);
                ch[0][5000] == if near >= 8 { 3000 } else { 1000 } && ch[1][5000] == 3000
            })
            .unwrap();
        out.push((format!("chain-near{near}"), block));
    }
    out
}

/// The early-out walk equals `find_best` on every test block, for every preset and K2 variant.
#[test]
fn k2_cap_early_out_matches_find_best_cpu() {
    let mut blocks = all_blocks();
    blocks.extend(k2_chain_blocks());
    blocks.extend(k2_fp_blocks());
    for params in GPU_PRESETS.iter().map(|p| p.1).chain(K2_VARIANTS) {
        for (name, block) in &blocks {
            let ch = chains(block, &params);
            assert!(find_best_early_out(block, &ch, &params) == find_best(block, &ch, &params), "{params:?} {name}");
        }
    }
}

/// K2's best[] equals `find_best` exactly, for every preset and K2 variant.
#[test]
fn k2_best_matches_find_best() {
    let mut blocks = all_blocks();
    blocks.extend(k2_chain_blocks());
    blocks.extend(k2_fp_blocks());
    let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
    for params in GPU_PRESETS.iter().map(|p| p.1).chain(K2_VARIANTS) {
        let (ctx, kernels) = setup(params);
        let got = best_from_blocks(&ctx, &kernels, &refs).expect("best_from_blocks");
        for ((name, block), got) in blocks.iter().zip(&got) {
            let want = find_best(block, &chains(block, &params), &params);
            if let Some(p) = (0..BLOCK_SIZE).find(|&p| got[p] != want[p]) {
                panic!("{params:?} {name}: best[{p}] gpu {:?} cpu {:?}", got[p], want[p]);
            }
        }
    }
}

/// Dfast, the case the cross-chain early-out must get right: the long chain caps at a far
/// candidate while the short chain holds a nearer (larger q) one. A nearer candidate shorter
/// than 8 bytes is only on the short chain and loses on length (the early-out skips it); one of
/// >= 8 bytes is on the long chain too, ahead of the far copy, and wins there (with search_cap
/// 8 it caps first, and the far copy is never compared).
#[test]
fn k2_dfast_nearer_short_chain_candidate() {
    for params in [LVL3, MatchParams { search_cap: 8, ..LVL3 }] {
        let (ctx, kernels) = setup(params);
        let blocks = k2_chain_blocks();
        let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
        let got = best_from_blocks(&ctx, &kernels, &refs).expect("best_from_blocks");
        for ((name, block), got) in blocks.iter().zip(&got) {
            let near: u32 = name.trim_start_matches("chain-near").parse().unwrap();
            let want = find_best(block, &chains(block, &params), &params);
            assert_eq!(got[5000], want[5000], "{params:?} {name}");
            // Depth 1: the long chain's first candidate is the nearer copy once it is >= 8 bytes.
            let expect = if near >= 8 {
                Match { offset: 2000, len: near.min(params.search_cap) }
            } else {
                Match { offset: 4000, len: params.search_cap }
            };
            assert_eq!(got[5000], expect, "{params:?} {name}");
        }
    }
}

// ---- K3 lazy / lazy2 ----

/// Lazy parses off the preset table: `min_match` above 4 (repcode matches may still be 4 bytes)
/// and a small `search_cap`, so the parse extends many capped `best[]` entries.
const LAZY_VARIANTS: [MatchParams; 4] = [
    MatchParams { min_match: 6, ..RUNG2 },
    MatchParams { hashes: gzc_core::params::Hashes::Single, min_match: 5, depth: 4, lazy: 2, search_cap: 8, segment_log2: 0 },
    // Segmented: 1 KiB lazy1 with min_match 6, and 2 KiB lazy2 extending many capped matches.
    MatchParams { min_match: 6, segment_log2: 10, ..RUNG2 },
    MatchParams { hashes: gzc_core::params::Hashes::Single, min_match: 5, depth: 4, lazy: 2, search_cap: 8, segment_log2: 11 },
];

#[test]
fn gpu_matches_reference_lazy_variants() {
    for params in LAZY_VARIANTS {
        eprintln!("{params:?}");
        let (ctx, kernels) = setup(params);
        check_batch(&ctx, &kernels, &all_blocks(), params);
    }
}

/// Every hand-built lazy case (gain ties / wins by 1, `continue` after a win at step 1 and 2,
/// catch-up bounds, the dedup-store rep rule, the immediate offset_2 loop, PARSE_END guards),
/// run through K3 on its scripted `best[]` under both lazy presets. The K3 parse must equal
/// `lazy_parse` (and the pinned sequences where the case lists them), and the frame must equal
/// `write_frame` of it. The parse is compared directly because most of these mostly-random
/// blocks become Raw frames, which would hide the sequences. Batched and one by one.
#[test]
fn k3_lazy_hand_built_best_matches_cpu() {
    let cases = lazy_test_cases();
    assert_eq!(cases.len(), 21);
    for params in [RUNG2, LVL9] {
        let (ctx, kernels) = setup_frames_for(params, true);
        let check = |cases: &[&LazyCase]| {
            let blocks: Vec<&[u8]> = cases.iter().map(|c| c.block.as_slice()).collect();
            let bests: Vec<Vec<Match>> = cases.iter().map(|c| c.best.clone()).collect();
            let parses = parses_from_best(&ctx, &kernels, &blocks, &bests).expect("parses_from_best");
            let frames = frames_from_best(&ctx, &kernels, &blocks, &bests).expect("frames_from_best");
            for ((c, got), frame) in cases.iter().zip(&parses).zip(&frames) {
                let want = lazy_parse(&c.block, &c.best, &params);
                assert!(*got == want, "{} lazy {}: K3 != lazy_parse; {}", c.name, params.lazy, first_diff(got, &want));
                if let Some((_, pinned)) = c.expect.iter().find(|(p, _)| *p == params) {
                    assert_eq!(got.sequences, *pinned, "{} lazy {}: pinned sequences", c.name, params.lazy);
                }
                let want_frame = write_frame(&c.block, &want, kernels.frame_options());
                assert!(*frame == want_frame, "{} lazy {}: frame; {}", c.name, params.lazy, first_byte_diff(frame, &want_frame));
            }
        };
        let all: Vec<&LazyCase> = cases.iter().collect();
        check(&all);
        for c in &all {
            check(std::slice::from_ref(c));
        }
    }
}

/// The segmented parse (k3_seg.wgsl) on hand-built `best[]`: the segment-boundary cases under
/// their own params, and every lazy case under lvl9seg and a 1 KiB lazy1 variant. The K3 parse
/// must equal `reference::parse` (and the pinned sequences), the frame `write_frame` of it.
/// Batched and one by one.
#[test]
fn k3_seg_hand_built_best_matches_cpu() {
    let seg_cases = segment_test_cases();
    let lazy_cases = lazy_test_cases();
    let mut runs: Vec<(MatchParams, Vec<&LazyCase>)> = Vec::new();
    for c in &seg_cases {
        for (p, _) in &c.expect {
            match runs.iter_mut().find(|(q, _)| q == p) {
                Some((_, v)) => v.push(c),
                None => runs.push((*p, vec![c])),
            }
        }
    }
    for p in [LVL9SEG, MatchParams { segment_log2: 10, ..RUNG2 }] {
        let all: Vec<&LazyCase> = lazy_cases.iter().chain(&seg_cases).filter(|c| c.expect.iter().all(|(q, _)| q.min_match == p.min_match)).collect();
        runs.push((p, all));
    }
    for (params, cases) in runs {
        eprintln!("{params:?}: {} cases", cases.len());
        let (ctx, kernels) = setup_frames_for(params, true);
        let check = |cases: &[&LazyCase]| {
            let blocks: Vec<&[u8]> = cases.iter().map(|c| c.block.as_slice()).collect();
            let bests: Vec<Vec<Match>> = cases.iter().map(|c| c.best.clone()).collect();
            let parses = parses_from_best(&ctx, &kernels, &blocks, &bests).expect("parses_from_best");
            let frames = frames_from_best(&ctx, &kernels, &blocks, &bests).expect("frames_from_best");
            for ((c, got), frame) in cases.iter().zip(&parses).zip(&frames) {
                let want = gzc_core::reference::parse(&c.block, &c.best, &params);
                assert!(*got == want, "{}: K3 != reference parse; {}", c.name, first_diff(got, &want));
                if let Some((_, pinned)) = c.expect.iter().find(|(p, _)| *p == params) {
                    assert_eq!(got.sequences, *pinned, "{}: pinned sequences", c.name);
                }
                let want_frame = write_frame(&c.block, &want, kernels.frame_options());
                assert!(*frame == want_frame, "{}: frame; {}", c.name, first_byte_diff(frame, &want_frame));
            }
        };
        check(&cases);
        for c in &cases {
            check(std::slice::from_ref(c));
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

/// The streaming pipeline (several slots in flight, a partial last batch reusing a slot with
/// stale blocks, the pipeline reused for a second run) for every non-lvl3 preset.
#[test]
fn stream_frames_match_cpu_non_lvl3_presets() {
    let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
    let distinct = frame_blocks();
    for (name, matching) in GPU_PRESETS.into_iter().filter(|(n, _)| *n != "lvl3") {
        eprintln!("preset {name}");
        let params = GpuParams { matching, emit_frames: true, huffman: true };
        let want: Vec<Vec<u8>> =
            distinct.iter().map(|(_, b)| write_frame(b, &compress_block(b, matching), params.frame_options())).collect();
        // 100 = 6 * 16 + 4 blocks: 7 batches over 3 slots, the last one partial.
        let blocks: Vec<&[u8]> = (0..100).map(|i| distinct[i % distinct.len()].1.as_slice()).collect();
        let mut pipe = Pipeline::new(&ctx, &PipelineConfig { batch: 16, inflight: 3, params }).expect("Pipeline::new");
        for _ in 0..2 {
            let mut sink = CollectFrames(vec![None; blocks.len()]);
            let stats = pipe.run_frames(&blocks, &mut sink).expect("run_frames");
            assert_eq!(stats.batches, 7);
            for (i, got) in sink.0.iter().enumerate() {
                let got = got.as_ref().unwrap_or_else(|| panic!("{name}: index {i} never delivered"));
                let (bname, _) = &distinct[i % distinct.len()];
                assert!(*got == want[i % distinct.len()], "{name}: index {i} ({bname}): GPU frame != CPU frame");
            }
        }
    }
}

/// Informal (not in the normal suite; reads the real corpus): a few hundred real .dds/.nif blocks
/// per preset, GPU frames against CPU frames.
/// `GZC_CORPUS=/path/to/data/corpus cargo test --release -p gzc-gpu --test differential corpus_blocks -- --ignored --nocapture`
/// Takes up to `GZC_CORPUS_BLOCKS` (default 300) blocks, spread over the files in sorted path order;
/// `GZC_CORPUS_PRESETS` (comma-separated) limits the presets.
#[test]
#[ignore]
fn corpus_blocks_match_cpu_per_preset() {
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
    let want_blocks: usize = std::env::var("GZC_CORPUS_BLOCKS").map(|v| v.parse().unwrap()).unwrap_or(300);
    let mut files = Vec::new();
    walk(Path::new(&root), &mut files);
    files.sort();
    // Every k-th file, up to 4 blocks from each, so the sample spans the corpus.
    let step = (files.len() / want_blocks).max(1);
    let mut blocks: Vec<(String, Vec<u8>)> = Vec::new();
    for f in files.iter().step_by(step) {
        if blocks.len() >= want_blocks {
            break;
        }
        let bytes = std::fs::read(f).unwrap();
        for (i, b) in chunk_file(&bytes).into_iter().take(4).enumerate() {
            blocks.push((format!("{}[{i}]", f.display()), b.data));
        }
    }
    blocks.truncate(want_blocks);
    eprintln!("{} blocks from {} files", blocks.len(), files.len());
    // `GZC_CORPUS_PRESETS=lvl9seg,...` limits the run to those presets.
    let only = std::env::var("GZC_CORPUS_PRESETS").ok();
    for (name, params) in GPU_PRESETS {
        if only.as_ref().is_some_and(|o| !o.split(',').any(|p| p == name)) {
            continue;
        }
        let (ctx, kernels) = setup_frames_for(params, true);
        check_frames(&ctx, &kernels, &blocks);
        let total: usize = blocks
            .iter()
            .map(|(_, b)| write_frame(b, &compress_block(b, params), kernels.frame_options()).len())
            .sum();
        eprintln!("preset {name}: {} blocks equal, {} -> {total} bytes", blocks.len(), blocks.len() * BLOCK_SIZE);
    }
}
