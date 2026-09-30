//! Differential tests: the GPU parse (K1→K2→K3) must equal `reference::compress_block` exactly,
//! and the GPU frames (K4) must equal `write_frame` on that parse byte for byte.
use gzc_core::block::chunk_file;
use gzc_core::config::BLOCK_SIZE;
use gzc_core::codes::{LL_DEFAULT_LOG, LL_DEFAULT_NORM, ML_BASE};
use gzc_core::frame::{FrameOptions, frame_header, write_frame, write_literals_raw};
use gzc_core::fse::{choose_table_log, cost_x256, normalize, write_ncount};
use gzc_core::reference::{LVL3, RefParams, compress_block};
use gzc_core::seq::{BlockOutput, INITIAL_REPS, Sequence, apply_off_base, off_base_for, reconstruct};
use gzc_core::seqenc::{SeqMode, StreamKind, StreamTable, histograms, write_sequences_section_auto};
use gzc_core::synth::{random, test_cases, text, zeros};
use gzc_gpu::compressor::{GpuParams, Kernels, compress_batch, compress_frames, frames_from_parses};
use gzc_gpu::context::GpuContext;

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

fn setup(depth: u32) -> (GpuContext, Kernels) {
    let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
    let kernels = Kernels::new(&ctx, GpuParams { depth, emit_frames: false });
    (ctx, kernels)
}

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
fn check_batch(ctx: &GpuContext, kernels: &Kernels, blocks: &[(String, Vec<u8>)], params: RefParams) -> Vec<BlockOutput> {
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
    let (ctx, kernels) = setup(1);
    // All blocks in one batch (every block has neighbours on both sides)...
    check_batch(&ctx, &kernels, &all_blocks(), LVL3);
    // ...and each block alone, so its end is followed only by the trailing zero word.
    for b in all_blocks() {
        check_batch(&ctx, &kernels, std::slice::from_ref(&b), LVL3);
    }
}

#[test]
fn gpu_matches_reference_depth4() {
    let (ctx, kernels) = setup(4);
    let blocks: Vec<_> = all_blocks().into_iter().step_by(2).collect();
    check_batch(&ctx, &kernels, &blocks, RefParams { depth: 4, ..LVL3 });
}

/// Depth 4, where MATCH_SEARCH_CAP changes the choice: at 2000 the long chain offers q=1500
/// (80-byte match) then q=0 (200 bytes). Both cap to 64, so the tie goes to the closer q=1500,
/// and the parse extends that match to its full 80 bytes.
#[test]
fn capped_search_prefers_closer_candidate() {
    let (ctx, kernels) = setup(4);
    let mut block = random(20, BLOCK_SIZE);
    let s = random(21, 200);
    block[..200].copy_from_slice(&s);
    block[1500..1580].copy_from_slice(&s[..80]);
    block[2000..2200].copy_from_slice(&s);
    let blocks = [("capped".to_string(), block)];
    let params = RefParams { depth: 4, ..LVL3 };
    let out = check_batch(&ctx, &kernels, &blocks, params);
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
    let (ctx, kernels) = setup(1);
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
    check_batch(&ctx, &kernels, &blocks, LVL3);
}

#[test]
fn gpu_frames_roundtrip() {
    let (ctx, kernels) = setup(1);
    let blocks = all_blocks();
    let outs = check_batch(&ctx, &kernels, &blocks, LVL3);
    for ((name, block), out) in blocks.iter().zip(&outs) {
        let frame = write_frame(block, out, FrameOptions::default());
        let back = zstd::bulk::decompress(&frame, BLOCK_SIZE).unwrap_or_else(|e| panic!("{name}: libzstd: {e}"));
        assert!(back == *block, "{name}: libzstd output differs from input block");
    }
}

#[test]
fn batch_of_300_mixed() {
    let (ctx, kernels) = setup(1);
    let blocks: Vec<_> = all_blocks().into_iter().cycle().take(300).collect();
    check_batch(&ctx, &kernels, &blocks, LVL3);
}

#[test]
fn empty_batch_is_empty() {
    let (ctx, kernels) = setup(1);
    assert!(compress_batch(&ctx, &kernels, &[]).unwrap().is_empty());
}

// ---- K4: complete frames (sequence entropy + block assembly) ----

/// The CPU frame the GPU must reproduce byte for byte (raw literals until the GPU does Huffman).
const NO_HUFFMAN: FrameOptions = FrameOptions { checksum: false, huffman: false };

fn setup_frames() -> (GpuContext, Kernels) {
    let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
    let kernels = Kernels::new(&ctx, GpuParams { depth: 1, emit_frames: true });
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

/// GPU frames for `blocks` must equal `write_frame` on the CPU reference parse, and decode.
fn check_frames(ctx: &GpuContext, kernels: &Kernels, blocks: &[(String, Vec<u8>)]) {
    let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
    let frames = compress_frames(ctx, kernels, &refs).expect("compress_frames");
    assert_eq!(frames.len(), blocks.len());
    for (i, ((name, block), got)) in blocks.iter().zip(&frames).enumerate() {
        let want = write_frame(block, &compress_block(block, LVL3), NO_HUFFMAN);
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
    let (ctx, kernels) = setup_frames();
    let blocks = frame_blocks();
    // All blocks in one batch, then each block alone.
    check_frames(&ctx, &kernels, &blocks);
    for b in &blocks {
        check_frames(&ctx, &kernels, std::slice::from_ref(b));
    }
}

#[test]
fn gpu_frames_batch_of_300_mixed() {
    let (ctx, kernels) = setup_frames();
    let blocks: Vec<_> = frame_blocks().into_iter().cycle().take(300).collect();
    check_frames(&ctx, &kernels, &blocks);
}

#[test]
fn compress_frames_needs_emit_frames() {
    let (ctx, kernels) = setup(1);
    let block = zeros(BLOCK_SIZE);
    assert!(compress_frames(&ctx, &kernels, &[&block]).is_err());
    assert!(frames_from_parses(&ctx, &kernels, &[&block], &[BlockOutput::default()]).is_err());
}

// ---- K4 alone on scripted parses: exact ties and boundaries the match finder rarely produces ----

/// (block, parse) from (lit_len, offset, match_len) steps, literals drawn from `seed`; the rest of
/// the block is literals.
fn scripted(script: &[(u32, u32, u32)], seed: u64) -> (Vec<u8>, BlockOutput) {
    let mut r = Lcg(seed);
    let mut out = BlockOutput::default();
    let mut reps = INITIAL_REPS;
    let mut data: Vec<u8> = Vec::new();
    let mut lit = |out: &mut BlockOutput, data: &mut Vec<u8>| {
        let b = r.next() as u8;
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

/// K4 frames for scripted (block, parse) pairs must equal `write_frame` on the same parse, and decode.
fn check_scripted(ctx: &GpuContext, kernels: &Kernels, cases: &[(String, Vec<u8>, BlockOutput)]) {
    let blocks: Vec<&[u8]> = cases.iter().map(|c| c.1.as_slice()).collect();
    let parses: Vec<BlockOutput> = cases.iter().map(|c| c.2.clone()).collect();
    let frames = frames_from_parses(ctx, kernels, &blocks, &parses).expect("frames_from_parses");
    for ((name, block, parse), got) in cases.iter().zip(&frames) {
        assert_eq!(reconstruct(parse).unwrap(), *block, "{name}: bad script");
        let want = write_frame(block, parse, NO_HUFFMAN);
        assert!(*got == want, "{name}: GPU frame != CPU frame; {}", first_byte_diff(got, &want));
        let back = zstd::bulk::decompress(got, BLOCK_SIZE).unwrap_or_else(|e| panic!("{name}: libzstd: {e}"));
        assert!(back == *block, "{name}: libzstd output differs from input block");
    }
}

/// Random short scripts over small code palettes: frequent ties in the normalization and in the
/// predefined-vs-computed cost, one- and two-sequence streams, RLE streams, repeat offsets.
#[test]
fn k4_random_scripts_match_cpu() {
    let (ctx, kernels) = setup_frames();
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

/// Hand-built cases pinning each decision boundary of the mode choice and the block type.
#[test]
fn k4_decision_boundaries_match_cpu() {
    let (ctx, kernels) = setup_frames();
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
