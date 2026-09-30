//! K1→K2→K3 differential test: the GPU parse must equal `reference::compress_block` exactly.
use gzc_core::block::chunk_file;
use gzc_core::config::BLOCK_SIZE;
use gzc_core::frame::{FrameOptions, write_frame};
use gzc_core::reference::{LVL3, RefParams, compress_block};
use gzc_core::seq::BlockOutput;
use gzc_core::synth::{random, test_cases, text, zeros};
use gzc_gpu::compressor::{GpuParams, Kernels, compress_batch};
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
    let kernels = Kernels::new(&ctx, GpuParams { depth });
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
