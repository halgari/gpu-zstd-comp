//! CPU reference compressor used as the correctness and quality baseline.
//!
//! This is the bit-exact oracle the GPU kernels are tested against: a hash-chain match
//! finder and parse driven by runtime `MatchParams`, using integer arithmetic only so the
//! GPU mirrors it exactly. The `LVL3` preset is the M3 level-3-style greedy parse.
use crate::config::{BLOCK_SIZE, NO_POS, PARSE_END};
use crate::frame::{write_frame, FrameOptions};
use crate::hash::{compute_preds, hash_long, hash_short, hash_width};
use crate::params::{cpu_supports, Hashes, MatchParams};
use crate::seq::{apply_off_base, off_base_for, BlockOutput, Sequence, INITIAL_REPS};

pub use crate::params::LVL3;

/// The reference compressor's parameters are the shared `MatchParams`.
pub type RefParams = MatchParams;

/// A candidate match at some position: offset back from that position, and length.
/// `len == 0` means no match was found (or none met `min_match`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Match {
    pub offset: u32,
    pub len: u32,
}

/// Length of the common prefix of `block[p..]` and `block[q..]`, bounded so the
/// comparison never reads past `BLOCK_SIZE`.
pub fn match_len(block: &[u8], p: usize, q: usize) -> usize {
    let max = BLOCK_SIZE - p;
    let mut n = 0usize;
    while n < max && block[p + n] == block[q + n] {
        n += 1;
    }
    n
}

/// `match_len` that stops comparing at `cap` bytes:
/// `min(match_len(block, p, q), cap)` without the cost of the full compare.
pub fn match_len_capped(block: &[u8], p: usize, q: usize, cap: usize) -> usize {
    let max = (BLOCK_SIZE - p).min(cap);
    let mut n = 0usize;
    while n < max && block[p + n] == block[q + n] {
        n += 1;
    }
    n
}

/// The predecessor chains `find_best` walks, in walk order: for `Dfast`, the long-hash
/// (8-byte) chain then the short-hash (5-byte) chain; for `Single`, one chain over
/// `hash_width(.., min_match)`.
pub fn chains(block: &[u8], p: &MatchParams) -> Vec<Vec<u32>> {
    match p.hashes {
        Hashes::Dfast => vec![compute_preds(block, hash_long), compute_preds(block, hash_short)],
        Hashes::Single => {
            let min_match = p.min_match;
            vec![compute_preds(block, move |b: &[u8], pos: usize| hash_width(b, pos, min_match))]
        }
    }
}

/// For every `p < PARSE_END`, find the best match by walking up to `depth` candidates along
/// each chain in turn (Dfast: long, then short), keeping the candidate with the largest
/// length capped at `search_cap` (ties broken by the larger `q`, i.e. the more recent/closer
/// candidate). The stored length is the capped one; the parse extends matches that hit the
/// cap. Only lengths `>= min_match` are stored. Positions `p >= PARSE_END` are left default.
pub fn find_best(block: &[u8], chains: &[Vec<u32>], params: &MatchParams) -> Vec<Match> {
    let mut best = vec![Match::default(); BLOCK_SIZE];
    let (depth, cap, min_match) = (params.depth, params.search_cap as usize, params.min_match as usize);
    for p in 0..PARSE_END {
        let mut best_len = 0usize;
        let mut best_q = 0usize;
        for preds in chains {
            let mut q = preds[p];
            for _ in 0..depth {
                if q == NO_POS {
                    break;
                }
                let qu = q as usize;
                let len = match_len_capped(block, p, qu, cap);
                if len > best_len || (len == best_len && qu > best_q) {
                    best_len = len;
                    best_q = qu;
                }
                q = preds[qu];
            }
        }
        if best_len >= min_match {
            best[p] = Match { offset: (p - best_q) as u32, len: best_len as u32 };
        }
    }
    best
}

/// Parse a block from its best matches: the greedy parse for `lazy == 0` (the lazy parse
/// arrives in Task 4).
pub fn parse(block: &[u8], best: &[Match], p: &MatchParams) -> BlockOutput {
    assert!(p.lazy == 0, "reference::parse: lazy {} is not implemented yet", p.lazy);
    greedy_parse(block, best, p)
}

/// Greedy parse: at each position, prefer a repeat-offset match (rep0) over the best
/// hash-chain match (extended to its full length when `find_best` capped it);
/// otherwise skip ahead with a mild acceleration as literal runs grow.
/// Trailing bytes from the final anchor to `BLOCK_SIZE` become the last literal run.
pub fn greedy_parse(block: &[u8], best: &[Match], params: &MatchParams) -> BlockOutput {
    let (min_match, cap) = (params.min_match as usize, params.search_cap as usize);
    let mut sequences = Vec::new();
    let mut literals = Vec::new();
    let mut p = 0usize;
    let mut anchor = 0usize;
    let mut reps = INITIAL_REPS;

    macro_rules! emit {
        ($off:expr, $len:expr) => {{
            let ll = (p - anchor) as u32;
            let ob = off_base_for($off, ll, &reps);
            apply_off_base(&mut reps, ob, ll);
            literals.extend_from_slice(&block[anchor..p]);
            sequences.push(Sequence { lit_len: ll, match_len: $len as u32, off_base: ob });
            p += $len;
            anchor = p;
        }};
    }

    while p < PARSE_END {
        if p > anchor && p >= reps[0] as usize {
            let l = match_len(block, p, p - reps[0] as usize);
            if l >= min_match {
                emit!(reps[0], l);
                continue;
            }
        }
        if best[p].len as usize >= min_match {
            let m = best[p];
            let len = if m.len as usize == cap {
                match_len(block, p, p - m.offset as usize)
            } else {
                m.len as usize
            };
            emit!(m.offset, len);
            continue;
        }
        p += 1 + ((p - anchor) >> 8);
    }
    literals.extend_from_slice(&block[anchor..BLOCK_SIZE]);
    BlockOutput { sequences, literals }
}

/// Compress one full-size block: compute the hash chains, find best matches, parse.
/// Panics if `params` is invalid or not implemented on the CPU yet (`params::cpu_supports`).
pub fn compress_block(block: &[u8], params: MatchParams) -> BlockOutput {
    assert_eq!(block.len(), BLOCK_SIZE);
    if let Err(e) = params.validate() {
        panic!("reference::compress_block: invalid params {params:?}: {e}");
    }
    assert!(cpu_supports(&params), "reference::compress_block: {params:?} is not implemented yet on cpu");
    let chains = chains(block, &params);
    let best = find_best(block, &chains, &params);
    parse(block, &best, &params)
}

/// Compress one full-size block straight to a zstd frame: `compress_block` followed
/// by `write_frame`. This is the CPU reference compressor's end-to-end entry point,
/// used by `gzc-bench ref` and exercised by `all_synthetic_frames_roundtrip` below.
pub fn compress_block_to_frame(block: &[u8], params: MatchParams, opts: FrameOptions) -> Vec<u8> {
    let out = compress_block(block, params);
    write_frame(block, &out, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::chunk_file;
    use crate::params::RUNG1;
    use crate::seq::reconstruct;
    use crate::synth;

    #[test]
    fn roundtrips_all_test_cases() {
        for (name, bytes) in synth::test_cases() {
            for (i, blk) in chunk_file(&bytes).into_iter().enumerate() {
                let out = compress_block(&blk.data, LVL3);
                let got = reconstruct(&out).unwrap_or_else(|e| panic!("{name} block {i}: {e}"));
                assert_eq!(got, blk.data, "{name} block {i}: roundtrip mismatch");
            }
        }
    }

    #[test]
    fn best_matches_are_real() {
        for (_, bytes) in synth::test_cases() {
            for blk in chunk_file(&bytes) {
                let block = &blk.data;
                let best = find_best(block, &chains(block, &LVL3), &LVL3);
                for (p, m) in best.iter().enumerate() {
                    if m.len == 0 {
                        continue;
                    }
                    let len = m.len as usize;
                    assert!(len >= LVL3.min_match as usize, "p={p} len={len} below min_match");
                    assert!(len <= LVL3.search_cap as usize, "p={p} len={len} above search_cap");
                    assert!(p + len <= BLOCK_SIZE, "p={p} len={len} runs past block end");
                    let q = p - m.offset as usize;
                    assert_eq!(&block[p..p + len], &block[q..q + len], "p={p} offset={} len={len} not a real match", m.offset);
                }
            }
        }
    }

    #[test]
    fn match_runs_to_block_end() {
        let (_, bytes) = synth::test_cases().into_iter().find(|(n, _)| *n == "period3").unwrap();
        assert_eq!(bytes.len(), BLOCK_SIZE);
        let out = compress_block(&bytes, LVL3);

        let mut pos: u32 = 0;
        let mut ends_at_block = false;
        for s in &out.sequences {
            pos += s.lit_len + s.match_len;
            if pos as usize == BLOCK_SIZE {
                ends_at_block = true;
            }
        }
        let trailing = BLOCK_SIZE - pos as usize;
        assert!(ends_at_block || trailing < 8, "no match reaches block end, and trailing literal run is {trailing} bytes");
    }

    #[test]
    fn zeros_is_one_long_match_found_fast() {
        let block = synth::zeros(BLOCK_SIZE);
        let chains = chains(&block, &LVL3);
        let t = std::time::Instant::now();
        let best = find_best(&block, &chains, &LVL3);
        let elapsed = t.elapsed();
        assert!(elapsed < std::time::Duration::from_secs(1), "find_best on zeros took {elapsed:?}");
        assert_eq!(best[1], Match { offset: 1, len: LVL3.search_cap });

        // One literal, then one (rep0) match to the block end.
        let out = greedy_parse(&block, &best, &LVL3);
        assert_eq!(out.sequences, vec![Sequence { lit_len: 1, match_len: BLOCK_SIZE as u32 - 1, off_base: 1 }]);
        assert_eq!(out.literals, vec![0]);
    }

    #[test]
    fn parse_extends_capped_best_match() {
        // period3: no repcode matches, so p=3 takes best[3] (offset 3, len capped) and must
        // extend it to the block end.
        let (_, block) = synth::test_cases().into_iter().find(|(n, _)| *n == "period3").unwrap();
        let best = find_best(&block, &chains(&block, &LVL3), &LVL3);
        assert_eq!(best[3], Match { offset: 3, len: LVL3.search_cap });
        let out = greedy_parse(&block, &best, &LVL3);
        assert_eq!(out.sequences, vec![Sequence { lit_len: 3, match_len: BLOCK_SIZE as u32 - 3, off_base: 3 + 3 }]);
        assert_eq!(out.literals, vec![1, 2, 3]);
    }

    #[test]
    fn text_compresses() {
        let bytes = synth::text(42, BLOCK_SIZE);
        let out = compress_block(&bytes, LVL3);
        assert!(out.literals.len() < BLOCK_SIZE / 2, "literals {} not < half block", out.literals.len());
    }

    #[test]
    fn random_has_few_sequences() {
        let bytes = synth::random(42, BLOCK_SIZE);
        let out = compress_block(&bytes, LVL3);
        assert!(out.sequences.len() < 50, "sequences {} not < 50", out.sequences.len());
    }

    #[test]
    fn strided_uses_repcodes() {
        let (_, bytes) = synth::test_cases().into_iter().find(|(n, _)| *n == "strided").unwrap();
        assert_eq!(bytes.len(), BLOCK_SIZE);
        let out = compress_block(&bytes, LVL3);
        assert!(out.sequences.iter().any(|s| s.off_base <= 3), "no repeat-offset sequences found");
    }

    #[test]
    fn deterministic() {
        let bytes = synth::dds_like(7, BLOCK_SIZE);
        let a = compress_block(&bytes, LVL3);
        let b = compress_block(&bytes, LVL3);
        assert_eq!(a, b);
    }

    #[test]
    fn all_synthetic_frames_roundtrip() {
        for (name, bytes) in synth::test_cases() {
            for (i, blk) in chunk_file(&bytes).into_iter().enumerate() {
                let frame = compress_block_to_frame(&blk.data, LVL3, FrameOptions::default());
                let dec = zstd::bulk::decompress(&frame, BLOCK_SIZE)
                    .unwrap_or_else(|e| panic!("{name} block {i}: libzstd rejected frame: {e}"));
                assert_eq!(dec, blk.data, "{name} block {i}: roundtrip mismatch");
            }
        }
    }

    #[test]
    #[should_panic(expected = "not implemented yet on cpu")]
    fn unsupported_params_panic_clearly() {
        compress_block(&synth::zeros(BLOCK_SIZE), crate::params::LVL9);
    }

    /// xxh64 of the concatenated lvl3 frames, captured on the unmodified M3 code (ddeee75).
    #[cfg(feature = "block-128k")]
    const M3_LVL3_ANCHOR: u64 = 0xc8ea1f5b1315917e;
    #[cfg(feature = "block-64k")]
    const M3_LVL3_ANCHOR: u64 = 0x2e488c60ec5d4e73;
    #[cfg(feature = "block-32k")]
    const M3_LVL3_ANCHOR: u64 = 0x0eab410abe5b93ef;
    #[cfg(feature = "block-16k")]
    const M3_LVL3_ANCHOR: u64 = 0x8ac4e7dead6b2d84;

    /// Pins the M3 lvl3 output: xxh64 over every synthetic test case's frames, concatenated.
    /// Any byte change in the lvl3 CPU path (and hence the GPU path, which must match it) fails here.
    #[test]
    fn lvl3_frames_match_m3_anchor() {
        let mut all = Vec::new();
        for (_, bytes) in synth::test_cases() {
            for blk in chunk_file(&bytes) {
                all.extend_from_slice(&compress_block_to_frame(&blk.data, LVL3, FrameOptions::default()));
            }
        }
        let h = xxhash_rust::xxh64::xxh64(&all, 0);
        println!("lvl3 anchor: {h:#018x} ({} bytes)", all.len());
        assert_eq!(h, M3_LVL3_ANCHOR, "lvl3 frames changed from the M3 anchor");
    }

    /// `rung1` (Single chain, min_match 4, greedy parse): every synthetic test case
    /// round-trips both through `reconstruct` and through a libzstd frame decode.
    #[test]
    fn rung1_roundtrips_all_cases() {
        for (name, bytes) in synth::test_cases() {
            for (i, blk) in chunk_file(&bytes).into_iter().enumerate() {
                let out = compress_block(&blk.data, RUNG1);
                let got = reconstruct(&out).unwrap_or_else(|e| panic!("{name} block {i}: {e}"));
                assert_eq!(got, blk.data, "{name} block {i}: reconstruct roundtrip mismatch");

                let frame = compress_block_to_frame(&blk.data, RUNG1, FrameOptions::default());
                let dec = zstd::bulk::decompress(&frame, BLOCK_SIZE)
                    .unwrap_or_else(|e| panic!("{name} block {i}: libzstd rejected frame: {e}"));
                assert_eq!(dec, blk.data, "{name} block {i}: frame roundtrip mismatch");
            }
        }
    }

    /// A block of 5-byte tokens ("AAAA" + a byte cycling 0..255) has no 5-byte repeats
    /// (the 5th byte of any two adjacent tokens always differs), but every 4-byte prefix
    /// repeats. `rung1`'s Single hash sees only `min_match` (4) bytes, so it must find
    /// these matches, each capped at exactly 4 bytes by the differing 5th byte.
    #[test]
    fn rung1_finds_4_byte_matches() {
        let mut block = vec![0u8; BLOCK_SIZE];
        let ntoken = BLOCK_SIZE / 5;
        for i in 0..ntoken {
            let p = i * 5;
            block[p..p + 4].copy_from_slice(b"AAAA");
            block[p + 4] = (i % 256) as u8;
        }
        let out = compress_block(&block, RUNG1);
        assert!(out.sequences.iter().any(|s| s.match_len == 4), "no length-4 sequence found");
    }

    /// Every token below shares the same 4-byte hash ("ABCD" ignores the 5th byte), so the
    /// Single chain links every earlier token, nearest first. Only the 3rd-nearest token
    /// (`q3`) shares a real 20-byte run with `p`; the two nearer ones (`q1`, `q2`) diverge
    /// right after the 4-byte prefix, so `find_best` only reaches `q3`'s match at `depth >= 3`.
    #[test]
    fn rung1_depth_reaches_older_candidates() {
        const W: usize = 30;
        const N: usize = 10;
        let long_match: &[u8; 20] = b"ABCDEFGHIJKLMNOPQRST";

        let mut block = vec![0u8; BLOCK_SIZE];
        for i in 0..=N {
            let p = i * W;
            if i == N || i == N - 3 {
                block[p..p + long_match.len()].copy_from_slice(long_match);
                // Diverge right after the shared run so the match can't run past 20 bytes.
                block[p + long_match.len()] = i as u8;
            } else {
                block[p..p + 4].copy_from_slice(b"ABCD");
                block[p + 4] = b'Z';
            }
        }
        let p = N * W;

        let depth1 = MatchParams { depth: 1, ..RUNG1 };
        let best1 = find_best(&block, &chains(&block, &depth1), &depth1);
        assert_eq!(best1[p], Match { offset: W as u32, len: 4 }, "depth 1 should only reach q1's 4-byte match");

        let best8 = find_best(&block, &chains(&block, &RUNG1), &RUNG1);
        assert_eq!(best8[p], Match { offset: (3 * W) as u32, len: 20 }, "depth 8 should reach q3's 20-byte match");
    }
}
