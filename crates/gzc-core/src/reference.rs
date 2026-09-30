//! CPU reference compressor used as the correctness and quality baseline.
//!
//! This is the bit-exact oracle the GPU kernels are tested against: a level-3-style
//! greedy parse over a hash-chain match finder, using integer arithmetic only so the
//! GPU mirrors it exactly.
use crate::config::{BLOCK_SIZE, MATCH_SEARCH_CAP, MIN_MATCH, NO_POS, PARSE_END};
use crate::hash::{compute_preds, hash_long, hash_short};
use crate::seq::{apply_off_base, off_base_for, BlockOutput, Sequence, INITIAL_REPS};

/// Match-finder tuning: minimum match length to accept, and hash-chain search depth.
#[derive(Clone, Copy, Debug)]
pub struct RefParams {
    pub min_match: usize,
    pub depth: usize,
}

/// Level-3 calibration: default min match, depth-1 chain search.
pub const LVL3: RefParams = RefParams { min_match: MIN_MATCH, depth: 1 };

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

/// `match_len` that stops comparing at `MATCH_SEARCH_CAP` bytes:
/// `min(match_len(block, p, q), MATCH_SEARCH_CAP)` without the cost of the full compare.
pub fn match_len_capped(block: &[u8], p: usize, q: usize) -> usize {
    let max = (BLOCK_SIZE - p).min(MATCH_SEARCH_CAP);
    let mut n = 0usize;
    while n < max && block[p + n] == block[q + n] {
        n += 1;
    }
    n
}

/// For every `p < PARSE_END`, find the best match by walking up to `depth` candidates
/// from the long-hash chain, then up to `depth` candidates from the short-hash chain,
/// keeping the candidate with the largest length capped at `MATCH_SEARCH_CAP` (ties broken
/// by the larger `q`, i.e. the more recent/closer candidate). The stored length is the capped
/// one; `greedy_parse` extends matches that hit the cap. Positions `p >= PARSE_END` are left
/// default.
pub fn find_best(block: &[u8], pred_long: &[u32], pred_short: &[u32], params: RefParams) -> Vec<Match> {
    let mut best = vec![Match::default(); BLOCK_SIZE];
    for p in 0..PARSE_END {
        let mut best_len = 0usize;
        let mut best_q = 0usize;
        for preds in [pred_long, pred_short] {
            let mut q = preds[p];
            for _ in 0..params.depth {
                if q == NO_POS {
                    break;
                }
                let qu = q as usize;
                let len = match_len_capped(block, p, qu);
                if len > best_len || (len == best_len && qu > best_q) {
                    best_len = len;
                    best_q = qu;
                }
                q = preds[qu];
            }
        }
        if best_len >= params.min_match {
            best[p] = Match { offset: (p - best_q) as u32, len: best_len as u32 };
        }
    }
    best
}

/// Greedy parse: at each position, prefer a repeat-offset match (rep0) over the best
/// hash-chain match (extended to its full length when `find_best` capped it);
/// otherwise skip ahead with a mild acceleration as literal runs grow.
/// Trailing bytes from the final anchor to `BLOCK_SIZE` become the last literal run.
pub fn greedy_parse(block: &[u8], best: &[Match], params: RefParams) -> BlockOutput {
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
            if l >= params.min_match {
                emit!(reps[0], l);
                continue;
            }
        }
        if best[p].len as usize >= params.min_match {
            let m = best[p];
            let len = if m.len as usize == MATCH_SEARCH_CAP {
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

/// Compress one full-size block: compute both hash chains, find best matches, greedy-parse.
pub fn compress_block(block: &[u8], params: RefParams) -> BlockOutput {
    assert_eq!(block.len(), BLOCK_SIZE);
    let pred_long = compute_preds(block, hash_long);
    let pred_short = compute_preds(block, hash_short);
    let best = find_best(block, &pred_long, &pred_short, params);
    greedy_parse(block, &best, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::chunk_file;
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
                let pred_long = compute_preds(block, hash_long);
                let pred_short = compute_preds(block, hash_short);
                let best = find_best(block, &pred_long, &pred_short, LVL3);
                for (p, m) in best.iter().enumerate() {
                    if m.len == 0 {
                        continue;
                    }
                    let len = m.len as usize;
                    assert!(len >= MIN_MATCH, "p={p} len={len} below min_match");
                    assert!(len <= MATCH_SEARCH_CAP, "p={p} len={len} above MATCH_SEARCH_CAP");
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
        let pred_long = compute_preds(&block, hash_long);
        let pred_short = compute_preds(&block, hash_short);
        let t = std::time::Instant::now();
        let best = find_best(&block, &pred_long, &pred_short, LVL3);
        let elapsed = t.elapsed();
        assert!(elapsed < std::time::Duration::from_secs(1), "find_best on zeros took {elapsed:?}");
        assert_eq!(best[1], Match { offset: 1, len: MATCH_SEARCH_CAP as u32 });

        // One literal, then one (rep0) match to the block end.
        let out = greedy_parse(&block, &best, LVL3);
        assert_eq!(out.sequences, vec![Sequence { lit_len: 1, match_len: BLOCK_SIZE as u32 - 1, off_base: 1 }]);
        assert_eq!(out.literals, vec![0]);
    }

    #[test]
    fn parse_extends_capped_best_match() {
        // period3: no repcode matches, so p=3 takes best[3] (offset 3, len capped) and must
        // extend it to the block end.
        let (_, block) = synth::test_cases().into_iter().find(|(n, _)| *n == "period3").unwrap();
        let pred_long = compute_preds(&block, hash_long);
        let pred_short = compute_preds(&block, hash_short);
        let best = find_best(&block, &pred_long, &pred_short, LVL3);
        assert_eq!(best[3], Match { offset: 3, len: MATCH_SEARCH_CAP as u32 });
        let out = greedy_parse(&block, &best, LVL3);
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
}
