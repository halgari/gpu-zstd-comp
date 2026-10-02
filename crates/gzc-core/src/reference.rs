//! CPU reference compressor used as the correctness and quality baseline.
//!
//! This is the bit-exact oracle the GPU kernels are tested against: a hash-chain match
//! finder and parse driven by runtime `MatchParams`, using integer arithmetic only so the
//! GPU mirrors it exactly. The `LVL3` preset is the M3 level-3-style greedy parse.
use crate::config::{BLOCK_SIZE, HASH_BITS, NO_POS, PARSE_END};
use crate::frame::{write_frame, FrameOptions};
use crate::hash::{compute_preds, hash3, hash_long, hash_short, hash_sparse, hash_width, key};
use crate::lazy::{lazy_parse, lazy_parse_segmented};
use crate::params::{Hashes, MatchParams, SparseChain, OPT_H3_DEPTH};
use crate::seq::{apply_off_base, off_base_for, BlockOutput, Sequence, INITIAL_REPS};

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
/// `hash_width(.., min_match)`; for `Opt3`, the 4-byte chain (`hash_width(.., 4)`), the
/// 3-byte chain (`hash3`), then one chain per `OptParams::sparse_chains` entry, in order
/// (`sparse_chain_preds`).
/// Each chain links equal *keys*: the hash's top `hash_bits` bits (`hash::key`; all 16 bits for
/// every chain preset).
pub fn chains(block: &[u8], p: &MatchParams) -> Vec<Vec<u32>> {
    let hb = p.hash_bits;
    match p.hashes {
        Hashes::Dfast => vec![
            compute_preds(block, move |b: &[u8], pos: usize| key(hash_long(b, pos), hb)),
            compute_preds(block, move |b: &[u8], pos: usize| key(hash_short(b, pos), hb)),
        ],
        Hashes::Single => {
            let min_match = p.min_match;
            vec![compute_preds(block, move |b: &[u8], pos: usize| key(hash_width(b, pos, min_match), hb))]
        }
        Hashes::Opt3 => {
            let mut v = vec![
                compute_preds(block, move |b: &[u8], pos: usize| key(hash_width(b, pos, 4), hb)),
                compute_preds(block, move |b: &[u8], pos: usize| key(hash3(b, pos), hb)),
            ];
            for c in p.opt.iter().flat_map(|o| o.sparse_chains.into_iter().flatten()) {
                v.push(sparse_chain_preds(block, &c));
            }
            v
        }
    }
}

/// Sparse chains hash positions `p < SPARSE_END` only (`hash::hash_sparse` reads up to 12 bytes).
pub const SPARSE_END: usize = BLOCK_SIZE - 12;

/// The predecessor chain of a sparse chain `c` (`params::SparseChain`): for each *sparse
/// position* `p` (`p % c.stride == 0 && p < SPARSE_END`) in increasing order,
/// `pred[p]` = the previous sparse position with the same `hash_sparse(block, p, c.width)`
/// (16 bits), else `NO_POS`. Every other position is `NO_POS`, so `pred` links sparse positions
/// only and `find_cands` walks the chain only from sparse positions.
pub fn sparse_chain_preds(block: &[u8], c: &SparseChain) -> Vec<u32> {
    assert_eq!(block.len(), BLOCK_SIZE);
    let mut head = vec![NO_POS; 1 << HASH_BITS];
    let mut pred = vec![NO_POS; BLOCK_SIZE];
    for p in (0..SPARSE_END).step_by(c.stride as usize) {
        let h = hash_sparse(block, p, c.width) as usize;
        pred[p] = head[h];
        head[h] = p as u32;
    }
    pred
}

/// One `find_cands` record: offset back from the position and capped length (`len == 0`: none).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cand {
    pub offset: u32,
    pub len: u32,
}

/// The two candidate words of a position (K2opt's output, 8 B per position):
/// `w[0] = offA | lenA << 16 | lenB << 24`, `w[1] = offB`, plus `DEAD_BIT` at a dead position
/// (`find_cands`). All zero when there is no candidate, apart from `DEAD_BIT`.
pub type CandWords = [u32; 2];

/// `w[1]`'s flag of a dead position (M6 A3, see `find_cands`); offsets are below 2^16.
pub const DEAD_BIT: u32 = 1 << 16;

/// Whether a position's candidate words mark it dead (`find_cands`).
pub fn is_dead(w: CandWords) -> bool {
    w[1] & DEAD_BIT != 0
}

/// Packs records `a` (nearest) and `b` (longest) into `CandWords`.
pub fn pack_cands(a: Cand, b: Cand) -> CandWords {
    debug_assert!(a.offset < 1 << 16 && b.offset < 1 << 16 && a.len < 256 && b.len < 256);
    [a.offset | a.len << 16 | b.len << 24, b.offset]
}

/// Unpacks `CandWords` into `(A, B)`.
pub fn unpack_cands(w: CandWords) -> (Cand, Cand) {
    (Cand { offset: w[0] & 0xFFFF, len: (w[0] >> 16) & 0xFF }, Cand { offset: w[1] & 0xFFFF, len: w[0] >> 24 })
}

/// Walk depth of each `Opt3` chain, in `chains` order: `h4` `params.depth`, `h3`
/// `OPT_H3_DEPTH`, then each sparse chain's `depth`.
pub fn cand_depths(params: &MatchParams) -> Vec<u32> {
    let mut d = vec![params.depth, OPT_H3_DEPTH];
    d.extend(params.opt.iter().flat_map(|o| o.sparse_chains.into_iter().flatten()).map(|c| c.depth));
    d
}

/// K2opt (m5-opt-design §2.1; M6 sparse chains), the optimal parse's candidates.
///
/// For every `p < PARSE_END`, the *visit list* of `p` is the union, sorted by position
/// descending (nearest first) with duplicates removed, of the first `depth_i` entries of each
/// `Opt3` chain `i` from `p` (`chain_i[p]`, `chain_i[chain_i[p]]`, ..., stopping at `NO_POS`):
/// chain 0 `h4` (`params.depth` deep), chain 1 `h3` (`OPT_H3_DEPTH`), chains 2.. the sparse
/// chains (`OptParams::sparse_chains`, each its own `depth`; `cand_depths`). A sparse chain
/// contributes only when `p` is a sparse position (`p % stride == 0 && p < SPARSE_END`):
/// elsewhere `chain_i[p] = NO_POS`. Each chain is strictly decreasing and a position on several
/// chains is visited once, but counts against each chain's depth.
///
/// Each visited `q` gets the capped length `c = match_len_capped(block, p, q, search_cap)`; a
/// *record* is a `q` whose `c` strictly beats every earlier `c` and the floor 2 (so records start
/// at length 3, and on equal lengths the nearer `q` wins). `A` = the first record (the nearest
/// match of at least 3 bytes), `B` = the last (longest); with one record `B == A`. The walk may
/// stop once `c` reaches `min(search_cap, BLOCK_SIZE - p)` (no later `q` can beat it). Only the
/// visit order matters, so any merge that visits the union nearest first gives the same words.
/// Lengths are capped: a stored 64 (`search_cap`) means "at least 64", extended by the parse.
/// Positions `>= PARSE_END` are zero. Requires `Opt3` chains (`chains(block, params)`).
///
/// This implementation merges the chains in one walk: the next `q` is the largest live head (a
/// chain is live while it has depth left and its head is not `NO_POS`); every live chain whose
/// head equals `q` advances to `chain_i[q]` and spends one depth.
///
/// Fingerprint caveat (for GPU filters): an `h4`-chain entry whose first 4 bytes differ from
/// `p`'s (a 16-bit hash collision) can still share 3 bytes and is then a valid 3-byte record, so
/// a 4-byte fingerprint mismatch may skip the compare only once `best >= 3` (or when the first 3
/// bytes differ too). The same holds for sparse-chain collisions: a sparse entry is a
/// candidate like any other, whatever its hashed bytes.
///
/// Dead positions (M6 A3, a09): `p` is *dead* when it has no record and its `h3` walk reached
/// the chain's end (`NO_POS`, within `OPT_H3_DEPTH` steps). Then no earlier position shares `p`'s
/// first 3 bytes: every such position has `p`'s `hash3` key, so it is on `p`'s `h3` chain (which
/// links every earlier position below `HASHED_POSITIONS` with that key), and the walk visited the
/// whole chain without a 3-byte match. So no candidate and no rep (whatever its offset) reaches 3
/// bytes there, and the parse's `get_all_matches` is empty under every rep state. A dead
/// position's words are `[0, DEAD_BIT]`. The parse oracle ignores the bit (`unpack_cands`);
/// K3opt skips dead positions. The test reads the `h3` walk only, so the sparse chains (which
/// cannot add a record at a dead position) do not change it.
pub fn find_cands(block: &[u8], chains: &[Vec<u32>], params: &MatchParams) -> Vec<CandWords> {
    assert_eq!(params.hashes, Hashes::Opt3, "find_cands: Opt3 chains only");
    let depths = cand_depths(params);
    assert_eq!(chains.len(), depths.len());
    let k = chains.len();
    let cap = params.search_cap as usize;
    let mut out = vec![[0u32; 2]; BLOCK_SIZE];
    let mut heads = vec![NO_POS; k];
    let mut left = vec![0u32; k];
    for p in 0..PARSE_END {
        let max_c = cap.min(BLOCK_SIZE - p);
        for i in 0..k {
            heads[i] = chains[i][p];
            left[i] = depths[i];
        }
        let mut best = 2usize;
        let (mut a, mut b) = (Cand::default(), Cand::default());
        loop {
            // Next position of the merged walk: the largest live head; equal heads all advance.
            let mut q = NO_POS;
            for i in 0..k {
                if left[i] > 0 && heads[i] != NO_POS && (q == NO_POS || heads[i] > q) {
                    q = heads[i];
                }
            }
            if q == NO_POS {
                break;
            }
            for i in 0..k {
                if left[i] > 0 && heads[i] == q {
                    heads[i] = chains[i][q as usize];
                    left[i] -= 1;
                }
            }
            let qu = q as usize;
            let c = match_len_capped(block, p, qu, cap);
            if c > best {
                best = c;
                b = Cand { offset: (p - qu) as u32, len: c as u32 };
                if a.len == 0 {
                    a = b;
                }
                if c == max_c {
                    break;
                }
            }
        }
        out[p] = pack_cands(a, b);
        // chain 1 is h3: its head is NO_POS iff the walk reached its end
        if a.len == 0 && heads[1] == NO_POS {
            out[p][1] |= DEAD_BIT;
        }
    }
    out
}

/// `find_best` over the bucket-sorted candidate array (`hash::bucket_sort` of the Single chain's
/// keys), the way the GPU's window K2 (`k2_window.wgsl`) walks it: for `p < PARSE_END` with slot
/// `s = rank[p]`, visit the `min(depth, s)` entries `sorted[s - 1], sorted[s - 2], ..` (nearest
/// first) and keep the largest (capped len, q), exactly as `find_best` does along a chain.
///
/// Equal to `find_best(block, &chains(block, params), params)`: inside p's bucket the entries
/// before slot s are p's chain in chain order (same key, positions descending), so the first
/// `min(depth, i)` of the window (i = p's index in its bucket) are the chain's first `depth`
/// candidates. The window's remaining entries (only when i < depth) belong to other buckets: a
/// different key means different first 4 bytes (the key hashes only bytes p..p+min_match; for
/// min_match > 4, different first min_match bytes), so their len < min_match and they cannot
/// win, nor stop the walk at the cap. Entries with q >= p (other buckets only) are skipped.
/// Panics unless `params.hashes` is `Single`.
pub fn find_best_window(block: &[u8], sorted: &[u32], rank: &[u32], params: &MatchParams) -> Vec<Match> {
    assert_eq!(params.hashes, Hashes::Single, "find_best_window: Single hash only");
    let mut best = vec![Match::default(); BLOCK_SIZE];
    let (depth, cap, min_match) = (params.depth as usize, params.search_cap as usize, params.min_match as usize);
    for p in 0..PARSE_END {
        let s = rank[p] as usize;
        let mut best_len = 0usize;
        let mut best_q = 0usize;
        for j in 1..=depth.min(s) {
            let qu = sorted[s - j] as usize;
            if qu >= p {
                continue;
            }
            let len = match_len_capped(block, p, qu, cap);
            if len > best_len || (len == best_len && qu > best_q) {
                best_len = len;
                best_q = qu;
                if len == cap {
                    break;
                }
            }
        }
        if best_len >= min_match {
            best[p] = Match { offset: (p - best_q) as u32, len: best_len as u32 };
        }
    }
    best
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

/// Parse a block from its best matches: the greedy parse for `lazy == 0`, otherwise the
/// libzstd lazy/lazy2 port (`lazy::lazy_parse`, or `lazy::lazy_parse_segmented` with
/// `segment_log2 > 0`). The optimal parse (`p.opt`) takes `find_cands` words instead: see
/// `parse_cands`; `compress_block` dispatches between the two.
pub fn parse(block: &[u8], best: &[Match], p: &MatchParams) -> BlockOutput {
    assert!(p.opt.is_none(), "reference::parse: opt params parse find_cands words (parse_cands)");
    if p.lazy == 0 {
        greedy_parse(block, best, p)
    } else if p.segment_log2 > 0 {
        lazy_parse_segmented(block, best, p)
    } else {
        lazy_parse(block, best, p)
    }
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

/// The optimal parse (`opt::parse`) of a block from its `find_cands` words. `p.opt` must be set.
pub fn parse_cands(block: &[u8], cands: &[CandWords], p: &MatchParams) -> BlockOutput {
    crate::opt::parse(block, cands, p)
}

/// Compress one full-size block: compute the hash chains, find best matches (or the optimal
/// parse's candidates), parse.
/// Panics if `params` is invalid (`MatchParams::validate`).
pub fn compress_block(block: &[u8], params: MatchParams) -> BlockOutput {
    assert_eq!(block.len(), BLOCK_SIZE);
    if let Err(e) = params.validate() {
        panic!("reference::compress_block: invalid params {params:?}: {e}");
    }
    let chains = chains(block, &params);
    if params.opt.is_some() {
        return parse_cands(block, &find_cands(block, &chains, &params), &params);
    }
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
    use crate::params::{LVL3, RUNG1};
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
    #[should_panic(expected = "invalid params")]
    fn invalid_params_panic_clearly() {
        compress_block(&synth::zeros(BLOCK_SIZE), MatchParams { lazy: 3, ..crate::params::LVL9 });
    }

    /// The window walk over the bucket-sorted array equals the chain walk, for 11/12/13-bit keys and
    /// the 16-bit one, at depths 4, 16 and 32 (shallow walks rarely reach other buckets).
    #[test]
    fn window_walk_equals_chain_walk() {
        use crate::params::{LVL9, LVL9S12, LVL9S12D16SEG};
        let s13 = MatchParams { hash_bits: 13, ..LVL9 };
        for params in [LVL9S12, LVL9S12D16SEG, s13, LVL9, MatchParams { depth: 4, ..s13 }, MatchParams { hash_bits: 11, ..LVL9 }] {
            for (name, bytes) in synth::test_cases() {
                for (i, blk) in chunk_file(&bytes).into_iter().enumerate() {
                    let block = &blk.data;
                    let want = find_best(block, &chains(block, &params), &params);
                    let (sorted, rank) = crate::hash::bucket_sort(block, &params);
                    let got = find_best_window(block, &sorted, &rank, &params);
                    assert!(got == want, "{name} block {i} {params:?}: window walk differs from the chain walk");
                }
            }
        }
    }

    /// `find_cands` by its definition: the union of every chain's first `depth_i` entries,
    /// sorted nearest first, filtered by "capped length strictly above the best so far, from 3".
    fn find_cands_by_definition(block: &[u8], params: &MatchParams) -> Vec<CandWords> {
        let ch = chains(block, params);
        let mut out = vec![[0u32; 2]; BLOCK_SIZE];
        let mut dead = vec![false; BLOCK_SIZE];
        for (p, w) in out.iter_mut().enumerate().take(PARSE_END) {
            let mut v = Vec::new();
            for (c, &d) in ch.iter().zip(&cand_depths(params)) {
                let mut q = c[p];
                for _ in 0..d {
                    if q == NO_POS {
                        break;
                    }
                    v.push(q);
                    q = c[q as usize];
                }
            }
            // Dead: the h3 chain ends within OPT_H3_DEPTH steps (and no record, below).
            let mut q = ch[1][p];
            let mut n = 0;
            while q != NO_POS && n < OPT_H3_DEPTH {
                q = ch[1][q as usize];
                n += 1;
            }
            dead[p] = q == NO_POS;
            v.sort_unstable_by(|a, b| b.cmp(a));
            v.dedup();
            let mut recs = Vec::new();
            let mut best = 2;
            for q in v {
                let c = match_len_capped(block, p, q as usize, params.search_cap as usize);
                if c > best {
                    best = c;
                    recs.push(Cand { offset: (p - q as usize) as u32, len: c as u32 });
                }
            }
            if let (Some(&a), Some(&b)) = (recs.first(), recs.last()) {
                *w = pack_cands(a, b);
                dead[p] = false;
            }
        }
        for (w, &d) in out.iter_mut().zip(&dead) {
            if d {
                w[1] |= DEAD_BIT;
            }
        }
        out
    }

    #[test]
    fn find_cands_matches_its_definition() {
        use crate::params::OPT16;
        use crate::params::{OptParams, SparseChain, OPT16P1};
        let one_sparse = OptParams { sparse_chains: [Some(SparseChain { width: 5, stride: 1, depth: 3 }), None, None], ..OPT16.opt.unwrap() };
        let variants = [
            OPT16,
            MatchParams { depth: 3, ..OPT16 },
            MatchParams { depth: 64, ..OPT16 },
            OPT16P1,
            MatchParams { depth: 1, ..OPT16P1 },
            MatchParams { opt: Some(one_sparse), ..OPT16 },
        ];
        for params in variants {
            for (name, bytes) in synth::test_cases() {
                for (i, blk) in chunk_file(&bytes).into_iter().enumerate() {
                    let got = find_cands(&blk.data, &chains(&blk.data, &params), &params);
                    let want = find_cands_by_definition(&blk.data, &params);
                    assert!(got == want, "{name} block {i} depth {}: find_cands differs from its definition", params.depth);
                    // Dead positions really have no earlier position with the same 3 bytes.
                    let mut seen = std::collections::HashSet::new();
                    let mut n_dead = 0;
                    for (p, &w) in got.iter().enumerate().take(PARSE_END) {
                        let k = &blk.data[p..p + 3];
                        if is_dead(w) {
                            assert!(!seen.contains(k), "p={p}: dead but an earlier position shares its 3 bytes");
                            assert_eq!(w, [0, DEAD_BIT], "p={p}: dead with a record");
                            n_dead += 1;
                        }
                        seen.insert(k);
                    }
                    if name == "random" {
                        assert!(n_dead > 0, "{name}: no dead position");
                    }
                    for (p, &w) in got.iter().enumerate() {
                        let (a, b) = unpack_cands(w);
                        assert_eq!(a.len == 0, b.len == 0, "p={p}");
                        if a.len > 0 {
                            assert!(a.len >= 3 && a.len <= b.len && b.len <= 64, "p={p} {a:?} {b:?}");
                            assert!(a.offset <= b.offset || a == b, "p={p}: A must be the nearer record");
                            for c in [a, b] {
                                let l = match_len_capped(&blk.data, p, p - c.offset as usize, 64) as u32;
                                assert_eq!(l, c.len, "p={p} {c:?} is not a real (capped) match");
                            }
                        }
                    }
                }
            }
        }
    }

    /// K2opt's record rules on one position of a random block: the h3-only neighbour (3 bytes)
    /// is A, a 4-byte h4/h3 match then a 10-byte one are records, a farther 10-byte tie is not;
    /// B is the nearer 10-byte match. A 100-byte match is stored capped at 64.
    #[test]
    fn find_cands_records_nearest_then_longest() {
        use crate::params::OPT16;
        let mut block = synth::random(77, BLOCK_SIZE);
        let p = 5000;
        let put = |block: &mut Vec<u8>, q: usize, len: usize| {
            for i in 0..len {
                block[q + i] = block[p + i];
            }
            block[q + len] = block[p + len] ^ 0xFF;
        };
        put(&mut block, 4990, 3);
        put(&mut block, 4900, 4);
        put(&mut block, 4000, 10);
        put(&mut block, 3000, 10);
        let c = find_cands(&block, &chains(&block, &OPT16), &OPT16);
        assert_eq!(unpack_cands(c[p]), (Cand { offset: 10, len: 3 }, Cand { offset: 1000, len: 10 }));
        let p2 = 9000;
        for i in 0..100 {
            block[p2 + i] = block[p2 - 700 + i];
        }
        let c = find_cands(&block, &chains(&block, &OPT16), &OPT16);
        let (a, b) = unpack_cands(c[p2]);
        assert_eq!(b, Cand { offset: 700, len: 64 });
        assert_eq!(unpack_cands(pack_cands(a, b)), (a, b));
    }

    /// `sparse_chain_preds`: only sparse positions (`p % stride == 0`, `p < SPARSE_END`) are
    /// linked, each to the previous sparse position with the same `hash_sparse`; on all-zero
    /// data that is `p - stride`, the last sparse position is `SPARSE_END - stride` (for strides
    /// dividing 12), and `chains` appends one such chain per `S3_CHAINS` entry.
    #[test]
    fn sparse_chain_links_sparse_positions_only() {
        use crate::params::{OPT16P1, S3_CHAINS};
        let zeros = synth::zeros(BLOCK_SIZE);
        for stride in [1u32, 2, 4, 8] {
            let c = SparseChain { width: 10, stride, depth: 16 };
            let pred = sparse_chain_preds(&zeros, &c);
            for (p, &q) in pred.iter().enumerate() {
                let sparse = p % stride as usize == 0 && p < SPARSE_END;
                let want = if sparse && p > 0 { (p - stride as usize) as u32 } else { NO_POS };
                assert_eq!(q, want, "stride {stride} p={p}");
            }
        }
        assert_eq!(sparse_chain_preds(&zeros, &S3_CHAINS[0].unwrap())[BLOCK_SIZE - 16], (BLOCK_SIZE - 20) as u32);
        assert_eq!(sparse_chain_preds(&zeros, &S3_CHAINS[0].unwrap())[BLOCK_SIZE - 12], NO_POS);
        let text = synth::text(3, BLOCK_SIZE);
        let ch = chains(&text, &OPT16P1);
        assert_eq!(ch.len(), 5);
        assert_eq!(cand_depths(&OPT16P1), [8, OPT_H3_DEPTH, 16, 16, 16]);
        for (i, c) in S3_CHAINS.iter().flatten().enumerate() {
            assert_eq!(ch[2 + i], sparse_chain_preds(&text, c));
            for (p, &q) in ch[2 + i].iter().enumerate() {
                if q != NO_POS {
                    assert!(p % 4 == 0 && (q as usize) < p && q % 4 == 0, "chain {i} p={p} q={q}");
                    assert_eq!(hash_sparse(&text, q as usize, c.width), hash_sparse(&text, p, c.width));
                }
            }
        }
    }

    /// The sparse chains reach a match the 8-deep h4 chain cannot: ten nearer 4-byte decoys
    /// fill the h4 (and h3) walk at `p`, a 16-byte match lies behind them. Found only when both
    /// `p` and the source are sparse positions (multiples of 4); at a non-sparse `p`, or with a
    /// non-sparse source, A = B = the nearest decoy. OPT16 (h4 32 deep) finds it either way.
    #[test]
    fn find_cands_sparse_chains_reach_past_h4_depth() {
        use crate::params::{OPT16, OPT16P1};
        let mut block = synth::random(79, BLOCK_SIZE);
        // (p, source): sparse/sparse, non-sparse p, non-sparse source
        let sites = [(20000usize, 15000usize), (30002, 25002), (40000, 35001)];
        for &(p, src) in &sites {
            for i in 0..16 {
                block[src + i] = block[p + i];
            }
            block[src + 16] = block[p + 16] ^ 0xFF;
            for d in 0..10 {
                let q = p - 100 - 40 * d;
                block.copy_within(p..p + 4, q);
                block[q + 4] = block[p + 4] ^ 0xFF;
            }
        }
        let decoy = Cand { offset: 100, len: 4 };
        let c1 = find_cands(&block, &chains(&block, &OPT16P1), &OPT16P1);
        let c16 = find_cands(&block, &chains(&block, &OPT16), &OPT16);
        for (k, &(p, src)) in sites.iter().enumerate() {
            let far = Cand { offset: (p - src) as u32, len: 16 };
            assert_eq!(unpack_cands(c16[p]), (decoy, far), "opt16 site {k}");
            let want = if k == 0 { (decoy, far) } else { (decoy, decoy) };
            assert_eq!(unpack_cands(c1[p]), want, "opt16p1 site {k}");
        }
    }

    /// A position on several chains is visited once but spends depth on each: with h4 and a
    /// 5-byte sparse chain at stride 1 (whose entries here are exactly h4's), a sparse depth of
    /// 1 adds nothing to h4 depth 1, and the 2nd-nearest decoy's 6-byte match needs depth 2 on
    /// either chain. Four nearer 3-byte decoys use up the h3 walk (A is the nearest of them).
    #[test]
    fn find_cands_shared_position_spends_each_depth() {
        use crate::params::{OptParams, OPT16};
        let mut block = synth::random(80, BLOCK_SIZE);
        let p = 12000;
        let (q1, q2) = (p - 50, p - 90);
        for i in 0..6 {
            block[q1 + i] = block[p + i];
            block[q2 + i] = block[p + i];
        }
        block[q1 + 5] = block[p + 5] ^ 0xFF; // q1: 5 bytes
        block[q2 + 6] = block[p + 6] ^ 0xFF; // q2: 6 bytes
        for d in [10, 15, 20, 25] {
            for i in 0..3 {
                block[p - d + i] = block[p + i];
            }
            block[p - d + 3] = block[p + 3] ^ 0xFF;
        }
        let a3 = Cand { offset: 10, len: 3 };
        let with = |h4: u32, sparse: u32| {
            let o = OptParams { sparse_chains: [Some(SparseChain { width: 5, stride: 1, depth: sparse }), None, None], ..OPT16.opt.unwrap() };
            let m = MatchParams { depth: h4, opt: Some(o), ..OPT16 };
            let c = find_cands(&block, &chains(&block, &m), &m);
            unpack_cands(c[p])
        };
        let near = Cand { offset: 50, len: 5 };
        let far = Cand { offset: 90, len: 6 };
        assert_eq!(with(1, 1), (a3, near));
        assert_eq!(with(1, 2), (a3, far));
        assert_eq!(with(2, 1), (a3, far));
    }

    /// xxh64 of the concatenated lvl3 frames, first captured on the unmodified M3 code (ddeee75);
/// re-captured when `synth::nif_like` switched to a platform-independent sine (the lvl3 code
/// itself unchanged: the old anchors still passed on Linux immediately before the switch).
    const M3_LVL3_ANCHOR: u64 = 0xd3354ac3c8f4a5d2;

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
