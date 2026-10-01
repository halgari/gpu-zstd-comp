//! CPU reference compressor used as the correctness and quality baseline.
//!
//! This is the bit-exact oracle the GPU kernels are tested against: a hash-chain match
//! finder and parse driven by runtime `MatchParams`, using integer arithmetic only so the
//! GPU mirrors it exactly. The `LVL3` preset is the M3 level-3-style greedy parse.
use crate::config::{BLOCK_SIZE, NO_POS, PARSE_END};
use crate::frame::{write_frame, FrameOptions};
use crate::hash::{compute_preds, hash3, hash_long, hash_short, hash_width, key};
use crate::lazy::{lazy_parse, lazy_parse_segmented};
use crate::params::{cpu_supports, Hashes, MatchParams, OPT_H3_DEPTH};
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
/// `hash_width(.., min_match)`; for `Opt3`, the 4-byte chain (`hash_width(.., 4)`) then the
/// 3-byte chain (`hash3`).
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
        Hashes::Opt3 => vec![
            compute_preds(block, move |b: &[u8], pos: usize| key(hash_width(b, pos, 4), hb)),
            compute_preds(block, move |b: &[u8], pos: usize| key(hash3(b, pos), hb)),
        ],
    }
}

/// One `find_cands` record: offset back from the position and capped length (`len == 0`: none).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cand {
    pub offset: u32,
    pub len: u32,
}

/// The two candidate words of a position (K2opt's output, 8 B per position):
/// `w[0] = offA | lenA << 16 | lenB << 24`, `w[1] = offB | dead_run << 16`. All zero when there
/// is no candidate, apart from the dead run (`find_cands`), which is nonzero only there.
pub type CandWords = [u32; 2];

/// Dead runs (M6 A3) stop at the end of the position's aligned tile of this many positions (K2opt's
/// workgroup), so each GPU workgroup computes its own runs.
pub const DEAD_TILE: usize = 256;

/// The dead run of a position's candidate words (`find_cands`): 0 when the position is not dead.
pub fn dead_run(w: CandWords) -> u32 {
    w[1] >> 16
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

/// K2opt (m5-opt-design §2.1), the optimal parse's candidates: for every `p < PARSE_END`, walk
/// chain 0 (`h4`) `params.depth` deep and chain 1 (`h3`) `OPT_H3_DEPTH` deep, merged by position
/// nearest first (both chains are strictly decreasing; a position on both is visited once). Each
/// visited `q` gets the capped length `c = match_len_capped(block, p, q, search_cap)`; a *record*
/// is a `q` whose `c` strictly beats every earlier `c` and the floor 2 (so records start at
/// length 3, and on equal lengths the nearer `q` wins). `A` = the first record (the nearest
/// match of at least 3 bytes), `B` = the last (longest); with one record `B == A`. The walk may
/// stop once `c`
/// reaches `min(search_cap, BLOCK_SIZE - p)` (no later `q` can beat it). Lengths are capped: a
/// stored 64 (`search_cap`) means "at least 64", extended by the parse. Positions `>= PARSE_END`
/// are zero. Requires `Opt3` chains (`chains(block, params)`).
///
/// Fingerprint caveat (for GPU filters): an `h4`-chain entry whose first 4 bytes differ from
/// `p`'s (a 16-bit hash collision) can still share 3 bytes and is then a valid 3-byte record, so
/// a 4-byte fingerprint mismatch may skip the compare only once `best >= 3` (or when the first 3
/// bytes differ too).
///
/// Dead positions (M6 A3, a09): `p` is *dead* when it has no record and its `h3` walk reached
/// the chain's end (`NO_POS`, within `OPT_H3_DEPTH` steps). Then no earlier position shares `p`'s
/// first 3 bytes: every such position has `p`'s `hash3` key, so it is on `p`'s `h3` chain (which
/// links every earlier position below `HASHED_POSITIONS` with that key), and the walk visited the
/// whole chain without a 3-byte match. So no candidate and no rep (whatever its offset) reaches 3
/// bytes there, and the parse's `get_all_matches` is empty under every rep state. `w[1]`'s high
/// half holds the dead run: the number of consecutive dead positions from `p` up to the end of
/// `p`'s `DEAD_TILE`-aligned tile (1..=`DEAD_TILE`), 0 when `p` is not dead. The parse oracle
/// ignores it (`unpack_cands`); K3opt uses it to skip dead positions.
pub fn find_cands(block: &[u8], chains: &[Vec<u32>], params: &MatchParams) -> Vec<CandWords> {
    assert_eq!(params.hashes, Hashes::Opt3, "find_cands: Opt3 chains only");
    assert_eq!(chains.len(), 2);
    let (h4, h3) = (&chains[0], &chains[1]);
    let cap = params.search_cap as usize;
    let mut out = vec![[0u32; 2]; BLOCK_SIZE];
    let mut dead = vec![false; BLOCK_SIZE];
    for p in 0..PARSE_END {
        let max_c = cap.min(BLOCK_SIZE - p);
        let (mut q4, mut n4) = (h4[p], params.depth);
        let (mut q3, mut n3) = (h3[p], OPT_H3_DEPTH);
        let mut best = 2usize;
        let (mut a, mut b) = (Cand::default(), Cand::default());
        loop {
            // Next position of the merged walk: the larger live head; equal heads advance both.
            let live4 = n4 > 0 && q4 != NO_POS;
            let live3 = n3 > 0 && q3 != NO_POS;
            let q = match (live4, live3) {
                (false, false) => break,
                (true, false) => q4,
                (false, true) => q3,
                (true, true) => q4.max(q3),
            };
            if live4 && q4 == q {
                q4 = h4[q as usize];
                n4 -= 1;
            }
            if live3 && q3 == q {
                q3 = h3[q as usize];
                n3 -= 1;
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
        dead[p] = a.len == 0 && q3 == NO_POS;
    }
    set_dead_runs(&mut out, &dead);
    out
}

/// Writes the dead runs of the flags `dead` into `w[1]`'s high half (see `find_cands`).
fn set_dead_runs(out: &mut [CandWords], dead: &[bool]) {
    let mut run = 0u32;
    for p in (0..BLOCK_SIZE).rev() {
        if p % DEAD_TILE == DEAD_TILE - 1 {
            run = 0;
        }
        run = if dead[p] { run + 1 } else { 0 };
        out[p][1] |= run << 16;
    }
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
/// Panics if `params` is invalid or not implemented on the CPU yet (`params::cpu_supports`).
pub fn compress_block(block: &[u8], params: MatchParams) -> BlockOutput {
    assert_eq!(block.len(), BLOCK_SIZE);
    if let Err(e) = params.validate() {
        panic!("reference::compress_block: invalid params {params:?}: {e}");
    }
    assert!(cpu_supports(&params), "reference::compress_block: {params:?} is not implemented yet on cpu");
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

    /// `find_cands` by its definition: the union of the two chains' first `depth` / 4 entries,
    /// sorted nearest first, filtered by "capped length strictly above the best so far, from 3".
    fn find_cands_by_definition(block: &[u8], params: &MatchParams) -> Vec<CandWords> {
        let ch = chains(block, params);
        let mut out = vec![[0u32; 2]; BLOCK_SIZE];
        let mut dead = vec![false; BLOCK_SIZE];
        for (p, w) in out.iter_mut().enumerate().take(PARSE_END) {
            let mut v = Vec::new();
            for (c, d) in [(&ch[0], params.depth), (&ch[1], OPT_H3_DEPTH)] {
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
        // Runs by definition: consecutive dead positions from p within p's tile.
        for p in 0..BLOCK_SIZE {
            let end = (p / DEAD_TILE + 1) * DEAD_TILE;
            let run = (p..end).take_while(|&q| dead[q]).count() as u32;
            out[p][1] |= run << 16;
        }
        out
    }

    #[test]
    fn find_cands_matches_its_definition() {
        // OPT16 (and its depth variants) only validate at blocks of at most 64 KiB.
        if crate::config::LOG2_BLOCK > 16 {
            return;
        }
        use crate::params::OPT16;
        for params in [OPT16, MatchParams { depth: 3, ..OPT16 }, MatchParams { depth: 64, ..OPT16 }] {
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
                        if dead_run(w) > 0 {
                            assert!(!seen.contains(k), "p={p}: dead but an earlier position shares its 3 bytes");
                            assert_eq!(w, [0, dead_run(w) << 16], "p={p}: dead with a record");
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

    /// xxh64 of the concatenated lvl3 frames, first captured on the unmodified M3 code (ddeee75);
/// re-captured when `synth::nif_like` switched to a platform-independent sine (the lvl3 code
/// itself unchanged: the old anchors still passed on Linux immediately before the switch).
    #[cfg(feature = "block-128k")]
    const M3_LVL3_ANCHOR: u64 = 0xc73510d0195192d6;
    #[cfg(feature = "block-64k")]
    const M3_LVL3_ANCHOR: u64 = 0xd3354ac3c8f4a5d2;
    #[cfg(feature = "block-32k")]
    const M3_LVL3_ANCHOR: u64 = 0x0a4f0a366675659b;
    #[cfg(feature = "block-16k")]
    const M3_LVL3_ANCHOR: u64 = 0x42fda1cbdb8957f5;

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
