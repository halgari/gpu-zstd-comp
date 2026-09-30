//! Lazy / lazy2 parse: a port of libzstd 1.5.7 `ZSTD_compressBlock_lazy_generic`
//! (`lib/compress/zstd_lazy.c`, `dictMode == ZSTD_noDict`, `depth = params.lazy`), spec §3.4.
//!
//! This is the normative oracle the GPU K3 lazy kernel mirrors bit-exactly: integer-only,
//! deterministic, no reads past `BLOCK_SIZE`. The structure follows the C function
//! statement by statement; names in comments (`ip`, `anchor`, `start`, `offBase`,
//! `matchLength`, `offset_1`, `offset_2`, `ilimit`) are zstd's.
//!
//! Substitutions (the "only change" allowed by the spec):
//! - `ZSTD_searchMax(ip)` is `best[ip]` (K2 output), extended to its true length with
//!   `match_len` when it hit `search_cap`. `len < min_match` means "no match". An explicit
//!   match has `offBase = OFFSET_TO_OFFBASE(offset) = offset + 3`.
//! - `MEM_read32(a) == MEM_read32(b)` followed by `ZSTD_count(...) + 4` is the bounded
//!   `match_len(block, a, b) >= 4` (see `rep_len`).
//! - `ilimit = iend - 8` is `PARSE_END`.
//! - Sequences are stored through the M3 `off_base_for` / `apply_off_base` bookkeeping, so
//!   the frame writer is unchanged. `offset_1` / `offset_2` always equal the decoder's
//!   `reps[0]` / `reps[1]` (see the `// deviation:` notes for the one case where zstd's own
//!   bookkeeping would differ).
use crate::config::{BLOCK_SIZE, PARSE_END};
use crate::params::MatchParams;
use crate::reference::{match_len, Match};
use crate::seq::{apply_off_base, off_base_for, BlockOutput, Reps, Sequence, INITIAL_REPS};

/// zstd `REPCODE1_TO_OFFBASE`.
const REPCODE1_TO_OFFBASE: u32 = 1;
/// zstd `ZSTD_REP_NUM`: explicit offsets are stored as `offset + ZSTD_REP_NUM`.
const ZSTD_REP_NUM: u32 = 3;
/// zstd `kSearchStrength`: literal-run skip `step = ((ip - anchor) >> 8) + 1`.
const K_SEARCH_STRENGTH: u32 = 8;

/// zstd `ZSTD_highbit32` (`x > 0`): index of the highest set bit.
fn highbit32(x: u32) -> i32 {
    debug_assert!(x > 0);
    31 - x.leading_zeros() as i32
}

/// Repeat-offset probe at `p`: zstd's `(off > 0) & (MEM_read32(p) == MEM_read32(p - off))`
/// then `ZSTD_count(p+4, p+4-off, iend) + 4`. Returns the match length, or 0 when the first
/// 4 bytes differ or `off` is unusable. `off > p` cannot happen for a real rep history (zstd
/// relies on its window check for that); it is guarded so nothing underflows.
fn rep_len(block: &[u8], p: usize, off: u32) -> u32 {
    debug_assert!(off as usize <= p, "rep offset {off} beyond position {p}");
    if off == 0 || off as usize > p {
        return 0;
    }
    let l = match_len(block, p, p - off as usize);
    if l >= 4 {
        l as u32
    } else {
        0
    }
}

/// `ZSTD_searchMax(ip)`: `(matchLength, offBase)` of `best[ip]`, or `None` when it has no
/// match of at least `min_match`. A capped length is extended to the true match length.
fn search_max(block: &[u8], best: &[Match], ip: usize, params: &MatchParams) -> Option<(u32, u32)> {
    debug_assert!(ip < PARSE_END, "best[] read at {ip} >= PARSE_END");
    let m = best[ip];
    if m.len < params.min_match {
        return None;
    }
    let len = if m.len == params.search_cap { match_len(block, ip, ip - m.offset as usize) as u32 } else { m.len };
    Some((len, m.offset + ZSTD_REP_NUM))
}

/// Appends one sequence (`lit_len` literals from `anchor`, then `ml` bytes at `offset`) with
/// the decoder's rep bookkeeping, like `greedy_parse`.
fn store(out: &mut BlockOutput, reps: &mut Reps, block: &[u8], anchor: usize, lit_len: usize, offset: u32, ml: u32) {
    let ll = lit_len as u32;
    let ob = off_base_for(offset, ll, reps);
    apply_off_base(reps, ob, ll);
    out.literals.extend_from_slice(&block[anchor..anchor + lit_len]);
    out.sequences.push(Sequence { lit_len: ll, match_len: ml, off_base: ob });
}

/// Lazy / lazy2 parse over precomputed best[] (spec §3.4). params.lazy ∈ {1,2}.
pub fn lazy_parse(block: &[u8], best: &[Match], params: &MatchParams) -> BlockOutput {
    assert!(params.lazy == 1 || params.lazy == 2, "lazy_parse: lazy {} not in 1..=2", params.lazy);
    assert_eq!(block.len(), BLOCK_SIZE);
    assert!(best.len() >= PARSE_END);
    let depth = params.lazy;
    let ilimit = PARSE_END;

    let mut out = BlockOutput::default();
    let mut reps: Reps = INITIAL_REPS;
    let mut anchor = 0usize;
    // zstd: `ip += (dictAndPrefixLength == 0)`. Every block is a fresh frame, so the parse
    // starts at 1.
    let mut ip = 1usize;
    // zstd noDict: offsets larger than `maxRep = curr - windowLow` (= ip = 1 here) start
    // disabled as 0. INITIAL_REPS = [1, 4, 8], so offset_1 = 1 is live and offset_2 is 0
    // (off) until the first explicit match sets it.
    let max_rep = ip as u32;
    let mut offset_1 = if reps[0] <= max_rep { reps[0] } else { 0 };
    let mut offset_2 = if reps[1] <= max_rep { reps[1] } else { 0 };

    while ip < ilimit {
        // offset_1 mirrors reps[0]; offset_2 mirrors reps[1] once enabled.
        debug_assert!(offset_1 == reps[0] && (offset_2 == 0 || offset_2 == reps[1]));
        let mut match_length: u32 = 0;
        let mut off_base: u32 = REPCODE1_TO_OFFBASE;
        let mut start = ip + 1;

        // check repCode (at ip+1). With depth >= 1 zstd does not `goto _storeSequence` here.
        let l = rep_len(block, ip + 1, offset_1);
        if l > 0 {
            match_length = l;
        }

        // first search (depth 0)
        if let Some((ml2, ob)) = search_max(block, best, ip, params) {
            if ml2 > match_length {
                match_length = ml2;
                start = ip;
                off_base = ob;
            }
        }

        if match_length < 4 {
            // jump faster over incompressible sections
            ip += ((ip - anchor) >> K_SEARCH_STRENGTH) + 1;
            // deviation: zstd also sets `ms->lazySkipping = step > kLazySkippingStep`, which
            // only stops hash-table insertion of skipped positions. K2 searched every
            // position, so it cannot change what `best[]` holds; not applicable.
            continue;
        }

        // let's try to find a better solution
        // deviation: zstd's `while (ip < ilimit) { ip++; ... }` can search at ip == ilimit;
        // best[] is only valid below PARSE_END, so deferral stops when ip + 1 >= PARSE_END
        // (here and in the depth-2 guard below).
        while ip + 1 < ilimit {
            ip += 1;
            // search depth 1: repcode at ip, ×3 rule. (zstd's `(offBase) &&` is always true.)
            let ml_rep = rep_len(block, ip, offset_1);
            if ml_rep >= 4 {
                let gain2 = (ml_rep * 3) as i32;
                let gain1 = (match_length * 3) as i32 - highbit32(off_base) + 1;
                if gain2 > gain1 {
                    match_length = ml_rep;
                    off_base = REPCODE1_TO_OFFBASE;
                    start = ip;
                }
            }
            if let Some((ml2, ob)) = search_max(block, best, ip, params) {
                let gain2 = (ml2 * 4) as i32 - highbit32(ob); // raw approx
                let gain1 = (match_length * 4) as i32 - highbit32(off_base) + 4;
                if ml2 >= 4 && gain2 > gain1 {
                    match_length = ml2;
                    off_base = ob;
                    start = ip;
                    continue; // search a better one
                }
            }

            // let's find an even better one
            if depth == 2 && ip + 1 < ilimit {
                ip += 1;
                // search depth 2: repcode at ip, ×4 rule.
                let ml_rep = rep_len(block, ip, offset_1);
                if ml_rep >= 4 {
                    let gain2 = (ml_rep * 4) as i32;
                    let gain1 = (match_length * 4) as i32 - highbit32(off_base) + 1;
                    if gain2 > gain1 {
                        match_length = ml_rep;
                        off_base = REPCODE1_TO_OFFBASE;
                        start = ip;
                    }
                }
                if let Some((ml2, ob)) = search_max(block, best, ip, params) {
                    let gain2 = (ml2 * 4) as i32 - highbit32(ob); // raw approx
                    let gain1 = (match_length * 4) as i32 - highbit32(off_base) + 7;
                    if ml2 >= 4 && gain2 > gain1 {
                        match_length = ml2;
                        off_base = ob;
                        start = ip;
                        continue;
                    }
                }
            }
            break; // nothing found: store previous solution
        }

        if off_base > ZSTD_REP_NUM {
            // catch up: `start > anchor && start - offset > prefixLowest` (block position 0)
            let offset = off_base - ZSTD_REP_NUM;
            let off = offset as usize;
            while start > anchor && start > off && block[start - 1] == block[start - 1 - off] {
                start -= 1;
                match_length += 1;
            }
            // store sequence. zstd: `offset_2 = offset_1; offset_1 = offset;`
            let lit_len = start - anchor;
            let zstd_view = (offset, offset_1);
            let dedup = lit_len > 0 && offset == reps[0];
            store(&mut out, &mut reps, block, anchor, lit_len, offset, match_length);
            if dedup {
                // deviation: zstd stores this match with an explicit offBase (decoded as reps
                // [o, o, r1]) and sets offset_2 = offset_1 = o. off_base_for stores it as
                // repcode 1, which the decoder resolves without touching the history, so
                // offset_2 becomes the old reps[1] (and, if it was still disabled, is enabled).
                // The decoder semantics win: the immediate loop below then probes the old
                // reps[1] where zstd would probe o, and can emit a (ll 0, repcode 1) match zstd
                // would not (test `dedup_store_enables_old_rep1_immediate`).
                // Common case: the first match of a block starting with a byte run (start 1,
                // offset 1 == reps[0], ll 1) lands here and enables offset_2 = INITIAL_REPS[1]
                // = 4, which zstd's maxRep rule keeps disabled. That one is output-equivalent
                // while reps[0] == 1: a store ending an offset-1 run has block[E] != block[E-1]
                // == block[E-4], so an offset-4 probe at its end can never match (test
                // `dedup_store_at_block_start_byte_run`).
                debug_assert_eq!(reps[0], zstd_view.0);
            } else {
                debug_assert_eq!((reps[0], reps[1]), zstd_view, "rep history diverged from zstd's");
            }
            offset_1 = reps[0];
            offset_2 = reps[1];
        } else {
            // repcode 1 found at ip+1 or later, so lit_len > 0: the decoder reads reps[0]
            // (= offset_1) and leaves the history unchanged, as zstd does.
            let lit_len = start - anchor;
            debug_assert!(lit_len > 0);
            let before = reps;
            store(&mut out, &mut reps, block, anchor, lit_len, offset_1, match_length);
            debug_assert_eq!(out.sequences.last().unwrap().off_base, REPCODE1_TO_OFFBASE);
            debug_assert_eq!(reps, before);
        }
        anchor = start + match_length as usize;
        ip = anchor;

        // check immediate repcode: an offset_2 match at ip is stored as (ll 0, repcode 1),
        // which the decoder resolves to reps[1] and swaps into reps[0] — zstd's swap.
        while ip <= ilimit && offset_2 > 0 {
            let ml = rep_len(block, ip, offset_2);
            if ml == 0 {
                break;
            }
            (offset_1, offset_2) = (offset_2, offset_1); // swap repcodes
            store(&mut out, &mut reps, block, anchor, 0, offset_1, ml);
            debug_assert_eq!(out.sequences.last().unwrap().off_base, REPCODE1_TO_OFFBASE);
            debug_assert_eq!((reps[0], reps[1]), (offset_1, offset_2));
            ip += ml as usize;
            anchor = ip;
        }
    }
    // last literals
    out.literals.extend_from_slice(&block[anchor..BLOCK_SIZE]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::chunk_file;
    use crate::config::{BLOCK_SIZE, PARSE_END};
    use crate::frame::{write_frame, FrameOptions};
    use crate::params::{LVL9, RUNG1, RUNG2};
    use crate::reference::{chains, compress_block, find_best, match_len};
    use crate::seq::{reconstruct, Sequence};
    use crate::synth;

    /// zstd `ZSTD_highbit32`.
    fn hb(x: u32) -> i32 {
        31 - x.leading_zeros() as i32
    }

    /// libzstd's explicit-match gain at step 1 / 2: `ml*4 - highbit(offBase)`.
    fn gain(ml: u32, off_base: u32) -> i32 {
        ml as i32 * 4 - hb(off_base)
    }

    fn seq(lit_len: u32, match_len: u32, off_base: u32) -> Sequence {
        Sequence { lit_len, match_len, off_base }
    }

    /// Random background: no accidental 4-byte repeats at the offsets the tests use.
    fn background(seed: u64) -> Vec<u8> {
        synth::random(seed, BLOCK_SIZE)
    }

    /// Make `block[dst..dst+len]` a match at offset `off` of exactly `len` bytes: copy the
    /// (random) destination bytes back to `dst-off`, break the byte after the source and the
    /// byte before it (so catch-up cannot extend the match backwards).
    fn plant(block: &mut [u8], dst: usize, off: usize, len: usize) {
        assert!(off > len, "plant: overlapping source");
        let src = dst - off;
        for i in 0..len {
            block[src + i] = block[dst + i];
        }
        if dst + len < BLOCK_SIZE {
            block[src + len] = block[dst + len] ^ 0xFF;
        }
        if src > 0 {
            block[src - 1] = block[dst - 1] ^ 0xFF;
        }
    }

    /// The real match at `p` with offset `off`, as K2 would store it (the tests keep lengths
    /// under `search_cap`, so this is also the capped length).
    fn real(block: &[u8], p: usize, off: usize) -> Match {
        Match { offset: off as u32, len: match_len(block, p, p - off) as u32 }
    }

    /// Parse and check that the output decodes back to `block`.
    fn run(block: &[u8], best: &[Match], params: &MatchParams) -> Vec<Sequence> {
        let out = lazy_parse(block, best, params);
        let got = reconstruct(&out).expect("reconstruct");
        assert_eq!(got, block, "lazy_parse output does not reconstruct the block");
        out.sequences
    }

    fn empty_best() -> Vec<Match> {
        vec![Match::default(); BLOCK_SIZE]
    }

    /// Step 1: the match at ip+1 wins by exactly 1 over `gain1 + 4` → the parse defers.
    #[test]
    fn lazy_prefers_later_longer_match() {
        let mut block = background(11);
        plant(&mut block, 200, 20, 8); // A: offBase 23, highbit 4
        plant(&mut block, 201, 130, 10); // B: offBase 133, highbit 7
        let mut best = empty_best();
        best[200] = real(&block, 200, 20);
        best[201] = real(&block, 201, 130);
        assert_eq!((best[200].len, best[201].len), (8, 10));
        assert_eq!(gain(10, 133), gain(8, 23) + 4 + 1, "B must win by exactly 1");
        assert_eq!(run(&block, &best, &RUNG2), vec![seq(201, 10, 133)]);
    }

    /// Step 1: equal gains → the current match is kept.
    #[test]
    fn lazy_keeps_on_gain_tie() {
        let mut block = background(12);
        plant(&mut block, 200, 14, 8); // A: offBase 17, highbit 4
        plant(&mut block, 201, 28, 9); // B: offBase 31, highbit 4
        let mut best = empty_best();
        best[200] = real(&block, 200, 14);
        best[201] = real(&block, 201, 28);
        assert_eq!((best[200].len, best[201].len), (8, 9));
        assert_eq!(gain(9, 31), gain(8, 17) + 4, "B must tie");
        assert_eq!(run(&block, &best, &RUNG2), vec![seq(200, 8, 17)]);
    }

    /// Step 2 (lazy2): a match at ip+2 must beat `gain1 + 7`. Winning by 1 defers; a tie keeps.
    /// lazy1 never looks at ip+2.
    #[test]
    fn lazy2_second_step_threshold() {
        // Win by 1: A at 200 (offBase 17, hb 4), nothing at 201, C at 202 (offBase 31, hb 4).
        let mut block = background(13);
        plant(&mut block, 200, 14, 8);
        plant(&mut block, 202, 28, 10);
        let mut best = empty_best();
        best[200] = real(&block, 200, 14);
        best[202] = real(&block, 202, 28);
        assert_eq!((best[200].len, best[202].len), (8, 10));
        assert_eq!(gain(10, 31), gain(8, 17) + 7 + 1);
        assert_eq!(run(&block, &best, &LVL9), vec![seq(202, 10, 31)]);
        assert_eq!(run(&block, &best, &RUNG2), vec![seq(200, 8, 17)], "lazy1 must not search ip+2");

        // Tie: C has offBase 43 (hb 5).
        let mut block = background(14);
        plant(&mut block, 200, 14, 8);
        plant(&mut block, 202, 40, 10);
        let mut best = empty_best();
        best[200] = real(&block, 200, 14);
        best[202] = real(&block, 202, 40);
        assert_eq!((best[200].len, best[202].len), (8, 10));
        assert_eq!(gain(10, 43), gain(8, 17) + 7);
        assert_eq!(run(&block, &best, &LVL9), vec![seq(200, 8, 17)]);
    }

    /// A first match M0 (offset 45) sets offset_1; later the explicit match A at 120 competes
    /// with a repcode match at 121 (step 1, ×3 rule) or 122 (lazy2 step 2, ×4 rule).
    fn rep_block(seed: u64, a_off: usize, rep_at: usize) -> (Vec<u8>, Vec<Match>) {
        let mut block = background(seed);
        plant(&mut block, 60, 45, 10); // M0: offBase 48
        plant(&mut block, rep_at, 45, 7); // repcode match (offset_1 = 45), 7 bytes
        plant(&mut block, 120, a_off, 8); // A
        let mut best = empty_best();
        best[60] = real(&block, 60, 45);
        best[120] = real(&block, 120, a_off);
        assert_eq!((best[60].len, best[120].len), (10, 8));
        assert_eq!(match_len(&block, rep_at, rep_at - 45), 7);
        (block, best)
    }

    #[test]
    fn lazy_rep_check_at_ip_plus_1() {
        // ×3 rule: gain2 = mlRep*3 vs gain1 = ml*3 - highbit(offBase) + 1.
        // A offBase 23 (hb 4): 21 vs 24 - 4 + 1 = 21 → tie, keep A.
        let (block, best) = rep_block(21, 20, 121);
        assert_eq!(run(&block, &best, &RUNG2), vec![seq(60, 10, 48), seq(50, 8, 23)]);
        // A offBase 35 (hb 5): 21 vs 20 → the repcode match at 121 wins by 1.
        let (block, best) = rep_block(22, 32, 121);
        assert_eq!(run(&block, &best, &RUNG2), vec![seq(60, 10, 48), seq(51, 7, 1)]);

        // lazy2 step 2, ×4 rule: gain2 = mlRep*4 vs gain1 = ml*4 - highbit(offBase) + 1.
        // A offBase 35 (hb 5): 28 vs 32 - 5 + 1 = 28 → tie, keep A.
        let (block, best) = rep_block(23, 32, 122);
        assert_eq!(run(&block, &best, &LVL9), vec![seq(60, 10, 48), seq(50, 8, 35)]);
        // A offBase 73 (hb 6): 28 vs 27 → the repcode match at 122 wins by 1.
        let (block, best) = rep_block(24, 70, 122);
        assert_eq!(run(&block, &best, &LVL9), vec![seq(60, 10, 48), seq(52, 7, 1)]);
        // lazy1 never checks ip+2: A is kept.
        assert_eq!(run(&block, &best, &RUNG2), vec![seq(60, 10, 48), seq(50, 8, 73)]);
    }

    /// Catch-up extends an explicit match backwards but never past `anchor` (the end of the
    /// previous match), even though the bytes before it still match.
    #[test]
    fn catch_up_stops_at_anchor() {
        let mut block = background(31);
        plant(&mut block, 200, 40, 10); // M0 → anchor 210
        // Offset-100 run covering 205..224: 5 bytes before anchor, 2 before the search hit.
        for p in 205..224 {
            block[p - 100] = block[p];
        }
        block[124] = block[224] ^ 0xFF;
        let mut best = empty_best();
        best[200] = real(&block, 200, 40);
        best[212] = real(&block, 212, 100);
        assert_eq!((best[200].len, best[212].len), (10, 12));
        assert_eq!(block[205..210], block[105..110], "run must extend past the anchor");
        assert_eq!(run(&block, &best, &RUNG2), vec![seq(200, 10, 43), seq(0, 14, 103)]);
    }

    /// Catch-up stops when the match source reaches position 0 (`start - offset > 0`).
    #[test]
    fn catch_up_at_block_start() {
        let mut block = background(32);
        plant(&mut block, 30, 30, 12); // source 0..12
        let mut best = empty_best();
        best[33] = real(&block, 33, 30);
        assert_eq!(best[33].len, 9);
        assert_eq!(run(&block, &best, &RUNG2), vec![seq(30, 12, 33)]);
    }

    /// After a store, repeated `offset_2` matches are stored immediately as (ll 0, repcode 1),
    /// swapping offset_1 and offset_2 each time.
    #[test]
    fn immediate_offset2_repcode() {
        let mut block = background(41);
        plant(&mut block, 40, 25, 8); // M0 → reps [25, 1, 4]
        plant(&mut block, 150, 100, 8); // M1 → reps [100, 25, 1]
        plant(&mut block, 158, 25, 6); // offset_2 = 25 at the end of M1
        plant(&mut block, 164, 100, 5); // then offset_2 = 100 again
        let mut best = empty_best();
        best[40] = real(&block, 40, 25);
        best[150] = real(&block, 150, 100);
        assert_eq!((best[40].len, best[150].len), (8, 8));
        let want = vec![seq(40, 8, 28), seq(102, 8, 103), seq(0, 6, 1), seq(0, 5, 1)];
        assert_eq!(run(&block, &best, &RUNG2), want);
        assert_eq!(run(&block, &best, &LVL9), want);
    }

    /// `best[]` is only valid below PARSE_END: deferral from the last parse positions must not
    /// look at `best[PARSE_END..]`. Poisoned entries there would win every gain comparison
    /// (and are not real matches, so using them breaks the reconstruct).
    #[test]
    fn lazy_near_parse_end() {
        let pe = PARSE_END;
        let build = |with_b: bool| {
            let mut block = background(51);
            plant(&mut block, pe - 2, pe - 12, 6); // A: source at 10
            if with_b {
                plant(&mut block, pe - 1, pe - 21, 9); // B: source at 20, runs to BLOCK_SIZE
            }
            // Offset-50 run over 100..pe-20 (found capped at 100, extended by the parse),
            // resetting the anchor close to the end so the last positions are all searched.
            block[99] = block[49] ^ 0xFF;
            for p in 100..pe - 20 {
                block[p] = block[p - 50];
            }
            block[pe - 20] = block[pe - 70] ^ 0xFF;
            let mut best = empty_best();
            best[100] = Match { offset: 50, len: RUNG2.search_cap };
            best[pe - 2] = real(&block, pe - 2, pe - 12);
            if with_b {
                best[pe - 1] = real(&block, pe - 1, pe - 21);
            }
            best[pe] = Match { offset: 1, len: 8 };
            best[pe + 1] = Match { offset: 1, len: 7 };
            (block, best)
        };
        let m0 = seq(100, (pe - 120) as u32, 53);

        // A at pe-2, B at pe-1 wins step 1 (lazy1 and lazy2); deferral stops there.
        let (block, best) = build(true);
        assert_eq!((best[pe - 2].len, best[pe - 1].len), (6, 9));
        let want = vec![m0, seq(19, 9, (pe - 21 + 3) as u32)];
        assert_eq!(run(&block, &best, &RUNG2), want);
        assert_eq!(run(&block, &best, &LVL9), want);

        // A at pe-2, nothing at pe-1: lazy2's step 2 must not search pe.
        let (block, best) = build(false);
        assert_eq!(best[pe - 2].len, 6);
        let want = vec![m0, seq(18, 6, (pe - 12 + 3) as u32)];
        assert_eq!(run(&block, &best, &RUNG2), want);
        assert_eq!(run(&block, &best, &LVL9), want);
    }

    /// With literal skipping (step 2 after 256 literals) position 295 is never repcode-checked,
    /// so the search at 296 finds an explicit match whose offset equals offset_1. It is stored
    /// as repcode 1 (off_base_for), and the decoder's rep history (not zstd's
    /// `offset_2 = offset_1`) is what the parse continues with.
    #[test]
    fn explicit_match_at_rep0_uses_decoder_reps() {
        let mut block = background(61);
        plant(&mut block, 30, 20, 8); // M0 → reps [20, 1, 4], anchor 38
        plant(&mut block, 296, 20, 6);
        let mut best = empty_best();
        best[30] = real(&block, 30, 20);
        best[296] = real(&block, 296, 20);
        assert_eq!((best[30].len, best[296].len), (8, 6));
        assert_eq!(run(&block, &best, &RUNG2), vec![seq(30, 8, 23), seq(258, 6, 1)]);
    }

    /// Deviation 3 changes output: the dedup store at 296 (offset 20 == reps[0]) leaves the
    /// decoder's reps at [20, 1, 4], so offset_2 = 1 (zstd: offset_2 = offset_1 = 20). An
    /// offset-1 run right after the match is then stored by the immediate loop as
    /// (ll 0, repcode 1); zstd would probe offset 20 there, which does not match.
    #[test]
    fn dedup_store_enables_old_rep1_immediate() {
        let mut block = background(61);
        plant(&mut block, 30, 20, 8); // M0 → reps [20, 1, 4], anchor 38
        block[301..307].fill(0x5A); // offset-1 run over 301..307 (301 is inside the match)
        block[307] = 0xA5;
        plant(&mut block, 296, 20, 6);
        let mut best = empty_best();
        best[30] = real(&block, 30, 20);
        best[296] = real(&block, 296, 20);
        assert_eq!((best[30].len, best[296].len), (8, 6));
        assert_eq!(match_len(&block, 302, 301), 5, "offset-1 run at the match end");
        assert_ne!(block[302], block[302 - 20], "zstd's offset_2 (20) must not match at 302");
        let want = vec![seq(30, 8, 23), seq(258, 6, 1), seq(0, 5, 1)];
        assert_eq!(run(&block, &best, &RUNG2), want);
        assert_eq!(run(&block, &best, &LVL9), want);
    }

    /// The common deviation-3 case: a block starting with a byte run. At ip 1 the explicit
    /// offset-1 match (start 1, 9 bytes) beats the ip+1 repcode (8 bytes); catch-up cannot
    /// move (start == offset), so it is stored as (ll 1, repcode 1) with reps unchanged
    /// [1, 4, 8] and offset_2 enabled as 4 (zstd keeps it at 1 after `offset_2 = offset_1`).
    /// The offset-4 probe at the run's end cannot match, so the output equals zstd's.
    #[test]
    fn dedup_store_at_block_start_byte_run() {
        let mut block = background(71);
        block[0..10].fill(0x41);
        block[10] = 0x14;
        let mut best = empty_best();
        best[1] = real(&block, 1, 1);
        assert_eq!(best[1].len, 9);
        let want = vec![seq(1, 9, 1)];
        assert_eq!(run(&block, &best, &RUNG2), want);
        assert_eq!(run(&block, &best, &LVL9), want);
    }

    /// Chained deferral: B wins at p+1, the parse `continue`s, and at p+2 C beats B by exactly
    /// 1 under the step-1 `+4` rule. Falling through to lazy2's depth-2 check instead would
    /// apply `+7` and keep B; `break` would store B.
    #[test]
    fn lazy2_chained_deferral_uses_step1_rule() {
        let mut block = background(81);
        plant(&mut block, 200, 14, 8); // A: offBase 17
        plant(&mut block, 201, 28, 10); // B: offBase 31
        plant(&mut block, 202, 130, 12); // C: offBase 133
        let mut best = empty_best();
        best[200] = real(&block, 200, 14);
        best[201] = real(&block, 201, 28);
        best[202] = real(&block, 202, 130);
        assert_eq!((best[200].len, best[201].len, best[202].len), (8, 10, 12));
        assert!(gain(10, 31) > gain(8, 17) + 4, "B wins at p+1");
        assert_eq!(gain(12, 133), gain(10, 31) + 4 + 1, "C beats B by exactly 1 under +4");
        assert!(gain(12, 133) <= gain(10, 31) + 7, "C would lose under +7");
        let want = vec![seq(202, 12, 133)];
        assert_eq!(run(&block, &best, &LVL9), want);
        assert_eq!(run(&block, &best, &RUNG2), want);
    }

    /// A step-2 win also `continue`s: C wins at p+2 (+7 rule), then D at p+3 beats C under the
    /// step-1 `+4` rule. `break` after C's win would store C.
    #[test]
    fn lazy2_step2_win_continues() {
        let mut block = background(82);
        plant(&mut block, 200, 14, 8); // A: offBase 17
        plant(&mut block, 202, 28, 10); // C: offBase 31
        plant(&mut block, 203, 130, 12); // D: offBase 133
        let mut best = empty_best();
        best[200] = real(&block, 200, 14);
        best[202] = real(&block, 202, 28);
        best[203] = real(&block, 203, 130);
        assert_eq!((best[200].len, best[202].len, best[203].len), (8, 10, 12));
        assert!(gain(10, 31) > gain(8, 17) + 7, "C wins at p+2");
        assert!(gain(12, 133) > gain(10, 31) + 4, "D beats C at p+3");
        assert_eq!(run(&block, &best, &LVL9), vec![seq(203, 12, 133)]);
        assert_eq!(run(&block, &best, &RUNG2), vec![seq(200, 8, 17)], "lazy1 stops at p+1");
    }

    #[test]
    fn rung2_and_lvl9_roundtrip_all_cases() {
        for params in [RUNG2, LVL9] {
            for (name, bytes) in synth::test_cases() {
                for (i, blk) in chunk_file(&bytes).into_iter().enumerate() {
                    let out = compress_block(&blk.data, params);
                    let got = reconstruct(&out).unwrap_or_else(|e| panic!("lazy {} {name} block {i}: {e}", params.lazy));
                    assert_eq!(got, blk.data, "lazy {} {name} block {i}: reconstruct mismatch", params.lazy);

                    let frame = write_frame(&blk.data, &out, FrameOptions::default());
                    let dec = zstd::bulk::decompress(&frame, BLOCK_SIZE)
                        .unwrap_or_else(|e| panic!("lazy {} {name} block {i}: libzstd rejected frame: {e}", params.lazy));
                    assert_eq!(dec, blk.data, "lazy {} {name} block {i}: frame mismatch", params.lazy);
                }
            }
        }
    }

    #[test]
    fn lvl9_ratio_not_worse_than_rung1_on_text_dds_nif() {
        let mut totals = [0usize; 2];
        for (name, bytes) in synth::test_cases() {
            if !matches!(name, "text" | "dds" | "nif") {
                continue;
            }
            for blk in chunk_file(&bytes) {
                for (t, params) in totals.iter_mut().zip([RUNG1, LVL9]) {
                    let best = find_best(&blk.data, &chains(&blk.data, &params), &params);
                    let out = crate::reference::parse(&blk.data, &best, &params);
                    *t += write_frame(&blk.data, &out, FrameOptions::default()).len();
                }
            }
        }
        println!("rung1 {} bytes, lvl9 {} bytes", totals[0], totals[1]);
        assert!(totals[1] <= totals[0], "lvl9 {} bytes > rung1 {} bytes", totals[1], totals[0]);
    }
}
