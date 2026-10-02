//! Lazy / lazy2 parse: a port of libzstd 1.5.7 `ZSTD_compressBlock_lazy_generic`
//! (`lib/compress/zstd_lazy.c`, `dictMode == ZSTD_noDict`, `depth = params.lazy`).
//!
//! This is the normative oracle the GPU K3 lazy kernel mirrors bit-exactly: integer-only,
//! deterministic, no reads past `BLOCK_SIZE`. The structure follows the C function
//! statement by statement; names in comments (`ip`, `anchor`, `start`, `offBase`,
//! `matchLength`, `offset_1`, `offset_2`, `ilimit`) are zstd's.
//!
//! Substitutions, the only departures from the C function:
//! - `ZSTD_searchMax(ip)` is `best[ip]` (K2 output), extended to its true length with
//!   `match_len` when it hit `search_cap`. `len < min_match` means "no match". An explicit
//!   match has `offBase = OFFSET_TO_OFFBASE(offset) = offset + 3`.
//! - `MEM_read32(a) == MEM_read32(b)` followed by `ZSTD_count(...) + 4` is the bounded
//!   `match_len(block, a, b) >= 4` (see `rep_len`).
//! - `ilimit = iend - 8` is `PARSE_END`.
//! - Sequences are stored through `seq::off_base_for` / `seq::apply_off_base`, the same
//!   bookkeeping the greedy parse uses. `offset_1` / `offset_2` always equal the decoder's
//!   `reps[0]` / `reps[1]` (see the `// deviation:` notes for the one case where zstd's own
//!   bookkeeping would differ).
use crate::config::{BLOCK_SIZE, PARSE_END};
use crate::params::MatchParams;
use crate::reference::{match_len_capped, Match};
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
/// then `ZSTD_count(p+4, p+4-off, iend) + 4`, with `iend = lim` (the block end, or the segment
/// end of a segmented parse). Returns the match length, or 0 when the first 4 bytes differ or
/// `off` is unusable. `off > p` cannot happen for a real rep history (zstd relies on its window
/// check for that); it is guarded so nothing underflows.
fn rep_len(block: &[u8], p: usize, off: u32, lim: usize) -> u32 {
    debug_assert!(off as usize <= p, "rep offset {off} beyond position {p}");
    if off == 0 || off as usize > p {
        return 0;
    }
    let l = match_len_capped(block, p, p - off as usize, lim - p);
    if l >= 4 {
        l as u32
    } else {
        0
    }
}

/// `ZSTD_searchMax(ip)`: `(matchLength, offBase)` of `best[ip]`, or `None` when it has no
/// match of at least `min_match`. A capped length is extended to the true match length (up to
/// `seg.lim`). With `seg.clamp` (segmented parse) a match running past `seg.lim` is cut there
/// and dropped when that (or the lim-bounded extension of a capped entry) leaves fewer than
/// `min_match` bytes.
fn search_max(block: &[u8], best: &[Match], ip: usize, params: &MatchParams, seg: &Seg) -> Option<(u32, u32)> {
    debug_assert!(ip < PARSE_END, "best[] read at {ip} >= PARSE_END");
    let m = best[ip];
    if m.len < params.min_match {
        return None;
    }
    let mut len = if m.len == params.search_cap {
        match_len_capped(block, ip, ip - m.offset as usize, seg.lim - ip) as u32
    } else {
        m.len
    };
    if seg.clamp {
        // Cut at lim; a capped entry's extension is already bounded by lim and can also end
        // below min_match there.
        len = len.min((seg.lim - ip) as u32);
        if len < params.min_match {
            return None;
        }
    }
    Some((len, m.offset + ZSTD_REP_NUM))
}

/// One parsed sequence before its offset is encoded: (lit_len, match_len, offset). The literals
/// are implicit (the bytes between matches).
pub type RawSeq = (u32, u32, u32);

/// Where and how `lazy_core` parses: the byte range `[anchor0, lim)` (matches never reach past
/// `lim`), starting at `ip0` with rep history `reps0`. The whole-block parse is
/// `Seg::whole_block()`; `lazy_parse_segmented` uses `Seg::segment`.
#[derive(Clone, Copy, Debug)]
pub struct Seg {
    /// Where the parse starts.
    pub ip0: usize,
    /// Start of the parsed range: the first literal.
    pub anchor0: usize,
    /// The rep history the parse starts with.
    pub reps0: Reps,
    /// End of the parsed range: match lengths are bounded by it.
    pub lim: usize,
    /// zstd's `ilimit`: the parse loop runs while `ip < pend` (`PARSE_END` for the whole block).
    pub pend: usize,
    /// zstd's literal-run skip acceleration `step = ((ip - anchor) >> 8) + 1` (else step 1).
    pub accel: bool,
    /// Cut `best[]` matches that run past `lim` (segmented parse only; see `search_max`).
    pub clamp: bool,
}

impl Seg {
    /// Today's `lazy_parse`: the whole block, zstd's start (`ip = 1`, `INITIAL_REPS`).
    pub fn whole_block() -> Self {
        Seg { ip0: 1, anchor0: 0, reps0: INITIAL_REPS, lim: BLOCK_SIZE, pend: PARSE_END, accel: true, clamp: false }
    }

    /// Segment `k` of `1 << log2` bytes: segment 0 keeps the block start (`ip = 1`,
    /// `INITIAL_REPS`), later ones start at `ip = anchor = k << log2` with an empty rep history
    /// (all zero: no rep probe can hit until the segment's first explicit match). The last
    /// segment ends at `BLOCK_SIZE` / `PARSE_END`, the others at their end / end - 4. No skip
    /// acceleration.
    pub fn segment(k: usize, log2: u32) -> Self {
        let n = BLOCK_SIZE >> log2;
        assert!(k < n, "segment {k} of {n}");
        let s = k << log2;
        let last = k + 1 == n;
        let lim = if last { BLOCK_SIZE } else { s + (1 << log2) };
        let pend = if last { PARSE_END } else { lim - 4 };
        let (ip0, reps0) = if k == 0 { (1, INITIAL_REPS) } else { (s, [0; 3]) };
        Seg { ip0, anchor0: s, reps0, lim, pend, accel: false, clamp: true }
    }
}

/// Appends one raw sequence (`lit_len` literals from the anchor, then `ml` bytes at `offset`)
/// and applies the decoder's rep bookkeeping to the parse's own history `reps`.
fn store(out: &mut Vec<RawSeq>, reps: &mut Reps, lit_len: usize, offset: u32, ml: u32) {
    let ll = lit_len as u32;
    let ob = off_base_for(offset, ll, reps);
    apply_off_base(reps, ob, ll);
    out.push((ll, ml, offset));
}

/// Lazy / lazy2 parse over precomputed best[]. params.lazy ∈ {1,2}.
/// Whole-block parse: `lazy_core` over `Seg::whole_block()`, then `encode_raw`.
pub fn lazy_parse(block: &[u8], best: &[Match], params: &MatchParams) -> BlockOutput {
    let mut raw = Vec::new();
    lazy_core(block, best, params, &Seg::whole_block(), &mut raw);
    encode_raw(block, &raw)
}

/// The segmented lazy / lazy2 parse (`params.segment_log2 > 0`, preset `lvl9seg`): each
/// `Seg::segment` is parsed on its own by `lazy_core`; the raw sequences are concatenated, each
/// segment's trailing literals carried into the first sequence of the next non-empty segment,
/// and `encode_raw` assigns every `off_base` from the block's true decoder rep history.
pub fn lazy_parse_segmented(block: &[u8], best: &[Match], params: &MatchParams) -> BlockOutput {
    let log2 = params.segment_log2;
    assert!(log2 > 0, "lazy_parse_segmented: segment_log2 0");
    let mut raw = Vec::new();
    // End of the last sequence so far (start of the pending literal run).
    let mut prev_end = 0usize;
    let mut part = Vec::new();
    for k in 0..BLOCK_SIZE >> log2 {
        let seg = Seg::segment(k, log2);
        part.clear();
        let end = lazy_core(block, best, params, &seg, &mut part);
        if let Some(first) = part.first_mut() {
            first.0 += (seg.anchor0 - prev_end) as u32;
            prev_end = end;
        }
        raw.extend_from_slice(&part);
    }
    encode_raw(block, &raw)
}

/// Encodes raw sequences with the decoder's rep history from `INITIAL_REPS` (`off_base_for` /
/// `apply_off_base`) and gathers the literals they leave uncovered.
pub fn encode_raw(block: &[u8], raw: &[RawSeq]) -> BlockOutput {
    let mut out = BlockOutput::default();
    let mut reps: Reps = INITIAL_REPS;
    let mut pos = 0usize;
    for &(ll, ml, offset) in raw {
        let ob = off_base_for(offset, ll, &reps);
        apply_off_base(&mut reps, ob, ll);
        out.literals.extend_from_slice(&block[pos..pos + ll as usize]);
        out.sequences.push(Sequence { lit_len: ll, match_len: ml, off_base: ob });
        pos += (ll + ml) as usize;
    }
    // last literals
    out.literals.extend_from_slice(&block[pos..BLOCK_SIZE]);
    out
}

/// The lazy / lazy2 parse of `seg` (see `Seg`), appending raw sequences to `out`; returns the
/// final anchor (the end of the last sequence, or `seg.anchor0` if none). `lit_len` of the first
/// sequence counts from `seg.anchor0`.
pub fn lazy_core(block: &[u8], best: &[Match], params: &MatchParams, seg: &Seg, out: &mut Vec<RawSeq>) -> usize {
    assert!(params.lazy == 1 || params.lazy == 2, "lazy_parse: lazy {} not in 1..=2", params.lazy);
    assert_eq!(block.len(), BLOCK_SIZE);
    assert!(best.len() >= PARSE_END);
    assert!(seg.ip0 >= seg.anchor0 && seg.lim <= BLOCK_SIZE && seg.pend <= PARSE_END.min(seg.lim), "{seg:?}");
    let depth = params.lazy;
    let ilimit = seg.pend;
    let lim = seg.lim;

    let mut reps: Reps = seg.reps0;
    let mut anchor = seg.anchor0;
    // zstd: `ip += (dictAndPrefixLength == 0)`. Every block is a fresh frame, so the parse
    // starts at 1 (a later segment starts at its anchor).
    let mut ip = seg.ip0;
    // zstd noDict: offsets larger than `maxRep = curr - windowLow` (= ip = 1 here) start
    // disabled as 0. INITIAL_REPS = [1, 4, 8], so offset_1 = 1 is live and offset_2 is 0
    // (off) until the first explicit match sets it. (A later segment's reps are all 0.)
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
        let l = rep_len(block, ip + 1, offset_1, lim);
        if l > 0 {
            match_length = l;
        }

        // first search (depth 0)
        if let Some((ml2, ob)) = search_max(block, best, ip, params, seg)
            && ml2 > match_length
        {
            match_length = ml2;
            start = ip;
            off_base = ob;
        }

        if match_length < 4 {
            // jump faster over incompressible sections
            ip += if seg.accel { ((ip - anchor) >> K_SEARCH_STRENGTH) + 1 } else { 1 };
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
            let ml_rep = rep_len(block, ip, offset_1, lim);
            if ml_rep >= 4 {
                let gain2 = (ml_rep * 3) as i32;
                let gain1 = (match_length * 3) as i32 - highbit32(off_base) + 1;
                if gain2 > gain1 {
                    match_length = ml_rep;
                    off_base = REPCODE1_TO_OFFBASE;
                    start = ip;
                }
            }
            if let Some((ml2, ob)) = search_max(block, best, ip, params, seg) {
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
                let ml_rep = rep_len(block, ip, offset_1, lim);
                if ml_rep >= 4 {
                    let gain2 = (ml_rep * 4) as i32;
                    let gain1 = (match_length * 4) as i32 - highbit32(off_base) + 1;
                    if gain2 > gain1 {
                        match_length = ml_rep;
                        off_base = REPCODE1_TO_OFFBASE;
                        start = ip;
                    }
                }
                if let Some((ml2, ob)) = search_max(block, best, ip, params, seg) {
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
            store(out, &mut reps, lit_len, offset, match_length);
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
            store(out, &mut reps, lit_len, offset_1, match_length);
            debug_assert_eq!(reps, before);
        }
        anchor = start + match_length as usize;
        ip = anchor;

        // check immediate repcode: an offset_2 match at ip is stored as (ll 0, repcode 1),
        // which the decoder resolves to reps[1] and swaps into reps[0]: zstd's swap.
        while ip <= ilimit && offset_2 > 0 {
            let ml = rep_len(block, ip, offset_2, lim);
            if ml == 0 {
                break;
            }
            (offset_1, offset_2) = (offset_2, offset_1); // swap repcodes
            store(out, &mut reps, 0, offset_1, ml);
            debug_assert_eq!((reps[0], reps[1]), (offset_1, offset_2));
            ip += ml as usize;
            anchor = ip;
        }
    }
    anchor
}


/// Hand-built `best[]` cases pinning each branch of `lazy_parse` (gain boundaries, `continue`
/// after a win, catch-up bounds, the dedup-store rep rule, the immediate offset_2 loop and the
/// PARSE_END guards). Public (but hidden) so the GPU K3 tests replay exactly the same cases;
/// every builder asserts its own gain arithmetic, so a case cannot drift silently.
#[doc(hidden)]
pub mod cases {
    use crate::config::{BLOCK_SIZE, PARSE_END};
    use crate::fixtures::{LVL9, RUNG2};
    use crate::params::{MatchParams, LVL9SEG};
    use crate::reference::{match_len, Match};
    use crate::seq::Sequence;
    use crate::synth;

    /// One block with its scripted `best[]` (BLOCK_SIZE entries) and the exact sequences
    /// `lazy_parse` must produce for each listed params.
    pub struct LazyCase {
        pub name: String,
        pub block: Vec<u8>,
        pub best: Vec<Match>,
        pub expect: Vec<(MatchParams, Vec<Sequence>)>,
    }

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

    fn empty_best() -> Vec<Match> {
        vec![Match::default(); BLOCK_SIZE]
    }

    fn case(name: &str, block: Vec<u8>, best: Vec<Match>, expect: Vec<(MatchParams, Vec<Sequence>)>) -> LazyCase {
        LazyCase { name: name.to_string(), block, best, expect }
    }

    /// Step 1: the match at ip+1 wins by exactly 1 over `gain1 + 4` → the parse defers.
    pub fn lazy_prefers_later_longer_match() -> Vec<LazyCase> {
        let mut block = background(11);
        plant(&mut block, 200, 20, 8); // A: offBase 23, highbit 4
        plant(&mut block, 201, 130, 10); // B: offBase 133, highbit 7
        let mut best = empty_best();
        best[200] = real(&block, 200, 20);
        best[201] = real(&block, 201, 130);
        assert_eq!((best[200].len, best[201].len), (8, 10));
        assert_eq!(gain(10, 133), gain(8, 23) + 4 + 1, "B must win by exactly 1");
        vec![case("lazy_prefers_later_longer_match", block, best, vec![(RUNG2, vec![seq(201, 10, 133)])])]
    }

    /// Step 1: equal gains → the current match is kept.
    pub fn lazy_keeps_on_gain_tie() -> Vec<LazyCase> {
        let mut block = background(12);
        plant(&mut block, 200, 14, 8); // A: offBase 17, highbit 4
        plant(&mut block, 201, 28, 9); // B: offBase 31, highbit 4
        let mut best = empty_best();
        best[200] = real(&block, 200, 14);
        best[201] = real(&block, 201, 28);
        assert_eq!((best[200].len, best[201].len), (8, 9));
        assert_eq!(gain(9, 31), gain(8, 17) + 4, "B must tie");
        vec![case("lazy_keeps_on_gain_tie", block, best, vec![(RUNG2, vec![seq(200, 8, 17)])])]
    }

    /// Step 2 (lazy2): a match at ip+2 must beat `gain1 + 7`. Winning by 1 defers; a tie keeps.
    /// lazy1 never looks at ip+2.
    pub fn lazy2_second_step_threshold() -> Vec<LazyCase> {
        // Win by 1: A at 200 (offBase 17, hb 4), nothing at 201, C at 202 (offBase 31, hb 4).
        let mut block = background(13);
        plant(&mut block, 200, 14, 8);
        plant(&mut block, 202, 28, 10);
        let mut best = empty_best();
        best[200] = real(&block, 200, 14);
        best[202] = real(&block, 202, 28);
        assert_eq!((best[200].len, best[202].len), (8, 10));
        assert_eq!(gain(10, 31), gain(8, 17) + 7 + 1);
        // lazy1 must not search ip+2.
        let win = case(
            "lazy2_second_step_threshold/win",
            block,
            best,
            vec![(LVL9, vec![seq(202, 10, 31)]), (RUNG2, vec![seq(200, 8, 17)])],
        );

        // Tie: C has offBase 43 (hb 5).
        let mut block = background(14);
        plant(&mut block, 200, 14, 8);
        plant(&mut block, 202, 40, 10);
        let mut best = empty_best();
        best[200] = real(&block, 200, 14);
        best[202] = real(&block, 202, 40);
        assert_eq!((best[200].len, best[202].len), (8, 10));
        assert_eq!(gain(10, 43), gain(8, 17) + 7);
        let tie = case("lazy2_second_step_threshold/tie", block, best, vec![(LVL9, vec![seq(200, 8, 17)])]);
        vec![win, tie]
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

    pub fn lazy_rep_check_at_ip_plus_1() -> Vec<LazyCase> {
        let mut out = Vec::new();
        // ×3 rule: gain2 = mlRep*3 vs gain1 = ml*3 - highbit(offBase) + 1.
        // A offBase 23 (hb 4): 21 vs 24 - 4 + 1 = 21 → tie, keep A.
        let (block, best) = rep_block(21, 20, 121);
        out.push(case("lazy_rep_check/x3_tie", block, best, vec![(RUNG2, vec![seq(60, 10, 48), seq(50, 8, 23)])]));
        // A offBase 35 (hb 5): 21 vs 20 → the repcode match at 121 wins by 1.
        let (block, best) = rep_block(22, 32, 121);
        out.push(case("lazy_rep_check/x3_win", block, best, vec![(RUNG2, vec![seq(60, 10, 48), seq(51, 7, 1)])]));

        // lazy2 step 2, ×4 rule: gain2 = mlRep*4 vs gain1 = ml*4 - highbit(offBase) + 1.
        // A offBase 35 (hb 5): 28 vs 32 - 5 + 1 = 28 → tie, keep A.
        let (block, best) = rep_block(23, 32, 122);
        out.push(case("lazy_rep_check/x4_tie", block, best, vec![(LVL9, vec![seq(60, 10, 48), seq(50, 8, 35)])]));
        // A offBase 73 (hb 6): 28 vs 27 → the repcode match at 122 wins by 1; lazy1 never
        // checks ip+2, so it keeps A.
        let (block, best) = rep_block(24, 70, 122);
        out.push(case(
            "lazy_rep_check/x4_win",
            block,
            best,
            vec![(LVL9, vec![seq(60, 10, 48), seq(52, 7, 1)]), (RUNG2, vec![seq(60, 10, 48), seq(50, 8, 73)])],
        ));
        out
    }

    /// Catch-up extends an explicit match backwards but never past `anchor` (the end of the
    /// previous match), even though the bytes before it still match.
    pub fn catch_up_stops_at_anchor() -> Vec<LazyCase> {
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
        vec![case("catch_up_stops_at_anchor", block, best, vec![(RUNG2, vec![seq(200, 10, 43), seq(0, 14, 103)])])]
    }

    /// Catch-up stops when the match source reaches position 0 (`start - offset > 0`).
    pub fn catch_up_at_block_start() -> Vec<LazyCase> {
        let mut block = background(32);
        plant(&mut block, 30, 30, 12); // source 0..12
        let mut best = empty_best();
        best[33] = real(&block, 33, 30);
        assert_eq!(best[33].len, 9);
        vec![case("catch_up_at_block_start", block, best, vec![(RUNG2, vec![seq(30, 12, 33)])])]
    }

    /// After a store, repeated `offset_2` matches are stored immediately as (ll 0, repcode 1),
    /// swapping offset_1 and offset_2 each time.
    pub fn immediate_offset2_repcode() -> Vec<LazyCase> {
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
        vec![case("immediate_offset2_repcode", block, best, vec![(RUNG2, want.clone()), (LVL9, want)])]
    }

    /// `best[]` is only valid below PARSE_END: deferral from the last parse positions must not
    /// look at `best[PARSE_END..]`. Poisoned entries there would win every gain comparison
    /// (and are not real matches, so using them breaks the reconstruct).
    pub fn lazy_near_parse_end() -> Vec<LazyCase> {
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
        let with_b = case("lazy_near_parse_end/b", block, best, vec![(RUNG2, want.clone()), (LVL9, want)]);

        // A at pe-2, nothing at pe-1: lazy2's step 2 must not search pe.
        let (block, best) = build(false);
        assert_eq!(best[pe - 2].len, 6);
        let want = vec![m0, seq(18, 6, (pe - 12 + 3) as u32)];
        let without_b = case("lazy_near_parse_end/no_b", block, best, vec![(RUNG2, want.clone()), (LVL9, want)]);
        vec![with_b, without_b]
    }

    /// With literal skipping (step 2 after 256 literals) position 295 is never repcode-checked,
    /// so the search at 296 finds an explicit match whose offset equals offset_1. It is stored
    /// as repcode 1 (off_base_for), and the decoder's rep history (not zstd's
    /// `offset_2 = offset_1`) is what the parse continues with.
    pub fn explicit_match_at_rep0_uses_decoder_reps() -> Vec<LazyCase> {
        let mut block = background(61);
        plant(&mut block, 30, 20, 8); // M0 → reps [20, 1, 4], anchor 38
        plant(&mut block, 296, 20, 6);
        let mut best = empty_best();
        best[30] = real(&block, 30, 20);
        best[296] = real(&block, 296, 20);
        assert_eq!((best[30].len, best[296].len), (8, 6));
        vec![case(
            "explicit_match_at_rep0_uses_decoder_reps",
            block,
            best,
            vec![(RUNG2, vec![seq(30, 8, 23), seq(258, 6, 1)])],
        )]
    }

    /// Deviation 3 changes output: the dedup store at 296 (offset 20 == `reps[0]`) leaves the
    /// decoder's reps at [20, 1, 4], so offset_2 = 1 (zstd: offset_2 = offset_1 = 20). An
    /// offset-1 run right after the match is then stored by the immediate loop as
    /// (ll 0, repcode 1); zstd would probe offset 20 there, which does not match.
    pub fn dedup_store_enables_old_rep1_immediate() -> Vec<LazyCase> {
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
        vec![case("dedup_store_enables_old_rep1_immediate", block, best, vec![(RUNG2, want.clone()), (LVL9, want)])]
    }

    /// The common deviation-3 case: a block starting with a byte run. At ip 1 the explicit
    /// offset-1 match (start 1, 9 bytes) beats the ip+1 repcode (8 bytes); catch-up cannot
    /// move (start == offset), so it is stored as (ll 1, repcode 1) with reps unchanged
    /// [1, 4, 8] and offset_2 enabled as 4 (zstd keeps it at 1 after `offset_2 = offset_1`).
    /// The offset-4 probe at the run's end cannot match, so the output equals zstd's.
    pub fn dedup_store_at_block_start_byte_run() -> Vec<LazyCase> {
        let mut block = background(71);
        block[0..10].fill(0x41);
        block[10] = 0x14;
        let mut best = empty_best();
        best[1] = real(&block, 1, 1);
        assert_eq!(best[1].len, 9);
        let want = vec![seq(1, 9, 1)];
        vec![case("dedup_store_at_block_start_byte_run", block, best, vec![(RUNG2, want.clone()), (LVL9, want)])]
    }

    /// Chained deferral: B wins at p+1, the parse `continue`s, and at p+2 C beats B by exactly
    /// 1 under the step-1 `+4` rule. Falling through to lazy2's depth-2 check instead would
    /// apply `+7` and keep B; `break` would store B.
    pub fn lazy2_chained_deferral_uses_step1_rule() -> Vec<LazyCase> {
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
        vec![case("lazy2_chained_deferral_uses_step1_rule", block, best, vec![(LVL9, want.clone()), (RUNG2, want)])]
    }

    /// A step-2 win also `continue`s: C wins at p+2 (+7 rule), then D at p+3 beats C under the
    /// step-1 `+4` rule. `break` after C's win would store C.
    pub fn lazy2_step2_win_continues() -> Vec<LazyCase> {
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
        // lazy1 stops at p+1.
        vec![case(
            "lazy2_step2_win_continues",
            block,
            best,
            vec![(LVL9, vec![seq(203, 12, 133)]), (RUNG2, vec![seq(200, 8, 17)])],
        )]
    }

    /// The immediate offset_2 loop runs while `ip <= PARSE_END` (zstd: `ip <= ilimit`): a store
    /// ending exactly at PARSE_END is followed by a 6-byte offset_2 repeat there, stored as
    /// (ll 0, repcode 1); only the last 2 bytes stay literals.
    pub fn immediate_offset2_at_parse_end() -> Vec<LazyCase> {
        let pe = PARSE_END;
        let mut block = background(91);
        // Offset-50 run over 100..pe-60 (found capped at 100): M0 → reps [50, 1, 4], anchor pe-60.
        block[99] = block[49] ^ 0xFF;
        for p in 100..pe - 60 {
            block[p] = block[p - 50];
        }
        block[pe - 60] = block[pe - 110] ^ 0xFF;
        plant(&mut block, pe, 50, 6); // offset_2 (= 50 after M1) repeat at PARSE_END
        plant(&mut block, pe - 10, 30, 10); // M1: ends at PARSE_END → reps [30, 50, 1]
        let mut best = empty_best();
        best[100] = Match { offset: 50, len: RUNG2.search_cap };
        best[pe - 10] = real(&block, pe - 10, 30);
        assert_eq!(best[pe - 10].len, 10);
        assert_eq!(match_len(&block, pe, pe - 50), 6);
        assert!(match_len(&block, pe - 60, pe - 61) < 4, "no offset-1 repeat after M0");
        let want = vec![seq(100, (pe - 160) as u32, 53), seq(50, 10, 33), seq(0, 6, 1)];
        vec![case("immediate_offset2_at_parse_end", block, best, vec![(RUNG2, want.clone()), (LVL9, want)])]
    }

    /// First search: `best[ip]` replaces the ip+1 repcode only when strictly longer. On a length
    /// tie the deferral loop normally re-finds the repcode at ip+1 (x3 rule), which hides the
    /// tie rule; at ip = PARSE_END-1 the deferral loop does not run, so the tie is visible: the
    /// repcode at PARSE_END (6 bytes) is kept over the 6-byte explicit match at PARSE_END-1.
    pub fn depth0_tie_keeps_repcode_at_parse_end() -> Vec<LazyCase> {
        let pe = PARSE_END;
        let mut block = background(92);
        // Offset-50 run over 100..pe-60 (found capped at 100): M0 → offset_1 = 50, anchor pe-60.
        block[99] = block[49] ^ 0xFF;
        for p in 100..pe - 60 {
            block[p] = block[p - 50];
        }
        block[pe - 60] = block[pe - 110] ^ 0xFF;
        plant(&mut block, pe, 50, 6); // repcode (offset_1 = 50) match at PARSE_END
        plant(&mut block, pe - 1, 30, 6); // explicit match at PARSE_END-1, same length
        let mut best = empty_best();
        best[100] = Match { offset: 50, len: RUNG2.search_cap };
        best[pe - 1] = real(&block, pe - 1, 30);
        assert_eq!(best[pe - 1].len, 6);
        assert_eq!(match_len(&block, pe, pe - 50), 6);
        assert!(match_len(&block, pe - 60, pe - 61) < 4, "no offset-1 repeat after M0");
        let want = vec![seq(100, (pe - 160) as u32, 53), seq(60, 6, 1)];
        vec![case("depth0_tie_keeps_repcode_at_parse_end", block, best, vec![(RUNG2, want.clone()), (LVL9, want)])]
    }

    /// RUNG2 with `min_match` raised to 6: the repcode check at ip+1 only needs 4 bytes
    /// (`rep_len`'s hardcoded floor) whatever `min_match` is, so a byte run of exactly 6 equal
    /// bytes at the block start (with no explicit match, `best[]` entirely empty) is still
    /// caught by the offset_1 = 1 repeat at ip+1: `match_length` 4 covering positions 2..6,
    /// `start` stays at ip+1 = 2 (the depth-0 first search at ip = 1 finds nothing, since
    /// `best[1]` is empty and every other `best[]` entry too), so the store is `(ll 2, rep1, 4)`.
    /// The expected output comes from running the CPU oracle (`lazy_parse`), not from a hand
    /// derivation.
    pub fn rung2_min_match6_byte_run_at_start() -> Vec<LazyCase> {
        let mut block = background(101);
        block[0..6].fill(0x77);
        block[6] = !0x77u8; // guaranteed different from the run byte
        let best = empty_best();
        let params = MatchParams { min_match: 6, ..RUNG2 };
        assert_eq!(match_len(&block, 2, 1), 4, "the offset-1 run at ip+1 is exactly 4 bytes");
        let want = vec![seq(2, 4, 1)];
        vec![case("rung2_min_match6_byte_run_at_start", block, best, vec![(params, want)])]
    }

    /// Segment size of `LVL9SEG` (4 KiB).
    const SEG: usize = 1 << LVL9SEG.segment_log2;

    /// A match running across the end of segment 0 is cut at the segment end (length 6 of 20);
    /// segment 1 then starts with the rest (ll 0), which the true reps store explicitly (offset
    /// == `reps[0]` with ll 0 is not a repcode).
    pub fn seg_match_clamped_at_segment_end() -> Vec<LazyCase> {
        let mut block = background(31);
        plant(&mut block, SEG - 6, 100, 20);
        let mut best = empty_best();
        best[SEG - 6] = real(&block, SEG - 6, 100);
        best[SEG] = real(&block, SEG, 100);
        assert_eq!((best[SEG - 6].len, best[SEG].len), (20, 14));
        let want = vec![seq(SEG as u32 - 6, 6, 103), seq(0, 14, 103)];
        vec![case("seg_match_clamped_at_segment_end", block, best, vec![(LVL9SEG, want)])]
    }

    /// min_match 6: a match 5 bytes before the segment end is cut below min_match and dropped;
    /// segment 1 takes the rest, its literals carried over the whole of segment 0.
    pub fn seg_clamp_drops_short_match() -> Vec<LazyCase> {
        let p = MatchParams { min_match: 6, ..LVL9SEG };
        let mut block = background(32);
        plant(&mut block, SEG - 5, 100, 20);
        let mut best = empty_best();
        best[SEG - 5] = real(&block, SEG - 5, 100);
        best[SEG] = real(&block, SEG, 100);
        assert_eq!(best[SEG].len, 15);
        vec![case("seg_clamp_drops_short_match", block, best, vec![(p, vec![seq(SEG as u32, 15, 103)])])]
    }

    /// min_match 6: a capped best[] entry 5 bytes before the segment end, whose extension stops
    /// at lim with 5 bytes (< min_match), is dropped; segment 1 takes the rest (capped too).
    pub fn seg_capped_extension_clamped_below_min_match() -> Vec<LazyCase> {
        let p = MatchParams { min_match: 6, ..LVL9SEG };
        let cap = p.search_cap;
        let mut block = background(35);
        plant(&mut block, SEG - 5, 100, 70);
        let mut best = empty_best();
        for q in [SEG - 5, SEG] {
            let m = real(&block, q, 100);
            best[q] = Match { offset: m.offset, len: m.len.min(cap) };
        }
        assert_eq!((best[SEG - 5].len, best[SEG].len, real(&block, SEG, 100).len), (cap, cap, 65));
        let want = vec![seq(SEG as u32, 65, 103)];
        vec![case("seg_capped_extension_clamped_below_min_match", block, best, vec![(p, want)])]
    }

    /// Segments 1 and 2 are empty: the literals after segment 0's match run into segment 3's
    /// first sequence.
    pub fn seg_empty_segments_carry_literals() -> Vec<LazyCase> {
        let mut block = background(33);
        plant(&mut block, 100, 50, 10);
        let far = 3 * SEG + 200;
        plant(&mut block, far, 300, 12);
        let mut best = empty_best();
        best[100] = real(&block, 100, 50);
        best[far] = real(&block, far, 300);
        let want = vec![seq(100, 10, 53), seq((far - 110) as u32, 12, 303)];
        vec![case("seg_empty_segments_carry_literals", block, best, vec![(LVL9SEG, want)])]
    }

    /// Segment 1 parses with empty reps (its first match is explicit there), but the true
    /// history makes it repcode 1 (ll > 0, offset == `reps[0]`); and a segment starting with a
    /// match at its first byte (ll 0) at the true `reps[1]` becomes repcode 1 as well.
    pub fn seg_true_reps_across_segments() -> Vec<LazyCase> {
        let mut block = background(34);
        plant(&mut block, 100, 50, 10);
        plant(&mut block, SEG - 6, 70, 6);
        plant(&mut block, SEG, 50, 10);
        plant(&mut block, 2 * SEG + 300, 50, 10);
        let mut best = empty_best();
        for p in [100, SEG, 2 * SEG + 300] {
            best[p] = real(&block, p, 50);
        }
        best[SEG - 6] = real(&block, SEG - 6, 70);
        assert_eq!(best[SEG - 6].len, 6);
        let want = vec![
            seq(100, 10, 53),
            seq(SEG as u32 - 116, 6, 73),
            // reps [70, 50, 1]: ll 0 and offset reps[1] → repcode 1, reps [50, 70, 1]
            seq(0, 10, 1),
            // ll > 0 and offset reps[0] → repcode 1
            seq(SEG as u32 + 300 - 10, 10, 1),
        ];
        vec![case("seg_true_reps_across_segments", block, best, vec![(LVL9SEG, want)])]
    }

    /// Every segmented-parse case above, in order.
    pub fn segment_test_cases() -> Vec<LazyCase> {
        [
            seg_match_clamped_at_segment_end,
            seg_clamp_drops_short_match,
            seg_capped_extension_clamped_below_min_match,
            seg_empty_segments_carry_literals,
            seg_true_reps_across_segments,
        ]
        .into_iter()
        .flat_map(|f| f())
        .collect()
    }

    /// Every hand-built case above, in order.
    pub fn lazy_test_cases() -> Vec<LazyCase> {
        [
            lazy_prefers_later_longer_match,
            lazy_keeps_on_gain_tie,
            lazy2_second_step_threshold,
            lazy_rep_check_at_ip_plus_1,
            catch_up_stops_at_anchor,
            catch_up_at_block_start,
            immediate_offset2_repcode,
            immediate_offset2_at_parse_end,
            depth0_tie_keeps_repcode_at_parse_end,
            lazy_near_parse_end,
            explicit_match_at_rep0_uses_decoder_reps,
            dedup_store_enables_old_rep1_immediate,
            dedup_store_at_block_start_byte_run,
            lazy2_chained_deferral_uses_step1_rule,
            lazy2_step2_win_continues,
            rung2_min_match6_byte_run_at_start,
        ]
        .into_iter()
        .flat_map(|f| f())
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::cases::{self, LazyCase};
    use super::*;
    use crate::block::chunk_file;
    use crate::frame::{write_frame, FrameOptions};
    use crate::fixtures::{LVL9, RUNG1, RUNG2};
    use crate::params::LVL9SEG;
    use crate::reference::{chains, compress_block, find_best};
    use crate::seq::reconstruct;
    use crate::synth;

    /// Parses every case with each of its params, checks the exact sequences and that the
    /// output decodes back to the block.
    fn check(cases: Vec<LazyCase>) {
        for c in cases {
            for (params, want) in &c.expect {
                let out = crate::reference::parse(&c.block, &c.best, params);
                let got = reconstruct(&out).expect("reconstruct");
                assert_eq!(got, c.block, "{} lazy {}: output does not reconstruct the block", c.name, params.lazy);
                assert_eq!(out.sequences, *want, "{} lazy {}", c.name, params.lazy);
            }
        }
    }

    #[test]
    fn lazy_prefers_later_longer_match() {
        check(cases::lazy_prefers_later_longer_match());
    }

    #[test]
    fn lazy_keeps_on_gain_tie() {
        check(cases::lazy_keeps_on_gain_tie());
    }

    #[test]
    fn lazy2_second_step_threshold() {
        check(cases::lazy2_second_step_threshold());
    }

    #[test]
    fn lazy_rep_check_at_ip_plus_1() {
        check(cases::lazy_rep_check_at_ip_plus_1());
    }

    #[test]
    fn catch_up_stops_at_anchor() {
        check(cases::catch_up_stops_at_anchor());
    }

    #[test]
    fn catch_up_at_block_start() {
        check(cases::catch_up_at_block_start());
    }

    #[test]
    fn immediate_offset2_repcode() {
        check(cases::immediate_offset2_repcode());
    }

    #[test]
    fn immediate_offset2_at_parse_end() {
        check(cases::immediate_offset2_at_parse_end());
    }

    #[test]
    fn depth0_tie_keeps_repcode_at_parse_end() {
        check(cases::depth0_tie_keeps_repcode_at_parse_end());
    }

    #[test]
    fn lazy_near_parse_end() {
        check(cases::lazy_near_parse_end());
    }

    #[test]
    fn explicit_match_at_rep0_uses_decoder_reps() {
        check(cases::explicit_match_at_rep0_uses_decoder_reps());
    }

    #[test]
    fn dedup_store_enables_old_rep1_immediate() {
        check(cases::dedup_store_enables_old_rep1_immediate());
    }

    #[test]
    fn dedup_store_at_block_start_byte_run() {
        check(cases::dedup_store_at_block_start_byte_run());
    }

    #[test]
    fn lazy2_chained_deferral_uses_step1_rule() {
        check(cases::lazy2_chained_deferral_uses_step1_rule());
    }

    #[test]
    fn lazy2_step2_win_continues() {
        check(cases::lazy2_step2_win_continues());
    }

    #[test]
    fn rung2_min_match6_byte_run_at_start() {
        check(cases::rung2_min_match6_byte_run_at_start());
    }

    /// `lazy_test_cases` (what the GPU tests replay) is exactly the union of the cases above.
    #[test]
    fn lazy_test_cases_lists_every_case() {
        let all = cases::lazy_test_cases();
        assert_eq!(all.len(), 21);
        let mut names: Vec<&str> = all.iter().map(|c| c.name.as_str()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), all.len(), "case names are unique");
        assert!(all.iter().all(|c| c.best.len() == BLOCK_SIZE && c.block.len() == BLOCK_SIZE && !c.expect.is_empty()));
    }

    #[test]
    fn seg_match_clamped_at_segment_end() {
        check(cases::seg_match_clamped_at_segment_end());
    }

    #[test]
    fn seg_clamp_drops_short_match() {
        check(cases::seg_clamp_drops_short_match());
    }

    #[test]
    fn seg_empty_segments_carry_literals() {
        check(cases::seg_empty_segments_carry_literals());
    }

    #[test]
    fn seg_true_reps_across_segments() {
        check(cases::seg_true_reps_across_segments());
    }

    #[test]
    fn seg_capped_extension_clamped_below_min_match() {
        check(cases::seg_capped_extension_clamped_below_min_match());
    }

    #[test]
    fn segment_test_cases_lists_every_case() {
        let all = cases::segment_test_cases();
        assert_eq!(all.len(), 5);
        assert!(all.iter().all(|c| c.expect.iter().all(|(p, _)| p.segment_log2 > 0)));
    }

    /// `lazy_parse` is `lazy_core` over the whole block followed by `encode_raw`, and a
    /// one-segment `lazy_core` run from the block start with `Seg::whole_block` limits but no
    /// acceleration equals the segmented parse with `segment_log2 == LOG2_BLOCK`.
    #[test]
    fn one_segment_is_the_unaccelerated_whole_block_parse() {
        use crate::config::LOG2_BLOCK;
        let p = MatchParams { segment_log2: LOG2_BLOCK, ..LVL9 };
        for (name, bytes) in synth::test_cases() {
            for blk in chunk_file(&bytes) {
                let best = find_best(&blk.data, &chains(&blk.data, &p), &p);
                let mut raw = Vec::new();
                let seg = Seg { accel: false, clamp: true, ..Seg::whole_block() };
                let end = lazy_core(&blk.data, &best, &p, &seg, &mut raw);
                assert_eq!(end, raw.iter().map(|q| (q.0 + q.1) as usize).sum::<usize>(), "{name}");
                assert_eq!(encode_raw(&blk.data, &raw), lazy_parse_segmented(&blk.data, &best, &p), "{name}");
            }
        }
    }

    /// The segmented parse (4 KiB lazy2 = lvl9seg, and 1 KiB lazy1) round-trips through libzstd,
    /// and no match crosses a segment boundary.
    #[test]
    fn segmented_roundtrip_and_matches_stay_in_their_segment() {
        for params in [LVL9SEG, MatchParams { segment_log2: 10, ..RUNG2 }] {
            let seg = 1usize << params.segment_log2;
            for (name, bytes) in synth::test_cases() {
                for (i, blk) in chunk_file(&bytes).into_iter().enumerate() {
                    let out = compress_block(&blk.data, params);
                    let mut pos = 0usize;
                    for q in &out.sequences {
                        let start = pos + q.lit_len as usize;
                        pos = start + q.match_len as usize;
                        assert_eq!(start / seg, (pos - 1) / seg, "{name}[{i}]: match {start}..{pos} crosses a segment");
                    }
                    let frame = write_frame(&blk.data, &out, FrameOptions::default());
                    let dec = zstd::bulk::decompress(&frame, BLOCK_SIZE)
                        .unwrap_or_else(|e| panic!("{params:?} {name}[{i}]: libzstd rejected frame: {e}"));
                    assert_eq!(dec, blk.data, "{params:?} {name}[{i}]: frame mismatch");
                }
            }
        }
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
