//! Optimal parse for the M5 presets `opt14` / `opt16`: an integer port of libzstd 1.5.7
//! `ZSTD_compressBlock_opt_generic` (`lib/compress/zstd_opt.c`, noDict, one block), design in
//! `docs/superpowers/m5/m5-opt-design.md` §1–§2.
//!
//! This is the normative oracle the GPU K3opt kernel mirrors bit-exactly: integer-only,
//! deterministic, every tie rule explicit. The DP body follows the C function statement by
//! statement; names in comments (`ip`, `anchor`, `cur`, `last_pos`, `lastStretch`, `opt[]`,
//! `sufficient_len`, `ll0`) are zstd's.
//!
//! Substitutions (all measured in the design, §2.6):
//! - `ZSTD_insertBtAndGetAllMatches` is `get_all_matches`: zstd's three rep probes, then the two
//!   `find_cands` records `A` (nearest >= 3 bytes) and `B` (longest) instead of the binary tree
//!   and the hash3 slot. No `nextToUpdate` skipped area.
//! - The block is parsed as independent segments of `1 << segment_log2` bytes (4 KiB, one GPU
//!   lane each), exactly like `lazy::lazy_parse_segmented`: segment `k > 0` starts at
//!   `ip = anchor = k << log2` with reps `[0, 0, 0]`, segment 0 at `ip = 1` with
//!   `INITIAL_REPS`; `iend` is the segment end and `ilimit = iend - 8`; every match length is
//!   clamped to `iend`. Trailing literals are carried into the next segment's first sequence and
//!   `lazy::encode_raw` re-encodes every offset against the block's true decoder reps.
//! - Prices are static per pass (no `ZSTD_updateStats` inside the pass): pass 0 from the seed
//!   (`Seed::BlockInit`: zstd's `ZSTD_rescaleFreqs` first-block tables; `Seed::Prior`: the
//!   `codes::OPT_PRIOR_*` tables plus the cover literals), every later pass from the previous
//!   pass's own output histogram (`Hist::of_output`, `off_base` under the decoder reps).
//!   Weights are always zstd's fractional `ZSTD_fracWeight` (optLevel 2 arithmetic).
//! - "Cheap" intermediate passes run optLevel-0 *control flow* (the `+128` skip, the early
//!   relaxation abort, no match+1-literal check) with `sufficient_len = target_length`; the final
//!   pass runs `OptParams::level`.
//! - After a series whose last stretch ends in literals, the next series starts after those
//!   literals (`ip = anchor + litlen`). zstd 1.5.7 intends this too, but its `} {` typo in the
//!   store loop makes it restart at the anchor; the prototype measured the intended form.
//!
//! Tie rules (pinned by `cases`): the literal extension replaces `opt[cur]` when its price is
//! `<=`; a relaxation replaces `opt[pos]` only when `<` (or `pos > last_pos`); relaxation lengths
//! run from each record's length *down* to the previous record's length + 1; records are the rep
//! candidates (zstd's `ll0` numbering) in increasing length, then `A`, then `B`, each kept only
//! when strictly longer than every earlier one; `ZSTD_newRep` updates a node's reps from its
//! predecessor when the node ends a match.
use crate::codes::{ll_code, ml_code, LL_BITS, ML_BITS, OPT_PRIOR_LL, OPT_PRIOR_ML, OPT_PRIOR_OF};
use crate::config::BLOCK_SIZE;
use crate::lazy::{encode_raw, RawSeq};
use crate::params::{MatchParams, OptParams, Seed};
use crate::reference::{match_len_capped, unpack_cands, CandWords};
use crate::seq::{apply_off_base, BlockOutput, Reps, INITIAL_REPS};

/// zstd `BITCOST_MULTIPLIER`: prices are in 1/256 bit.
pub const BITCOST_MULTIPLIER: u32 = 256;
/// zstd `ZSTD_MAX_PRICE`.
pub const MAX_PRICE: i32 = 1 << 30;
/// zstd `ZSTD_OPT_NUM`: the longest series.
pub const OPT_NUM: usize = 1 << 12;
/// zstd `MINMATCH` for `minMatch == 3`.
const MIN_MATCH: usize = 3;
/// The constant per-match fee in `ZSTD_getMatchPrice`: `BITCOST_MULTIPLIER / 5`.
const MATCH_FEE: i32 = (BITCOST_MULTIPLIER / 5) as i32;
/// `search_cap` of every opt preset: a stored candidate length of 64 means "at least 64".
const CAND_CAP: u32 = 64;

fn highbit32(x: u32) -> u32 {
    debug_assert!(x > 0);
    31 - x.leading_zeros()
}

/// zstd `ZSTD_fracWeight` (optLevel >= 1): `WEIGHT(stat)` in 1/256 bit, the highest bit of
/// `stat + 1` plus a linear interpolation of the rest.
pub fn frac_weight(raw: u32) -> u32 {
    let stat = raw + 1;
    let hb = highbit32(stat);
    hb * BITCOST_MULTIPLIER + ((stat << 8) >> hb)
}

/// Symbol counts of one parse: literal bytes, and LL / ML / OF codes of its sequences.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hist {
    pub lit: [u32; 256],
    pub ll: [u32; 36],
    pub ml: [u32; 53],
    pub of: [u32; 32],
}

impl Hist {
    /// The histogram of a block's parse: every literal byte, and each sequence's LL code, ML code
    /// and OF code (`highbit(off_base)`, i.e. under the decoder's reps).
    pub fn of_output(out: &BlockOutput) -> Hist {
        let mut h = Hist { lit: [0; 256], ll: [0; 36], ml: [0; 53], of: [0; 32] };
        for &b in &out.literals {
            h.lit[b as usize] += 1;
        }
        for s in &out.sequences {
            h.ll[ll_code(s.lit_len) as usize] += 1;
            h.ml[ml_code(s.match_len) as usize] += 1;
            h.of[highbit32(s.off_base) as usize] += 1;
        }
        h
    }

    /// Text dump, one table per line (`lit:`, `ll:`, `ml:`, `of:` then the counts), for
    /// comparing GPU passes against the oracle.
    pub fn to_text(&self) -> String {
        let row = |name: &str, t: &[u32]| {
            let v: Vec<String> = t.iter().map(|x| x.to_string()).collect();
            format!("{name}: {}\n", v.join(" "))
        };
        row("lit", &self.lit) + &row("ll", &self.ll) + &row("ml", &self.ml) + &row("of", &self.of)
    }
}

/// Static price tables of one DP pass, in 1/256 bit. For zstd frequency tables `f` with sums
/// `S` (each sum at least 1): `lit[b] = W(S_lit) - min(W(f_lit[b]), W(S_lit) - 256)`
/// (`ZSTD_rawLiteralsCost`), `ll[c] = LL_BITS[c] * 256 + W(S_ll) - W(f_ll[c])`,
/// `ml[c] = ML_BITS[c] * 256 + W(S_ml) - W(f_ml[c])`, `of[c] = c * 256 + W(S_of) - W(f_of[c])`,
/// with `W = frac_weight`. Every entry is in 0..65536 (fits the GPU's u16 tables).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prices {
    pub lit: [i32; 256],
    pub ll: [i32; 36],
    pub ml: [i32; 53],
    pub of: [i32; 32],
}

impl Prices {
    /// Price tables from frequency tables (see `Prices`).
    pub fn from_freqs(lit: &[u32; 256], ll: &[u32; 36], ml: &[u32; 53], of: &[u32; 32]) -> Prices {
        let base = |t: &[u32]| frac_weight(t.iter().sum::<u32>().max(1));
        let (lb, llb, mlb, ofb) = (base(lit), base(ll), base(ml), base(of));
        let m = BITCOST_MULTIPLIER;
        Prices {
            lit: std::array::from_fn(|b| (lb - frac_weight(lit[b]).min(lb - m)) as i32),
            ll: std::array::from_fn(|c| (LL_BITS[c] as u32 * m + llb - frac_weight(ll[c])) as i32),
            ml: std::array::from_fn(|c| (ML_BITS[c] as u32 * m + mlb - frac_weight(ml[c])) as i32),
            of: std::array::from_fn(|c| (c as u32 * m + ofb - frac_weight(of[c])) as i32),
        }
    }

    /// zstd's first-block statistics (`ZSTD_rescaleFreqs`, no dictionary): literal frequencies
    /// `(c > 0) + (c >> 8)` of the block's byte counts `c`, LL `{4, 2, 1, ...}`, ML all 1, OF
    /// `{6, 2, 1, 1, 2, 3, 4, 4, 4, 3, 2, 1, ...}`.
    pub fn block_init(block: &[u8]) -> Prices {
        let mut lit = [0u32; 256];
        for &b in block {
            lit[b as usize] += 1;
        }
        for c in lit.iter_mut() {
            *c = (*c > 0) as u32 + (*c >> 8);
        }
        let mut ll = [1u32; 36];
        ll[..2].copy_from_slice(&[4, 2]);
        let mut of = [1u32; 32];
        of[..11].copy_from_slice(&[6, 2, 1, 1, 2, 3, 4, 4, 4, 3, 2]);
        Prices::from_freqs(&lit, &ll, &[1; 53], &of)
    }

    /// Static prices from a histogram: frequencies `c + (c > 0)` (a seen symbol never costs as
    /// much as an unseen one).
    pub fn from_hist(h: &Hist) -> Prices {
        fn f<const N: usize>(t: &[u32; N]) -> [u32; N] {
            t.map(|c| c + (c > 0) as u32)
        }
        Prices::from_freqs(&f(&h.lit), &f(&h.ll), &f(&h.ml), &f(&h.of))
    }

    /// `ZSTD_rawLiteralsCost` of one byte.
    #[inline]
    fn lit_price(&self, byte: u8) -> i32 {
        self.lit[byte as usize]
    }

    /// `ZSTD_litLengthPrice`.
    #[inline]
    fn ll_price(&self, litlen: u32) -> i32 {
        self.ll[ll_code(litlen) as usize]
    }

    /// `ZSTD_getMatchPrice(offBase, matchLength)`. (Its optLevel < 2 long-offset penalty needs
    /// offCode >= 20, impossible for blocks <= 64 KiB.)
    #[inline]
    fn match_price(&self, off_base: u32, mlen: u32) -> i32 {
        self.of[highbit32(off_base) as usize] + self.ml[ml_code(mlen) as usize] + MATCH_FEE
    }
}

/// Cover literals (`Seed::Prior`): the byte histogram of the positions no candidate covers.
/// `p` is covered when some `p' <= p` has `p' + lenB(p') > p` (a running max of `p + lenB` over
/// the candidate words, capped lengths as stored). When every byte is covered, the whole
/// block's histogram.
pub fn cover_literals(block: &[u8], cands: &[CandWords]) -> [u32; 256] {
    let mut lit = [0u32; 256];
    let mut reach = 0usize;
    let mut n = 0u32;
    for p in 0..BLOCK_SIZE {
        let (_, b) = unpack_cands(cands[p]);
        reach = reach.max(p + b.len as usize);
        if reach <= p {
            lit[block[p] as usize] += 1;
            n += 1;
        }
    }
    if n == 0 {
        for &b in block {
            lit[b as usize] += 1;
        }
    }
    lit
}

/// Pass-0 prices for `seed`.
pub fn seed_prices(block: &[u8], cands: &[CandWords], seed: Seed) -> Prices {
    match seed {
        Seed::BlockInit => Prices::block_init(block),
        Seed::Prior => Prices::from_hist(&Hist {
            lit: cover_literals(block, cands),
            ll: OPT_PRIOR_LL,
            ml: OPT_PRIOR_ML,
            of: OPT_PRIOR_OF,
        }),
    }
}

/// zstd `ZSTD_newRep`: the reps after a match with `off_base` preceded by literals iff `!ll0`
/// (the decoder's update, `seq::apply_off_base`).
fn new_rep(rep: Reps, off_base: u32, ll0: bool) -> Reps {
    let mut r = rep;
    apply_off_base(&mut r, off_base, !ll0 as u32);
    r
}

/// zstd `ZSTD_optimal_t`: a *stretch* ending at this position (a match of `mlen` at `off`
/// (offBase), then `litlen` literals), its price from the series start, and the reps after it.
#[derive(Clone, Copy, Debug, Default)]
struct Node {
    price: i32,
    off: u32,
    mlen: u32,
    litlen: u32,
    rep: Reps,
}

/// One segment's parse limits (see the module doc).
#[derive(Clone, Copy, Debug)]
struct Seg {
    ip0: usize,
    anchor0: usize,
    iend: usize,
    ilimit: usize,
    reps0: Reps,
}

impl Seg {
    fn new(k: usize, log2: u32) -> Seg {
        let s = k << log2;
        let iend = s + (1 << log2);
        let (ip0, reps0) = if k == 0 { (1, INITIAL_REPS) } else { (s, [0; 3]) };
        Seg { ip0, anchor0: s, iend, ilimit: iend - 8, reps0 }
    }
}

/// One DP pass's constants and scratch.
struct Dp<'a> {
    block: &'a [u8],
    cands: &'a [CandWords],
    prices: &'a Prices,
    /// optLevel of this pass (0 or 2).
    level: u8,
    /// `sufficient_len = min(targetLength, ZSTD_OPT_NUM - 1)`.
    sufficient: usize,
    opt: Vec<Node>,
    /// `(offBase, len)` records of the last `get_all_matches`, strictly increasing length.
    matches: Vec<(u32, u32)>,
    /// Debug only: the series in which each `opt[]` entry was last written, to prove the DP
    /// never reads an entry left over from an earlier series (the GPU keeps a ring instead).
    #[cfg(debug_assertions)]
    stamp: Vec<u32>,
    #[cfg(debug_assertions)]
    series: u32,
    /// `Engine::Ring`: `target_length + 1` nodes, series position `pos` in slot
    /// `pos % ring.len()`.
    ring: Vec<Node>,
    /// `Engine::Ring`: (mlen, litlen, offBase) of every block position's node, written when the
    /// node becomes final (the GPU's trace buffer).
    trace: Vec<[u32; 3]>,
    /// Debug: the series position each ring slot holds.
    #[cfg(debug_assertions)]
    ring_pos: Vec<usize>,
}

impl Dp<'_> {
    #[inline]
    fn touch(&mut self, _i: usize) {
        #[cfg(debug_assertions)]
        {
            self.stamp[_i] = self.series;
        }
    }

    #[inline]
    fn check(&self, _i: usize) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(self.stamp[_i], self.series, "opt[{_i}] read before being written in this series");
    }

    /// `ZSTD_insertBtAndGetAllMatches` over the candidate words: fills `self.matches` with the
    /// records at `p` given the node's `rep` and `ll0`, every length clamped to `iend - p`.
    /// 1. Rep candidates `rep[ll0 .. ll0 + 3)` (index 3 = `rep[0] - 1`), each valid when
    ///    `1 <= off <= p`, matching when the first 3 bytes agree; recorded as offBase
    ///    `index - ll0 + 1` when longer than the best so far. A rep longer than `sufficient_len`
    ///    or reaching `iend` returns at once.
    /// 2. `A` then `B` from the candidate words (a stored length of 64 extended to the true
    ///    length), recorded as offBase `offset + 3` when longer than the best so far; one
    ///    reaching `iend` ends the list.
    fn get_all_matches(&mut self, p: usize, rep: &Reps, ll0: bool, iend: usize) {
        self.matches.clear();
        let lim = iend - p;
        let mut best = MIN_MATCH - 1;
        let ll0u = ll0 as usize;
        for rc in ll0u..ll0u + 3 {
            let ro = if rc == 3 { rep[0].wrapping_sub(1) } else { rep[rc] };
            if ro.wrapping_sub(1) < p as u32 {
                let rl = match_len_capped(self.block, p, p - ro as usize, lim);
                if rl >= MIN_MATCH && rl > best {
                    best = rl;
                    self.matches.push(((rc - ll0u + 1) as u32, rl as u32));
                    if rl > self.sufficient || rl == lim {
                        return;
                    }
                }
            }
        }
        let (a, b) = unpack_cands(self.cands[p]);
        for c in [a, b] {
            if c.len == 0 {
                continue;
            }
            let q = p - c.offset as usize;
            let l = if c.len == CAND_CAP { match_len_capped(self.block, p, q, lim) } else { (c.len as usize).min(lim) };
            if l > best {
                best = l;
                self.matches.push((c.offset + 3, l as u32));
                if l == lim {
                    break;
                }
            }
        }
    }

    /// `ZSTD_compressBlock_opt_generic`'s match loop over one segment. Appends the segment's
    /// sequences (literal lengths counted from `seg.anchor0`, real offsets) to `out` and returns
    /// the final anchor.
    fn segment(&mut self, seg: &Seg, out: &mut Vec<RawSeq>) -> usize {
        let (iend, ilimit) = (seg.iend, seg.ilimit);
        let pr = self.prices;
        let block = self.block;
        let mut st = SegState { ip: seg.ip0, anchor: seg.anchor0, rep: seg.reps0, seq_reps: seg.reps0 };
        while st.ip < ilimit {
            let ip = st.ip;
            let mut last_pos;
            let last_stretch;
            let cur_end;
            #[cfg(debug_assertions)]
            {
                self.series += 1;
            }
            // find first match
            {
                let litlen = (ip - st.anchor) as u32;
                let rep = st.rep;
                self.get_all_matches(ip, &rep, litlen == 0, iend);
                if self.matches.is_empty() {
                    st.ip += 1;
                    continue;
                }
                self.opt[0] = Node { mlen: 0, litlen, price: pr.ll_price(litlen), off: 0, rep };
                self.touch(0);
                let (max_off, max_ml) = *self.matches.last().unwrap();
                if max_ml as usize > self.sufficient {
                    // large match -> immediate encoding
                    let ls = Node { litlen: 0, mlen: max_ml, off: max_off, price: 0, rep: [0; 3] };
                    self.commit(ls, 0, max_ml as usize, &mut st, out);
                    continue;
                }
                let mut pos = 1usize;
                while pos < MIN_MATCH {
                    self.opt[pos] = Node { price: MAX_PRICE, mlen: 0, litlen: litlen + pos as u32, off: 0, rep: [0; 3] };
                    self.touch(pos);
                    pos += 1;
                }
                for mi in 0..self.matches.len() {
                    let (ob, end) = self.matches[mi];
                    while pos <= end as usize {
                        let price = self.opt[0].price + pr.match_price(ob, pos as u32) + pr.ll_price(0);
                        self.opt[pos] = Node { mlen: pos as u32, off: ob, litlen: 0, price, rep: [0; 3] };
                        self.touch(pos);
                        pos += 1;
                    }
                }
                last_pos = pos - 1;
                self.opt[pos].price = MAX_PRICE;
                self.touch(pos);
            }

            // check further positions
            let mut early: Option<Node> = None;
            let mut cur = 1usize;
            while cur <= last_pos {
                let inr = ip + cur;
                // Fix current position with one literal if cheaper (tie: the literal wins).
                {
                    self.check(cur - 1);
                    self.check(cur);
                    let litlen = self.opt[cur - 1].litlen + 1;
                    let price = self.opt[cur - 1].price + pr.lit_price(block[ip + cur - 1]) + (pr.ll_price(litlen) - pr.ll_price(litlen - 1));
                    if price <= self.opt[cur].price {
                        let prev_match = self.opt[cur];
                        self.opt[cur] = self.opt[cur - 1];
                        self.opt[cur].litlen = litlen;
                        self.opt[cur].price = price;
                        let ll_inc1 = pr.ll_price(1) - pr.ll_price(0);
                        if self.level >= 1 && prev_match.litlen == 0 && ll_inc1 < 0 && ip + cur < iend {
                            // check next position, in case it would be cheaper
                            let next_lit = pr.lit_price(block[ip + cur]);
                            let with1 = prev_match.price + next_lit + ll_inc1;
                            let with_more = price + next_lit + (pr.ll_price(litlen + 1) - pr.ll_price(litlen));
                            self.check(cur + 1);
                            if with1 < with_more && with1 < self.opt[cur + 1].price {
                                let prev = cur - prev_match.mlen as usize;
                                self.check(prev);
                                let nr = new_rep(self.opt[prev].rep, prev_match.off, self.opt[prev].litlen == 0);
                                self.opt[cur + 1] = Node { rep: nr, litlen: 1, price: with1, ..prev_match };
                                self.touch(cur + 1);
                                if last_pos < cur + 1 {
                                    last_pos = cur + 1;
                                }
                            }
                        }
                    }
                }
                // just finished a match => alter offset history
                if self.opt[cur].litlen == 0 {
                    let prev = cur - self.opt[cur].mlen as usize;
                    self.check(prev);
                    self.opt[cur].rep = new_rep(self.opt[prev].rep, self.opt[cur].off, self.opt[prev].litlen == 0);
                }
                // last match must start at a minimum distance of 8 from oend
                if inr > ilimit {
                    cur += 1;
                    continue;
                }
                if cur == last_pos {
                    break;
                }
                if self.level == 0 {
                    self.check(cur + 1);
                    if self.opt[cur + 1].price <= self.opt[cur].price + (BITCOST_MULTIPLIER / 2) as i32 {
                        cur += 1;
                        continue;
                    }
                }
                let ll0 = self.opt[cur].litlen == 0;
                let base_price = self.opt[cur].price + pr.ll_price(0);
                let rep = self.opt[cur].rep;
                self.get_all_matches(inr, &rep, ll0, iend);
                if self.matches.is_empty() {
                    cur += 1;
                    continue;
                }
                let (max_off, longest) = *self.matches.last().unwrap();
                let longest = longest as usize;
                if longest > self.sufficient || cur + longest >= OPT_NUM || ip + cur + longest >= iend {
                    early = Some(Node { mlen: longest as u32, off: max_off, litlen: 0, price: 0, rep: [0; 3] });
                    last_pos = cur + longest;
                    break;
                }
                // set prices using matches found at position == cur (lengths downward)
                for mi in 0..self.matches.len() {
                    let (ob, last_ml) = self.matches[mi];
                    let start_ml = if mi > 0 { self.matches[mi - 1].1 + 1 } else { MIN_MATCH as u32 };
                    let mut mlen = last_ml;
                    while mlen >= start_ml {
                        let pos = cur + mlen as usize;
                        let price = base_price + pr.match_price(ob, mlen);
                        if pos <= last_pos {
                            self.check(pos);
                        }
                        if pos > last_pos || price < self.opt[pos].price {
                            while last_pos < pos {
                                // fill empty positions, for future comparisons
                                last_pos += 1;
                                self.opt[last_pos].price = MAX_PRICE;
                                self.opt[last_pos].litlen = 1;
                                self.touch(last_pos);
                            }
                            let n = &mut self.opt[pos];
                            n.mlen = mlen;
                            n.off = ob;
                            n.litlen = 0;
                            n.price = price;
                        } else if self.level == 0 {
                            break; // early update abort
                        }
                        mlen -= 1;
                    }
                }
                self.opt[last_pos + 1].price = MAX_PRICE;
                self.touch(last_pos + 1);
                cur += 1;
            }
            match early {
                Some(ls) => {
                    last_stretch = ls;
                    cur_end = last_pos - ls.mlen as usize;
                }
                None => {
                    self.check(last_pos);
                    last_stretch = self.opt[last_pos];
                    cur_end = last_pos - last_stretch.mlen as usize;
                }
            }
            self.commit(last_stretch, cur_end, last_pos, &mut st, out);
        }
        st.anchor
    }

    /// zstd's `_shortestPath`: the backward trace from `last` (ending at `last_pos`, with
    /// `cur == last_pos - last.mlen`), storing the series' sequences with real offsets.
    fn commit(&mut self, last: Node, cur: usize, last_pos: usize, st: &mut SegState, out: &mut Vec<RawSeq>) {
        if last.mlen == 0 {
            // no solution: all matches have been converted into literals
            st.ip += last_pos;
            return;
        }
        let mut cur = cur;
        if last.litlen == 0 {
            // finishing on a match: update offset history
            self.check(cur);
            st.rep = new_rep(self.opt[cur].rep, last.off, self.opt[cur].litlen == 0);
        } else {
            st.rep = last.rep;
            cur -= last.litlen as usize;
        }
        // Stretches (match, then literals) back to the series start become sequences
        // (literals, then match): each match takes the preceding stretch's literal count.
        let first = out.len();
        let (mut mlen, mut off) = (last.mlen, last.off);
        let mut stretch_pos = cur;
        loop {
            self.check(stretch_pos);
            let next = self.opt[stretch_pos];
            out.push((next.litlen, mlen, off));
            if next.mlen == 0 {
                break;
            }
            (mlen, off) = (next.mlen, next.off);
            stretch_pos -= (next.litlen + next.mlen) as usize;
        }
        out[first..].reverse();
        for s in &mut out[first..] {
            // offBase -> real offset against the segment's own history
            s.2 = apply_off_base(&mut st.seq_reps, s.2, s.0);
            debug_assert!(s.2 >= 1 && s.2 as usize <= st.anchor + s.0 as usize, "offset {} at {}", s.2, st.anchor);
            st.anchor += (s.0 + s.1) as usize;
            st.ip = st.anchor;
        }
        debug_assert_eq!(st.rep, st.seq_reps, "DP reps differ from the stored sequences' reps");
        if last.litlen > 0 {
            st.ip = st.anchor + last.litlen as usize;
        }
    }

    /// Ring slot of series position `pos`.
    #[inline]
    fn slot(&self, pos: usize) -> usize {
        pos % self.ring.len()
    }

    /// Debug: records that `pos` now owns its slot.
    #[inline]
    fn own(&mut self, _pos: usize) {
        #[cfg(debug_assertions)]
        {
            let s = self.slot(_pos);
            self.ring_pos[s] = _pos;
        }
    }

    /// Debug: `pos` must still own its slot (not aliased by a later position).
    #[inline]
    fn owns(&self, _pos: usize) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(self.ring_pos[self.slot(_pos)], _pos, "ring slot of {_pos} was overwritten");
    }

    /// `segment` in ring form (`Engine::Ring`, the GPU K3opt spec; see `Engine`).
    fn segment_ring(&mut self, seg: &Seg, out: &mut Vec<RawSeq>) -> usize {
        let (iend, ilimit) = (seg.iend, seg.ilimit);
        let pr = self.prices;
        let block = self.block;
        let mut st = SegState { ip: seg.ip0, anchor: seg.anchor0, rep: seg.reps0, seq_reps: seg.reps0 };
        while st.ip < ilimit {
            let ip = st.ip;
            let litlen = (ip - st.anchor) as u32;
            let rep = st.rep;
            self.get_all_matches(ip, &rep, litlen == 0, iend);
            if self.matches.is_empty() {
                st.ip += 1;
                continue;
            }
            let n0 = Node { mlen: 0, litlen, price: pr.ll_price(litlen), off: 0, rep };
            #[cfg(debug_assertions)]
            self.ring_pos.fill(usize::MAX);
            self.ring[0] = n0;
            self.own(0);
            self.trace[ip] = [0, litlen, 0];
            let (max_off, max_ml) = *self.matches.last().unwrap();
            if max_ml as usize > self.sufficient {
                let ls = Node { litlen: 0, mlen: max_ml, off: max_off, price: 0, rep: new_rep(rep, max_off, litlen == 0) };
                self.commit_ring(ls, ip, max_ml as usize, &mut st, out);
                continue;
            }
            let mut pos = 1usize;
            while pos < MIN_MATCH {
                let s = self.slot(pos);
                self.ring[s] = Node { price: MAX_PRICE, mlen: 0, litlen: litlen + pos as u32, off: 0, rep: [0; 3] };
                self.own(pos);
                pos += 1;
            }
            for mi in 0..self.matches.len() {
                let (ob, end) = self.matches[mi];
                let mrep = new_rep(rep, ob, litlen == 0);
                while pos <= end as usize {
                    let price = n0.price + pr.match_price(ob, pos as u32) + pr.ll_price(0);
                    let s = self.slot(pos);
                    self.ring[s] = Node { mlen: pos as u32, off: ob, litlen: 0, price, rep: mrep };
                    self.own(pos);
                    pos += 1;
                }
            }
            let mut last_pos = pos - 1;
            let mut early: Option<Node> = None;
            let mut cur = 1usize;
            while cur <= last_pos {
                let inr = ip + cur;
                self.owns(cur - 1);
                self.owns(cur);
                let (sp, sc) = (self.slot(cur - 1), self.slot(cur));
                let prev = self.ring[sp];
                let litlen = prev.litlen + 1;
                let price = prev.price + pr.lit_price(block[ip + cur - 1]) + (pr.ll_price(litlen) - pr.ll_price(litlen - 1));
                if price <= self.ring[sc].price {
                    let prev_match = self.ring[sc];
                    self.ring[sc] = Node { litlen, price, ..prev };
                    let ll_inc1 = pr.ll_price(1) - pr.ll_price(0);
                    if self.level >= 1 && prev_match.litlen == 0 && ll_inc1 < 0 && ip + cur < iend {
                        let next_lit = pr.lit_price(block[ip + cur]);
                        let with1 = prev_match.price + next_lit + ll_inc1;
                        let with_more = price + next_lit + (pr.ll_price(litlen + 1) - pr.ll_price(litlen));
                        // virtual sentinel: positions past last_pos cost MAX_PRICE
                        let next_price = if cur < last_pos {
                            self.owns(cur + 1);
                            self.ring[self.slot(cur + 1)].price
                        } else {
                            MAX_PRICE
                        };
                        if with1 < with_more && with1 < next_price {
                            // prev_match.rep was computed when it was relaxed
                            let s = self.slot(cur + 1);
                            self.ring[s] = Node { litlen: 1, price: with1, ..prev_match };
                            self.own(cur + 1);
                            if last_pos < cur + 1 {
                                last_pos = cur + 1;
                            }
                        }
                    }
                }
                let n = self.ring[sc];
                self.trace[inr] = [n.mlen, n.litlen, n.off];
                if inr > ilimit {
                    cur += 1;
                    continue;
                }
                if cur == last_pos {
                    break;
                }
                if self.level == 0 {
                    self.owns(cur + 1);
                    if self.ring[self.slot(cur + 1)].price <= n.price + (BITCOST_MULTIPLIER / 2) as i32 {
                        cur += 1;
                        continue;
                    }
                }
                let ll0 = n.litlen == 0;
                let base_price = n.price + pr.ll_price(0);
                self.get_all_matches(inr, &n.rep, ll0, iend);
                if self.matches.is_empty() {
                    cur += 1;
                    continue;
                }
                let (max_off, longest) = *self.matches.last().unwrap();
                let longest = longest as usize;
                if longest > self.sufficient || cur + longest >= OPT_NUM || ip + cur + longest >= iend {
                    early = Some(Node { mlen: longest as u32, off: max_off, litlen: 0, price: 0, rep: new_rep(n.rep, max_off, ll0) });
                    last_pos = cur + longest;
                    break;
                }
                for mi in 0..self.matches.len() {
                    let (ob, last_ml) = self.matches[mi];
                    let start_ml = if mi > 0 { self.matches[mi - 1].1 + 1 } else { MIN_MATCH as u32 };
                    let mrep = new_rep(n.rep, ob, ll0);
                    let mut mlen = last_ml;
                    while mlen >= start_ml {
                        let pos = cur + mlen as usize;
                        debug_assert!(pos - cur < self.ring.len());
                        let price = base_price + pr.match_price(ob, mlen);
                        let improves = pos > last_pos || {
                            self.owns(pos);
                            price < self.ring[self.slot(pos)].price
                        };
                        if improves {
                            while last_pos < pos {
                                last_pos += 1;
                                let s = self.slot(last_pos);
                                self.ring[s].price = MAX_PRICE;
                                self.ring[s].litlen = 1;
                                self.own(last_pos);
                            }
                            let s = self.slot(pos);
                            self.ring[s] = Node { mlen, off: ob, litlen: 0, price, rep: mrep };
                        } else if self.level == 0 {
                            break;
                        }
                        mlen -= 1;
                    }
                }
                cur += 1;
            }
            let last = match early {
                Some(ls) => ls,
                None => {
                    self.owns(last_pos);
                    self.ring[self.slot(last_pos)]
                }
            };
            self.commit_ring(last, ip, last_pos, &mut st, out);
        }
        st.anchor
    }

    /// `commit` in ring form: the next series' reps are the last stretch's own (computed when
    /// its match was relaxed, or with the immediate encoding), and the backward trace reads the
    /// per-position `trace` (mlen, litlen, offBase), written when each node became final.
    fn commit_ring(&mut self, last: Node, ip: usize, last_pos: usize, st: &mut SegState, out: &mut Vec<RawSeq>) {
        if last.mlen == 0 {
            st.ip += last_pos;
            return;
        }
        st.rep = last.rep;
        let mut stretch_pos = last_pos - last.mlen as usize - last.litlen as usize;
        let first = out.len();
        let (mut mlen, mut off) = (last.mlen, last.off);
        loop {
            let [nm, nl, no] = self.trace[ip + stretch_pos];
            out.push((nl, mlen, off));
            if nm == 0 {
                break;
            }
            (mlen, off) = (nm, no);
            stretch_pos -= (nl + nm) as usize;
        }
        out[first..].reverse();
        for s in &mut out[first..] {
            s.2 = apply_off_base(&mut st.seq_reps, s.2, s.0);
            st.anchor += (s.0 + s.1) as usize;
            st.ip = st.anchor;
        }
        debug_assert_eq!(st.rep, st.seq_reps, "ring reps differ from the stored sequences' reps");
        if last.litlen > 0 {
            st.ip = st.anchor + last.litlen as usize;
        }
    }
}

/// A segment's running parse state: zstd's `ip`, `anchor`, `rep`, and the rep history of the
/// sequences stored so far (`seq_reps`, equal to `rep` after every series).
struct SegState {
    ip: usize,
    anchor: usize,
    rep: Reps,
    seq_reps: Reps,
}

/// How the DP stores its nodes. Both engines give identical output (tested on every synthetic
/// block and a corpus sample, all variants).
///
/// - `Linear`: zstd's layout, statement by statement: `opt[0 ..= ZSTD_OPT_NUM]`, a match node's
///   reps computed when the node is *visited* (`ZSTD_newRep(opt[cur - mlen].rep, ..)`), the
///   sentinel `opt[last_pos + 1].price = MAX` written, the backward trace over `opt[]`.
/// - `Ring`: **the GPU K3opt spec**. Nodes live in a ring of `target_length + 1` (33) slots,
///   position `pos` of the series in slot `pos % 33`. At the visit of `cur` the live positions
///   are `cur - 1 ..= cur + 32` (34 of them): the literal extension reads `cur - 1` first, and
///   only then may a relaxation to `cur + 32` reuse its slot. To make that sufficient:
///   1. a match node's reps are computed when it is *relaxed* (or seeded at the series start),
///      from the source node `cur`, which is final then: `rep = ZSTD_newRep(node[cur].rep, off,
///      node[cur].litlen == 0)`, stored with the node. The visit-time rep update disappears;
///      the match + 1 literal node takes `prevMatch.rep` as is; the immediate encoding computes
///      `newRep(node[cur].rep, off, ll0)` itself, and the commit uses `lastStretch.rep`.
///   2. no sentinel is written: `price(pos) = MAX_PRICE` for every `pos > last_pos` (only the
///      match + 1 literal check reads there, at `cur + 1`); the relaxation's fill loop still
///      writes `price = MAX, litlen = 1` into each slot it claims.
///   3. when node `cur` is final (after its literal extension), its (mlen, litlen, offBase) go
///      to `trace[ip + cur]` (8 B per position on the GPU); the series start writes
///      `trace[ip] = (0, litlen, 0)`. The backward trace walks `trace`, never the ring. A node
///      can sit at `iend` itself (a match reaching the segment end); its entry is written but
///      never read (the walk starts at `last_pos - mlen - litlen < iend`), so the GPU may drop
///      that write and keep 4096 entries per segment.
///
///   Debug builds assert that every ring read finds the position it expects in its slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Linear,
    Ring,
}

/// One DP pass over the block at optLevel `level` with static `prices`: every segment parsed
/// on its own, literals carried across segments, offsets encoded with the true decoder reps.
pub fn dp_pass(block: &[u8], cands: &[CandWords], params: &MatchParams, prices: &Prices, level: u8, target_length: u32) -> BlockOutput {
    dp_pass_with(block, cands, params, prices, level, target_length, Engine::Linear)
}

/// `dp_pass` with an explicit `Engine`.
pub fn dp_pass_with(
    block: &[u8],
    cands: &[CandWords],
    params: &MatchParams,
    prices: &Prices,
    level: u8,
    target_length: u32,
    engine: Engine,
) -> BlockOutput {
    assert_eq!(block.len(), BLOCK_SIZE);
    assert_eq!(cands.len(), BLOCK_SIZE);
    let log2 = params.segment_log2;
    assert!(log2 > 0, "opt::dp_pass: segment_log2 0");
    let mut dp = Dp {
        block,
        cands,
        prices,
        level,
        sufficient: (target_length as usize).min(OPT_NUM - 1),
        opt: vec![Node::default(); OPT_NUM + 3],
        matches: Vec::with_capacity(8),
        #[cfg(debug_assertions)]
        stamp: vec![0; OPT_NUM + 3],
        #[cfg(debug_assertions)]
        series: 0,
        ring: vec![Node::default(); (target_length as usize).min(OPT_NUM - 1) + 1],
        trace: if engine == Engine::Ring { vec![[0; 3]; BLOCK_SIZE + 1] } else { Vec::new() },
        #[cfg(debug_assertions)]
        ring_pos: vec![usize::MAX; (target_length as usize).min(OPT_NUM - 1) + 1],
    };
    let mut raw: Vec<RawSeq> = Vec::new();
    let mut prev_end = 0usize;
    let mut part = Vec::new();
    for k in 0..BLOCK_SIZE >> log2 {
        let seg = Seg::new(k, log2);
        part.clear();
        let end = match engine {
            Engine::Linear => dp.segment(&seg, &mut part),
            Engine::Ring => dp.segment_ring(&seg, &mut part),
        };
        if let Some(first) = part.first_mut() {
            first.0 += (seg.anchor0 - prev_end) as u32;
            prev_end = end;
        }
        raw.extend_from_slice(&part);
    }
    encode_raw(block, &raw)
}

/// One pass of `passes`: the prices it ran with, its optLevel, and its output.
#[derive(Clone, Debug)]
pub struct Pass {
    pub prices: Prices,
    pub level: u8,
    pub out: BlockOutput,
}

/// Every DP pass of the optimal parse, in order: `o.passes` cheap passes (optLevel 0), then the
/// final pass at `o.level`. Pass 0 is priced by `seed_prices`, pass `n + 1` by
/// `Prices::from_hist(&Hist::of_output(&pass[n].out))`. The GPU passes are checked against these
/// one at a time (`Hist::of_output(&pass.out).to_text()` is the per-pass histogram dump).
pub fn passes(block: &[u8], cands: &[CandWords], params: &MatchParams) -> Vec<Pass> {
    let o: OptParams = params.opt.expect("opt::passes: params.opt is None");
    let mut prices = seed_prices(block, cands, o.seed);
    let mut v = Vec::with_capacity(o.passes as usize + 1);
    for i in 0..=o.passes {
        let level = if i == o.passes { o.level } else { 0 };
        let out = dp_pass(block, cands, params, &prices, level, o.target_length);
        let next = Prices::from_hist(&Hist::of_output(&out));
        v.push(Pass { prices: std::mem::replace(&mut prices, next), level, out });
    }
    v
}

/// The optimal parse of `block` from its `find_cands` words: the final pass of `passes`.
pub fn parse(block: &[u8], cands: &[CandWords], params: &MatchParams) -> BlockOutput {
    parse_with(block, cands, params, Engine::Linear)
}

/// `parse` with an explicit `Engine` (every pass runs on it).
pub fn parse_with(block: &[u8], cands: &[CandWords], params: &MatchParams, engine: Engine) -> BlockOutput {
    let o: OptParams = params.opt.expect("opt::parse: params.opt is None");
    let mut prices = seed_prices(block, cands, o.seed);
    for _ in 0..o.passes {
        let out = dp_pass_with(block, cands, params, &prices, 0, o.target_length, engine);
        prices = Prices::from_hist(&Hist::of_output(&out));
    }
    dp_pass_with(block, cands, params, &prices, o.level, o.target_length, engine)
}

/// Hand-built blocks with scripted candidate words and the exact sequences the optimal parse
/// must produce: one case per DP tie rule, segment boundary and rep-numbering rule (the GPU
/// K3opt replays every case). Blocks are random bytes with planted matches, so no other match
/// exists at the rep offsets the cases use.
///
/// A case runs one of two ways (`run_case`): with `prices: Some(p)`, one `dp_pass` at
/// `params.opt.level` with the price tables `p` verbatim (the GPU loads them in place of its
/// pass-0 prologue); with `prices: None`, the full `parse` of `params` (seed and passes).
/// Every case fits the first four 4 KiB segments, so it runs at 16, 32 and 64 KiB blocks.
pub mod cases {
    use super::{dp_pass_with, parse_with, Engine, Prices, BITCOST_MULTIPLIER};
    use crate::config::BLOCK_SIZE;
    use crate::params::{MatchParams, OptParams, OPT14, OPT16};
    use crate::reference::{match_len, pack_cands, Cand, CandWords};
    use crate::seq::{BlockOutput, Sequence};
    use crate::synth;

    /// One scripted block: `expect` lists `(params, prices, sequences)` runs.
    pub struct OptCase {
        pub name: String,
        pub block: Vec<u8>,
        pub cands: Vec<CandWords>,
        pub expect: Vec<(MatchParams, Option<Prices>, Vec<Sequence>)>,
    }

    /// The output of one `expect` entry of a case (see the module doc), on `engine`.
    pub fn run_case(block: &[u8], cands: &[CandWords], params: &MatchParams, prices: Option<&Prices>, engine: Engine) -> BlockOutput {
        match prices {
            Some(p) => {
                let o = params.opt.expect("opt case params");
                dp_pass_with(block, cands, params, p, o.level, o.target_length, engine)
            }
            None => parse_with(block, cands, params, engine),
        }
    }

    /// Segment size of the opt presets.
    const SEG: usize = 1 << OPT16.segment_log2;
    /// `M` = one bit.
    const M: i32 = BITCOST_MULTIPLIER as i32;

    /// A single final pass at `level` (no cheap passes, block-init seed unused).
    fn pass(level: u8) -> MatchParams {
        MatchParams { opt: Some(OptParams { level, passes: 0, ..OPT16.opt.unwrap() }), ..OPT16 }
    }

    /// Flat prices: every literal 4 bits, LL and ML codes free, OF code `c` costs `c` bits, so a
    /// match costs `of[oc] + 51`.
    fn flat() -> Prices {
        Prices { lit: [4 * M; 256], ll: [0; 36], ml: [0; 53], of: std::array::from_fn(|c| c as i32 * M) }
    }

    fn seq(lit_len: u32, match_len: u32, off_base: u32) -> Sequence {
        Sequence { lit_len, match_len, off_base }
    }

    fn oc(offset: usize) -> usize {
        (31 - (offset as u32 + 3).leading_zeros()) as usize
    }

    /// Make `block[dst..dst+len]` a match at offset `off` of exactly `len` bytes (copy the
    /// destination bytes back to `dst - off`, break the bytes after and before the source).
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

    /// The candidate words of a single record (A = B).
    fn one(offset: usize, len: u32) -> CandWords {
        let c = Cand { offset: offset as u32, len };
        pack_cands(c, c)
    }

    fn two(a: (usize, u32), b: (usize, u32)) -> CandWords {
        pack_cands(Cand { offset: a.0 as u32, len: a.1 }, Cand { offset: b.0 as u32, len: b.1 })
    }

    /// Checks that the planted match at `p` with `offset` is exactly `len` bytes.
    fn real(block: &[u8], p: usize, offset: usize, len: usize) {
        assert_eq!(match_len(block, p, p - offset), len, "planted match at {p} offset {offset}");
    }

    fn empty() -> Vec<CandWords> {
        vec![[0; 2]; BLOCK_SIZE]
    }

    fn case(name: &str, block: Vec<u8>, cands: Vec<CandWords>, expect: Vec<(MatchParams, Option<Prices>, Vec<Sequence>)>) -> OptCase {
        OptCase { name: name.to_string(), block, cands, expect }
    }

    /// Literal extension ties go to the literal (`<=`): a lone 3-byte match costing exactly
    /// three literals is replaced by them at `cur == last_pos`, the series has no match left
    /// (`lastStretch.mlen == 0`) and emits nothing. One unit cheaper, the match is kept.
    pub fn literal_tie_drops_match() -> Vec<OptCase> {
        let mut block = synth::random(101, BLOCK_SIZE);
        plant(&mut block, 100, 50, 3);
        real(&block, 100, 50, 3);
        let mut cands = empty();
        cands[100] = one(50, 3);
        let (mut tie, mut cheaper) = (flat(), flat());
        tie.of[oc(50)] = 3 * 4 * M - 51;
        cheaper.of[oc(50)] = 3 * 4 * M - 52;
        vec![case("literal_tie_drops_match", block, cands, vec![(pass(2), Some(tie), vec![]), (pass(2), Some(cheaper), vec![seq(100, 3, 53)])])]
    }

    /// Relaxation ties keep the earlier node (`<`): the series start's record (offset 1000,
    /// length 5) and one literal + a record at the next position (offset 100, length 4) reach
    /// position 5 at the same price; the first stays. One unit cheaper, the second wins.
    pub fn relax_tie_keeps_earlier() -> Vec<OptCase> {
        let p = 2000;
        let mut block = synth::random(102, BLOCK_SIZE);
        plant(&mut block, p, 1000, 5);
        plant(&mut block, p + 1, 100, 4);
        real(&block, p, 1000, 5);
        real(&block, p + 1, 100, 4);
        let mut cands = empty();
        cands[p] = one(1000, 5);
        cands[p + 1] = one(100, 4);
        let (mut tie, mut cheaper) = (flat(), flat());
        for (pr, d) in [(&mut tie, 0), (&mut cheaper, 1)] {
            pr.of[oc(1000)] = 2000;
            pr.of[oc(100)] = 2000 - 4 * M - d;
        }
        let want_tie = vec![seq(p as u32, 5, 1003)];
        let want_cheaper = vec![seq(p as u32 + 1, 4, 103)];
        vec![case("relax_tie_keeps_earlier", block, cands, vec![(pass(2), Some(tie), want_tie), (pass(2), Some(cheaper), want_cheaper)])]
    }

    /// Records must be strictly longer than every earlier one: at `p` the rep0 match (6 bytes)
    /// is recorded first, so the explicit candidate of the same length is dropped although its
    /// offset is far cheaper; the parse takes the (expensive) repcode. At `p2` the same with 40
    /// bytes: the rep, longer than `sufficient_len`, ends the search and is encoded at once.
    pub fn equal_length_explicit_after_rep_dropped() -> Vec<OptCase> {
        let (p0, p, p2) = (1000, 1100, 1300);
        let mut block = synth::random(103, BLOCK_SIZE);
        plant(&mut block, p0, 700, 5);
        plant(&mut block, p, 700, 6);
        plant(&mut block, p, 300, 6);
        plant(&mut block, p2, 700, 40);
        plant(&mut block, p2, 250, 40);
        real(&block, p0, 700, 5);
        real(&block, p, 700, 6);
        real(&block, p, 300, 6);
        real(&block, p2, 700, 40);
        real(&block, p2, 250, 40);
        let mut cands = empty();
        cands[p0] = one(700, 5);
        cands[p] = one(300, 6);
        cands[p2] = one(250, 40);
        let mut pr = flat();
        pr.of[0] = 3000; // repcode 1 (offBase 1)
        pr.of[oc(300)] = 500;
        let want = vec![seq(p0 as u32, 5, 703), seq((p - p0 - 5) as u32, 6, 1), seq((p2 - p - 6) as u32, 40, 1)];
        vec![case("equal_length_explicit_after_rep_dropped", block, cands, vec![(pass(2), Some(pr), want)])]
    }

    /// `startML`: a later record is priced only at lengths above the earlier record's. A series
    /// opens at `p - 1` (an unused, expensive 3-byte match); at `p`, A (offset 40, 3 bytes,
    /// expensive) and B (offset 600, 8 bytes, cheap); a 30-byte match at `p + 3` makes ending
    /// the first match at `p + 3` best. Only A may end there, so the parse uses A although B's
    /// first 3 bytes would be cheaper.
    pub fn later_record_not_priced_below_earlier() -> Vec<OptCase> {
        let p = 3000;
        let mut block = synth::random(104, BLOCK_SIZE);
        plant(&mut block, p - 1, 2500, 3);
        plant(&mut block, p, 40, 3);
        plant(&mut block, p, 600, 8);
        plant(&mut block, p + 3, 2000, 30);
        real(&block, p, 40, 3);
        real(&block, p, 600, 8);
        real(&block, p + 3, 2000, 30);
        real(&block, p - 1, 2500, 3);
        let mut cands = empty();
        cands[p - 1] = one(2500, 3);
        cands[p] = two((40, 3), (600, 8));
        cands[p + 3] = one(2000, 30);
        let mut pr = flat();
        pr.of[oc(2500)] = 6000;
        pr.of[oc(40)] = 2000;
        pr.of[oc(600)] = 100;
        let want = vec![seq(p as u32, 3, 43), seq(0, 30, 2003)];
        vec![case("later_record_not_priced_below_earlier", block, cands, vec![(pass(2), Some(pr), want)])]
    }

    /// Rep candidates with `ll0` (right after a match) are rep[1], rep[2], rep[0] - 1 (zstd's
    /// numbering, `ZSTD_newRep` on every node that ends a match): after 300 then 900, a match
    /// at offset 300 (= rep[1]) and then one at 299 (= rep[0] - 1 after the repcode) are found
    /// with no explicit candidate. An unused expensive 24-byte record at `p` keeps all four in
    /// one series, so the node reps (not only the committed ones) are exercised.
    pub fn rep_candidates_after_a_match() -> Vec<OptCase> {
        let p = 5000;
        let mut block = synth::random(105, BLOCK_SIZE);
        plant(&mut block, p, 2000, 24);
        plant(&mut block, p, 300, 6);
        plant(&mut block, p + 6, 900, 5);
        plant(&mut block, p + 11, 300, 7);
        plant(&mut block, p + 18, 299, 5);
        real(&block, p, 300, 6);
        real(&block, p + 6, 900, 5);
        real(&block, p + 11, 300, 7);
        real(&block, p + 18, 299, 5);
        real(&block, p, 2000, 24);
        let mut cands = empty();
        cands[p] = two((300, 6), (2000, 24));
        cands[p + 6] = one(900, 5);
        let mut pr = flat();
        pr.of[oc(2000)] = 20000;
        let want = vec![seq(p as u32, 6, 303), seq(0, 5, 903), seq(0, 7, 1), seq(0, 5, 3)];
        vec![case("rep_candidates_after_a_match", block, cands, vec![(pass(2), Some(pr.clone()), want.clone()), (pass(0), Some(pr), want)])]
    }

    /// optLevel 0 stops a record's downward length scan at the first length that does not
    /// improve its target (`if (optLevel==0) break`); optLevel 2 keeps scanning. At `p` R0
    /// (offset 1000, 10 bytes), at `p + 1` R1 (offset 100, 6 bytes), at `p + 6` R2 (offset
    /// 3000, 20 bytes). ML code prices make R1 lose at position 7 but win at 6: level 2 reaches
    /// `p + 6` through R1 (1 literal + 5), level 0 through R0 (6 bytes).
    pub fn level0_relaxation_abort() -> Vec<OptCase> {
        let p = 7000;
        let mut block = synth::random(106, BLOCK_SIZE);
        plant(&mut block, p, 1000, 10);
        plant(&mut block, p + 1, 100, 6);
        plant(&mut block, p + 6, 3000, 20);
        real(&block, p, 1000, 10);
        real(&block, p + 1, 100, 6);
        real(&block, p + 6, 3000, 20);
        let mut cands = empty();
        cands[p] = one(1000, 10);
        cands[p + 1] = one(100, 6);
        cands[p + 6] = one(3000, 20);
        let mut pr = flat();
        pr.of[oc(1000)] = 2000;
        pr.of[oc(100)] = 1300;
        pr.ml[3] = 500; // match length 6
        pr.ml[4] = 700; // match length 7
        let l2 = vec![seq(p as u32 + 1, 5, 103), seq(0, 20, 3003)];
        let l0 = vec![seq(p as u32, 6, 1003), seq(0, 20, 3003)];
        vec![case("level0_relaxation_abort", block, cands, vec![(pass(2), Some(pr.clone()), l2), (pass(0), Some(pr), l0)])]
    }

    /// optLevel 0 skips the match search at `cur` when `opt[cur+1].price <= opt[cur].price +
    /// 128`: R0 (offset 1000, 10 bytes) makes position 7 cheap, so a 20-byte match at `p + 6`
    /// is never searched at level 0 (R0 whole is emitted), while level 2 takes it.
    pub fn level0_skip_rule() -> Vec<OptCase> {
        let p = 8000;
        let mut block = synth::random(107, BLOCK_SIZE);
        plant(&mut block, p, 1000, 10);
        plant(&mut block, p + 6, 3000, 20);
        real(&block, p, 1000, 10);
        real(&block, p + 6, 3000, 20);
        let mut cands = empty();
        cands[p] = one(1000, 10);
        cands[p + 6] = one(3000, 20);
        let mut pr = flat();
        pr.of[oc(1000)] = 2000;
        pr.of[oc(3000)] = 100;
        let l2 = vec![seq(p as u32, 6, 1003), seq(0, 20, 3003)];
        let l0 = vec![seq(p as u32, 10, 1003)];
        vec![case("level0_skip_rule", block, cands, vec![(pass(2), Some(pr.clone()), l2), (pass(0), Some(pr), l0)])]
    }

    /// optLevel 2's match + 1 literal check (LL code 1 cheaper than code 0): at `p` A (offset
    /// 20, 3 bytes) and B (offset 700, 4 bytes). At position 4 "A + 1 literal" beats B, and
    /// then "B + 1 literal" beats "A + 2 literals" at position 5, so level 2 emits B and skips
    /// the literal after it; level 0 emits A.
    pub fn match_plus_one_literal() -> Vec<OptCase> {
        let p = 6000;
        let mut block = synth::random(108, BLOCK_SIZE);
        plant(&mut block, p, 20, 3);
        plant(&mut block, p, 700, 4);
        real(&block, p, 20, 3);
        real(&block, p, 700, 4);
        let mut cands = empty();
        cands[p] = two((20, 3), (700, 4));
        let mut pr = flat();
        pr.ll[0] = 600;
        pr.of[oc(20)] = 0;
        pr.of[oc(700)] = 500;
        let l2 = vec![seq(p as u32, 4, 703)];
        let l0 = vec![seq(p as u32, 3, 23)];
        vec![case("match_plus_one_literal", block, cands, vec![(pass(2), Some(pr.clone()), l2), (pass(0), Some(pr), l0)])]
    }

    /// `sufficient_len`: a match longer than `target_length` (32) at a series start is taken
    /// at once, even though one literal + a cheaper 45-byte match at the next position would
    /// cost less. At `p2` a rep0 match longer than `sufficient_len` returns from the match
    /// search at once, so the longer explicit candidate there is never seen.
    pub fn long_match_commits_immediately() -> Vec<OptCase> {
        let (p, p2) = (9000, 9140);
        let mut block = synth::random(109, BLOCK_SIZE);
        plant(&mut block, p + 1, 2000, 45);
        plant(&mut block, p, 500, 40);
        plant(&mut block, p2, 3000, 45);
        plant(&mut block, p2, 500, 40);
        real(&block, p, 500, 40);
        real(&block, p + 1, 2000, 45);
        real(&block, p2, 500, 40);
        real(&block, p2, 3000, 45);
        let mut cands = empty();
        cands[p] = one(500, 40);
        cands[p + 1] = one(2000, 45);
        cands[p2] = one(3000, 45);
        let mut pr = flat();
        pr.of[oc(500)] = 5000;
        pr.of[oc(2000)] = 0;
        let want = vec![seq(p as u32, 40, 503), seq((p2 - p - 40) as u32, 40, 1)];
        vec![case("long_match_commits_immediately", block, cands, vec![(pass(2), Some(pr), want)])]
    }

    /// A candidate running past the segment end is clamped to it (`iend`), which ends the
    /// series; the next segment starts at its first byte with empty reps and takes the rest,
    /// stored explicitly by the true reps (offset == reps[0] with ll 0 is not a repcode).
    pub fn seg_candidate_clamped_at_segment_end() -> Vec<OptCase> {
        let p = SEG - 10;
        let mut block = synth::random(110, BLOCK_SIZE);
        plant(&mut block, p, 100, 20);
        real(&block, p, 100, 20);
        let mut cands = empty();
        cands[p] = one(100, 20);
        cands[SEG] = one(100, 10);
        let want = vec![seq(p as u32, 10, 103), seq(0, 10, 103)];
        vec![case("seg_candidate_clamped_at_segment_end", block, cands, vec![(pass(2), Some(flat()), want)])]
    }

    /// Only positions `<= ilimit = iend - 8` search matches; a series running into `ilimit`
    /// still searches there: from `SEG - 12` (6 bytes, offset 300) the parse cuts to 4 bytes to
    /// take the 30-byte match at `ilimit` (clamped to 8, reaching `iend`: immediate encoding of
    /// the longest record, however expensive; a DP over it would prefer R0 + literals);
    /// segment 1 takes the remaining 22 bytes.
    pub fn seg_match_at_ilimit_in_series() -> Vec<OptCase> {
        let p = SEG - 12;
        let mut block = synth::random(111, BLOCK_SIZE);
        plant(&mut block, SEG - 8, 1500, 30);
        plant(&mut block, SEG - 8, 20, 3);
        plant(&mut block, p, 300, 6);
        real(&block, p, 300, 6);
        real(&block, SEG - 8, 1500, 30);
        real(&block, SEG - 8, 20, 3);
        let mut cands = empty();
        cands[p] = one(300, 6);
        cands[SEG - 8] = two((20, 3), (1500, 30));
        cands[SEG] = one(1500, 22);
        let mut pr = flat();
        pr.of[oc(1500)] = 20000;
        let want = vec![seq(p as u32, 4, 303), seq(0, 8, 1503), seq(0, 22, 1503)];
        vec![case("seg_match_at_ilimit_in_series", block, cands, vec![(pass(2), Some(pr), want)])]
    }

    /// Segments 1 and 2 are empty: the literals after segment 0's match run into segment 3's
    /// first sequence.
    pub fn seg_empty_segments_carry_literals() -> Vec<OptCase> {
        let far = 3 * SEG + 200;
        let mut block = synth::random(112, BLOCK_SIZE);
        plant(&mut block, 100, 50, 10);
        plant(&mut block, far, 300, 12);
        real(&block, 100, 50, 10);
        real(&block, far, 300, 12);
        let mut cands = empty();
        cands[100] = one(50, 10);
        cands[far] = one(300, 12);
        let want = vec![seq(100, 10, 53), seq((far - 110) as u32, 12, 303)];
        vec![case("seg_empty_segments_carry_literals", block, cands, vec![(pass(2), Some(flat()), want)])]
    }

    /// Offsets are re-encoded with the block's true reps: segment 1's first match (offset 50,
    /// explicit inside the segment) is repcode 2 after the carried literals (true reps
    /// [70, 50, 1]), and segment 2's is then repcode 1.
    pub fn seg_true_reps_across_segments() -> Vec<OptCase> {
        let mut block = synth::random(113, BLOCK_SIZE);
        let ps = [100, SEG - 20, SEG, 2 * SEG + 300];
        let offs = [50, 70, 50, 50];
        for (&p, &o) in ps.iter().zip(&offs) {
            plant(&mut block, p, o, if o == 70 { 6 } else { 10 });
        }
        let mut cands = empty();
        for (&p, &o) in ps.iter().zip(&offs) {
            let len = if o == 70 { 6 } else { 10 };
            real(&block, p, o, len);
            cands[p] = one(o, len as u32);
        }
        let want = vec![seq(100, 10, 53), seq(SEG as u32 - 130, 6, 73), seq(14, 10, 2), seq(SEG as u32 + 290, 10, 1)];
        vec![case("seg_true_reps_across_segments", block, cands, vec![(pass(2), Some(flat()), want)])]
    }

    /// A stored length of 64 (the compare cap) is extended to the true length, bounded by the
    /// segment end: 50 bytes before the end of segment 0, then 70 more in segment 1.
    pub fn seg_capped_candidate_extended() -> Vec<OptCase> {
        let p = SEG - 50;
        let mut block = synth::random(114, BLOCK_SIZE);
        plant(&mut block, p, 200, 120);
        real(&block, p, 200, 120);
        let mut cands = empty();
        cands[p] = one(200, 64);
        cands[SEG] = one(200, 64);
        let want = vec![seq(p as u32, 50, 203), seq(0, 70, 203)];
        vec![case("seg_capped_candidate_extended", block, cands, vec![(pass(2), Some(flat()), want)])]
    }

    /// The presets end to end (seeded prices and every pass): long planted matches (over
    /// `target_length`) are committed whatever the prices; the second, at the same offset after
    /// literals, is found as rep0 first, so the equal explicit candidate is not recorded.
    pub fn presets_commit_long_matches() -> Vec<OptCase> {
        let (p1, p2) = (1500, 3500);
        let mut block = synth::random(115, BLOCK_SIZE);
        plant(&mut block, p1, 1200, 40);
        plant(&mut block, p2, 1200, 40);
        real(&block, p1, 1200, 40);
        real(&block, p2, 1200, 40);
        let mut cands = empty();
        cands[p1] = one(1200, 40);
        cands[p2] = one(1200, 40);
        let want = vec![seq(p1 as u32, 40, 1203), seq((p2 - p1 - 40) as u32, 40, 1)];
        vec![case("presets_commit_long_matches", block, cands, vec![(OPT14, None, want.clone()), (OPT16, None, want)])]
    }

    /// After a series whose last stretch ends in literals, the next series starts *after* them
    /// (`ip = anchor + litlen`, the documented deviation from zstd 1.5.7, whose `} {` typo
    /// restarts at the anchor). Level 0, literals cost 100: the `+128` skip leaves `p + 5`
    /// unsearched inside the series, and the literal extension at `p + 6` beats R0's 6th byte,
    /// so the series ends as R0 (5 bytes) + 1 literal. The 20-byte match at `p + 5` is then
    /// never seen (restarting at the anchor would take it).
    pub fn tail_literals_are_not_reparsed() -> Vec<OptCase> {
        let p = 10000;
        let mut block = synth::random(116, BLOCK_SIZE);
        plant(&mut block, p, 1000, 6);
        plant(&mut block, p + 5, 400, 20);
        real(&block, p, 1000, 6);
        real(&block, p + 5, 400, 20);
        let mut cands = empty();
        cands[p] = one(1000, 6);
        cands[p + 5] = one(400, 20);
        let mut pr = flat();
        pr.lit = [100; 256];
        pr.of[oc(1000)] = 0;
        pr.of[oc(400)] = 0;
        pr.ml[3] = 110; // match length 6 costs 110 more than 5
        let want = vec![seq(p as u32, 5, 1003)];
        vec![case("tail_literals_are_not_reparsed", block, cands, vec![(pass(0), Some(pr), want)])]
    }

    /// A series that ends in a match commits it and the next series starts right there with
    /// `ll0`: its rep candidates come from `opt[0].rep` (the committed reps) with zstd's `ll0`
    /// numbering. After 300 then 900 (each its own series), the match at offset 300 (= rep[1])
    /// and then at 299 (= rep[0] - 1) start series of their own, with no explicit candidate.
    pub fn rep_candidates_at_series_start() -> Vec<OptCase> {
        let p = 11000;
        let mut block = synth::random(117, BLOCK_SIZE);
        plant(&mut block, p, 300, 6);
        plant(&mut block, p + 6, 900, 5);
        plant(&mut block, p + 11, 300, 7);
        plant(&mut block, p + 18, 299, 5);
        real(&block, p, 300, 6);
        real(&block, p + 6, 900, 5);
        real(&block, p + 11, 300, 7);
        real(&block, p + 18, 299, 5);
        let mut cands = empty();
        cands[p] = one(300, 6);
        cands[p + 6] = one(900, 5);
        let want = vec![seq(p as u32, 6, 303), seq(0, 5, 903), seq(0, 7, 1), seq(0, 5, 3)];
        vec![case("rep_candidates_at_series_start", block, cands, vec![(pass(2), Some(flat()), want.clone()), (pass(0), Some(flat()), want)])]
    }

    /// The last segment ends at the block end (`ilimit = PARSE_END`): a 12-byte match starting
    /// 12 bytes before it is priced through a DP series whose last node sits at `iend` itself.
    pub fn last_segment_match_to_block_end() -> Vec<OptCase> {
        let p = BLOCK_SIZE - 12;
        let mut block = synth::random(118, BLOCK_SIZE);
        plant(&mut block, p, 600, 12);
        real(&block, p, 600, 12);
        let mut cands = empty();
        cands[p] = one(600, 12);
        let want = vec![seq(p as u32, 12, 603)];
        vec![case("last_segment_match_to_block_end", block, cands, vec![(pass(2), Some(flat()), want.clone()), (pass(0), Some(flat()), want)])]
    }

    /// Every case above, in order.
    pub fn opt_test_cases() -> Vec<OptCase> {
        [
            literal_tie_drops_match,
            relax_tie_keeps_earlier,
            equal_length_explicit_after_rep_dropped,
            later_record_not_priced_below_earlier,
            rep_candidates_after_a_match,
            level0_relaxation_abort,
            level0_skip_rule,
            match_plus_one_literal,
            long_match_commits_immediately,
            seg_candidate_clamped_at_segment_end,
            seg_match_at_ilimit_in_series,
            seg_empty_segments_carry_literals,
            seg_true_reps_across_segments,
            seg_capped_candidate_extended,
            presets_commit_long_matches,
            tail_literals_are_not_reparsed,
            rep_candidates_at_series_start,
            last_segment_match_to_block_end,
        ]
        .into_iter()
        .flat_map(|f| f())
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::cases::{self, run_case};
    use super::*;
    use crate::block::chunk_file;
    use crate::frame::{write_frame, FrameOptions};
    use crate::params::{OPT14, OPT16};
    use crate::reference::{chains, compress_block, find_cands};
    use crate::seq::reconstruct;
    use crate::synth;

    /// The presets and the oracle variants the GPU is checked against: passes 0 / 1 / 3 with
    /// both seeds, and an optLevel-0 final pass.
    fn variants() -> Vec<MatchParams> {
        let mut v = vec![OPT14, OPT16];
        for passes in [0, 1, 3] {
            for seed in [Seed::BlockInit, Seed::Prior] {
                v.push(MatchParams { opt: Some(OptParams { passes, seed, ..OPT16.opt.unwrap() }), ..OPT16 });
            }
        }
        v.push(MatchParams { opt: Some(OptParams { level: 0, ..OPT16.opt.unwrap() }), ..OPT16 });
        v
    }

    /// Every synthetic block round-trips through libzstd for every variant, and no match crosses
    /// a 4 KiB segment.
    #[test]
    fn synthetic_roundtrip_all_variants() {
        // opt only validates at blocks of at most 64 KiB.
        if crate::config::LOG2_BLOCK > 16 {
            return;
        }
        for params in variants() {
            for (name, bytes) in synth::test_cases() {
                for (i, blk) in chunk_file(&bytes).into_iter().enumerate() {
                    let out = compress_block(&blk.data, params);
                    let mut pos = 0usize;
                    for q in &out.sequences {
                        let start = pos + q.lit_len as usize;
                        pos = start + q.match_len as usize;
                        assert!(q.match_len >= 3);
                        assert_eq!(start >> 12, (pos - 1) >> 12, "{name}[{i}]: match {start}..{pos} crosses a segment");
                    }
                    let frame = write_frame(&blk.data, &out, FrameOptions::default());
                    let dec = zstd::bulk::decompress(&frame, BLOCK_SIZE)
                        .unwrap_or_else(|e| panic!("{params:?} {name}[{i}]: libzstd rejected frame: {e}"));
                    assert_eq!(dec, blk.data, "{params:?} {name}[{i}]: frame mismatch");
                }
            }
        }
    }

    /// The ring engine (GPU spec) equals the zstd-layout engine on every synthetic block, for
    /// every variant.
    #[test]
    fn ring_engine_matches_linear_synthetic() {
        for params in variants() {
            for (name, bytes) in synth::test_cases() {
                for (i, blk) in chunk_file(&bytes).into_iter().enumerate() {
                    let cands = find_cands(&blk.data, &chains(&blk.data, &params), &params);
                    let a = parse_with(&blk.data, &cands, &params, Engine::Linear);
                    let b = parse_with(&blk.data, &cands, &params, Engine::Ring);
                    assert!(a == b, "{params:?} {name}[{i}]: ring engine differs");
                }
            }
        }
    }

    /// `passes` is `parse` with every pass exposed: `o.passes + 1` passes, the last is `parse`'s
    /// output at `o.level`, pass 0 runs the seed prices and pass n + 1 the prices of pass n's
    /// histogram; every price entry fits the GPU's u16 tables.
    #[test]
    fn passes_chain_prices_and_end_in_parse() {
        for params in [OPT14, OPT16] {
            let o = params.opt.unwrap();
            for (name, bytes) in synth::test_cases() {
                let blk = &chunk_file(&bytes)[0];
                let cands = find_cands(&blk.data, &chains(&blk.data, &params), &params);
                let ps = passes(&blk.data, &cands, &params);
                assert_eq!(ps.len(), o.passes as usize + 1, "{name}");
                assert_eq!(ps[0].prices, seed_prices(&blk.data, &cands, o.seed), "{name}");
                for w in ps.windows(2) {
                    assert_eq!(w[1].prices, Prices::from_hist(&Hist::of_output(&w[0].out)), "{name}");
                    assert_eq!(w[0].level, 0, "{name}");
                }
                assert_eq!(ps.last().unwrap().level, o.level);
                assert_eq!(ps.last().unwrap().out, parse(&blk.data, &cands, &params), "{name}");
                for pass in &ps {
                    let p = &pass.prices;
                    let all = p.lit.iter().chain(&p.ll).chain(&p.ml).chain(&p.of);
                    assert!(all.clone().all(|&x| (0..65536).contains(&x)), "{name}: price out of u16 range");
                    let h = Hist::of_output(&pass.out);
                    assert_eq!(h.lit.iter().sum::<u32>() as usize, pass.out.literals.len());
                    assert_eq!(h.ll.iter().sum::<u32>() as usize, pass.out.sequences.len());
                    assert_eq!(h.to_text().lines().count(), 4);
                }
            }
        }
    }

    /// zstd's first-block prices on a known block: every byte value 256 times (frequency 2 each,
    /// sum 512), so a literal costs `W(512) - W(2)`.
    #[test]
    fn block_init_prices_follow_zstd() {
        let block: Vec<u8> = (0..BLOCK_SIZE).map(|i| (i % 256) as u8).collect();
        let p = Prices::block_init(&block);
        let f = (BLOCK_SIZE / 256) as u32;
        let (fl, sum) = ((f > 0) as u32 + (f >> 8), 256 * ((f > 0) as u32 + (f >> 8)));
        assert!(p.lit.iter().all(|&x| x == (frac_weight(sum) - frac_weight(fl).min(frac_weight(sum) - 256)) as i32));
        assert_eq!(frac_weight(0), 256);
        assert_eq!(frac_weight(1), 512);
        assert_eq!(frac_weight(2), 256 + 384);
        // LL {4, 2, 1 x 34}: sum 40; OF {6, 2, 1, 1, 2, 3, 4, 4, 4, 3, 2, 1 x 21}: sum 53.
        assert_eq!(p.ll[0], (frac_weight(40) - frac_weight(4)) as i32);
        assert_eq!(p.ll[35], (16 * 256 + frac_weight(40) - frac_weight(1)) as i32);
        assert_eq!(p.of[2], (2 * 256 + frac_weight(53) - frac_weight(1)) as i32);
        assert_eq!(p.ml[52], (16 * 256 + frac_weight(53) - frac_weight(1)) as i32);
    }

    #[test]
    fn deterministic() {
        // opt only validates at blocks of at most 64 KiB.
        if crate::config::LOG2_BLOCK > 16 {
            return;
        }
        let bytes = synth::dds_like(9, BLOCK_SIZE);
        for params in [OPT14, OPT16] {
            assert_eq!(compress_block(&bytes, params), compress_block(&bytes, params));
        }
    }

    /// Informal (reads the real corpus): 4000 blocks spread over `data/corpus` (every
    /// `total / 4000`-th .dds/.nif block in path order) round-trip through libzstd for every
    /// variant, and every other one of them (2000, i.e. every ~50th block at 64 KiB) gives the
    /// same output on the ring engine.
    /// `GZC_CORPUS=/path/to/data/corpus cargo test --release -p gzc-core opt_corpus_roundtrip -- --ignored`
    #[test]
    #[ignore]
    fn opt_corpus_roundtrip() {
        use std::path::{Path, PathBuf};
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let mut ents: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
            ents.sort();
            for p in ents {
                let e = p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
                if p.is_dir() {
                    walk(&p, out);
                } else if e == "dds" || e == "nif" {
                    out.push(p);
                }
            }
        }
        let root = std::env::var("GZC_CORPUS")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/corpus").to_string());
        let mut files = Vec::new();
        walk(Path::new(&root), &mut files);
        let mut blocks = Vec::new();
        for f in files {
            blocks.extend(chunk_file(&std::fs::read(&f).unwrap()).into_iter().map(|b| b.data));
        }
        let stride = (blocks.len() / 4000).max(1);
        let sample: Vec<Vec<u8>> = blocks.into_iter().step_by(stride).take(4000).collect();
        let vars = variants();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(b) = sample.get(i) else { break };
                    let cands = find_cands(b, &chains(b, &OPT16), &OPT16);
                    for params in &vars {
                        let out = parse(b, &cands, params);
                        if i.is_multiple_of(2) {
                            assert!(parse_with(b, &cands, params, Engine::Ring) == out, "block {i} {params:?}: ring engine differs");
                        }
                        let frame = write_frame(b, &out, FrameOptions::default());
                        let dec = zstd::bulk::decompress(&frame, BLOCK_SIZE).expect("libzstd rejected an opt frame");
                        assert!(dec == *b, "block {i} {params:?}: round trip mismatch");
                    }
                });
            }
        });
        println!("{} corpus blocks x {} variants round-tripped", sample.len(), vars.len());
    }

    #[test]
    fn opt_cases() {
        let mut failed = Vec::new();
        for c in cases::opt_test_cases() {
            for (i, (params, prices, want)) in c.expect.iter().enumerate() {
                let out = run_case(&c.block, &c.cands, params, prices.as_ref(), Engine::Linear);
                let ring = run_case(&c.block, &c.cands, params, prices.as_ref(), Engine::Ring);
                assert_eq!(ring, out, "{} [{i}]: ring engine differs", c.name);
                assert_eq!(reconstruct(&out).expect("reconstruct"), c.block, "{} [{i}]: output does not reconstruct", c.name);
                if out.sequences != *want {
                    failed.push(format!("{} [{i}] level {}: got {:?}\n    want {:?}", c.name, params.opt.unwrap().level, out.sequences, want));
                }
            }
        }
        assert!(failed.is_empty(), "\n{}", failed.join("\n"));
    }
}
