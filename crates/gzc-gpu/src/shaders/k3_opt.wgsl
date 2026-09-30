// K3opt: one DP pass of the M5 optimal parse (presets opt14 / opt16), == gzc_core::opt::dp_pass_with
// on Engine::Ring (the normative GPU spec, documented on opt::Engine), statement by statement.
// Two entry points, dispatched back to back:
//
// main_opt (one invocation per 4 KiB segment, NSEG per block; WG lanes per workgroup, BPW blocks
//   per workgroup, or a block across NSEG / WG workgroups): a prologue builds the block's static
//   price tables in workgroup memory (PRICE_MODE 0: zstd's first-block statistics,
//   opt::Prices::block_init, the literal histogram counted by the workgroup's lanes of the block;
//   PRICE_MODE 1: the tables in `prices`, opt::Prices verbatim, 377 words per block: lit[256]
//   ll[36] ml[53] of[32]). Then each lane runs opt::Dp::segment_ring over its segment: segment 0
//   from ip 1 with INITIAL_REPS, segment k > 0 from ip = anchor = k*SEG with reps [0, 0, 0];
//   iend = the segment end, ilimit = iend - 8.
//   Phase 1, the DP:
//   - The lane loop is flattened (one trip per series start probe or series position, no
//     `continue`) so that the lanes' get_all_matches calls run together; each trip issues its
//     position's loads (data, candidate words) first.
//   - The DP nodes live in a ring of RING_N = sufficient_len + 1 slots per lane (position pos of
//     the series in slot pos % RING_N), in workgroup memory or in private memory (the fallback
//     when the adapter's workgroup storage is too small); both declared by the host (rix).
//   - `trace` (the dead K1 pred buffer, 2 words per position) gets (mlen | litlen << 8, offBase)
//     of every node when it becomes final; the backward trace reads it, never the ring. The entry
//     at iend (a node at the segment end) is never read and not written (it would be the next
//     segment's first position).
//   - The end of a series (commit_ring) only updates the parse state (end_series: ip, anchor and
//     reps follow from the last stretch alone) and logs the series in `seqs` (free until
//     main_fixup); no lane waits for another's backward trace inside the DP loop.
//   Phase 2: the logged series' backward traces, last series first (emit_series), write the
//   segment's sequences in reverse order (RAW_REVERSED) into its own words of `best` (the
//   candidate words, 2 per position, no longer read): sequence i from the segment's end at
//   best[wbase + 3*i ..], trailer (n_seq, final anchor, sum of match_len, final reps) at
//   best[wbase + SEG_META ..], for k3_fixup.wgsl's main_fixup (at most SEG / 3 sequences).
// main_fixup (k3_fixup.wgsl): concatenation, literal carry and off_base re-encoding against the
//   block's true reps (== lazy::encode_raw), one workgroup per block.
//
// Candidate words (reference::CandWords, K2opt): best[2*(b*BLOCK_SIZE + p)] = offA | lenA << 16 |
// lenB << 24, best[.. + 1] = offB; lengths capped at SEARCH_CAP (a stored SEARCH_CAP is extended).
//
// Precondition (K2opt's output satisfies it): every candidate record lies in the block before its
// position, and its length is the true common length capped at SEARCH_CAP.
//
// Passes (M5 T4, opt::passes): pass n + 1 is priced from pass n's own output histogram
// (opt::Hist::of_output of the fixed-up block parse: literal bytes, and each sequence's LL code
// with the literal carry, ML code, and OF code of the off_base under the block's true decoder
// reps). A HIST_OUT pass (the cheap ones) keeps the candidate words: its raw sequences go to the
// segment's series-log words of `seqs` instead of `best`, and a workgroup epilogue (hist_epilogue)
// histograms them: every lane counts its own sequences and literal bytes, then the block's
// segment-0 lane applies what main_fixup would change (the literal carry of each segment's first
// sequence, and the OF codes of the sequences it re-encodes against the true reps), and the
// block's lanes store the Hist (377 words: lit[256] ll[36] ml[53] of[32]) at `prices`[b * 377].
// The next pass's prologue (PRICE_MODE 3) turns it into price tables (opt::Prices::from_hist).
// No fix-up runs after a HIST_OUT pass. A HIST_OUT pass needs each block's segments in one
// workgroup (LPB == NSEG).
//
// PRICE_MODE: 0 block-init (opt::Prices::block_init), 1 explicit tables in `prices`, 2 the prior
// seed (codes::OPT_PRIOR_* tables plus opt::cover_literals of the block's candidate words; needs
// LPB == NSEG), 3 opt::Prices::from_hist of the Hist in `prices`.
//
// Consts injected by the host: MIN_MATCH, SEARCH_CAP, MAX_SEQS, SEG_LOG2, WG, SUFF
// (sufficient_len), LEVEL (optLevel 0 | 2), PRICE_MODE, HIST_OUT, BI_LL / BI_ML / BI_OF
// (block-init LL / ML / OF prices), PR_LL / PR_ML / PR_OF (prior-seed LL / ML / OF prices),
// LL_BITS / ML_BITS / ML_CODE (codes.rs), and the ring declarations (ring_p/a/b/c, rix).

const SEG: u32 = 1u << SEG_LOG2;
const NSEG: u32 = BLOCK_SIZE >> SEG_LOG2;
const SEG_WORDS: u32 = 2u * SEG;
const SEG_META: u32 = SEG_WORDS - 6u;
const RAW_REVERSED: bool = true;
// Lanes of one block in a workgroup, and blocks per workgroup (WG < NSEG: a block spans
// NSEG / WG workgroups, each with its own copy of the block's price tables).
const LPB: u32 = min(WG, NSEG);
const BPW: u32 = WG / LPB;
const RING_N: u32 = SUFF + 1u;
const MAXP: i32 = 1 << 30;
const OPT_NUM: u32 = 4096u;
const MATCH_FEE: i32 = 51;
const PRICE_WORDS: u32 = 377u;
// seqs words per segment for the series log: at most SEG / 3 series with a match (each covers at
// least one 3-byte match, and series are disjoint), 3 words each.
const LOG_WORDS: u32 = (3u * MAX_SEQS) / NSEG;
// HIST_OUT: the segment's raw sequences, in order, end at word lbase + 3 * LOG_SEQS (written last
// first while phase 2 still reads the series log from the front: raw sequence j from the end
// lands at slot LOG_SEQS - 1 - j, which holds a series entry already read, since each unread
// series still has at least one sequence to emit and a segment has at most SEG / 3).
const LOG_SEQS: u32 = LOG_WORDS / 3u;
const_assert 3u * (SEG / 3u) <= LOG_WORDS;
const_assert SEG / 3u <= LOG_SEQS;
const HIST_WORDS: u32 = 377u;
// Sizes of the HIST_OUT-only workgroup arrays (1 when unused).
const HO: u32 = select(0u, 1u, HIST_OUT);
const_assert BLOCK_SIZE <= 65536u;
const_assert WG % NSEG == 0u || NSEG % WG == 0u;
const_assert 3u * (SEG / 3u) <= SEG_META;
const_assert NSEG * (SEG / 3u) <= MAX_SEQS;
// ML code == mlen - 3 for every priced length (mlen <= SUFF <= 34).
const_assert SUFF <= 34u;
const_assert MIN_MATCH == 3u;

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> best: array<u32>;
@group(0) @binding(2) var<storage, read_write> seqs: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> trace: array<u32>;
// PRICE_MODE 1: opt::Prices per block; PRICE_MODE 3 / HIST_OUT: opt::Hist per block (the
// previous pass's, replaced by this pass's). PRICE_WORDS == HIST_WORDS words per block.
@group(0) @binding(5) var<storage, read_write> prices: array<u32>;

// zstd LL_Code for lit_len < 64 (codes::ll_code).
const LL_CODE: array<u32, 64> = array<u32, 64>(
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u,
    16u, 16u, 17u, 17u, 18u, 18u, 19u, 19u, 20u, 20u, 20u, 20u, 21u, 21u, 21u, 21u,
    22u, 22u, 22u, 22u, 22u, 22u, 22u, 22u, 23u, 23u, 23u, 23u, 23u, 23u, 23u, 23u,
    24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u);

// Price tables of the workgroup's blocks (local block pb at pb * size).
var<workgroup> p_lit: array<i32, 256u * BPW>;
// ll_price(litlen) for litlen < 64.
var<workgroup> p_lls: array<i32, 64u * BPW>;
// LL price by code (litlen >= 64: code highbit(litlen) + 19).
var<workgroup> p_llc: array<i32, 36u * BPW>;
// ML price by match length (3..=SUFF).
var<workgroup> p_ml: array<i32, RING_N * BPW>;
var<workgroup> p_of: array<i32, 32u * BPW>;
// The prologue's literal frequencies; with HIST_OUT, then the pass's literal histogram.
var<workgroup> hist: array<atomic<u32>, 256u * BPW>;
// Per block: the table sums (lit, ll, ml, of) and, for PRICE_MODE 2, the uncovered bytes.
var<workgroup> hsum: array<atomic<u32>, 5u * BPW>;
// HIST_OUT: the pass's LL / ML / OF code histograms, and each lane's segment summary (sequences,
// final anchor, final reps).
var<workgroup> o_ll: array<atomic<u32>, max(36u * BPW * HO, 1u)>;
var<workgroup> o_ml: array<atomic<u32>, max(53u * BPW * HO, 1u)>;
var<workgroup> o_of: array<atomic<u32>, max(32u * BPW * HO, 1u)>;
var<workgroup> sg_n: array<u32, max(WG * HO, 1u)>;
var<workgroup> sg_anchor: array<u32, max(WG * HO, 1u)>;
var<workgroup> sg_rep: array<vec3<u32>, max(WG * HO, 1u)>;

// Lane constants.
var<private> lane: u32;
var<private> pb: u32;
var<private> base: u32;
var<private> cbase: u32;
var<private> tbase: u32;
var<private> wbase: u32;

// Records of the last get_all_matches: (offBase, length), strictly increasing length.
var<private> m_ob: array<u32, 5>;
var<private> m_len: array<u32, 5>;
var<private> m_n: u32;

// Segment state (opt::SegState) and output counters.
var<private> st_ip: u32;
var<private> st_anchor: u32;
var<private> st_rep: vec3<u32>;
var<private> n_seq: u32;
var<private> ml_sum: u32;
// Phase 1 series log: seqs[lbase + 3*i ..] (free until main_fixup).
var<private> lbase: u32;
var<private> n_series: u32;

// opt::frac_weight
fn frac_weight(raw: u32) -> u32 {
    let stat = raw + 1u;
    let hb = firstLeadingBit(stat);
    return hb * 256u + ((stat << 8u) >> hb);
}

fn ll_price(litlen: u32) -> i32 {
    if (litlen < 64u) { return p_lls[pb * 64u + litlen]; }
    return p_llc[pb * 36u + firstLeadingBit(litlen) + 19u];
}

// Literal price of byte value `c`.
fn lit_cost(c: u32) -> i32 {
    return p_lit[pb * 256u + c];
}

fn of_price(ob: u32) -> i32 {
    return p_of[pb * 32u + firstLeadingBit(ob)];
}

fn match_price(ob: u32, mlen: u32) -> i32 {
    return of_price(ob) + p_ml[pb * RING_N + mlen] + MATCH_FEE;
}

// seq::apply_off_base on a history value: the reps after offBase `ob` with `ll` literals.
fn rep_after(r: vec3<u32>, ob: u32, ll: u32) -> vec3<u32> {
    if (ob > 3u) { return vec3<u32>(ob - 3u, r.x, r.y); }
    let idx = ob - 1u + select(0u, 1u, ll == 0u);
    switch (idx) {
        case 0u: { return r; }
        case 1u: { return vec3<u32>(r.y, r.x, r.z); }
        case 2u: { return vec3<u32>(r.z, r.x, r.y); }
        default: { return vec3<u32>(r.x - 1u, r.x, r.y); }
    }
}

// opt::new_rep (ZSTD_newRep).
fn new_rep(r: vec3<u32>, ob: u32, ll0: bool) -> vec3<u32> {
    return rep_after(r, ob, select(1u, 0u, ll0));
}

// A DP node (opt::Node). In the ring: ring_p = price, ring_a = rep0 | rep1 << 16,
// ring_b = rep2 | litlen << 16, ring_c = mlen | offBase << 8 (mlen <= SUFF, reps < 65536).
struct Node {
    price: i32,
    r: vec3<u32>,
    litlen: u32,
    mlen: u32,
    ob: u32,
}

fn slot(pos: u32) -> u32 { return pos % RING_N; }

fn ld(pos: u32) -> Node {
    let i = rix(slot(pos));
    let a = ring_a[i];
    let b = ring_b[i];
    let c = ring_c[i];
    return Node(ring_p[i], vec3<u32>(a & 0xFFFFu, a >> 16u, b & 0xFFFFu), b >> 16u, c & 0xFFu, c >> 8u);
}

fn ld_price(pos: u32) -> i32 {
    return ring_p[rix(slot(pos))];
}

fn st(pos: u32, n: Node) {
    let i = rix(slot(pos));
    ring_p[i] = n.price;
    ring_a[i] = n.r.x | (n.r.y << 16u);
    ring_b[i] = n.r.z | (n.litlen << 16u);
    ring_c[i] = n.mlen | (n.ob << 8u);
}

// The relaxation's fill: price MAX, litlen 1, the rest left as it was.
fn fill(pos: u32) {
    let i = rix(slot(pos));
    ring_p[i] = MAXP;
    ring_b[i] = (ring_b[i] & 0xFFFFu) | (1u << 16u);
}

fn push_match(ob: u32, len: u32) {
    m_ob[m_n] = ob;
    m_len[m_n] = len;
    m_n += 1u;
}

// One rep probe of get_all_matches: offBase `ob`, source offset `ro` (valid: 1 <= ro <= p), `x`
// and `y` the 4 bytes at p and p - ro. Records it when its length (the common prefix within
// `lim`) is at least MIN_MATCH and beats `bestl`; true when the search ends there.
fn rep_probe(p: u32, lim: u32, ob: u32, ro: u32, valid: bool, x: u32, y: u32, bestl: ptr<function, u32>) -> bool {
    let d = x ^ y;
    if (!valid || (d & 0xFFFFFFu) != 0u) { return false; }
    // lim >= 8 at every searched position (p <= ilimit = iend - 8).
    var rl = 3u;
    if (d == 0u) { rl = 4u + match_len(base, p + 4u, p - ro + 4u, lim - 4u); }
    if (rl <= *bestl) { return false; }
    *bestl = rl;
    push_match(ob, rl);
    return rl > SUFF || rl == lim;
}

// opt::Dp::get_all_matches: records at p (<= ilimit) for reps `r` and `ll0`, lengths clamped
// to iend - p. `x` holds the 4 bytes at p and (w0, w1) the candidate words of p, loaded by the
// caller; the three rep sources are loaded together before any is compared.
fn get_all_matches(p: u32, r: vec3<u32>, ll0: bool, iend: u32, x: u32, w0: u32, w1: u32) {
    m_n = 0u;
    let lim = iend - p;
    var bestl = MIN_MATCH - 1u;
    // Rep indices ll0 .. ll0 + 2 (index 3 = rep0 - 1), offBase 1 .. 3.
    let ro0 = select(r.x, r.y, ll0);
    let ro1 = select(r.y, r.z, ll0);
    let ro2 = select(r.z, r.x - 1u, ll0);
    let v0 = ro0 - 1u < p;
    let v1 = ro1 - 1u < p;
    let v2 = ro2 - 1u < p;
    let y0 = load_u32_at(base, p - select(0u, ro0, v0));
    let y1 = load_u32_at(base, p - select(0u, ro1, v1));
    let y2 = load_u32_at(base, p - select(0u, ro2, v2));
    if (rep_probe(p, lim, 1u, ro0, v0, x, y0, &bestl)) { return; }
    if (rep_probe(p, lim, 2u, ro1, v1, x, y1, &bestl)) { return; }
    if (rep_probe(p, lim, 3u, ro2, v2, x, y2, &bestl)) { return; }
    for (var j = 0u; j < 2u; j += 1u) {
        let off = select(w1 & 0xFFFFu, w0 & 0xFFFFu, j == 0u);
        let len = select(w0 >> 24u, (w0 >> 16u) & 0xFFu, j == 0u);
        if (len == 0u) { continue; }
        var l = min(len, lim);
        if (len == SEARCH_CAP) { l = match_len(base, p, p - off, lim); }
        if (l > bestl) {
            bestl = l;
            push_match(off + 3u, l);
            if (l == lim) { break; }
        }
    }
}

// == codes::ll_code / codes::ml_code.
fn ll_code(ll: u32) -> u32 {
    if (ll > 63u) { return firstLeadingBit(ll) + 19u; }
    return LL_CODE[ll];
}
fn ml_code(ml: u32) -> u32 {
    let b = ml - 3u;
    if (b > 127u) { return firstLeadingBit(b) + 36u; }
    return ML_CODE[b];
}

// HIST_OUT: counts the literal bytes at [a, e) of the lane's block.
fn hist_lits(a: u32, e: u32) {
    // Terminates: q rises to e.
    for (var q = a; q < e; q += 1u) {
        atomicAdd(&hist[pb * 256u + load_byte(base, q)], 1u);
    }
}

// HIST_OUT: counts one raw sequence (its match at block position `m`) under the segment's own
// history: its literal bytes and its LL / ML / OF codes (hist_epilogue fixes the rest).
fn hist_seq(m: u32, ll: u32, ml: u32, ob: u32) {
    hist_lits(m - ll, m);
    atomicAdd(&o_ll[pb * 36u + ll_code(ll)], 1u);
    atomicAdd(&o_ml[pb * 53u + ml_code(ml)], 1u);
    atomicAdd(&o_of[pb * 32u + firstLeadingBit(ob)], 1u);
}

fn trace_put(p: u32, n: Node) {
    let t = tbase + 2u * p;
    trace[t] = n.mlen | (n.litlen << 8u);
    trace[t + 1u] = n.ob;
}

// Phase 1 end of a series (opt::Dp::commit_ring's effect on the parse state): `last` ends the
// series started at `sip`, at series position `last_pos`. The stored sequences cover the literals
// before the series and every stretch up to the end of the last match, so the anchor moves to
// sip + last_pos - last.litlen and ip to sip + last_pos. A series with a match is logged for
// phase 2; its trace entries (positions sip ..= sip + last_pos - 1) are never written again.
fn end_series(last: Node, sip: u32, last_pos: u32) {
    st_ip = sip + last_pos;
    if (last.mlen == 0u) { return; }
    st_rep = last.r;
    st_anchor = st_ip - last.litlen;
    let l = lbase + 3u * n_series;
    seqs[l] = sip | (last.mlen << 16u);
    seqs[l + 1u] = last_pos | (last.litlen << 16u);
    seqs[l + 2u] = last.ob;
    n_series += 1u;
}

// Phase 2 (opt::Dp::commit_ring's output): the backward trace of one logged series, appending
// its sequences last first to the segment's words of `best`. Series are emitted last first too,
// so the segment's raw sequences end up in reverse order (RAW_REVERSED).
//
// The DP's offBases are stored as they are: each is seq::off_base_for(offset, lit_len) under the
// DP's own history, as main_fixup needs. A rep record (index i, zstd's ll0 numbering) is only
// kept when no earlier rep index holds the same offset (that one gives the same length first),
// which is off_base_for's order; an explicit record whose offset equals a probed rep has that
// rep's length (candidate lengths are true capped lengths, extended at the cap, both clamped to
// iend) and so is never recorded after it; the last rep index needs rep0 > 1 in both. So the
// DP's history is the canonical one and the trailer's reps are st_rep.
//
// HIST_OUT: the sequences go to the segment's series-log words instead (see LOG_SEQS), and each
// is counted in the pass's histograms with its literal bytes (hist_seq).
fn emit_series(sip: u32, last_pos: u32, last_mlen: u32, last_litlen: u32, last_ob: u32) {
    var sp = last_pos - last_mlen - last_litlen;
    var mlen = last_mlen;
    var ob = last_ob;
    loop {
        let t = tbase + 2u * (sip + sp);
        let t0 = trace[t];
        let nm = t0 & 0xFFu;
        let nl = t0 >> 8u;
        if (HIST_OUT) {
            let o = lbase + 3u * (LOG_SEQS - 1u - n_seq);
            seqs[o] = nl;
            seqs[o + 1u] = mlen;
            seqs[o + 2u] = ob;
            hist_seq(sip + sp, nl, mlen, ob);
        } else {
            let o = wbase + 3u * n_seq;
            best[o] = nl;
            best[o + 1u] = mlen;
            best[o + 2u] = ob;
        }
        n_seq += 1u;
        ml_sum += mlen;
        // Terminates: sp falls by nl + nm >= 3 per step (a node ending a match has mlen >= 3);
        // the series start (sp = 0) has nm = 0.
        if (nm == 0u || sp < nl + nm) { break; }
        mlen = nm;
        ob = trace[t + 1u];
        sp -= nl + nm;
    }
}

// Literal price from the frequency `f` and the base weight `lb` (opt::Prices::from_freqs).
fn lit_from(f: u32, lb: u32) -> i32 {
    return i32(lb - min(frac_weight(f), lb - 256u));
}

// opt::Prices::from_hist's frequency of a count.
fn seen(c: u32) -> u32 {
    return c + select(0u, 1u, c > 0u);
}

// The block's literal byte counts into hist (LPB lanes striding over its words).
fn count_block(blk: u32, kl: u32, b: u32) {
    let w0 = block_base(b);
    for (var w = kl; w < BLOCK_SIZE / 4u; w += LPB) {
        let v = data[w0 + w];
        atomicAdd(&hist[blk * 256u + (v & 0xFFu)], 1u);
        atomicAdd(&hist[blk * 256u + ((v >> 8u) & 0xFFu)], 1u);
        atomicAdd(&hist[blk * 256u + ((v >> 16u) & 0xFFu)], 1u);
        atomicAdd(&hist[blk * 256u + (v >> 24u)], 1u);
    }
}

// PRICE_MODE 2 (opt::cover_literals): the bytes of lane kl's chunk of the block that no
// candidate covers into hist, and their number into hsum[blk * 5 + 4]. Position p is covered
// when some p' <= p has p' + lenB(p') > p; lenB <= SEARCH_CAP, so the chunk's running maximum
// starts from the SEARCH_CAP positions before it.
fn cover_chunk(blk: u32, kl: u32, b: u32) {
    const C: u32 = BLOCK_SIZE / LPB;
    let cb = 2u * b * BLOCK_SIZE;
    let bb = block_base(b);
    let p0 = kl * C;
    var reach = 0u;
    // Terminates: p rises to p0.
    for (var p = p0 - min(p0, SEARCH_CAP); p < p0; p += 1u) {
        reach = max(reach, p + (best[cb + 2u * p] >> 24u));
    }
    var n = 0u;
    // Terminates: p rises to p0 + C.
    for (var p = p0; p < p0 + C; p += 1u) {
        reach = max(reach, p + (best[cb + 2u * p] >> 24u));
        if (reach <= p) {
            atomicAdd(&hist[blk * 256u + load_byte(bb, p)], 1u);
            n += 1u;
        }
    }
    atomicAdd(&hsum[blk * 5u + 4u], n);
}

// The workgroup's price tables (see the header). Every lane calls it (it has barriers).
fn prologue(valid: bool, b: u32) {
    let blk = lane / LPB;
    let kl = lane % LPB;
    if (PRICE_MODE == 0u || PRICE_MODE == 2u) {
        for (var i = lane; i < 256u * BPW; i += WG) { atomicStore(&hist[i], 0u); }
        for (var i = lane; i < 5u * BPW; i += WG) { atomicStore(&hsum[i], 0u); }
        workgroupBarrier();
        if (valid) {
            if (PRICE_MODE == 0u) { count_block(blk, kl, b); } else { cover_chunk(blk, kl, b); }
        }
        workgroupBarrier();
        // Every byte covered: the whole block's histogram.
        if (PRICE_MODE == 2u && valid && atomicLoad(&hsum[blk * 5u + 4u]) == 0u) {
            count_block(blk, kl, b);
        }
        workgroupBarrier();
        for (var e = kl; e < 256u; e += LPB) {
            let c = atomicLoad(&hist[blk * 256u + e]);
            var f = seen(c);
            if (PRICE_MODE == 0u) { f = select(0u, 1u, c > 0u) + (c >> 8u); }
            atomicStore(&hist[blk * 256u + e], f);
            atomicAdd(&hsum[blk * 5u], f);
        }
        workgroupBarrier();
        let lb = frac_weight(max(atomicLoad(&hsum[blk * 5u]), 1u));
        for (var e = kl; e < 256u; e += LPB) {
            p_lit[blk * 256u + e] = lit_from(atomicLoad(&hist[blk * 256u + e]), lb);
        }
        if (PRICE_MODE == 0u) {
            for (var c = kl; c < 36u; c += LPB) { p_llc[blk * 36u + c] = BI_LL[c]; }
            for (var l = kl; l < 64u; l += LPB) { p_lls[blk * 64u + l] = BI_LL[LL_CODE[l]]; }
            for (var m = kl; m < RING_N; m += LPB) { p_ml[blk * RING_N + m] = BI_ML[max(m, 3u) - 3u]; }
            for (var c = kl; c < 32u; c += LPB) { p_of[blk * 32u + c] = BI_OF[c]; }
        } else {
            for (var c = kl; c < 36u; c += LPB) { p_llc[blk * 36u + c] = PR_LL[c]; }
            for (var l = kl; l < 64u; l += LPB) { p_lls[blk * 64u + l] = PR_LL[LL_CODE[l]]; }
            for (var m = kl; m < RING_N; m += LPB) { p_ml[blk * RING_N + m] = PR_ML[max(m, 3u) - 3u]; }
            for (var c = kl; c < 32u; c += LPB) { p_of[blk * 32u + c] = PR_OF[c]; }
        }
    } else if (PRICE_MODE == 1u) {
        if (valid) {
            let q = b * PRICE_WORDS;
            for (var e = kl; e < 256u; e += LPB) { p_lit[blk * 256u + e] = i32(prices[q + e]); }
            for (var c = kl; c < 36u; c += LPB) { p_llc[blk * 36u + c] = i32(prices[q + 256u + c]); }
            for (var l = kl; l < 64u; l += LPB) { p_lls[blk * 64u + l] = i32(prices[q + 256u + LL_CODE[l]]); }
            for (var m = kl; m < RING_N; m += LPB) { p_ml[blk * RING_N + m] = i32(prices[q + 292u + max(m, 3u) - 3u]); }
            for (var c = kl; c < 32u; c += LPB) { p_of[blk * 32u + c] = i32(prices[q + 345u + c]); }
        }
    } else {
        // PRICE_MODE 3: opt::Prices::from_hist of the block's Hist (lit, ll, ml, of).
        for (var i = lane; i < 5u * BPW; i += WG) { atomicStore(&hsum[i], 0u); }
        workgroupBarrier();
        let q = b * HIST_WORDS;
        if (valid) {
            var sl = 0u;
            var sll = 0u;
            var sml = 0u;
            var sof = 0u;
            for (var e = kl; e < 256u; e += LPB) { sl += seen(prices[q + e]); }
            for (var c = kl; c < 36u; c += LPB) { sll += seen(prices[q + 256u + c]); }
            for (var c = kl; c < 53u; c += LPB) { sml += seen(prices[q + 292u + c]); }
            for (var c = kl; c < 32u; c += LPB) { sof += seen(prices[q + 345u + c]); }
            atomicAdd(&hsum[blk * 5u], sl);
            atomicAdd(&hsum[blk * 5u + 1u], sll);
            atomicAdd(&hsum[blk * 5u + 2u], sml);
            atomicAdd(&hsum[blk * 5u + 3u], sof);
        }
        workgroupBarrier();
        if (valid) {
            let lb = frac_weight(max(atomicLoad(&hsum[blk * 5u]), 1u));
            let llb = frac_weight(max(atomicLoad(&hsum[blk * 5u + 1u]), 1u));
            let mlb = frac_weight(max(atomicLoad(&hsum[blk * 5u + 2u]), 1u));
            let ofb = frac_weight(max(atomicLoad(&hsum[blk * 5u + 3u]), 1u));
            for (var e = kl; e < 256u; e += LPB) { p_lit[blk * 256u + e] = lit_from(seen(prices[q + e]), lb); }
            for (var c = kl; c < 36u; c += LPB) {
                p_llc[blk * 36u + c] = i32(LL_BITS[c] * 256u + llb - frac_weight(seen(prices[q + 256u + c])));
            }
            for (var l = kl; l < 64u; l += LPB) {
                let c = LL_CODE[l];
                p_lls[blk * 64u + l] = i32(LL_BITS[c] * 256u + llb - frac_weight(seen(prices[q + 256u + c])));
            }
            for (var m = kl; m < RING_N; m += LPB) {
                let c = max(m, 3u) - 3u;
                p_ml[blk * RING_N + m] = i32(ML_BITS[c] * 256u + mlb - frac_weight(seen(prices[q + 292u + c])));
            }
            for (var c = kl; c < 32u; c += LPB) {
                p_of[blk * 32u + c] = i32(c * 256u + ofb - frac_weight(seen(prices[q + 345u + c])));
            }
        }
    }
    workgroupBarrier();
    if (HIST_OUT) {
        for (var i = lane; i < 256u * BPW; i += WG) { atomicStore(&hist[i], 0u); }
        for (var i = lane; i < 36u * BPW; i += WG) { atomicStore(&o_ll[i], 0u); }
        for (var i = lane; i < 53u * BPW; i += WG) { atomicStore(&o_ml[i], 0u); }
        for (var i = lane; i < 32u * BPW; i += WG) { atomicStore(&o_of[i], 0u); }
        workgroupBarrier();
    }
}

// HIST_OUT: the pass's Hist of block b (lane of segment k), after every lane's phase 2. Each lane
// has counted its own raw sequences and literal bytes; the segment-0 lane then applies main_fixup's
// changes (the literal carry of each segment's first sequence, and the off_bases it re-encodes
// against the true reps, walking the segments in order exactly as main_fixup does), and the
// block's lanes store the Hist at prices[b * HIST_WORDS]. Every lane calls it (barriers).
fn hist_epilogue(valid: bool, b: u32, k: u32) {
    storageBarrier();
    workgroupBarrier();
    if (valid && k == 0u) {
        var reps = vec3<u32>(1u, 4u, 8u);
        var prev_end = 0u;
        for (var kk = 0u; kk < NSEG; kk += 1u) {
            let li = pb * NSEG + kk;
            let n = sg_n[li];
            let carry = kk * SEG - prev_end;
            if (n > 0u) { prev_end = sg_anchor[li]; }
            let src = b * (3u * MAX_SEQS) + kk * LOG_WORDS + 3u * (LOG_SEQS - n);
            if (n > 0u && carry > 0u) {
                let own = seqs[src];
                atomicSub(&o_ll[pb * 36u + ll_code(own)], 1u);
                atomicAdd(&o_ll[pb * 36u + ll_code(own + carry)], 1u);
            }
            var spec = select(vec3<u32>(0u), vec3<u32>(1u, 4u, 8u), kk == 0u);
            var i = 0u;
            // Terminates: i rises to n.
            loop {
                if (i >= n || all(reps == spec)) { break; }
                let r = src + 3u * i;
                let ll = seqs[r] + select(0u, carry, i == 0u);
                let spec_ob = seqs[r + 2u];
                let offset = offset_of(spec, spec_ob, ll);
                spec = applied(spec, spec_ob, ll);
                let ob = ob_for(reps, offset, ll);
                reps = applied(reps, ob, ll);
                let c0 = firstLeadingBit(spec_ob);
                let c1 = firstLeadingBit(ob);
                if (c0 != c1) {
                    atomicSub(&o_of[pb * 32u + c0], 1u);
                    atomicAdd(&o_of[pb * 32u + c1], 1u);
                }
                i += 1u;
            }
            if (i < n) { reps = sg_rep[li]; }
        }
    }
    workgroupBarrier();
    if (valid) {
        let q = b * HIST_WORDS;
        for (var e = k; e < HIST_WORDS; e += NSEG) {
            var v: u32;
            if (e < 256u) {
                v = atomicLoad(&hist[pb * 256u + e]);
            } else if (e < 292u) {
                v = atomicLoad(&o_ll[pb * 36u + e - 256u]);
            } else if (e < 345u) {
                v = atomicLoad(&o_ml[pb * 53u + e - 292u]);
            } else {
                v = atomicLoad(&o_of[pb * 32u + e - 345u]);
            }
            prices[q + e] = v;
        }
    }
}

@compute @workgroup_size(WG)
fn main_opt(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let g = wid.x * WG + lid;
    let b = g / NSEG;
    let k = g % NSEG;
    let valid = b < arrayLength(&counts) / 2u;
    lane = lid;
    pb = lid / LPB;
    prologue(valid, b);
    if (valid) { dp(b, k); }
    if (HIST_OUT) { hist_epilogue(valid, b, k); }
}

// The DP pass of segment k of block b (phases 1 and 2).
fn dp(b: u32, k: u32) {
    base = block_base(b);
    cbase = 2u * b * BLOCK_SIZE;
    tbase = cbase;
    wbase = cbase + k * SEG_WORDS;
    let s = k * SEG;
    let iend = s + SEG;
    let ilimit = iend - 8u;
    st_anchor = s;
    if (k == 0u) {
        st_ip = 1u;
        st_rep = vec3<u32>(1u, 4u, 8u);
    } else {
        st_ip = s;
        st_rep = vec3<u32>(0u);
    }
    n_seq = 0u;
    ml_sum = 0u;
    n_series = 0u;
    lbase = b * (3u * MAX_SEQS) + k * LOG_WORDS;
    let ll_inc1 = ll_price(1u) - ll_price(0u);
    let ll_p0 = ll_price(0u);

    var in_series = false;
    var sip = 0u;       // the series' ip
    var cur = 0u;
    var last_pos = 0u;
    // Terminates: every trip either advances st_ip (outside a series: by 1, or a commit moves it
    // past the series start) or cur (inside one, cur <= last_pos <= iend - sip, then a commit).
    loop {
        if (!in_series && st_ip >= ilimit) { break; }
        // The trip's position and its loads, issued before anything depends on them: the 4 bytes
        // at p (x), the byte before it, and p's candidate words (p is clamped to the segment: a
        // series may reach iend, where nothing is searched and x is unused).
        let p = select(st_ip, sip + cur, in_series);
        let pc = min(p, iend - 1u);
        let x = load_u32_at(base, pc);
        let xprev = load_byte(base, p - 1u);
        let ci = cbase + 2u * pc;
        let w0 = best[ci];
        let w1 = best[ci + 1u];

        // Part 1: what to search, if anything.
        var grep = vec3<u32>(0u);
        var gll0 = false;
        var search = false;
        var n = Node(0, vec3<u32>(0u), 0u, 0u, 0u);
        var advance = false;
        var finish = false;
        if (!in_series) {
            grep = st_rep;
            gll0 = st_ip == st_anchor;
            search = true;
        } else {
            let inr = p;
            let prev = ld(cur - 1u);
            let litlen = prev.litlen + 1u;
            let price = prev.price + lit_cost(xprev) + (ll_price(litlen) - ll_price(litlen - 1u));
            n = ld(cur);
            if (price <= n.price) {
                let pm = n;
                n = prev;
                n.litlen = litlen;
                n.price = price;
                st(cur, n);
                if (LEVEL >= 1u && pm.litlen == 0u && ll_inc1 < 0 && inr < iend) {
                    let next_lit = lit_cost(x & 0xFFu);
                    let with1 = pm.price + next_lit + ll_inc1;
                    let with_more = price + next_lit + (ll_price(litlen + 1u) - ll_price(litlen));
                    var next_price = MAXP;
                    if (cur < last_pos) { next_price = ld_price(cur + 1u); }
                    if (with1 < with_more && with1 < next_price) {
                        var q = pm;
                        q.litlen = 1u;
                        q.price = with1;
                        st(cur + 1u, q);
                        last_pos = max(last_pos, cur + 1u);
                    }
                }
            }
            if (inr < iend) { trace_put(inr, n); }
            if (inr > ilimit) {
                advance = true;
            } else if (cur == last_pos) {
                finish = true;
            } else if (LEVEL == 0u && ld_price(cur + 1u) <= n.price + 128) {
                advance = true;
            } else {
                grep = n.r;
                gll0 = n.litlen == 0u;
                search = true;
            }
        }

        // The lanes' match searches, together.
        if (search) {
            get_all_matches(p, grep, gll0, iend, x, w0, w1);
        }

        // Part 2 (no `continue`: every lane reaches the end of the trip, so the lanes stay
        // together for the next trip's get_all_matches).
        if (!in_series) {
            if (m_n == 0u) {
                st_ip += 1u;
            } else {
                let ip = st_ip;
                let litlen = ip - st_anchor;
                let n0 = Node(ll_price(litlen), st_rep, litlen, 0u, 0u);
                st(0u, n0);
                trace_put(ip, n0);
                let max_ob = m_ob[m_n - 1u];
                let max_ml = m_len[m_n - 1u];
                if (max_ml > SUFF) {
                    // large match -> immediate encoding
                    end_series(Node(0, new_rep(st_rep, max_ob, litlen == 0u), 0u, max_ml, max_ob), ip, max_ml);
                } else {
                    st(1u, Node(MAXP, vec3<u32>(0u), litlen + 1u, 0u, 0u));
                    st(2u, Node(MAXP, vec3<u32>(0u), litlen + 2u, 0u, 0u));
                    var pos = MIN_MATCH;
                    for (var mi = 0u; mi < m_n; mi += 1u) {
                        let ob = m_ob[mi];
                        let end = m_len[mi];
                        let mrep = new_rep(st_rep, ob, litlen == 0u);
                        let mp = n0.price + of_price(ob) + MATCH_FEE + ll_p0;
                        for (; pos <= end; pos += 1u) {
                            st(pos, Node(mp + p_ml[pb * RING_N + pos], mrep, 0u, pos, ob));
                        }
                    }
                    last_pos = pos - 1u;
                    sip = ip;
                    cur = 1u;
                    in_series = true;
                }
            }
        } else if (search) {
            let inr = sip + cur;
            let ll0 = gll0;
            if (m_n == 0u) {
                advance = true;
            } else {
                let max_ob = m_ob[m_n - 1u];
                let longest = m_len[m_n - 1u];
                if (longest > SUFF || cur + longest >= OPT_NUM || inr + longest >= iend) {
                    last_pos = cur + longest;
                    end_series(Node(0, new_rep(n.r, max_ob, ll0), 0u, longest, max_ob), sip, last_pos);
                    in_series = false;
                } else {
                    // set prices using matches found at position == cur (lengths downward)
                    let base_price = n.price + ll_p0;
                    var start_ml = MIN_MATCH;
                    for (var mi = 0u; mi < m_n; mi += 1u) {
                        let ob = m_ob[mi];
                        let last_ml = m_len[mi];
                        let mrep = new_rep(n.r, ob, ll0);
                        let mp = base_price + of_price(ob) + MATCH_FEE;
                        // Terminates: mlen falls to start_ml >= 3.
                        for (var mlen = last_ml; mlen >= start_ml; mlen -= 1u) {
                            let pos = cur + mlen;
                            let price = mp + p_ml[pb * RING_N + mlen];
                            if (pos > last_pos || price < ld_price(pos)) {
                                // Terminates: last_pos rises to pos.
                            while (last_pos < pos) {
                                    last_pos += 1u;
                                    fill(last_pos);
                                }
                                st(pos, Node(price, mrep, 0u, mlen, ob));
                            } else if (LEVEL == 0u) {
                                break; // early update abort
                            }
                        }
                        start_ml = last_ml + 1u;
                    }
                    advance = true;
                }
            }
        }
        if (advance) {
            cur += 1u;
            finish = cur > last_pos;
        }
        if (finish) {
            end_series(ld(last_pos), sip, last_pos);
            in_series = false;
        }
    }
    // Phase 2: the logged series' sequences, last first.
    // Terminates: i falls to 0.
    for (var i = n_series; i > 0u; i -= 1u) {
        let l = lbase + 3u * (i - 1u);
        let a = seqs[l];
        let c = seqs[l + 1u];
        emit_series(a & 0xFFFFu, c & 0xFFFFu, a >> 16u, c >> 16u, seqs[l + 2u]);
    }
    if (HIST_OUT) {
        // The literals after the segment's last match; the summary for hist_epilogue. The
        // candidate words stay (no trailer in `best`).
        hist_lits(st_anchor, iend);
        sg_n[lane] = n_seq;
        sg_anchor[lane] = st_anchor;
        sg_rep[lane] = st_rep;
        return;
    }
    best[wbase + SEG_META] = n_seq;
    best[wbase + SEG_META + 1u] = st_anchor;
    best[wbase + SEG_META + 2u] = ml_sum;
    best[wbase + SEG_META + 3u] = st_rep.x;
    best[wbase + SEG_META + 4u] = st_rep.y;
    best[wbase + SEG_META + 5u] = st_rep.z;
}
