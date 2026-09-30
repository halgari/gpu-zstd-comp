// K3opt: one DP pass of the M5 optimal parse (presets opt14 / opt16), == gzc_core::opt::dp_pass_with
// on Engine::Ring (the normative GPU spec, documented on opt::Engine), statement by statement.
// Two entry points, dispatched back to back:
//
// main_opt (one invocation per 4 KiB segment, NSEG per block, WG lanes = BPW blocks per
//   workgroup): a prologue builds the block's static price tables in workgroup memory (PRICE_MODE
//   0: zstd's first-block statistics, opt::Prices::block_init, the literal histogram counted by
//   the block's lanes; PRICE_MODE 1: the tables in `prices`, opt::Prices verbatim, 377 words per
//   block: lit[256] ll[36] ml[53] of[32]). Then each lane runs opt::Dp::segment_ring over its
//   segment: segment 0 from ip 1 with INITIAL_REPS, segment k > 0 from ip = anchor = k*SEG with
//   reps [0, 0, 0]; iend = the segment end, ilimit = iend - 8.
//   - The lane loop is flattened (one trip per series start probe or series position) so that the
//     lanes' get_all_matches calls run together.
//   - The DP nodes live in a ring of RING_N = sufficient_len + 1 slots per lane (position pos of
//     the series in slot pos % RING_N), in workgroup memory (RING_WG) or in private memory (the
//     fallback when the adapter's workgroup storage is too small); both declared by the host.
//   - `trace` (the dead K1 pred buffer, 2 words per position) gets (mlen | litlen << 8, offBase)
//     of every node when it becomes final; the backward trace reads it, never the ring. The entry
//     at iend (a node at the segment end) is never read and not written (it would be the next
//     segment's first position).
//   - Sequences go into the segment's own words of `best` (the candidate words, 2 per position):
//     sequence i at best[wbase + 3*i ..], trailer (n_seq, final anchor, sum of match_len, final
//     reps) at best[wbase + SEG_META ..], as k3_seg.wgsl, for k3_fixup.wgsl's main_fixup. Each
//     sequence covers at least 3 bytes, so once i sequences are stored the anchor is at least
//     3*i past the segment start and every later candidate read (at a position >= anchor, words
//     >= 6*i) is above the 3*i words written.
//   - Stored off_bases are canonical (seq::off_base_for) under the segment's own canonical
//     history, as main_fixup needs: the DP's offBase is decoded to an offset under the DP's own
//     history (the oracle's seq_reps), then re-encoded.
// main_fixup (k3_fixup.wgsl): concatenation, literal carry and off_base re-encoding against the
//   block's true reps (== lazy::encode_raw), one workgroup per block.
//
// Candidate words (reference::CandWords, K2opt): best[2*(b*BLOCK_SIZE + p)] = offA | lenA << 16 |
// lenB << 24, best[.. + 1] = offB; lengths capped at SEARCH_CAP (a stored SEARCH_CAP is extended).
//
// Consts injected by the host: MIN_MATCH, SEARCH_CAP, MAX_SEQS, SEG_LOG2, WG, SUFF
// (sufficient_len), LEVEL (optLevel 0 | 2), PRICE_MODE, BI_LL / BI_ML / BI_OF (block-init
// LL / ML / OF prices), and the ring declarations (ring_p/a/b/c, rix).

const SEG: u32 = 1u << SEG_LOG2;
const NSEG: u32 = BLOCK_SIZE >> SEG_LOG2;
const SEG_WORDS: u32 = 2u * SEG;
const SEG_META: u32 = SEG_WORDS - 6u;
const BPW: u32 = WG / NSEG;
const RING_N: u32 = SUFF + 1u;
const MAXP: i32 = 1 << 30;
const OPT_NUM: u32 = 4096u;
const MATCH_FEE: i32 = 51;
const PRICE_WORDS: u32 = 377u;
const_assert BLOCK_SIZE <= 65536u;
const_assert WG % NSEG == 0u;
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
@group(0) @binding(5) var<storage, read> prices: array<u32>;

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
var<workgroup> hist: array<atomic<u32>, 256u * BPW>;
var<workgroup> hsum: array<atomic<u32>, BPW>;

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
var<private> dreps: vec3<u32>;
var<private> creps: vec3<u32>;
var<private> n_seq: u32;
var<private> ml_sum: u32;

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

// Literal price of the byte at block position p.
fn lit_price(p: u32) -> i32 {
    return p_lit[pb * 256u + load_byte(base, p)];
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

// The offset offBase `ob` stands for (seq::apply_off_base's return value).
fn rep_offset(r: vec3<u32>, ob: u32, ll: u32) -> u32 {
    if (ob > 3u) { return ob - 3u; }
    switch (ob - 1u + select(0u, 1u, ll == 0u)) {
        case 0u: { return r.x; }
        case 1u: { return r.y; }
        case 2u: { return r.z; }
        default: { return r.x - 1u; }
    }
}

// seq::off_base_for on a history value.
fn rep_ob(r: vec3<u32>, offset: u32, ll: u32) -> u32 {
    if (ll > 0u) {
        if (offset == r.x) { return 1u; }
        if (offset == r.y) { return 2u; }
        if (offset == r.z) { return 3u; }
    } else {
        if (offset == r.y) { return 1u; }
        if (offset == r.z) { return 2u; }
        if (r.x > 1u && offset == r.x - 1u) { return 3u; }
    }
    return offset + 3u;
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

// opt::Dp::get_all_matches: records at p for reps `r` and `ll0`, lengths clamped to iend - p.
fn get_all_matches(p: u32, r: vec3<u32>, ll0: bool, iend: u32) {
    m_n = 0u;
    let lim = iend - p;
    var bestl = MIN_MATCH - 1u;
    let l0 = select(0u, 1u, ll0);
    for (var rc = l0; rc < l0 + 3u; rc += 1u) {
        var ro = r.x - 1u;
        if (rc == 0u) { ro = r.x; } else if (rc == 1u) { ro = r.y; } else if (rc == 2u) { ro = r.z; }
        if (ro - 1u < p) {
            let rl = match_len(base, p, p - ro, lim);
            if (rl >= MIN_MATCH && rl > bestl) {
                bestl = rl;
                push_match(rc - l0 + 1u, rl);
                if (rl > SUFF || rl == lim) { return; }
            }
        }
    }
    let ci = cbase + 2u * p;
    let w0 = best[ci];
    let w1 = best[ci + 1u];
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

fn trace_put(p: u32, n: Node) {
    let t = tbase + 2u * p;
    trace[t] = n.mlen | (n.litlen << 8u);
    trace[t + 1u] = n.ob;
}

// opt::Dp::commit_ring: `last` ends the series started at `sip`, at series position `last_pos`.
fn commit(last: Node, sip: u32, last_pos: u32) {
    if (last.mlen == 0u) {
        st_ip = sip + last_pos;
        return;
    }
    st_rep = last.r;
    // Backward trace: the series' sequences, last first, into the output slots.
    let w = wbase + 3u * n_seq;
    var sp = last_pos - last.mlen - last.litlen;
    var mlen = last.mlen;
    var ob = last.ob;
    var cnt = 0u;
    loop {
        let t = tbase + 2u * (sip + sp);
        let t0 = trace[t];
        let nm = t0 & 0xFFu;
        let nl = t0 >> 8u;
        let o = w + 3u * cnt;
        best[o] = nl;
        best[o + 1u] = mlen;
        best[o + 2u] = ob;
        cnt += 1u;
        // Terminates: sp falls by nl + nm >= 3 per step (a node ending a match has mlen >= 3);
        // the series start (sp = 0) has nm = 0.
        if (nm == 0u || sp < nl + nm) { break; }
        mlen = nm;
        ob = trace[t + 1u];
        sp -= nl + nm;
    }
    // Into order.
    for (var i = 0u; i < cnt / 2u; i += 1u) {
        let x = w + 3u * i;
        let y = w + 3u * (cnt - 1u - i);
        for (var j = 0u; j < 3u; j += 1u) {
            let v = best[x + j];
            best[x + j] = best[y + j];
            best[y + j] = v;
        }
    }
    // Offsets under the DP's history, stored canonical under the segment's own.
    for (var i = 0u; i < cnt; i += 1u) {
        let o = w + 3u * i;
        let ll = best[o];
        let ml = best[o + 1u];
        let dob = best[o + 2u];
        let off = rep_offset(dreps, dob, ll);
        dreps = rep_after(dreps, dob, ll);
        let cob = rep_ob(creps, off, ll);
        creps = rep_after(creps, cob, ll);
        best[o + 2u] = cob;
        st_anchor += ll + ml;
        ml_sum += ml;
    }
    n_seq += cnt;
    st_ip = st_anchor;
    if (last.litlen > 0u) {
        st_ip = st_anchor + last.litlen;
    }
}

// The workgroup's price tables (see the header). Every lane calls it (it has barriers).
fn prologue(valid: bool, b: u32, k: u32) {
    let blk = lane / NSEG;
    if (PRICE_MODE == 0u) {
        for (var i = lane; i < 256u * BPW; i += WG) { atomicStore(&hist[i], 0u); }
        if (lane < BPW) { atomicStore(&hsum[lane], 0u); }
        workgroupBarrier();
        if (valid) {
            let w0 = block_base(b) + k * (SEG / 4u);
            for (var w = 0u; w < SEG / 4u; w += 1u) {
                let v = data[w0 + w];
                atomicAdd(&hist[blk * 256u + (v & 0xFFu)], 1u);
                atomicAdd(&hist[blk * 256u + ((v >> 8u) & 0xFFu)], 1u);
                atomicAdd(&hist[blk * 256u + ((v >> 16u) & 0xFFu)], 1u);
                atomicAdd(&hist[blk * 256u + (v >> 24u)], 1u);
            }
        }
        workgroupBarrier();
        for (var e = k; e < 256u; e += NSEG) {
            let c = atomicLoad(&hist[blk * 256u + e]);
            let f = select(0u, 1u, c > 0u) + (c >> 8u);
            atomicStore(&hist[blk * 256u + e], f);
            atomicAdd(&hsum[blk], f);
        }
        workgroupBarrier();
        let lb = frac_weight(max(atomicLoad(&hsum[blk]), 1u));
        for (var e = k; e < 256u; e += NSEG) {
            let f = atomicLoad(&hist[blk * 256u + e]);
            p_lit[blk * 256u + e] = i32(lb - min(frac_weight(f), lb - 256u));
        }
        for (var c = k; c < 36u; c += NSEG) { p_llc[blk * 36u + c] = BI_LL[c]; }
        for (var l = k; l < 64u; l += NSEG) { p_lls[blk * 64u + l] = BI_LL[LL_CODE[l]]; }
        for (var m = k; m < RING_N; m += NSEG) { p_ml[blk * RING_N + m] = BI_ML[max(m, 3u) - 3u]; }
        for (var c = k; c < 32u; c += NSEG) { p_of[blk * 32u + c] = BI_OF[c]; }
    } else if (valid) {
        let q = b * PRICE_WORDS;
        for (var e = k; e < 256u; e += NSEG) { p_lit[blk * 256u + e] = i32(prices[q + e]); }
        for (var c = k; c < 36u; c += NSEG) { p_llc[blk * 36u + c] = i32(prices[q + 256u + c]); }
        for (var l = k; l < 64u; l += NSEG) { p_lls[blk * 64u + l] = i32(prices[q + 256u + LL_CODE[l]]); }
        for (var m = k; m < RING_N; m += NSEG) { p_ml[blk * RING_N + m] = i32(prices[q + 292u + max(m, 3u) - 3u]); }
        for (var c = k; c < 32u; c += NSEG) { p_of[blk * 32u + c] = i32(prices[q + 345u + c]); }
    }
    workgroupBarrier();
}

@compute @workgroup_size(WG)
fn main_opt(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let g = wid.x * WG + lid;
    let b = g / NSEG;
    let k = g % NSEG;
    let valid = b < arrayLength(&counts) / 2u;
    lane = lid;
    pb = lid / NSEG;
    prologue(valid, b, k);
    if (!valid) { return; }

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
    dreps = st_rep;
    creps = st_rep;
    n_seq = 0u;
    ml_sum = 0u;
    let ll_inc1 = ll_price(1u) - ll_price(0u);
    let ll_p0 = ll_price(0u);

    var in_series = false;
    var sip = 0u;       // the series' ip
    var cur = 0u;
    var last_pos = 0u;
    // Terminates: every trip either advances st_ip (outside a series: by 1, or a commit moves it
    // past the series start) or cur (inside one, cur <= last_pos <= iend - sip, then a commit).
    loop {
        // Part 1: what to search, if anything.
        var gp = 0u;
        var grep = vec3<u32>(0u);
        var gll0 = false;
        var search = false;
        var n = Node(0, vec3<u32>(0u), 0u, 0u, 0u);
        var advance = false;
        var finish = false;
        if (!in_series) {
            if (st_ip >= ilimit) { break; }
            gp = st_ip;
            grep = st_rep;
            gll0 = st_ip == st_anchor;
            search = true;
        } else {
            let inr = sip + cur;
            let prev = ld(cur - 1u);
            let litlen = prev.litlen + 1u;
            let price = prev.price + lit_price(inr - 1u) + (ll_price(litlen) - ll_price(litlen - 1u));
            n = ld(cur);
            if (price <= n.price) {
                let pm = n;
                n = prev;
                n.litlen = litlen;
                n.price = price;
                st(cur, n);
                if (LEVEL >= 1u && pm.litlen == 0u && ll_inc1 < 0 && inr < iend) {
                    let next_lit = lit_price(inr);
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
                gp = inr;
                grep = n.r;
                gll0 = n.litlen == 0u;
                search = true;
            }
        }

        // The lanes' match searches, together.
        if (search) {
            get_all_matches(gp, grep, gll0, iend);
        }

        // Part 2.
        if (!in_series) {
            if (m_n == 0u) {
                st_ip += 1u;
                continue;
            }
            let ip = st_ip;
            let litlen = ip - st_anchor;
            let n0 = Node(ll_price(litlen), st_rep, litlen, 0u, 0u);
            st(0u, n0);
            trace_put(ip, n0);
            let max_ob = m_ob[m_n - 1u];
            let max_ml = m_len[m_n - 1u];
            if (max_ml > SUFF) {
                // large match -> immediate encoding
                commit(Node(0, new_rep(st_rep, max_ob, litlen == 0u), 0u, max_ml, max_ob), ip, max_ml);
                continue;
            }
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
            continue;
        }
        if (search) {
            let inr = sip + cur;
            let ll0 = gll0;
            if (m_n == 0u) {
                advance = true;
            } else {
                let max_ob = m_ob[m_n - 1u];
                let longest = m_len[m_n - 1u];
                if (longest > SUFF || cur + longest >= OPT_NUM || inr + longest >= iend) {
                    last_pos = cur + longest;
                    commit(Node(0, new_rep(n.r, max_ob, ll0), 0u, longest, max_ob), sip, last_pos);
                    in_series = false;
                    continue;
                }
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
        if (advance) {
            cur += 1u;
            finish = cur > last_pos;
        }
        if (finish) {
            commit(ld(last_pos), sip, last_pos);
            in_series = false;
        }
    }
    best[wbase + SEG_META] = n_seq;
    best[wbase + SEG_META + 1u] = st_anchor;
    best[wbase + SEG_META + 2u] = ml_sum;
    best[wbase + SEG_META + 3u] = creps.x;
    best[wbase + SEG_META + 4u] = creps.y;
    best[wbase + SEG_META + 5u] = creps.z;
}
