// K3drop (M6 B3): the drop pass of the optimal parse (gzc_core::opt::drop_pass, OptParams::
// drop_max_len), after K3opt's final pass and its fix-up (main_fixup, k3_fixup.wgsl), as in the
// oracle: opt::parse applies drop_pass to the fixed-up output of the final DP pass. One workgroup
// of DROP_WG lanes per block (main_drop):
//
// 0. Lane 0 walks the segments as main_fixup does (the trailers K3opt left in `best`, and the
//    fixed-up off_bases in `seqs`): each segment's sequence count, first index in `seqs`, literal
//    carry, and the block's true decoder reps before its first sequence (R_k).
// 1. The output's histogram (opt::Hist::of_output): every lane counts LL / ML / OF codes of a
//    stride of the sequences; the literal bytes are counted per segment (LPS lanes per segment,
//    each every LPS-th byte of the segment's literal runs).
// 2. The drop prices (opt::Prices::from_hist of that histogram), or with DROP_PRICES_IN (a test
//    hook) the opt::Prices tables at prices[b * 377], verbatim.
// 3. One lane per 4 KiB segment k (opt::drop_decisions, segment-local): its sequences are those
//    whose match starts in segment k, K3opt's raw sequences of segment k. The lane starts from
//    reps r = R_k and carry 0 (nothing from the previous lane) and walks them in order, decoding
//    each one's real offset with the true reps (rin, which follows the input), and decides:
//    ll = ll_j + carry, ob = off_base_for(off_j, ll, r); a candidate (ml <= DROP_MAX, ob > 3, a
//    successor j + 1 in the block, possibly in a later segment, with its input values) is dropped
//    when lits(s_j .. s_j + ml) + price(ll + ml + ll', ml', ob'(r)) < price(ll, ml, ob) +
//    price(ll', ml', ob'(r after ob)) (strict), which carries ll + ml and leaves r; a kept one
//    updates r and stores (ll, ml, ob), its off_base under the lane's own history r, in the
//    segment's words of `best` (K3opt's raw sequences, read in step 0 only; in order), with the
//    segment's kept count, first match start, last match end, match bytes and final r.
// 4. opt::apply_drops: the kept sequences are concatenated into `seqs` (a segment's first kept
//    sequence takes every literal since the previous kept match, which a dropped tail of earlier
//    segments lengthens), then lane 0 re-encodes the off_bases against the true reps, segment by
//    segment, as main_fixup does: from the true incoming history until it equals the lane's own
//    (started at R_k, followed with the lane's lit_lens) with an unchanged lit_len; from there on
//    the lane's off_bases are the true ones and its final r is the true history.
//    counts = (n_seq, n_lit).
//
// Consts injected by the host: MIN_MATCH, SEARCH_CAP, ... (params), MAX_SEQS, SEG_LOG2,
// DROP_MAX (OptParams::drop_max_len), DROP_PRICES_IN, LL_BITS / ML_BITS / ML_CODE (codes.rs).
// k3_fixup.wgsl is appended for its rep helpers (its main_fixup entry is not built here).

const SEG: u32 = 1u << SEG_LOG2;
const NSEG: u32 = BLOCK_SIZE >> SEG_LOG2;
const SEG_WORDS: u32 = 2u * SEG;
const SEG_META: u32 = SEG_WORDS - 6u;
// K3opt stores a segment's raw sequences last first (k3_opt.wgsl); the kept ones are in order.
const RAW_REVERSED: bool = true;
const DROP_WG: u32 = 64u;
// Lanes per segment counting literal bytes.
const LPS: u32 = DROP_WG / NSEG;
const_assert DROP_WG % NSEG == 0u;
const_assert NSEG <= DROP_WG;
const MATCH_FEE: i32 = 51;
const HIST_WORDS: u32 = 377u;
// opt::Hist / opt::Prices layout: lit[256] ll[36] ml[53] of[32].
const H_LL: u32 = 256u;
const H_ML: u32 = 292u;
const H_OF: u32 = 345u;

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> best: array<u32>;
@group(0) @binding(2) var<storage, read_write> seqs: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
// DROP_PRICES_IN only: opt::Prices per block (HIST_WORDS words).
@group(0) @binding(4) var<storage, read> prices: array<u32>;

// zstd LL_Code for lit_len < 64 (codes::ll_code).
const LL_CODE: array<u32, 64> = array<u32, 64>(
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u,
    16u, 16u, 17u, 17u, 18u, 18u, 19u, 19u, 20u, 20u, 20u, 20u, 21u, 21u, 21u, 21u,
    22u, 22u, 22u, 22u, 22u, 22u, 22u, 22u, 23u, 23u, 23u, 23u, 23u, 23u, 23u, 23u,
    24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u);

fn ll_code(ll: u32) -> u32 {
    if (ll > 63u) { return firstLeadingBit(ll) + 19u; }
    return LL_CODE[ll];
}
fn ml_code(ml: u32) -> u32 {
    let b = ml - 3u;
    if (b > 127u) { return firstLeadingBit(b) + 36u; }
    return ML_CODE[b];
}

// opt::frac_weight
fn frac_weight(raw: u32) -> u32 {
    let stat = raw + 1u;
    let hb = firstLeadingBit(stat);
    return hb * 256u + ((stat << 8u) >> hb);
}

var<workgroup> d_hist: array<atomic<u32>, HIST_WORDS>;
// The from_hist table sums (lit, ll, ml, of).
var<workgroup> d_sum: array<atomic<u32>, 4>;
// The drop prices, opt::Prices layout.
var<workgroup> d_price: array<i32, HIST_WORDS>;
// Per segment: K3opt's sequences (count, first index in seqs, literal carry into the first) and
// the true reps before the first (R_k).
var<workgroup> d_n: array<u32, NSEG>;
var<workgroup> d_first: array<u32, NSEG>;
var<workgroup> d_carry: array<u32, NSEG>;
var<workgroup> d_r0: array<vec3<u32>, NSEG>;
// Per segment after step 3: kept count, the first kept match's start, the last kept match's end,
// kept match bytes, the lane's final reps; step 4: first output index and the first kept
// sequence's true lit_len.
var<workgroup> d_kn: array<u32, NSEG>;
var<workgroup> d_kstart: array<u32, NSEG>;
var<workgroup> d_kend: array<u32, NSEG>;
var<workgroup> d_kml: array<u32, NSEG>;
var<workgroup> d_kr: array<vec3<u32>, NSEG>;
var<workgroup> d_kfirst: array<u32, NSEG>;
var<workgroup> d_kll: array<u32, NSEG>;
var<workgroup> d_total: u32;

// opt::drop_decisions' price(ll, ml, ob): ll_price + match_price.
fn seq_price(ll: u32, ml: u32, ob: u32) -> i32 {
    return d_price[H_LL + ll_code(ll)] + d_price[H_ML + ml_code(ml)] + d_price[H_OF + firstLeadingBit(ob)] + MATCH_FEE;
}

// Counts the literal bytes at [a, e) of the block at word base `db` whose position is sub mod LPS.
fn count_lits(db: u32, a: u32, e: u32, sub: u32) {
    // Terminates: q rises by LPS to e.
    for (var q = a + (sub + LPS - a % LPS) % LPS; q < e; q += LPS) {
        atomicAdd(&d_hist[load_byte(db, q)], 1u);
    }
}

@compute @workgroup_size(DROP_WG)
fn main_drop(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let b = wid.x;
    // Uniform per workgroup, so the barriers below stay in uniform control flow.
    if (b >= arrayLength(&counts) / 2u) { return; }
    let bbase = b * NSEG * SEG_WORDS;
    let sbase = b * MAX_SEQS * 3u;
    let db = block_base(b);
    let init = vec3<u32>(1u, 4u, 8u);

    // Step 0.
    for (var i = t; i < HIST_WORDS; i += DROP_WG) { atomicStore(&d_hist[i], 0u); }
    if (t < 4u) { atomicStore(&d_sum[t], 0u); }
    if (t == 0u) {
        var first = 0u;
        var prev_end = 0u;
        var reps = init;
        for (var k = 0u; k < NSEG; k += 1u) {
            let src = bbase + k * SEG_WORDS;
            let n = min(best[src + SEG_META], SEG / 3u);
            let carry = k * SEG - prev_end;
            d_n[k] = n;
            d_first[k] = first;
            d_carry[k] = carry;
            d_r0[k] = reps;
            // main_fixup's walk: the true reps follow the fixed-up off_bases until they equal the
            // segment parse's own (spec); from there its trailer's reps are the true ones.
            var spec = select(vec3<u32>(0u), init, k == 0u);
            var i = 0u;
            // Terminates: i rises to n.
            loop {
                if (i >= n || all(reps == spec)) { break; }
                let r = raw_at(src, n, i);
                let ll = best[r] + select(0u, carry, i == 0u);
                spec = applied(spec, best[r + 2u], ll);
                reps = applied(reps, seqs[sbase + 3u * (first + i) + 2u], ll);
                i += 1u;
            }
            if (i < n) { reps = vec3<u32>(best[src + SEG_META + 3u], best[src + SEG_META + 4u], best[src + SEG_META + 5u]); }
            if (n > 0u) { prev_end = best[src + SEG_META + 1u]; }
            first += n;
        }
        d_total = first;
    }
    workgroupBarrier();

    // Step 1.
    let n_all = d_total;
    if (!DROP_PRICES_IN) {
        // Terminates: i rises by DROP_WG to n_all.
        for (var i = t; i < n_all; i += DROP_WG) {
            let q = sbase + 3u * i;
            atomicAdd(&d_hist[H_LL + ll_code(seqs[q])], 1u);
            atomicAdd(&d_hist[H_ML + ml_code(seqs[q + 1u])], 1u);
            atomicAdd(&d_hist[H_OF + firstLeadingBit(seqs[q + 2u])], 1u);
        }
        let k = t % NSEG;
        let sub = t / NSEG;
        let n = d_n[k];
        let f = d_first[k];
        var pos = k * SEG;
        // Terminates: i rises to n.
        for (var i = 0u; i < n; i += 1u) {
            let q = sbase + 3u * (f + i);
            let ll = seqs[q] - select(0u, d_carry[k], i == 0u);
            count_lits(db, pos, pos + ll, sub);
            pos += ll + seqs[q + 1u];
        }
        count_lits(db, pos, (k + 1u) * SEG, sub);
    }
    workgroupBarrier();

    // Step 2.
    if (DROP_PRICES_IN) {
        for (var i = t; i < HIST_WORDS; i += DROP_WG) { d_price[i] = i32(prices[b * HIST_WORDS + i]); }
    } else {
        var s = vec4<u32>(0u);
        for (var i = t; i < HIST_WORDS; i += DROP_WG) {
            let c = atomicLoad(&d_hist[i]);
            let f = c + select(0u, 1u, c > 0u);
            if (i < H_LL) { s.x += f; } else if (i < H_ML) { s.y += f; } else if (i < H_OF) { s.z += f; } else { s.w += f; }
        }
        atomicAdd(&d_sum[0], s.x);
        atomicAdd(&d_sum[1], s.y);
        atomicAdd(&d_sum[2], s.z);
        atomicAdd(&d_sum[3], s.w);
        workgroupBarrier();
        let lb = frac_weight(max(atomicLoad(&d_sum[0]), 1u));
        let llb = frac_weight(max(atomicLoad(&d_sum[1]), 1u));
        let mlb = frac_weight(max(atomicLoad(&d_sum[2]), 1u));
        let ofb = frac_weight(max(atomicLoad(&d_sum[3]), 1u));
        for (var i = t; i < HIST_WORDS; i += DROP_WG) {
            let c = atomicLoad(&d_hist[i]);
            let w = frac_weight(c + select(0u, 1u, c > 0u));
            var p: u32;
            if (i < H_LL) {
                p = lb - min(w, lb - 256u);
            } else if (i < H_ML) {
                p = LL_BITS[i - H_LL] * 256u + llb - w;
            } else if (i < H_OF) {
                p = ML_BITS[i - H_ML] * 256u + mlb - w;
            } else {
                p = (i - H_OF) * 256u + ofb - w;
            }
            d_price[i] = i32(p);
        }
    }
    workgroupBarrier();

    // Step 3: lane k decides segment k.
    if (t < NSEG) {
        let k = t;
        let n = d_n[k];
        let f = d_first[k];
        let dst = bbase + k * SEG_WORDS;
        var r = d_r0[k];
        var rin = r;
        var carry = 0u;
        var kept = 0u;
        var kstart = 0u;
        var kend = 0u;
        var kml = 0u;
        // The match start of the current sequence, the end of the previous one's match.
        var prev_end = k * SEG - d_carry[k];
        // Terminates: i rises to n.
        for (var i = 0u; i < n; i += 1u) {
            let j = f + i;
            let q = sbase + 3u * j;
            let ll_j = seqs[q];
            let ml = seqs[q + 1u];
            let ob_in = seqs[q + 2u];
            let s = prev_end + ll_j;
            prev_end = s + ml;
            let off = offset_of(rin, ob_in, ll_j);
            rin = applied(rin, ob_in, ll_j);
            let ll = ll_j + carry;
            let ob = ob_for(r, off, ll);
            var dropped = false;
            if (ml <= DROP_MAX && ob > 3u && j + 1u < n_all) {
                let nll = seqs[q + 3u];
                let nml = seqs[q + 4u];
                // rin: the true reps before j + 1.
                let noff = offset_of(rin, seqs[q + 5u], nll);
                let after = applied(r, ob, ll);
                let keep = seq_price(ll, ml, ob) + seq_price(nll, nml, ob_for(after, noff, nll));
                var lits = 0;
                // Terminates: e rises to s + ml.
                for (var e = s; e < s + ml; e += 1u) { lits += d_price[load_byte(db, e)]; }
                let mll = ll + ml + nll;
                let drop = lits + seq_price(mll, nml, ob_for(r, noff, mll));
                dropped = drop < keep;
            }
            if (dropped) {
                carry = ll + ml;
            } else {
                let o = dst + 3u * kept;
                best[o] = ll;
                best[o + 1u] = ml;
                best[o + 2u] = ob;
                if (kept == 0u) { kstart = s; }
                kept += 1u;
                kend = s + ml;
                kml += ml;
                r = applied(r, ob, ll);
                carry = 0u;
            }
        }
        d_kn[k] = kept;
        d_kstart[k] = kstart;
        d_kend[k] = kend;
        d_kml[k] = kml;
        d_kr[k] = r;
    }
    storageBarrier();
    workgroupBarrier();

    // Step 4.
    if (t == 0u) {
        var first = 0u;
        var prev_end = 0u;
        var ml = 0u;
        for (var k = 0u; k < NSEG; k += 1u) {
            d_kfirst[k] = first;
            if (d_kn[k] > 0u) {
                d_kll[k] = d_kstart[k] - prev_end;
                prev_end = d_kend[k];
            }
            first += d_kn[k];
            ml += d_kml[k];
        }
        counts[b * 2u] = first;
        counts[b * 2u + 1u] = BLOCK_SIZE - ml;
    }
    workgroupBarrier();
    for (var k = 0u; k < NSEG; k += 1u) {
        let n = d_kn[k];
        let src = bbase + k * SEG_WORDS;
        let dst = sbase + 3u * d_kfirst[k];
        // Terminates: i rises by DROP_WG to n.
        for (var i = t; i < n; i += DROP_WG) {
            seqs[dst + 3u * i] = select(best[src + 3u * i], d_kll[k], i == 0u);
            seqs[dst + 3u * i + 1u] = best[src + 3u * i + 1u];
            seqs[dst + 3u * i + 2u] = best[src + 3u * i + 2u];
        }
    }
    storageBarrier();
    workgroupBarrier();
    if (t == 0u) {
        var reps = init;
        for (var k = 0u; k < NSEG; k += 1u) {
            let n = d_kn[k];
            let src = bbase + k * SEG_WORDS;
            let dst = sbase + 3u * d_kfirst[k];
            var spec = d_r0[k];
            var i = 0u;
            // Terminates: i rises to n.
            loop {
                if (i >= n) { break; }
                let lls = best[src + 3u * i];
                let llt = select(lls, d_kll[k], i == 0u);
                if (all(reps == spec) && llt == lls) { break; }
                let sob = best[src + 3u * i + 2u];
                let off = offset_of(spec, sob, lls);
                spec = applied(spec, sob, lls);
                let ob = ob_for(reps, off, llt);
                reps = applied(reps, ob, llt);
                seqs[dst + 3u * i + 2u] = ob;
                i += 1u;
            }
            if (i < n) { reps = d_kr[k]; }
        }
    }
}
