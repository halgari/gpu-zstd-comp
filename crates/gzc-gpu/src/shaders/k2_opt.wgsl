// K2opt: the optimal parse's two candidate records per position (== gzc_core::reference::find_cands,
// m5-opt-design §2.1). Appended to k2_best.wgsl (whose bindings, `funnel` and `match_len_capped` it
// uses); entry point main_opt, built only for Opt3 params (K1 wrote pred[b*PRED_PER_BLOCK ..] =
// the h4 chain, with pred_fp fingerprints, pred[b*PRED_PER_BLOCK + BLOCK_SIZE ..] = the h3 chain,
// with pred_fp3 fingerprints, and with M6 sparse long chains (N_SPARSE > 0, `chains::layout_wgsl`)
// chain k compactly at pred[b*PRED_PER_BLOCK + SP_OFF{k} + p / SP_S{k}], with pred_fp
// fingerprints).
// Dispatch (BLOCK_SIZE / 256, n_blocks, 1): one thread per (position, block). Output (the `best`
// binding, 2 words per position): best[2*(b*BLOCK_SIZE + p) ..] = (offA | lenA << 16 | lenB << 24,
// offB | DEAD_BIT at a dead position, below): all zero without a record or for p >= PARSE_END,
// except [0, DEAD_BIT] at a dead position.
//
// The walk: the h4 chain DEPTH deep and the h3 chain H3_DEPTH deep, plus each sparse long chain k
// SP_D{k} deep when p is on its slots (p % SP_S{k} == 0; its head at p is PRED_NONE past its hashed
// slots), merged nearest first (the largest live head; a position on several chains is visited
// once and advances all of them). A visited q
// with capped length c (match_len_capped, cap SEARCH_CAP) is a record when c > best (best starts
// at 2); A = the first record, B = the last. The walk stops at a record with c == max.
// Fingerprint skips, byte-identical: a q is compared only if an upper bound on its length, taken
// from the fingerprints in the pred words of q the walk loads anyway, exceeds best (otherwise c <=
// best and q is no record). Bounds (both valid for any bytes, so a q on both chains takes the
// smaller):
// - h4 word (pred_fp: 7-bit hash of bytes 0..4, byte 4): hash field differs -> the first 4 bytes
//   differ -> c <= 2; byte field differs -> c <= 4. The c <= 2 is where the spec's fingerprint
//   caveat applies (an h4-chain entry whose first 4 bytes differ may still share 3 bytes, unless
//   the first 3 bytes differ as well), and they always do here: q is on p's h4 chain, so
//   hash_width(q, 4) == hash_width(p, 4) (all 16 bits, hash_bits is 16 for opt), and that hash,
//   mix(lo, 0) = (lo * 0x9E3779B1 * 0xC2B2AE3D) >> 16, is injective in byte 3 for fixed bytes
//   0..3: byte 3 enters lo * K (K odd) only as (byte3 * K) << 24, a bijection of the top 8 bits,
//   which the hash keeps. So equal h4 hashes with differing 4 bytes differ in the first 3 bytes
//   (tests/cands.rs checks this property exhaustively over byte 3);
// - h3 word (pred_fp3: 7-bit hash of bytes 0..3, byte 3): hash field differs -> c <= 2 (always
//   skipped); byte field differs -> c <= 3;
// - sparse chain word (pred_fp, as h4's, but nothing is known about the first 3 bytes of a q on a
//   long chain): hash field differs -> c <= 3; byte field differs -> c <= 4.
// A skipped q still counts toward its chain's depth, and never has c == max (>= 8), so the walk
// and its early stop are unchanged.
// Indices: pred words hold positions < HASHED_POSITIONS or PRED_NONE (K1's output, run before
// this in the same submission), so pred[pb + q] and the byte loads stay in the block.
// Dead positions (M6 A3, reference::find_cands): p is dead when the walk found no record and its
// h3 head reached PRED_NONE (the whole h3 chain visited). Every visited q then had c <= 2 (a
// fingerprint skip only drops a q with c <= 2 <= best), and every earlier position with p's
// first 3 bytes has p's hash3 key, so it is on p's h3 chain: there is none. Its second word gets
// DEAD_BIT (offB < 2^16 leaves the high half free).
const DEAD_BIT: u32 = 0x10000u;

// Candidate output index of (b, p).
fn cand_index(b: u32, p: u32) -> u32 { return 2u * (b * BLOCK_SIZE + p); }

@compute @workgroup_size(256)
fn main_opt(@builtin(global_invocation_id) gid: vec3<u32>) {
    var p = gid.x;
    if (N_SPARSE > 0u) {
        // Only positions p % 4 == 0 walk the sparse chains (strides 4 and 8), much longer walks:
        // the first 64 invocations of each group of 256 take its 64 positions p % 4 == 0, the
        // other 192 the rest, so the long walks share subgroups instead of stalling every
        // subgroup on its quarter of them (a bijection of the group's positions).
        let l = gid.x & 255u;
        let g = gid.x - l;
        if (l < 64u) {
            p = g + 4u * l;
        } else {
            let r = l - 64u;
            p = g + 4u * (r / 3u) + 1u + r % 3u;
        }
    }
    let b = gid.y;
    let o = cand_index(b, p);
    if (p >= PARSE_END) {
        best[o] = 0u;
        best[o + 1u] = 0u;
        return;
    }
    let base = block_base(b);
    let mx = min(BLOCK_SIZE - p, SEARCH_CAP);
    let pw = base + (p >> 2u);
    let sp = (p & 3u) * 8u;
    let p0 = load_u32_at(base, p);
    let p1 = load_u32_at(base, p + 4u);
    let fp4 = pred_fp(p0, p1);
    let fp3 = pred_fp3(p0);
    let pb4 = b * PRED_PER_BLOCK;
    let pb3 = pb4 + BLOCK_SIZE;
    // Chain heads q4 / q3 (PRED_NONE: done), their pred words w4 / w3 (q's successor and
    // fingerprint; loaded one step ahead), and the steps left n4 / n3.
    var q4 = pred[pb4 + p] & PRED_POS;
    var w4 = 0u;
    if (q4 != PRED_NONE) { w4 = pred[pb4 + q4]; }
    var n4 = DEPTH;
    var q3 = pred[pb3 + p] & PRED_POS;
    var w3 = 0u;
    if (q3 != PRED_NONE) { w3 = pred[pb3 + q3]; }
    var n3 = H3_DEPTH;
    // Sparse long chains 0, 1, 2 (when N_SPARSE > k): heads qa / qb / qc, words wa / wb / wc and
    // steps left na / nb / nc as above; n = 0 when p is not on the chain's slots.
    let pba = pb4 + SP_OFF0;
    let pbb = pb4 + SP_OFF1;
    let pbc = pb4 + SP_OFF2;
    var qa = PRED_NONE;
    var wa = 0u;
    var na = 0u;
    if (N_SPARSE > 0u && p % SP_S0 == 0u) {
        qa = pred[pba + p / SP_S0] & PRED_POS;
        if (qa != PRED_NONE) { wa = pred[pba + qa / SP_S0]; }
        na = SP_D0;
    }
    var qb = PRED_NONE;
    var wb = 0u;
    var nb = 0u;
    if (N_SPARSE > 1u && p % SP_S1 == 0u) {
        qb = pred[pbb + p / SP_S1] & PRED_POS;
        if (qb != PRED_NONE) { wb = pred[pbb + qb / SP_S1]; }
        nb = SP_D1;
    }
    var qc = PRED_NONE;
    var wc = 0u;
    var nc = 0u;
    if (N_SPARSE > 2u && p % SP_S2 == 0u) {
        qc = pred[pbc + p / SP_S2] & PRED_POS;
        if (qc != PRED_NONE) { wc = pred[pbc + qc / SP_S2]; }
        nc = SP_D2;
    }
    var best_len = 2u;
    var a = 0u;
    var off_b = 0u;
    // Terminates: every iteration decrements the steps left of at least one live head.
    loop {
        let live4 = n4 > 0u && q4 != PRED_NONE;
        let live3 = n3 > 0u && q3 != PRED_NONE;
        var livea = false;
        var liveb = false;
        var livec = false;
        var q = q3;
        if (N_SPARSE == 0u) {
            if (!live4 && !live3) { break; }
            if (live4 && (!live3 || q4 > q3)) { q = q4; }
        } else {
            // The largest live head, as q + 1 (0: none live).
            livea = na > 0u && qa != PRED_NONE;
            liveb = nb > 0u && qb != PRED_NONE;
            livec = nc > 0u && qc != PRED_NONE;
            let m = max(max(select(0u, q4 + 1u, live4), select(0u, q3 + 1u, live3)),
                max(select(0u, qa + 1u, livea), max(select(0u, qb + 1u, liveb), select(0u, qc + 1u, livec))));
            if (m == 0u) { break; }
            q = m - 1u;
        }
        var ub = 0xFFFFFFFFu;
        if (live4 && q4 == q) {
            let x = (w4 ^ fp4) & ~PRED_POS;
            if ((x & PRED_FP_LO) != 0u) {
                ub = 2u;
            } else if (x != 0u) {
                ub = 4u;
            }
            q4 = w4 & PRED_POS;
            n4 -= 1u;
            w4 = 0u;
            if (q4 != PRED_NONE && n4 > 0u) { w4 = pred[pb4 + q4]; }
        }
        if (live3 && q3 == q) {
            let x = (w3 ^ fp3) & ~PRED_POS;
            if ((x & PRED_FP_LO) != 0u) {
                ub = 2u;
            } else if (x != 0u) {
                ub = min(ub, 3u);
            }
            q3 = w3 & PRED_POS;
            n3 -= 1u;
            w3 = 0u;
            if (q3 != PRED_NONE && n3 > 0u) { w3 = pred[pb3 + q3]; }
        }
        if (livea && qa == q) {
            let x = (wa ^ fp4) & ~PRED_POS;
            if ((x & PRED_FP_LO) != 0u) {
                ub = min(ub, 3u);
            } else if (x != 0u) {
                ub = min(ub, 4u);
            }
            qa = wa & PRED_POS;
            na -= 1u;
            wa = 0u;
            if (qa != PRED_NONE && na > 0u) { wa = pred[pba + qa / SP_S0]; }
        }
        if (liveb && qb == q) {
            let x = (wb ^ fp4) & ~PRED_POS;
            if ((x & PRED_FP_LO) != 0u) {
                ub = min(ub, 3u);
            } else if (x != 0u) {
                ub = min(ub, 4u);
            }
            qb = wb & PRED_POS;
            nb -= 1u;
            wb = 0u;
            if (qb != PRED_NONE && nb > 0u) { wb = pred[pbb + qb / SP_S1]; }
        }
        if (livec && qc == q) {
            let x = (wc ^ fp4) & ~PRED_POS;
            if ((x & PRED_FP_LO) != 0u) {
                ub = min(ub, 3u);
            } else if (x != 0u) {
                ub = min(ub, 4u);
            }
            qc = wc & PRED_POS;
            nc -= 1u;
            wc = 0u;
            if (qc != PRED_NONE && nc > 0u) { wc = pred[pbc + qc / SP_S2]; }
        }
        if (ub > best_len) {
            let c = match_len_capped(pw, sp, p0, p1, base, q, mx);
            if (c > best_len) {
                best_len = c;
                off_b = p - q;
                if (a == 0u) { a = off_b | (c << 16u); }
                if (c == mx) { break; }
            }
        }
    }
    best[o] = select(0u, a | (best_len << 24u), a != 0u);
    best[o + 1u] = off_b | select(0u, DEAD_BIT, a == 0u && q3 == PRED_NONE);
}
