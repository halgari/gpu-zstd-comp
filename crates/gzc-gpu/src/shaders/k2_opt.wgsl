// K2opt: the optimal parse's two candidate records per position (== gzc_core::reference::find_cands,
// m5-opt-design §2.1). Appended to k2_best.wgsl (whose bindings, `funnel` and `match_len_capped` it
// uses); entry point main_opt, built only for Opt3 params (K1 wrote pred[(b*2 + 0)*BLOCK_SIZE ..]
// = the h4 chain, with pred_fp fingerprints, and pred[(b*2 + 1)*BLOCK_SIZE ..] = the h3 chain,
// with pred_fp3 fingerprints).
// Dispatch (BLOCK_SIZE / 256, n_blocks, 1): one thread per (position, block). Output (the `best`
// binding, 2 words per position): best[2*(b*BLOCK_SIZE + p) ..] = (offA | lenA << 16 | lenB << 24,
// offB), all zero without a record or for p >= PARSE_END.
//
// The walk: the h4 chain DEPTH deep and the h3 chain H3_DEPTH deep, merged nearest first (the
// larger live head; a position on both chains is visited once and advances both). A visited q
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
//   skipped); byte field differs -> c <= 3.
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
    let p = gid.x;
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
    let pb4 = N_HASHES * b * BLOCK_SIZE;
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
    var best_len = 2u;
    var a = 0u;
    var off_b = 0u;
    // Terminates: every iteration decrements n4 or n3 (at least one head is live).
    loop {
        let live4 = n4 > 0u && q4 != PRED_NONE;
        let live3 = n3 > 0u && q3 != PRED_NONE;
        if (!live4 && !live3) { break; }
        var q = q3;
        if (live4 && (!live3 || q4 > q3)) { q = q4; }
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
