// K2: best match per position (== gzc_core::reference::find_best).
// Dispatch (BLOCK_SIZE / 256, n_blocks, 1): one thread per (position, block).
// For p < PARSE_END it walks up to DEPTH candidates of each of the N_HASHES chains in turn
// (Dfast: long, then short), keeping the longest match capped at SEARCH_CAP (ties: larger q).
// best[b*BLOCK_SIZE + p] = (capped len << BEST_OFF_BITS) | offset, 0 when the best length is
// below MIN_MATCH or p >= PARSE_END (BEST_OFF_BITS is injected by the host: offset < BLOCK_SIZE
// <= 2^BEST_OFF_BITS, len <= SEARCH_CAP <= 256). K3 extends matches whose len == SEARCH_CAP.
// Cap early-out, byte-identical to the full walk: (1) within a chain q strictly decreases
// (pred[q] < q), so after a candidate reaches SEARCH_CAP every later one has len <= SEARCH_CAP
// and a smaller q, and loses (a tie keeps the larger q). (2) Across chains (Dfast: long chain
// first, then short): a later-chain candidate q' could only win with len == SEARCH_CAP and
// q' > best_q. SEARCH_CAP >= 8 (MatchParams::validate) = the long hash width, so such a q'
// shares p's first 8 bytes, hence p's long hash, so it is on the long chain above best_q, i.e.
// the long walk reached it before best_q, found len == SEARCH_CAP there and stopped at it:
// best_q >= q', a contradiction. So the whole walk can stop at the first cap-length candidate.
// Fingerprint skips (S8), also byte-identical. The result is the largest (len, q) over the
// visited candidates (lexicographic: longer wins, a tie goes to the larger q), or none if that
// len < MIN_MATCH; so a candidate can be skipped whenever its (len, q) is below the current
// (best_len, best_q) or its len is below MIN_MATCH. The pred word loaded for a candidate q (to get
// the next one) also holds q's fingerprint (common.wgsl `pred_fp`), compared with p's:
// - the hash bits of bytes 0..4 differ: the first 4 bytes differ, len < 4 <= MIN_MATCH: skip;
// - otherwise, byte 4 differs: len <= 4 (max >= 8), skip if MIN_MATCH > 4, or best_len > 4, or
//   best_len == 4 and q < best_q (q == best_q, the same q on both Dfast chains, cannot update
//   either).
// A skipped candidate still counts toward DEPTH, and never has len == SEARCH_CAP (>= 8), so the
// walk and its cap early-out are unchanged. Skips cut K2's data reads: 58 % of lvl9 candidates on
// the corpus are skipped before any of their bytes is loaded.
// MIN_MATCH, SEARCH_CAP, DEPTH and N_HASHES come from the MatchParams the host injects per
// Kernels (`context::params_wgsl`). pred layout (K1): [block][chain][pos], N_HASHES chains per
// block, as pred words (predecessor | fingerprint, see common.wgsl).

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> pred: array<u32>;
@group(0) @binding(2) var<storage, read_write> best: array<u32>;

// Bytes [sh/8, sh/8 + 4) of the little-endian pair (lo, hi); sh in {0, 8, 16, 24}. The second
// shift is split so sh == 0 shifts hi out entirely (WGSL masks shift amounts to 5 bits).
fn funnel(lo: u32, hi: u32, sh: u32) -> u32 {
    return (lo >> sh) | ((hi << (31u - sh)) << 1u);
}

// == match_len(base, p, q, SEARCH_CAP) for max = min(BLOCK_SIZE - p, SEARCH_CAP), which is >= 8
// (p < PARSE_END = BLOCK_SIZE - 8 and SEARCH_CAP >= 8). p is given as its word index
// pw = base + p / 4 and shift sp = (p & 3) * 8, plus its first 8 bytes p0, p1, which the
// thread loads once: most candidates are decided within them, from three q loads. Beyond them
// both sides stream aligned words, one new word per side per 4 bytes (match_len's load_u32_at
// loads two), and the last step is masked to the max - n bytes still in range. Every word read
// is at most word (p + max - 1) / 4 + 1 <= base + BLOCK_SIZE / 4 (q < p likewise): the next
// block's first word or the packed buffer's trailing zero word, and only bytes past max come
// from it.
fn match_len_capped(pw: u32, sp: u32, p0: u32, p1: u32, base: u32, q: u32, max: u32) -> u32 {
    let qw = base + (q >> 2u);
    let sq = (q & 3u) * 8u;
    let q0 = data[qw];
    let q1 = data[qw + 1u];
    var qlo = data[qw + 2u];
    var x = p0 ^ funnel(q0, q1, sq);
    if (x != 0u) { return countTrailingZeros(x) >> 3u; }
    x = p1 ^ funnel(q1, qlo, sq);
    if (x != 0u) { return 4u + (countTrailingZeros(x) >> 3u); }
    var n = 8u;
    var plo = data[pw + 2u];
    var i = 3u;
    loop {
        if (n >= max) { break; }
        let phi = data[pw + i];
        let qhi = data[qw + i];
        x = funnel(plo, phi, sp) ^ funnel(qlo, qhi, sq);
        let left = max - n;
        if (left < 4u) {
            x &= (1u << (left * 8u)) - 1u;
        }
        if (x != 0u) { return n + (countTrailingZeros(x) >> 3u); }
        n += 4u;
        plo = phi;
        qlo = qhi;
        i += 1u;
    }
    return max;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    let b = gid.y;
    let o = b * BLOCK_SIZE + p;
    if (p >= PARSE_END) {
        best[o] = 0u;
        return;
    }
    let base = block_base(b);
    var best_len = 0u;
    var best_q = 0u;
    let max = min(BLOCK_SIZE - p, SEARCH_CAP);
    let pw = base + (p >> 2u);
    let sp = (p & 3u) * 8u;
    let p0 = load_u32_at(base, p);
    let p1 = load_u32_at(base, p + 4u);
    let fpp = pred_fp(p0, p1);
    // Cap early-out: once best_len == SEARCH_CAP nothing later can win, so the whole walk
    // stops (identical to walking on; see the header).
    for (var chain = 0u; chain < N_HASHES && best_len < SEARCH_CAP; chain++) {
        let pb = (b * N_HASHES + chain) * BLOCK_SIZE;
        var q = pred[pb + p] & PRED_POS;
        for (var d = 0u; d < DEPTH; d++) {
            if (q == PRED_NONE) { break; }
            // q's successor and q's fingerprint; compare q only if the fingerprint allows it to
            // win (see the header).
            let wq = pred[pb + q];
            let x = (wq ^ fpp) & ~PRED_POS;
            if ((x & PRED_FP_LO) == 0u
                && (x == 0u || (MIN_MATCH <= 4u && (best_len < 4u || (best_len == 4u && q > best_q))))) {
                let len = match_len_capped(pw, sp, p0, p1, base, q, max);
                if (len > best_len || (len == best_len && q > best_q)) {
                    best_len = len;
                    best_q = q;
                    if (len == SEARCH_CAP) { break; }
                }
            }
            q = wq & PRED_POS;
        }
    }
    best[o] = select(0u, (best_len << BEST_OFF_BITS) | (p - best_q), best_len >= MIN_MATCH);
}
