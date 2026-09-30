// K2 over the bucket-sorted candidate array (speed2 E2; == gzc_core::reference::find_best_window,
// which equals find_best over the key's hash chains). Appended to k2_best.wgsl (its bindings and
// match_len_capped); entry point `main_window`, Single hash only.
// Input `pred` holds per block the sorted array (k1_sort_sg.wgsl or any builder of the same
// array, gzc_core::hash::bucket_sort): slot s < HASHED_POSITIONS holds q | pred_fp(q) for the
// position q in that slot, positions ordered by key, ascending inside a key. Every position of
// the block gets exactly one best[] word: the thread of slot s writes position sorted[s]'s, and
// the threads of the unused slots HASHED_POSITIONS.. write the positions HASHED_POSITIONS.. (no
// slot; >= PARSE_END, so 0).
// Dispatch (BLOCK_SIZE / 256, n_blocks): one thread per slot, 256 consecutive slots per
// workgroup, which stages their windows (slots s0 - DEPTH .. s0 + 256) in workgroup memory with
// each entry's key (hash_width(q, MIN_MATCH) >> KEY_SHIFT, from q's bytes; empty slots below 0 get
// NO_KEY). The thread of slot s holding p (< PARSE_END) walks the entries below s nearest first
// while they hold p's key, at most DEPTH: exactly p's chain, in chain order (every entry has
// q < p). Fingerprint skips, the tie rule and the cap early-out are k2_best's (see its header),
// so the result is find_best's.

const WIN: u32 = 256u + DEPTH;
const NO_KEY: u32 = 0xFFFFFFFFu;
var<workgroup> win: array<u32, WIN>;
var<workgroup> wkey: array<u32, WIN>;

@compute @workgroup_size(256)
fn main_window(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.y;
    let s0 = wid.x * 256u;
    let sb = b * BLOCK_SIZE;
    let base = block_base(b);
    for (var i = lid; i < WIN; i += 256u) {
        // slot s0 + i - DEPTH, if in 0..HASHED_POSITIONS.
        let s = s0 + i;
        var w = PRED_NONE;
        var k = NO_KEY;
        if (s >= DEPTH && s - DEPTH < HASHED_POSITIONS) {
            w = pred[sb + s - DEPTH];
            k = hash_width(base, w & PRED_POS, MIN_MATCH) >> KEY_SHIFT;
        }
        win[i] = w;
        wkey[i] = k;
    }
    workgroupBarrier();
    let s = s0 + lid;
    if (s >= HASHED_POSITIONS) {
        best[sb + s] = 0u;
        return;
    }
    let wp = win[DEPTH + lid];
    let p = wp & PRED_POS;
    if (p >= PARSE_END) {
        best[sb + p] = 0u;
        return;
    }
    let key = wkey[DEPTH + lid];
    let fpp = wp & ~PRED_POS;
    var best_len = 0u;
    var best_q = 0u;
    let max = min(BLOCK_SIZE - p, SEARCH_CAP);
    let pw = base + (p >> 2u);
    let sp = (p & 3u) * 8u;
    var loaded = false;
    var p0 = 0u;
    var p1 = 0u;
    for (var j = 1u; j <= DEPTH; j++) {
        if (wkey[DEPTH + lid - j] != key) { break; }
        let w = win[DEPTH + lid - j];
        let q = w & PRED_POS;
        let x = (w ^ fpp) & ~PRED_POS;
        if ((x & PRED_FP_LO) == 0u
            && (x == 0u || (MIN_MATCH <= 4u && (best_len < 4u || (best_len == 4u && q > best_q))))) {
            if (!loaded) {
                p0 = load_u32_at(base, p);
                p1 = load_u32_at(base, p + 4u);
                loaded = true;
            }
            let len = match_len_capped(pw, sp, p0, p1, base, q, max);
            if (len > best_len || (len == best_len && q > best_q)) {
                best_len = len;
                best_q = q;
                if (len == SEARCH_CAP) { break; }
            }
        }
    }
    best[sb + p] = select(0u, (best_len << BEST_OFF_BITS) | (p - best_q), best_len >= MIN_MATCH);
}
