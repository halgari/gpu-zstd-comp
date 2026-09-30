// K1, bucket-sorted candidates (speed2 E2, option C; == gzc_core::hash::bucket_sort): per block,
// every hashed position p < HASHED_POSITIONS ordered by key (hash_width(p, MIN_MATCH) >> KEY_SHIFT,
// KEY_BITS bits), ascending inside a key; slot s of block b holds the pred-style word
// q | pred_fp(q) at sorted[b*BLOCK_SIZE + s] (slots HASHED_POSITIONS.. are not written). K2 walks
// the slots below each position's own (k2_window.wgsl). Single hash only.
//
// One workgroup of 32 lanes per block runs a counting sort over 2^KEY_BITS counters in workgroup
// memory, 16 bits each (two per word) when blocks have at most 2^16 positions (a count or cursor
// is at most HASHED_POSITIONS < 2^16 then, so an add never carries into the other half), else 32
// bits:
// 1. histogram: every lane adds its position's key (atomicAdd);
// 2. exclusive scan of the counters in place;
// 3. ranking, one 32-position tile at a time in position order: the lanes holding equal keys
//    are found; the lowest such lane advances its key's cursor by their count and the others take
//    cursor + (equal lanes below them). A barrier after each tile orders one tile's cursor
//    updates before the next tile's, so the order inside a key is the position order. Each
//    position's slot is written to rankw[p] (coalesced) with its fingerprint;
// 4. `scatter`, a second dispatch, moves every position to its slot.
// Two versions of 1-3: `main` below needs no subgroups (lanes compare keys through workgroup
// memory); `main_sg` (k1_sort_sg.wgsl, appended when the device has subgroups of >= 32 lanes)
// uses ballots and shuffles. Their output is identical. No device-scope state: nothing is carried
// between blocks or dispatches.

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> sorted: array<u32>;
@group(0) @binding(2) var<storage, read_write> rankw: array<u32>;

const NKEYS: u32 = 1u << KEY_BITS;
const CNT16: bool = LOG2_BLOCK <= 16u;
const NWORDS: u32 = select(NKEYS, NKEYS / 2u, CNT16);
var<workgroup> cnt: array<atomic<u32>, NWORDS>;

// Adds n to key k's counter and returns its old value.
fn cnt_add(k: u32, n: u32) -> u32 {
    if (CNT16) {
        let sh = (k & 1u) * 16u;
        return (atomicAdd(&cnt[k >> 1u], n << sh) >> sh) & 0xFFFFu;
    }
    return atomicAdd(&cnt[k], n);
}

// Key (x) and pred-style word p | pred_fp (y) of p < HASHED_POSITIONS, from its 8 bytes.
fn key_word(base: u32, p: u32) -> vec2<u32> {
    let lo = load_u32_at(base, p);
    let hi = load_u32_at(base, p + 4u);
    let m = MIN_MATCH - 4u;
    var mask = 0xFFFFFFFFu;
    if (m == 0u) {
        mask = 0u;
    } else if (m < 4u) {
        mask = (1u << (8u * m)) - 1u;
    }
    return vec2<u32>(mix(lo, hi & mask) >> KEY_SHIFT, p | pred_fp(lo, hi));
}

// Without subgroups: the tile's keys (NO_TILE_KEY for dead lanes) and each group's cursor.
const NO_TILE_KEY: u32 = 0xFFFFFFFFu;
var<workgroup> tkey: array<u32, 32>;
var<workgroup> tbase: array<u32, 32>;
// Scan partials, one per lane.
var<workgroup> tsum: array<u32, 32>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let b = wid.x;
    let base = block_base(b);
    let sb = b * BLOCK_SIZE;
    for (var i = lane; i < NWORDS; i += 32u) {
        atomicStore(&cnt[i], 0u);
    }
    workgroupBarrier();

    // 1. Histogram.
    for (var t0 = 0u; t0 < HASHED_POSITIONS; t0 += 32u) {
        let p = t0 + lane;
        if (p < HASHED_POSITIONS) { cnt_add(key_word(base, p).x, 1u); }
    }
    workgroupBarrier();

    // 2. Exclusive scan: lane l owns counter words [l * PER, (l + 1) * PER); its partial sum's
    // prefix over the lanes below comes from tsum.
    const PER: u32 = NWORDS / 32u;
    var sum = 0u;
    for (var i = 0u; i < PER; i++) {
        let x = atomicLoad(&cnt[lane * PER + i]);
        sum += select(x, (x & 0xFFFFu) + (x >> 16u), CNT16);
    }
    tsum[lane] = sum;
    workgroupBarrier();
    var run = 0u;
    for (var l = 0u; l < lane; l++) {
        run += tsum[l];
    }
    for (var i = 0u; i < PER; i++) {
        let x = atomicLoad(&cnt[lane * PER + i]);
        if (CNT16) {
            let lo = x & 0xFFFFu;
            atomicStore(&cnt[lane * PER + i], run | ((run + lo) << 16u));
            run += lo + (x >> 16u);
        } else {
            atomicStore(&cnt[lane * PER + i], run);
            run += x;
        }
    }
    workgroupBarrier();

    // 3. Ranking, tile by tile in position order.
    for (var t0 = 0u; t0 < HASHED_POSITIONS; t0 += 32u) {
        let p = t0 + lane;
        let live = p < HASHED_POSITIONS;
        var kw = vec2<u32>(NO_TILE_KEY, 0u);
        if (live) { kw = key_word(base, p); }
        tkey[lane] = kw.x;
        workgroupBarrier();
        // leader: the first lane holding this key; below: equal lanes below this one; n: all.
        var leader = lane;
        var below = 0u;
        var n = 0u;
        for (var l = 0u; l < 32u; l++) {
            if (tkey[l] == kw.x) {
                if (l < lane) {
                    below += 1u;
                    leader = min(leader, l);
                }
                n += 1u;
            }
        }
        if (live && leader == lane) { tbase[lane] = cnt_add(kw.x, n); }
        workgroupBarrier();
        if (live) { rankw[sb + p] = (tbase[leader] + below) | (kw.y & ~PRED_POS); }
        // The next tile writes tkey after the barrier above (every read of it precedes that
        // barrier) and tbase after its own first barrier (every read of it precedes that).
    }
}

// Scatter (second dispatch of the K1 pass, (BLOCK_SIZE / 256, n_blocks) like K2, so only a few
// blocks' output regions are live at a time and the L2 merges the scattered 4-byte writes;
// scattering from `main` with ~1000 blocks in flight was DRAM-bound): rankw[p] holds p's
// slot | pred_fp(p) from `main`; sorted[slot] = p | pred_fp(p).
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= HASHED_POSITIONS) { return; }
    let sb = gid.y * BLOCK_SIZE;
    let w = rankw[sb + p];
    sorted[sb + (w & PRED_POS)] = p | (w & ~PRED_POS);
}
