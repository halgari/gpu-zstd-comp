// K1, bucket-sorted candidates (speed2 E2, option C; == gzc_core::hash::bucket_sort): per block,
// every hashed position p < HASHED_POSITIONS ordered by key (hash_width(p, MIN_MATCH) >> KEY_SHIFT,
// KEY_BITS bits), ascending inside a key; slot s of block b holds the pred-style word
// q | pred_fp(q) at sorted[b*BLOCK_SIZE + s] (slots HASHED_POSITIONS.. are not written). K2 walks
// the slots below each position's own (k2_window.wgsl). Single hash only.
//
// One workgroup of 32 lanes per block, which must be one subgroup (subgroup size >= 32; checked
// by the host's self-test). A counting sort over 2^KEY_BITS counters in workgroup memory, 16 bits
// each (two per word) when blocks have at most 2^16 positions (a count or cursor is at most
// HASHED_POSITIONS < 2^16 then, so an add never carries into the other half), else 32 bits:
// 1. histogram: every lane adds its position's key (atomicAdd; one add per tile when all 32 keys
//    are equal, as in constant runs);
// 2. exclusive scan of the counters in place (32 at a time, subgroupExclusiveAdd);
// 3. ranking, one 32-position tile at a time in position order: the lanes holding equal keys
//    are found by bit-sliced ballots (KEY_BITS of them, skipped when all keys are equal); the
//    lowest such lane advances its key's cursor by their count and the others take
//    cursor + (equal lanes below them). A barrier after each tile orders one tile's cursor
//    updates before the next tile's, so the order inside a key is the position order. Each
//    position's slot is written to rankw[p] (coalesced) with its fingerprint;
// 4. `scatter`, a second dispatch, moves every position to its slot.
// No device-scope state: nothing is carried between blocks or dispatches.

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

// The 32 words of chunk c (bytes 128c .. 128c + 128 of the block), one per lane; past the block
// they come from the next block (or the buffer's last word) and only reach dead positions.
fn chunk_words(base: u32, c: u32, lane: u32) -> u32 {
    return data[min(base + 32u * c + lane, arrayLength(&data) - 1u)];
}

// Word i (0..34) of the chunk pair (w: this chunk, wn: the next), for every lane.
fn word_at(w: u32, wn: u32, i: u32) -> u32 {
    return select(subgroupShuffle(wn, i & 31u), subgroupShuffle(w, i & 31u), i < 32u);
}

// Key (x) and pred-style word p | pred_fp (y) of position p = 128c + 32j + lane, from the chunk
// pair: == hash_width(p, MIN_MATCH) >> KEY_SHIFT and common.wgsl pred_fp.
fn key_word_chunk(w: u32, wn: u32, c: u32, j: u32, lane: u32) -> vec2<u32> {
    let i = 8u * j + (lane >> 2u);
    let sh = (lane & 3u) * 8u;
    let a = word_at(w, wn, i);
    let b = word_at(w, wn, i + 1u);
    let d = word_at(w, wn, i + 2u);
    // Bytes [sh/8, sh/8 + 4) of each word pair; the second shift is split so sh == 0 shifts the
    // high word out entirely.
    let lo = (a >> sh) | ((b << (31u - sh)) << 1u);
    let hi = (b >> sh) | ((d << (31u - sh)) << 1u);
    let k = MIN_MATCH - 4u;
    var mask = 0xFFFFFFFFu;
    if (k == 0u) {
        mask = 0u;
    } else if (k < 4u) {
        mask = (1u << (8u * k)) - 1u;
    }
    let p = 128u * c + 32u * j + lane;
    return vec2<u32>(mix(lo, hi & mask) >> KEY_SHIFT, p | pred_fp(lo, hi));
}

fn match_bit(k: u32, i: u32) -> u32 {
    let bit = (k >> i) & 1u;
    return subgroupBallot(bit != 0u).x ^ (bit - 1u);
}

// Lanes whose key equals this lane's among `live` (bit-sliced ballots over KEY_BITS bits; written
// out, since naga keeps a loop over the bits rolled).
fn match_key(k: u32, live: u32) -> u32 {
    var eq = live;
    if (KEY_BITS > 0u) { eq &= match_bit(k, 0u); }
    if (KEY_BITS > 1u) { eq &= match_bit(k, 1u); }
    if (KEY_BITS > 2u) { eq &= match_bit(k, 2u); }
    if (KEY_BITS > 3u) { eq &= match_bit(k, 3u); }
    if (KEY_BITS > 4u) { eq &= match_bit(k, 4u); }
    if (KEY_BITS > 5u) { eq &= match_bit(k, 5u); }
    if (KEY_BITS > 6u) { eq &= match_bit(k, 6u); }
    if (KEY_BITS > 7u) { eq &= match_bit(k, 7u); }
    if (KEY_BITS > 8u) { eq &= match_bit(k, 8u); }
    if (KEY_BITS > 9u) { eq &= match_bit(k, 9u); }
    if (KEY_BITS > 10u) { eq &= match_bit(k, 10u); }
    if (KEY_BITS > 11u) { eq &= match_bit(k, 11u); }
    if (KEY_BITS > 12u) { eq &= match_bit(k, 12u); }
    if (KEY_BITS > 13u) { eq &= match_bit(k, 13u); }
    if (KEY_BITS > 14u) { eq &= match_bit(k, 14u); }
    if (KEY_BITS > 15u) { eq &= match_bit(k, 15u); }
    return eq;
}

const CHUNKS: u32 = (HASHED_POSITIONS + 127u) / 128u;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
    let b = wid.x;
    let base = block_base(b);
    let sb = b * BLOCK_SIZE;
    let below = (1u << lane) - 1u;
    for (var i = lane; i < NWORDS; i += 32u) {
        atomicStore(&cnt[i], 0u);
    }
    workgroupBarrier();

    // 1. Histogram, in chunks of 128 positions (4 tiles); the next two chunks' words are loaded
    // ahead.
    var w = chunk_words(base, 0u, lane);
    var wn = chunk_words(base, 1u, lane);
    for (var c = 0u; c < CHUNKS; c++) {
        let wnn = chunk_words(base, c + 2u, lane);
        for (var j = 0u; j < 4u; j++) {
            let live = 128u * c + 32u * j + lane < HASHED_POSITIONS;
            let k = key_word_chunk(w, wn, c, j, lane).x;
            let k0 = subgroupBroadcastFirst(k);
            let lv = subgroupBallot(live).x;
            if (subgroupAll(k == k0 || !live)) {
                if (lane == 0u && lv != 0u) { cnt_add(k0, countOneBits(lv)); }
            } else if (live) {
                cnt_add(k, 1u);
            }
        }
        w = wn;
        wn = wnn;
    }
    workgroupBarrier();

    // 2. Exclusive scan: each counter becomes its key's first slot.
    var run = 0u;
    for (var i0 = 0u; i0 < NWORDS; i0 += 32u) {
        let x = atomicLoad(&cnt[i0 + lane]);
        if (CNT16) {
            let lo = x & 0xFFFFu;
            let ex = run + subgroupExclusiveAdd(lo + (x >> 16u));
            atomicStore(&cnt[i0 + lane], ex | ((ex + lo) << 16u));
            run += subgroupAdd(lo + (x >> 16u));
        } else {
            atomicStore(&cnt[i0 + lane], run + subgroupExclusiveAdd(x));
            run += subgroupAdd(x);
        }
    }
    workgroupBarrier();

    // 3. Ranking, tile by tile in position order.
    w = chunk_words(base, 0u, lane);
    wn = chunk_words(base, 1u, lane);
    for (var c = 0u; c < CHUNKS; c++) {
        let wnn = chunk_words(base, c + 2u, lane);
        for (var j = 0u; j < 4u; j++) {
            let p = 128u * c + 32u * j + lane;
            let live = p < HASHED_POSITIONS;
            let kw = key_word_chunk(w, wn, c, j, lane);
            let k = kw.x;
            let lv = subgroupBallot(live).x;
            let k0 = subgroupBroadcastFirst(k);
            var eq = lv;
            if (!subgroupAll(k == k0 || !live)) { eq = match_key(k, lv); }
            let leader = select(0u, firstTrailingBit(eq), eq != 0u);
            var old = 0u;
            if (live && lane == leader) { old = cnt_add(k, countOneBits(eq)); }
            let slot = subgroupShuffle(old, leader) + countOneBits(eq & below);
            if (live) { rankw[sb + p] = slot | (kw.y & ~PRED_POS); }
            workgroupBarrier();
        }
        w = wn;
        wn = wnn;
    }
}

// Scatter (second dispatch of the K1 pass, (BLOCK_SIZE / 256, n_blocks) like K2, so only ~2
// blocks' 512 KiB output regions are live at a time and the L2 merges the scattered 4-byte
// writes; scattering from `main` with ~500 blocks in flight was DRAM-bound): rankw[p] holds p's
// slot | pred_fp(p) from `main`; sorted[slot] = p | pred_fp(p).
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= HASHED_POSITIONS) { return; }
    let sb = gid.y * BLOCK_SIZE;
    let w = rankw[sb + p];
    sorted[sb + (w & PRED_POS)] = p | (w & ~PRED_POS);
}
