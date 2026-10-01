// K1 bucket sort, subgroup version (`main_sg`), appended to k1_sort.wgsl (its bindings, counters
// and `scatter`; the steps are described there) when the device has subgroups of >= 32 lanes: one
// workgroup of 32 lanes per block, which must be one subgroup (checked by the host's self-test).
// It loads each 128-position chunk's words once, one per lane, and builds the keys with shuffles;
// the histogram adds once per tile when all 32 keys are equal (constant runs); the scan uses
// subgroupExclusiveAdd; the ranking finds equal keys with bit-sliced ballots (KEY_BITS of them,
// skipped when all keys are equal).
//
// Every subgroup operation runs in subgroup-uniform control flow, and no operand comes out of a
// lane-dependent branch (`|` instead of `||`, which naga lowers to an `if`), so nothing relies on
// the lanes reconverging after divergence (VK_KHR_shader_maximal_reconvergence is not enabled;
// .superpowers/m6-research/subgroup-audit.md).

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

// The ranking's per-tile handoff of each leader's first slot (two tiles' worth, by parity).
var<workgroup> first_slot: array<u32, 64>;

@compute @workgroup_size(32)
fn main_sg(@builtin(workgroup_id) wid: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
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
            if (subgroupAll((k == k0) | !live)) {
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
        let lo = x & 0xFFFFu;
        let ex = run + subgroupExclusiveAdd(lo + (x >> 16u));
        atomicStore(&cnt[i0 + lane], ex | ((ex + lo) << 16u));
        run += subgroupAdd(lo + (x >> 16u));
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
            if (!subgroupAll((k == k0) | !live)) { eq = match_key(k, lv); }
            let leader = select(0u, firstTrailingBit(eq), eq != 0u);
            // The leader's first slot reaches its equal lanes through workgroup memory across the
            // tile's barrier (not a shuffle right after the leader-only branch, which would need the
            // lanes reconverged there). The barrier also orders the tiles' cnt_adds; the handoff is
            // double-buffered by tile parity, so tile j + 1's leaders do not overwrite what tile j's
            // lanes read before tile j + 1's barrier.
            let hb = (j & 1u) * 32u;
            if (live && lane == leader) { first_slot[hb + lane] = cnt_add(k, countOneBits(eq)); }
            workgroupBarrier();
            let slot = first_slot[hb + leader] + countOneBits(eq & below);
            if (live) { rankw[sb + p] = slot | (kw.y & ~PRED_POS); }
        }
        w = wn;
        wn = wnn;
    }
}
