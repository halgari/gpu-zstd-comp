// K1, subgroup kernel (needs Features::SUBGROUP and subgroups of 32..=128 lanes, and passes a
// self-test at ChainsKernel::new; otherwise the fallback k1_chains.wgsl runs). Same output as the
// fallback: pred[p] = most recent q < p with hash(q) == hash(p), else none (== gzc_core
// compute_preds), layout pred[(b*N_HASHES + chain)*BLOCK_SIZE ..], stored as pred words with p's
// fingerprint (common.wgsl `pred_word`; the tail p >= HASHED_POSITIONS holds PRED_NONE). pred is
// bound to exactly this dispatch's chains, so n_tasks = arrayLength(pred) / PRED_PER_BLOCK *
// N_HASHES.
// M6 sparse long chains (Opt3 chains N_FULL.., `chains::layout_wgsl`): chain N_FULL + k hashes
// only the slots i < SP_N{k}, position p = i * SP_S{k} (stride 4 or 8, so the key's words are
// word aligned), on `long_hash` (== gzc_core::reference::sparse_chain_preds), stored compactly at
// pred[b*PRED_PER_BLOCK + SP_OFF{k} + i] (the slots from SP_N{k} on hold PRED_NONE), with
// predecessor *positions* and pred_fp fingerprints. The build is the same, over slots instead of
// positions: tiles, head entries and links are slot indices, scaled by the stride on output.
//
// Persistent grid. A task is one chain t = b*N_HASHES + chain; workgroup w of the G dispatched
// builds tasks w, w + G, w + 2G, .. in order, all in its own head table head[w << HASH_BITS ..].
// G is kept small enough for the G live tables (256 KiB each) to stay in L2: with one table per
// block (1638 x 256 KiB live) K1 was DRAM-bound. Each workgroup clears its table once at the start
// of the dispatch; its j-th task stamps entries (tag << LOG2_BLOCK) | (pos + 1) with tag = j + 1,
// and only entries of the current tag count (pos + 1 <= HASHED_POSITIONS < BLOCK_SIZE fits
// LOG2_BLOCK bits). So the table is not cleared between the workgroup's tasks, and no tag state
// outlives the dispatch (the host ensures a workgroup has at most MAX_TAG tasks).
//
// A task walks its block in tiles of T = 256 positions, one per invocation. The tile-local index
// li = subgroup_id * subgroup_size + subgroup_invocation_id (position t0 + li) splits the tile
// into 8 chunks of 32 lanes, each inside one subgroup:
// 1. Each chunk matches equal hashes by bit-slicing the hash through ballots (a subgroup "match
//    any") and publishes its 16 bit-ballots plus its live ballot to workgroup memory. A lane with
//    an equal lane below it in its chunk links to the highest one.
// 2. After a barrier, every lane compares its hash against the published ballots of all 8 chunks:
//    a chunk-first lane links to the highest equal lane of the highest earlier chunk; with none it
//    is the first of its hash in the tile, links to head[h] and stores the tile's last position of
//    h (highest equal lane of the highest chunk) into head[h].
// That is exactly the sequential chain build, for any subgroup size in 32..=128.
// head[] is accessed with relaxed atomics (so the speculative load below never races with the
// tile-first lane's store) and a storageBarrier ends every tile, which orders one tile's head
// stores before the next tile's loads. The published ballots are double-buffered by tile parity:
// tile k + 2 overwrites tile k's buffer only after tile k + 1's middle barrier, which every read
// of tile k precedes. So a tile has two barriers (the fallback's bitonic sort has 38).
//
// Assumes a workgroup's subgroups are full and equally sized (256 is a multiple of the subgroup
// size), so li covers 0..256 exactly once.
//
// Every subgroup operation runs in subgroup-uniform control flow, and no operand comes straight
// out of a lane-dependent branch: the hash is computed before a barrier (see `h` below) and
// chunk_first uses `&` (naga lowers `&&` to an `if`), so nothing relies on the lanes reconverging
// after divergence (VK_KHR_shader_maximal_reconvergence is not enabled;
// .superpowers/m6-research/subgroup-audit.md).

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> head: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> pred_out: array<u32>;


const T: u32 = 256u;
const CHUNKS: u32 = T / 32u;
const POS_MASK: u32 = (1u << LOG2_BLOCK) - 1u;

// Per parity and chunk, 5 vec4: the 16 hash-bit ballots (bit i at [i / 4][i % 4]), then the live
// ballot in [4].x.
var<workgroup> bal: array<vec4<u32>, 2u * CHUNKS * 5u>;

const_assert HASH_BITS == 16u;

// The chain's hash at p from the words w = data[base + p / 4 ..][0..3] (lo and hi are
// load_u32_at(base, p) and load_u32_at(base, p + 4)): == hash_width(base, p, MIN_MATCH) for Single,
// hash_long / hash_short for Dfast chain 0 / 1, hash_width(.., 4) / hash3 for Opt3 chain 0 / 1, reduced to the chain key (>> KEY_SHIFT).
fn chain_hash_words(w: vec3<u32>, p: u32, chain: u32) -> u32 {
    if (N_SPARSE > 0u && chain >= N_FULL) {
        // A sparse long chain: p is word aligned, w its key's three words.
        return long_hash(w.x, w.y, w.z, sp_width(chain - N_FULL));
    }
    let sh = (p & 3u) * 8u;
    var lo = w.x;
    var hi = w.y;
    if (sh != 0u) {
        lo = (w.x >> sh) | (w.y << (32u - sh));
        hi = (w.y >> sh) | (w.z << (32u - sh));
    }
    if (N_HASHES == 1u) {
        let k = MIN_MATCH - 4u;
        var mask = 0xFFFFFFFFu;
        if (k == 0u) {
            mask = 0u;
        } else if (k < 4u) {
            // `& 31u`: a no-op for k < 4, but this dead branch is still const-evaluated where
            // MIN_MATCH < 4 (Opt3, k wraps) once naga folds k, as the GZC_EMULATE_* rewrite of
            // naga's output does, and a shift by >= 32 there failed the module.
            mask = (1u << ((8u * k) & 31u)) - 1u;
        }
        return mix(lo, hi & mask) >> KEY_SHIFT;
    } else if (OPT3) {
        if (chain == 0u) { return mix(lo, 0u) >> KEY_SHIFT; }
        return ((lo << 8u) * 506832829u) >> (32u - HASH_BITS) >> KEY_SHIFT;
    } else if (chain == 0u) {
        return mix(lo, hi) >> KEY_SHIFT;
    }
    return mix(lo, hi & 0xFFu) >> KEY_SHIFT;
}

// The pred word fingerprint at p from the same words: pred_fp, or pred_fp3 on the Opt3 h3 chain.
fn fp_words(w: vec3<u32>, p: u32, chain: u32) -> u32 {
    let sh = (p & 3u) * 8u;
    var lo = w.x;
    var hi = w.y;
    if (sh != 0u) {
        lo = (w.x >> sh) | (w.y << (32u - sh));
        hi = (w.y >> sh) | (w.z << (32u - sh));
    }
    if (OPT3 && chain == 1u) { return pred_fp3(lo); }
    return pred_fp(lo, hi);
}

// The words chain_hash_words needs at tile index i, position i * stride (zeros, without loading,
// if i >= n_idx: not hashed).
fn load_words(base: u32, i: u32, stride: u32, n_idx: u32) -> vec3<u32> {
    var w = vec3<u32>(0u);
    if (i < n_idx) {
        let x = base + ((i * stride) >> 2u);
        w = vec3<u32>(data[x], data[x + 1u], data[x + 2u]);
    }
    return w;
}

// Ballot of bit i of h over this lane's chunk (ballot word `word`; K1_BALLOT_WORD is `.x` for
// subgroups of at most 32 lanes, else `[word]`, host-generated), published by the chunk's lane i
// at bal[at ..]; returns the lanes whose bit i equals this lane's. One lane stores one component
// per call, the calls in order: safe where a component store is a read-modify-write of the whole
// vec4 (Metal; `GZC_EMULATE_VEC_RMW`), unlike several lanes storing components at once (K4's old
// wg_scan). Storing whole vec4s from one lane instead cost this kernel 7 % (RTX 5090, lvl9).
fn publish_bit(h: u32, i: u32, word: u32, cl: u32, at: u32) -> u32 {
    let bit = (h >> i) & 1u;
    let m = subgroupBallot(bit != 0u)K1_BALLOT_WORD;
    if (cl == i) { bal[at + (i >> 2u)][i & 3u] = m; }
    return m ^ (bit - 1u);
}

// Publishes this chunk's ballots at bal[at ..] and returns its live lanes holding hash h. Written
// out: naga's loop bounding keeps a loop over the bits rolled, which made the ballots several
// times slower.
fn publish(h: u32, live: bool, word: u32, cl: u32, at: u32) -> u32 {
    let lv = subgroupBallot(live)K1_BALLOT_WORD;
    if (cl == 16u) { bal[at + 4u].x = lv; }
    return lv
        & publish_bit(h, 0u, word, cl, at) & publish_bit(h, 1u, word, cl, at)
        & publish_bit(h, 2u, word, cl, at) & publish_bit(h, 3u, word, cl, at)
        & publish_bit(h, 4u, word, cl, at) & publish_bit(h, 5u, word, cl, at)
        & publish_bit(h, 6u, word, cl, at) & publish_bit(h, 7u, word, cl, at)
        & publish_bit(h, 8u, word, cl, at) & publish_bit(h, 9u, word, cl, at)
        & publish_bit(h, 10u, word, cl, at) & publish_bit(h, 11u, word, cl, at)
        & publish_bit(h, 12u, word, cl, at) & publish_bit(h, 13u, word, cl, at)
        & publish_bit(h, 14u, word, cl, at) & publish_bit(h, 15u, word, cl, at);
}

// Lanes of the chunk published at bal[at ..] that are live and hold the hash whose bits are
// nb = (bit i of h) - 1 (4 vec4, bit i at [i / 4][i % 4]).
fn match_published(at: u32, nb0: vec4<u32>, nb1: vec4<u32>, nb2: vec4<u32>, nb3: vec4<u32>) -> u32 {
    let a = (bal[at] ^ nb0) & (bal[at + 1u] ^ nb1) & (bal[at + 2u] ^ nb2) & (bal[at + 3u] ^ nb3);
    return bal[at + 4u].x & a.x & a.y & a.z & a.w;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(subgroup_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_lane: u32,
) {
    let li = sg_id * sg_size + sg_lane;
    let chunk = li >> 5u;
    let cl = li & 31u;
    let word = sg_lane >> 5u;
    let below = (1u << cl) - 1u;
    let hb = wid.x << HASH_BITS;
    let n_tasks = arrayLength(&pred_out) / PRED_PER_BLOCK * N_HASHES;
    var parity = 0u;

    // Stale entries of earlier dispatches may carry any tag: clear this workgroup's table.
    for (var i = li; i < (1u << HASH_BITS); i += T) {
        atomicStore(&head[hb + i], 0u);
    }
    storageBarrier();

    for (var j = 0u; ; j++) {
        let t = wid.x + nwg.x * j;
        if (t >= n_tasks) { break; }
        let b = t / N_HASHES;
        let chain = t % N_HASHES;
        let base = block_base(b);
        // The task's tile indices i < n_idx are positions i * stride; its pred words start at pb
        // (n_len of them, the ones from n_idx on PRED_NONE).
        var stride = 1u;
        var n_idx = HASHED_POSITIONS;
        var n_len = BLOCK_SIZE;
        // Without sparse chains PRED_PER_BLOCK == N_HASHES * BLOCK_SIZE, so pb == t * BLOCK_SIZE,
        // written so: `b * PRED_PER_BLOCK + chain * BLOCK_SIZE` there made the driver's code for
        // the full chains ~5 % slower, this form ~10 % faster than before M6 (RTX 5090, opt16
        // and lvl3, B2 report).
        var pb = t * BLOCK_SIZE;
        if (N_SPARSE > 0u) { pb = b * PRED_PER_BLOCK + chain * BLOCK_SIZE; }
        if (N_SPARSE > 0u && chain >= N_FULL) {
            let k = chain - N_FULL;
            stride = sp_stride(k);
            n_idx = sp_slots(k);
            n_len = BLOCK_SIZE / stride;
            pb = b * PRED_PER_BLOCK + sp_off(k);
        }
        let tag = (j + 1u) << LOG2_BLOCK;
        // This lane's data words for the current tile; the next tile's are loaded a tile ahead.
        var words = load_words(base, li, stride, n_idx);
        // This lane's hash for the current tile, computed a tile ahead (before the previous tile's
        // closing storageBarrier; the first tile's here, before a barrier of its own). The
        // ballots that consume h then follow a barrier, not the per-lane branches that built it
        // (load_words' bound, chain_hash_words' byte shift). It also hides the hash's latency:
        // K1 11 % faster than computing it at the top of the tile (RTX 5090, lvl9seg).
        var h = chain_hash_words(words, li * stride, chain);
        workgroupBarrier();

        for (var t0 = 0u; t0 < n_idx; t0 += T) {
            let mb = parity * CHUNKS * 5u;
            parity ^= 1u;

            // Tile index i (a position, or a sparse chain's slot), position p.
            let i = t0 + li;
            let p = i * stride;
            let live = i < n_idx;
            let cur = words;
            words = load_words(base, i + T, stride, n_idx);
            // A dead lane's h (from zero words) is masked out by the live ballot. chunk_first is
            // built without a lane-dependent branch (`&`, not `&&`, which naga lowers to an `if`).
            let eq = publish(h, live, word, cl, mb + chunk * 5u);
            let lower = eq & below;
            let chunk_first = live & (lower == 0u);
            // A chunk-first lane may be its hash's first in the tile: load head[h] now so the load
            // overlaps the barrier and the matching; it is only used (and head[h] only written)
            // by the tile-first lane, after the barrier.
            var old = 0u;
            if (chunk_first) { old = atomicLoad(&head[hb + h]); }
            workgroupBarrier();

            // e[c] = lanes of chunk c holding h. Skipped by subgroups without a chunk-first lane.
            var e = array<u32, 8>();
            if (subgroupAny(chunk_first)) {
                let hv = vec4<u32>(h, h >> 1u, h >> 2u, h >> 3u);
                let one = vec4<u32>(1u);
                let nb0 = (hv & one) - one;
                let nb1 = ((hv >> vec4<u32>(4u)) & one) - one;
                let nb2 = ((hv >> vec4<u32>(8u)) & one) - one;
                let nb3 = ((hv >> vec4<u32>(12u)) & one) - one;
                e[0] = match_published(mb, nb0, nb1, nb2, nb3);
                e[1] = match_published(mb + 5u, nb0, nb1, nb2, nb3);
                e[2] = match_published(mb + 10u, nb0, nb1, nb2, nb3);
                e[3] = match_published(mb + 15u, nb0, nb1, nb2, nb3);
                e[4] = match_published(mb + 20u, nb0, nb1, nb2, nb3);
                e[5] = match_published(mb + 25u, nb0, nb1, nb2, nb3);
                e[6] = match_published(mb + 30u, nb0, nb1, nb2, nb3);
                e[7] = match_published(mb + 35u, nb0, nb1, nb2, nb3);
            }
            if (live) {
                var pr = t0 + 32u * chunk + firstLeadingBit(lower);
                if (lower == 0u) {
                    let nz = select(0u, 1u, e[0] != 0u) | select(0u, 2u, e[1] != 0u)
                        | select(0u, 4u, e[2] != 0u) | select(0u, 8u, e[3] != 0u)
                        | select(0u, 16u, e[4] != 0u) | select(0u, 32u, e[5] != 0u)
                        | select(0u, 64u, e[6] != 0u) | select(0u, 128u, e[7] != 0u);
                    let earlier = nz & ((1u << chunk) - 1u);
                    // Highest earlier chunk holding h, else the highest chunk (h's last in the tile).
                    let c2 = firstLeadingBit(select(nz, earlier, earlier != 0u));
                    var ec = e[0];
                    if (c2 == 1u) { ec = e[1]; }
                    if (c2 == 2u) { ec = e[2]; }
                    if (c2 == 3u) { ec = e[3]; }
                    if (c2 == 4u) { ec = e[4]; }
                    if (c2 == 5u) { ec = e[5]; }
                    if (c2 == 6u) { ec = e[6]; }
                    if (c2 == 7u) { ec = e[7]; }
                    let q = t0 + 32u * c2 + firstLeadingBit(ec);
                    if (earlier != 0u) {
                        pr = q;
                    } else {
                        atomicStore(&head[hb + h], tag | (q + 1u));
                        pr = select(NO_POS, (old & POS_MASK) - 1u, (old & ~POS_MASK) == tag);
                    }
                }
                // pr is a tile index: the predecessor's position is pr * stride.
                if (pr != NO_POS) { pr *= stride; }
                pred_out[pb + i] = pred_word(pr, fp_words(cur, p, chain));
            }
            h = chain_hash_words(words, (i + T) * stride, chain);
            storageBarrier();
        }

        if (li < n_len - n_idx) {
            pred_out[pb + n_idx + li] = PRED_NONE;
        }
    }
}
