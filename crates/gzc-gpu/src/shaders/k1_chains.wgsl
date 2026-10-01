// K1 fallback (no subgroups; see k1_chains_sg.wgsl): builds the hash-chain predecessor arrays K2
// walks (== gzc_core::reference::chains). A task is one chain t = b*N_HASHES + chain: with
// N_HASHES == 2 (Dfast) chain 0 is hash_long and chain 1 hash_short; with OPT3 (Opt3 params)
// chain 0 is hash_width(4) and chain 1 hash3, whose words carry pred_fp3 instead of pred_fp, and
// chains N_FULL.. are the M6 sparse long chains (below); with N_HASHES == 1 (Single) chain 0 is
// hash_width(MIN_MATCH). N_HASHES and MIN_MATCH come from the injected MatchParams
// (`context::params_wgsl`); the chains link equal keys, the hash's top MatchParams::hash_bits bits
// (hash >> KEY_SHIFT, `chains::finder_wgsl`). pred[b*PRED_PER_BLOCK + chain*BLOCK_SIZE + p] = most
// recent q < p with key(q) == key(p), else none (== gzc_core compute_preds), stored as pred words
// with p's fingerprint (common.wgsl `pred_word`; PRED_NONE in the tail p >= HASHED_POSITIONS);
// pred is bound to exactly the dispatch's tasks, so n_tasks = arrayLength(pred) / PRED_PER_BLOCK *
// N_HASHES.
// Sparse long chain N_FULL + k (`chains::layout_wgsl`): only the slots i < SP_N{k}, position
// p = i * SP_S{k} (word aligned), are hashed, on `long_hash` (== reference::sparse_chain_preds); the
// chain is stored compactly at pred[b*PRED_PER_BLOCK + SP_OFF{k} + i] (PRED_NONE from slot SP_N{k}
// on) with predecessor positions and pred_fp fingerprints. The build runs over slots instead of
// positions.
// Persistent grid (like the subgroup kernel, so both fit the same head buffer, sized for at most
// chains::HEAD_TABLES tables): workgroup w of G builds tasks w, w + G, .. in table
// head[w << HASH_BITS ..], which it clears before each task.
// Positions (slots) are processed in tiles of 256: each tile is sorted by (hash, lane) so equal
// hashes are adjacent in position order; the first of a run links to head[], the others to their
// sorted neighbour, and the last of a run updates head[].
// head[] holds index + 1 (0 = none).

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> head: array<u32>;
@group(0) @binding(2) var<storage, read_write> pred_out: array<u32>;

const WG: u32 = 256u;

var<workgroup> keys: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let n_tasks = arrayLength(&pred_out) / PRED_PER_BLOCK * N_HASHES;
    let hbase = wid.x << HASH_BITS;
    for (var t = wid.x; t < n_tasks; t += nwg.x) {
        let b = t / N_HASHES;
        let chain = t % N_HASHES;
        let base = block_base(b);
        // Tile indices i < n_idx are positions i * stride; the task's pred words start at pbase
        // (n_len of them, the ones from n_idx on PRED_NONE).
        var stride = 1u;
        var n_idx = HASHED_POSITIONS;
        var n_len = BLOCK_SIZE;
        var pbase = b * PRED_PER_BLOCK + chain * BLOCK_SIZE;
        var sparse = false;
        if (N_SPARSE > 0u && chain >= N_FULL) {
            let k = chain - N_FULL;
            sparse = true;
            stride = sp_stride(k);
            n_idx = sp_slots(k);
            n_len = BLOCK_SIZE / stride;
            pbase = b * PRED_PER_BLOCK + sp_off(k);
        }
        for (var i = lid; i < (1u << HASH_BITS); i += WG) {
            head[hbase + i] = 0u;
        }
        storageBarrier();

        let n_tiles = (n_idx + WG - 1u) / WG;
        for (var tile = 0u; tile < n_tiles; tile++) {
            let t0 = tile * WG;
            let p = (t0 + lid) * stride;
            var h = 0xFFFFFFu; // sorts after every real hash (HASH_BITS <= 24)
            if (t0 + lid < n_idx) {
                if (sparse) {
                    h = long_hash(load_u32_at(base, p), load_u32_at(base, p + 4u), load_u32_at(base, p + 8u),
                        sp_width(chain - N_FULL));
                } else if (N_HASHES == 1u) {
                    h = hash_width(base, p, MIN_MATCH) >> KEY_SHIFT;
                } else if (OPT3) {
                    h = select(hash3(base, p), hash_width(base, p, 4u), chain == 0u) >> KEY_SHIFT;
                } else if (chain == 0u) {
                    h = hash_long(base, p) >> KEY_SHIFT;
                } else {
                    h = hash_short(base, p) >> KEY_SHIFT;
                }
            }
            keys[lid] = (h << 8u) | lid;
            workgroupBarrier();

            // Bitonic sort of keys ascending.
            for (var k = 2u; k <= WG; k <<= 1u) {
                for (var j = k >> 1u; j > 0u; j >>= 1u) {
                    let ixj = lid ^ j;
                    if (ixj > lid) {
                        let a = keys[lid];
                        let c = keys[ixj];
                        let up = (lid & k) == 0u;
                        if ((a > c) == up) {
                            keys[lid] = c;
                            keys[ixj] = a;
                        }
                    }
                    workgroupBarrier();
                }
            }

            let key = keys[lid];
            let hk = key >> 8u;
            let idx = t0 + (key & 255u);
            let live = idx < n_idx;
            if (live) {
                var pr: u32;
                if (lid > 0u && (keys[lid - 1u] >> 8u) == hk) {
                    pr = t0 + (keys[lid - 1u] & 255u);
                } else {
                    let hv = head[hbase + hk];
                    pr = select(hv - 1u, NO_POS, hv == 0u);
                }
                if (pr != NO_POS) { pr *= stride; }
                let pos = idx * stride;
                let lo = load_u32_at(base, pos);
                var fp = pred_fp(lo, load_u32_at(base, pos + 4u));
                if (OPT3 && chain == 1u) { fp = pred_fp3(lo); }
                pred_out[pbase + idx] = pred_word(pr, fp);
            }
            storageBarrier();
            workgroupBarrier();
            if (live && (lid == WG - 1u || (keys[lid + 1u] >> 8u) != hk)) {
                head[hbase + hk] = idx + 1u;
            }
            storageBarrier();
            workgroupBarrier();
        }

        if (lid < n_len - n_idx) {
            pred_out[pbase + n_idx + lid] = PRED_NONE;
        }
    }
}
