// K1 fallback (no subgroups; see k1_chains_sg.wgsl): builds the hash-chain predecessor arrays K2
// walks (== gzc_core::reference::chains). A task is one chain t = b*N_HASHES + chain: with
// N_HASHES == 2 (Dfast) chain 0 is hash_long and chain 1 hash_short; with N_HASHES == 1 (Single)
// chain 0 is hash_width(MIN_MATCH). N_HASHES and MIN_MATCH come from the injected MatchParams
// (`context::params_wgsl`). pred[t*BLOCK_SIZE + p] = most recent q < p with hash(q) == hash(p),
// else none (== gzc_core compute_preds), stored as pred words with p's fingerprint (common.wgsl
// `pred_word`; PRED_NONE in the tail p >= HASHED_POSITIONS); pred is bound to exactly the
// dispatch's tasks, so n_tasks = arrayLength(pred) / BLOCK_SIZE.
// Persistent grid (like the subgroup kernel, so both fit the same head buffer, sized for at most
// chains::HEAD_TABLES tables): workgroup w of G builds tasks w, w + G, .. in table
// head[w << HASH_BITS ..], which it clears before each task.
// Positions are processed in tiles of 256: each tile is sorted by (hash, lane) so equal hashes
// are adjacent in position order; the first of a run links to head[], the others to their
// sorted neighbour, and the last of a run updates head[].
// head[] holds pos + 1 (0 = none).

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> head: array<u32>;
@group(0) @binding(2) var<storage, read_write> pred_out: array<u32>;

const WG: u32 = 256u;
const N_TILES: u32 = (HASHED_POSITIONS + WG - 1u) / WG;

var<workgroup> keys: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let n_tasks = arrayLength(&pred_out) / BLOCK_SIZE;
    let hbase = wid.x << HASH_BITS;
    for (var t = wid.x; t < n_tasks; t += nwg.x) {
        let b = t / N_HASHES;
        let chain = t % N_HASHES;
        let base = block_base(b);
        let pbase = t * BLOCK_SIZE;
        for (var i = lid; i < (1u << HASH_BITS); i += WG) {
            head[hbase + i] = 0u;
        }
        storageBarrier();

        for (var tile = 0u; tile < N_TILES; tile++) {
            let t0 = tile * WG;
            let p = t0 + lid;
            var h = 0xFFFFFFu; // sorts after every real hash (HASH_BITS <= 24)
            if (p < HASHED_POSITIONS) {
                if (N_HASHES == 1u) {
                    h = hash_width(base, p, MIN_MATCH);
                } else if (chain == 0u) {
                    h = hash_long(base, p);
                } else {
                    h = hash_short(base, p);
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
            let pos = t0 + (key & 255u);
            let live = pos < HASHED_POSITIONS;
            if (live) {
                var pr: u32;
                if (lid > 0u && (keys[lid - 1u] >> 8u) == hk) {
                    pr = t0 + (keys[lid - 1u] & 255u);
                } else {
                    let hv = head[hbase + hk];
                    pr = select(hv - 1u, NO_POS, hv == 0u);
                }
                pred_out[pbase + pos] = pred_word(pr, pred_fp(load_u32_at(base, pos), load_u32_at(base, pos + 4u)));
            }
            storageBarrier();
            workgroupBarrier();
            if (live && (lid == WG - 1u || (keys[lid + 1u] >> 8u) != hk)) {
                head[hbase + hk] = pos + 1u;
            }
            storageBarrier();
            workgroupBarrier();
        }

        if (lid < BLOCK_SIZE - HASHED_POSITIONS) {
            pred_out[pbase + HASHED_POSITIONS + lid] = PRED_NONE;
        }
    }
}
