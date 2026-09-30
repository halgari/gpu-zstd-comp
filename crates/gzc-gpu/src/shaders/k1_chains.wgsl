// K1: builds hash-chain predecessor arrays for long and short match hashes.
// Dispatch (n_blocks, 2, 1): workgroup_id.x = block, workgroup_id.y = width (0 long, 1 short).
// pred[p] = most recent q < p with hash(q) == hash(p), else NO_POS (== gzc_core compute_preds).
// Positions are processed in tiles of 256: each tile is sorted by (hash, lane) so equal hashes
// are adjacent in position order; the first of a run links to head[], the others to their
// sorted neighbour, and the last of a run updates head[].
// head[] holds pos + 1 (0 = none) so a buffer clear initializes it.

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> head: array<u32>;
@group(0) @binding(2) var<storage, read_write> pred_out: array<u32>;

const WG: u32 = 256u;
const N_TILES: u32 = (HASHED_POSITIONS + WG - 1u) / WG;

var<workgroup> keys: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.x;
    let width = wid.y;
    let base = block_base(b);
    let hbase = (b * 2u + width) << HASH_BITS;
    let pbase = (b * 2u + width) * BLOCK_SIZE;

    for (var tile = 0u; tile < N_TILES; tile++) {
        let t0 = tile * WG;
        let p = t0 + lid;
        var h = 0xFFFFFFu; // sorts after every real hash (HASH_BITS <= 24)
        if (p < HASHED_POSITIONS) {
            if (width == 0u) { h = hash_long(base, p); } else { h = hash_short(base, p); }
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
            pred_out[pbase + pos] = pr;
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
        pred_out[pbase + HASHED_POSITIONS + lid] = NO_POS;
    }
}
