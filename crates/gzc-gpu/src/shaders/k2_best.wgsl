// K2: best match per position (== gzc_core::reference::find_best).
// Dispatch (BLOCK_SIZE / 256, n_blocks, 1): one thread per (position, block).
// For p < PARSE_END it walks up to DEPTH candidates of each of the N_HASHES chains in turn
// (Dfast: long, then short), keeping the longest match capped at SEARCH_CAP (ties: larger q).
// best[(b*BLOCK_SIZE + p)*2 ..] = (offset, capped len), (0, 0) when the best length is below
// MIN_MATCH or p >= PARSE_END. K3 extends matches whose len == SEARCH_CAP.
// MIN_MATCH, SEARCH_CAP, DEPTH and N_HASHES come from the MatchParams the host injects per
// Kernels (`context::params_wgsl`). pred layout (K1): [block][chain][pos], N_HASHES chains per block.

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> pred: array<u32>;
@group(0) @binding(2) var<storage, read_write> best: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    let b = gid.y;
    let o = (b * BLOCK_SIZE + p) * 2u;
    if (p >= PARSE_END) {
        best[o] = 0u;
        best[o + 1u] = 0u;
        return;
    }
    let base = block_base(b);
    var best_len = 0u;
    var best_q = 0u;
    for (var chain = 0u; chain < N_HASHES; chain++) {
        let pb = (b * N_HASHES + chain) * BLOCK_SIZE;
        var q = pred[pb + p];
        for (var d = 0u; d < DEPTH; d++) {
            if (q == NO_POS) { break; }
            let len = match_len(base, p, q, SEARCH_CAP);
            if (len > best_len || (len == best_len && q > best_q)) {
                best_len = len;
                best_q = q;
            }
            q = pred[pb + q];
        }
    }
    if (best_len >= MIN_MATCH) {
        best[o] = p - best_q;
        best[o + 1u] = best_len;
    } else {
        best[o] = 0u;
        best[o + 1u] = 0u;
    }
}
