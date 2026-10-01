// K3opt's block schedule (M6 A4, a07): the persistent K3opt passes (k3_opt.wgsl main_opt_persist)
// take their blocks heaviest first. Recorded once per batch, after K2opt and before the first pass,
// on the candidate words K2opt left in `best` (no pass reads the schedule's words but K3opt).
//
// `sched` (k3opt::sched_bytes, bound to exactly SCHED_HDR + 2 n words for an n-block batch):
//   [0]                       the pass's block counter (cleared before every persistent pass);
//   [1 .. SCHED_HDR)          unused;
//   [SCHED_HDR + b]           block b's weight (main_weight);
//   [SCHED_HDR + n + i]       the i-th block to run (main_order): blocks by descending weight,
//                             equal weights by ascending block id.
//
// Weight (a07's cost proxy, Spearman 0.925 against the block's K3opt time): the block's positions
// whose longest candidate (max(lenA, lenB)) is 3..32. Longer matches are encoded at once and are
// cheap; dead and match-free positions have lenA = lenB = 0. Only every WEIGHT_STRIDE-th position
// is read (see WEIGHT_STRIDE). The order only steers timing: the passes' output is the same for
// any order (blocks are independent), and the order itself is deterministic.
//
// Consts injected by the host: SCHED_HDR, WEIGHT_STRIDE (plus the block constants).

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> best: array<u32>;
@group(0) @binding(2) var<storage, read_write> sched: array<u32>;

fn n_blocks() -> u32 { return (arrayLength(&sched) - SCHED_HDR) / 2u; }

var<workgroup> wsum: atomic<u32>;

// One workgroup of 256 per block: its weight into sched[SCHED_HDR + b].
@compute @workgroup_size(256)
fn main_weight(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.x;
    let cb = 2u * b * BLOCK_SIZE;
    var c = 0u;
    // Terminates: p rises by 256 * WEIGHT_STRIDE to BLOCK_SIZE.
    for (var p = lid * WEIGHT_STRIDE; p < BLOCK_SIZE; p += 256u * WEIGHT_STRIDE) {
        let w0 = best[cb + 2u * p];
        let m = max((w0 >> 16u) & 0xFFu, w0 >> 24u);
        c += select(0u, 1u, m >= 3u && m <= 32u);
    }
    atomicAdd(&wsum, c);
    workgroupBarrier();
    if (lid == 0u && b < n_blocks()) { sched[SCHED_HDR + b] = atomicLoad(&wsum); }
}

var<workgroup> tile: array<u32, 256>;

// Rank sort, one invocation per block: block i goes to slot #{j : w_j > w_i, or w_j == w_i and
// j < i} (a permutation: the ranks are distinct). O(n^2) compares from workgroup tiles, a few
// microseconds for thousands of blocks.
@compute @workgroup_size(256)
fn main_order(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n = n_blocks();
    let i = wid.x * 256u + lid;
    let wi = sched[SCHED_HDR + min(i, n - 1u)];
    var r = 0u;
    // Terminates: t rises by 256 to n.
    for (var t = 0u; t < n; t += 256u) {
        workgroupBarrier();
        tile[lid] = sched[SCHED_HDR + min(t + lid, n - 1u)];
        workgroupBarrier();
        let m = min(256u, n - t);
        // Terminates: k rises to m.
        for (var k = 0u; k < m; k += 1u) {
            let wj = tile[k];
            r += select(0u, 1u, wj > wi || (wj == wi && t + k < i));
        }
    }
    if (i < n) { sched[SCHED_HDR + n + r] = i; }
}
