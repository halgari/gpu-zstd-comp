// K3opt's block schedule (M6 A4, a07): the persistent K3opt passes (k3_opt.wgsl main_opt_persist)
// take their blocks heaviest first. Recorded once per batch, after K2opt and before the first pass,
// on the candidate words K2opt left in `best` (no pass reads the schedule's words but K3opt).
//
// `sched` (sizing::sched_bytes, bound to exactly SCHED_HDR + 3 n words for an n-block batch):
//   [0]                       the pass's block counter (cleared before every persistent pass);
//   [1 .. SCHED_HDR)          unused;
//   [SCHED_HDR + b]           block b's weight (main_weight);
//   [SCHED_HDR + n + i]       the i-th block to run (main_scatter): blocks by descending weight,
//                             equal weights by ascending block id;
//   [SCHED_HDR + 2 n + b]     block b's place in that order (main_rank).
//
// Weight (a07's cost proxy, Spearman 0.925 against the block's K3opt time): the block's positions
// whose longest candidate (max(lenA, lenB)) is 3..32. Longer matches are encoded at once and are
// cheap; dead and match-free positions have lenA = lenB = 0. Only runs of WEIGHT_RUN positions,
// one every WEIGHT_STRIDE positions, are read (see k3opt::WEIGHT_STRIDE: a full scan of the
// candidate words cost about 1.4 % of opt16's K3 time). The order only steers timing: the passes' output is the same for any order (blocks are
// independent), and the order itself is deterministic.
//
// Dispatches, in order (one compute pass): main_weight (n, 1, 1), main_rank (ceil(n / 256),
// ceil(n / 256), 1), main_scatter (ceil(n / 256), 1, 1).
//
// Consts injected by the host: SCHED_HDR, WEIGHT_RUN, WEIGHT_STRIDE (plus the block constants).

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> best: array<u32>;
@group(0) @binding(2) var<storage, read_write> sched: array<atomic<u32>>;

fn n_blocks() -> u32 { return (arrayLength(&sched) - SCHED_HDR) / 3u; }

var<workgroup> wsum: atomic<u32>;

// One workgroup of 256 per block: its weight, and its rank cleared for main_rank.
@compute @workgroup_size(256)
fn main_weight(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.x;
    let cb = 2u * b * BLOCK_SIZE;
    var c = 0u;
    // Terminates: r rises by 256 * WEIGHT_STRIDE to BLOCK_SIZE, j to WEIGHT_RUN.
    for (var r = lid * WEIGHT_STRIDE; r < BLOCK_SIZE; r += 256u * WEIGHT_STRIDE) {
        for (var j = 0u; j < WEIGHT_RUN; j += 1u) {
            let p = r + j;
            let w0 = best[cb + 2u * min(p, BLOCK_SIZE - 1u)];
            let m = max((w0 >> 16u) & 0xFFu, w0 >> 24u);
            c += select(0u, 1u, p < BLOCK_SIZE && m >= 3u && m <= 32u);
        }
    }
    atomicAdd(&wsum, c);
    workgroupBarrier();
    let n = n_blocks();
    if (lid == 0u && b < n) {
        atomicStore(&sched[SCHED_HDR + b], atomicLoad(&wsum));
        atomicStore(&sched[SCHED_HDR + 2u * n + b], 0u);
    }
}

var<workgroup> tile: array<u32, 256>;

// Rank sort, tile against tile: invocation i of tile wid.x counts the blocks j of tile wid.y
// that go before it (w_j > w_i, or w_j == w_i and j < i) into its rank. The ranks are distinct,
// so they form a permutation.
@compute @workgroup_size(256)
fn main_rank(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n = n_blocks();
    let i = wid.x * 256u + lid;
    let t = wid.y * 256u;
    tile[lid] = atomicLoad(&sched[SCHED_HDR + min(t + lid, n - 1u)]);
    workgroupBarrier();
    if (i >= n) { return; }
    let wi = atomicLoad(&sched[SCHED_HDR + i]);
    let m = min(256u, n - t);
    var r = 0u;
    // Terminates: k rises to m.
    for (var k = 0u; k < m; k += 1u) {
        let wj = tile[k];
        r += select(0u, 1u, wj > wi || (wj == wi && t + k < i));
    }
    if (r > 0u) { atomicAdd(&sched[SCHED_HDR + 2u * n + i], r); }
}

// The order: block i at its rank.
@compute @workgroup_size(256)
fn main_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = n_blocks();
    let i = gid.x;
    if (i < n) { atomicStore(&sched[SCHED_HDR + n + atomicLoad(&sched[SCHED_HDR + 2u * n + i])], i); }
}
