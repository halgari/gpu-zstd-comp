// K3t: cuts the parse of each partial block (a file's last block, zero-padded to BLOCK_SIZE) to
// the block's real length, so its frame holds no byte past it. == gzc_core::seq::truncate_output:
// sequences that end by len are kept; one whose match crosses len keeps its match cut to end
// there when at least ZSTD_MIN_MATCH bytes of it are left, else it and every later one are
// dropped; counts become (kept sequences, len - their match bytes). Runs between K3 and K5, and
// only for batches holding a partial block; full blocks (len == BLOCK_SIZE) are left alone.
//
// One workgroup of 256 threads per block. Each thread sums the bytes and the match bytes of its
// chunk of ceil(n_seq / 256) sequences; thread 0 then skips the chunks that end by len and walks
// only the chunk the cut falls in (a sequential walk of all n_seq sequences cost 0.75 ms per
// batch on an RTX 5090, 2.4 % of lvl9s12seg's pipeline time).
// The host binds exactly n_blocks words of `lens`, so arrayLength(&lens) is the batch size.
// Prepended by the host: MAX_SEQS (and common.wgsl, whose helpers name `data`).

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> lens: array<u32>;
@group(0) @binding(2) var<storage, read_write> seqs: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;

const WG: u32 = 256u;
const ZSTD_MIN_MATCH: u32 = 3u;

// Per chunk: the bytes its sequences cover (literals + matches), and their match bytes.
var<workgroup> chunk_bytes: array<u32, WG>;
var<workgroup> chunk_match: array<u32, WG>;
var<workgroup> n_seq_wg: u32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.x;
    if (b >= arrayLength(&lens)) { return; }
    let len = lens[b];
    if (len >= BLOCK_SIZE) { return; }
    // Thread 0 alone reads counts[b] (it alone overwrites it below) and shares it.
    if (lid == 0u) { n_seq_wg = min(counts[2u * b], MAX_SEQS); }
    let n_seq = workgroupUniformLoad(&n_seq_wg);
    let sbase = b * MAX_SEQS * 3u;
    let per = (n_seq + WG - 1u) / WG;
    let lo = min(lid * per, n_seq);
    let hi = min(lo + per, n_seq);
    var bytes = 0u;
    var matched = 0u;
    for (var i = lo; i < hi; i++) {
        let s = sbase + 3u * i;
        bytes += seqs[s] + seqs[s + 1u];
        matched += seqs[s + 1u];
    }
    chunk_bytes[lid] = bytes;
    chunk_match[lid] = matched;
    // Thread 0's store into seqs below comes after every thread's loads above.
    storageBarrier();
    workgroupBarrier();
    if (lid != 0u) { return; }

    // Whole chunks that end by len: all their sequences are kept. (An empty chunk past n_seq is
    // skipped too; once pos == len every later non-empty chunk ends past len.)
    var pos = 0u;
    var match_bytes = 0u;
    var c = 0u;
    while (c < WG && pos + chunk_bytes[c] <= len) {
        pos += chunk_bytes[c];
        match_bytes += chunk_match[c];
        c += 1u;
    }
    var kept = min(c * per, n_seq);
    // The chunk the cut falls in. Terminates: i counts up to n_seq <= MAX_SEQS (and breaks within
    // `per` sequences, at the one that ends past len).
    for (var i = kept; i < n_seq; i++) {
        let s = sbase + 3u * i;
        let start = pos + seqs[s];
        let end = start + seqs[s + 1u];
        var ml = seqs[s + 1u];
        if (end > len) {
            if (start + ZSTD_MIN_MATCH > len) { break; }
            ml = len - start;
            seqs[s + 1u] = ml;
        }
        kept += 1u;
        match_bytes += ml;
        pos = start + ml;
        if (pos == len) { break; }
    }
    counts[2u * b] = kept;
    counts[2u * b + 1u] = len - match_bytes;
}
