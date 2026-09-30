// Frame packing: copies each block's frame (K4 output, FRAME_WORDS words per block) into the
// pipeline's mappable readback buffer, contiguous, so the host reads back only the frames' bytes.
// Constants prepended by the host: FRAME_WORDS (words per block in `frames`, a multiple of 4)
// and PACK_BASE (offset of the packed frames region in `packed`, in words, a multiple of 4).
//
// Layout of `packed` (as u32 words): word b = frame_len[b]; from PACK_BASE on, the frames in block
// order, each starting on a 16-byte boundary: block b's frame starts
// 16 * sum_{j<b} ceil(frame_len[j] / 16) bytes after PACK_BASE. The host recomputes those offsets
// from the lengths. All stores are 16 bytes wide (the buffer may live in host memory).
//
// One workgroup per block: it sums the 16-byte chunk counts of the frames before it (at most
// batch-size loads, spread over the workgroup) and copies its own frame. No cross-workgroup
// communication, so the result does not depend on scheduling.

@group(0) @binding(0) var<storage, read> frame_len: array<u32>;
@group(0) @binding(1) var<storage, read> frames: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> packed: array<vec4<u32>>;

const WG: u32 = 256u;
var<workgroup> part: array<u32, WG>;

fn len_at(j: u32, n: u32) -> u32 {
    if (j < n) {
        return frame_len[j];
    }
    return 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let b = wg.x;
    let n = arrayLength(&frame_len);
    var s = 0u;
    for (var j = t; j < b; j += WG) {
        s += (frame_len[j] + 15u) >> 4u;
    }
    part[t] = s;
    workgroupBarrier();
    for (var h = WG / 2u; h > 0u; h >>= 1u) {
        if (t < h) {
            part[t] += part[t + h];
        }
        workgroupBarrier();
    }
    // Lengths, four per 16-byte store: block 4k's workgroup writes words 4k..4k+3.
    if (t == 0u && (b & 3u) == 0u) {
        packed[b >> 2u] = vec4<u32>(len_at(b, n), len_at(b + 1u, n), len_at(b + 2u, n), len_at(b + 3u, n));
    }
    let dst = (PACK_BASE >> 2u) + part[0];
    // K4 never reports more than FRAME_STRIDE bytes; the clamp keeps a bad length from reading
    // into the next block's frame (the host rejects it anyway).
    let chunks = min((frame_len[b] + 15u) >> 4u, FRAME_WORDS >> 2u);
    let src = b * (FRAME_WORDS >> 2u);
    for (var k = t; k < chunks; k += WG) {
        packed[dst + k] = frames[src + k];
    }
}
