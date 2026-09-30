// Shared byte-load and hash helpers, mirroring gzc_core::hash bit-exactly.
// Prepended (after the generated constants) to every kernel. Each kernel declares
//   @group(0) @binding(0) var<storage, read> data: array<u32>;
// which these helpers reference (WGSL module-scope declarations are order-independent).

// Word index of block b in the packed data buffer.
fn block_base(b: u32) -> u32 { return b * (BLOCK_SIZE / 4u); }

// Unaligned little-endian u32 load at byte offset byte_off of the block at word base.
// Precondition: byte_off + 4 <= BLOCK_SIZE for all four bytes to come from this block.
// There is NO upper-bound check: for byte_off > BLOCK_SIZE - 4 the high bytes come from the
// next block (or from the packed buffer's trailing zero word, which keeps data[w + 1] in
// bounds for the last block). Callers must either respect the precondition or make sure the
// out-of-block bytes cannot influence their result.
fn load_u32_at(base: u32, byte_off: u32) -> u32 {
    let w = base + (byte_off >> 2u);
    let sh = (byte_off & 3u) * 8u;
    if (sh == 0u) { return data[w]; }
    return (data[w] >> sh) | (data[w + 1u] << (32u - sh));
}

fn load_byte(base: u32, byte_off: u32) -> u32 {
    return (data[base + (byte_off >> 2u)] >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

fn mix(lo: u32, hi: u32) -> u32 {
    return ((lo * 0x9E3779B1u) ^ (hi * 0x85EBCA77u)) * 0xC2B2AE3Du >> (32u - HASH_BITS);
}

fn hash_long(base: u32, p: u32) -> u32 { return mix(load_u32_at(base, p), load_u32_at(base, p + 4u)); }
fn hash_short(base: u32, p: u32) -> u32 { return mix(load_u32_at(base, p), load_byte(base, p + 4u)); }

// Length of the common prefix of block[p..] and block[q..] for q < p, bounded by
// min(BLOCK_SIZE - p, cap): == gzc_core::reference::match_len with cap = 0xFFFFFFFFu, and
// == match_len_capped with cap = MATCH_SEARCH_CAP. Compares 4 bytes at a time only while
// n + 4 <= max, so every load_u32_at stays inside the block (q + n + 4 <= p + n + 4
// <= BLOCK_SIZE); the tail is compared byte by byte.
fn match_len(base: u32, p: u32, q: u32, cap: u32) -> u32 {
    let max = min(BLOCK_SIZE - p, cap);
    var n = 0u;
    loop {
        if (n + 4u > max) { break; }
        let x = load_u32_at(base, p + n) ^ load_u32_at(base, q + n);
        if (x != 0u) { return n + (countTrailingZeros(x) >> 3u); }
        n += 4u;
    }
    loop {
        if (n >= max || load_byte(base, p + n) != load_byte(base, q + n)) { break; }
        n += 1u;
    }
    return n;
}
