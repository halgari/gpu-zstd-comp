// Shared byte-load and hash helpers, mirroring gzc_core::hash bit-exactly.
// Prepended (after the generated constants) to every kernel. Each kernel declares
//   @group(0) @binding(0) var<storage, read> data: array<u32>;
// which these helpers reference (WGSL module-scope declarations are order-independent).

// Word index of block b in the packed data buffer.
fn block_base(b: u32) -> u32 { return b * (BLOCK_SIZE / 4u); }

// Unaligned little-endian u32 load at byte offset byte_off of the block at word base.
// The packed buffer carries one trailing zero word, so data[w + 1] stays in bounds.
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
