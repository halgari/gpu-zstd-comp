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

// == gzc_core::hash::hash_width: hash of `width` bytes at p (width in 4..=8), the second word
// masked to width - 4 bytes. mask(4) is spelled out: WGSL, like Rust, must not shift by 32.
fn hash_width(base: u32, p: u32, width: u32) -> u32 {
    let k = width - 4u;
    var mask = 0xFFFFFFFFu;
    if (k == 0u) {
        mask = 0u;
    } else if (k < 4u) {
        mask = (1u << (8u * k)) - 1u;
    }
    return mix(load_u32_at(base, p), load_u32_at(base, p + 4u) & mask);
}

// == gzc_core::hash::hash3 (zstd's ZSTD_hash3Ptr, 16 bits): the 3 bytes at p (byte p + 3 is
// shifted out). The Opt3 short chain's key.
fn hash3(base: u32, p: u32) -> u32 { return ((load_u32_at(base, p) << 8u) * 506832829u) >> (32u - HASH_BITS); }

// Length of the common prefix of block[p..] and block[q..] for q < p, bounded by
// min(BLOCK_SIZE - p, cap): == gzc_core::reference::match_len with cap = 0xFFFFFFFFu, and
// == match_len_capped with cap = SEARCH_CAP. Compares 4 bytes at a time only while
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

// K1's pred words, which K2 walks: bits 0..17 hold the predecessor (PRED_NONE = none; positions are
// below HASHED_POSITIONS < 2^17), bits 17..32 a fingerprint of the word's own position p, whose
// bytes lo = p..p+4 and hi = p+4..p+8: bits 17..24 are 7 bits of a hash of lo, bits 24..32 byte
// p + 4. Equal bytes give equal fingerprints, so K2 can rule out a candidate from the pred word it
// loads anyway: a differing lo field means len < 4, a differing byte field len <= 4.
const PRED_POS: u32 = 0x1FFFFu;
const PRED_NONE: u32 = 0x1FFFFu;
const PRED_FP_LO: u32 = 0x7Fu << 17u;
const_assert LOG2_BLOCK == 16u;
fn pred_fp(lo: u32, hi: u32) -> u32 {
    return (((lo * 0x85EBCA6Bu) >> 25u) << 17u) | ((hi & 0xFFu) << 24u);
}
// The Opt3 h3 chain's fingerprint (chain 1 of Opt3 params, `chains::pred_fp3`): bits 17..24
// are 7 bits of a hash of the 3 bytes p..p+3, bits 24..32 byte p + 3 (lo = the 4 bytes at p).
// A differing lo field means len < 3, a differing byte field len <= 3.
fn pred_fp3(lo: u32) -> u32 {
    return ((((lo & 0xFFFFFFu) * 0x85EBCA6Bu) >> 25u) << 17u) | (lo & 0xFF000000u);
}
// A pred word for predecessor pr (NO_POS = none) and fingerprint fp.
fn pred_word(pr: u32, fp: u32) -> u32 {
    return select(pr, PRED_NONE, pr == NO_POS) | fp;
}
