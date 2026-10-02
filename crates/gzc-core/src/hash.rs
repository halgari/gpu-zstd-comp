//! Long/short match-finding hashes and predecessor-chain computation.
use crate::config::{BLOCK_SIZE, HASH_BITS, HASHED_POSITIONS, NO_POS};
use crate::params::{Hashes, MatchParams};

pub fn read_u32(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}

/// Keep in mind when changing: `k2_opt.wgsl`'s `ub = 2` skip on an h4 fingerprint mismatch
/// (entries on the `Opt3` h4 chain whose first 4 bytes differ share < 3 bytes) relies on
/// `hash_width(.., 4) = mix(lo, 0)` being injective in byte 3 for fixed bytes 0..3. That holds
/// because both multipliers are odd, `hi = 0`, and the 16-bit key keeps bits 24..31 of the
/// product, where byte 3 enters as `(byte3 * K) << 24`, a bijection. Checked by
/// `gzc-gpu/tests/cands.rs::h4_hash_is_injective_in_byte_3`.
fn mix(lo: u32, hi: u32) -> u32 {
    (lo.wrapping_mul(0x9E37_79B1) ^ hi.wrapping_mul(0x85EB_CA77)).wrapping_mul(0xC2B2_AE3D) >> (32 - HASH_BITS)
}

pub fn hash_long(b: &[u8], p: usize) -> u32 {
    mix(read_u32(b, p), read_u32(b, p + 4))
}

pub fn hash_short(b: &[u8], p: usize) -> u32 {
    mix(read_u32(b, p), b[p + 4] as u32)
}

/// Hash of `width` bytes at `p` (`width` in `4..=8`):
/// `mix(read_u32(b, p), read_u32(b, p + 4) & mask(width - 4))`, where `mask(0) = 0`,
/// `mask(k) = (1 << (8 * k)) - 1` for `k < 4`, and `mask(4) = u32::MAX`.
///
/// Invariants (mirrored bit-exactly in WGSL by Task 3): `hash_width(b, p, 5) == hash_short(b, p)`
/// and `hash_width(b, p, 8) == hash_long(b, p)`. u32 wrapping arithmetic only, so the GPU port
/// matches exactly.
pub fn hash_width(b: &[u8], p: usize, width: u32) -> u32 {
    let k = width.wrapping_sub(4);
    let mask: u32 = if k == 0 {
        0
    } else if k < 4 {
        (1u32 << (8 * k)) - 1
    } else {
        u32::MAX
    };
    mix(read_u32(b, p), read_u32(b, p + 4) & mask)
}

/// The low `k` bytes of a little-endian word: `mask(0) = 0`, `mask(k) = (1 << 8k) - 1` for
/// `k < 4`, `mask(k) = u32::MAX` for `k >= 4`.
fn byte_mask(k: u32) -> u32 {
    if k >= 4 { u32::MAX } else { (1u32 << (8 * k)) - 1 }
}

/// The 16-bit (`HASH_BITS`) key of a sparse candidate chain (`params::SparseChain`, M6 a06/r3):
/// the `width` bytes at `p`, `5 <= width <= 12`. With `lo`, `hi`, `h2` the little-endian words
/// at `p`, `p + 4`, `p + 8`:
///
/// `hash_sparse = ((lo * 0x9E3779B1) ^ ((hi & mask(width - 4)) * 0x85EBCA77)
///                 ^ ((h2 & mask(width - 8)) * 0x27D4EB2F)) * 0xC2B2AE3D >> 16`
///
/// in wrapping u32 arithmetic, `mask` as in `hash_width`, and the `h2` term 0 (not read) for
/// `width <= 8`, where this equals `hash_width(b, p, width)`. The `S3_CHAINS` keys, with
/// `K1 = 0x9E3779B1`, `K2 = 0x85EBCA77`, `K3 = 0x27D4EB2F`, `K4 = 0xC2B2AE3D`:
/// - width 6: `((lo * K1) ^ ((hi & 0xFFFF) * K2)) * K4 >> 16`, i.e. `hash_width(b, p, 6)`
///   (reads bytes `p..p + 8`);
/// - width 10: `((lo * K1) ^ (hi * K2) ^ ((h2 & 0xFFFF) * K3)) * K4 >> 16` (reads `p..p + 12`);
/// - width 12: `((lo * K1) ^ (hi * K2) ^ (h2 * K3)) * K4 >> 16` (reads `p..p + 12`).
///
/// Width 10 is a06's h10 GPU prototype hash.
pub fn hash_sparse(b: &[u8], p: usize, width: u32) -> u32 {
    debug_assert!((5..=12).contains(&width), "hash_sparse width {width}");
    let lo = read_u32(b, p);
    let hi = read_u32(b, p + 4) & byte_mask(width - 4);
    let h2 = if width > 8 { read_u32(b, p + 8) & byte_mask(width - 8) } else { 0 };
    (lo.wrapping_mul(0x9E37_79B1) ^ hi.wrapping_mul(0x85EB_CA77) ^ h2.wrapping_mul(0x27D4_EB2F)).wrapping_mul(0xC2B2_AE3D) >> (32 - HASH_BITS)
}

/// zstd's `ZSTD_hash3Ptr` with `hBits = HASH_BITS` (16, zstd's hashLog3 at 64 KiB): the 3 bytes at
/// `p`, `((MEM_readLE32(p) << 8) * 506832829) >> (32 - 16)` in wrapping u32 arithmetic. Reads 4
/// bytes (`p + 4 <= BLOCK_SIZE`); byte `p + 3` is shifted out. The `Opt3` short chain's key.
pub fn hash3(b: &[u8], p: usize) -> u32 {
    (read_u32(b, p) << 8).wrapping_mul(506_832_829) >> (32 - HASH_BITS)
}

/// The top `bits` bits of a `HASH_BITS`-bit hash `h`: the match finder's key
/// (`MatchParams::hash_bits`). `key(h, HASH_BITS) == h`.
pub fn key(h: u32, bits: u32) -> u32 {
    h >> (HASH_BITS - bits)
}

/// The bucket-sorted candidate array of a Single-hash `params` (speed2 E2): every hashed position
/// `p < HASHED_POSITIONS`, ordered by key (`key(hash_width(block, p, min_match), hash_bits)`)
/// ascending, positions ascending inside a key (a stable counting sort by key). Returns
/// `(sorted, rank)`: `sorted[s]` is the position in slot `s` (`HASHED_POSITIONS` slots, the rest
/// of the `BLOCK_SIZE`-long array is `NO_POS`), and `rank[p]` is p's slot (`NO_POS` for
/// `p >= HASHED_POSITIONS`). p's hash-chain predecessor is `sorted[rank[p] - 1]` when that slot is
/// in p's bucket, so the chain of depth d is the (up to) d entries just below p's slot.
pub fn bucket_sort(block: &[u8], params: &MatchParams) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(block.len(), BLOCK_SIZE);
    assert_eq!(params.hashes, Hashes::Single, "bucket_sort: Single hash only");
    let keys: Vec<u32> =
        (0..HASHED_POSITIONS).map(|p| key(hash_width(block, p, params.min_match), params.hash_bits)).collect();
    let mut start = vec![0u32; (1 << params.hash_bits) + 1];
    for &k in &keys {
        start[k as usize + 1] += 1;
    }
    for i in 1..start.len() {
        start[i] += start[i - 1];
    }
    let mut sorted = vec![NO_POS; BLOCK_SIZE];
    let mut rank = vec![NO_POS; BLOCK_SIZE];
    for (p, &k) in keys.iter().enumerate() {
        let s = start[k as usize];
        start[k as usize] += 1;
        sorted[s as usize] = p as u32;
        rank[p] = s;
    }
    (sorted, rank)
}

/// `pred[p]` = most recent q < p with hash(q) == hash(p), else NO_POS. len == BLOCK_SIZE.
pub fn compute_preds<F: Fn(&[u8], usize) -> u32>(block: &[u8], hash: F) -> Vec<u32> {
    assert_eq!(block.len(), BLOCK_SIZE);
    let mut head = vec![NO_POS; 1 << HASH_BITS];
    let mut pred = vec![NO_POS; BLOCK_SIZE];
    for (p, pr) in pred.iter_mut().enumerate().take(HASHED_POSITIONS) {
        let h = hash(block, p) as usize;
        *pr = head[h];
        head[h] = p as u32;
    }
    pred
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HASHED_POSITIONS, NO_POS};

    #[test]
    fn preds_point_back_to_equal_hash() {
        let b = crate::synth::text(1, crate::config::BLOCK_SIZE);
        let pl = compute_preds(&b, hash_long);
        for (p, &q) in pl.iter().enumerate().take(HASHED_POSITIONS) {
            if q != NO_POS {
                assert!((q as usize) < p);
                assert_eq!(hash_long(&b, q as usize), hash_long(&b, p));
            }
        }
        assert!(pl[HASHED_POSITIONS..].iter().all(|&x| x == NO_POS));
    }

    #[test]
    fn preds_are_most_recent() {
        let b = crate::synth::zeros(crate::config::BLOCK_SIZE);
        let ps = compute_preds(&b, hash_short);
        assert_eq!(ps[0], NO_POS);
        for (p, &q) in ps.iter().enumerate().take(HASHED_POSITIONS).skip(1) {
            assert_eq!(q, (p - 1) as u32);
        }
    }

    #[test]
    fn hash_width_matches_long_and_short() {
        for b in [crate::synth::text(1, crate::config::BLOCK_SIZE), crate::synth::random(2, crate::config::BLOCK_SIZE)] {
            for p in 0..HASHED_POSITIONS {
                assert_eq!(hash_width(&b, p, 5), hash_short(&b, p), "p={p}");
                assert_eq!(hash_width(&b, p, 8), hash_long(&b, p), "p={p}");
            }
        }
    }

    #[test]
    fn hash3_ignores_byte_3_and_matches_zstd() {
        let mut b = crate::synth::random(4, crate::config::BLOCK_SIZE);
        for p in (0..HASHED_POSITIONS).step_by(89) {
            let h0 = hash3(&b, p);
            assert!(h0 < 1 << HASH_BITS);
            let original = b[p + 3];
            for delta in 1u8..=255 {
                b[p + 3] = original.wrapping_add(delta);
                assert_eq!(hash3(&b, p), h0, "p={p}");
            }
            b[p + 3] = original;
        }
        // ((0x00636261 << 8) * 506832829) >> 16, computed by hand for "abc".
        let mut v = vec![0u8; 8];
        v[..4].copy_from_slice(b"abcX");
        assert_eq!(hash3(&v, 0), (0x6362_6100u32.wrapping_mul(506_832_829)) >> 16);
    }

    /// `hash_sparse` is `hash_width` up to 8 bytes, depends on exactly its `width` bytes, and
    /// matches its formula for the S3 widths on hand-built words.
    #[test]
    fn hash_sparse_formula_and_width() {
        let b = crate::synth::random(5, crate::config::BLOCK_SIZE);
        for p in (0..crate::config::BLOCK_SIZE - 12).step_by(61) {
            for w in 5..=8 {
                assert_eq!(hash_sparse(&b, p, w), hash_width(&b, p, w), "p={p} w={w}");
            }
            for w in 5..=12u32 {
                let h = hash_sparse(&b, p, w);
                assert!(h < 1 << HASH_BITS);
                let mut c = b.clone();
                for i in w as usize..12 {
                    c[p + i] ^= 0x5A; // bytes past the width do not matter
                }
                assert_eq!(hash_sparse(&c, p, w), h, "p={p} w={w}");
            }
        }
        let v: Vec<u8> = (1..=16).collect();
        let (lo, hi, h2) = (0x0403_0201u32, 0x0807_0605u32, 0x0C0B_0A09u32);
        let f = |hi: u32, h2: u32| (lo.wrapping_mul(0x9E37_79B1) ^ hi.wrapping_mul(0x85EB_CA77) ^ h2.wrapping_mul(0x27D4_EB2F)).wrapping_mul(0xC2B2_AE3D) >> 16;
        assert_eq!(hash_sparse(&v, 0, 6), f(hi & 0xFFFF, 0));
        assert_eq!(hash_sparse(&v, 0, 10), f(hi, h2 & 0xFFFF));
        assert_eq!(hash_sparse(&v, 0, 12), f(hi, h2));
    }

    #[test]
    fn hash_width_4_ignores_byte_4() {
        let mut b = crate::synth::random(3, crate::config::BLOCK_SIZE);
        for p in (0..HASHED_POSITIONS).step_by(97) {
            let h0 = hash_width(&b, p, 4);
            let original = b[p + 4];
            for delta in 1u8..=255 {
                b[p + 4] = original.wrapping_add(delta);
                assert_eq!(hash_width(&b, p, 4), h0, "p={p} byte4={}", b[p + 4]);
            }
            b[p + 4] = original;
        }
    }
}
