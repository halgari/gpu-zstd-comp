//! Long/short match-finding hashes and predecessor-chain computation.
use crate::config::{BLOCK_SIZE, HASH_BITS, HASHED_POSITIONS, NO_POS};
use crate::params::{Hashes, MatchParams};

pub fn read_u32(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}

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

/// pred[p] = most recent q < p with hash(q) == hash(p), else NO_POS. len == BLOCK_SIZE.
pub fn compute_preds<F: Fn(&[u8], usize) -> u32>(block: &[u8], hash: F) -> Vec<u32> {
    assert_eq!(block.len(), BLOCK_SIZE);
    let mut head = vec![NO_POS; 1 << HASH_BITS];
    let mut pred = vec![NO_POS; BLOCK_SIZE];
    for p in 0..HASHED_POSITIONS {
        let h = hash(block, p) as usize;
        pred[p] = head[h];
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
        for p in 0..HASHED_POSITIONS {
            if pl[p] != NO_POS {
                assert!((pl[p] as usize) < p);
                assert_eq!(hash_long(&b, pl[p] as usize), hash_long(&b, p));
            }
        }
        assert!(pl[HASHED_POSITIONS..].iter().all(|&x| x == NO_POS));
    }

    #[test]
    fn preds_are_most_recent() {
        let b = crate::synth::zeros(crate::config::BLOCK_SIZE);
        let ps = compute_preds(&b, hash_short);
        assert_eq!(ps[0], NO_POS);
        for p in 1..HASHED_POSITIONS {
            assert_eq!(ps[p], (p - 1) as u32);
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
