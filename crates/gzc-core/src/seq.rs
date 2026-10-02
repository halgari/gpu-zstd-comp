//! Sequence and repeat-offset model shared by CPU and GPU parsers.

/// One zstd sequence: `lit_len` literals, then a match of `match_len` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Sequence {
    /// Literal bytes before the match.
    pub lit_len: u32,
    /// Match length in bytes, at least `ZSTD_MIN_MATCH`.
    pub match_len: u32,
    /// zstd's offBase: 1..=3 are repeat codes, larger values are `offset + 3`
    /// (`off_base_for`).
    pub off_base: u32,
}

/// A block's parse: its sequences, and every literal byte in order. The literals past the last
/// sequence's are the block's trailing literals.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockOutput {
    /// The sequences, in block order.
    pub sequences: Vec<Sequence>,
    /// The literal bytes of every sequence, then the trailing literals.
    pub literals: Vec<u8>,
}

/// The repeat-offset history, most recent first.
pub type Reps = [u32; 3];
/// The history at the start of a frame (RFC 8878 §3.1.1.5).
pub const INITIAL_REPS: Reps = [1, 4, 8];

/// zstd offBase for a match at `offset` preceded by `lit_len` literals (RFC 8878 §3.1.1.5).
pub fn off_base_for(offset: u32, lit_len: u32, reps: &Reps) -> u32 {
    if lit_len > 0 {
        if offset == reps[0] {
            return 1;
        }
        if offset == reps[1] {
            return 2;
        }
        if offset == reps[2] {
            return 3;
        }
    } else {
        if offset == reps[1] {
            return 1;
        }
        if offset == reps[2] {
            return 2;
        }
        if reps[0] > 1 && offset == reps[0] - 1 {
            return 3;
        }
    }
    offset + 3
}

/// Resolve `off_base` to an offset and update the repeat history exactly as a decoder does.
pub fn apply_off_base(reps: &mut Reps, off_base: u32, lit_len: u32) -> u32 {
    if off_base > 3 {
        let off = off_base - 3;
        *reps = [off, reps[0], reps[1]];
        return off;
    }
    let idx = off_base - 1 + (lit_len == 0) as u32;
    let off = match idx {
        0 => reps[0],
        1 => reps[1],
        2 => reps[2],
        _ => reps[0].wrapping_sub(1),
    };
    if idx > 0 {
        if idx > 1 {
            reps[2] = reps[1];
        }
        reps[1] = reps[0];
        reps[0] = off;
    }
    off
}

/// zstd's shortest match (RFC 8878: Match_Length codes start at 3).
pub const ZSTD_MIN_MATCH: u32 = 3;

/// Cuts `out`, the parse of the zero-padded block `block` (BLOCK_SIZE bytes), to the parse of its
/// first `len` bytes (`1..=BLOCK_SIZE`, `Block::real_len`), so the frame holds no byte past them.
/// Sequences that end by `len` are kept; one whose match crosses `len` keeps its match cut to
/// end there when at least `ZSTD_MIN_MATCH` bytes of it are left, else it is dropped with every
/// later one; the bytes from the last kept sequence to `len` are the trailing literals. Offsets
/// and repeat codes of kept sequences are unchanged (the history before them is). The identity
/// at `len == BLOCK_SIZE`. K3t (`k3_trunc.wgsl`) is the GPU twin.
pub fn truncate_output(out: &BlockOutput, block: &[u8], len: usize) -> BlockOutput {
    assert!((1..=block.len()).contains(&len), "truncate_output: len {len} not in 1..={}", block.len());
    let (mut pos, mut lits) = (0usize, 0usize);
    let mut sequences = Vec::with_capacity(out.sequences.len());
    for s in &out.sequences {
        let start = pos + s.lit_len as usize;
        let end = start + s.match_len as usize;
        if end <= len {
            sequences.push(*s);
        } else if start + ZSTD_MIN_MATCH as usize <= len {
            sequences.push(Sequence { match_len: (len - start) as u32, ..*s });
        } else {
            break;
        }
        lits += s.lit_len as usize;
        pos = end.min(len);
        if pos == len {
            break;
        }
    }
    let mut literals = out.literals[..lits].to_vec();
    literals.extend_from_slice(&block[pos..len]);
    BlockOutput { sequences, literals }
}

/// Decode a BlockOutput the way a zstd decoder would. Used as a test oracle.
pub fn reconstruct(out: &BlockOutput) -> Result<Vec<u8>, String> {
    let mut dst = Vec::new();
    let mut lit = 0usize;
    let mut reps = INITIAL_REPS;
    for (i, s) in out.sequences.iter().enumerate() {
        let ll = s.lit_len as usize;
        let lits = out.literals.get(lit..lit + ll).ok_or_else(|| format!("seq {i}: literals overrun"))?;
        dst.extend_from_slice(lits);
        lit += ll;
        if s.match_len < 3 {
            return Err(format!("seq {i}: match_len {} < 3", s.match_len));
        }
        if s.off_base == 0 {
            return Err(format!("seq {i}: off_base 0"));
        }
        let off = apply_off_base(&mut reps, s.off_base, s.lit_len) as usize;
        if off == 0 || off > dst.len() {
            return Err(format!("seq {i}: offset {off} with {} bytes decoded", dst.len()));
        }
        let start = dst.len() - off;
        for k in 0..s.match_len as usize {
            let b = dst[start + k];
            dst.push(b);
        }
    }
    dst.extend_from_slice(out.literals.get(lit..).ok_or("literals overrun at end")?);
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_off_base_explicit() {
        let mut r = INITIAL_REPS;
        assert_eq!(apply_off_base(&mut r, 100 + 3, 5), 100);
        assert_eq!(r, [100, 1, 4]);
    }

    #[test]
    fn apply_off_base_rep_with_literals() {
        let mut r = [10, 20, 30];
        assert_eq!(apply_off_base(&mut r, 1, 3), 10);
        assert_eq!(r, [10, 20, 30]);
        assert_eq!(apply_off_base(&mut r, 2, 3), 20);
        assert_eq!(r, [20, 10, 30]);
        assert_eq!(apply_off_base(&mut r, 3, 3), 30);
        assert_eq!(r, [30, 20, 10]);
    }

    #[test]
    fn apply_off_base_ll0_shifts() {
        let mut r = [10, 20, 30];
        assert_eq!(apply_off_base(&mut r, 1, 0), 20);
        assert_eq!(r, [20, 10, 30]);
        let mut r = [10, 20, 30];
        assert_eq!(apply_off_base(&mut r, 2, 0), 30);
        assert_eq!(r, [30, 10, 20]);
        let mut r = [10, 20, 30];
        assert_eq!(apply_off_base(&mut r, 3, 0), 9);
        assert_eq!(r, [9, 10, 20]);
    }

    #[test]
    fn off_base_for_inverts_apply() {
        // seeded walk: for many (offset, lit_len), mapping then applying yields the same offset
        let mut state = 12345u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut r = INITIAL_REPS;
        for _ in 0..10_000 {
            let offset = match next() % 4 {
                0 => r[0],
                1 => r[1],
                2 => r[2],
                _ => 1 + next() % 5000,
            };
            let ll = next() % 3;
            let mut r2 = r;
            let ob = off_base_for(offset, ll, &r);
            assert_eq!(apply_off_base(&mut r2, ob, ll), offset);
            r = r2;
        }
    }

    #[test]
    fn reconstruct_overlapping_match() {
        let out = BlockOutput {
            sequences: vec![Sequence { lit_len: 2, match_len: 6, off_base: 2 + 3 }],
            literals: b"abXY".to_vec(),
        };
        assert_eq!(reconstruct(&out).unwrap(), b"ababababXY".to_vec());
    }

    #[test]
    fn reconstruct_rejects_bad_offset() {
        let out = BlockOutput {
            sequences: vec![Sequence { lit_len: 1, match_len: 3, off_base: 5 + 3 }],
            literals: b"a".to_vec(),
        };
        assert!(reconstruct(&out).is_err());
    }
}
