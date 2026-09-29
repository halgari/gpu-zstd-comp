//! zstd frame writer: header, block header, literals and sequences sections.
//!
//! Every block becomes one single-segment zstd frame holding exactly one zstd block
//! (RFC 8878 §3.1). Frame content size is always `BLOCK_SIZE`.

use xxhash_rust::xxh64::xxh64;

use crate::config::BLOCK_SIZE;
use crate::seq::BlockOutput;
use crate::seqenc::write_sequences_section_auto;

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameOptions {
    pub checksum: bool,
}

const MAGIC: u32 = 0xFD2F_B528;

const BLOCK_RAW: u32 = 0;
const BLOCK_RLE: u32 = 1;
const BLOCK_COMPRESSED: u32 = 2;

/// Raw_Literals_Block: literals section header (type 0) followed by the literal bytes.
pub fn write_literals_raw(lits: &[u8], out: &mut Vec<u8>) {
    let n = lits.len() as u32;
    assert!(n < 1 << 20, "too many literals: {n}");
    if n < 32 {
        out.push((n << 3) as u8);
    } else if n < 4096 {
        out.extend_from_slice(&(((n << 4) | 0b0100) as u16).to_le_bytes());
    } else {
        out.extend_from_slice(&((n << 4) | 0b1100).to_le_bytes()[..3]);
    }
    out.extend_from_slice(lits);
}

/// Magic number + Frame_Header_Descriptor + Frame_Content_Size (single segment, no dictionary).
pub fn frame_header(opts: FrameOptions) -> Vec<u8> {
    let mut h = MAGIC.to_le_bytes().to_vec();
    // FCS_Field_Size flag 1 (2 bytes, value - 256) covers 256..=65791; flag 2 is 4 bytes.
    let fcs_flag: u8 = if BLOCK_SIZE < 65536 + 256 { 1 } else { 2 };
    h.push((fcs_flag << 6) | (1 << 5) | ((opts.checksum as u8) << 2));
    if fcs_flag == 1 {
        h.extend_from_slice(&((BLOCK_SIZE - 256) as u16).to_le_bytes());
    } else {
        h.extend_from_slice(&(BLOCK_SIZE as u32).to_le_bytes());
    }
    h
}

/// Block_Header for the only (last) block of the frame.
fn block_header(block_type: u32, size: usize, out: &mut Vec<u8>) {
    debug_assert!(size < 1 << 21);
    let v = 1 | (block_type << 1) | ((size as u32) << 3);
    out.extend_from_slice(&v.to_le_bytes()[..3]);
}

/// Encode `block` (exactly `BLOCK_SIZE` bytes) as one zstd frame, using `out` as its parse.
pub fn write_frame(block: &[u8], out: &BlockOutput, opts: FrameOptions) -> Vec<u8> {
    assert_eq!(block.len(), BLOCK_SIZE, "frames always hold exactly one full block");
    let mut f = frame_header(opts);
    if block.iter().all(|&b| b == block[0]) {
        block_header(BLOCK_RLE, BLOCK_SIZE, &mut f);
        f.push(block[0]);
    } else {
        debug_assert_eq!(
            out.literals.len() as u64 + out.sequences.iter().map(|s| s.match_len as u64).sum::<u64>(),
            BLOCK_SIZE as u64,
            "BlockOutput does not cover the block"
        );
        let mut content = Vec::with_capacity(out.literals.len() + 16 + out.sequences.len() * 4);
        write_literals_raw(&out.literals, &mut content);
        write_sequences_section_auto(&out.sequences, &mut content);
        if content.len() < BLOCK_SIZE {
            block_header(BLOCK_COMPRESSED, content.len(), &mut f);
            f.extend_from_slice(&content);
        } else {
            block_header(BLOCK_RAW, BLOCK_SIZE, &mut f);
            f.extend_from_slice(block);
        }
    }
    if opts.checksum {
        f.extend_from_slice(&(xxh64(block, 0) as u32).to_le_bytes());
    }
    f
}

#[cfg(test)]
pub(crate) mod testutil {
    use crate::config::BLOCK_SIZE;
    use crate::seq::{BlockOutput, INITIAL_REPS, Sequence, apply_off_base, off_base_for};

    /// Build a (block, BlockOutput) pair from a script of (lit_len, offset, match_len), literals drawn from `lits`.
    pub(crate) fn scripted(script: &[(u32, u32, u32)], lits: &mut impl FnMut() -> u8) -> (Vec<u8>, BlockOutput) {
        let mut out = BlockOutput::default();
        let mut reps = INITIAL_REPS;
        let mut data: Vec<u8> = Vec::new();
        for &(ll, off, ml) in script {
            for _ in 0..ll {
                let b = lits();
                out.literals.push(b);
                data.push(b);
            }
            let ob = off_base_for(off, ll, &reps);
            apply_off_base(&mut reps, ob, ll);
            let start = data.len() - off as usize;
            for k in 0..ml as usize {
                let b = data[start + k];
                data.push(b);
            }
            out.sequences.push(Sequence { lit_len: ll, match_len: ml, off_base: ob });
        }
        while data.len() < BLOCK_SIZE {
            let b = lits();
            out.literals.push(b);
            data.push(b);
        }
        assert_eq!(data.len(), BLOCK_SIZE, "script overran the block");
        (data, out)
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::scripted;
    use super::*;
    use crate::seq::{INITIAL_REPS, apply_off_base, off_base_for, reconstruct};
    use crate::synth;

    const RAW: u8 = 0;
    const RLE: u8 = 1;
    const COMPRESSED: u8 = 2;

    /// Small deterministic PRNG for scripts and literal bytes.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    fn lit_source(seed: u64) -> impl FnMut() -> u8 {
        let mut r = Lcg(seed);
        move || r.next() as u8
    }

    fn block_type(frame: &[u8], opts: FrameOptions) -> u8 {
        let h = frame_header(opts).len();
        (frame[h] >> 1) & 3
    }

    /// Encode, check libzstd decodes it to `block`, and return the frame.
    fn roundtrip(block: &[u8], out: &BlockOutput, opts: FrameOptions) -> Vec<u8> {
        assert_eq!(reconstruct(out).unwrap(), block, "test input is not a valid parse");
        let frame = write_frame(block, out, opts);
        let dec = zstd::bulk::decompress(&frame, BLOCK_SIZE).expect("libzstd rejected frame");
        assert_eq!(dec.len(), BLOCK_SIZE);
        assert!(dec == block, "decoded block differs");
        frame
    }

    fn roundtrip_script(script: &[(u32, u32, u32)], seed: u64) -> (BlockOutput, Vec<u8>) {
        let (block, out) = scripted(script, &mut lit_source(seed));
        let frame = roundtrip(&block, &out, FrameOptions::default());
        (out, frame)
    }

    #[test]
    fn literals_raw_header_sizes() {
        let hdr = |n: usize| {
            let mut v = Vec::new();
            write_literals_raw(&vec![7u8; n], &mut v);
            assert_eq!(&v[v.len() - n..], &vec![7u8; n][..]);
            v[..v.len() - n].to_vec()
        };
        assert_eq!(hdr(0), vec![0]);
        assert_eq!(hdr(31), vec![31 << 3]);
        assert_eq!(hdr(32), ((32u16 << 4) | 0b0100).to_le_bytes().to_vec());
        assert_eq!(hdr(4095), ((4095u16 << 4) | 0b0100).to_le_bytes().to_vec());
        assert_eq!(hdr(4096), ((4096u32 << 4) | 0b1100).to_le_bytes()[..3].to_vec());
        assert_eq!(hdr(BLOCK_SIZE), (((BLOCK_SIZE as u32) << 4) | 0b1100).to_le_bytes()[..3].to_vec());
    }

    #[test]
    fn frame_roundtrip_no_sequences() {
        let block = synth::text(3, BLOCK_SIZE);
        let out = BlockOutput { sequences: vec![], literals: block.clone() };
        roundtrip(&block, &out, FrameOptions::default());
    }

    #[test]
    fn frame_roundtrip_simple() {
        let (_, frame) = roundtrip_script(&[(10, 10, 20), (5, 3, 100), (0, 7, 9), (40, 50, 300)], 1);
        assert_eq!(block_type(&frame, FrameOptions::default()), COMPRESSED);
    }

    #[test]
    fn frame_roundtrip_repcodes_ll0() {
        let bs = BLOCK_SIZE as u32;
        let script = [(4, 4, 4), (0, 8, 4), (0, 4, 4), (3, 7, 5), (0, 6, 6), (10, 10, bs / 4)];
        let (out, frame) = roundtrip_script(&script, 2);
        assert_eq!(block_type(&frame, FrameOptions::default()), COMPRESSED);
        for ob in 1..=3 {
            assert!(
                out.sequences.iter().any(|s| s.lit_len == 0 && s.off_base == ob),
                "no ll=0 sequence with off_base {ob}: {:?}",
                out.sequences
            );
        }
    }

    #[test]
    fn frame_roundtrip_large_codes() {
        let bs = BLOCK_SIZE as u32;
        // 128K: ll = 73728 (LL code 35), ml = 40960 (ML code 51).
        let ll = bs / 2 + bs / 16;
        let ml = bs / 4 + bs / 16;
        let (out, frame) = roundtrip_script(&[(ll, 1000, ml), (100, 3, bs / 16)], 3);
        assert_eq!(block_type(&frame, FrameOptions::default()), COMPRESSED);
        assert_eq!(out.sequences[0].lit_len, ll);
    }

    #[test]
    fn frame_roundtrip_max_offset() {
        let bs = BLOCK_SIZE as u32;
        // second match copies from position 0 at distance BLOCK_SIZE - 16 and ends at the block end
        let script = [(8, 8, bs / 2), (bs / 2 - 24, bs - 16, 16)];
        let (out, frame) = roundtrip_script(&script, 4);
        assert_eq!(block_type(&frame, FrameOptions::default()), COMPRESSED);
        assert_eq!(out.sequences[1].off_base, bs - 16 + 3);
        assert!(out.literals.len() == (8 + bs / 2 - 24) as usize);
    }

    #[test]
    fn match_to_block_end() {
        let bs = BLOCK_SIZE as u32;
        let (out, frame) = roundtrip_script(&[(10, 3, bs - 10)], 5);
        assert_eq!(block_type(&frame, FrameOptions::default()), COMPRESSED);
        assert_eq!(out.literals.len(), 10);
    }

    fn random_script(r: &mut Lcg, nseq: u32) -> (Vec<(u32, u32, u32)>, u32) {
        let bs = BLOCK_SIZE as u32;
        let ml_cap = [32, 500, bs / 8][r.below(3) as usize];
        let ll_cap = [16, 200, bs / 32][r.below(3) as usize];
        let mut script = Vec::new();
        let mut reps = INITIAL_REPS;
        let mut pos = 0u32;
        let mut total_ml = 0;
        for _ in 0..nseq {
            let mut ll = match r.below(4) {
                0 => 0,
                1 => r.below(16),
                _ => r.below(ll_cap),
            };
            if pos == 0 {
                ll = ll.max(1);
            }
            let ml = 3 + r.below(ml_cap);
            if pos + ll + ml > bs {
                break;
            }
            let avail = pos + ll;
            let cand = match r.below(5) {
                0 => reps[0],
                1 => reps[1],
                2 => reps[2],
                3 => reps[0].wrapping_sub(1),
                _ => 1 + r.below(avail),
            };
            let off = if (1..=avail).contains(&cand) { cand } else { 1 + r.below(avail) };
            let ob = off_base_for(off, ll, &reps);
            apply_off_base(&mut reps, ob, ll);
            script.push((ll, off, ml));
            pos += ll + ml;
            total_ml += ml;
        }
        (script, total_ml)
    }

    #[test]
    fn frame_roundtrip_many_random_scripts() {
        let mut r = Lcg(0x5eed);
        let mut compressed = 0;
        for i in 0..200 {
            let nseq = 1 + r.below(2000);
            let (script, total_ml) = random_script(&mut r, nseq);
            let (block, out) = scripted(&script, &mut lit_source(1000 + i));
            let frame = roundtrip(&block, &out, FrameOptions::default());
            let ty = block_type(&frame, FrameOptions::default());
            // literals header <= 3, sequences header <= 4, <= 66 bits per sequence + end mark
            if total_ml > 16 + 9 * script.len() as u32 {
                assert_eq!(ty, COMPRESSED, "script {i} should compress");
            }
            compressed += (ty == COMPRESSED) as u32;
        }
        assert!(compressed >= 150, "only {compressed}/200 frames used a compressed block");
    }

    #[test]
    fn many_sequences_nbseq_header_forms() {
        // 4 bytes per sequence: 128K reaches the 3-byte nbSeq form (>= 0x7F00), smaller blocks the 2-byte form.
        let n = (BLOCK_SIZE / 4 - 1).min(0x7F00 + 100);
        let (out, frame) = roundtrip_script(&vec![(1, 1, 3); n], 13);
        assert_eq!(out.sequences.len(), n);
        assert_eq!(block_type(&frame, FrameOptions::default()), COMPRESSED);
    }

    #[test]
    fn random_block_falls_back_to_raw() {
        let block = synth::random(9, BLOCK_SIZE);
        let out = BlockOutput { sequences: vec![], literals: block.clone() };
        let opts = FrameOptions::default();
        let frame = roundtrip(&block, &out, opts);
        assert_eq!(block_type(&frame, opts), RAW);
        assert_eq!(frame.len(), frame_header(opts).len() + 3 + BLOCK_SIZE);
    }

    #[test]
    fn short_match_block_falls_back_to_raw() {
        // One tiny match: the compressed form would be >= BLOCK_SIZE, so Raw must be chosen.
        let (block, out) = scripted(&[(8, 4, 3)], &mut lit_source(11));
        let opts = FrameOptions::default();
        let frame = roundtrip(&block, &out, opts);
        assert_eq!(block_type(&frame, opts), RAW);
    }

    #[test]
    fn all_zero_block_is_rle() {
        let block = synth::zeros(BLOCK_SIZE);
        let out = BlockOutput { sequences: vec![], literals: block.clone() };
        let opts = FrameOptions::default();
        let frame = roundtrip(&block, &out, opts);
        assert_eq!(block_type(&frame, opts), RLE);
        assert_eq!(frame.len(), frame_header(opts).len() + 3 + 1);
    }

    #[test]
    fn fcs_field_matches_block_size() {
        let opts = FrameOptions::default();
        let h = frame_header(opts);
        assert_eq!(&h[..4], &[0x28, 0xB5, 0x2F, 0xFD]);
        let fhd = h[4];
        assert_eq!(fhd & 0b0010_0000, 0b0010_0000, "single segment");
        assert_eq!(fhd & 0b0000_0111, 0, "no checksum, no dict id");
        let (flag, len) = if BLOCK_SIZE < 65536 + 256 { (1, 7) } else { (2, 9) };
        assert_eq!(fhd >> 6, flag);
        assert_eq!(h.len(), len);
        for block in [synth::zeros(BLOCK_SIZE), synth::text(1, BLOCK_SIZE)] {
            let out = BlockOutput { sequences: vec![], literals: block.clone() };
            let frame = roundtrip(&block, &out, opts);
            assert_eq!(zstd::zstd_safe::get_frame_content_size(&frame).unwrap(), Some(BLOCK_SIZE as u64));
        }
    }

    #[test]
    fn checksum_option_roundtrips() {
        let opts = FrameOptions { checksum: true };
        assert_eq!(frame_header(opts)[4] & 0b100, 0b100);
        let bs = BLOCK_SIZE as u32;
        let cases = [
            scripted(&[(10, 10, 20), (5, 3, bs / 2)], &mut lit_source(7)),
            (synth::zeros(BLOCK_SIZE), BlockOutput { sequences: vec![], literals: synth::zeros(BLOCK_SIZE) }),
            (synth::random(8, BLOCK_SIZE), BlockOutput { sequences: vec![], literals: synth::random(8, BLOCK_SIZE) }),
        ];
        for (block, out) in &cases {
            let mut frame = roundtrip(block, out, opts);
            let want = (xxhash_rust::xxh64::xxh64(block, 0) as u32).to_le_bytes();
            assert_eq!(&frame[frame.len() - 4..], &want);
            // libzstd must actually verify it
            *frame.last_mut().unwrap() ^= 1;
            assert!(zstd::bulk::decompress(&frame, BLOCK_SIZE).is_err());
        }
    }

    #[test]
    fn sequence_with_all_code_extremes() {
        // exercise the max-bits LL/ML/OF extra fields in one stream alongside tiny sequences
        let bs = BLOCK_SIZE as u32;
        let script = [(1, 1, 3), (bs / 4, bs / 4, bs / 4), (0, 1, 3), (2, 2, bs / 4)];
        let (_, frame) = roundtrip_script(&script, 12);
        assert_eq!(block_type(&frame, FrameOptions::default()), COMPRESSED);
    }
}
