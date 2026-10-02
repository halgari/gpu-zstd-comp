//! zstd frame writer: header, block header, literals and sequences sections.
//!
//! Every block becomes one single-segment zstd frame holding exactly one zstd block
//! (RFC 8878 §3.1). The frame's content size is the block's real length (`Block::real_len`):
//! `BLOCK_SIZE`, or less for a file's last block.

use xxhash_rust::xxh64::xxh64;

use crate::config::BLOCK_SIZE;
use crate::huffman::{write_literals_section, write_raw_rle_header};
use crate::seq::BlockOutput;
use crate::seqenc::write_sequences_section_auto;

#[derive(Clone, Copy, Debug)]
pub struct FrameOptions {
    /// Append the 4-byte xxh64 content checksum.
    pub checksum: bool,
    /// Entropy-code literals (Huffman / RLE) instead of always storing them raw.
    pub huffman: bool,
}

impl Default for FrameOptions {
    fn default() -> Self {
        FrameOptions { checksum: false, huffman: true }
    }
}

const MAGIC: u32 = 0xFD2F_B528;

const BLOCK_RAW: u32 = 0;
const BLOCK_RLE: u32 = 1;
const BLOCK_COMPRESSED: u32 = 2;

/// Raw_Literals_Block: literals section header (type 0) followed by the literal bytes.
pub fn write_literals_raw(lits: &[u8], out: &mut Vec<u8>) {
    write_raw_rle_header(0, lits.len(), out);
    out.extend_from_slice(lits);
}

/// Magic number + Frame_Header_Descriptor + Frame_Content_Size (single segment, no dictionary)
/// of a full block's frame.
pub fn frame_header(opts: FrameOptions) -> Vec<u8> {
    frame_header_for(opts, BLOCK_SIZE)
}

/// Content sizes below this get the 1-byte Frame_Content_Size field (a 6-byte header, against 7).
pub const SHORT_FCS_LIMIT: usize = 256;

/// `frame_header` of a frame holding `content_size` bytes (`1..=BLOCK_SIZE`): FCS_Field_Size flag
/// 1 (2 bytes holding size - 256, which covers 256..=65791), or flag 0 (1 byte) below
/// `SHORT_FCS_LIMIT`.
pub fn frame_header_for(opts: FrameOptions, content_size: usize) -> Vec<u8> {
    const _: () = assert!(BLOCK_SIZE >= SHORT_FCS_LIMIT && BLOCK_SIZE < 65536 + 256);
    assert!((1..=BLOCK_SIZE).contains(&content_size), "frame content size {content_size}");
    let mut h = MAGIC.to_le_bytes().to_vec();
    let checksum = (opts.checksum as u8) << 2;
    if content_size >= SHORT_FCS_LIMIT {
        h.push((1 << 6) | (1 << 5) | checksum);
        h.extend_from_slice(&((content_size - 256) as u16).to_le_bytes());
    } else {
        h.push((1 << 5) | checksum);
        h.push(content_size as u8);
    }
    h
}

/// Block_Header for the only (last) block of the frame.
fn block_header(block_type: u32, size: usize, out: &mut Vec<u8>) {
    debug_assert!(size < 1 << 21);
    let v = 1 | (block_type << 1) | ((size as u32) << 3);
    out.extend_from_slice(&v.to_le_bytes()[..3]);
}

/// Encode `block` (`1..=BLOCK_SIZE` bytes: a block's real bytes) as one zstd frame, using `out`
/// as its parse (covering exactly `block`; `seq::truncate_output` cuts a padded block's parse).
/// The block is RLE when every byte is equal, else Compressed when that is smaller than the block,
/// else Raw. Below `SHORT_FCS_LIMIT` bytes it is never Compressed: the GPU writes a compressed
/// block's literals section at a fixed offset after the 7-byte header.
pub fn write_frame(block: &[u8], out: &BlockOutput, opts: FrameOptions) -> Vec<u8> {
    let n = block.len();
    assert!((1..=BLOCK_SIZE).contains(&n), "a frame holds 1..=BLOCK_SIZE bytes, not {n}");
    let mut f = frame_header_for(opts, n);
    if block.iter().all(|&b| b == block[0]) {
        block_header(BLOCK_RLE, n, &mut f);
        f.push(block[0]);
    } else if n < SHORT_FCS_LIMIT {
        block_header(BLOCK_RAW, n, &mut f);
        f.extend_from_slice(block);
    } else {
        debug_assert_eq!(
            out.literals.len() as u64 + out.sequences.iter().map(|s| s.match_len as u64).sum::<u64>(),
            n as u64,
            "BlockOutput does not cover the block"
        );
        let mut content = Vec::with_capacity(out.literals.len() + 16 + out.sequences.len() * 4);
        if opts.huffman {
            write_literals_section(&out.literals, &mut content);
        } else {
            write_literals_raw(&out.literals, &mut content);
        }
        write_sequences_section_auto(&out.sequences, &mut content);
        if content.len() < n {
            block_header(BLOCK_COMPRESSED, content.len(), &mut f);
            f.extend_from_slice(&content);
        } else {
            block_header(BLOCK_RAW, n, &mut f);
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

    /// Literals section type (0 raw, 1 RLE, 2 compressed) of a Compressed block's frame.
    fn literals_type(frame: &[u8], opts: FrameOptions) -> u8 {
        assert_eq!(block_type(frame, opts), COMPRESSED);
        frame[frame_header(opts).len() + 3] & 3
    }

    /// Byte length of the literals section starting at `sec[0]`.
    fn literals_section_len(sec: &[u8]) -> usize {
        let b0 = sec[0] as usize;
        let sf = (b0 >> 2) & 3;
        match b0 & 3 {
            0 | 1 => {
                let (hdr, n) = match sf {
                    0 | 2 => (1, b0 >> 3),
                    1 => (2, (b0 >> 4) | (sec[1] as usize) << 4),
                    _ => (3, (b0 >> 4) | (sec[1] as usize) << 4 | (sec[2] as usize) << 12),
                };
                hdr + if b0 & 3 == 0 { n } else { 1 }
            }
            _ => {
                let v = u64::from_le_bytes([sec[0], sec[1], sec[2], sec[3], sec[4], 0, 0, 0]);
                let (hdr, bits) = match sf {
                    0 | 1 => (3, 10),
                    2 => (4, 14),
                    _ => (5, 18),
                };
                hdr + ((v >> (4 + bits)) & ((1 << bits) - 1)) as usize
            }
        }
    }

    #[test]
    fn frame_roundtrip_no_sequences() {
        // All-literal text with no matches: Huffman literals make the block Compressed with nbSeq = 0.
        let block = synth::text(3, BLOCK_SIZE);
        let out = BlockOutput { sequences: vec![], literals: block.clone() };
        let opts = FrameOptions::default();
        let frame = roundtrip(&block, &out, opts);
        assert_eq!(literals_type(&frame, opts), COMPRESSED);
        let content = &frame[frame_header(opts).len() + 3..];
        let lit_len = literals_section_len(content);
        assert_eq!(content.len(), lit_len + 1, "only the nbSeq byte follows the literals");
        assert_eq!(content[lit_len], 0, "nbSeq == 0");
        assert!(frame.len() < BLOCK_SIZE * 3 / 4, "text literals should shrink: {}", frame.len());
        // without Huffman the same block cannot compress
        let raw_opts = FrameOptions { huffman: false, ..opts };
        assert_eq!(block_type(&roundtrip(&block, &out, raw_opts), raw_opts), RAW);
    }

    #[test]
    fn synthetic_cases_roundtrip_with_huffman_literals() {
        let opts = FrameOptions::default();
        let mut direct = 0;
        for (name, data) in synth::test_cases() {
            for (i, b) in crate::block::chunk_file(&data).into_iter().enumerate() {
                let out = crate::reference::compress_block(&b.data, crate::params::LVL3);
                let frame = roundtrip(&b.data, &out, opts);
                if matches!(name, "text" | "nif" | "exact_block") {
                    assert_eq!(literals_type(&frame, opts), COMPRESSED, "{name}#{i}: literals not Huffman-coded");
                    // text-like literals stay below 128: direct 4-bit weights (header byte >= 128)
                    let content = &frame[frame_header(opts).len() + 3..];
                    let hdr = [3, 3, 4, 5][((content[0] >> 2) & 3) as usize];
                    if name != "nif" {
                        assert!(content[hdr] >= 128, "{name}#{i}: expected direct weights");
                        direct += 1;
                    }
                }
            }
        }
        assert!(direct >= 2);
    }

    #[test]
    fn fse_weights_roundtrip_in_frames() {
        // Skewed random literals reaching past 127 (max_symbol >= 128) force FSE-compressed weights.
        let opts = FrameOptions::default();
        let mut r = Lcg(21);
        let mut lits = move || (r.next() & r.next() & r.next()) as u8;
        let bs = BLOCK_SIZE as u32;
        for script in [vec![(bs / 2, 1000, bs / 4)], vec![(300, 7, 20); (bs / 640) as usize], vec![(5, 3, 8); (bs / 26) as usize]] {
            let (block, out) = scripted(&script, &mut lits);
            let frame = roundtrip(&block, &out, opts);
            assert_eq!(literals_type(&frame, opts), COMPRESSED);
            let content = &frame[frame_header(opts).len() + 3..];
            let hdr = [3, 3, 4, 5][((content[0] >> 2) & 3) as usize];
            assert!(content[hdr] < 128, "expected FSE-compressed weights, header {}", content[hdr]);
        }
    }

    #[test]
    fn huffman_frames_never_grow() {
        let on = FrameOptions::default();
        let off = FrameOptions { huffman: false, ..on };
        let mut cases: Vec<(Vec<u8>, BlockOutput)> = Vec::new();
        for (_, data) in synth::test_cases() {
            for b in crate::block::chunk_file(&data) {
                let out = crate::reference::compress_block(&b.data, crate::params::LVL3);
                cases.push((b.data, out));
            }
        }
        let mut r = Lcg(0xabc);
        for i in 0..60 {
            let nseq = 1 + r.below(3000);
            let (script, _) = random_script(&mut r, nseq);
            cases.push(scripted(&script, &mut lit_source(500 + i)));
        }
        let mut saved = 0i64;
        for (i, (block, out)) in cases.iter().enumerate() {
            let a = roundtrip(block, out, on).len();
            let b = roundtrip(block, out, off).len();
            assert!(a <= b + 1, "case {i}: huffman frame {a} > raw-literals frame {b} + 1");
            saved += b as i64 - a as i64;
        }
        assert!(saved > 0);
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
        // ll = 36864 (LL code 34), ml = 20480 (ML code 50).
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
        // 4 bytes per sequence: 16383 sequences, the 2-byte nbSeq form (the 3-byte form needs >= 0x7F00).
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
        assert_eq!(fhd >> 6, 1, "2-byte frame content size");
        assert_eq!(h.len(), 7);
        for block in [synth::zeros(BLOCK_SIZE), synth::text(1, BLOCK_SIZE)] {
            let out = BlockOutput { sequences: vec![], literals: block.clone() };
            let frame = roundtrip(&block, &out, opts);
            assert_eq!(zstd::zstd_safe::get_frame_content_size(&frame).unwrap(), Some(BLOCK_SIZE as u64));
        }
    }

    #[test]
    fn checksum_option_roundtrips() {
        let opts = FrameOptions { checksum: true, ..Default::default() };
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

    /// Ratio report over the real corpus (not part of the normal suite; needs `data/`):
    /// `GZC_CORPUS=/path/to/data/corpus GZC_CORPUS_MB=500 cargo test --release -p gzc-core corpus_ratio -- --ignored --nocapture`
    /// Takes the first `GZC_CORPUS_MB` (default 500) MB of .dds/.nif files (only `GZC_CORPUS_EXT` if set)
    /// in sorted path order and
    /// prints real bytes / frame bytes for the reference parse with Huffman literals, with raw
    /// literals, and libzstd level 3 on the same padded blocks, overall and per extension.
    #[test]
    #[ignore]
    fn corpus_ratio() {
        use std::path::Path;
        fn ext(p: &Path) -> String {
            p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default()
        }
        let Some(root) = crate::testdata::corpus_dir() else { return };
        let limit: u64 = std::env::var("GZC_CORPUS_MB").map(|v| v.parse().unwrap()).unwrap_or(500) * 1_000_000;
        let only = std::env::var("GZC_CORPUS_EXT").ok();
        let files: Vec<_> =
            crate::testdata::corpus_files(&root).into_iter().filter(|f| only.as_deref().is_none_or(|o| o == ext(f))).collect();
        let mut jobs: Vec<(String, std::sync::Arc<crate::block::Block>)> = Vec::new();
        let mut taken = 0u64;
        for f in files {
            if taken >= limit {
                break;
            }
            let bytes = std::fs::read(&f).unwrap();
            taken += bytes.len() as u64;
            for b in crate::block::chunk_file(&bytes) {
                jobs.push((ext(&f), std::sync::Arc::new(b)));
            }
        }
        // per extension: [real, huffman, raw-literals, libzstd L3]
        let totals = std::sync::Mutex::new(std::collections::BTreeMap::<String, [u64; 4]>::new());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((e, b)) = jobs.get(i) else { break };
                        let out = crate::reference::compress_block(&b.data, crate::params::LVL3);
                        let on = write_frame(&b.data, &out, FrameOptions::default());
                        let off = write_frame(&b.data, &out, FrameOptions { huffman: false, ..Default::default() });
                        let dec = zstd::bulk::decompress(&on, BLOCK_SIZE).expect("libzstd rejected frame");
                        assert!(dec == b.data, "decoded block differs");
                        let l3 = zstd::bulk::compress(&b.data, 3).unwrap();
                        let row = [b.real_len, on.len(), off.len(), l3.len()].map(|x| x as u64);
                        let mut t = totals.lock().unwrap();
                        for key in [e.clone(), "ALL".to_string()] {
                            let acc = t.entry(key).or_default();
                            for k in 0..4 {
                                acc[k] += row[k];
                            }
                        }
                    }
                });
            }
        });
        println!("{:>5} {:>12} {:>9} {:>9} {:>9}", "ext", "real bytes", "huffman", "raw-lits", "zstd-L3");
        for (e, t) in totals.into_inner().unwrap() {
            let r = |k: usize| t[0] as f64 / t[k] as f64;
            println!("{e:>5} {:>12} {:>9.4} {:>9.4} {:>9.4}", t[0], r(1), r(2), r(3));
        }
    }
}
