//! Sequences section encoding: per-stream mode selection (predefined / RLE / computed FSE table)
//! and the FSE-coded LL/OF/ML bitstream.

use std::borrow::Cow;
use std::sync::LazyLock;

use crate::bits::BitWriter;
use crate::codes::*;
use crate::fse::{FseCTable, FseState, choose_table_log, cost_x256, normalize, write_ncount};
use crate::seq::Sequence;

static LL_TABLE: LazyLock<FseCTable> = LazyLock::new(|| FseCTable::from_normalized(&LL_DEFAULT_NORM, LL_DEFAULT_LOG));
static ML_TABLE: LazyLock<FseCTable> = LazyLock::new(|| FseCTable::from_normalized(&ML_DEFAULT_NORM, ML_DEFAULT_LOG));
static OF_TABLE: LazyLock<FseCTable> = LazyLock::new(|| FseCTable::from_normalized(&OF_DEFAULT_NORM, OF_DEFAULT_LOG));

/// Symbol compression mode of one stream (RFC 8878 §3.1.1.3.2.2). Repeat_Mode is never used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeqMode {
    Predefined,
    /// Every sequence uses this one code.
    Rle(u8),
    Compressed,
}

impl SeqMode {
    /// Two-bit Symbol_Compression_Modes value.
    fn bits(self) -> u8 {
        match self {
            SeqMode::Predefined => 0,
            SeqMode::Rle(_) => 1,
            SeqMode::Compressed => 2,
        }
    }
}

/// The three sequence streams, in table-description order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamKind {
    LiteralLength,
    Offset,
    MatchLength,
}

impl StreamKind {
    /// Histogram size: LL codes 0..=35, OF codes 0..=31 (MaxOff), ML codes 0..=52.
    pub const fn alphabet(self) -> usize {
        match self {
            StreamKind::LiteralLength => 36,
            StreamKind::Offset => 32,
            StreamKind::MatchLength => 53,
        }
    }

    /// Largest table log libzstd accepts for this stream (LLFSELog / OffFSELog / MLFSELog).
    pub const fn max_log(self) -> u32 {
        match self {
            StreamKind::LiteralLength | StreamKind::MatchLength => 9,
            StreamKind::Offset => 8,
        }
    }

    fn default_norm(self) -> (&'static [i16], u32) {
        match self {
            StreamKind::LiteralLength => (&LL_DEFAULT_NORM, LL_DEFAULT_LOG),
            StreamKind::Offset => (&OF_DEFAULT_NORM, OF_DEFAULT_LOG),
            StreamKind::MatchLength => (&ML_DEFAULT_NORM, ML_DEFAULT_LOG),
        }
    }

    fn default_table(self) -> &'static FseCTable {
        match self {
            StreamKind::LiteralLength => &LL_TABLE,
            StreamKind::Offset => &OF_TABLE,
            StreamKind::MatchLength => &ML_TABLE,
        }
    }
}

/// How one stream is encoded: its mode, the FSE table the encoder uses and the table
/// description written after the modes byte (empty / 1 RLE byte / NCount header).
#[derive(Clone, Debug)]
pub struct StreamTable {
    pub mode: SeqMode,
    pub table: Cow<'static, FseCTable>,
    pub description: Vec<u8>,
}

impl StreamTable {
    /// zstd's predefined distribution for `kind`.
    pub fn predefined(kind: StreamKind) -> StreamTable {
        StreamTable { mode: SeqMode::Predefined, table: Cow::Borrowed(kind.default_table()), description: Vec::new() }
    }

    /// Every sequence uses `code`: one description byte, no state bits.
    pub fn rle(code: u8) -> StreamTable {
        StreamTable { mode: SeqMode::Rle(code), table: Cow::Owned(FseCTable::rle(code)), description: vec![code] }
    }

    /// Table computed from the histogram `counts` (indexed by code, at least one nonzero).
    pub fn computed(kind: StreamKind, counts: &[u32]) -> StreamTable {
        Self::computed_with_norm(kind, counts).0
    }

    /// `computed`, also returning the normalized counts and table log.
    fn computed_with_norm(kind: StreamKind, counts: &[u32]) -> (StreamTable, Vec<i16>, u32) {
        let last = counts.iter().rposition(|&c| c > 0).expect("empty histogram");
        let counts = &counts[..=last];
        let total: u32 = counts.iter().sum();
        let used = counts.iter().filter(|&&c| c > 0).count();
        let log = choose_table_log(total, used, kind.max_log());
        let norm = normalize(counts, total, log);
        let mut description = Vec::new();
        write_ncount(&norm, log, &mut description);
        let table = FseCTable::from_normalized(&norm, log);
        (StreamTable { mode: SeqMode::Compressed, table: Cow::Owned(table), description }, norm, log)
    }

    /// Mode choice for one stream with histogram `counts` over `nb_seq` sequences (integer-only;
    /// the GPU port mirrors it):
    /// 1. a single distinct code and `nb_seq > 2` -> RLE;
    /// 2. predefined is allowed only if every used code has a nonzero default probability;
    /// 3. otherwise pick computed iff `cost(computed) + 256 * 8 * ncount_bytes < cost(predefined)`
    ///    (costs from [`cost_x256`]; ties go to predefined).
    pub fn choose(kind: StreamKind, counts: &[u32], nb_seq: usize) -> StreamTable {
        let mut used = counts.iter().enumerate().filter(|&(_, &c)| c > 0);
        let (first, _) = used.next().expect("empty histogram");
        if used.next().is_none() && nb_seq > 2 {
            return StreamTable::rle(first as u8);
        }
        let (default_norm, default_log) = kind.default_norm();
        let predefined_ok = counts.iter().enumerate().all(|(s, &c)| c == 0 || default_norm.get(s).is_some_and(|&n| n != 0));
        let (computed, norm, log) = Self::computed_with_norm(kind, counts);
        if !predefined_ok {
            return computed;
        }
        let computed_cost = cost_x256(counts, &norm, log) + 256 * 8 * computed.description.len() as u64;
        if computed_cost < cost_x256(counts, default_norm, default_log) {
            computed
        } else {
            StreamTable::predefined(kind)
        }
    }
}

/// Per-sequence codes and extra-bit fields.
struct Coded {
    ll: u8,
    ml: u8,
    of: u8,
    ll_extra: u32,
    ml_extra: u32,
    of_extra: u32,
}

impl Coded {
    fn new(s: &Sequence) -> Self {
        let ll = ll_code(s.lit_len);
        let ml = ml_code(s.match_len);
        let of = of_code(s.off_base);
        Coded {
            ll,
            ml,
            of,
            ll_extra: s.lit_len - LL_BASE[ll as usize],
            ml_extra: s.match_len - ML_BASE[ml as usize],
            of_extra: s.off_base - (1 << of),
        }
    }

    /// Extra bits in decoder order: LL, ML, OF.
    ///
    /// Why `value - BASE[code]` equals libzstd's masked `BIT_addBits(raw, nbBits)`: the raw values are
    /// lit_len, match_len - 3 and off_base, whose code bases (LL_BASE, ML_BASE - 3, 1 << of) are
    /// multiples of `1 << nbBits`, so the low `nbBits` of `raw` are exactly `raw - base`.
    fn write_extras(&self, w: &mut BitWriter) {
        w.add_bits(self.ll_extra, LL_BITS[self.ll as usize] as u32);
        w.add_bits(self.ml_extra, ML_BITS[self.ml as usize] as u32);
        w.add_bits(self.of_extra, self.of as u32);
    }
}

/// Number_of_Sequences field (RFC 8878 §3.1.1.3.2.1).
fn write_nb_seq(n: usize, out: &mut Vec<u8>) {
    const LONGNBSEQ: usize = 0x7F00;
    if n < 128 {
        out.push(n as u8);
    } else if n < LONGNBSEQ {
        out.extend_from_slice(&[((n >> 8) + 0x80) as u8, n as u8]);
    } else {
        assert!(n - LONGNBSEQ <= 0xFFFF, "too many sequences: {n}");
        out.push(0xFF);
        out.extend_from_slice(&((n - LONGNBSEQ) as u16).to_le_bytes());
    }
}

/// Per-stream code histograms `[LL, OF, ML]`, sized by [`StreamKind::alphabet`].
pub fn histograms(seqs: &[Sequence]) -> [Vec<u32>; 3] {
    let mut h = [StreamKind::LiteralLength, StreamKind::Offset, StreamKind::MatchLength].map(|k| vec![0u32; k.alphabet()]);
    for s in seqs {
        h[0][ll_code(s.lit_len) as usize] += 1;
        h[1][of_code(s.off_base) as usize] += 1;
        h[2][ml_code(s.match_len) as usize] += 1;
    }
    h
}

/// Sequences section with all three streams predefined.
pub fn write_sequences_section(seqs: &[Sequence], out: &mut Vec<u8>) {
    let tables = [StreamKind::LiteralLength, StreamKind::Offset, StreamKind::MatchLength].map(StreamTable::predefined);
    write_sequences_section_with(seqs, &tables, out);
}

/// Sequences section with each stream's mode chosen by [`StreamTable::choose`].
pub fn write_sequences_section_auto(seqs: &[Sequence], out: &mut Vec<u8>) {
    if seqs.is_empty() {
        write_nb_seq(0, out);
        return;
    }
    let h = histograms(seqs);
    let kinds = [StreamKind::LiteralLength, StreamKind::Offset, StreamKind::MatchLength];
    let tables = [0, 1, 2].map(|i| StreamTable::choose(kinds[i], &h[i], seqs.len()));
    write_sequences_section_with(seqs, &tables, out);
}

/// Sequences section (header, modes byte, table descriptions, bitstream) using `tables`
/// in `[LL, OF, ML]` order.
pub fn write_sequences_section_with(seqs: &[Sequence], tables: &[StreamTable; 3], out: &mut Vec<u8>) {
    write_nb_seq(seqs.len(), out);
    let Some(last) = seqs.last() else { return };
    let [ll_t, of_t, ml_t] = tables;
    out.push((ll_t.mode.bits() << 6) | (of_t.mode.bits() << 4) | (ml_t.mode.bits() << 2));
    for t in tables {
        out.extend_from_slice(&t.description);
    }

    // Port of ZSTD_encodeSequences_body: sequences are encoded last-to-first so the
    // decoder reads them first-to-last.
    let mut w = BitWriter::new();
    let c = Coded::new(last);
    let mut ml_state = FseState::init(&ml_t.table, c.ml);
    let mut of_state = FseState::init(&of_t.table, c.of);
    let mut ll_state = FseState::init(&ll_t.table, c.ll);
    c.write_extras(&mut w);
    for s in seqs[..seqs.len() - 1].iter().rev() {
        let c = Coded::new(s);
        of_state.encode(&mut w, c.of);
        ml_state.encode(&mut w, c.ml);
        ll_state.encode(&mut w, c.ll);
        c.write_extras(&mut w);
    }
    ml_state.flush(&mut w);
    of_state.flush(&mut w);
    ll_state.flush(&mut w);
    out.extend_from_slice(&w.finish_with_end_mark());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::chunk_file;
    use crate::config::BLOCK_SIZE;
    use crate::frame::testutil::scripted;
    use crate::frame::{FrameOptions, frame_header, write_frame, write_literals_raw};
    use crate::reference::{LVL3, compress_block};
    use crate::seq::{BlockOutput, reconstruct};
    use crate::synth;

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

    /// Reference-compressor outputs for every full block of every synthetic case.
    fn synthetic_outputs() -> Vec<(String, Vec<u8>, BlockOutput)> {
        let mut v = Vec::new();
        for (name, data) in synth::test_cases() {
            for (i, b) in chunk_file(&data).into_iter().enumerate() {
                let out = compress_block(&b.data, LVL3);
                v.push((format!("{name}#{i}"), b.data, out));
            }
        }
        v
    }

    /// Scripted outputs: uniform streams, a wide spread of codes, and a few sequences.
    fn scripted_outputs() -> Vec<(String, Vec<u8>, BlockOutput)> {
        let bs = BLOCK_SIZE as u32;
        let mut v = Vec::new();
        let mut push = |name: &str, script: &[(u32, u32, u32)], seed| {
            let (block, out) = scripted(script, &mut lit_source(seed));
            v.push((name.to_string(), block, out));
        };
        push("single", &[(10, 10, 20)], 1);
        push("two", &[(10, 10, 20), (5, 3, 100)], 2);
        push("uniform_all", &vec![(1, 1, 3); 500], 3);
        let mut r = Lcg(77);
        for k in 0..6u32 {
            let mut script = Vec::new();
            let mut pos = 0u32;
            let n = [3, 20, 200, 2000, 5000, 9000][k as usize];
            for _ in 0..n {
                let ll = [r.below(4), r.below(40), r.below(1000)][r.below(3) as usize] + (pos == 0) as u32;
                let ml = 3 + [r.below(3), r.below(60), r.below(3000)][r.below(3) as usize];
                if pos + ll + ml > bs {
                    break;
                }
                let near = [8, 1000, bs][r.below(3) as usize];
                let off = 1 + r.below((pos + ll).min(near));
                script.push((ll, off, ml));
                pos += ll + ml;
            }
            push(&format!("random{k}"), &script, 10 + k as u64);
        }
        v
    }

    /// Wrap literals + a given sequences section into a one-block frame; None if it would not fit.
    fn frame_with_section(out: &BlockOutput, section: &[u8]) -> Option<Vec<u8>> {
        let mut content = Vec::new();
        write_literals_raw(&out.literals, &mut content);
        content.extend_from_slice(section);
        if content.len() > BLOCK_SIZE {
            return None;
        }
        let mut f = frame_header(FrameOptions::default());
        f.extend_from_slice(&(1 | (2 << 1) | ((content.len() as u32) << 3)).to_le_bytes()[..3]);
        f.extend_from_slice(&content);
        Some(f)
    }

    /// Modes byte of a non-empty section.
    fn modes_byte(section: &[u8]) -> u8 {
        let hdr = match section[0] {
            0..128 => 1,
            255 => 3,
            _ => 2,
        };
        section[hdr]
    }

    #[test]
    fn write_ncount_roundtrips_through_libzstd() {
        let kinds = [StreamKind::LiteralLength, StreamKind::Offset, StreamKind::MatchLength];
        let mut decoded = 0;
        for (name, block, out) in synthetic_outputs().into_iter().chain(scripted_outputs()) {
            assert_eq!(reconstruct(&out).unwrap(), block, "{name}: not a valid parse");
            if out.sequences.is_empty() {
                continue;
            }
            let h = histograms(&out.sequences);
            let tables = [0, 1, 2].map(|i| StreamTable::computed(kinds[i], &h[i]));
            let mut section = Vec::new();
            write_sequences_section_with(&out.sequences, &tables, &mut section);
            assert_eq!(modes_byte(&section), 0b1010_1000, "{name}: all three streams compressed");
            let Some(frame) = frame_with_section(&out, &section) else { continue };
            let dec = zstd::bulk::decompress(&frame, BLOCK_SIZE).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(dec == block, "{name}: decoded block differs");
            decoded += 1;
        }
        assert!(decoded >= 12, "only {decoded} frames checked");
    }

    /// ML codes {0, big} leave a gap of more than 24 zero-probability codes, so `write_ncount` takes
    /// its 24-zero-run (0xFFFF) branch; libzstd must still decode the table. The long match is sized
    /// to the block (code 52 at 128K, code 49 at 16K: both gaps exceed 24).
    #[test]
    fn write_ncount_long_zero_run_roundtrips() {
        let big = BLOCK_SIZE as u32 - 2 * 50 * 4 - 1 - 64;
        let mut script = vec![(1, 1, 3); 50];
        script.push((1, 1, big));
        script.extend(vec![(1, 1, 3); 50]);
        let (block, out) = scripted(&script, &mut lit_source(24));
        let h = histograms(&out.sequences);
        let used: Vec<usize> = (0..h[2].len()).filter(|&c| h[2][c] > 0).collect();
        assert_eq!(used[0], 0);
        assert_eq!(used.len(), 2);
        assert!(used[1] > 24 + 1, "gap too small: {used:?}");

        // ML forced computed; LL and OF as chosen (both RLE here).
        let kinds = [StreamKind::LiteralLength, StreamKind::Offset, StreamKind::MatchLength];
        let n = out.sequences.len();
        let tables = [StreamTable::choose(kinds[0], &h[0], n), StreamTable::choose(kinds[1], &h[1], n), StreamTable::computed(kinds[2], &h[2])];
        let mut section = Vec::new();
        write_sequences_section_with(&out.sequences, &tables, &mut section);
        assert_eq!(modes_byte(&section), 0b0101_1000, "LL/OF RLE, ML compressed");
        let frame = frame_with_section(&out, &section).expect("fits");
        assert!(zstd::bulk::decompress(&frame, BLOCK_SIZE).unwrap() == block, "decoded block differs");

        // The auto choice picks the computed table for ML too, and the full frame round-trips.
        let mut auto = Vec::new();
        write_sequences_section_auto(&out.sequences, &mut auto);
        assert_eq!(auto, section);
        let opts = FrameOptions { checksum: false, huffman: false };
        assert!(zstd::bulk::decompress(&write_frame(&block, &out, opts), BLOCK_SIZE).unwrap() == block);
    }

    #[test]
    fn rle_mode_used_for_uniform_ml() {
        let mut r = Lcg(5);
        let mut pos = 0;
        let script: Vec<_> = (0..300)
            .map(|_| {
                let ll = 1 + r.below(50);
                let off = 1 + r.below(pos + ll);
                pos += ll + 20;
                (ll, off, 20)
            })
            .collect();
        let (block, out) = scripted(&script, &mut lit_source(6));
        let mut section = Vec::new();
        write_sequences_section_auto(&out.sequences, &mut section);
        let modes = modes_byte(&section);
        assert_eq!((modes >> 2) & 3, 1, "ML mode must be RLE, modes {modes:#010b}");
        assert_ne!((modes >> 6) & 3, 1, "LL is not uniform");
        let frame = write_frame(&block, &out, FrameOptions::default());
        assert_eq!(zstd::bulk::decompress(&frame, BLOCK_SIZE).unwrap(), block);
    }

    #[test]
    fn rle_needs_more_than_two_sequences() {
        let s = vec![Sequence { lit_len: 1, match_len: 3, off_base: 1 }; 2];
        let mut section = Vec::new();
        write_sequences_section_auto(&s, &mut section);
        assert_eq!(modes_byte(&section) & 0b0101_0100, 0, "no RLE mode with nbSeq <= 2");
        let s = vec![Sequence { lit_len: 1, match_len: 3, off_base: 1 }; 3];
        let mut section = Vec::new();
        write_sequences_section_auto(&s, &mut section);
        // modes byte, then one RLE byte per stream in LL, OF, ML order, then the (bit-free) stream
        assert_eq!(&section[1..], &[0b0101_0100, 1, 0, 0, 1]);
    }

    #[test]
    fn auto_never_bigger_than_predefined() {
        let mut saved = 0i64;
        let mut computed = 0;
        for (name, _, out) in synthetic_outputs().into_iter().chain(scripted_outputs()) {
            let mut pre = Vec::new();
            write_sequences_section(&out.sequences, &mut pre);
            let mut auto = Vec::new();
            write_sequences_section_auto(&out.sequences, &mut auto);
            assert!(auto.len() <= pre.len() + 1, "{name}: auto {} > predefined {} + 1", auto.len(), pre.len());
            saved += pre.len() as i64 - auto.len() as i64;            if !out.sequences.is_empty() {
                let m = modes_byte(&auto);
                computed += [m >> 6, m >> 4, m >> 2].iter().filter(|&&x| x & 3 == 2).count();
            }
        }
        assert!(computed > 0, "computed tables never chosen");
        assert!(saved > 0, "auto saved nothing");
    }

    #[test]
    fn predefined_wrapper_matches_explicit_tables() {
        for (_, _, out) in scripted_outputs() {
            let mut a = Vec::new();
            write_sequences_section(&out.sequences, &mut a);
            let tables = [StreamKind::LiteralLength, StreamKind::Offset, StreamKind::MatchLength].map(StreamTable::predefined);
            let mut b = Vec::new();
            write_sequences_section_with(&out.sequences, &tables, &mut b);
            assert_eq!(a, b);
        }
    }

    fn section_for(n: usize) -> Vec<u8> {
        let s = vec![Sequence { lit_len: 1, match_len: 3, off_base: 1 }; n];
        let mut out = Vec::new();
        write_sequences_section(&s, &mut out);
        out
    }

    #[test]
    fn empty_is_single_zero_byte() {
        assert_eq!(section_for(0), vec![0]);
    }

    #[test]
    fn nbseq_header_forms() {
        // header bytes followed by the modes byte (all predefined = 0)
        assert_eq!(&section_for(1)[..2], &[1, 0]);
        assert_eq!(&section_for(127)[..2], &[127, 0]);
        assert_eq!(&section_for(128)[..3], &[0x80, 128, 0]);
        assert_eq!(&section_for(0x7EFF)[..3], &[0x80 + 0x7E, 0xFF, 0]);
        assert_eq!(&section_for(0x7F00)[..4], &[0xFF, 0, 0, 0]);
        assert_eq!(&section_for(0x7F00 + 0x123)[..4], &[0xFF, 0x23, 0x01, 0]);
    }

    /// With `min_match = 4` a 128K block can hold more than 0x7F00 sequences, so the 3-byte nbSeq
    /// form is reachable. 4 literals then 32767 matches of length 4 at offset 4 (ll = 0; repcodes
    /// after the first two) fill the block exactly; the frame must be Compressed and decode. At
    /// 16K..64K `BLOCK_SIZE / 4 < 0x7F00`, so the form is unreachable there.
    #[cfg(feature = "block-128k")]
    #[test]
    fn nbseq_three_byte_form_roundtrips() {
        let n = BLOCK_SIZE / 4 - 1;
        let mut script = vec![(4, 4, 4)];
        script.extend(std::iter::repeat_n((0, 4, 4), n - 1));
        let (block, out) = scripted(&script, &mut lit_source(0x7f00));
        assert_eq!(out.sequences.len(), n);
        assert!(out.sequences.len() >= 0x7F00);
        assert_eq!(reconstruct(&out).unwrap(), block);
        let mut section = Vec::new();
        write_sequences_section_auto(&out.sequences, &mut section);
        let extra = (n - 0x7F00) as u16;
        assert_eq!(section[..3], [0xFF, extra as u8, (extra >> 8) as u8], "3-byte nbSeq header");
        for opts in [FrameOptions::default(), FrameOptions { checksum: false, huffman: false }] {
            let frame = write_frame(&block, &out, opts);
            assert_eq!((frame[frame_header(opts).len()] >> 1) & 3, 2, "Compressed block");
            let dec = zstd::bulk::decompress(&frame, BLOCK_SIZE).expect("libzstd rejected frame");
            assert!(dec == block, "decoded block differs");
        }
    }
}
