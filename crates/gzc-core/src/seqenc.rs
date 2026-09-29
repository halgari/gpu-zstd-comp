//! Sequences section encoding: mode selection and FSE-coded LL/ML/OF streams.

use std::sync::LazyLock;

use crate::bits::BitWriter;
use crate::codes::*;
use crate::fse::{FseCTable, FseState};
use crate::seq::Sequence;

static LL_TABLE: LazyLock<FseCTable> = LazyLock::new(|| FseCTable::from_normalized(&LL_DEFAULT_NORM, LL_DEFAULT_LOG));
static ML_TABLE: LazyLock<FseCTable> = LazyLock::new(|| FseCTable::from_normalized(&ML_DEFAULT_NORM, ML_DEFAULT_LOG));
static OF_TABLE: LazyLock<FseCTable> = LazyLock::new(|| FseCTable::from_normalized(&OF_DEFAULT_NORM, OF_DEFAULT_LOG));

/// Symbol_Compression_Modes value for Predefined_Mode.
const MODE_PREDEFINED: u8 = 0;

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

/// Sequences section (header + modes byte + bitstream). M1: all three predefined.
pub fn write_sequences_section(seqs: &[Sequence], out: &mut Vec<u8>) {
    write_nb_seq(seqs.len(), out);
    let Some(last) = seqs.last() else { return };
    out.push((MODE_PREDEFINED << 6) | (MODE_PREDEFINED << 4) | (MODE_PREDEFINED << 2));

    // Port of ZSTD_encodeSequences_body: sequences are encoded last-to-first so the
    // decoder reads them first-to-last.
    let mut w = BitWriter::new();
    let c = Coded::new(last);
    let mut ml_state = FseState::init(&ML_TABLE, c.ml);
    let mut of_state = FseState::init(&OF_TABLE, c.of);
    let mut ll_state = FseState::init(&LL_TABLE, c.ll);
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
}
