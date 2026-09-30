//! Bit writer for zstd's LSB-first bitstream encoding.

/// zstd bitstream writer: bits are appended LSB-first into little-endian bytes.
#[derive(Debug, Default)]
pub struct BitWriter {
    acc: u64,
    nbits: u32,
    out: Vec<u8>,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append the low `nbits` bits of `value` (`nbits <= 32`).
    pub fn add_bits(&mut self, value: u32, nbits: u32) {
        debug_assert!(nbits <= 32);
        let masked = value as u64 & ((1u64 << nbits) - 1);
        // Invariant: fewer than 8 bits are pending here, so at most 39 bits after the add.
        self.acc |= masked << self.nbits;
        self.nbits += nbits;
        while self.nbits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }

    /// Append the end mark (a single 1 bit), zero-pad to a byte boundary and return the bytes.
    pub fn finish_with_end_mark(mut self) -> Vec<u8> {
        self.add_bits(1, 1);
        if self.nbits > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_pack_lsb_first() {
        let mut w = BitWriter::new();
        w.add_bits(0b101, 3);
        w.add_bits(1, 1);
        assert_eq!(w.finish_with_end_mark(), vec![0b0001_1101]);
    }

    #[test]
    fn bits_masks_and_spans_bytes() {
        let mut w = BitWriter::new();
        w.add_bits(0xFFFF_FFFF, 4); // only the low 4 bits are kept
        w.add_bits(0xABCD_1234, 32);
        w.add_bits(0, 0);
        w.add_bits(0x3, 2);
        // 4 + 32 + 2 = 38 bits, end mark at bit 38 -> 5 bytes
        let expect: u64 = 0xF | (0xABCD_1234u64 << 4) | (0x3u64 << 36) | (1u64 << 38);
        assert_eq!(w.finish_with_end_mark(), expect.to_le_bytes()[..5].to_vec());
    }

    #[test]
    fn end_mark_alone_on_byte_boundary() {
        let mut w = BitWriter::new();
        w.add_bits(0xA5, 8);
        assert_eq!(w.finish_with_end_mark(), vec![0xA5, 0x01]);
    }
}
