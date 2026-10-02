//! FSE compression table construction and encoding (port of libzstd's
//! `FSE_buildCTable_wksp`, `FSE_initCState2`, `FSE_encodeSymbol`, `FSE_flushCState`).

use crate::bits::BitWriter;

/// FSE compression table for one normalized distribution.
#[derive(Clone, Debug)]
pub struct FseCTable {
    /// Next-state table, indexed by `cumul[symbol] + rank`; entries are `table_size + position`.
    state_table: Vec<u16>,
    /// Per symbol: (delta_find_state, delta_nb_bits).
    symbol_tt: Vec<(i32, u32)>,
    table_log: u32,
}

impl FseCTable {
    /// Build from a normalized count (`-1` marks a low-probability symbol); `table_log` in 5..=12
    /// (libzstd's `FSE_MIN_TABLELOG..=FSE_MAX_TABLELOG`). For a single-symbol stream use [`FseCTable::rle`].
    pub fn from_normalized(norm: &[i16], table_log: u32) -> FseCTable {
        assert!(
            (FSE_MIN_TABLE_LOG..=FSE_MAX_TABLE_LOG).contains(&table_log),
            "table_log {table_log} out of range"
        );
        assert!(norm.len() <= 256, "at most 256 symbols");
        let size = 1u32 << table_log;
        let mask = size - 1;
        let step = (size >> 1) + (size >> 3) + 3;

        // Symbol start positions; low-probability (-1) symbols take the top cells.
        let mut cumul = vec![0u32; norm.len() + 1];
        let mut table_symbol = vec![0u8; size as usize];
        let mut high = size - 1;
        for (s, &n) in norm.iter().enumerate() {
            if n == -1 {
                cumul[s + 1] = cumul[s] + 1;
                table_symbol[high as usize] = s as u8;
                high = high.wrapping_sub(1);
            } else {
                assert!(n >= 0, "invalid normalized count {n}");
                cumul[s + 1] = cumul[s] + n as u32;
            }
        }
        assert_eq!(cumul[norm.len()], size, "normalized counts must sum to the table size");

        // Spread the remaining symbols.
        let mut pos = 0u32;
        for (s, &n) in norm.iter().enumerate() {
            for _ in 0..n.max(0) {
                table_symbol[pos as usize] = s as u8;
                pos = (pos + step) & mask;
                while pos > high {
                    pos = (pos + step) & mask;
                }
            }
        }
        assert_eq!(pos, 0, "spread did not cover the table");

        // State table, sorted by symbol.
        let mut state_table = vec![0u16; size as usize];
        for (u, &s) in table_symbol.iter().enumerate() {
            let c = &mut cumul[s as usize];
            state_table[*c as usize] = (size + u as u32) as u16;
            *c += 1;
        }

        // Symbol transformation table.
        let mut symbol_tt = vec![(0i32, 0u32); norm.len()];
        let mut total = 0i32;
        for (s, &n) in norm.iter().enumerate() {
            symbol_tt[s] = match n {
                0 => (0, ((table_log + 1) << 16) - size),
                -1 | 1 => {
                    let tt = (total - 1, (table_log << 16) - size);
                    total += 1;
                    tt
                }
                _ => {
                    let n = n as u32;
                    let max_bits_out = table_log - (31 - (n - 1).leading_zeros());
                    let min_state_plus = n << max_bits_out;
                    let tt = (total - n as i32, (max_bits_out << 16) - min_state_plus);
                    total += n as i32;
                    tt
                }
            };
        }

        FseCTable { state_table, symbol_tt, table_log }
    }

    /// Table for a stream that repeats one symbol (port of `FSE_buildCTable_rle`): table log 0,
    /// `deltaNbBits` 0 and a one-entry state table, so `FseState` emits no bits at all.
    pub fn rle(symbol: u8) -> FseCTable {
        FseCTable { state_table: vec![0], symbol_tt: vec![(0, 0); symbol as usize + 1], table_log: 0 }
    }

    /// The table's log2 size.
    pub fn table_log(&self) -> u32 {
        self.table_log
    }

    /// Next-state table (`1 << table_log` entries; one entry for an RLE table).
    pub fn state_table(&self) -> &[u16] {
        &self.state_table
    }

    /// Per symbol: (delta_find_state, delta_nb_bits).
    pub fn symbol_tt(&self) -> &[(i32, u32)] {
        &self.symbol_tt
    }

    #[inline]
    fn next_state(&self, value: u32, nb: u32, symbol: u8) -> u32 {
        let dfs = self.symbol_tt[symbol as usize].0;
        self.state_table[((value >> nb) as i32 + dfs) as usize] as u32
    }
}

/// Encoder state bound to one table.
pub struct FseState<'a> {
    value: u32,
    table: &'a FseCTable,
}

impl<'a> FseState<'a> {
    /// Initialize with the first symbol to encode (the last one the decoder reads), emitting no bits.
    pub fn init(t: &'a FseCTable, symbol: u8) -> Self {
        let dnb = t.symbol_tt[symbol as usize].1;
        let nb = (dnb + (1 << 15)) >> 16;
        // No u32 underflow: dnb = (k << 16) - m with 0 < m <= 2 * size <= 2^13 (m = n << maxBitsOut for
        // n >= 2, m = size for n in {-1, 0, 1}), so nb rounds up to k and v = m. This needs
        // 2 * size <= 2^15, i.e. table_log <= 14; at most 12 is allowed. RLE tables: dnb = 0, v = 0.
        let v = (nb << 16) - dnb;
        FseState { value: t.next_state(v, nb, symbol), table: t }
    }

    /// Emit the bits for `symbol` and move to the next state.
    pub fn encode(&mut self, w: &mut BitWriter, symbol: u8) {
        let dnb = self.table.symbol_tt[symbol as usize].1;
        let nb = (self.value + dnb) >> 16;
        w.add_bits(self.value, nb);
        self.value = self.table.next_state(self.value, nb, symbol);
    }

    /// Emit the final state (`table_log` low bits).
    pub fn flush(&self, w: &mut BitWriter) {
        w.add_bits(self.value, self.table.table_log);
    }
}

/// Smallest table log a table description can carry (`FSE_MIN_TABLELOG`).
pub const FSE_MIN_TABLE_LOG: u32 = 5;
/// Largest table log built (`FSE_MAX_TABLELOG`); sequence streams stop at 9 (LL/ML) and 8 (OF).
pub const FSE_MAX_TABLE_LOG: u32 = 12;

/// Floor log2 of a nonzero value.
#[inline]
fn highbit(v: u32) -> u32 {
    debug_assert!(v != 0);
    31 - v.leading_zeros()
}

/// `FRAC[i] = round(256 * log2(1 + i/256))`: fractional part of log2 in 1/256 bits.
/// Public so the GPU cost model (K4) can upload the exact table.
pub const FRAC: [u16; 256] = [
    0, 1, 3, 4, 6, 7, 9, 10, 11, 13, 14, 16, 17, 18, 20, 21, //
    22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, //
    44, 45, 46, 47, 49, 50, 51, 52, 54, 55, 56, 57, 59, 60, 61, 62, //
    63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, //
    82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, //
    100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, //
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, //
    134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, //
    150, 151, 152, 153, 154, 155, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, //
    165, 166, 167, 168, 169, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 178, //
    179, 180, 181, 182, 183, 184, 185, 185, 186, 187, 188, 189, 190, 191, 192, 192, //
    193, 194, 195, 196, 197, 198, 198, 199, 200, 201, 202, 203, 203, 204, 205, 206, //
    207, 208, 208, 209, 210, 211, 212, 212, 213, 214, 215, 216, 216, 217, 218, 219, //
    220, 220, 221, 222, 223, 224, 224, 225, 226, 227, 228, 228, 229, 230, 231, 231, //
    232, 233, 234, 234, 235, 236, 237, 238, 238, 239, 240, 241, 241, 242, 243, 244, //
    244, 245, 246, 247, 247, 248, 249, 249, 250, 251, 252, 252, 253, 254, 255, 255,
];

/// Integer log2 in 1/256 bits for `1 <= x <= 1 << FSE_MAX_TABLE_LOG`.
#[inline]
fn log2_x256(x: u32) -> u32 {
    let hb = highbit(x);
    256 * hb + FRAC[((x << 8 >> hb) & 255) as usize] as u32
}

/// Integer-only normalization (GPU-portable). `counts`: histogram; `total` = its sum; returns `norm`
/// (same length as `counts`) summing to `1 << table_log`, where every symbol with count > 0 gets
/// norm >= 1 (never -1 in this encoder) and absent symbols get 0.
///
/// `n = max(1, c * size / total)`, then the rounding error `size - sum(n)` is fixed up: a surplus goes
/// to the symbol with the largest count (lowest index on ties); a deficit is taken 1 at a time from the
/// symbol with the largest `n > 1` (lowest index on ties). `c <= 2^17` and `size <= 2^9` in practice, so
/// `c * size` fits in u32 (checked). Requires more table cells than used symbols (see `choose_table_log`).
pub fn normalize(counts: &[u32], total: u32, table_log: u32) -> Vec<i16> {
    assert!((FSE_MIN_TABLE_LOG..=FSE_MAX_TABLE_LOG).contains(&table_log), "table_log {table_log} out of range");
    assert!(total > 0, "empty histogram");
    let size = 1u32 << table_log;
    let mut norm = vec![0i16; counts.len()];
    let mut sum = 0u32;
    let mut largest = 0usize; // symbol with the largest count, lowest index on ties
    for (s, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let n = (c.checked_mul(size).expect("count * table size overflows u32") / total).max(1);
        norm[s] = n as i16;
        sum += n;
        if c > counts[largest] {
            largest = s;
        }
    }
    assert!(sum > 0 && counts.iter().sum::<u32>() == total, "total does not match counts");
    if sum < size {
        norm[largest] += (size - sum) as i16;
    }
    while sum > size {
        let mut best = usize::MAX;
        for (s, &n) in norm.iter().enumerate() {
            if n > 1 && (best == usize::MAX || n > norm[best]) {
                best = s;
            }
        }
        assert!(best != usize::MAX, "more used symbols than table cells");
        norm[best] -= 1;
        sum -= 1;
    }
    norm
}

/// Table log for a stream with `total` symbols of which `nb_used` are distinct:
/// `clamp(highbit(total) - 2, 5, max_log)`, but at least `highbit(nb_used) + 1` so the table has more
/// cells than used symbols (normalization needs `size >= nb_used`). `max_log` is 9 for LL/ML and 8 for OF;
/// the result is always within `5..=max_log` (every sequence alphabet has <= 53 symbols, so the
/// symbol bound (<= 6) never exceeds `max_log`).
pub fn choose_table_log(total: u32, nb_used: usize, max_log: u32) -> u32 {
    assert!((FSE_MIN_TABLE_LOG..=FSE_MAX_TABLE_LOG).contains(&max_log), "max_log {max_log} out of range");
    assert!(total > 0 && nb_used > 0 && nb_used as u32 <= total);
    let by_total = highbit(total).saturating_sub(2).clamp(FSE_MIN_TABLE_LOG, max_log);
    let by_symbols = highbit(nb_used as u32) + 1;
    let log = by_total.max(by_symbols);
    assert!(log <= max_log, "{nb_used} symbols do not fit a table of log {max_log}");
    log
}

/// Append an `FSE_writeNCount`-compatible table description (RFC 8878 §4.1.1) for `norm`
/// (port of `FSE_writeNCount_generic` from libzstd's `fse_compress.c`). Trailing zero entries of
/// `norm` are never written: the encoding stops as soon as the whole table is accounted for.
pub fn write_ncount(norm: &[i16], table_log: u32, out: &mut Vec<u8>) {
    assert!((FSE_MIN_TABLE_LOG..=FSE_MAX_TABLE_LOG).contains(&table_log), "table_log {table_log} out of range");
    let table_size = 1i32 << table_log;
    let mut bit_stream: u32 = table_log - FSE_MIN_TABLE_LOG;
    let mut bit_count: u32 = 4;
    let mut remaining = table_size + 1; // +1 for extra accuracy
    let mut threshold = table_size;
    let mut nb_bits = table_log + 1;
    let mut symbol = 0usize;
    let mut previous_is0 = false;

    // Flush 16 bits once more than 16 are pending (never more than 32 are pending).
    let flush16 = |out: &mut Vec<u8>, bit_stream: &mut u32, bit_count: &mut u32| {
        out.extend_from_slice(&(*bit_stream as u16).to_le_bytes());
        *bit_stream >>= 16;
        *bit_count -= 16;
    };

    while symbol < norm.len() && remaining > 1 {
        if previous_is0 {
            // Run of zero-probability symbols: repeat flags of 2 bits (3 = "3 more zeros").
            let mut start = symbol;
            while symbol < norm.len() && norm[symbol] == 0 {
                symbol += 1;
            }
            assert!(symbol < norm.len(), "distribution ends in zeros before the table is full");
            while symbol >= start + 24 {
                start += 24;
                // bit_count <= 16 here, so the shift fits; 16 bits go out and 16 flag bits come in,
                // leaving bit_count unchanged.
                bit_stream += 0xFFFF << bit_count;
                out.extend_from_slice(&(bit_stream as u16).to_le_bytes());
                bit_stream >>= 16;
            }
            while symbol >= start + 3 {
                start += 3;
                bit_stream += 3 << bit_count;
                bit_count += 2;
            }
            bit_stream += ((symbol - start) as u32) << bit_count;
            bit_count += 2;
            if bit_count > 16 {
                flush16(out, &mut bit_stream, &mut bit_count);
            }
        }
        let mut count = norm[symbol] as i32;
        symbol += 1;
        let max = (2 * threshold - 1) - remaining;
        remaining -= count.abs();
        count += 1; // +1 for extra accuracy
        if count >= threshold {
            count += max; // [0..max[ [max..threshold[ (...) [threshold+max 2*threshold[
        }
        bit_stream += (count as u32) << bit_count;
        bit_count += nb_bits;
        bit_count -= (count < max) as u32;
        previous_is0 = count == 1;
        assert!(remaining >= 1, "normalized counts exceed the table size");
        while remaining < threshold {
            nb_bits -= 1;
            threshold >>= 1;
        }
        if bit_count > 16 {
            flush16(out, &mut bit_stream, &mut bit_count);
        }
    }
    assert_eq!(remaining, 1, "normalized counts do not sum to the table size");
    out.extend_from_slice(&bit_stream.to_le_bytes()[..bit_count.div_ceil(8) as usize]);
}

/// Estimated cost, in 1/256 bits, of encoding a stream with histogram `counts` using `norm`:
/// `sum counts[s] * (256 * table_log - log2_x256(norm[s]))`, i.e. `-log2(p)` per symbol with -1
/// (low probability) treated as 1. Every symbol with a count must have a nonzero `norm` entry.
///
/// Integer-only (GPU-portable). The sum is bounded by `nbSeq * 256 * table_log <= 2^17 * 256 * 9 < 2^29`,
/// so the GPU port can use u32 (saturating as a guard); u64 here is a host convenience only.
pub fn cost_x256(counts: &[u32], norm: &[i16], table_log: u32) -> u64 {
    let full = 256 * table_log;
    let mut cost = 0u64;
    for (s, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let n = norm.get(s).copied().unwrap_or(0);
        assert!(n != 0, "symbol {s} has a count but no probability");
        let n = n.unsigned_abs() as u32; // -1 costs like 1
        cost += c as u64 * (full - log2_x256(n)) as u64;
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::*;

    fn check_permutation(norm: &[i16], log: u32) {
        let t = FseCTable::from_normalized(norm, log);
        let size = 1u16 << log;
        let mut v = t.state_table.clone();
        v.sort_unstable();
        assert_eq!(v, (size..2 * size).collect::<Vec<u16>>());
        assert_eq!(t.symbol_tt.len(), norm.len());
    }

    #[test]
    fn default_tables_state_table_is_permutation() {
        check_permutation(&LL_DEFAULT_NORM, LL_DEFAULT_LOG);
        check_permutation(&ML_DEFAULT_NORM, ML_DEFAULT_LOG);
        check_permutation(&OF_DEFAULT_NORM, OF_DEFAULT_LOG);
    }

    #[test]
    fn no_low_prob_symbols_table_builds() {
        // 32-entry table without -1 entries (the spread never skips a high area).
        check_permutation(&[8, 8, 4, 4, 2, 2, 1, 1, 1, 1], 5);
    }

    #[test]
    fn accessors_expose_table() {
        let t = FseCTable::from_normalized(&OF_DEFAULT_NORM, OF_DEFAULT_LOG);
        assert_eq!(t.table_log(), OF_DEFAULT_LOG);
        assert_eq!(t.state_table().len(), 1 << OF_DEFAULT_LOG);
        assert_eq!(t.symbol_tt().len(), OF_DEFAULT_NORM.len());
    }

    #[test]
    #[should_panic(expected = "table_log")]
    fn from_normalized_rejects_log_below_5() {
        FseCTable::from_normalized(&[8, 8], 4);
    }

    #[test]
    #[should_panic(expected = "table_log")]
    fn from_normalized_rejects_log_above_12() {
        FseCTable::from_normalized(&[4096, 4096], 13);
    }

    #[test]
    fn rle_table_emits_no_bits() {
        let t = FseCTable::rle(7);
        assert_eq!(t.table_log(), 0);
        assert_eq!(t.symbol_tt()[7], (0, 0));
        let mut w = BitWriter::new();
        let mut st = FseState::init(&t, 7);
        for _ in 0..100 {
            st.encode(&mut w, 7);
        }
        st.flush(&mut w);
        assert_eq!(w.finish_with_end_mark(), vec![1], "RLE stream must not emit state bits");
    }

    #[test]
    fn init_never_underflows_at_max_log() {
        // A symbol owning the whole table, skewed and even splits, at the max log 12.
        for norm in [vec![4096i16], vec![4095, 1], vec![1, 4095], vec![2048, 2048]] {
            let t = FseCTable::from_normalized(&norm, 12);
            for s in 0..norm.len() as u8 {
                let st = FseState::init(&t, s);
                assert!((4096..8192).contains(&st.value), "state {} out of range", st.value);
            }
        }
    }

    /// Small deterministic PRNG.
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

    #[test]
    fn normalize_random_histograms() {
        let mut r = Lcg(0xf5e);
        for i in 0..1000 {
            let nsym = 1 + r.below(53) as usize;
            let cap = [2, 50, 5000, 1 << 17][r.below(4) as usize];
            let mut counts: Vec<u32> = (0..nsym).map(|_| if r.below(3) == 0 { 0 } else { r.below(cap) }).collect();
            if counts.iter().all(|&c| c == 0) {
                counts[r.below(nsym as u32) as usize] = 1;
            }
            // keep the total within a block's worth of sequences
            while counts.iter().sum::<u32>() > 1 << 17 {
                for c in counts.iter_mut() {
                    *c = c.div_ceil(2);
                }
            }
            let total: u32 = counts.iter().sum();
            let used = counts.iter().filter(|&&c| c > 0).count();
            let max_log = [8, 9][i % 2];
            let log = choose_table_log(total, used, max_log);
            assert!((5..=max_log).contains(&log));
            let norm = normalize(&counts, total, log);
            assert_eq!(norm.len(), counts.len());
            assert_eq!(norm.iter().map(|&n| n as i32).sum::<i32>(), 1 << log, "case {i}: {counts:?}");
            for (s, (&c, &n)) in counts.iter().zip(&norm).enumerate() {
                if c > 0 {
                    assert!(n >= 1, "case {i}: symbol {s} lost");
                } else {
                    assert_eq!(n, 0, "case {i}: symbol {s} gained");
                }
            }
            FseCTable::from_normalized(&norm, log);
        }
    }

    #[test]
    fn normalize_diff_rules() {
        assert_eq!(normalize(&[2, 2, 4], 8, 5), vec![8, 8, 16]);
        // diff > 0 goes to the largest count, lowest index on ties
        assert_eq!(normalize(&[1, 1, 1], 3, 5), vec![12, 10, 10]);
        assert_eq!(normalize(&[1, 3, 3], 7, 5), vec![4, 15, 13]);
        // diff < 0 comes off the largest n > 1, lowest index on ties
        assert_eq!(normalize(&[97, 1, 1, 1], 100, 5), vec![29, 1, 1, 1]);
        assert_eq!(normalize(&[0, 50, 50, 1, 1, 1, 1], 104, 5), vec![0, 14, 14, 1, 1, 1, 1]);
    }

    #[test]
    fn choose_table_log_bounds() {
        assert_eq!(choose_table_log(1000, 10, 9), 7);
        assert_eq!(choose_table_log(3, 2, 9), 5);
        assert_eq!(choose_table_log(1, 1, 8), 5);
        assert_eq!(choose_table_log(1 << 17, 30, 9), 9);
        assert_eq!(choose_table_log(1 << 17, 30, 8), 8);
        // enough cells for every used symbol
        assert_eq!(choose_table_log(40, 36, 9), 6);
    }

    #[test]
    fn frac_table_matches_log2() {
        for (i, &f) in FRAC.iter().enumerate() {
            let want = (256.0 * (1.0 + i as f64 / 256.0).log2()).round() as u16;
            assert_eq!(f, want, "FRAC[{i}]");
        }
        assert_eq!(log2_x256(1), 0);
        assert_eq!(log2_x256(16), 4 * 256);
        assert_eq!(log2_x256(3), 256 + FRAC[128] as u32);
    }

    #[test]
    fn cost_x256_examples() {
        // two equiprobable symbols: 1 bit each
        assert_eq!(cost_x256(&[4, 4], &[16, 16], 5), 8 * 256);
        // -1 (low probability) costs the full table log; absent symbols cost nothing
        assert_eq!(cost_x256(&[1, 0, 3], &[-1, 1, 30], 5), 5 * 256 + 3 * (5 * 256 - log2_x256(30) as u64));
        // a better-fitting distribution is cheaper
        let counts = [90, 5, 5];
        assert!(cost_x256(&counts, &[28, 2, 2], 5) < cost_x256(&counts, &[12, 10, 10], 5));
    }

    #[test]
    fn write_ncount_hand_computed() {
        // log 5: nibble 0; symbol 0: 16+1 = 17 in 5 bits (17 < max 30); symbol 1: 17 >= threshold 16
        // -> 17 + max 14 = 31 in 5 bits. Bits: 17 << 4 | 31 << 9 = 0x3F10, 14 bits -> 2 bytes.
        let mut out = Vec::new();
        write_ncount(&[16, 16], 5, &mut out);
        assert_eq!(out, vec![0x10, 0x3F]);
        let mut out = vec![0xAA];
        write_ncount(&[256, 256], 9, &mut out);
        assert_eq!(out[0], 0xAA, "appends");
        assert_eq!(out[1] & 0xF, 4);
    }
}
