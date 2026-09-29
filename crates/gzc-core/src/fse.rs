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
    /// Build from a normalized count (`-1` marks a low-probability symbol).
    pub fn from_normalized(norm: &[i16], table_log: u32) -> FseCTable {
        assert!((1..16).contains(&table_log), "table_log {table_log} out of range");
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
}
