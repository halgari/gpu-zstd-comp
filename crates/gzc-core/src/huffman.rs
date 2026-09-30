//! Huffman table construction and encoding for the literals section.
//!
//! Everything that decides a bit of output (tree construction, length limiting, code assignment,
//! weight compression, section choice) is integer-only and deterministic so a GPU kernel can
//! mirror it exactly. Ground truth: libzstd `huf_compress.c` and `zstd_compress_literals.c`.

use crate::bits::BitWriter;
use crate::fse::{FseCTable, FseState, choose_table_log, normalize, write_ncount};

/// Longest code we emit (zstd's `LitHufLog`; the format allows at most 11).
pub const HUF_MAX_BITS: u32 = 11;
/// Largest table log for the FSE-compressed weights (`MAX_FSE_TABLELOG_FOR_HUFF_HEADER`).
const WEIGHTS_MAX_LOG: u32 = 6;
/// Weights are 0..=11 (a weight is at most `max_bits`), so 12 histogram cells.
const WEIGHT_ALPHABET: usize = HUF_MAX_BITS as usize + 1;
/// Below this many literals we never try Huffman.
pub const MIN_HUF_LITERALS: usize = 64;

/// Huffman code for the literal alphabet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HufTable {
    /// Code length per symbol; 0 for absent symbols.
    pub nb_bits: [u8; 256],
    /// Canonical code per symbol (RFC 8878 §4.2.1), written with `nb_bits` bits.
    pub code: [u16; 256],
    /// Longest code length (`Max_Number_of_Bits`).
    pub max_bits: u32,
    /// Largest present symbol; its weight is implied in the table description.
    pub max_symbol: usize,
}

impl HufTable {
    /// Weight of `s`: `max_bits + 1 - nb_bits`, or 0 if absent.
    pub fn weight(&self, s: usize) -> u8 {
        match self.nb_bits[s] {
            0 => 0,
            b => (self.max_bits + 1 - b as u32) as u8,
        }
    }
}

/// Build a length-limited (<= 11 bits) Huffman code for the histogram `counts`
/// (sum < 2^31). `None` if fewer than two symbols are present.
///
/// Normative construction:
/// 1. sort present symbols by `(count asc, symbol asc)`;
/// 2. two-queue Huffman merge, taking the leaf on count ties;
/// 3. depths are made non-increasing along that order (a no-op for the two-queue tree, kept as a
///    cheap guarantee), then limited to 11 with a port of libzstd's `HUF_setMaxHeight` run on the
///    reversed (count-descending) order;
/// 4. canonical codes per RFC 8878 (see [`from_lengths`]).
pub fn build_table(counts: &[u32; 256]) -> Option<HufTable> {
    // 1. ascending (count, symbol)
    let mut leaves: Vec<(u32, u8)> = (0..256).filter(|&s| counts[s] > 0).map(|s| (counts[s], s as u8)).collect();
    let n = leaves.len();
    if n < 2 {
        return None;
    }
    leaves.sort_unstable();

    // 2. two-queue merge. Node ids: leaves 0..n, internal nodes n..2n-1 in creation order.
    let mut weight: Vec<u32> = leaves.iter().map(|&(c, _)| c).collect();
    weight.reserve(n - 1);
    let mut parent = vec![0usize; 2 * n - 1];
    let (mut next_leaf, mut next_internal) = (0usize, n);
    for _ in 0..n - 1 {
        let mut pick = || {
            let take_leaf =
                next_leaf < n && (next_internal == weight.len() || weight[next_leaf] <= weight[next_internal]);
            if take_leaf {
                next_leaf += 1;
                next_leaf - 1
            } else {
                next_internal += 1;
                next_internal - 1
            }
        };
        let (a, b) = (pick(), pick());
        let new_id = weight.len();
        parent[a] = new_id;
        parent[b] = new_id;
        weight.push(weight[a] + weight[b]);
    }
    // Parents are always created after their children: walk ids downward from the root.
    let mut depth = vec![0u32; 2 * n - 1];
    for id in (0..2 * n - 2).rev() {
        depth[id] = depth[parent[id]] + 1;
    }

    // 3. count-descending order with non-decreasing depths, then limit.
    let mut depths: Vec<u32> = depth[..n].to_vec();
    depths.sort_unstable_by(|a, b| b.cmp(a)); // longest codes go to the smallest counts
    let mut nodes: Vec<(u32, u32)> = (0..n).rev().map(|i| (leaves[i].0, depths[i])).collect();
    set_max_height(&mut nodes, HUF_MAX_BITS);

    let mut nb_bits = [0u8; 256];
    for (i, &(_, bits)) in nodes.iter().enumerate() {
        nb_bits[leaves[n - 1 - i].1 as usize] = bits as u8;
    }
    Some(from_lengths(nb_bits))
}

/// Floor log2 of a nonzero value.
#[inline]
fn highbit(v: u32) -> u32 {
    31 - v.leading_zeros()
}

/// Port of libzstd's `HUF_setMaxHeight`: `nodes` holds `(count, nb_bits)` sorted by count descending
/// with `nb_bits` non-decreasing; afterwards no length exceeds `target` and the Kraft sum is exactly 1.
/// Returns the resulting maximum length.
fn set_max_height(nodes: &mut [(u32, u32)], target: u32) -> u32 {
    const NO_SYMBOL: u32 = 0xF0F0_F0F0;
    let last = nodes.len() - 1;
    let largest = nodes[last].1;
    if largest <= target {
        return largest;
    }
    assert!(largest - target < 30, "tree too deep");

    // Clamp every over-long code to `target`; total_cost is the Kraft excess in units of 2^-largest.
    let base_cost = 1i32 << (largest - target);
    let mut total_cost = 0i32;
    let mut n = last as i32;
    while nodes[n as usize].1 > target {
        total_cost += base_cost - (1i32 << (largest - nodes[n as usize].1));
        nodes[n as usize].1 = target;
        n -= 1;
    }
    // n: largest position (smallest count) using fewer than `target` bits.
    while nodes[n as usize].1 == target {
        n -= 1;
    }
    // renormalize to units of 2^-target (always a multiple of base_cost)
    debug_assert_eq!(total_cost & (base_cost - 1), 0);
    total_cost >>= largest - target;
    debug_assert!(total_cost > 0);

    // rank_last[k]: position of the smallest-count symbol using `target - k` bits.
    let mut rank_last = [NO_SYMBOL; HUF_MAX_BITS as usize + 3];
    let mut current = target;
    for pos in (0..=n).rev() {
        let bits = nodes[pos as usize].1;
        if bits >= current {
            continue;
        }
        current = bits;
        rank_last[(target - current) as usize] = pos as u32;
    }

    while total_cost > 0 {
        // Lengthen one code of the rank that pays back the next power of 2 above total_cost,
        // stepping down a rank while lengthening two codes there is cheaper.
        let mut nb_dec = highbit(total_cost as u32) + 1;
        while nb_dec > 1 {
            let high = rank_last[nb_dec as usize];
            let low = rank_last[nb_dec as usize - 1];
            if high == NO_SYMBOL {
                nb_dec -= 1;
                continue;
            }
            if low == NO_SYMBOL {
                break;
            }
            // lengthen `high` unless two `low` symbols would be cheaper
            if nodes[high as usize].0 <= 2 * nodes[low as usize].0 {
                break;
            }
            nb_dec -= 1;
        }
        while nb_dec <= HUF_MAX_BITS + 1 && rank_last[nb_dec as usize] == NO_SYMBOL {
            nb_dec += 1;
        }
        let pos = rank_last[nb_dec as usize];
        assert!(pos != NO_SYMBOL, "no symbol left to lengthen");
        total_cost -= 1i32 << (nb_dec - 1);
        nodes[pos as usize].1 += 1;

        // The lengthened symbol is now the largest-count one of the next rank, or its only one.
        if rank_last[nb_dec as usize - 1] == NO_SYMBOL {
            rank_last[nb_dec as usize - 1] = pos;
        }
        // Its old rank's smallest symbol is the previous position, if it still belongs to that rank.
        if pos == 0 {
            rank_last[nb_dec as usize] = NO_SYMBOL;
        } else {
            rank_last[nb_dec as usize] = pos - 1;
            if nodes[pos as usize - 1].1 != target - nb_dec {
                rank_last[nb_dec as usize] = NO_SYMBOL;
            }
        }
    }

    // Overshoot: give bits back to the largest-count `target`-bit symbols.
    while total_cost < 0 {
        if rank_last[1] == NO_SYMBOL {
            while nodes[n as usize].1 == target {
                n -= 1;
            }
            nodes[(n + 1) as usize].1 -= 1;
            rank_last[1] = (n + 1) as u32;
            total_cost += 1;
            continue;
        }
        nodes[rank_last[1] as usize + 1].1 -= 1;
        rank_last[1] += 1;
        total_cost += 1;
    }
    target
}

/// Canonical codes for the given lengths (a complete prefix code, lengths <= 11), RFC 8878 §4.2.1:
/// codes are handed out from the longest length (lowest weight) up, in increasing symbol order
/// within a length, the longest codes starting at 0 (libzstd's `nbPerRank` / `valPerRank`).
fn from_lengths(nb_bits: [u8; 256]) -> HufTable {
    let max_bits = nb_bits.iter().copied().max().unwrap_or(0) as u32;
    assert!((1..=HUF_MAX_BITS).contains(&max_bits), "max_bits {max_bits} out of range");
    let max_symbol = nb_bits.iter().rposition(|&b| b > 0).unwrap();
    let mut nb_per_rank = [0u32; HUF_MAX_BITS as usize + 1];
    for &b in &nb_bits {
        nb_per_rank[b as usize] += 1;
    }
    let kraft: u32 = (1..=max_bits).map(|b| nb_per_rank[b as usize] << (max_bits - b)).sum();
    assert_eq!(kraft, 1 << max_bits, "code lengths do not form a complete prefix code");
    let mut val_per_rank = [0u32; HUF_MAX_BITS as usize + 1];
    let mut min = 0u32;
    for b in (1..=max_bits).rev() {
        val_per_rank[b as usize] = min;
        min = (min + nb_per_rank[b as usize]) >> 1;
    }
    let mut code = [0u16; 256];
    for s in 0..256 {
        let b = nb_bits[s] as usize;
        if b > 0 {
            code[s] = val_per_rank[b] as u16;
            val_per_rank[b] += 1;
        }
    }
    HufTable { nb_bits, code, max_bits, max_symbol }
}

/// Huffman tree description (RFC 8878 §4.2.1.1): weights of symbols `0..max_symbol` (the last one
/// is implied). Direct 4-bit form if `max_symbol < 128`, otherwise FSE-compressed weights.
/// `None` if the table cannot be described: the FSE form needs at least two distinct weight values
/// (and some value used twice) and must fit in 127 bytes.
pub fn table_description(t: &HufTable) -> Option<Vec<u8>> {
    let weights: Vec<u8> = (0..t.max_symbol).map(|s| t.weight(s)).collect();
    let mut out = Vec::with_capacity(1 + weights.len().div_ceil(2));
    if weights.len() < 128 {
        out.push(127 + weights.len() as u8);
        for pair in weights.chunks(2) {
            out.push((pair[0] << 4) | pair.get(1).copied().unwrap_or(0));
        }
    } else {
        let c = compress_weights(&weights)?;
        if c.len() >= 128 {
            return None;
        }
        out.push(c.len() as u8);
        out.extend_from_slice(&c);
    }
    Some(out)
}

/// Append the table description; panics if [`table_description`] returns `None`.
pub fn write_table_description(t: &HufTable, out: &mut Vec<u8>) {
    out.extend_from_slice(&table_description(t).expect("Huffman table cannot be described"));
}

/// FSE-compress the weights (port of `HUF_compressWeights` + `FSE_compress_usingCTable`): NCount
/// header (table log <= 6) followed by one bitstream driven by two interleaved states over one table.
/// `None` where libzstd reports "not compressible" (a single weight value, or every value at most once).
fn compress_weights(weights: &[u8]) -> Option<Vec<u8>> {
    let n = weights.len();
    if n <= 2 {
        return None;
    }
    let mut counts = [0u32; WEIGHT_ALPHABET];
    for &w in weights {
        counts[w as usize] += 1;
    }
    let max_count = *counts.iter().max().unwrap();
    if max_count == n as u32 || max_count == 1 {
        return None;
    }
    let max_w = counts.iter().rposition(|&c| c > 0).unwrap();
    let counts = &counts[..=max_w];
    let used = counts.iter().filter(|&&c| c > 0).count();
    // <= 12 used symbols -> the symbol bound (<= 4) is below 6 and the result is 5 or 6,
    // within FseCTable's 5..=12.
    let log = choose_table_log(n as u32, used, WEIGHTS_MAX_LOG);
    let norm = normalize(counts, n as u32, log);
    let mut out = Vec::new();
    write_ncount(&norm, log, &mut out);
    let table = FseCTable::from_normalized(&norm, log);

    // Symbols go in last -> first. State 1 is flushed last, so the decoder reads it first.
    let mut w = BitWriter::new();
    let mut i = n;
    let (mut s1, mut s2);
    if n % 2 == 1 {
        s1 = FseState::init(&table, weights[n - 1]);
        s2 = FseState::init(&table, weights[n - 2]);
        s1.encode(&mut w, weights[n - 3]);
        i -= 3;
    } else {
        s2 = FseState::init(&table, weights[n - 1]);
        s1 = FseState::init(&table, weights[n - 2]);
        i -= 2;
    }
    // i is even here: alternate state 2, state 1.
    while i > 0 {
        s2.encode(&mut w, weights[i - 1]);
        s1.encode(&mut w, weights[i - 2]);
        i -= 2;
    }
    s2.flush(&mut w);
    s1.flush(&mut w);
    out.extend_from_slice(&w.finish_with_end_mark());
    Some(out)
}

/// One Huffman bitstream: symbols encoded last -> first (so the decoder emits them first -> last),
/// then the end mark.
pub fn encode_stream(t: &HufTable, lits: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::new();
    for &b in lits.iter().rev() {
        let nb = t.nb_bits[b as usize] as u32;
        debug_assert!(nb > 0, "symbol {b} not in table");
        w.add_bits(t.code[b as usize] as u32, nb);
    }
    w.finish_with_end_mark()
}

/// Size of a Raw literals section: 1/2/3-byte header + the bytes.
fn raw_section_len(n: usize) -> usize {
    n + 1 + (n >= 32) as usize + (n >= 4096) as usize
}

/// Raw (type 0) / RLE (type 1) literals header: Size_Format 00/01/11 with a 5/12/20-bit size.
pub(crate) fn write_raw_rle_header(lit_type: u32, n: usize, out: &mut Vec<u8>) {
    let n = n as u32;
    assert!(n < 1 << 20, "too many literals: {n}");
    if n < 32 {
        out.push((lit_type | (n << 3)) as u8);
    } else if n < 4096 {
        out.extend_from_slice(&((lit_type | 0b0100 | (n << 4)) as u16).to_le_bytes());
    } else {
        out.extend_from_slice(&(lit_type | 0b1100 | (n << 4)).to_le_bytes()[..3]);
    }
}

/// Compressed_Literals_Block (type 2) for `lits`, or `None` if Huffman cannot code them.
/// One stream if `lits.len() < 256`, else four; the header size follows the larger of the two sizes.
fn compressed_section(lits: &[u8]) -> Option<Vec<u8>> {
    let mut counts = [0u32; 256];
    for &b in lits {
        counts[b as usize] += 1;
    }
    let t = build_table(&counts)?;
    let mut payload = table_description(&t)?;
    let regen = lits.len();
    let single = regen < 256;
    if single {
        payload.extend_from_slice(&encode_stream(&t, lits));
    } else {
        // (regen + 3) / 4 bytes per stream, the last one takes the remainder
        let seg = regen.div_ceil(4);
        let streams: Vec<Vec<u8>> = lits.chunks(seg).map(|c| encode_stream(&t, c)).collect();
        assert_eq!(streams.len(), 4);
        for s in &streams[..3] {
            assert!(s.len() <= 0xFFFF, "stream too large for the jump table");
            payload.extend_from_slice(&(s.len() as u16).to_le_bytes());
        }
        for s in &streams {
            payload.extend_from_slice(s);
        }
    }
    let size = regen.max(payload.len()) as u32;
    let (regen, comp) = (regen as u32, payload.len() as u32);
    let mut out = Vec::with_capacity(5 + payload.len());
    if size < 1 << 10 {
        let sf = if single { 0 } else { 1 };
        out.extend_from_slice(&(2 | (sf << 2) | (regen << 4) | (comp << 14)).to_le_bytes()[..3]);
    } else if size < 1 << 14 {
        out.extend_from_slice(&(2 | (2 << 2) | (regen << 4) | (comp << 18)).to_le_bytes());
    } else {
        assert!(size < 1 << 18, "literals too large: {size}");
        out.extend_from_slice(&(2 | (3 << 2) | (regen << 4) | (comp << 22)).to_le_bytes());
        out.push((comp >> 10) as u8);
    }
    out.extend_from_slice(&payload);
    Some(out)
}

/// Literals section (RFC 8878 §3.1.1.3.1), smallest of Raw / RLE / Compressed:
/// RLE when there are >= 2 literals, all equal; Huffman (Compressed) when there are >= 64 literals
/// and the compressed section is strictly smaller than the raw one; Raw otherwise.
pub fn write_literals_section(lits: &[u8], out: &mut Vec<u8>) {
    if lits.len() >= 2 && lits.iter().all(|&b| b == lits[0]) {
        write_raw_rle_header(1, lits.len(), out);
        out.push(lits[0]);
        return;
    }
    if lits.len() >= MIN_HUF_LITERALS
        && let Some(sec) = compressed_section(lits)
        && sec.len() < raw_section_len(lits.len())
    {
        out.extend_from_slice(&sec);
        return;
    }
    crate::frame::write_literals_raw(lits, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BLOCK_SIZE;

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

    fn histogram(lits: &[u8]) -> [u32; 256] {
        let mut c = [0u32; 256];
        for &b in lits {
            c[b as usize] += 1;
        }
        c
    }

    /// Kraft sum in units of 2^-max_bits; must equal 2^max_bits for a complete prefix code.
    fn kraft(t: &HufTable) -> u32 {
        t.nb_bits.iter().filter(|&&b| b > 0).map(|&b| 1u32 << (t.max_bits - b as u32)).sum()
    }

    fn check_table(t: &HufTable, counts: &[u32; 256]) {
        assert!((1..=11).contains(&t.max_bits), "max_bits {}", t.max_bits);
        assert_eq!(kraft(t), 1 << t.max_bits, "code is not complete");
        for (s, (&c, &b)) in counts.iter().zip(&t.nb_bits).enumerate() {
            assert_eq!(c > 0, b > 0, "symbol {s}");
            assert!(b as u32 <= t.max_bits);
        }
        assert_eq!(t.max_symbol, counts.iter().rposition(|&c| c > 0).unwrap());
        assert_eq!(t.nb_bits.iter().map(|&b| b as u32).max().unwrap(), t.max_bits);
        // prefix-free: no code is a prefix of another
        let present: Vec<usize> = (0..256).filter(|&s| t.nb_bits[s] > 0).collect();
        for &a in &present {
            assert!((t.code[a] as u32) < 1 << t.nb_bits[a], "code of {a} too wide");
            for &b in &present {
                if a != b && t.nb_bits[a] <= t.nb_bits[b] {
                    let shift = t.nb_bits[b] - t.nb_bits[a];
                    assert_ne!(t.code[b] >> shift, t.code[a], "code of {a} is a prefix of {b}");
                }
            }
        }
    }

    #[test]
    fn fewer_than_two_symbols_is_none() {
        assert!(build_table(&[0; 256]).is_none());
        let mut c = [0u32; 256];
        c[42] = 1000;
        assert!(build_table(&c).is_none());
        c[7] = 1;
        let t = build_table(&c).unwrap();
        assert_eq!((t.nb_bits[7], t.nb_bits[42], t.max_bits, t.max_symbol), (1, 1, 1, 42));
    }

    #[test]
    fn canonical_codes_match_rfc_example() {
        // RFC 8878 §4.2.1.1 example: weights 4,3,2,0,1,1 -> bits 1,2,3,0,4,4 -> codes 1,01,001,-,0000,0001
        let mut nb = [0u8; 256];
        nb[..6].copy_from_slice(&[1, 2, 3, 0, 4, 4]);
        let t = from_lengths(nb);
        assert_eq!(t.max_bits, 4);
        assert_eq!(t.max_symbol, 5);
        assert_eq!(&t.code[..6], &[1, 1, 1, 0, 0, 1]);
    }

    #[test]
    fn code_lengths_limited_to_11() {
        // Fibonacci counts give a maximally skewed tree (unlimited depth would be 24).
        let mut c = [0u32; 256];
        let (mut a, mut b) = (1u32, 1u32);
        for s in 0..25 {
            c[s * 3] = a;
            (a, b) = (b, a + b);
        }
        let t = build_table(&c).unwrap();
        check_table(&t, &c);
        assert_eq!(t.max_bits, 11);
        // the most frequent symbol keeps the shortest code
        assert_eq!(t.nb_bits[72], 1);
    }

    #[test]
    fn random_histograms_give_valid_codes() {
        let mut r = Lcg(0x4f);
        for _ in 0..500 {
            let mut c = [0u32; 256];
            let nsym = 2 + r.below(255);
            let cap = [2, 100, 5000, 1 << 16][r.below(4) as usize];
            for _ in 0..nsym {
                c[r.below(256) as usize] += 1 + r.below(cap);
            }
            if c.iter().filter(|&&x| x > 0).count() < 2 {
                continue;
            }
            let t = build_table(&c).unwrap();
            check_table(&t, &c);
            assert_eq!(build_table(&c).unwrap(), t, "deterministic");
        }
    }

    #[test]
    fn length_limiting_stress() {
        // Steep, noisy geometric histograms over many symbols: most need limiting.
        let mut r = Lcg(0x11);
        let mut limited = 0;
        for i in 0..300 {
            let nsym = 20 + r.below(237) as usize;
            let decay = 1 + r.below(12);
            let mut c = [0u32; 256];
            for s in 0..nsym {
                let base = (1u32 << 17) >> ((s as u32 * 8 / decay).min(17));
                c[(s * 7 + i) % 256] = base.max(1) + r.below(3);
            }
            let t = build_table(&c).unwrap();
            check_table(&t, &c);
            limited += (t.max_bits == 11) as u32;
            // literals with the same (scaled) shape must round-trip through libzstd
            let total: u32 = c.iter().sum();
            let shift = highbit(total).saturating_sub(15); // keep the literals under ~64K
            let lits: Vec<u8> =
                (0..256usize).flat_map(|s| std::iter::repeat_n(s as u8, (c[s] >> shift) as usize)).collect();
            let mut sec = Vec::new();
            write_literals_section(&lits, &mut sec);
            decode_literals(&sec, &lits);
        }
        assert!(limited > 100, "only {limited} tables hit the limit");
    }

    #[test]
    fn huffman_is_optimal_when_unconstrained() {
        // counts 1,1,2,4 -> depths 3,3,2,1
        let mut c = [0u32; 256];
        c[..4].copy_from_slice(&[1, 1, 2, 4]);
        let t = build_table(&c).unwrap();
        assert_eq!(&t.nb_bits[..4], &[3, 3, 2, 1]);
        // equal counts: ties resolved by symbol, lengths balanced
        let mut c = [0u32; 256];
        c[..4].copy_from_slice(&[5, 5, 5, 5]);
        assert_eq!(&build_table(&c).unwrap().nb_bits[..4], &[2, 2, 2, 2]);
    }

    #[test]
    fn direct_weights_description() {
        let mut nb = [0u8; 256];
        nb[..6].copy_from_slice(&[1, 2, 3, 0, 4, 4]);
        let t = from_lengths(nb);
        let mut out = Vec::new();
        write_table_description(&t, &mut out);
        // 5 weights written (last implied): 4,3,2,0,1 -> 0x43, 0x20, 0x10
        assert_eq!(out, vec![127 + 5, 0x43, 0x20, 0x10]);
    }

    #[test]
    fn fse_weights_description_for_large_alphabets() {
        // skewed literals over all 256 byte values: max_symbol 255 needs FSE-compressed weights
        let mut r = Lcg(3);
        let mut lits: Vec<u8> = (0..20000).map(|_| (r.next() & r.next()) as u8).collect();
        lits.push(255);
        let t = build_table(&histogram(&lits)).unwrap();
        assert_eq!(t.max_symbol, 255);
        let d = table_description(&t).unwrap();
        assert!(d[0] < 128 && d.len() == 1 + d[0] as usize, "header {}", d[0]);
    }

    #[test]
    fn uniform_weights_cannot_use_fse() {
        // 256 symbols with equal counts: all weights equal -> no FSE description (and no direct form)
        let c = [10u32; 256];
        let t = build_table(&c).unwrap();
        assert!(table_description(&t).is_none());
        let lits: Vec<u8> = (0..2560).map(|i| i as u8).collect();
        let mut out = Vec::new();
        write_literals_section(&lits, &mut out);
        assert_eq!(out[0] & 3, 0, "falls back to raw");
    }

    #[test]
    fn stream_is_end_marked_and_reversed() {
        let mut nb = [0u8; 256];
        nb[..6].copy_from_slice(&[1, 2, 3, 0, 4, 4]);
        let t = from_lengths(nb);
        // symbols encoded last -> first: 5 (0001), then 0 (1), then 1 (01), end mark
        // bits LSB-first: 0001 | 1 | 01 | 1 -> value 0b1_01_1_0001
        assert_eq!(encode_stream(&t, &[1, 0, 5]), vec![0b1011_0001]);
        // one more bit spills into a second byte holding only the end mark
        assert_eq!(encode_stream(&t, &[0, 1, 0, 5]), vec![0b1011_0001, 0b1]);
    }

    /// Literals section type (low 2 bits of the first header byte).
    fn section_type(sec: &[u8]) -> u8 {
        sec[0] & 3
    }

    /// Wrap a literals section plus an empty sequences section into a frame and decode with libzstd.
    fn decode_literals(section: &[u8], lits: &[u8]) {
        let mut content = section.to_vec();
        content.push(0); // nbSeq = 0
        // no Frame_Content_Size (so any length decodes): descriptor 0, window descriptor 2^(10+8)
        let mut f = vec![0x28, 0xB5, 0x2F, 0xFD, 0x00, 8 << 3];
        f.extend_from_slice(&(1 | (2 << 1) | ((content.len() as u32) << 3)).to_le_bytes()[..3]);
        f.extend_from_slice(&content);
        let dec = zstd::bulk::decompress(&f, BLOCK_SIZE.max(lits.len())).expect("libzstd rejected literals");
        assert!(dec == lits, "decoded literals differ");
    }

    #[test]
    fn literals_sections_roundtrip_through_libzstd() {
        let mut r = Lcg(99);
        let sizes = [64, 100, 255, 256, 257, 1023, 1024, 5000, 16383, 16384, 70000, BLOCK_SIZE];
        let mut seen = std::collections::BTreeSet::new();
        for &n in &sizes {
            if n > BLOCK_SIZE {
                continue;
            }
            for kind in 0..3 {
                let lits: Vec<u8> = (0..n)
                    .map(|_| match kind {
                        0 => (r.below(6) * r.below(6)) as u8,
                        1 => (r.next() & r.next()) as u8,
                        _ => (r.below(40) + r.below(40)) as u8 + 32,
                    })
                    .collect();
                let mut sec = Vec::new();
                write_literals_section(&lits, &mut sec);
                if kind == 0 || n >= 1024 {
                    assert_eq!(section_type(&sec), 2, "n={n} kind={kind} not compressed");
                }
                if section_type(&sec) == 2 {
                    seen.insert((sec[0] >> 2) & 3);
                }
                decode_literals(&sec, &lits);
            }
        }
        let want: std::collections::BTreeSet<u8> = [0, 1, 2, 3].into();
        assert_eq!(seen, want, "size formats exercised");
    }

    #[test]
    fn literals_section_raw_rle_choices() {
        for n in [0usize, 1, 5, 31, 32, 63, 4095, 4096] {
            // short or incompressible -> raw
            let mut r = Lcg(n as u64);
            let lits: Vec<u8> = (0..n).map(|_| r.next() as u8).collect();
            let mut sec = Vec::new();
            write_literals_section(&lits, &mut sec);
            let mut raw = Vec::new();
            crate::frame::write_literals_raw(&lits, &mut raw);
            if n < 64 || section_type(&sec) != 2 {
                assert_eq!(sec, raw, "n={n}");
            }
            if n > 0 {
                decode_literals(&sec, &lits);
            }
        }
        // compressible but shorter than 64 -> raw
        let lits = b"aaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbc".to_vec();
        let mut sec = Vec::new();
        write_literals_section(&lits, &mut sec);
        assert_eq!(section_type(&sec), 0);
        // all equal -> RLE, header scheme like raw
        for (n, hdr) in [(2usize, vec![(2 << 3) | 1]), (31, vec![(31 << 3) | 1]), (32, vec![0x05, 0x02]), (4096, vec![0x0D, 0x00, 0x01])] {
            let lits = vec![0x61u8; n];
            let mut sec = Vec::new();
            write_literals_section(&lits, &mut sec);
            let mut want = hdr.clone();
            want.push(0x61);
            assert_eq!(sec, want, "rle n={n}");
            decode_literals(&sec, &lits);
        }
    }
}
