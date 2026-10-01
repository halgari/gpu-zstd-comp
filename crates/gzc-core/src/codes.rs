//! Literal-length, match-length and offset code tables, and zstd's predefined FSE distributions.
//!
//! Values checked against libzstd 1.5.7 (`common/zstd_internal.h`,
//! `decompress/zstd_decompress_internal.h`, `compress/zstd_compress_internal.h`).

pub const LL_BASE: [u32; 36] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, //
    16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096, //
    8192, 16384, 32768, 65536,
];
pub const LL_BITS: [u8; 36] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, //
    13, 14, 15, 16,
];
pub const ML_BASE: [u32; 53] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, //
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, //
    35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, //
    4099, 8195, 16387, 32771, 65539,
];
pub const ML_BITS: [u8; 53] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, //
    12, 13, 14, 15, 16,
];

pub const LL_DEFAULT_NORM: [i16; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, //
    2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, //
    -1, -1, -1, -1,
];
pub const LL_DEFAULT_LOG: u32 = 6;
pub const ML_DEFAULT_NORM: [i16; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, //
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, //
    -1, -1, -1, -1, -1,
];
pub const ML_DEFAULT_LOG: u32 = 6;
pub const OF_DEFAULT_NORM: [i16; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, //
    1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];
pub const OF_DEFAULT_LOG: u32 = 5;

/// Prior LL-code frequencies for the optimal parse's `Seed::Prior` (`opt::seed_prices`, preset
/// `opt14`). Trained on blocks block-disjoint from the m5-opt-design 1/50 evaluation sample
/// (every 50th block, offset 0): the summed LL/ML/OF code histograms of `opt16`'s output over every 50th
/// 64 KiB block at offset 25 of `data/corpus` (`--ext dds,nif` order, 2015 blocks), each table
/// scaled to 65536 (round to nearest), produced by
/// `cargo run --release -p gzc-core --example opt_sample -- train data/corpus 50 25`.
pub const OPT_PRIOR_LL: [u32; 36] = [10175, 5362, 2777, 4261, 18544, 5612, 2888, 1485, 2814, 954, 666, 1004, 3403, 1926, 526, 248, 333, 190, 397, 187, 336, 561, 224, 238, 183, 181, 49, 13, 3, 0, 0, 0, 0, 0, 0, 0];
/// Prior ML-code frequencies (see `OPT_PRIOR_LL`).
pub const OPT_PRIOR_ML: [u32; 53] = [25759, 27428, 4277, 1151, 574, 3224, 654, 830, 269, 673, 131, 66, 68, 101, 18, 22, 24, 13, 6, 6, 12, 18, 5, 7, 9, 18, 5, 4, 4, 5, 5, 4, 6, 3, 7, 6, 9, 10, 9, 15, 15, 8, 12, 17, 11, 7, 3, 6, 0, 0, 0, 0, 0];
/// Prior OF-code frequencies (see `OPT_PRIOR_LL`).
pub const OPT_PRIOR_OF: [u32; 32] = [4961, 3407, 239, 594, 1319, 1327, 1638, 2138, 2903, 3911, 5472, 7608, 9595, 9986, 7960, 2479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
/// Prior LL-code frequencies of `PriorTables::S3` (`Seed::Prior` in preset `opt16p1`, M6 B0).
///
/// Training set: the same blocks as `OPT_PRIOR_*`, every 50th 64 KiB block at offset 25 of
/// `data/corpus` (`--ext dds,nif` order, 2015 blocks), so block-disjoint from the 1/50
/// evaluation sample (every 50th block, offset 0). Training parse: `opt16`'s schedule
/// (`BlockInit` seed, 3 cheap passes, the optLevel-2 final pass) over `opt16p1`'s candidates (h4
/// 8 deep, h3, the S3 sparse chains) and gap3 segment ends, with no relaxation pruning and no
/// drop pass (`opt_sample.rs`'s `s3_train_params`). Each table is the summed code histogram of
/// that parse's output, scaled to 65536 (round to nearest). Regenerate (byte-identical) with
/// `cargo run --release -p gzc-core --example opt_sample -- train-s3 data/corpus 50 25`.
///
/// The B0 study measured the sensitivity on the full corpus at 64 KiB (without pruning): these
/// tables 1.37235, block-disjoint cross-fitted tables 1.37236, cross-mod tables 1.37215, the M5
/// `OPT_PRIOR_*` tables 1.37211.
pub const OPT_PRIOR_S3_LL: [u32; 36] = [9299, 5308, 2743, 3874, 19329, 5865, 3657, 1371, 2248, 969, 685, 949, 3552, 1971, 534, 253, 338, 193, 400, 190, 337, 570, 227, 240, 185, 183, 50, 13, 3, 0, 0, 0, 0, 0, 0, 0];
/// Prior ML-code frequencies of `PriorTables::S3` (see `OPT_PRIOR_S3_LL`).
pub const OPT_PRIOR_S3_ML: [u32; 53] = [25790, 27806, 3674, 1205, 579, 1714, 470, 1670, 446, 1431, 169, 72, 74, 100, 17, 23, 24, 14, 5, 5, 10, 22, 5, 8, 10, 18, 5, 4, 4, 6, 4, 3, 5, 3, 9, 7, 9, 10, 10, 14, 15, 8, 13, 18, 11, 7, 4, 6, 0, 0, 0, 0, 0];
/// Prior OF-code frequencies of `PriorTables::S3` (see `OPT_PRIOR_S3_LL`).
pub const OPT_PRIOR_S3_OF: [u32; 32] = [3643, 3103, 243, 608, 1327, 1337, 1659, 2174, 2954, 4059, 5697, 7886, 9904, 10252, 8139, 2550, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// Floor log2 of a nonzero value.
fn highbit(v: u32) -> u32 {
    debug_assert!(v != 0);
    31 - v.leading_zeros()
}

/// Literal-length code (port of `ZSTD_LLcode`).
pub fn ll_code(lit_len: u32) -> u8 {
    const LL_CODE: [u8; 64] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, //
        16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, //
        22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, //
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    ];
    if lit_len > 63 { (highbit(lit_len) + 19) as u8 } else { LL_CODE[lit_len as usize] }
}

/// Match-length code for a real match length (>= 3) (port of `ZSTD_MLcode(matchLength - MINMATCH)`).
pub fn ml_code(match_len: u32) -> u8 {
    const ML_CODE: [u8; 128] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, //
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, //
        32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, //
        38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, //
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, //
        41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, //
        42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, //
        42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    ];
    debug_assert!(match_len >= 3);
    let ml_base = match_len - 3;
    if ml_base > 127 { (highbit(ml_base) + 36) as u8 } else { ML_CODE[ml_base as usize] }
}

/// Offset code: floor log2 of `off_base`; the code is also its number of extra bits.
pub fn of_code(off_base: u32) -> u8 {
    highbit(off_base) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference definition: the largest code whose base is <= value.
    fn largest_base_le(base: &[u32], v: u32) -> u8 {
        base.iter().rposition(|&b| b <= v).unwrap() as u8
    }

    #[test]
    fn ll_code_matches_base_table() {
        // lit_len < 128K always (a sequence also carries a match of >= 3 bytes)
        for ll in 0..(1u32 << 17) {
            let c = ll_code(ll);
            assert_eq!(c, largest_base_le(&LL_BASE, ll), "ll {ll}");
            assert!(ll - LL_BASE[c as usize] < (1 << LL_BITS[c as usize]), "ll {ll}");
        }
    }

    #[test]
    fn ml_code_matches_base_table() {
        for ml in 3..=(1u32 << 17) {
            let c = ml_code(ml);
            assert_eq!(c, largest_base_le(&ML_BASE, ml), "ml {ml}");
            assert!(ml - ML_BASE[c as usize] < (1 << ML_BITS[c as usize]), "ml {ml}");
        }
    }

    #[test]
    fn of_code_is_highbit() {
        assert_eq!(of_code(1), 0);
        assert_eq!(of_code(2), 1);
        assert_eq!(of_code(3), 1);
        assert_eq!(of_code(4), 2);
        assert_eq!(of_code((1 << 17) + 3), 17);
    }

    #[test]
    fn default_norms_sum_to_table_size() {
        let sum = |n: &[i16]| n.iter().map(|&x| x.unsigned_abs() as u32).sum::<u32>();
        assert_eq!(sum(&LL_DEFAULT_NORM), 1 << LL_DEFAULT_LOG);
        assert_eq!(sum(&ML_DEFAULT_NORM), 1 << ML_DEFAULT_LOG);
        assert_eq!(sum(&OF_DEFAULT_NORM), 1 << OF_DEFAULT_LOG);
    }
}
