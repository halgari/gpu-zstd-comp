//! Runtime match-finder / parse parameters and the named presets shared by CPU, GPU and CLI.
use crate::config::{HASH_BITS, LOG2_BLOCK, MATCH_SEARCH_CAP};

/// Which hash chains the match finder walks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hashes {
    /// Two chains: `hash_long` (8 bytes), then `hash_short` (5 bytes). Requires `min_match == 5`.
    Dfast,
    /// One chain over a hash of `min_match` bytes.
    Single,
}

/// Match-finder and parse tuning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchParams {
    pub hashes: Hashes,
    /// Shortest match the finder stores and the parse accepts.
    pub min_match: u32,
    /// Candidates walked per chain.
    pub depth: u32,
    /// 0: greedy parse; 1/2: libzstd lazy/lazy2 deferral depth.
    pub lazy: u32,
    /// Bytes compared per candidate; the parse extends matches that hit the cap.
    pub search_cap: u32,
    /// Bits of the match-finder hash key (`hash::key`: the 16-bit hash's top `hash_bits` bits).
    /// 16 for the chain presets; 12 for the `lvl9s12*` presets, whose GPU finder bucket-sorts the
    /// candidates by key with 2^hash_bits counters in workgroup memory (`gzc_gpu::sorted`).
    /// Candidates, and so `find_best`, are the hash chains over this key either way.
    pub hash_bits: u32,
    /// 0: the lazy parse runs over the whole block. Otherwise log2 of the parse segment
    /// (`lazy::lazy_parse_segmented`): segments of `1 << segment_log2` bytes are parsed
    /// independently (empty rep state, matches clamped to the segment, no skip acceleration),
    /// then the offsets are re-encoded with the block's true rep history. Needs `lazy > 0`.
    pub segment_log2: u32,
}

impl MatchParams {
    /// Ok when every field is in its supported range: min_match 4..=8, depth 1..=64,
    /// lazy 0..=2, search_cap 8..=256, hash_bits 11..=16, and Dfast only with min_match 5.
    pub fn validate(&self) -> Result<(), String> {
        let MatchParams { hashes, min_match, depth, lazy, search_cap, hash_bits, segment_log2 } = *self;
        if !(4..=8).contains(&min_match) {
            return Err(format!("min_match {min_match} not in 4..=8"));
        }
        if !(1..=64).contains(&depth) {
            return Err(format!("depth {depth} not in 1..=64"));
        }
        if lazy > 2 {
            return Err(format!("lazy {lazy} not in 0..=2"));
        }
        if !(8..=256).contains(&search_cap) {
            return Err(format!("search_cap {search_cap} not in 8..=256"));
        }
        if !(11..=HASH_BITS).contains(&hash_bits) {
            return Err(format!("hash_bits {hash_bits} not in 11..={HASH_BITS}"));
        }
        if hashes == Hashes::Dfast && min_match != 5 {
            return Err(format!("Dfast hashes need min_match 5, got {min_match}"));
        }
        if segment_log2 != 0 && !(lazy > 0 && (10..=LOG2_BLOCK).contains(&segment_log2)) {
            return Err(format!("segment_log2 {segment_log2} needs lazy > 0 and 10..={LOG2_BLOCK}"));
        }
        Ok(())
    }

    /// Number of hash chains: Dfast 2, Single 1.
    pub fn n_hashes(&self) -> u32 {
        match self.hashes {
            Hashes::Dfast => 2,
            Hashes::Single => 1,
        }
    }

    /// Shortest sequence the parse can emit (bounds the sequence count per block): `min_match`
    /// for the greedy parse; 4 for lazy/lazy2, whose repcode matches (zstd's `MEM_read32`
    /// checks) need only 4 bytes whatever `min_match` is.
    pub fn min_seq_len(&self) -> u32 {
        if self.lazy > 0 { 4 } else { self.min_match }
    }
}

/// Level-3 calibration (the M3 output, byte for byte): dfast chains, depth 1, greedy.
pub const LVL3: MatchParams =
    MatchParams { hashes: Hashes::Dfast, min_match: 5, depth: 1, lazy: 0, search_cap: MATCH_SEARCH_CAP as u32, hash_bits: HASH_BITS, segment_log2: 0 };
/// Single 4-byte hash, depth 8, greedy (compare against libzstd L5).
pub const RUNG1: MatchParams =
    MatchParams { hashes: Hashes::Single, min_match: 4, depth: 8, lazy: 0, search_cap: MATCH_SEARCH_CAP as u32, hash_bits: HASH_BITS, segment_log2: 0 };
/// Rung 1 with a lazy parse (compare against libzstd L6).
pub const RUNG2: MatchParams =
    MatchParams { hashes: Hashes::Single, min_match: 4, depth: 8, lazy: 1, search_cap: MATCH_SEARCH_CAP as u32, hash_bits: HASH_BITS, segment_log2: 0 };
/// Single 4-byte hash, depth 32, lazy2 (compare against libzstd L9).
pub const LVL9: MatchParams =
    MatchParams { hashes: Hashes::Single, min_match: 4, depth: 32, lazy: 2, search_cap: MATCH_SEARCH_CAP as u32, hash_bits: HASH_BITS, segment_log2: 0 };

/// lvl9 with the parse split into independent 4 KiB segments (speed-2 E1): same match finder,
/// a parse that runs one GPU lane per segment. Ratio ~0.01 % below lvl9, above libzstd L9.
pub const LVL9SEG: MatchParams = MatchParams { segment_log2: 12, ..LVL9 };

/// `lvl9` with a 12-bit hash key (speed2 E2): the GPU builds its candidates as a per-block
/// bucket-sorted array (a counting sort over 2^12 keys in workgroup memory, `gzc_gpu::sorted`)
/// instead of 16-bit hash chains. Full corpus at 64 KiB: 1.33927 (lvl9 1.33932).
pub const LVL9S12: MatchParams = MatchParams { hash_bits: 12, ..LVL9 };
/// `lvl9s12` with the segmented parse (E2 + E1): 1.33926 at 64 KiB, 1.35456 at 128 KiB.
pub const LVL9S12SEG: MatchParams = MatchParams { hash_bits: 12, ..LVL9SEG };
/// `lvl9s12seg` walking 16 candidates instead of 32 (E2 + E1 + E4): 1.33860 at 64 KiB (above libzstd L9,
/// 1.3379). Validated for blocks of at most 64 KiB only: at 128 KiB it gives 1.35159, below L9 (1.3532).
pub const LVL9S12D16SEG: MatchParams = MatchParams { depth: 16, ..LVL9S12SEG };

/// Every named preset, in CLI order.
pub const PRESETS: [(&str, MatchParams); 8] = [
    ("lvl3", LVL3),
    ("rung1", RUNG1),
    ("rung2", RUNG2),
    ("lvl9", LVL9),
    ("lvl9seg", LVL9SEG),
    ("lvl9s12", LVL9S12),
    ("lvl9s12seg", LVL9S12SEG),
    ("lvl9s12d16seg", LVL9S12D16SEG),
];

/// The preset called `name`; an unknown name is an error listing the valid ones.
pub fn preset(name: &str) -> Result<MatchParams, String> {
    PRESETS.iter().find(|(n, _)| *n == name).map(|&(_, p)| p).ok_or_else(|| {
        let names: Vec<&str> = PRESETS.iter().map(|(n, _)| *n).collect();
        format!("unknown preset '{name}' (valid: {})", names.join(", "))
    })
}

/// Whether the CPU reference implements `p` (chains, best match and parse). Since Task 4
/// (the lazy/lazy2 parse) it implements every preset and every valid `MatchParams`.
pub fn cpu_supports(p: &MatchParams) -> bool {
    // Now just `validate()`: kept as a separate hook for the CLI's per-engine preset check.
    p.validate().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_validate() {
        // The spec's preset table, verbatim.
        let m = |hashes, min_match, depth, lazy| MatchParams { hashes, min_match, depth, lazy, search_cap: 64, hash_bits: 16, segment_log2: 0 };
        assert_eq!(LVL3, m(Hashes::Dfast, 5, 1, 0));
        assert_eq!(RUNG1, m(Hashes::Single, 4, 8, 0));
        assert_eq!(RUNG2, m(Hashes::Single, 4, 8, 1));
        assert_eq!(LVL9, m(Hashes::Single, 4, 32, 2));
        assert_eq!(LVL9S12, MatchParams { hash_bits: 12, ..m(Hashes::Single, 4, 32, 2) });
        assert_eq!(LVL9S12D16SEG, MatchParams { hash_bits: 12, segment_log2: 12, ..m(Hashes::Single, 4, 16, 2) });
        assert_eq!(LVL9SEG, MatchParams { segment_log2: 12, ..m(Hashes::Single, 4, 32, 2) });
        assert_eq!(LVL9S12SEG, MatchParams { hash_bits: 12, segment_log2: 12, ..LVL9 });
        for (name, p) in PRESETS {
            assert_eq!(p.validate(), Ok(()), "{name}");
            assert_eq!(preset(name), Ok(p), "{name}");
        }
        assert_eq!(LVL3.n_hashes(), 2);
        assert_eq!(LVL9.n_hashes(), 1);
        assert_eq!(LVL3.min_seq_len(), 5);
        assert_eq!(LVL9.min_seq_len(), 4);
        assert_eq!(MatchParams { min_match: 8, ..RUNG1 }.min_seq_len(), 8);
        assert_eq!(MatchParams { min_match: 8, ..RUNG2 }.min_seq_len(), 4);
    }

    #[test]
    fn unknown_preset_is_rejected() {
        let e = preset("lvl10").unwrap_err();
        assert!(e.contains("lvl10"), "{e}");
        for (name, _) in PRESETS {
            assert!(e.contains(name), "error does not list {name}: {e}");
        }
        assert!(preset("").is_err());
        assert!(preset("LVL3").is_err());
    }

    #[test]
    fn validate_rejects_out_of_range() {
        let single = RUNG1;
        let bad = [
            MatchParams { min_match: 3, ..single },
            MatchParams { min_match: 9, ..single },
            MatchParams { depth: 0, ..single },
            MatchParams { depth: 65, ..single },
            MatchParams { lazy: 3, ..single },
            MatchParams { search_cap: 7, ..single },
            MatchParams { search_cap: 257, ..single },
            MatchParams { min_match: 4, ..LVL3 },
            MatchParams { hash_bits: 10, ..single },
            MatchParams { hash_bits: 17, ..single },
            MatchParams { segment_log2: 12, ..RUNG1 },
            MatchParams { segment_log2: 9, ..LVL9 },
            MatchParams { segment_log2: LOG2_BLOCK + 1, ..LVL9 },
        ];
        for p in bad {
            assert!(p.validate().is_err(), "{p:?} accepted");
        }
        for p in [MatchParams { min_match: 8, depth: 64, search_cap: 256, ..single }, MatchParams { search_cap: 8, ..single }] {
            assert_eq!(p.validate(), Ok(()), "{p:?}");
        }
    }

    #[test]
    fn cpu_supports_all_presets() {
        for (name, p) in PRESETS {
            assert!(cpu_supports(&p), "{name}");
        }
        assert!(cpu_supports(&MatchParams { depth: 4, ..LVL3 }));
        assert!(cpu_supports(&MatchParams { min_match: 6, lazy: 1, ..RUNG2 }));
        assert!(!cpu_supports(&MatchParams { lazy: 3, ..LVL9 }));
    }
}
