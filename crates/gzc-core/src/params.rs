//! Match-finder and parse parameters, and the named presets the CPU, the GPU and the CLI share.
//!
//! A [`MatchParams`] value fixes the compressed output: the CPU reference and every GPU produce
//! the same bytes for it. Take one from [`preset`] or [`PRESETS`], or build one and check it
//! with [`MatchParams::validate`]. `docs/reference.md` has each preset's ratio.
use crate::config::{HASH_BITS, LOG2_BLOCK, MATCH_SEARCH_CAP};

/// Which hash chains the match finder walks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hashes {
    /// Two chains: `hash_long` (8 bytes), then `hash_short` (5 bytes). Requires `min_match == 5`.
    Dfast,
    /// One chain over a hash of `min_match` bytes.
    Single,
    /// Optimal-parse candidates: a 4-byte hash chain walked `depth` deep, a 3-byte chain walked
    /// `OPT_H3_DEPTH` deep, and any `OptParams::sparse_chains`, merged nearest-first by
    /// `reference::find_cands`. Requires `opt`.
    Opt3,
}

/// Depth of the 3-byte hash chain walked by `Hashes::Opt3`.
pub const OPT_H3_DEPTH: u32 = 4;

/// How the optimal parse prices its first pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seed {
    /// zstd's first-block statistics (`ZSTD_rescaleFreqs`): literal counts `(c > 0) + (c >> 8)`
    /// over the block, the baseline LL/ML/OF tables.
    BlockInit,
    /// The constant LL/ML/OF prior (`codes::OPT_PRIOR_*`) and the histogram of the bytes no
    /// candidate covers ("cover literals", `opt::cover_literals`).
    Prior,
}

/// Optimal-parse settings (see the `opt` module). The parse is `passes` cheap DP passes with
/// optLevel-0 control flow, each re-pricing the next from its own output, then one final pass at
/// `level`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptParams {
    /// zstd optLevel of the final pass: 0 (btopt control flow) or 2 (btultra).
    pub level: u8,
    /// zstd `targetLength` (`sufficient_len`): 8..=32. Longer matches are committed at once.
    pub target_length: u32,
    /// Cheap intermediate passes before the final one (0..=7).
    pub passes: u8,
    /// Prices of the first pass.
    pub seed: Seed,
    /// Candidate records per position from `find_cands` (only 2 is implemented: A and B).
    pub k: u8,
    /// The LL/ML/OF tables of `Seed::Prior` (`opt::seed_prices`). Must be `Base` unless `seed`
    /// is `Prior`.
    pub prior: PriorTables,
    /// Extra sparse candidate chains, walked after `h4` and `h3` by `reference::find_cands`, in
    /// this order. The `Some` entries come first.
    pub sparse_chains: [Option<SparseChain>; 3],
    /// Segment-end gap (zstd's `ilimit = iend - gap`): an inner segment starts matches only at
    /// positions `<= iend - inner_gap`. The block's last segment always uses 8 (`PARSE_END`).
    /// 8, or 3 ("gap3": a match may start 3 bytes before a segment end).
    pub inner_gap: u8,
    /// Relaxation pruning: `Some(n)` relaxes only the `n` longest lengths of each explicit
    /// record (offBase > 3), in every DP pass. Rep records keep every length. `None` relaxes
    /// every length. See the `opt` module doc.
    pub relax_lengths: Option<u8>,
    /// Drop pass (`opt::drop_pass`): after the final DP pass, explicit matches of at most
    /// `drop_max_len` bytes become literals where that is cheaper at the parse's own prices.
    /// 0 turns it off.
    pub drop_max_len: u8,
}

impl OptParams {
    /// True when every optional DP feature is off: no sparse chains, the base prior tables,
    /// gap 8, no relaxation pruning and no drop pass.
    pub fn is_baseline(&self) -> bool {
        self.prior == PriorTables::Base
            && self.sparse_chains == [None; 3]
            && self.inner_gap == 8
            && self.relax_lengths.is_none()
            && self.drop_max_len == 0
    }
}

/// Which prior LL/ML/OF frequency tables `Seed::Prior` uses (`codes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PriorTables {
    /// `codes::OPT_PRIOR_{LL,ML,OF}`, trained on `opt16`'s output. `opt14` uses them.
    Base,
    /// `codes::OPT_PRIOR_SPARSE_{LL,ML,OF}`, trained on the opt16 schedule over `OPT16P1`'s
    /// candidates and segment ends. `opt16p1` uses them.
    Sparse,
}

/// A sparse candidate chain. Only *sparse positions* `p % stride == 0 && p < SPARSE_END`
/// (`reference::SPARSE_END`) are hashed and chained, on the 16-bit
/// `hash::hash_sparse(block, p, width)`. Every other position has no predecessor (`NO_POS`), so
/// the chain is walked only from sparse positions, `depth` entries deep.
/// `reference::sparse_chain_preds` builds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseChain {
    /// Bytes hashed: 5..=12.
    pub width: u32,
    /// Position stride: 1, 2, 4 or 8.
    pub stride: u32,
    /// Chain entries walked per position: 1..=64.
    pub depth: u32,
}

/// The three sparse chains of `OPT16P1`: 6-, 10- and 12-byte keys at every 4th position, 16 deep.
pub const SPARSE_CHAINS: [Option<SparseChain>; 3] = [
    Some(SparseChain { width: 6, stride: 4, depth: 16 }),
    Some(SparseChain { width: 10, stride: 4, depth: 16 }),
    Some(SparseChain { width: 12, stride: 4, depth: 16 }),
];

/// Match-finder and parse parameters. The output is a function of these and the input alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchParams {
    /// Which hash chains the match finder walks.
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
    /// 16 for the chain presets; 12 for `lvl9s12seg`, whose GPU finder bucket-sorts the
    /// candidates by key with 2^hash_bits counters in workgroup memory (`gzc_gpu::sorted`).
    /// Candidates, and so `find_best`, are the hash chains over this key either way.
    pub hash_bits: u32,
    /// 0: the parse runs over the whole block. A lazy parse then runs on the CPU oracle only:
    /// the GPU parses lazy in segments. Otherwise log2 of the parse segment
    /// (`lazy::lazy_parse_segmented`): segments of `1 << segment_log2` bytes are parsed
    /// independently (empty rep state, matches clamped to the segment, no skip acceleration),
    /// then the offsets are re-encoded with the block's true rep history. Needs `lazy > 0`, or
    /// `opt`, whose DP runs per segment the same way.
    pub segment_log2: u32,
    /// `Some`: the optimal parse (`opt::parse`) over `find_cands` candidates. `None`:
    /// `find_best` with a greedy or lazy parse.
    pub opt: Option<OptParams>,
}

impl MatchParams {
    /// Ok when every field is in its supported range.
    ///
    /// Without `opt`: min_match 4..=8, depth 1..=64, lazy 0..=2, search_cap 8..=256, hash_bits
    /// 11..=16, Dfast only with min_match 5, and segments (10..=16) only with a lazy parse.
    ///
    /// With `opt`: hashes `Opt3`, min_match 3, lazy 0, search_cap 64, hash_bits 16,
    /// segment_log2 12, level 0 or 2, target_length 8..=32, passes 0..=7, k 2. Prior tables other
    /// than `Base` need seed `Prior`. Sparse chains are packed first, each with width 5..=12,
    /// stride 1/2/4/8 and depth 1..=64. inner_gap is 8 or 3, relax_lengths `None` or 1..=32,
    /// drop_max_len 0 or 3..=32.
    pub fn validate(&self) -> Result<(), String> {
        let MatchParams { hashes, min_match, depth, lazy, search_cap, hash_bits, segment_log2, opt } = *self;
        if let Some(o) = opt {
            return validate_opt(self, &o);
        }
        if hashes == Hashes::Opt3 {
            return Err("Opt3 hashes need opt".to_string());
        }
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

    /// Number of hash chains: Dfast 2, Single 1, Opt3 2 (h4, h3) plus one per sparse chain.
    pub fn n_hashes(&self) -> u32 {
        match self.hashes {
            Hashes::Dfast => 2,
            Hashes::Single => 1,
            Hashes::Opt3 => 2 + self.opt.map_or(0, |o| o.sparse_chains.iter().flatten().count() as u32),
        }
    }

    /// Shortest sequence the parse can emit (bounds the sequence count per block): `min_match`
    /// for the greedy parse; 4 for lazy/lazy2, whose repcode matches (zstd's `MEM_read32`
    /// checks) need only 4 bytes whatever `min_match` is; 3 for the optimal parse.
    pub fn min_seq_len(&self) -> u32 {
        if self.opt.is_some() {
            3
        } else if self.lazy > 0 {
            4
        } else {
            self.min_match
        }
    }
}

// The opt parse packs offsets in 16 bits.
const _: () = assert!(LOG2_BLOCK <= 16, "opt needs offsets that fit 16 bits");

/// `validate` for `opt` params (see there).
fn validate_opt(p: &MatchParams, o: &OptParams) -> Result<(), String> {
    let want = |ok: bool, what: String| if ok { Ok(()) } else { Err(what) };
    want(p.hashes == Hashes::Opt3, format!("opt needs Opt3 hashes, got {:?}", p.hashes))?;
    want(p.min_match == 3, format!("opt needs min_match 3, got {}", p.min_match))?;
    want((1..=64).contains(&p.depth), format!("depth {} not in 1..=64", p.depth))?;
    want(p.lazy == 0, format!("opt needs lazy 0, got {}", p.lazy))?;
    want(p.search_cap == 64, format!("opt needs search_cap 64, got {}", p.search_cap))?;
    want(p.hash_bits == HASH_BITS, format!("opt needs hash_bits {HASH_BITS}, got {}", p.hash_bits))?;
    want(p.segment_log2 == 12, format!("opt needs segment_log2 12, got {}", p.segment_log2))?;
    want(o.level == 0 || o.level == 2, format!("opt level {} not 0 or 2", o.level))?;
    want((8..=32).contains(&o.target_length), format!("opt target_length {} not in 8..=32", o.target_length))?;
    want(o.passes <= 7, format!("opt passes {} not in 0..=7", o.passes))?;
    want(o.k == 2, format!("opt k {} not 2", o.k))?;
    want(o.prior == PriorTables::Base || o.seed == Seed::Prior, format!("opt prior {:?} needs seed Prior", o.prior))?;
    let n = o.sparse_chains.iter().flatten().count();
    want(o.sparse_chains[..n].iter().all(Option::is_some), format!("opt sparse chains not packed first: {:?}", o.sparse_chains))?;
    for c in o.sparse_chains.iter().flatten() {
        let ok = (5..=12).contains(&c.width) && [1, 2, 4, 8].contains(&c.stride) && (1..=64).contains(&c.depth);
        want(ok, format!("opt sparse chain {c:?}: width 5..=12, stride 1/2/4/8, depth 1..=64"))?;
    }
    want(o.inner_gap == 8 || o.inner_gap == 3, format!("opt inner_gap {} not 8 or 3", o.inner_gap))?;
    want(o.relax_lengths.is_none_or(|n| (1..=32).contains(&n)), format!("opt relax_lengths {:?} not None or 1..=32", o.relax_lengths))?;
    want(o.drop_max_len == 0 || (3..=32).contains(&o.drop_max_len), format!("opt drop_max_len {} not 0 or 3..=32", o.drop_max_len))
}

/// `lvl3`: two hash chains (8 and 5 bytes, as zstd's dfast), depth 1, greedy parse. Compares
/// with libzstd level 3.
pub const LVL3: MatchParams =
    MatchParams { hashes: Hashes::Dfast, min_match: 5, depth: 1, lazy: 0, search_cap: MATCH_SEARCH_CAP as u32, hash_bits: HASH_BITS, segment_log2: 0, opt: None };

/// `lvl9seg`: a single 4-byte hash, depth 32, lazy2, with the parse split into independent 4 KiB
/// segments, so the GPU runs one lane per segment. Its ratio is above libzstd L9's.
pub const LVL9SEG: MatchParams =
    MatchParams { hashes: Hashes::Single, min_match: 4, depth: 32, lazy: 2, search_cap: MATCH_SEARCH_CAP as u32, hash_bits: HASH_BITS, segment_log2: 12, opt: None };

/// `lvl9s12seg`: `lvl9seg` with a 12-bit hash key, so the GPU can bucket-sort the candidates in
/// workgroup memory instead of building 16-bit hash chains (`gzc_gpu`'s sorted finder). Its
/// ratio is above libzstd L9's.
pub const LVL9S12SEG: MatchParams = MatchParams { hash_bits: 12, ..LVL9SEG };

/// `opt16`: an optimal parse aimed at libzstd L16 (btultra). `Opt3` candidates (the h4 chain 32
/// deep and the h3 chain 4 deep, two records per position), zstd's optimal-parse DP per 4 KiB
/// segment with exact rep history, prices seeded from zstd's block init, 3 cheap re-pricing
/// passes, then an optLevel-2 final pass. Its ratio is above libzstd L16's.
pub const OPT16: MatchParams = MatchParams {
    hashes: Hashes::Opt3,
    min_match: 3,
    depth: 32,
    lazy: 0,
    search_cap: 64,
    hash_bits: HASH_BITS,
    segment_log2: 12,
    opt: Some(OptParams {
        level: 2,
        target_length: 32,
        passes: 3,
        seed: Seed::BlockInit,
        k: 2,
        prior: PriorTables::Base,
        sparse_chains: [None; 3],
        inner_gap: 8,
        relax_lengths: None,
        drop_max_len: 0,
    }),
};

/// `opt14`: `opt16`'s candidates and DP aimed at libzstd L14. The first pass is priced from the
/// corpus prior plus the block's cover literals; 1 cheap pass and the optLevel-2 final pass
/// follow. Its ratio is above libzstd L14's.
pub const OPT14: MatchParams = MatchParams { opt: Some(OptParams { passes: 1, seed: Seed::Prior, ..OPT16.opt.unwrap() }), ..OPT16 };

/// `opt16p1`: an L16-class optimal parse with one DP pass.
/// - Candidates: the h4 chain 8 deep, h3 4 deep and the three `SPARSE_CHAINS`, merged by
///   `reference::find_cands`.
/// - Pass-0 prices: `Seed::Prior` with the `PriorTables::Sparse` tables plus the cover literals.
/// - One optLevel-2 pass (`passes: 0`) with gap3 segment ends (`inner_gap: 3`) and top-4
///   relaxation pruning (`relax_lengths: Some(4)`).
/// - Then the drop pass for explicit matches of at most 6 bytes (`opt::drop_pass`).
///
/// Its ratio is above `opt16`'s, in one pass instead of four.
pub const OPT16P1: MatchParams = MatchParams {
    depth: 8,
    opt: Some(OptParams {
        passes: 0,
        seed: Seed::Prior,
        prior: PriorTables::Sparse,
        sparse_chains: SPARSE_CHAINS,
        inner_gap: 3,
        relax_lengths: Some(4),
        drop_max_len: 6,
        ..OPT16.opt.unwrap()
    }),
    ..OPT16
};

/// Every named preset, in CLI order.
pub const PRESETS: [(&str, MatchParams); 6] = [
    ("lvl3", LVL3),
    ("lvl9seg", LVL9SEG),
    ("lvl9s12seg", LVL9S12SEG),
    ("opt14", OPT14),
    ("opt16", OPT16),
    ("opt16p1", OPT16P1),
];

/// The preset called `name`; an unknown name is an error listing the valid ones.
pub fn preset(name: &str) -> Result<MatchParams, String> {
    PRESETS.iter().find(|(n, _)| *n == name).map(|&(_, p)| p).ok_or_else(|| {
        let names: Vec<&str> = PRESETS.iter().map(|(n, _)| *n).collect();
        format!("unknown preset '{name}' (valid: {})", names.join(", "))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{LVL9, RUNG1, RUNG2};

    #[test]
    fn presets_validate() {
        // Every preset and fixture, field by field.
        let m = |hashes, min_match, depth, lazy| MatchParams { hashes, min_match, depth, lazy, search_cap: 64, hash_bits: 16, segment_log2: 0, opt: None };
        assert_eq!(LVL3, m(Hashes::Dfast, 5, 1, 0));
        assert_eq!(RUNG1, m(Hashes::Single, 4, 8, 0));
        assert_eq!(RUNG2, m(Hashes::Single, 4, 8, 1));
        assert_eq!(LVL9, m(Hashes::Single, 4, 32, 2));
        assert_eq!(LVL9SEG, MatchParams { segment_log2: 12, ..m(Hashes::Single, 4, 32, 2) });
        assert_eq!(LVL9S12SEG, MatchParams { hash_bits: 12, segment_log2: 12, ..LVL9 });
        for (name, p) in PRESETS {
            assert_eq!(p.validate(), Ok(()), "{name}");
            assert_eq!(preset(name), Ok(p), "{name}");
        }
        assert_eq!(PRESETS.map(|(n, _)| n), ["lvl3", "lvl9seg", "lvl9s12seg", "opt14", "opt16", "opt16p1"]);
        for removed in ["rung1", "rung2", "lvl9", "lvl9s12", "lvl9s12d16seg"] {
            assert!(preset(removed).is_err(), "{removed}");
        }
        let opt = OptParams {
            level: 2,
            target_length: 32,
            passes: 3,
            seed: Seed::BlockInit,
            k: 2,
            prior: PriorTables::Base,
            sparse_chains: [None; 3],
            inner_gap: 8,
            relax_lengths: None,
            drop_max_len: 0,
        };
        let o16 = MatchParams { hashes: Hashes::Opt3, min_match: 3, depth: 32, lazy: 0, search_cap: 64, hash_bits: 16, segment_log2: 12, opt: Some(opt) };
        assert_eq!(OPT16, o16);
        assert_eq!(OPT14, MatchParams { opt: Some(OptParams { passes: 1, seed: Seed::Prior, ..opt }), ..o16 });
        let sparse = |width| Some(SparseChain { width, stride: 4, depth: 16 });
        let p1 = OptParams {
            passes: 0,
            seed: Seed::Prior,
            prior: PriorTables::Sparse,
            sparse_chains: [sparse(6), sparse(10), sparse(12)],
            inner_gap: 3,
            relax_lengths: Some(4),
            drop_max_len: 6,
            ..opt
        };
        assert_eq!(OPT16P1, MatchParams { depth: 8, opt: Some(p1), ..o16 });
        assert!(OPT14.opt.unwrap().is_baseline() && OPT16.opt.unwrap().is_baseline() && !p1.is_baseline());
        assert_eq!((OPT16.n_hashes(), OPT16.min_seq_len(), OPT14.min_seq_len()), (2, 3, 3));
        assert_eq!((OPT16P1.n_hashes(), OPT16P1.min_seq_len()), (5, 3));
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
    fn validate_opt() {
        let o = OPT16.opt.unwrap();
        let bad = [
            MatchParams { min_match: 4, ..OPT16 },
            MatchParams { hashes: Hashes::Single, ..OPT16 },
            MatchParams { lazy: 2, ..OPT16 },
            MatchParams { search_cap: 32, ..OPT16 },
            MatchParams { hash_bits: 12, ..OPT16 },
            MatchParams { segment_log2: 11, ..OPT16 },
            MatchParams { segment_log2: 0, ..OPT16 },
            MatchParams { depth: 0, ..OPT16 },
            MatchParams { opt: Some(OptParams { level: 1, ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { target_length: 33, ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { target_length: 7, ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { passes: 8, ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { k: 3, ..o }), ..OPT16 },
            MatchParams { opt: None, ..OPT16 },
            // The optional DP features
            MatchParams { opt: Some(OptParams { prior: PriorTables::Sparse, ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { sparse_chains: [None, SPARSE_CHAINS[0], None], ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { sparse_chains: [Some(SparseChain { width: 4, stride: 4, depth: 16 }), None, None], ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { sparse_chains: [Some(SparseChain { width: 13, stride: 4, depth: 16 }), None, None], ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { sparse_chains: [Some(SparseChain { width: 8, stride: 3, depth: 16 }), None, None], ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { sparse_chains: [Some(SparseChain { width: 8, stride: 16, depth: 16 }), None, None], ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { sparse_chains: [Some(SparseChain { width: 8, stride: 4, depth: 0 }), None, None], ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { sparse_chains: [Some(SparseChain { width: 8, stride: 4, depth: 65 }), None, None], ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { inner_gap: 4, ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { inner_gap: 2, ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { relax_lengths: Some(0), ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { relax_lengths: Some(33), ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { drop_max_len: 2, ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { drop_max_len: 33, ..o }), ..OPT16 },
            MatchParams { hashes: Hashes::Opt3, ..LVL9 },
        ];
        for p in bad {
            assert!(p.validate().is_err(), "{p:?} accepted");
        }
        let good = [
            MatchParams { depth: 1, ..OPT16 },
            MatchParams { depth: 64, ..OPT14 },
            MatchParams { opt: Some(OptParams { level: 0, target_length: 8, passes: 0, ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { passes: 7, seed: Seed::Prior, ..o }), ..OPT16 },
            OPT16P1,
            MatchParams { opt: Some(OptParams { prior: PriorTables::Sparse, seed: Seed::Prior, ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { sparse_chains: [Some(SparseChain { width: 5, stride: 1, depth: 1 }), None, None], ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { sparse_chains: [Some(SparseChain { width: 12, stride: 8, depth: 64 }); 3], ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { inner_gap: 3, relax_lengths: Some(1), drop_max_len: 3, ..o }), ..OPT16 },
            MatchParams { opt: Some(OptParams { relax_lengths: Some(32), drop_max_len: 32, ..o }), ..OPT16 },
        ];
        for p in good {
            assert_eq!(p.validate(), Ok(()), "{p:?}");
        }
    }

    #[test]
    fn every_preset_validates() {
        for (name, p) in PRESETS {
            assert!(p.validate().is_ok(), "{name}");
        }
        assert!(MatchParams { depth: 4, ..LVL3 }.validate().is_ok());
        assert!(MatchParams { min_match: 6, lazy: 1, ..RUNG2 }.validate().is_ok());
        assert!(MatchParams { lazy: 3, ..LVL9 }.validate().is_err());
    }
}
