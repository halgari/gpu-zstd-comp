//! Parameter sets the tests use. They are not presets.
//!
//! `RUNG1` is a single greedy chain; it runs on the GPU like any valid greedy parameters. `RUNG2`
//! and `LVL9` are lazy parses over the whole block. Only the CPU oracle implements those: the GPU
//! parses lazy in segments. The hand-built deferral cases of `lazy::cases` pin the lazy rules on
//! them.
use crate::config::{HASH_BITS, MATCH_SEARCH_CAP};
use crate::params::{Hashes, LVL9SEG, MatchParams};

/// Single 4-byte hash, depth 8, greedy.
pub const RUNG1: MatchParams =
    MatchParams { hashes: Hashes::Single, min_match: 4, depth: 8, lazy: 0, search_cap: MATCH_SEARCH_CAP as u32, hash_bits: HASH_BITS, segment_log2: 0, opt: None };
/// CPU only: `RUNG1` with a lazy parse over the whole block.
pub const RUNG2: MatchParams = MatchParams { lazy: 1, ..RUNG1 };
/// CPU only: `LVL9SEG`'s finder with a lazy2 parse over the whole block.
pub const LVL9: MatchParams = MatchParams { segment_log2: 0, ..LVL9SEG };
