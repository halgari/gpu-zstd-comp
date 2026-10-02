//! Compile-time block configuration.
/// Blocks are always 64 KiB.
pub const LOG2_BLOCK: u32 = 16;
pub const BLOCK_SIZE: usize = 1 << LOG2_BLOCK;
pub const HASH_BITS: u32 = 16;
/// Match search/parse only starts matches at p < PARSE_END.
pub const PARSE_END: usize = BLOCK_SIZE - 8;
/// Positions 0..HASHED_POSITIONS have an 8-byte hash window inside the block.
pub const HASHED_POSITIONS: usize = BLOCK_SIZE - 7;
pub const NO_POS: u32 = u32::MAX;
/// Default `MatchParams::search_cap`: find_best compares at most this many bytes per candidate;
/// the parse extends capped matches.
pub const MATCH_SEARCH_CAP: usize = 64;
