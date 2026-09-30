//! Compile-time block configuration.
#[cfg(any(
    all(feature = "block-16k", any(feature = "block-32k", feature = "block-64k", feature = "block-128k")),
    all(feature = "block-32k", any(feature = "block-64k", feature = "block-128k")),
    all(feature = "block-64k", feature = "block-128k"),
))]
compile_error!("enable exactly one block-* feature (use --no-default-features to pick a non-default size)");
#[cfg(not(any(feature = "block-16k", feature = "block-32k", feature = "block-64k", feature = "block-128k")))]
compile_error!("enable one block-* feature");

#[cfg(feature = "block-16k")]
pub const LOG2_BLOCK: u32 = 14;
#[cfg(feature = "block-32k")]
pub const LOG2_BLOCK: u32 = 15;
#[cfg(feature = "block-64k")]
pub const LOG2_BLOCK: u32 = 16;
#[cfg(feature = "block-128k")]
pub const LOG2_BLOCK: u32 = 17;

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
