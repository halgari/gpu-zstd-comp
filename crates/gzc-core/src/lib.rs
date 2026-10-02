//! The CPU reference encoder for `gzc-gpu`, and the pieces the two share.
//!
//! Input is cut into independent 64 KiB blocks ([`block::chunk_file`]). Each block becomes one
//! standard zstd frame. For a given [`params::MatchParams`] this crate and every GPU write the
//! same bytes, so this crate is both the test oracle for the GPU kernels and a CPU fallback.
//!
//! Entry points:
//! - [`params::preset`] and [`params::PRESETS`]: the named parameter sets.
//! - [`reference::compress_block_to_frame`]: one block to its frame.
//! - [`reference::compress_block`] and [`frame::write_frame`]: the same in two steps, the parse
//!   and then the frame.
//!
//! The rest is the encoder itself: match finding (`hash`, `reference`), the parses (`reference`
//! greedy, `lazy`, `opt`), entropy coding (`fse`, `huffman`, `seqenc`, `codes`, `bits`) and the
//! frame writer (`frame`). `synth` generates test data.
//!
//! ```
//! use gzc_core::block::chunk_file;
//! use gzc_core::frame::FrameOptions;
//! use gzc_core::params::LVL3;
//! use gzc_core::reference::compress_block_to_frame;
//!
//! let data = vec![7u8; 100_000];
//! let frames: Vec<Vec<u8>> = chunk_file(&data)
//!     .iter()
//!     .map(|block| compress_block_to_frame(block.real(), LVL3, FrameOptions::default()))
//!     .collect();
//! assert_eq!(frames.len(), 2); // 65536 bytes, then 34464
//! ```
pub mod config;
pub mod block;
pub mod seq;
pub mod hash;
pub mod synth;
pub mod bits;
pub mod codes;
pub mod fse;
pub mod frame;
pub mod seqenc;
pub mod huffman;
pub mod params;
pub mod lazy;
pub mod reference;
pub mod opt;
#[doc(hidden)]
pub mod fixtures;
#[doc(hidden)]
pub mod testdata;
