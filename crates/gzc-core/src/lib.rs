//! Shared primitives for the GPU zstd compression prototype: block config,
//! sequence/rep-offset model, hashing, synthetic corpora, and CPU-side
//! entropy coding / frame writing used as the reference implementation.
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
pub mod testdata;
