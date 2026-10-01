//! wgpu-backed GPU compression pipeline: device context, chain-hash and
//! best-match kernels, host-side compressor orchestration, and streaming pipeline.
pub mod context;
pub mod emulate;
pub mod chains;
pub mod sorted;
pub mod compressor;
pub mod k3opt;
pub mod pipeline;
pub mod poison;
pub mod multiqueue;
pub mod transfer;
#[doc(hidden)]
pub mod test_support;
