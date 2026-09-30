//! wgpu-backed GPU compression pipeline: device context, chain-hash and
//! best-match kernels, host-side compressor orchestration, and streaming pipeline.
pub mod context;
pub mod chains;
pub mod compressor;
pub mod pipeline;
