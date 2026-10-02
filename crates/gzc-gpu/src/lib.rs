//! wgpu-backed GPU compression pipeline: device context, chain-hash and
//! best-match kernels, host-side compressor orchestration, and streaming pipeline.
mod chains;
mod context;
mod emulate;
mod k3opt;
mod kernels;
pub mod pipeline;
mod poison;
mod sizing;
mod sorted;
#[doc(hidden)]
pub mod testing;
mod transfer;

pub use context::{GpuContext, GpuOptions, K3Kernel};
pub use emulate::Emulation;
pub use gzc_core::config::BLOCK_SIZE;
pub use gzc_core::params::MatchParams;
pub use kernels::{GpuParams, K3Mode, gpu_supports};
pub use sizing::max_batch_blocks;

// Pins the thread-safety the streaming API relies on (`FrameBatch`es go to writer threads, a
// `Pipeline` may move to another thread), so a dependency upgrade cannot drop it silently.
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    const fn send<T: Send>() {}
    send_sync::<context::GpuContext>();
    send_sync::<kernels::Kernels>();
    send_sync::<pipeline::FrameBatch>();
    send::<pipeline::Pipeline>();
};
