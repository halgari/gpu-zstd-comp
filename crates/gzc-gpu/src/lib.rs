//! wgpu-backed GPU compression pipeline: device context, chain-hash and
//! best-match kernels, host-side compressor orchestration, and streaming pipeline.
pub mod context;
pub mod emulate;
pub mod chains;
pub mod sorted;
pub mod compressor;
pub mod sizing;
pub mod k3opt;
pub mod pipeline;
pub mod poison;
pub mod transfer;
#[doc(hidden)]
pub mod testing;

pub use context::{GpuContext, GpuOptions, K3Kernel};
pub use emulate::Emulation;

// Pins the thread-safety the streaming API relies on (`FrameBatch`es go to writer threads, a
// `Pipeline` may move to another thread), so a dependency upgrade cannot drop it silently.
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    const fn send<T: Send>() {}
    send_sync::<context::GpuContext>();
    send_sync::<compressor::Kernels>();
    send_sync::<pipeline::FrameBatch>();
    send::<pipeline::Pipeline>();
};
