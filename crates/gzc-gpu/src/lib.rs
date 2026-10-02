//! A zstd compressor that runs on the GPU, through wgpu (Vulkan and Metal).
//!
//! The input is cut into independent 64 KiB blocks ([`BLOCK_SIZE`]). The GPU turns each block
//! into one complete zstd frame. A short last block gets a frame of its real length. Any zstd
//! decoder reads the frames; this crate does not decompress.
//!
//! # Compressing a buffer
//!
//! ```no_run
//! use gzc_gpu::{Compressor, Level};
//!
//! let data = std::fs::read("textures.bin")?;
//! let compressor = Compressor::new(Level::Zstd16)?;
//! let frames = compressor.compress(&data)?;
//! println!("{} blocks, {} -> {} bytes", frames.len(), data.len(), frames.as_bytes().len());
//! let first_block = zstd::bulk::decompress(frames.frame(0), gzc_gpu::BLOCK_SIZE)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! [`Compressor::compress`] returns the frames in one buffer ([`Frames`]): `frame(i)` is block
//! `i`, and the whole buffer is a zstd stream of the whole input. Build one [`Compressor`] and
//! reuse it; it owns the device and its buffers.
//!
//! # Levels
//!
//! A [`Level`] is named after the libzstd level whose ratio it matches. The output for a level
//! is the same on every GPU, byte for byte.
//!
//! # Other entry points
//!
//! - [`Compressor::compress_blocks`] takes blocks that are already split, such as the chunks
//!   of a virtual file system. Any block may be shorter than 64 KiB.
//! - [`Compressor::stream`] compresses while data arrives. The caller writes payloads straight
//!   into GPU upload memory and gets finished batches of frames back, with no copy in between.
//! - [`CompressorOptions`] sets the GPU memory budget (6144 MiB by default, for an 8 GB card),
//!   the batches in flight, the match parameters and the [`GpuOptions`].
//!
//! # Errors
//!
//! [`Error`] tells apart a missing adapter, unsupported parameters, an allocation failure, a
//! lost device and bad input, so a caller can fall back to the CPU or shrink the budget.
//!
//! # The environment
//!
//! Nothing here reads the environment unless asked. [`GpuOptions::from_env`] and
//! [`CompressorOptions::from_env`] apply the `GZC_*` variables, which the benchmark and the
//! tests use.
//!
//! # The pipeline underneath
//!
//! [`pipeline`] is the layer the compressor is built on: a [`pipeline::Pipeline`] on a shared
//! [`GpuContext`], with timing statistics, the parse-only path and raw upload slots. Its upload
//! slots leave the block lengths to the caller. Prefer [`Compressor::stream`].
mod chains;
mod compressor;
mod context;
mod emulate;
mod error;
mod k3opt;
mod kernels;
pub mod pipeline;
mod poison;
mod sizing;
mod sorted;
#[doc(hidden)]
pub mod testing;
mod transfer;

pub use compressor::{Batch, Compressor, CompressorOptions, Frames, FramesIter, Level, Payload, Stream};
pub use context::{GpuContext, GpuOptions, K3Kernel};
pub use emulate::Emulation;
pub use error::Error;
pub use gzc_core::config::BLOCK_SIZE;
pub use gzc_core::params::MatchParams;
pub use kernels::{GpuParams, K3Mode, gpu_supports};
pub use pipeline::{FrameBatch, PipelineStats};
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
