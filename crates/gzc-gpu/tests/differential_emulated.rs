//! The differential suite (`differential.rs`) once more, every context emulating the other GPUs'
//! semantics (`Emulation::ALL`): shifts by `n % 32` (Apple, AMD) and workgroup vector component
//! stores as whole-vector read-modify-writes (Apple). On an NVIDIA card this catches what only
//! broke on an M4 Pro (K4's prefix sum lost values stored into neighbouring vec4 components).
use gzc_gpu::context::Emulation;

const EMULATION: Emulation = Emulation::ALL;

#[path = "differential.rs"]
mod differential;
