//! Test and benchmark support. Not part of the public API: anything here may change.
//!
//! Shared by the lib tests and the integration tests (each test binary is its own process).
//!
//! `gpu_test_slot`: tests that open a device of their own and build full-size pipelines or batch
//! buffers take one of `GZC_GPU_TEST_SLOTS` (default 2) slots first, so a parallel `cargo test`
//! does not run a dozen of them on one GPU at once. On an M4 Pro (Metal) twelve such tests at
//! once lost devices and left buffers invalid ("Buffer with 'pipeline.upload' label is
//! invalid"); with two at a time they pass. Light tests (one small kernel, CPU-only checks) need
//! no slot and keep running beside them. Each test binary has its own slots (a runner that
//! starts several binaries at once, such as nextest, can still put more on the GPU).

use std::sync::{Arc, Condvar, Mutex, PoisonError};

use crate::context::{GpuContext, GpuOptions};

/// A context opened with `GpuOptions::from_env()`. Panics without a GPU.
pub fn gpu() -> Arc<GpuContext> {
    gpu_with(GpuOptions::from_env())
}

/// A context opened with `options`. Panics without a GPU.
pub fn gpu_with(options: GpuOptions) -> Arc<GpuContext> {
    Arc::new(GpuContext::new(options).expect("GPU required for gzc-gpu tests"))
}

static SLOTS: Mutex<Option<usize>> = Mutex::new(None);
static FREED: Condvar = Condvar::new();

/// One of the process's GPU test slots, held until dropped.
#[must_use = "the slot is released when the guard drops"]
pub struct GpuTestSlot(());

impl Drop for GpuTestSlot {
    fn drop(&mut self) {
        let mut free = SLOTS.lock().unwrap_or_else(PoisonError::into_inner);
        // Set: `gpu_test_slot` initialized it before handing this slot out.
        if let Some(n) = free.as_mut() {
            *n += 1;
        }
        FREED.notify_one();
    }
}

/// Waits for a free GPU test slot. Poison-tolerant: a test that panics while holding a slot
/// still releases it (the guard drops while unwinding), and a poisoned lock is used as is.
pub fn gpu_test_slot() -> GpuTestSlot {
    let mut free = SLOTS.lock().unwrap_or_else(PoisonError::into_inner);
    loop {
        let n = free.get_or_insert_with(slots_from_env);
        if *n > 0 {
            *n -= 1;
            return GpuTestSlot(());
        }
        free = FREED.wait(free).unwrap_or_else(PoisonError::into_inner);
    }
}

/// `GZC_GPU_TEST_SLOTS` (at least 1), else 2.
fn slots_from_env() -> usize {
    std::env::var("GZC_GPU_TEST_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(2).max(1)
}
