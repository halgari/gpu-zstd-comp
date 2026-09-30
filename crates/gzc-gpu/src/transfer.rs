//! A dedicated transfer queue next to wgpu's queue (speed-2 E3).
//!
//! wgpu runs every submission on one queue, strictly one after another, so the readback copy of
//! batch i (frames -> host) used to sit between batch i's K4 and batch i+1's K1. On the RTX 5090
//! the copy engine behind a transfer-only queue family runs that copy concurrently with the next
//! batch's kernels at no measurable cost to them (`multiqueue::tests::transfer_probe`).
//!
//! `GpuContext::with_gpu_options` creates the `VkDevice` itself (with one extra queue of a
//! transfer-only family) when the adapter is Vulkan 1.2 with timeline semaphores and has such a
//! family, and wraps its queue 0 of family 0 in the usual `wgpu::Device` (`device_from_raw` +
//! `create_device_from_hal`); the extra queue is driven here with raw Vulkan (ash): a command
//! pool, timeline semaphores and the buffers both queues touch, created `CONCURRENT` over the two
//! families (so no queue-family ownership transfers are needed) and imported into wgpu with
//! `create_buffer_from_hal`.
//!
//! Lifetimes: every object here holds an `Arc<DeviceOwner>`, which destroys the `VkDevice` after
//! the last of them (and the wgpu device) is gone and keeps the Vulkan instance alive until then.
//! GPU-side lifetimes (a buffer or semaphore must outlive the submissions using it) are the
//! caller's: the entry points that create such uses are `unsafe` or `pub(crate)`.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, anyhow};
use ash::vk;

/// Destroys the `VkDevice` once the wgpu device and every user of this module are gone. Holds the
/// adapter so the Vulkan instance outlives the device (a `VkDevice` destroyed after its instance
/// crashes the NVIDIA driver).
pub(crate) struct DeviceOwner {
    pub device: ash::Device,
    _adapter: wgpu::Adapter,
}

impl DeviceOwner {
    pub(crate) fn new(device: ash::Device, adapter: wgpu::Adapter) -> Self {
        Self { device, _adapter: adapter }
    }
}

impl Drop for DeviceOwner {
    fn drop(&mut self) {
        // SAFETY: the last user of the VkDevice (a hal device's drop callback or an object of this
        // module) is gone, and with it every object created on the device.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}

/// The extra queue of a transfer-only family on the context's `VkDevice`.
pub struct TransferQueue {
    owner: Arc<DeviceOwner>,
    /// `vkQueueSubmit` / `vkQueueWaitIdle` need external synchronization of the queue.
    queue: Mutex<vk::Queue>,
    /// The transfer queue's family (the wgpu queue is family 0).
    pub family: u32,
    memory: vk::PhysicalDeviceMemoryProperties,
    /// Held while semaphore waits/signals are staged on the wgpu queue and that queue's next
    /// submission is made (`submit_wgpu`), so no other submission through this crate picks them
    /// up.
    wgpu_submit: Mutex<()>,
    /// Set while a transfer-readback `Pipeline` exists (`StreamingGuard`): the transfer path
    /// assumes a single submitter per context, so a second one is refused.
    streaming: AtomicBool,
}

/// Exclusive use of a `TransferQueue` by one streaming `Pipeline` (`TransferQueue::begin_streaming`);
/// released on drop.
pub(crate) struct StreamingGuard {
    tq: Arc<TransferQueue>,
}

impl Drop for StreamingGuard {
    fn drop(&mut self) {
        self.tq.streaming.store(false, Ordering::Release);
    }
}

impl TransferQueue {
    pub(crate) fn new(owner: Arc<DeviceOwner>, family: u32, memory: vk::PhysicalDeviceMemoryProperties) -> Self {
        // SAFETY: the device was created with one queue of `family`.
        let queue = unsafe { owner.device.get_device_queue(family, 0) };
        let (wgpu_submit, streaming) = (Mutex::new(()), AtomicBool::new(false));
        Self { owner, queue: Mutex::new(queue), family, memory, wgpu_submit, streaming }
    }

    /// Claims the queue for one transfer-readback `Pipeline` for as long as the guard lives
    /// (try-lock: never blocks). Errors when another transfer-readback `Pipeline` on the same
    /// context is still alive: its staged semaphores and timelines assume it is the only one
    /// submitting.
    pub(crate) fn begin_streaming(self: &Arc<Self>) -> anyhow::Result<StreamingGuard> {
        anyhow::ensure!(
            self.streaming.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok(),
            "another transfer-readback Pipeline is alive on this GpuContext (one per context: drop it first, \
             or open this context with GpuOptions::transfer_queue off / GZC_TRANSFER_QUEUE=0)"
        );
        Ok(StreamingGuard { tq: self.clone() })
    }

    fn device(&self) -> &ash::Device {
        &self.owner.device
    }

    /// The first memory type allowed by `bits` that `pick` accepts, in `prefs` order.
    fn memory_type(&self, bits: u32, prefs: &[&dyn Fn(vk::MemoryPropertyFlags) -> bool]) -> Option<(u32, bool)> {
        let types = &self.memory.memory_types[..self.memory.memory_type_count as usize];
        // Never protected memory, nor AMD's uncached/device-coherent types.
        let excluded = vk::MemoryPropertyFlags::PROTECTED
            | vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD
            | vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD;
        prefs.iter().find_map(|pick| {
            types
                .iter()
                .enumerate()
                .find(|(i, t)| bits & (1 << i) != 0 && !t.property_flags.intersects(excluded) && pick(t.property_flags))
                .map(|(i, t)| (i as u32, t.property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)))
        })
    }

    /// A buffer of `size` bytes with its own memory: device-local and shared (`CONCURRENT`)
    /// between the wgpu queue's family 0 and the transfer family, or (`host`) host-visible memory
    /// (cached system memory preferred) used by the transfer queue only and mapped for good.
    pub(crate) fn buffer(&self, size: u64, usage: vk::BufferUsageFlags, host: bool) -> anyhow::Result<RawBuffer> {
        let families = [0, self.family];
        let mut info = vk::BufferCreateInfo::default().size(size.max(4)).usage(usage);
        info = if host {
            info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        } else {
            info.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&families)
        };
        let d = self.device();
        use vk::MemoryPropertyFlags as F;
        // SAFETY: plain object creation on a live device; every object is destroyed by RawBuffer
        // (or here on failure).
        unsafe {
            let buffer = d.create_buffer(&info, None).context("vkCreateBuffer")?;
            let req = d.get_buffer_memory_requirements(buffer);
            let pick = if host {
                self.memory_type(
                    req.memory_type_bits,
                    &[
                        &|f| f.contains(F::HOST_VISIBLE | F::HOST_CACHED) && !f.contains(F::DEVICE_LOCAL),
                        &|f| f.contains(F::HOST_VISIBLE | F::HOST_CACHED),
                        &|f| f.contains(F::HOST_VISIBLE | F::HOST_COHERENT),
                    ],
                )
            } else {
                // Plain VRAM first: a DEVICE_LOCAL | HOST_VISIBLE (ReBAR) type would spend the
                // BAR window on buffers the host never maps.
                self.memory_type(
                    req.memory_type_bits,
                    &[&|f| f.contains(F::DEVICE_LOCAL) && !f.contains(F::HOST_VISIBLE), &|f| f.contains(F::DEVICE_LOCAL)],
                )
            };
            let Some((type_index, coherent)) = pick else {
                d.destroy_buffer(buffer, None);
                return Err(anyhow!("no suitable memory type for a {size}-byte transfer buffer"));
            };
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(type_index)
                .push_next(&mut dedicated);
            let memory = match d.allocate_memory(&alloc, None) {
                Ok(m) => m,
                Err(e) => {
                    d.destroy_buffer(buffer, None);
                    return Err(anyhow!("vkAllocateMemory({size} bytes): {e}"));
                }
            };
            let mut raw =
                RawBuffer { owner: self.owner.clone(), buffer, memory, size, ptr: std::ptr::null_mut(), coherent };
            d.bind_buffer_memory(buffer, memory, 0).context("vkBindBufferMemory")?;
            if host {
                raw.ptr =
                    d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).context("vkMapMemory")?.cast();
            }
            Ok(raw)
        }
    }

    /// A timeline semaphore at 0.
    pub(crate) fn timeline(&self) -> anyhow::Result<Timeline> {
        let mut ty =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        // SAFETY: plain object creation; destroyed by Timeline.
        let sem =
            unsafe { self.device().create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut ty), None) }
                .context("vkCreateSemaphore")?;
        Ok(Timeline { owner: self.owner.clone(), sem })
    }

    /// A command pool of the transfer family with `n` resettable primary command buffers.
    pub(crate) fn commands(&self, n: u32) -> anyhow::Result<Commands> {
        let d = self.device();
        // SAFETY: plain object creation; destroyed by Commands.
        unsafe {
            let pool = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .context("vkCreateCommandPool")?;
            let mut commands = Commands { owner: self.owner.clone(), pool, buffers: Vec::new() };
            let buffers = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(n),
                )
                .context("vkAllocateCommandBuffers")?;
            commands.buffers = buffers;
            Ok(commands)
        }
    }

    /// Records `copies` (src, src offset, dst, dst offset, bytes) into `cmd`, followed by a
    /// barrier that makes them visible to the host, and submits it once `wait` reaches
    /// `wait_value`; signals `signal` to `signal_value` when done.
    ///
    /// # Safety
    /// - `cmd` belongs to a `Commands` of this queue and is not pending (its previous submission
    ///   completed).
    /// - Every buffer in `copies` is valid for the copy and outlives the submission, and so do
    ///   `wait` and `signal`: the caller keeps them until `signal` reaches `signal_value` (or the
    ///   queue is idle).
    /// - Whatever writes the sources is ordered before `wait` reaches `wait_value`.
    pub(crate) unsafe fn copy(
        &self,
        cmd: vk::CommandBuffer,
        copies: &[(vk::Buffer, u64, vk::Buffer, u64, u64)],
        wait: &Timeline,
        wait_value: u64,
        signal: &Timeline,
        signal_value: u64,
    ) -> anyhow::Result<()> {
        let d = self.device();
        // SAFETY: the caller's contract above; the queue is externally synchronized by the lock.
        unsafe {
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            for &(src, so, dst, dof, size) in copies {
                if size > 0 {
                    d.cmd_copy_buffer(cmd, src, dst, &[vk::BufferCopy { src_offset: so, dst_offset: dof, size }]);
                }
            }
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            d.end_command_buffer(cmd)?;
            let (waits, wait_values, stages) = ([wait.sem], [wait_value], [vk::PipelineStageFlags::TRANSFER]);
            let (signals, signal_values) = ([signal.sem], [signal_value]);
            let cmds = [cmd];
            let mut values = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values);
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&waits)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&cmds)
                .signal_semaphores(&signals)
                .push_next(&mut values);
            let queue = self.queue.lock().unwrap();
            d.queue_submit(*queue, &[submit], vk::Fence::null()).context("vkQueueSubmit (transfer)")?;
        }
        Ok(())
    }

    /// Submits `cmd` on `wgpu_queue` (the wgpu queue of this queue's device) with a wait for
    /// `wait` and a signal of `signal` attached (timeline values), and returns its submission
    /// index. The staging and the submission happen under one lock, so no other submission made
    /// through this function picks up the semaphores. Every submission a transfer-readback
    /// `Pipeline` makes goes through here (with `None`/`None` when it stages nothing), and only
    /// one such pipeline may exist per context (`begin_streaming`). Code submitting to the same
    /// `wgpu::Queue` directly from another thread while this runs could still pick the semaphores
    /// up: see `GpuContext::transfer` for that rule.
    ///
    /// If wgpu does not hand the submission to the hal queue (a validation error, or a panic in
    /// the error handler), the staged semaphores are removed again (`remove_wait_semaphore` /
    /// `remove_signal_semaphore`, also while unwinding), so no later submission waits for or
    /// signals a stale value; the function then errors. A removed signal leaves the timeline
    /// below the value the caller counted on, which `Pipeline::abandon` catches up from the host.
    ///
    /// # Safety
    /// `wait` and `signal` outlive the submission (the caller waits for it before dropping them).
    pub(crate) unsafe fn submit_wgpu(
        &self,
        wgpu_queue: &wgpu::Queue,
        cmd: wgpu::CommandBuffer,
        wait: Option<(&Timeline, u64)>,
        signal: Option<(&Timeline, u64)>,
    ) -> anyhow::Result<wgpu::SubmissionIndex> {
        let _guard = self.wgpu_submit.lock().unwrap_or_else(|e| e.into_inner());
        {
            // SAFETY: see the contract; the hal queue is only used to stage the semaphores.
            let hal = unsafe { wgpu_queue.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| anyhow!("not a Vulkan queue"))?;
            if let Some((t, v)) = wait {
                hal.add_wait_semaphore(t.sem, Some(v), vk::PipelineStageFlags::ALL_COMMANDS);
            }
            if let Some((t, v)) = signal {
                hal.add_signal_semaphore(t.sem, Some(v));
            }
        }
        /// Unstages whatever the submission did not consume (a no-op after a successful one).
        struct Unstage<'q> {
            queue: &'q wgpu::Queue,
            wait: Option<vk::Semaphore>,
            signal: Option<vk::Semaphore>,
        }
        impl Unstage<'_> {
            /// True when a staged semaphore was still pending (the submission never reached hal).
            fn run(&mut self) -> bool {
                let (wait, signal) = (self.wait.take(), self.signal.take());
                if wait.is_none() && signal.is_none() {
                    return false;
                }
                // SAFETY: only used to unstage semaphores this call staged.
                let Some(hal) = (unsafe { self.queue.as_hal::<wgpu::hal::api::Vulkan>() }) else { return false };
                let w = wait.is_some_and(|s| hal.remove_wait_semaphore(s));
                let s = signal.is_some_and(|s| hal.remove_signal_semaphore(s));
                w || s
            }
        }
        impl Drop for Unstage<'_> {
            fn drop(&mut self) {
                self.run();
            }
        }
        let mut unstage =
            Unstage { queue: wgpu_queue, wait: wait.map(|(t, _)| t.sem), signal: signal.map(|(t, _)| t.sem) };
        let index = wgpu_queue.submit([cmd]);
        anyhow::ensure!(
            !unstage.run(),
            "wgpu submission did not reach the queue (validation error?); its staged semaphores were removed"
        );
        Ok(index)
    }

    /// Blocks until `t` reaches `value`.
    pub(crate) fn wait(&self, t: &Timeline, value: u64) -> anyhow::Result<()> {
        let (sems, values) = ([t.sem], [value]);
        // SAFETY: plain wait on a live semaphore.
        unsafe {
            self.device().wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&values), u64::MAX)
        }
        .context("vkWaitSemaphores")
    }

    /// The current value of `t`.
    pub(crate) fn value(&self, t: &Timeline) -> anyhow::Result<u64> {
        // SAFETY: plain query on a live semaphore.
        unsafe { self.device().get_semaphore_counter_value(t.sem) }.context("vkGetSemaphoreCounterValue")
    }

    /// Signals `t` to `value` from the host if it is below (checked with `value` first, so it
    /// never moves the timeline backwards).
    ///
    /// # Safety
    /// No submission that signals `t` is pending (both queues are idle) and none is staged on the
    /// wgpu queue (`submit_wgpu` unstages its semaphores when a submission fails), so the host
    /// signal cannot race a GPU one or be followed by a signal of a smaller value.
    pub(crate) unsafe fn catch_up(&self, t: &Timeline, value: u64) -> anyhow::Result<()> {
        if self.value(t)? < value {
            // SAFETY: the caller's contract; `value` is larger than the current value.
            unsafe {
                self.device().signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(t.sem).value(value))
            }
            .context("vkSignalSemaphore")?;
        }
        Ok(())
    }

    /// Blocks until the transfer queue is idle.
    pub fn idle(&self) {
        let queue = self.queue.lock().unwrap();
        // SAFETY: plain wait; the queue is externally synchronized by the lock.
        let _ = unsafe { self.device().queue_wait_idle(*queue) };
    }
}

/// A buffer and its dedicated memory (see `TransferQueue::buffer`); host buffers stay mapped.
pub(crate) struct RawBuffer {
    owner: Arc<DeviceOwner>,
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub size: u64,
    ptr: *mut u8,
    coherent: bool,
}

// SAFETY: the mapping is only read through `mapped`, whose contract rules out concurrent GPU
// writes; the handles are plain Vulkan handles.
unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

impl RawBuffer {
    /// The mapped bytes of a host buffer (invalidates non-coherent memory first).
    ///
    /// # Safety
    /// Every GPU write to the buffer has completed (the caller waited for the submission that
    /// wrote it, e.g. its timeline value), and none is pending while the slice lives: the GPU may
    /// otherwise change bytes behind a shared reference.
    pub unsafe fn mapped(&self) -> &[u8] {
        assert!(!self.ptr.is_null(), "not a host buffer");
        if !self.coherent {
            let range = vk::MappedMemoryRange::default().memory(self.memory).offset(0).size(vk::WHOLE_SIZE);
            // SAFETY: the memory is mapped.
            let _ = unsafe { self.owner.device.invalidate_mapped_memory_ranges(&[range]) };
        }
        // SAFETY: the mapping covers `size` bytes and lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    /// The buffer as a `wgpu::Buffer` of `device`, with `usage`. The wgpu buffer does not own the
    /// `VkBuffer` (wgpu never destroys or maps it).
    ///
    /// # Safety
    /// - `device` is a wgpu device on this buffer's `VkDevice`.
    /// - `usage` is covered by the Vulkan usage flags the buffer was created with, and wgpu only
    ///   uses the buffer on queues of the families it was created for.
    /// - `self` outlives every GPU use of the returned buffer (drop it only after the device is
    ///   idle).
    pub unsafe fn import(&self, device: &wgpu::Device, label: &str, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        // SAFETY: the caller's contract; externally owned, so wgpu-hal never vkDestroyBuffers it
        // (`Buffer::from_raw` would).
        unsafe {
            let hal = wgpu::hal::vulkan::Buffer::from_raw_externally_owned(self.buffer, Box::new(|| {}));
            device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                hal,
                &wgpu::BufferDescriptor { label: Some(label), size: self.size, usage, mapped_at_creation: false },
            )
        }
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        // SAFETY: the owner waited for every GPU use to finish.
        unsafe {
            if !self.ptr.is_null() {
                self.owner.device.unmap_memory(self.memory);
            }
            self.owner.device.destroy_buffer(self.buffer, None);
            self.owner.device.free_memory(self.memory, None);
        }
    }
}

/// A timeline semaphore.
pub(crate) struct Timeline {
    owner: Arc<DeviceOwner>,
    pub sem: vk::Semaphore,
}

impl Drop for Timeline {
    fn drop(&mut self) {
        // SAFETY: no pending submission uses it (the owner waited).
        unsafe { self.owner.device.destroy_semaphore(self.sem, None) };
    }
}

/// A command pool and its command buffers.
pub(crate) struct Commands {
    owner: Arc<DeviceOwner>,
    pool: vk::CommandPool,
    pub buffers: Vec<vk::CommandBuffer>,
}

impl Drop for Commands {
    fn drop(&mut self) {
        // SAFETY: no command buffer is pending (the owner waited).
        unsafe { self.owner.device.destroy_command_pool(self.pool, None) };
    }
}

#[cfg(test)]
mod tests {
    use crate::context::GpuContext;

    /// A context with a transfer queue tears down cleanly (the VkDevice goes before the Vulkan
    /// instance), also when a clone of the queue outlives the context.
    #[test]
    fn context_with_transfer_queue_drops_cleanly() {
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
        let tq = ctx.transfer.clone();
        drop(ctx);
        drop(tq);
        let again = GpuContext::new().unwrap();
        drop(again);
    }
}
