//! A dedicated transfer queue next to wgpu's queue (speed-2 E3, stage 1).
//!
//! wgpu runs every submission on one queue, strictly one after another, so the readback copy of
//! batch i (frames -> host) used to sit between batch i's K4 and batch i+1's K1. On the RTX 5090
//! the copy engine behind a transfer-only queue family runs that copy concurrently with the next
//! batch's kernels at no measurable cost to them (`multiqueue::tests::transfer_probe`).
//!
//! `GpuContext::with_options` creates the `VkDevice` itself (with one extra queue of a
//! transfer-only family) when the adapter is Vulkan and has such a family, and wraps its queue 0
//! of family 0 in the usual `wgpu::Device` (`device_from_raw` + `create_device_from_hal`); the
//! extra queue is driven here with raw Vulkan (ash): a command pool, timeline semaphores and the
//! buffers both queues touch, created `CONCURRENT` over the two families (so no queue-family
//! ownership transfers are needed) and imported into wgpu with `create_buffer_from_hal`.
use std::sync::Arc;

use anyhow::{Context as _, anyhow};
use ash::vk;

/// Destroys the `VkDevice` once the wgpu device and every `TransferQueue` user are gone.
pub(crate) struct DeviceOwner(pub ash::Device);

impl Drop for DeviceOwner {
    fn drop(&mut self) {
        // SAFETY: the last user of the VkDevice (the hal device's drop callback or a
        // TransferQueue) is gone, and with it every object created on the device.
        unsafe {
            let _ = self.0.device_wait_idle();
            self.0.destroy_device(None);
        }
    }
}

/// The extra queue of a transfer-only family on the context's `VkDevice`.
pub struct TransferQueue {
    pub(crate) device: ash::Device,
    pub(crate) queue: vk::Queue,
    /// The transfer queue's family (the wgpu queue is family 0).
    pub family: u32,
    memory: vk::PhysicalDeviceMemoryProperties,
    _owner: Arc<DeviceOwner>,
    /// Keeps the Vulkan instance alive until the device is gone (declared after `_owner`).
    _adapter: wgpu::Adapter,
}

impl TransferQueue {
    pub(crate) fn new(
        owner: Arc<DeviceOwner>,
        family: u32,
        memory: vk::PhysicalDeviceMemoryProperties,
        adapter: wgpu::Adapter,
    ) -> Self {
        let device = owner.0.clone();
        // SAFETY: the device was created with one queue of `family`.
        let queue = unsafe { device.get_device_queue(family, 0) };
        Self { device, queue, family, memory, _owner: owner, _adapter: adapter }
    }

    /// The first memory type allowed by `bits` with all of `want`, else with all of `fallback`.
    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags, fallback: vk::MemoryPropertyFlags) -> Option<(u32, bool)> {
        let types = &self.memory.memory_types[..self.memory.memory_type_count as usize];
        let find = |flags: vk::MemoryPropertyFlags| {
            types.iter().enumerate().find(|(i, t)| bits & (1 << i) != 0 && t.property_flags.contains(flags))
        };
        find(want)
            .or_else(|| find(fallback))
            .map(|(i, t)| (i as u32, t.property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)))
    }

    /// A buffer of `size` bytes with its own memory: device-local and shared (`CONCURRENT`)
    /// between the wgpu queue's family 0 and the transfer family, or (`host`) host-visible,
    /// preferably cached, used by the transfer queue only and mapped for good.
    pub fn buffer(&self, size: u64, usage: vk::BufferUsageFlags, host: bool) -> anyhow::Result<RawBuffer> {
        let families = [0, self.family];
        let mut info = vk::BufferCreateInfo::default().size(size.max(4)).usage(usage);
        info = if host {
            info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        } else {
            info.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&families)
        };
        let d = &self.device;
        // SAFETY: plain object creation on a live device; every object is destroyed by RawBuffer.
        unsafe {
            let buffer = d.create_buffer(&info, None).context("vkCreateBuffer")?;
            let req = d.get_buffer_memory_requirements(buffer);
            use vk::MemoryPropertyFlags as F;
            let pick = if host {
                self.memory_type(req.memory_type_bits, F::HOST_VISIBLE | F::HOST_CACHED, F::HOST_VISIBLE | F::HOST_COHERENT)
            } else {
                self.memory_type(req.memory_type_bits, F::DEVICE_LOCAL, F::DEVICE_LOCAL)
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
            let mut raw = RawBuffer { device: d.clone(), buffer, memory, size, ptr: std::ptr::null_mut(), coherent };
            d.bind_buffer_memory(buffer, memory, 0).context("vkBindBufferMemory")?;
            if host {
                raw.ptr = d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).context("vkMapMemory")?.cast();
            }
            Ok(raw)
        }
    }

    /// A timeline semaphore at 0.
    pub fn timeline(&self) -> anyhow::Result<Timeline> {
        let mut ty = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        // SAFETY: plain object creation; destroyed by Timeline.
        let sem = unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut ty), None) }
            .context("vkCreateSemaphore")?;
        Ok(Timeline { device: self.device.clone(), sem })
    }

    /// A command pool of the transfer family with `n` resettable primary command buffers.
    pub fn commands(&self, n: u32) -> anyhow::Result<Commands> {
        let d = &self.device;
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
            let buffers = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(n),
                )
                .context("vkAllocateCommandBuffers")?;
            Ok(Commands { device: d.clone(), pool, buffers })
        }
    }

    /// Records `copies` (src, src offset, dst, dst offset, bytes) into `cmd`, followed by a
    /// barrier that makes them visible to the host, and submits it once `wait` reaches
    /// `wait_value`; signals `signal` to `signal_value` when done. `cmd` must not be pending.
    pub fn copy(
        &self,
        cmd: vk::CommandBuffer,
        copies: &[(vk::Buffer, u64, vk::Buffer, u64, u64)],
        wait: &Timeline,
        wait_value: u64,
        signal: &Timeline,
        signal_value: u64,
    ) -> anyhow::Result<()> {
        let d = &self.device;
        // SAFETY: `cmd` is idle (its previous submission was waited for), every buffer outlives
        // the submission (the caller keeps them until `signal` reaches `signal_value`).
        unsafe {
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
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
            d.queue_submit(self.queue, &[submit], vk::Fence::null()).context("vkQueueSubmit (transfer)")?;
        }
        Ok(())
    }

    /// Blocks until `t` reaches `value`.
    pub fn wait(&self, t: &Timeline, value: u64) -> anyhow::Result<()> {
        let (sems, values) = ([t.sem], [value]);
        // SAFETY: plain wait on a live semaphore.
        unsafe { self.device.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&values), u64::MAX) }
            .context("vkWaitSemaphores")
    }

    /// The current value of `t`.
    pub fn value(&self, t: &Timeline) -> anyhow::Result<u64> {
        // SAFETY: plain query on a live semaphore.
        unsafe { self.device.get_semaphore_counter_value(t.sem) }.context("vkGetSemaphoreCounterValue")
    }

    /// Signals `t` to `value` from the host if it is below (after an error, with both queues idle).
    pub fn catch_up(&self, t: &Timeline, value: u64) -> anyhow::Result<()> {
        if self.value(t)? < value {
            // SAFETY: no pending GPU signal of `t` (the queues are idle), and `value` is larger.
            unsafe { self.device.signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(t.sem).value(value)) }
                .context("vkSignalSemaphore")?;
        }
        Ok(())
    }

    /// Blocks until the transfer queue is idle.
    pub fn idle(&self) {
        // SAFETY: plain wait.
        let _ = unsafe { self.device.queue_wait_idle(self.queue) };
    }
}

/// A buffer and its dedicated memory (see `TransferQueue::buffer`); host buffers stay mapped.
pub struct RawBuffer {
    device: ash::Device,
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub size: u64,
    ptr: *mut u8,
    coherent: bool,
}

// SAFETY: the mapping is only read through `&self` after the GPU finished writing it; the handles
// are plain Vulkan handles.
unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

impl RawBuffer {
    /// The mapped bytes of a host buffer (after the GPU writes to them completed: invalidates
    /// non-coherent memory first).
    pub fn mapped(&self) -> &[u8] {
        assert!(!self.ptr.is_null(), "not a host buffer");
        if !self.coherent {
            let range = vk::MappedMemoryRange::default().memory(self.memory).offset(0).size(vk::WHOLE_SIZE);
            // SAFETY: the memory is mapped.
            let _ = unsafe { self.device.invalidate_mapped_memory_ranges(&[range]) };
        }
        // SAFETY: the mapping covers `size` bytes and lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    /// The buffer as a `wgpu::Buffer` of `device` (which must live on the same `VkDevice`), with
    /// `usage` (a subset of what it was created with). The wgpu buffer does not own it: `self` must
    /// outlive every GPU use of it.
    pub fn import(&self, device: &wgpu::Device, label: &str, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        // SAFETY: the VkBuffer is valid, bound to memory we manage, and outlives the wgpu buffer's
        // GPU uses (see above); wgpu never maps it.
        unsafe {
            // Externally owned: wgpu-hal must not vkDestroyBuffer it (`from_raw` would).
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
                self.device.unmap_memory(self.memory);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// A timeline semaphore.
pub struct Timeline {
    device: ash::Device,
    pub sem: vk::Semaphore,
}

impl Drop for Timeline {
    fn drop(&mut self) {
        // SAFETY: no pending submission uses it (the owner waited).
        unsafe { self.device.destroy_semaphore(self.sem, None) };
    }
}

/// A command pool and its command buffers.
pub struct Commands {
    device: ash::Device,
    pool: vk::CommandPool,
    pub buffers: Vec<vk::CommandBuffer>,
}

impl Drop for Commands {
    fn drop(&mut self) {
        // SAFETY: no command buffer is pending (the owner waited).
        unsafe { self.device.destroy_command_pool(self.pool, None) };
    }
}

/// The wgpu queue's hal side, for staging semaphore waits/signals on its next submission.
/// `None` on non-Vulkan backends.
pub(crate) fn stage_on_next_submit(
    queue: &wgpu::Queue,
    wait: Option<(&Timeline, u64)>,
    signal: Option<(&Timeline, u64)>,
) -> anyhow::Result<()> {
    // SAFETY: the semaphores outlive the submission that uses them (the owner waits for it).
    let hal = unsafe { queue.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(|| anyhow!("not a Vulkan queue"))?;
    if let Some((t, v)) = wait {
        hal.add_wait_semaphore(t.sem, Some(v), vk::PipelineStageFlags::ALL_COMMANDS);
    }
    if let Some((t, v)) = signal {
        hal.add_signal_semaphore(t.sem, Some(v));
    }
    Ok(())
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
