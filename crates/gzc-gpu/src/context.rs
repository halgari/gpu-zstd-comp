//! GpuContext: device/queue setup, buffer helpers, shader templating.
use anyhow::{Context as _, anyhow};
use gzc_core::config::{BLOCK_SIZE, HASH_BITS, HASHED_POSITIONS, LOG2_BLOCK, NO_POS, PARSE_END};
use gzc_core::params::MatchParams;

const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    /// True when the device was created with `Features::TIMESTAMP_QUERY`.
    pub timestamps: bool,
    /// True when the device was created with `Features::SUBGROUP` (S3: cooperative K3).
    pub subgroups: bool,
}

impl GpuContext {
    /// Opens the high-performance adapter with its full storage-buffer and dispatch limits,
    /// enabling timestamp queries when available.
    pub fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("no GPU adapter found (wgpu): {e}"))?;

        let al = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
            max_buffer_size: al.max_buffer_size,
            max_compute_workgroups_per_dimension: al.max_compute_workgroups_per_dimension,
            max_storage_buffers_per_shader_stage: al.max_storage_buffers_per_shader_stage,
            ..wgpu::Limits::default()
        };
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let mut required_features = if timestamps { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() };
        // S3: subgroup operations for the cooperative K3 parse, when the adapter has them.
        let subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
        if subgroups {
            required_features |= wgpu::Features::SUBGROUP;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gzc"),
            required_features,
            required_limits,
            ..Default::default()
        }))
        .context("request_device")?;

        Ok(Self { device, queue, adapter_info: adapter.get_info(), timestamps, subgroups })
    }

    /// Compiles `body` with the block constants and `common.wgsl` prepended.
    pub fn shader(&self, label: &str, body: &str) -> wgpu::ShaderModule {
        let src = format!("{}\n{}\n{}", constants_wgsl(), COMMON_WGSL, body);
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        })
    }

    /// STORAGE | COPY_DST buffer, plus COPY_SRC when it will be read back or copied from.
    pub fn storage_buffer(&self, label: &str, size: u64, copy_src: bool) -> wgpu::Buffer {
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if copy_src {
            usage |= wgpu::BufferUsages::COPY_SRC;
        }
        self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
    }

    /// Copies `count` elements of `T` starting at byte `offset` of `buf` (which needs COPY_SRC)
    /// into a staging buffer, waits for the GPU, and returns them.
    pub fn read_buffer<T: bytemuck::Pod>(&self, buf: &wgpu::Buffer, offset: u64, count: usize) -> Vec<T> {
        let bytes = (count * size_of::<T>()) as u64;
        let mut out = vec![T::zeroed(); count];
        if bytes == 0 {
            return out;
        }
        let padded = bytes.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: padded,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_buffer_to_buffer(buf, offset, &staging, 0, padded);
        self.queue.submit([enc.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
        rx.recv().unwrap().expect("map readback buffer");
        {
            let view = staging.get_mapped_range(..).expect("mapped range");
            bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(&view[..bytes as usize]);
        }
        staging.unmap();
        out
    }
}

/// WGSL `const` declarations mirroring `gzc_core::config` (compile-time block constants,
/// shared by every shader).
pub fn constants_wgsl() -> String {
    format!(
        "const BLOCK_SIZE: u32 = {BLOCK_SIZE}u;\n\
         const LOG2_BLOCK: u32 = {LOG2_BLOCK}u;\n\
         const HASH_BITS: u32 = {HASH_BITS}u;\n\
         const PARSE_END: u32 = {PARSE_END}u;\n\
         const HASHED_POSITIONS: u32 = {HASHED_POSITIONS}u;\n\
         const NO_POS: u32 = 0x{NO_POS:08X}u;\n"
    )
}

/// WGSL `const` declarations for one set of runtime `MatchParams`, prepended to the body of each
/// params-dependent shader, so kernels built for different presets coexist in one process.
pub fn params_wgsl(p: &MatchParams) -> String {
    format!(
        "const MIN_MATCH: u32 = {}u;\n\
         const SEARCH_CAP: u32 = {}u;\n\
         const DEPTH: u32 = {}u;\n\
         const N_HASHES: u32 = {}u;\n\
         const LAZY: u32 = {}u;\n",
        p.min_match,
        p.search_cap,
        p.depth,
        p.n_hashes(),
        p.lazy
    )
}

/// Concatenates BLOCK_SIZE blocks as little-endian words, plus one trailing zero word so
/// unaligned two-word loads at the end of the last block stay in bounds.
pub fn pack_blocks(blocks: &[&[u8]]) -> Vec<u32> {
    let mut out = Vec::with_capacity(blocks.len() * BLOCK_SIZE / 4 + 1);
    for b in blocks {
        assert_eq!(b.len(), BLOCK_SIZE, "pack_blocks: every block must be BLOCK_SIZE bytes");
        out.extend(b.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())));
    }
    out.push(0);
    out
}
