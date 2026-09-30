//! GpuContext: device/queue setup, buffer helpers, shader templating.
use anyhow::{Context as _, anyhow};
use gzc_core::config::{BLOCK_SIZE, HASH_BITS, HASHED_POSITIONS, LOG2_BLOCK, NO_POS, PARSE_END};
use gzc_core::params::MatchParams;

const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");

pub struct GpuContext {
    /// A queue of a transfer-only family on the same `VkDevice` (speed-2 E3): frame-path
    /// pipelines read their frames back through it, concurrently with the next batch's kernels
    /// (`pipeline`). Vulkan adapters with such a family (NVIDIA: family 1; AMD: SDMA when the
    /// driver exposes it) of Vulkan 1.2+ with timeline semaphores, unless
    /// `GpuOptions::transfer_queue` is off or frame packing is on. The queue and the wgpu device
    /// share one `transfer::DeviceOwner`, which destroys the `VkDevice` after both are gone and
    /// keeps the Vulkan instance alive until then, so the field order does not matter.
    ///
    /// Single submitter: the transfer path stages timeline-semaphore waits/signals on `queue` for
    /// its next submission (`TransferQueue::submit_wgpu`, under a lock that every submission of a
    /// transfer-readback `Pipeline` takes). While such a pipeline exists, **nothing else may
    /// submit to `queue` concurrently with it** (from another thread: `queue.submit`,
    /// `write_buffer`, `read_buffer`, the one-shot `compressor` functions, another pipeline's
    /// `run`), or that submission could take the staged semaphores. Only one transfer-readback
    /// `Pipeline` may exist per context: `Pipeline::new` errors on a second one while the first is
    /// alive (it does not fall back to the main-queue readback, which would still submit to the
    /// same queue). Other pipelines (parse path, packed frames) and other work on the same thread
    /// are fine. Open a context with `GpuOptions::transfer_queue` off for anything else.
    pub transfer: Option<std::sync::Arc<crate::transfer::TransferQueue>>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    /// True when the device was created with `Features::TIMESTAMP_QUERY`.
    pub timestamps: bool,
    /// True when the device was created with `Features::SUBGROUP`. K1 then runs its subgroup
    /// kernel (`k1_chains_sg.wgsl`) if the subgroup sizes suit it and its self-test passes;
    /// otherwise the workgroup-sort fallback. K3 runs its cooperative kernel
    /// (`k3_coop.wgsl`) if its lane probe passes (see `compressor::k3_mode`).
    pub subgroups: bool,
    /// True when the device was created with `Features::MAPPABLE_PRIMARY_BUFFERS` (mappable
    /// buffers may also be storage buffers): requested for frame packing (`pack_frames`) and for
    /// the direct upload (`direct_upload`).
    pub mappable_storage: bool,
    /// Frame-path pipelines pack their frames (`pipeline::PackKernel`): asked for (`GZC_PACK`,
    /// `with_options`) and `mappable_storage`.
    pub pack_frames: bool,
    /// The kernels read each batch straight from its slot's mapped upload buffer, with no upload
    /// copy (speed-2 E8): `mappable_storage` and the upload buffers land in device-local memory
    /// (full ReBAR / SAM, see `rebar`). `GZC_DIRECT_UPLOAD=0` turns it off, `=1` forces it on
    /// whenever `MAPPABLE_PRIMARY_BUFFERS` exists (for measurements: without ReBAR the kernels
    /// would read the batch over PCIe).
    pub direct_upload: bool,
}

/// How `GpuContext::with_gpu_options` sets up the device. `GpuOptions::from_env` is what
/// `GpuContext::new` uses; tests build it directly to cover every path whatever the environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuOptions {
    /// Enable subgroups when the adapter has them (`GZC_NO_SUBGROUPS` turns them off).
    pub subgroups: bool,
    /// Frame packing (`GZC_PACK`): requests `MAPPABLE_PRIMARY_BUFFERS` for `pipeline::PackKernel`.
    pub pack_frames: bool,
    /// The direct upload (E8): `None` = on with full ReBAR (`rebar`), `Some(b)` = forced (on only
    /// where `MAPPABLE_PRIMARY_BUFFERS` exists). `GZC_DIRECT_UPLOAD=0/1`.
    pub direct_upload: Option<bool>,
    /// The transfer-queue readback (E3) where the adapter supports it (`GZC_TRANSFER_QUEUE=0`
    /// turns it off).
    pub transfer_queue: bool,
}

impl Default for GpuOptions {
    /// Everything on where supported, ignoring the environment.
    fn default() -> Self {
        Self { subgroups: true, pack_frames: false, direct_upload: None, transfer_queue: true }
    }
}

impl GpuOptions {
    /// The defaults with the `GZC_*` environment overrides applied.
    pub fn from_env() -> Self {
        Self {
            subgroups: !env_on("GZC_NO_SUBGROUPS"),
            pack_frames: env_on("GZC_PACK"),
            direct_upload: match std::env::var("GZC_DIRECT_UPLOAD").as_deref() {
                Ok("0") => Some(false),
                Ok("1") => Some(true),
                _ => None,
            },
            transfer_queue: !env_off("GZC_TRANSFER_QUEUE"),
        }
    }
}

/// Boolean `GZC_*` knobs follow one convention: a knob that is off by default (`GZC_PACK`,
/// `GZC_NO_SUBGROUPS`, `GZC_NO_TIMESTAMPS`, `GZC_CHECKED_SHADERS`) is turned on by any value but
/// `0` (this function); one that is on by default (`GZC_TRANSFER_QUEUE`, `GZC_SORTED`) is turned
/// off by `0` only (`env_off`). `GZC_DIRECT_UPLOAD` is tri-state (unset: auto).
pub(crate) fn env_on(key: &str) -> bool {
    std::env::var(key).is_ok_and(|v| v != "0")
}

/// A default-on knob set to `0` (see `env_on`).
pub(crate) fn env_off(key: &str) -> bool {
    std::env::var(key).is_ok_and(|v| v == "0")
}

impl GpuContext {
    /// Opens the high-performance adapter with its full storage-buffer and dispatch limits,
    /// enabling timestamp queries and subgroups when available (`GpuOptions::from_env`):
    /// `GZC_NO_SUBGROUPS` leaves subgroups off, which selects K1's fallback kernel and, since
    /// `compressor::k3_mode` also checks `ctx.subgroups`, the sequential K3; `GZC_PACK` turns on
    /// the pipeline's frame packing; `GZC_DIRECT_UPLOAD` and `GZC_TRANSFER_QUEUE` see
    /// `GpuOptions`.
    pub fn new() -> anyhow::Result<Self> {
        Self::with_gpu_options(GpuOptions::from_env())
    }

    /// `new`, with subgroups (and so K1's subgroup kernel) enabled only if `allow` and the adapter
    /// supports them. `with_subgroups(false)` gives the context a device without subgroup support
    /// gets; tests use it to cover K1's fallback kernel. The rest as in `new`.
    pub fn with_subgroups(allow: bool) -> anyhow::Result<Self> {
        Self::with_gpu_options(GpuOptions { subgroups: allow, ..GpuOptions::from_env() })
    }

    /// Subgroups as in `with_subgroups(allow_subgroups)`; with `mappable`, also the native-only
    /// `MAPPABLE_PRIMARY_BUFFERS` feature when the adapter has it (`pack_frames`), with which
    /// frame-path pipelines pack their frames (`pipeline::PackKernel`, opt-in: slower than the
    /// fixed-stride copy on an RTX 5090, kept for PCIe x8 cards). The rest as in `new`.
    pub fn with_options(allow_subgroups: bool, mappable: bool) -> anyhow::Result<Self> {
        Self::with_gpu_options(GpuOptions {
            subgroups: allow_subgroups,
            pack_frames: mappable,
            ..GpuOptions::from_env()
        })
    }

    /// Opens the device as `opts` says. With `opts.transfer_queue`, no packing, and a Vulkan
    /// adapter that has a usable transfer-only family (`multiqueue::transfer_family`), the
    /// `VkDevice` is created here with that extra queue (`transfer`); if that fails the context
    /// falls back to wgpu's own device (with a warning), as it does everywhere else.
    pub fn with_gpu_options(opts: GpuOptions) -> anyhow::Result<Self> {
        let p = Prepared::new(opts)?;
        let family =
            (opts.transfer_queue && !p.pack_frames).then(|| crate::multiqueue::transfer_family(&p.adapter)).flatten();
        if let Some(family) = family {
            let with_transfer = || -> anyhow::Result<Self> {
                let rd = crate::multiqueue::RawDevice::new(&p, &[(family, 0)])?;
                let mut ctx = rd.context(&p, 0, 0)?;
                let tq = crate::transfer::TransferQueue::new(rd.owner.clone(), family, rd.memory);
                ctx.transfer = Some(std::sync::Arc::new(tq));
                Ok(ctx)
            };
            match with_transfer() {
                Ok(ctx) => return Ok(ctx),
                Err(e) => eprintln!("gzc: transfer queue unavailable ({e:#}); reading back on the main queue"),
            }
        }
        let (device, queue) =
            pollster::block_on(p.adapter.request_device(&p.descriptor())).context("request_device")?;
        Ok(p.context(device, queue))
    }
}

/// The adapter and the device features/limits `GpuContext::with_options` settles on, before a
/// device exists (shared with `multiqueue`, which creates the device itself).
pub(crate) struct Prepared {
    pub adapter: wgpu::Adapter,
    pub info: wgpu::AdapterInfo,
    pub timestamps: bool,
    pub subgroups: bool,
    pub mappable_storage: bool,
    pub pack_frames: bool,
    pub direct_upload: bool,
    pub required_features: wgpu::Features,
    pub required_limits: wgpu::Limits,
}

impl Prepared {
    pub fn descriptor(&self) -> wgpu::DeviceDescriptor<'static> {
        wgpu::DeviceDescriptor {
            label: Some("gzc"),
            required_features: self.required_features,
            required_limits: self.required_limits.clone(),
            ..Default::default()
        }
    }

    pub fn context(&self, device: wgpu::Device, queue: wgpu::Queue) -> GpuContext {
        GpuContext {
            device,
            queue,
            adapter_info: self.info.clone(),
            timestamps: self.timestamps,
            subgroups: self.subgroups,
            mappable_storage: self.mappable_storage,
            pack_frames: self.pack_frames,
            direct_upload: self.direct_upload,
            transfer: None,
        }
    }

    pub fn new(opts: GpuOptions) -> anyhow::Result<Self> {
        let (allow_subgroups, mappable) = (opts.subgroups, opts.pack_frames);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("no GPU adapter found (wgpu): {e}"))?;

        let al = adapter.limits();
        let info = adapter.get_info();
        let subgroups = allow_subgroups && adapter.features().contains(wgpu::Features::SUBGROUP);
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
            max_buffer_size: al.max_buffer_size,
            max_compute_workgroups_per_dimension: al.max_compute_workgroups_per_dimension,
            max_storage_buffers_per_shader_stage: al.max_storage_buffers_per_shader_stage,
            // K3opt keeps its DP rings in workgroup memory when the adapter allows more than
            // wgpu's 16 KiB default (M5; `k3opt::RingMem`). The raised limit also reaches
            // `SortKernel::new` (sorted.rs), which picks its kernel by it: it builds a version only
            // when `sorted::workgroup_bytes` fits, so a table that did not fit 16 KiB may now run
            // sorted. The other kernels do not read the limit.
            max_compute_workgroup_storage_size: al
                .max_compute_workgroup_storage_size
                .max(wgpu::Limits::default().max_compute_workgroup_storage_size),
            ..wgpu::Limits::default()
        };
        // GZC_NO_TIMESTAMPS (anything but 0) leaves timestamp queries off, to time runs without them.
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY)
            && !env_on("GZC_NO_TIMESTAMPS");
        let mut required_features = wgpu::Features::empty();
        if timestamps {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if subgroups {
            required_features |= wgpu::Features::SUBGROUP;
        }
        // Native-only feature, requested for frame packing (`pipeline::PackKernel`) and for the
        // direct upload.
        let has_mappable = adapter.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        let direct_upload = has_mappable && opts.direct_upload.unwrap_or_else(|| rebar(&adapter));
        let pack_frames = mappable && has_mappable;
        let mappable_storage = pack_frames || direct_upload;
        if mappable_storage {
            required_features |= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        }
        Ok(Self {
            adapter,
            info,
            timestamps,
            subgroups,
            mappable_storage,
            pack_frames,
            direct_upload,
            required_features,
            required_limits,
        })
    }
}

impl GpuContext {
    /// Compiles `body` with the block constants and `common.wgsl` prepended.
    pub fn shader(&self, label: &str, body: &str) -> wgpu::ShaderModule {
        self.shader_with(label, body, wgpu::ShaderRuntimeChecks::checked())
    }

    /// `shader` without naga's forced loop bounding (bounds checks stay on). Every loop in `body`
    /// must provably terminate (a loop that does not is undefined behaviour for the driver).
    pub fn shader_unbounded_loops(&self, label: &str, body: &str) -> wgpu::ShaderModule {
        let checks = wgpu::ShaderRuntimeChecks { force_loop_bounding: false, ..wgpu::ShaderRuntimeChecks::checked() };
        self.shader_with(label, body, checks)
    }

    /// `shader` without naga's forced loop bounding and without its index clamps (speed-2 E9):
    /// indices into function-local and workgroup arrays are not clamped, and on backends without
    /// hardware-robust buffer access neither are storage-buffer indices. Integer-division checks
    /// stay on (turning them off measured nothing). The caller must guarantee, for every input
    /// the kernel can be given, that
    /// - every loop in `body` terminates, and
    /// - every array index is in bounds (an out-of-bounds index is undefined behaviour, where the
    ///   clamped module would have silently read/written a clamped element).
    ///
    /// K2, K4 and K5 are built with this; their arguments are in `.superpowers/speed2/e3-report.md`
    /// (E9). A loop or index added to them must come with the same argument. RTX 5090, lvl9,
    /// 64 KiB blocks: K2 −9 %, K4 −18 %, K5 −7 % (128 KiB: −6 / −17 / −5 %). K1 stays checked: its
    /// subgroup kernel got 7 % slower at 64 KiB (9 % faster at 128 KiB); K3 keeps
    /// `shader_unbounded_loops` (the index clamps cost it nothing).
    /// `GZC_CHECKED_SHADERS=1` builds these modules fully checked instead (debugging aid).
    pub fn shader_trusted(&self, label: &str, body: &str) -> wgpu::ShaderModule {
        let checks = if env_on("GZC_CHECKED_SHADERS") {
            wgpu::ShaderRuntimeChecks::checked()
        } else {
            wgpu::ShaderRuntimeChecks {
                bounds_checks: false,
                force_loop_bounding: false,
                ..wgpu::ShaderRuntimeChecks::checked()
            }
        };
        self.shader_with(label, body, checks)
    }

    fn shader_with(&self, label: &str, body: &str, checks: wgpu::ShaderRuntimeChecks) -> wgpu::ShaderModule {
        let src = format!("{}\n{}\n{}", constants_wgsl(), COMMON_WGSL, body);
        // SAFETY: with loop bounding off the caller guarantees every loop terminates, and with
        // bounds checks off every index is in bounds (`shader_unbounded_loops`, `shader_trusted`;
        // the K3 argument is at its call site in `Kernels::new`).
        unsafe {
            self.device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) },
                checks,
            )
        }
    }

    /// STORAGE | COPY_DST buffer, plus COPY_SRC when it will be read back or copied from.
    pub fn storage_buffer(&self, label: &str, size: u64, copy_src: bool) -> wgpu::Buffer {
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if copy_src {
            usage |= wgpu::BufferUsages::COPY_SRC;
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
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

/// True when host-visible memory is device-local and as large as VRAM (full ReBAR / SAM), so
/// wgpu's mappable upload buffers (gpu-allocator `CpuToGpu`, which prefers DEVICE_LOCAL |
/// HOST_VISIBLE) sit in VRAM and kernels read them at VRAM speed. The legacy 256 MiB BAR window
/// does not count: a batch's upload buffers would spill into host memory. Vulkan only (other
/// backends: false).
fn rebar(adapter: &wgpu::Adapter) -> bool {
    use ash::vk::{MemoryHeapFlags, MemoryPropertyFlags as F};
    // SAFETY: the raw handles are only used for a read-only property query while `adapter` lives.
    let Some(hal) = (unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }) else { return false };
    let props = unsafe {
        hal.shared_instance().raw_instance().get_physical_device_memory_properties(hal.raw_physical_device())
    };
    let heaps = &props.memory_heaps[..props.memory_heap_count as usize];
    let vram = heaps.iter().filter(|h| h.flags.contains(MemoryHeapFlags::DEVICE_LOCAL)).map(|h| h.size).max();
    let Some(vram) = vram else { return false };
    props.memory_types[..props.memory_type_count as usize].iter().any(|t| {
        t.property_flags.contains(F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT)
            && heaps[t.heap_index as usize].size >= vram / 10 * 9
    })
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
