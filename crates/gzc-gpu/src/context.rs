//! `GpuContext`: device and queue setup, `GpuOptions`, buffer helpers, shader templating.
use anyhow::{Context as _, anyhow};
use gzc_core::config::{BLOCK_SIZE, HASH_BITS, HASHED_POSITIONS, LOG2_BLOCK, NO_POS, PARSE_END};
use gzc_core::params::MatchParams;

use crate::emulate::Emulation;

const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");

/// An open GPU device and what it runs with. Create one with [`GpuContext::new`]; a
/// [`crate::Compressor`] creates its own.
///
/// It is `Send + Sync`. Several pipelines may share one context, with one limit: a context
/// that reads frames back through a transfer queue ([`GpuContext::transfer_readback`]) serves one
/// frame pipeline at a time, and nothing else may submit to its queue from another thread while
/// that pipeline runs.
pub struct GpuContext {
    /// A queue of a transfer-only family on the same `VkDevice` (speed-2 E3): frame-path
    /// pipelines read their frames back through it, concurrently with the next batch's kernels
    /// (`pipeline`). Vulkan adapters with such a family (NVIDIA: family 1; AMD: SDMA when the
    /// driver exposes it) of Vulkan 1.2+ with timeline semaphores, unless
    /// `GpuOptions::transfer_queue` is off. The queue and the wgpu device
    /// share one `transfer::DeviceOwner`, which destroys the `VkDevice` after both are gone and
    /// keeps the Vulkan instance alive until then, so the field order does not matter.
    ///
    /// Single submitter: the transfer path stages timeline-semaphore waits/signals on `queue` for
    /// its next submission (`TransferQueue::submit_wgpu`, under a lock that every submission of a
    /// transfer-readback `Pipeline` takes). While such a pipeline exists, **nothing else may
    /// submit to `queue` concurrently with it** (from another thread: `queue.submit`,
    /// `write_buffer`, `read_buffer`, the one-shot `testing` functions, another pipeline's
    /// `run`), or that submission could take the staged semaphores. Only one transfer-readback
    /// `Pipeline` may exist per context: `Pipeline::new` errors on a second one while the first is
    /// alive (it does not fall back to the main-queue readback, which would still submit to the
    /// same queue). Other pipelines (parse path) and other work on the same thread
    /// are fine. Open a context with `GpuOptions::transfer_queue` off for anything else.
    pub(crate) transfer: Option<std::sync::Arc<crate::transfer::TransferQueue>>,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) adapter_info: wgpu::AdapterInfo,
    /// True when the device was created with `Features::TIMESTAMP_QUERY`.
    pub(crate) timestamps: bool,
    /// The adapter can also write timestamps inside encoders
    /// (`Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`, not requested). Without it (Metal on Apple
    /// GPUs: counters sampled at stage boundaries only) the pipeline's timers are pass-boundary
    /// samples that the device may leave unwritten, see `PipelineStats::kernel_ms`.
    pub(crate) timestamps_inside_encoders: bool,
    /// True when the device was created with `Features::SUBGROUP`. K1 then runs its subgroup
    /// kernel (`k1_chains_sg.wgsl`) if the subgroup sizes suit it and its self-test passes;
    /// otherwise the workgroup-sort fallback. K3 runs its cooperative kernel
    /// (`k3_coop.wgsl`) if its lane probe passes (see `kernels::k3_mode`).
    pub(crate) subgroups: bool,
    /// The kernels read each batch straight from its slot's mapped upload buffer, with no upload
    /// copy (speed-2 E8): the device was created with `Features::MAPPABLE_PRIMARY_BUFFERS`
    /// (mappable buffers may also be storage buffers; requested only for this) and the upload
    /// buffers land in device-local memory (full ReBAR / SAM, see `rebar`), or
    /// `GpuOptions::direct_upload` forced it (for measurements: without ReBAR the kernels would
    /// read the batch over PCIe).
    pub(crate) direct_upload: bool,
    /// The options the context was opened with (`subgroups`, `direct_upload` and `timestamps`
    /// above hold what the adapter then allowed).
    pub(crate) opts: GpuOptions,
    /// Set by the device-lost callback (`device_lost`).
    lost: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// Memory poisoning (`GpuOptions::poison`, see `poison`).
    pub(crate) poisoner: std::sync::OnceLock<crate::poison::Poisoner>,
    pub(crate) poison_seq: std::sync::atomic::AtomicU32,
    pub(crate) poison_base: u64,
}

/// Which K3 kernel runs the greedy parse over the whole block ([`GpuOptions::k3_kernel`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum K3Kernel {
    /// One lane per block.
    Seq,
    /// One subgroup per block. Opening a pipeline fails where the device cannot run it.
    Coop,
}

/// How a [`GpuContext`] sets up its device and builds its kernels.
///
/// [`GpuOptions::default`] turns on every fast path the adapter supports and reads nothing from
/// the environment. [`GpuOptions::from_env`] applies the `GZC_*` variables on top; the benchmark
/// and the tests use it. No option changes the compressed output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuOptions {
    /// Use subgroup operations when the adapter has them. `GZC_NO_SUBGROUPS` turns them off.
    pub subgroups: bool,
    /// Let the kernels read each batch from the mapped upload buffer, with no upload copy.
    /// `None` turns it on where upload buffers sit in device memory (full ReBAR). `Some(b)`
    /// forces it on or off; on needs `MAPPABLE_PRIMARY_BUFFERS`. `GZC_DIRECT_UPLOAD=0/1`.
    pub direct_upload: Option<bool>,
    /// Read frames back through a transfer-only queue where the adapter has one.
    /// `GZC_TRANSFER_QUEUE=0` turns it off.
    pub transfer_queue: bool,
    /// Time each kernel with timestamp queries when the adapter has them. `GZC_NO_TIMESTAMPS`
    /// turns them off.
    pub timestamps: bool,
    /// Use the bucket-sorted match finder for the presets it serves (a single hash of at most 13
    /// key bits, such as `lvl9s12seg`). `GZC_SORTED=0` turns it off.
    pub sorted_finder: bool,
    /// Threads that copy a batch into its upload buffer. `None` means 4. The pipeline never uses
    /// more than the machine has. `GZC_UPLOAD_THREADS`.
    pub upload_threads: Option<usize>,
    /// Workgroups per K1 dispatch, which is the number of hash tables in use at once. `None`
    /// picks 128 for the subgroup kernel and every table otherwise. `GZC_K1_GROUPS`.
    pub k1_groups: Option<u32>,
    /// Force one K3 kernel for the greedy parse over the whole block. `None` picks the
    /// cooperative kernel where its probe passes. `GZC_K3_MODE=seq|coop`.
    pub k3_kernel: Option<K3Kernel>,
    /// Lanes of the cooperative K3: a power of two in 4..=64, at most the adapter's smallest
    /// subgroup. `None` derives it from the adapter. `GZC_K3_W`.
    pub k3_width: Option<u32>,
    /// Test aid: every workgroup of the cooperative K3 takes its sequential fallback.
    /// `GZC_K3_FORCE_FALLBACK=1`.
    pub k3_force_fallback: bool,
    /// Debugging aid: build every shader with bounds checks. `GZC_CHECKED_SHADERS`.
    pub checked_shaders: bool,
    /// Test aid: rewrite every shader to behave as it would on other GPUs (see [`Emulation`]).
    /// `GZC_EMULATE_SHIFT_MOD32`, `GZC_EMULATE_VEC_RMW`, `GZC_EMULATE_SKEW`.
    pub emulate: Emulation,
    /// Test aid: fill every buffer and workgroup memory with garbage before each batch.
    /// `GZC_POISON`.
    pub poison: bool,
    /// The first poison pattern's seed. `None` takes it from the clock. `GZC_POISON_SEED`.
    pub poison_seed: Option<u64>,
    /// Debugging aid: write every shader module's final source to this directory.
    /// `GZC_DUMP_WGSL`.
    pub dump_wgsl: Option<std::path::PathBuf>,
}

impl Default for GpuOptions {
    /// Every fast path on where the adapter supports it, every test aid off. Reads nothing from
    /// the environment.
    fn default() -> Self {
        Self {
            subgroups: true,
            direct_upload: None,
            transfer_queue: true,
            timestamps: true,
            sorted_finder: true,
            upload_threads: None,
            k1_groups: None,
            k3_kernel: None,
            k3_width: None,
            k3_force_fallback: false,
            checked_shaders: false,
            emulate: Emulation::NONE,
            poison: false,
            poison_seed: None,
            dump_wgsl: None,
        }
    }
}

impl GpuOptions {
    /// The defaults with the `GZC_*` environment variables applied. Each field's documentation
    /// names its variable. This is the only place the crate reads them.
    ///
    /// A switch that is off by default (`GZC_NO_SUBGROUPS`, `GZC_POISON`, ...) is turned on by
    /// any value but `0`. A switch that is on by default (`GZC_TRANSFER_QUEUE`, `GZC_SORTED`) is
    /// turned off by `0` only.
    ///
    /// # Panics
    ///
    /// Panics when a variable holds a value it cannot parse, naming the variable.
    pub fn from_env() -> Self {
        Self {
            subgroups: !env_on("GZC_NO_SUBGROUPS"),
            direct_upload: match std::env::var("GZC_DIRECT_UPLOAD").as_deref() {
                Ok("0") => Some(false),
                Ok("1") => Some(true),
                _ => None,
            },
            transfer_queue: !env_off("GZC_TRANSFER_QUEUE"),
            timestamps: !env_on("GZC_NO_TIMESTAMPS"),
            sorted_finder: !env_off("GZC_SORTED"),
            upload_threads: env_number("GZC_UPLOAD_THREADS"),
            k1_groups: env_number("GZC_K1_GROUPS"),
            k3_kernel: match std::env::var("GZC_K3_MODE").as_deref() {
                Err(_) | Ok("") => None,
                Ok("seq") => Some(K3Kernel::Seq),
                Ok("coop") => Some(K3Kernel::Coop),
                Ok(v) => panic!("GZC_K3_MODE={v}: expected seq or coop"),
            },
            k3_width: env_number("GZC_K3_W"),
            k3_force_fallback: std::env::var("GZC_K3_FORCE_FALLBACK").is_ok_and(|v| v == "1"),
            checked_shaders: env_on("GZC_CHECKED_SHADERS"),
            emulate: Emulation {
                shift_mod32: env_on("GZC_EMULATE_SHIFT_MOD32"),
                vector_rmw: env_on("GZC_EMULATE_VEC_RMW"),
                skew: env_on("GZC_EMULATE_SKEW"),
            },
            poison: env_on("GZC_POISON"),
            poison_seed: env_number("GZC_POISON_SEED"),
            dump_wgsl: std::env::var_os("GZC_DUMP_WGSL").map(Into::into),
        }
    }
}

/// A switch that is off by default: on for any value but `0`.
fn env_on(key: &str) -> bool {
    std::env::var(key).is_ok_and(|v| v != "0")
}

/// A switch that is on by default: off for `0` only.
fn env_off(key: &str) -> bool {
    std::env::var(key).is_ok_and(|v| v == "0")
}

/// A numeric variable; unset is `None`. Panics on a value that is not a number.
fn env_number<T: std::str::FromStr>(key: &str) -> Option<T> {
    let v = std::env::var(key).ok()?;
    Some(v.parse().unwrap_or_else(|_| panic!("{key}={v}: not a number")))
}

/// Writes one module's final source (after emulation rewriting) to `<dir>/<label>.<n>.wgsl`
/// (`GpuOptions::dump_wgsl`), `n` counting modules in creation order, for offline
/// register / shared-memory statistics (WGSL → SPIR-V → driver pipeline statistics).
fn dump_wgsl(dir: &std::path::Path, label: &str, src: &str) {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // The label becomes one file name inside `dir`: no path separators, no `..`.
    let label = label.replace(['/', '\\'], "_").replace("..", "_");
    let path = dir.join(format!("{label}.{n}.wgsl"));
    if let Err(e) = std::fs::create_dir_all(dir).and_then(|_| std::fs::write(&path, src)) {
        eprintln!("GpuOptions::dump_wgsl: {}: {e}", path.display());
    }
}

impl GpuContext {
    /// Opens the high-performance adapter with its full storage-buffer and dispatch limits, as
    /// `options` says. Pass [`GpuOptions::default`], or [`GpuOptions::from_env`] to honour the
    /// `GZC_*` variables.
    ///
    /// With `options.transfer_queue` and a Vulkan adapter that has a usable transfer-only family
    /// (`transfer::transfer_family`), the `VkDevice` is created here with that extra queue
    /// (`transfer`); if that fails the context falls back to wgpu's own device (with a warning),
    /// as it does everywhere else.
    pub fn new(options: GpuOptions) -> anyhow::Result<Self> {
        let p = Prepared::new(options)?;
        let family = p.opts.transfer_queue.then(|| crate::transfer::transfer_family(&p.adapter)).flatten();
        if let Some(family) = family {
            let with_transfer = || -> anyhow::Result<Self> {
                let rd = crate::transfer::RawDevice::new(&p, family)?;
                let mut ctx = rd.context(&p)?;
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

    /// The wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The wgpu queue. See the type's documentation before submitting to it from another thread.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The adapter the device was opened on.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// The options the context was opened with.
    pub fn options(&self) -> &GpuOptions {
        &self.opts
    }

    /// True when the device has subgroup operations and the options allowed them.
    pub fn subgroups(&self) -> bool {
        self.subgroups
    }

    /// True when the device has timestamp queries and the options allowed them.
    pub fn timestamps(&self) -> bool {
        self.timestamps
    }

    /// True when the adapter can also write timestamps inside encoders. Without it the device
    /// may leave a kernel's timer unwritten; that kernel is then missing from
    /// `PipelineStats::kernel_ms`.
    pub fn timestamps_inside_encoders(&self) -> bool {
        self.timestamps_inside_encoders
    }

    /// True when the kernels read each batch straight from the mapped upload buffer.
    pub fn direct_upload(&self) -> bool {
        self.direct_upload
    }

    /// True when frame pipelines read their frames back through a transfer-only queue.
    pub fn transfer_readback(&self) -> bool {
        self.transfer.is_some()
    }
}

/// The adapter and the device features/limits `GpuContext::new` settles on, before a
/// device exists (shared with `transfer::RawDevice`, which creates the device itself).
pub(crate) struct Prepared {
    pub adapter: wgpu::Adapter,
    pub info: wgpu::AdapterInfo,
    pub timestamps: bool,
    pub timestamps_inside_encoders: bool,
    pub subgroups: bool,
    pub direct_upload: bool,
    pub opts: GpuOptions,
    pub required_features: wgpu::Features,
    pub required_limits: wgpu::Limits,
}

impl Prepared {
    pub(crate) fn descriptor(&self) -> wgpu::DeviceDescriptor<'static> {
        wgpu::DeviceDescriptor {
            label: Some("gzc"),
            required_features: self.required_features,
            required_limits: self.required_limits.clone(),
            ..Default::default()
        }
    }

    pub(crate) fn context(&self, device: wgpu::Device, queue: wgpu::Queue) -> GpuContext {
        // wgpu-core reports a device loss only through this callback: errors of the lost type
        // (the hal error that lost the device, e.g. a Metal counter sample buffer that could not
        // be created, and every later `create_buffer` on it) reach no error scope and no
        // uncaptured-error handler; they leave invalid objects that fail much later.
        let lost = std::sync::Arc::new(std::sync::Mutex::new(None));
        let sink = std::sync::Arc::clone(&lost);
        device.set_device_lost_callback(move |reason, message| {
            let mut g = sink.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            g.get_or_insert_with(|| {
                if message.is_empty() { format!("{reason:?}") } else { format!("{reason:?}: {message}") }
            });
        });
        GpuContext {
            lost,
            device,
            queue,
            adapter_info: self.info.clone(),
            timestamps: self.timestamps,
            timestamps_inside_encoders: self.timestamps_inside_encoders,
            subgroups: self.subgroups,
            direct_upload: self.direct_upload,
            transfer: None,
            poisoner: std::sync::OnceLock::new(),
            poison_seq: std::sync::atomic::AtomicU32::new(0),
            poison_base: self.opts.poison_seed.unwrap_or_else(|| {
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
            }),
            opts: self.opts.clone(),
        }
    }

    pub(crate) fn new(opts: GpuOptions) -> anyhow::Result<Self> {
        let allow_subgroups = opts.subgroups;
        #[allow(unused_mut)]
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        // Apple builds compile wgpu's Vulkan backend (for `hal::api::Vulkan`, see Cargo.toml); keep
        // them on Metal so an installed MoltenVK is never picked unless WGPU_BACKEND asks for it.
        #[cfg(target_vendor = "apple")]
        if std::env::var_os("WGPU_BACKEND").is_none() {
            desc.backends = wgpu::Backends::METAL;
        }
        // One thread at a time: the Vulkan loader's first ICD scan is not thread-safe. Two test
        // threads opening contexts at once crashed in it (SIGSEGV, a null call inside
        // libvulkan.so.1's vkEnumerateInstanceExtensionProperties while the other thread was in
        // the NVIDIA ICD's vk_icdNegotiateLoaderICDInterfaceVersion).
        static INSTANCE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let guard = INSTANCE_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("no GPU adapter found (wgpu): {e}"))?;
        drop(guard);

        let al = adapter.limits();
        let info = adapter.get_info();
        let subgroups = allow_subgroups && adapter.features().contains(wgpu::Features::SUBGROUP);
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
            max_buffer_size: al.max_buffer_size,
            max_compute_workgroups_per_dimension: al.max_compute_workgroups_per_dimension,
            max_storage_buffers_per_shader_stage: al.max_storage_buffers_per_shader_stage,
            // The adapter's workgroup storage, where it is above wgpu's 16 KiB default:
            // `SortKernel::new` (sorted.rs) picks its kernel by this limit (it builds a version
            // only when `sorted::workgroup_bytes` fits, so a table that does not fit 16 KiB may
            // still run sorted). K3opt's rings and tables fit the default (`k3opt::ring_for`); the
            // other kernels do not read the limit.
            max_compute_workgroup_storage_size: al
                .max_compute_workgroup_storage_size
                .max(wgpu::Limits::default().max_compute_workgroup_storage_size),
            ..wgpu::Limits::default()
        };
        // `timestamps: false` leaves timestamp queries off, to time runs without them.
        let timestamps = opts.timestamps && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let mut required_features = wgpu::Features::empty();
        if timestamps {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let timestamps_inside_encoders = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        if subgroups {
            required_features |= wgpu::Features::SUBGROUP;
        }
        // Native-only feature, requested for the direct upload.
        let has_mappable = adapter.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        let direct_upload = has_mappable && opts.direct_upload.unwrap_or_else(|| rebar(&adapter));
        if direct_upload {
            required_features |= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        }
        Ok(Self {
            adapter,
            info,
            timestamps,
            timestamps_inside_encoders,
            subgroups,
            direct_upload,
            opts,
            required_features,
            required_limits,
        })
    }
}

impl GpuContext {
    /// Why the device was lost (reason and wgpu's message), once it is.
    pub fn device_lost(&self) -> Option<String> {
        self.lost.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    /// Wait for the GPU with non-blocking polls only (`PollType::Poll`), never `PollType::Wait`:
    /// on Metal. wgpu-hal 30's Metal `Device::wait` can fail with a spurious `DeviceError::Lost`
    /// (which loses the device) when it runs while another thread is inside `queue.submit` (see
    /// `pipeline::Completion::poll_only`); a non-blocking poll never calls it.
    pub(crate) fn poll_only(&self) -> bool {
        self.adapter_info.backend == wgpu::Backend::Metal
    }

    /// Waits for the callback behind `rx` (a buffer map or a submitted-work-done callback, which
    /// fires once `submission` completes; `None`: all submitted work) and returns its value.
    /// With `poll_only` it polls without blocking (every 200 µs) instead of `PollType::Wait`, and
    /// gives up once the device is lost.
    pub(crate) fn wait_callback<T>(
        &self,
        rx: &std::sync::mpsc::Receiver<T>,
        submission: Option<wgpu::SubmissionIndex>,
        poll_only: bool,
    ) -> anyhow::Result<T> {
        use std::sync::mpsc::RecvTimeoutError;
        if let Ok(r) = rx.try_recv() {
            return Ok(r);
        }
        if !poll_only {
            let wait = match submission {
                Some(s) => wgpu::PollType::Wait { submission_index: Some(s), timeout: None },
                None => wgpu::PollType::wait_indefinitely(),
            };
            self.device.poll(wait).context("device poll")?;
            return rx.recv().context("wgpu callback dropped");
        }
        loop {
            self.device.poll(wgpu::PollType::Poll).context("device poll")?;
            match rx.recv_timeout(std::time::Duration::from_micros(200)) {
                Ok(r) => return Ok(r),
                Err(RecvTimeoutError::Timeout) => {
                    if let Some(why) = self.device_lost() {
                        anyhow::bail!("GPU device lost: {why}");
                    }
                }
                Err(RecvTimeoutError::Disconnected) => anyhow::bail!("wgpu callback dropped"),
            }
        }
    }

    /// Waits until all work submitted so far is done: `PollType::Wait`, or with `poll_only`
    /// non-blocking polls until a submitted-work-done callback fires (`wait_callback`).
    pub(crate) fn wait_idle(&self, poll_only: bool) -> anyhow::Result<()> {
        if !poll_only {
            self.device.poll(wgpu::PollType::wait_indefinitely()).context("device poll")?;
            return Ok(());
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        self.wait_callback(&rx, None, true)?;
        // Deliver the map callbacks of the work just completed, as `PollType::Wait` would have.
        self.device.poll(wgpu::PollType::Poll).context("device poll")?;
        Ok(())
    }

    /// One line naming the adapter and what the context runs with, e.g.
    /// `adapter: NVIDIA GeForce RTX 5090 (Vulkan, driver NVIDIA 610.57.04); subgroups: on (32..=32);
    /// timestamps: on; direct upload: on; transfer queue: on; workgroup storage: 49152 B`.
    pub fn describe(&self) -> String {
        let i = &self.adapter_info;
        let on = |b: bool| if b { "on" } else { "off" };
        let subgroups = if self.subgroups {
            format!("on ({}..={})", i.subgroup_min_size, i.subgroup_max_size)
        } else {
            "off".to_string()
        };
        let mut line = format!(
            "adapter: {} ({:?}, driver {} {}); subgroups: {subgroups}; timestamps: {}; direct upload: {}; \
             transfer queue: {}; workgroup storage: {} B",
            i.name,
            i.backend,
            i.driver,
            i.driver_info,
            on(self.timestamps),
            on(self.direct_upload),
            on(self.transfer.is_some()),
            self.device.limits().max_compute_workgroup_storage_size,
        );
        if self.opts.emulate.any() {
            line += &format!("; emulating {:?}", self.opts.emulate);
        }
        if self.opts.poison {
            line += "; poisoning memory";
        }
        line
    }

    /// Compiles `body` with the block constants and `common.wgsl` prepended.
    pub(crate) fn shader(&self, label: &str, body: &str) -> wgpu::ShaderModule {
        self.shader_with(label, body, wgpu::ShaderRuntimeChecks::checked())
    }

    /// `shader` without naga's forced loop bounding (bounds checks stay on). Every loop in `body`
    /// must provably terminate (a loop that does not is undefined behaviour for the driver).
    pub(crate) fn shader_unbounded_loops(&self, label: &str, body: &str) -> wgpu::ShaderModule {
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
    /// 64 KiB blocks: K2 −9 %, K4 −18 %, K5 −7 %. K1 stays checked: its subgroup kernel got 7 %
    /// slower; K3 keeps
    /// `shader_unbounded_loops` (the index clamps cost it nothing).
    /// `GpuOptions::checked_shaders` builds these modules fully checked instead (debugging aid).
    pub(crate) fn shader_trusted(&self, label: &str, body: &str) -> wgpu::ShaderModule {
        let checks = if self.opts.checked_shaders {
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
        self.wgsl_module(label, &src, checks)
    }

    /// Creates a module from complete WGSL `src` (no templating) with `checks`, rewritten for
    /// `emulate` (a test aid). Every module of the crate goes through here.
    pub(crate) fn wgsl_module(&self, label: &str, src: &str, checks: wgpu::ShaderRuntimeChecks) -> wgpu::ShaderModule {
        let src: std::borrow::Cow<str> = if self.opts.emulate.any() {
            self.opts.emulate.rewrite(src).unwrap_or_else(|e| panic!("shader emulation: rewriting {label}: {e}")).into()
        } else {
            src.into()
        };
        if let Some(dir) = &self.opts.dump_wgsl {
            dump_wgsl(dir, label, &src);
        }
        // SAFETY: with loop bounding off the caller guarantees every loop terminates, and with
        // bounds checks off every index is in bounds (`shader_unbounded_loops`, `shader_trusted`;
        // the K3 argument is at its call site in `Kernels::new`). Fully checked modules need no
        // guarantee.
        unsafe {
            self.device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src) },
                checks,
            )
        }
    }

    /// STORAGE | COPY_DST buffer, plus COPY_SRC when it will be read back or copied from. When
    /// poisoning, `poison::POISON_PAD` bytes larger. Test support outside the crate.
    #[doc(hidden)]
    pub fn storage_buffer(&self, label: &str, size: u64, copy_src: bool) -> wgpu::Buffer {
        let size = size + self.poison_pad();
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
    /// into a staging buffer, waits for the GPU, and returns them. Test support outside the crate.
    #[doc(hidden)]
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
        self.wait_callback(&rx, None, self.poll_only()).expect("device poll").expect("map readback buffer");
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
pub(crate) fn constants_wgsl() -> String {
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
pub(crate) fn params_wgsl(p: &MatchParams) -> String {
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

/// wgpu out-of-memory and validation error scopes, pushed together on this thread and popped
/// together (wgpu 30's scopes are per thread: push and pop them on the thread doing the work).
/// Popping also reports a device lost meanwhile: wgpu-core hands errors of that type to no scope.
pub(crate) struct ErrorScopes<'c> {
    // Fields drop in declaration order and wgpu requires scopes to pop in reverse push order:
    // `validation` (pushed last) must come first, so an early return drops them correctly.
    validation: wgpu::ErrorScopeGuard,
    oom: wgpu::ErrorScopeGuard,
    ctx: &'c GpuContext,
}

/// What `ErrorScopes` caught, first one first.
enum ScopeError {
    Validation(String),
    Oom(String),
    Lost(String),
}

impl<'c> ErrorScopes<'c> {
    pub(crate) fn push(ctx: &'c GpuContext) -> Self {
        let oom = ctx.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        Self { validation, oom, ctx }
    }

    /// The first captured error (validation before out-of-memory), else a device loss.
    pub(crate) fn pop(self) -> anyhow::Result<()> {
        match self.take() {
            None => Ok(()),
            Some(ScopeError::Validation(e)) => Err(anyhow!("wgpu validation error: {e}")),
            Some(ScopeError::Oom(e)) => Err(anyhow!("wgpu out of memory: {e}")),
            Some(ScopeError::Lost(why)) => Err(anyhow!("GPU device lost: {why}")),
        }
    }

    /// `pop` for a constructor that allocated `bytes` of buffers for `what`: the error names the
    /// allocation (and wgpu's message the failing buffer's label).
    pub(crate) fn pop_alloc(self, what: &str, bytes: u64) -> anyhow::Result<()> {
        const HINT: &str = "try a smaller --batch or --vram-budget-mb";
        let mib = bytes.div_ceil(1 << 20);
        match self.take() {
            None => Ok(()),
            Some(ScopeError::Oom(e)) => {
                Err(anyhow!("GPU allocation of {mib} MiB for {what} failed: out of memory ({HINT}): {e}"))
            }
            Some(ScopeError::Validation(e)) => {
                Err(anyhow!("GPU allocation of {mib} MiB for {what} failed ({HINT}): {e}"))
            }
            Some(ScopeError::Lost(why)) => Err(anyhow!("GPU device lost while allocating {mib} MiB for {what}: {why}")),
        }
    }

    fn take(self) -> Option<ScopeError> {
        let Self { validation, oom, ctx } = self;
        // A validation error's Display is wgpu's full description; an out-of-memory one's is
        // just "Out of Memory": its source chain names the failing call and the object's label.
        let validation = pollster::block_on(validation.pop()).map(|e| e.to_string());
        let oom = pollster::block_on(oom.pop()).map(|e| format!("{:#}", anyhow::Error::new(e)));
        validation
            .map(ScopeError::Validation)
            .or(oom.map(ScopeError::Oom))
            .or_else(|| ctx.device_lost().map(ScopeError::Lost))
    }
}
