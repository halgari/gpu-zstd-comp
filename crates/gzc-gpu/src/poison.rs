//! Test aid: memory poisoning (`GpuOptions::poison`). The output must never
//! depend on memory the kernels did not write in the same batch. On an idle GPU such memory is
//! consistent (zeros from wgpu's lazy zero-init, or the same stale bytes every run), so a kernel
//! that reads it can pass every test and still go wrong under memory pressure or another tenant.
//! With poisoning on, a context:
//! - allocates every storage buffer (`GpuContext::storage_buffer`), the pipeline's upload slots
//!   and the transfer queue's frame buffers `POISON_PAD` bytes larger than asked, so a read past a
//!   buffer's logical end lands on garbage instead of robust-access zeros;
//! - fills, before every batch, every scratch and output buffer of the batch with a fresh
//!   garbage pattern (`GpuContext::poison_from`), and likewise the input `data` past the batch's
//!   trailing zero word and all the padding. Only the protocol's inputs survive: the batch's
//!   blocks and their trailing zero word, and host-written kernel inputs;
//! - builds every pipeline without workgroup-memory zero-initialisation and, before each batch,
//!   runs a kernel that leaves garbage in the workgroup memory of every SM
//!   (`GpuContext::poison_workgroup_memory`), so a kernel that reads workgroup memory it did not
//!   write sees that garbage.
//!
//! The pattern changes per fill (`GpuOptions::poison_seed` fixes the first seed): random words, all
//! ones, small random words (0..256) or random positions (17 bits), so stale values range from
//! absurd to plausible.
use crate::context::GpuContext;
use std::sync::atomic::Ordering;

/// Bytes every buffer gets past its logical size when poisoning.
pub(crate) const POISON_PAD: u64 = 4096;

const FILL_WGSL: &str = r#"
struct Params { start: u32, end: u32, seed: u32, mode: u32 }
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> p: Params;

fn hash(x: u32) -> u32 {
    var h = x * 0x9E3779B1u + 0x7F4A7C15u;
    h ^= h >> 16u;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13u;
    h *= 0xC2B2AE35u;
    h ^= h >> 16u;
    return h;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let stride = nw.x * 256u;
    for (var i = p.start + gid.x; i < p.end; i += stride) {
        let h = hash(i ^ hash(p.seed));
        var v = h;
        if (p.mode == 1u) { v = 0xFFFFFFFFu; }
        if (p.mode == 2u) { v = h & 0xFFu; }
        if (p.mode == 3u) { v = h & 0x1FFFFu; }
        buf[i] = v;
        if (i > 0xFFFFFFFFu - stride) { break; }
    }
}
"#;

/// Workgroup words `poison_workgroup_memory` fills per workgroup: the device's limit minus 1 KiB of
/// headroom, at most 32 KiB (more than any kernel here declares). Never the full limit: with
/// `Emulation::skew` the skew rewrite adds a workgroup counter to this kernel too, and a kernel 4 B
/// over the limit crashes a GTX 1660 Super's channel (Xid 13 `SKEDCHECK18_L1_CONFIG_TOO_SMALL`)
/// instead of failing validation.
fn dirty_words(ctx: &GpuContext) -> u32 {
    (ctx.device.limits().max_compute_workgroup_storage_size.saturating_sub(1024) / 4).min(8192)
}

#[allow(non_snake_case)]
fn dirty_wgsl(DIRTY_WORDS: u32) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> sink: array<u32>;
@group(0) @binding(1) var<uniform> seed: vec4<u32>;
var<workgroup> junk: array<u32, {DIRTY_WORDS}>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let s = seed.x ^ (wid.x * 0x9E3779B1u);
    for (var i = lid; i < {DIRTY_WORDS}u; i += 256u) {{
        var h = (i ^ s) * 0x85EBCA6Bu;
        h ^= h >> 13u;
        h *= 0xC2B2AE35u;
        junk[i] = select(h ^ (h >> 16u), 0xFFFFFFFFu, seed.y == 1u);
    }}
    workgroupBarrier();
    // Keeps the stores: no seed makes this true for every lane, but the compiler cannot know.
    if (junk[(lid * 37u) % {DIRTY_WORDS}u] == seed.z) {{ sink[0] = lid; }}
}}
"#
    )
}

/// The poison kernels, built on first use.
pub(crate) struct Poisoner {
    fill: wgpu::ComputePipeline,
    fill_layout: wgpu::BindGroupLayout,
    dirty: wgpu::ComputePipeline,
    dirty_layout: wgpu::BindGroupLayout,
    sink: wgpu::Buffer,
}

fn layout(ctx: &GpuContext, label: &str) -> wgpu::BindGroupLayout {
    let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    };
    let storage = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let uniform =
        wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
    ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[entry(0, storage), entry(1, uniform)],
    })
}

impl Poisoner {
    fn new(ctx: &GpuContext) -> Self {
        let fill_layout = layout(ctx, "poison.fill");
        let module = ctx.wgsl_module("poison.fill", FILL_WGSL, wgpu::ShaderRuntimeChecks::checked());
        let fill = crate::kernels::pipeline_from_module(ctx, "poison.fill", &fill_layout, &module, "main");
        let dirty_layout = layout(ctx, "poison.dirty");
        let module = ctx.wgsl_module("poison.dirty", &dirty_wgsl(dirty_words(ctx)), wgpu::ShaderRuntimeChecks::checked());
        let dirty = crate::kernels::pipeline_from_module(ctx, "poison.dirty", &dirty_layout, &module, "main");
        let sink = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("poison.sink"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Self { fill, fill_layout, dirty, dirty_layout, sink }
    }
}

fn uniform(ctx: &GpuContext, words: [u32; 4]) -> wgpu::Buffer {
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("poison.params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    // Fails only on a lost device, whose work never runs.
    if let Ok(mut view) = buf.slice(..).get_mapped_range_mut() {
        view.slice(..).copy_from_slice(bytemuck::cast_slice(&words));
    }
    buf.unmap();
    buf
}

impl GpuContext {
    /// True when this context poisons memory (see the module docs).
    #[doc(hidden)]
    pub fn poisoning(&self) -> bool {
        self.opts.poison
    }

    /// Every compute pipeline's compilation options: the defaults (workgroup memory zeroed at
    /// dispatch), but without the zeroing when poisoning.
    pub(crate) fn compilation_options(&self) -> wgpu::PipelineCompilationOptions<'static> {
        wgpu::PipelineCompilationOptions { zero_initialize_workgroup_memory: !self.opts.poison, ..Default::default() }
    }

    /// Bytes to add to a buffer of logical size `size` (`POISON_PAD` when poisoning).
    pub(crate) fn poison_pad(&self) -> u64 {
        if self.opts.poison { POISON_PAD } else { 0 }
    }

    fn next_seed(&self) -> u32 {
        let k = self.poison_seq.fetch_add(1, Ordering::Relaxed);
        let s = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.poison_base;
        (s ^ (s >> 29)) as u32
    }

    fn poisoner(&self) -> &Poisoner {
        self.poisoner.get_or_init(|| Poisoner::new(self))
    }

    /// When poisoning: records a fill of `buf` from byte `from` (rounded up to a word) to its end
    /// with a fresh garbage pattern. `buf` needs STORAGE usage. No-op otherwise.
    pub(crate) fn poison_from(&self, enc: &mut wgpu::CommandEncoder, buf: &wgpu::Buffer, from: u64) {
        if !self.opts.poison {
            return;
        }
        let start = from.div_ceil(4);
        let end = buf.size() / 4;
        if start >= end {
            return;
        }
        let seed = self.next_seed();
        let mode = seed % 4;
        let p = self.poisoner();
        let params = uniform(self, [start as u32, end as u32, seed, mode]);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("poison.fill"),
            layout: &p.fill_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("poison.fill"), timestamp_writes: None });
        pass.set_pipeline(&p.fill);
        pass.set_bind_group(0, &bind, &[]);
        let groups = (end - start).div_ceil(256).min(4096) as u32;
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// When poisoning: records a kernel that leaves garbage in the workgroup memory of every SM.
    /// No-op otherwise.
    pub(crate) fn poison_workgroup_memory(&self, enc: &mut wgpu::CommandEncoder) {
        if !self.opts.poison {
            return;
        }
        let seed = self.next_seed();
        let p = self.poisoner();
        let params = uniform(self, [seed, (seed >> 8) % 4, seed.rotate_left(7) | 1, 0]);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("poison.dirty"),
            layout: &p.dirty_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: p.sink.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("poison.dirty"), timestamp_writes: None });
        pass.set_pipeline(&p.dirty);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1024, 1, 1);
    }
}
