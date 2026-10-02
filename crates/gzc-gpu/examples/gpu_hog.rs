//! Test aid: a second GPU tenant. Holds `vram_gib` GiB of device memory and keeps the GPU busy
//! with memory-bound compute (read-modify-write sweeps over all of it) for `seconds`, so that a
//! test run beside it sees a slow, contended, preempted GPU (the conditions of the GTX 1660 Super
//! nondeterminism report).
//!
//! `cargo run --release -p gzc-gpu --example gpu_hog -- [vram_gib (default 4)] [seconds (default 60)]`
use std::time::{Duration, Instant};

const SWEEP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let n = arrayLength(&buf);
    let stride = nw.x * 256u;
    var x = gid.x;
    for (var i = gid.x; i < n; i += stride) {
        x = x * 1664525u + buf[i] + 1013904223u;
        buf[i] = x;
    }
}
"#;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gib: u64 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(4);
    let seconds: u64 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(60);
    let ctx = gzc_gpu::GpuContext::new(gzc_gpu::GpuOptions { transfer_queue: false, ..Default::default() }).expect("GPU");
    eprintln!("gpu_hog: {} — {gib} GiB, {seconds} s", ctx.describe());
    let chunk: u64 = 256 << 20;
    let bufs: Vec<wgpu::Buffer> = (0..gib * 4)
        .map(|_| {
            ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("hog"),
                size: chunk,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect();
    let module = ctx.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hog"),
        source: wgpu::ShaderSource::Wgsl(SWEEP_WGSL.into()),
    });
    let pipeline = ctx.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hog"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);
    let binds: Vec<wgpu::BindGroup> = bufs
        .iter()
        .map(|b| {
            ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hog"),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: b.as_entire_binding() }],
            })
        })
        .collect();
    let end = Instant::now() + Duration::from_secs(seconds);
    let mut sweeps = 0u64;
    while Instant::now() < end {
        let mut enc = ctx.device().create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for bg in &binds {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(8192, 1, 1);
            }
        }
        ctx.queue().submit([enc.finish()]);
        ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        sweeps += 1;
    }
    eprintln!("gpu_hog: {sweeps} sweeps of {gib} GiB");
}
