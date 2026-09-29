//! Host side of the K1 hash-chain build kernel.
use crate::context::{GpuContext, pack_blocks};
use gzc_core::config::{BLOCK_SIZE, HASH_BITS};

const K1_WGSL: &str = include_str!("shaders/k1_chains.wgsl");

/// Per-block predecessor chains, each BLOCK_SIZE long; identical to
/// `gzc_core::hash::compute_preds` with `hash_long` / `hash_short`.
pub struct Preds {
    pub long: Vec<u32>,
    pub short: Vec<u32>,
}

/// Bytes of the `head` buffer K1 needs for `n_blocks`.
pub fn head_bytes(n_blocks: u32) -> u64 {
    n_blocks as u64 * 2 * (1u64 << HASH_BITS) * 4
}

/// Bytes of the `pred` buffer K1 writes for `n_blocks`.
pub fn pred_bytes(n_blocks: u32) -> u64 {
    n_blocks as u64 * 2 * BLOCK_SIZE as u64 * 4
}

/// Computes long and short hash chains for BLOCK_SIZE blocks on the GPU,
/// splitting into as many K1 dispatches as device limits require.
pub fn gpu_preds(ctx: &GpuContext, blocks: &[&[u8]]) -> anyhow::Result<Vec<Preds>> {
    let kernel = ChainsKernel::new(ctx);
    let limits = ctx.device.limits();
    let max_blocks = (limits.max_storage_buffer_binding_size.min(limits.max_buffer_size) / pred_bytes(1))
        .min(limits.max_compute_workgroups_per_dimension as u64)
        .max(1) as usize;

    let mut out = Vec::with_capacity(blocks.len());
    for batch in blocks.chunks(max_blocks) {
        let n = batch.len() as u32;
        let packed = pack_blocks(batch);
        let data = ctx.storage_buffer("k1.data", (packed.len() * 4) as u64, false);
        ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&packed));
        let head = ctx.storage_buffer("k1.head", head_bytes(n), false);
        let pred = ctx.storage_buffer("k1.pred", pred_bytes(n), true);

        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("k1") });
        kernel.record(ctx, &mut enc, &data, &head, &pred, n);
        ctx.queue.submit([enc.finish()]);

        let all: Vec<u32> = ctx.read_buffer(&pred, 0, 2 * BLOCK_SIZE * batch.len());
        out.extend(all.chunks_exact(2 * BLOCK_SIZE).map(|c| Preds {
            long: c[..BLOCK_SIZE].to_vec(),
            short: c[BLOCK_SIZE..].to_vec(),
        }));
    }
    Ok(out)
}

/// K1 pipeline; `record` lets later stages run it on their own buffers.
pub struct ChainsKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl ChainsKernel {
    pub fn new(ctx: &GpuContext) -> Self {
        let entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("k1"),
            entries: &[entry(0, true), entry(1, false), entry(2, false)],
        });
        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("k1"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = ctx.shader("k1_chains", K1_WGSL);
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k1_chains"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// data: packed blocks; head: n_blocks*2*2^HASH_BITS u32 (cleared by this call via encoder.clear_buffer);
    /// pred: n_blocks*2*BLOCK_SIZE u32, layout [block][width 0=long,1=short][pos]
    pub fn record(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        data: &wgpu::Buffer,
        head: &wgpu::Buffer,
        pred: &wgpu::Buffer,
        n_blocks: u32,
    ) {
        if n_blocks == 0 {
            return;
        }
        enc.clear_buffer(head, 0, Some(head_bytes(n_blocks)));
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k1"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: head.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pred.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k1"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_blocks, 2, 1);
    }
}
