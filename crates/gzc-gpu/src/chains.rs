//! Host side of the K1 hash-chain build kernel.
use crate::context::{GpuContext, pack_blocks};
use gzc_core::config::{BLOCK_SIZE, HASH_BITS, LOG2_BLOCK};

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

/// Largest `n_blocks` one `ChainsKernel::record` call may take under `limits`: the data
/// (n*BLOCK_SIZE + 4 bytes), head and pred buffers each fit one storage binding and one buffer,
/// the dispatch fits `max_compute_workgroups_per_dimension`, and the kernel's u32 indices
/// `(b*2+width) << HASH_BITS` and `(b*2+width) * BLOCK_SIZE` cannot wrap. 0 if one block doesn't fit.
pub fn max_blocks_per_batch(limits: &wgpu::Limits) -> u32 {
    let limit = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
    let by_data = limit.saturating_sub(4) / BLOCK_SIZE as u64;
    let by_buffers = by_data.min(limit / head_bytes(1)).min(limit / pred_bytes(1));
    let by_index = (1u64 << (32 - HASH_BITS.max(LOG2_BLOCK))) / 2;
    by_buffers.min(by_index).min(limits.max_compute_workgroups_per_dimension as u64) as u32
}

/// Computes long and short hash chains for BLOCK_SIZE blocks on the GPU,
/// splitting into as many K1 dispatches as device limits require.
pub fn gpu_preds(ctx: &GpuContext, blocks: &[&[u8]]) -> anyhow::Result<Vec<Preds>> {
    let kernel = ChainsKernel::new(ctx);
    let max_blocks = max_blocks_per_batch(&ctx.device.limits()) as usize;
    anyhow::ensure!(max_blocks > 0, "device limits too small for one K1 block");

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
    ///
    /// Precondition: `n_blocks <= max_blocks_per_batch(&ctx.device.limits())`. That keeps every
    /// buffer within the binding/buffer limits, the dispatch within the workgroup limit, and
    /// `n_blocks * 2 <= 2^(32 - max(HASH_BITS, LOG2_BLOCK))` so the shader's u32 head/pred
    /// indices do not wrap.
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

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn limits(binding: u64, buffer: u64, wg: u32) -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffer_binding_size: binding,
            max_buffer_size: buffer,
            max_compute_workgroups_per_dimension: wg,
            ..wgpu::Limits::default()
        }
    }

    fn fits(n: u32, limit: u64) -> bool {
        head_bytes(n) <= limit && pred_bytes(n) <= limit && n as u64 * BLOCK_SIZE as u64 + 4 <= limit
    }

    #[test]
    fn batch_fits_every_buffer_at_default_limits() {
        let n = max_blocks_per_batch(&limits(128 * MIB, 256 * MIB, 65535));
        #[cfg(feature = "block-128k")]
        assert_eq!(n, 128); // pred-bound: 1 MiB per block
        #[cfg(feature = "block-16k")]
        assert_eq!(n, 256); // head-bound: 512 KiB per block (pred is only 128 KiB)
        assert!(fits(n, 128 * MIB) && !fits(n + 1, 128 * MIB));
    }

    #[test]
    fn batch_respects_buffer_size_and_workgroup_cap() {
        assert_eq!(max_blocks_per_batch(&limits(128 * MIB, 128 * MIB, 3)), 3);
        let n = max_blocks_per_batch(&limits(u64::MAX, 4 * MIB, 65535));
        assert!(n > 0 && fits(n, 4 * MIB) && !fits(n + 1, 4 * MIB));
    }

    #[test]
    fn batch_keeps_u32_indices_in_range() {
        // With unlimited buffers the cap keeps (b*2+width) << HASH_BITS and * BLOCK_SIZE in u32.
        let n = max_blocks_per_batch(&limits(u64::MAX, u64::MAX, u32::MAX));
        assert_eq!(n as u64 * 2, 1u64 << (32 - HASH_BITS.max(LOG2_BLOCK)));
    }

    #[test]
    fn batch_is_zero_when_one_block_does_not_fit() {
        assert_eq!(max_blocks_per_batch(&limits(256 * 1024, 256 * 1024, 65535)), 0);
    }
}
