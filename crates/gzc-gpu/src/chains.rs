//! Host side of the K1 hash-chain build kernel.
use crate::context::{GpuContext, pack_blocks, params_wgsl};
use gzc_core::config::{BLOCK_SIZE, HASH_BITS, LOG2_BLOCK};
use gzc_core::params::MatchParams;

const K1_WGSL: &str = include_str!("shaders/k1_chains.wgsl");

/// Bytes of the `head` buffer K1 needs for `n_blocks` with `n_hashes` chains per block
/// (`MatchParams::n_hashes`).
pub fn head_bytes(n_blocks: u32, n_hashes: u32) -> u64 {
    n_blocks as u64 * n_hashes as u64 * (1u64 << HASH_BITS) * 4
}

/// Bytes of the `pred` buffer K1 writes for `n_blocks` with `n_hashes` chains per block.
pub fn pred_bytes(n_blocks: u32, n_hashes: u32) -> u64 {
    n_blocks as u64 * n_hashes as u64 * BLOCK_SIZE as u64 * 4
}

/// Largest `n_blocks` one `ChainsKernel::record` call may take under `limits` with `n_hashes`
/// chains per block: the data (n*BLOCK_SIZE + 4 bytes), head and pred buffers each fit one
/// storage binding and one buffer, the dispatch fits `max_compute_workgroups_per_dimension`, and
/// the kernel's u32 indices `(b*n_hashes+chain) << HASH_BITS` and `(b*n_hashes+chain) * BLOCK_SIZE`
/// cannot wrap. 0 if one block doesn't fit.
pub fn max_blocks_per_batch(limits: &wgpu::Limits, n_hashes: u32) -> u32 {
    let limit = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
    let by_data = limit.saturating_sub(4) / BLOCK_SIZE as u64;
    let by_buffers = by_data.min(limit / head_bytes(1, n_hashes)).min(limit / pred_bytes(1, n_hashes));
    let by_index = (1u64 << (32 - HASH_BITS.max(LOG2_BLOCK))) / n_hashes as u64;
    by_buffers.min(by_index).min(limits.max_compute_workgroups_per_dimension as u64) as u32
}

/// Computes the hash chains of `params` for BLOCK_SIZE blocks on the GPU, splitting into as many
/// K1 dispatches as device limits require. Per block: one BLOCK_SIZE-long pred array per chain,
/// in K2's walk order, identical to `gzc_core::reference::chains`.
pub fn gpu_preds(ctx: &GpuContext, blocks: &[&[u8]], params: &MatchParams) -> anyhow::Result<Vec<Vec<Vec<u32>>>> {
    let kernel = ChainsKernel::new(ctx, params)?;
    let nh = params.n_hashes();
    let max_blocks = max_blocks_per_batch(&ctx.device.limits(), nh) as usize;
    anyhow::ensure!(max_blocks > 0, "device limits too small for one K1 block");

    let mut out = Vec::with_capacity(blocks.len());
    for batch in blocks.chunks(max_blocks) {
        let n = batch.len() as u32;
        let packed = pack_blocks(batch);
        let data = ctx.storage_buffer("k1.data", (packed.len() * 4) as u64, false);
        ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&packed));
        let head = ctx.storage_buffer("k1.head", head_bytes(n, nh), false);
        let pred = ctx.storage_buffer("k1.pred", pred_bytes(n, nh), true);

        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("k1") });
        kernel.record(ctx, &mut enc, &data, &head, &pred, n);
        ctx.queue.submit([enc.finish()]);

        let per_block = nh as usize * BLOCK_SIZE;
        let all: Vec<u32> = ctx.read_buffer(&pred, 0, per_block * batch.len());
        for block in all.chunks_exact(per_block) {
            out.push(block.chunks_exact(BLOCK_SIZE).map(|c| c.to_vec()).collect());
        }
    }
    Ok(out)
}

/// K1 pipeline, built for one `MatchParams`' chains; `record` lets later stages run it on their
/// own buffers.
pub struct ChainsKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    n_hashes: u32,
}

impl ChainsKernel {
    /// Builds K1 for the chains of `params` (its `N_HASHES` and `MIN_MATCH` are injected as WGSL
    /// constants). Errors if `params` is invalid.
    pub fn new(ctx: &GpuContext, params: &MatchParams) -> anyhow::Result<Self> {
        params.validate().map_err(|e| anyhow::anyhow!("invalid match params {params:?}: {e}"))?;
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
        let module = ctx.shader("k1_chains", &format!("{}{K1_WGSL}", params_wgsl(params)));
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k1_chains"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self { pipeline, layout, n_hashes: params.n_hashes() })
    }

    /// Chains per block (`MatchParams::n_hashes`) this kernel builds.
    pub fn n_hashes(&self) -> u32 {
        self.n_hashes
    }

    /// data: packed blocks; head: `head_bytes(n_blocks, n_hashes)` (cleared by this call via
    /// encoder.clear_buffer); pred: `pred_bytes(n_blocks, n_hashes)`, layout [block][chain][pos]
    /// (Dfast: chain 0 long, 1 short; Single: chain 0 over `hash_width(min_match)`).
    ///
    /// Precondition: `n_blocks <= max_blocks_per_batch(&ctx.device.limits(), n_hashes)`. That
    /// keeps every buffer within the binding/buffer limits, the dispatch within the workgroup
    /// limit, and `n_blocks * n_hashes <= 2^(32 - max(HASH_BITS, LOG2_BLOCK))` so the shader's u32
    /// head/pred indices do not wrap.
    pub fn record(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        data: &wgpu::Buffer,
        head: &wgpu::Buffer,
        pred: &wgpu::Buffer,
        n_blocks: u32,
    ) {
        self.record_timed(ctx, enc, data, head, pred, n_blocks, None);
    }

    /// `record`, with the K1 compute pass writing `timestamp_writes` (if any).
    #[allow(clippy::too_many_arguments)]
    pub fn record_timed(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        data: &wgpu::Buffer,
        head: &wgpu::Buffer,
        pred: &wgpu::Buffer,
        n_blocks: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites>,
    ) {
        if n_blocks == 0 {
            return;
        }
        enc.clear_buffer(head, 0, Some(head_bytes(n_blocks, self.n_hashes)));
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k1"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: head.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pred.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k1"), timestamp_writes });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_blocks, self.n_hashes, 1);
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

    fn fits(n: u32, nh: u32, limit: u64) -> bool {
        head_bytes(n, nh) <= limit && pred_bytes(n, nh) <= limit && n as u64 * BLOCK_SIZE as u64 + 4 <= limit
    }

    #[test]
    fn batch_fits_every_buffer_at_default_limits() {
        let n2 = max_blocks_per_batch(&limits(128 * MIB, 256 * MIB, 65535), 2);
        let n1 = max_blocks_per_batch(&limits(128 * MIB, 256 * MIB, 65535), 1);
        #[cfg(feature = "block-128k")]
        assert_eq!((n2, n1), (128, 256)); // pred-bound: 1 MiB per block (512 KiB with one chain)
        #[cfg(feature = "block-16k")]
        assert_eq!((n2, n1), (256, 512)); // head-bound: 512 KiB per block (256 KiB with one chain)
        assert!(fits(n2, 2, 128 * MIB) && !fits(n2 + 1, 2, 128 * MIB));
        assert!(fits(n1, 1, 128 * MIB) && !fits(n1 + 1, 1, 128 * MIB));
    }

    #[test]
    fn batch_respects_buffer_size_and_workgroup_cap() {
        for nh in [1, 2] {
            assert_eq!(max_blocks_per_batch(&limits(128 * MIB, 128 * MIB, 3), nh), 3);
            let n = max_blocks_per_batch(&limits(u64::MAX, 4 * MIB, 65535), nh);
            assert!(n > 0 && fits(n, nh, 4 * MIB) && !fits(n + 1, nh, 4 * MIB));
        }
    }

    #[test]
    fn batch_keeps_u32_indices_in_range() {
        // With unlimited buffers the cap keeps (b*n_hashes+chain) << HASH_BITS and * BLOCK_SIZE in u32.
        for nh in [1, 2] {
            let n = max_blocks_per_batch(&limits(u64::MAX, u64::MAX, u32::MAX), nh);
            assert_eq!(n as u64 * nh as u64, 1u64 << (32 - HASH_BITS.max(LOG2_BLOCK)));
        }
    }

    #[test]
    fn batch_is_zero_when_one_block_does_not_fit() {
        assert_eq!(max_blocks_per_batch(&limits(256 * 1024, 256 * 1024, 65535), 2), 0);
        assert_eq!(max_blocks_per_batch(&limits(BLOCK_SIZE as u64, BLOCK_SIZE as u64, 65535), 1), 0);
    }
}
