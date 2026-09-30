//! Host side of the K1 hash-chain build kernel.
//!
//! Two kernels build the same chains from the same `head` and `pred` buffers:
//! - with subgroups (`GpuContext::subgroups`, subgroup sizes 32..=128) `k1_chains_sg.wgsl`: a
//!   persistent grid of `DEFAULT_GROUPS` workgroups, each reusing one L2-resident head table for
//!   its blocks, with tag-stamped entries so `head` is never cleared per batch, and ballot-matched
//!   tiles with 2 barriers per 256 positions;
//! - otherwise the original `k1_chains.wgsl`: one workgroup and head table per chain, a bitonic
//!   sort per 256-position tile (38 barriers), `head` cleared per batch.
use crate::context::{GpuContext, K1_IMMEDIATE_BYTES, pack_blocks, params_wgsl};
use gzc_core::config::{BLOCK_SIZE, HASH_BITS, LOG2_BLOCK};
use gzc_core::params::MatchParams;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

const K1_WGSL: &str = include_str!("shaders/k1_chains.wgsl");
const K1_SG_WGSL: &str = include_str!("shaders/k1_chains_sg.wgsl");

/// Largest gen the subgroup kernel's `head` tags can hold: entries are
/// `(gen << LOG2_BLOCK) | (pos + 1)` in a u32.
pub const MAX_GEN: u32 = (1u32 << (32 - LOG2_BLOCK)) - 1;

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
        out.extend(kernel.run(ctx, &data, &head, &pred, n));
    }
    Ok(out)
}

/// Options of the subgroup kernel (ignored by the fallback). `Default` is what `ChainsKernel::new`
/// uses.
#[derive(Clone, Copy, Debug)]
pub struct ChainsOptions {
    /// Largest gen handed out before the `head` buffer is cleared and gens restart at 1 (at most
    /// `MAX_GEN`; tests lower it to exercise the wrap).
    pub max_gen: u32,
    /// Workgroups (= live head tables) per dispatch; `None` = `DEFAULT_GROUPS`. Environment
    /// override: `GZC_K1_GROUPS`. Raised when needed so a dispatch uses at most `max_gen` tags.
    pub groups: Option<u32>,
    /// Pick each lane's ballot word at run time even for subgroups of at most 32 lanes (tests use
    /// it to cover the code path of wider subgroups).
    pub wide_masks: bool,
}

impl Default for ChainsOptions {
    fn default() -> Self {
        let groups = std::env::var("GZC_K1_GROUPS").ok().and_then(|v| v.parse().ok()).filter(|&g| g > 0);
        Self { max_gen: MAX_GEN, groups, wide_masks: false }
    }
}

/// Default workgroups of the subgroup kernel: 256 head tables of 256 KiB (64 MiB) is what stays
/// L2-resident enough on the RTX 5090 (96 MB L2); more tables made K1 DRAM-bound.
pub const DEFAULT_GROUPS: u32 = 256;

/// Last gen (tag) handed out per `head` buffer, keyed by the buffer's hash. Invariant: a key's
/// value is at least the largest tag in its buffer (every entry of a buffer carries a tag handed
/// out for it, or 0). A new buffer is zeroed; a key shared by two buffers (hash collision, or a
/// dropped buffer's id reused) only makes the value larger than needed; a wrap clears the buffer.
/// So a dispatch's fresh tags never match a stale entry.
static GENS: Mutex<BTreeMap<u64, u32>> = Mutex::new(BTreeMap::new());

/// Advances `last` (the last tag handed out) by `count` tags: returns the first of them and
/// whether the buffer must be cleared first because tags `last + 1 ..= last + count` would pass
/// `max_gen` (tags then restart at 1). Needs `1 <= count <= max_gen`.
fn advance_gen(last: &mut u32, count: u32, max_gen: u32) -> (u32, bool) {
    assert!((1..=max_gen).contains(&count), "{count} tags per dispatch, max {max_gen}");
    if max_gen - count < *last {
        *last = count;
        (1, true)
    } else {
        *last += count;
        (*last - count + 1, false)
    }
}

/// `advance_gen` for the tags of `head` (see `GENS`).
fn next_gen(head: &wgpu::Buffer, count: u32, max_gen: u32) -> (u32, bool) {
    let mut h = std::hash::DefaultHasher::new();
    head.hash(&mut h);
    advance_gen(GENS.lock().unwrap().entry(h.finish()).or_insert(0), count, max_gen)
}

/// K1 pipeline, built for one `MatchParams`' chains; `record` lets later stages run it on their
/// own buffers.
pub struct ChainsKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    n_hashes: u32,
    /// Some (with its options): the subgroup kernel; None: the fallback.
    sg: Option<ChainsOptions>,
}

impl ChainsKernel {
    /// Builds K1 for the chains of `params` (its `N_HASHES` and `MIN_MATCH` are injected as WGSL
    /// constants): the subgroup kernel when `ctx.subgroups`, else the fallback. Errors if `params`
    /// is invalid.
    pub fn new(ctx: &GpuContext, params: &MatchParams) -> anyhow::Result<Self> {
        Self::with_options(ctx, params, ChainsOptions::default())
    }

    /// `new` with explicit subgroup-kernel options.
    pub fn with_options(ctx: &GpuContext, params: &MatchParams, opts: ChainsOptions) -> anyhow::Result<Self> {
        params.validate().map_err(|e| anyhow::anyhow!("invalid match params {params:?}: {e}"))?;
        anyhow::ensure!((1..=MAX_GEN).contains(&opts.max_gen), "max_gen {} not in 1..={MAX_GEN}", opts.max_gen);
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
        // The subgroup kernel splits its 256-lane tiles into 32-lane chunks, each inside one
        // subgroup, and reads ballots of up to 128 lanes.
        let info = &ctx.adapter_info;
        let sg_ok = ctx.subgroups && info.subgroup_min_size >= 32 && info.subgroup_max_size <= 128;
        let sg = sg_ok.then_some(opts);
        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("k1"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: if sg.is_some() { K1_IMMEDIATE_BYTES } else { 0 },
        });
        let module = match &sg {
            // naga (wgpu 30) takes subgroup operations from Features::SUBGROUP and rejects the
            // `enable subgroups;` directive.
            Some(o) => ctx.shader(
                "k1_chains_sg",
                &format!("{}{}", params_wgsl(params), K1_SG_WGSL).replace(
                    "K1_BALLOT_WORD",
                    if o.wide_masks || info.subgroup_max_size > 32 { "[word]" } else { ".x" },
                ),
            ),
            None => ctx.shader("k1_chains", &format!("{}{K1_WGSL}", params_wgsl(params))),
        };
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k1_chains"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self { pipeline, layout, n_hashes: params.n_hashes(), sg })
    }

    /// Chains per block (`MatchParams::n_hashes`) this kernel builds.
    pub fn n_hashes(&self) -> u32 {
        self.n_hashes
    }

    /// True when this is the subgroup kernel (`GpuContext::subgroups`), false for the fallback.
    pub fn uses_subgroups(&self) -> bool {
        self.sg.is_some()
    }

    /// Records K1 on `data` into `head`/`pred` for `n_blocks`, submits it and reads `pred` back:
    /// per block, one BLOCK_SIZE-long pred array per chain. `pred` needs COPY_SRC.
    pub fn run(
        &self,
        ctx: &GpuContext,
        data: &wgpu::Buffer,
        head: &wgpu::Buffer,
        pred: &wgpu::Buffer,
        n_blocks: u32,
    ) -> Vec<Vec<Vec<u32>>> {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("k1") });
        self.record(ctx, &mut enc, data, head, pred, n_blocks);
        ctx.queue.submit([enc.finish()]);
        let per_block = self.n_hashes as usize * BLOCK_SIZE;
        let all: Vec<u32> = ctx.read_buffer(pred, 0, per_block * n_blocks as usize);
        all.chunks_exact(per_block).map(|block| block.chunks_exact(BLOCK_SIZE).map(|c| c.to_vec()).collect()).collect()
    }

    /// data: packed blocks; head: at least `head_bytes(n_blocks, n_hashes)` (the fallback clears
    /// it via encoder.clear_buffer; the subgroup kernel uses its first min(n_blocks * n_hashes,
    /// groups) tables, tags their entries with fresh per-buffer tags instead and clears the whole
    /// buffer only when tags run out, once per ~32767 blocks per table); pred:
    /// `pred_bytes(n_blocks, n_hashes)`, layout [block][chain][pos] (Dfast: chain 0 long, 1 short;
    /// Single: chain 0 over `hash_width(min_match)`).
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
        let mut generation = 0;
        let mut dispatch_groups = 0;
        match &self.sg {
            None => enc.clear_buffer(head, 0, Some(head_bytes(n_blocks, self.n_hashes))),
            Some(o) => {
                // Workgroup w builds chains w, w + groups, .. (one head table, one tag per chain).
                let n_tasks = n_blocks * self.n_hashes;
                let max_wg = ctx.device.limits().max_compute_workgroups_per_dimension;
                let groups = o.groups.unwrap_or(DEFAULT_GROUPS).max(n_tasks.div_ceil(o.max_gen));
                dispatch_groups = groups.min(n_tasks).min(max_wg);
                let clear;
                (generation, clear) = next_gen(head, n_tasks.div_ceil(dispatch_groups), o.max_gen);
                if clear {
                    enc.clear_buffer(head, 0, None);
                }
            }
        }
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
        match &self.sg {
            None => pass.dispatch_workgroups(n_blocks, self.n_hashes, 1),
            Some(_) => {
                pass.set_immediates(0, bytemuck::cast_slice(&[generation, n_blocks * self.n_hashes, 0, 0]));
                pass.dispatch_workgroups(dispatch_groups, 1, 1);
            }
        }
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
    fn tags_advance_and_wrap_with_a_clear() {
        let mut last = 0;
        assert_eq!(advance_gen(&mut last, 2, 5), (1, false));
        assert_eq!(advance_gen(&mut last, 2, 5), (3, false));
        // 5..=6 would pass 5: clear, restart at 1.
        assert_eq!(advance_gen(&mut last, 2, 5), (1, true));
        assert_eq!(advance_gen(&mut last, 3, 5), (3, false));
        assert_eq!(last, 5);
        assert_eq!(advance_gen(&mut last, 1, 5), (1, true));
        assert_eq!(advance_gen(&mut last, 5, 5), (1, true));
        // A key's value above max_gen (another kernel's larger max_gen) also wraps.
        let mut last = 9;
        assert_eq!(advance_gen(&mut last, 1, 5), (1, true));
        assert_eq!(MAX_GEN as u64 * (1u64 << LOG2_BLOCK) + (1u64 << LOG2_BLOCK) - 1, u32::MAX as u64);
    }

    #[test]
    fn batch_is_zero_when_one_block_does_not_fit() {
        assert_eq!(max_blocks_per_batch(&limits(256 * 1024, 256 * 1024, 65535), 2), 0);
        assert_eq!(max_blocks_per_batch(&limits(BLOCK_SIZE as u64, BLOCK_SIZE as u64, 65535), 1), 0);
    }
}
