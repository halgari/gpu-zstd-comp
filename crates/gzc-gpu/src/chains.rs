//! Host side of the K1 hash-chain build kernel.
//!
//! Two kernels build the same chains from the same `head` and `pred` buffers. Both run a
//! persistent grid of at most `HEAD_TABLES` workgroups, each reusing one head table for the chains
//! it builds (workgroup w: chains w, w + G, ..), so `head` holds at most `HEAD_TABLES` tables
//! (64 MiB) however large the batch, and the live tables stay in L2 (one table per block, 410 MB
//! for a 1638-block batch, made K1 DRAM-bound):
//! - with subgroups (`GpuContext::subgroups`, subgroup sizes 32..=128) `k1_chains_sg.wgsl`:
//!   tag-stamped table entries, so `head` is never cleared, and ballot-matched tiles with 2
//!   barriers per 256 positions;
//! - otherwise `k1_chains.wgsl`: a table clear per chain and a bitonic sort per 256-position tile
//!   (38 barriers).
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

/// Head tables K1 uses at most (one per workgroup of its persistent grid; the subgroup kernel's
/// default grid). 256 x 256 KiB = 64 MiB stays L2-resident enough on the RTX 5090 (96 MB L2);
/// more live tables made K1 DRAM-bound, fewer leave it latency-bound.
pub const HEAD_TABLES: u32 = 256;

/// Bytes of one head table (2^HASH_BITS u32 entries).
const TABLE_BYTES: u64 = (1u64 << HASH_BITS) * 4;

/// Bytes of the `head` buffer K1 needs for `n_blocks` with `n_hashes` chains per block
/// (`MatchParams::n_hashes`): one table per chain, at most `HEAD_TABLES`.
pub fn head_bytes(n_blocks: u32, n_hashes: u32) -> u64 {
    (n_blocks as u64 * n_hashes as u64).min(HEAD_TABLES as u64) * TABLE_BYTES
}

/// Bytes of the `pred` buffer K1 writes for `n_blocks` with `n_hashes` chains per block.
pub fn pred_bytes(n_blocks: u32, n_hashes: u32) -> u64 {
    n_blocks as u64 * n_hashes as u64 * BLOCK_SIZE as u64 * 4
}

/// Largest `n_blocks` one `ChainsKernel::record` call may take under `limits` with `n_hashes`
/// chains per block: the data (n*BLOCK_SIZE + 4 bytes), head and pred buffers each fit one
/// storage binding and one buffer, and the kernels' u32 pred indices `(b*n_hashes+chain) *
/// BLOCK_SIZE` cannot wrap (head indices `table << HASH_BITS` stay below 2^24). The block count is
/// also kept within `max_compute_workgroups_per_dimension`, which K2 dispatches over. 0 if one
/// block doesn't fit.
pub fn max_blocks_per_batch(limits: &wgpu::Limits, n_hashes: u32) -> u32 {
    let limit = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
    let nh = n_hashes as u64;
    let by_data = limit.saturating_sub(4) / BLOCK_SIZE as u64;
    // head_bytes(n) <= limit: always once HEAD_TABLES tables fit, else n * nh tables must.
    let by_head = if HEAD_TABLES as u64 * TABLE_BYTES <= limit { u64::MAX } else { limit / TABLE_BYTES / nh };
    let by_buffers = by_data.min(by_head).min(limit / pred_bytes(1, n_hashes));
    let by_index = (1u64 << (32 - LOG2_BLOCK)) / nh;
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
    /// Workgroups (= live head tables) per dispatch, at most the tables of the head buffer
    /// (`HEAD_TABLES` for a full batch); `None` = all of them. Environment override:
    /// `GZC_K1_GROUPS` (fewer tables suit GPUs with a smaller L2). Raised when needed so a
    /// dispatch uses at most `max_gen` tags.
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

    /// data: packed blocks; head: at least `head_bytes(n_blocks, n_hashes)` (scratch: the fallback
    /// clears a table before each chain; the subgroup kernel stamps entries with fresh per-buffer
    /// tags instead and clears the whole buffer only when tags run out, once per ~32767 chains per
    /// table); pred: at least `pred_bytes(n_blocks, n_hashes)`, layout [block][chain][pos] (Dfast:
    /// chain 0 long, 1 short; Single: chain 0 over `hash_width(min_match)`).
    ///
    /// Precondition: `n_blocks <= max_blocks_per_batch(&ctx.device.limits(), n_hashes)`. That
    /// keeps every buffer within the binding/buffer limits and `n_blocks * n_hashes <= 2^(32 -
    /// LOG2_BLOCK)` so the shaders' u32 pred indices do not wrap.
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
        // Workgroup w builds chains w, w + groups, .. in head table w.
        let n_tasks = n_blocks * self.n_hashes;
        let tables = (head.size() / TABLE_BYTES).min(HEAD_TABLES as u64) as u32;
        assert!(tables >= n_tasks.min(HEAD_TABLES), "head buffer smaller than head_bytes({n_blocks}, {})", self.n_hashes);
        let max_wg = ctx.device.limits().max_compute_workgroups_per_dimension;
        let mut groups = n_tasks.min(tables).min(max_wg);
        let mut generation = 0;
        if let Some(o) = &self.sg {
            groups = groups.min(o.groups.unwrap_or(u32::MAX).max(n_tasks.div_ceil(o.max_gen)));
            // One tag per chain a workgroup builds.
            let clear;
            (generation, clear) = next_gen(head, n_tasks.div_ceil(groups), o.max_gen);
            if clear {
                enc.clear_buffer(head, 0, None);
            }
        }
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k1"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: head.as_entire_binding() },
                // Exactly this dispatch's chains: the fallback reads n_tasks from its length.
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: pred,
                        offset: 0,
                        size: std::num::NonZeroU64::new(pred_bytes(n_blocks, self.n_hashes)),
                    }),
                },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k1"), timestamp_writes });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        if self.sg.is_some() {
            pass.set_immediates(0, bytemuck::cast_slice(&[generation, n_tasks, 0, 0]));
        }
        pass.dispatch_workgroups(groups, 1, 1);
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
        // pred-bound: 1 MiB per block at 128K (512 KiB with one chain), 128 KiB at 16K; head is
        // at most 64 MiB.
        #[cfg(feature = "block-128k")]
        assert_eq!((n2, n1), (128, 256));
        #[cfg(feature = "block-16k")]
        assert_eq!((n2, n1), (1024, 2048));
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
        // With unlimited buffers the cap keeps (b*n_hashes+chain) * BLOCK_SIZE in u32.
        for nh in [1, 2] {
            let n = max_blocks_per_batch(&limits(u64::MAX, u64::MAX, u32::MAX), nh);
            assert_eq!(n as u64 * nh as u64, 1u64 << (32 - LOG2_BLOCK));
        }
        assert!((HEAD_TABLES as u64) << HASH_BITS <= 1 << 32);
    }

    #[test]
    fn head_holds_one_table_per_chain_up_to_head_tables() {
        assert_eq!(head_bytes(3, 2), 6 * TABLE_BYTES);
        assert_eq!(head_bytes(HEAD_TABLES, 1), HEAD_TABLES as u64 * TABLE_BYTES);
        assert_eq!(head_bytes(1638, 1), head_bytes(HEAD_TABLES, 1));
        assert_eq!(head_bytes(1365, 2), head_bytes(HEAD_TABLES, 1));
        // A limit below the full head bounds the chains to the tables that fit.
        let n = max_blocks_per_batch(&limits(u64::MAX, 40 * TABLE_BYTES, 65535), 2);
        assert!(fits(n, 2, 40 * TABLE_BYTES) && !fits(n + 1, 2, 40 * TABLE_BYTES));
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
