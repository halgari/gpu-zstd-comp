//! Host side of the K1 hash-chain build kernel.
//!
//! Two kernels build the same chains from the same `head` and `pred` buffers. Both run a
//! persistent grid of at most `HEAD_TABLES` workgroups, each reusing one head table for the chains
//! it builds (workgroup w: chains w, w + G, ..), so `head` holds at most `HEAD_TABLES` tables
//! (64 MiB) however large the batch, and the live tables stay in L2 (one table per block, 410 MB
//! for a 1638-block batch, made K1 DRAM-bound):
//! - with subgroups (`GpuContext::subgroups`, subgroup sizes 32..=128, and a passing self-test)
//!   `k1_chains_sg.wgsl`: a table clear per workgroup per dispatch, tag-stamped entries so the
//!   table is not cleared between chains, and ballot-matched tiles with 2 barriers per 256
//!   positions;
//! - otherwise `k1_chains.wgsl`: a table clear per chain and a bitonic sort per 256-position tile
//!   (38 barriers).
//!
//! Neither keeps state across dispatches: `head` is pure per-dispatch scratch.
use crate::context::{GpuContext, pack_blocks, params_wgsl};
use gzc_core::config::{BLOCK_SIZE, HASHED_POSITIONS, HASH_BITS, LOG2_BLOCK, NO_POS};
use gzc_core::params::MatchParams;

const K1_WGSL: &str = include_str!("shaders/k1_chains.wgsl");
const K1_SG_WGSL: &str = include_str!("shaders/k1_chains_sg.wgsl");

/// Most chains one workgroup of the subgroup kernel may build in a dispatch: its j-th chain stamps
/// head entries `(j + 1) << LOG2_BLOCK | (pos + 1)` in a u32.
pub const MAX_TAG: u32 = (1u32 << (32 - LOG2_BLOCK)) - 1;

/// Head tables K1 allocates at most (one per workgroup of its persistent grid).
pub const HEAD_TABLES: u32 = 256;

/// Workgroups (live head tables) of the subgroup kernel by default: 128 x 256 KiB = 32 MiB, sized
/// for the ~32 MB L2 of 8 GB-class target GPUs. The RTX 5090 (96 MB L2) is fastest at 224-256
/// (`GZC_K1_GROUPS`).
pub const DEFAULT_SG_GROUPS: u32 = 128;

/// Predecessor bits of a K1 pred word (see `pred_fp`); `PRED_NONE` there means none.
pub const PRED_POS: u32 = 0x1_FFFF;
const _: () = assert!(LOG2_BLOCK <= 17);

/// The predecessor a K1 pred word holds (`NO_POS` for none), without its fingerprint.
pub fn pred_of_word(w: u32) -> u32 {
    let pr = w & PRED_POS;
    if pr == PRED_POS { NO_POS } else { pr }
}

/// Fingerprint bits K1 stores in the pred word of position `p < HASHED_POSITIONS` of `block`
/// (common.wgsl `pred_fp`): bits 17..24 hash bytes p..p+4, bits 24..32 are byte p + 4.
pub fn pred_fp(block: &[u8], p: usize) -> u32 {
    let lo = u32::from_le_bytes(block[p..p + 4].try_into().unwrap());
    ((lo.wrapping_mul(0x85EB_CA6B) >> 25) << 17) | ((block[p + 4] as u32) << 24)
}

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
    gpu_preds_with(ctx, &kernel, blocks)
}

/// `gpu_preds` with a given kernel.
pub fn gpu_preds_with(ctx: &GpuContext, kernel: &ChainsKernel, blocks: &[&[u8]]) -> anyhow::Result<Vec<Vec<Vec<u32>>>> {
    let nh = kernel.n_hashes();
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

/// K1 options. `Default` is what `ChainsKernel::new` uses.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChainsOptions {
    /// Workgroups (= live head tables) per dispatch, capped by the tables of the head buffer;
    /// `None` = `GZC_K1_GROUPS` if set, else `DEFAULT_SG_GROUPS` for the subgroup kernel and all
    /// tables for the fallback. Fewer live tables suit GPUs with a smaller L2.
    pub groups: Option<u32>,
    /// Pick each lane's ballot word at run time even for subgroups of at most 32 lanes (tests use
    /// it to cover the code path of wider subgroups).
    pub wide_masks: bool,
    /// Test only: build the subgroup kernel with a deliberately wrong in-chunk link, so its
    /// self-test fails and `ChainsKernel::with_options` falls back.
    pub break_subgroup_kernel: bool,
}

/// K1 pipeline, built for one `MatchParams`' chains; `record` lets later stages run it on their
/// own buffers.
pub struct ChainsKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    n_hashes: u32,
    subgroups: bool,
    opts: ChainsOptions,
    /// `GZC_K1_GROUPS`, parsed and validated once at construction (`build`), not per `record`
    /// call: `None` if it's unset, else the workgroup count it named. A non-numeric or
    /// non-positive value errors clearly here, the same way `GZC_K3_MODE`/`GZC_K3_W` do.
    env_groups: Option<u32>,
}

impl ChainsKernel {
    /// Builds K1 for the chains of `params` (its `N_HASHES` and `MIN_MATCH` are injected as WGSL
    /// constants): the subgroup kernel when `ctx.subgroups`, the subgroup sizes suit it and its
    /// self-test passes, else the fallback. Errors if `params` is invalid.
    pub fn new(ctx: &GpuContext, params: &MatchParams) -> anyhow::Result<Self> {
        Self::with_options(ctx, params, ChainsOptions::default())
    }

    /// `new` with explicit options.
    pub fn with_options(ctx: &GpuContext, params: &MatchParams, opts: ChainsOptions) -> anyhow::Result<Self> {
        params.validate().map_err(|e| anyhow::anyhow!("invalid match params {params:?}: {e}"))?;
        // The subgroup kernel splits its 256-lane tiles into 32-lane chunks, each inside one
        // subgroup, and reads ballots of up to 128 lanes.
        let info = &ctx.adapter_info;
        if ctx.subgroups && info.subgroup_min_size >= 32 && info.subgroup_max_size <= 128 {
            // Build + self-test inside their own error scope (like `probe_lanes`), so a wgpu
            // validation error from the subgroup shader/pipeline (not just a wrong self-test
            // result) also falls back here instead of surfacing uncaptured, which wgpu may
            // attribute to a later, unrelated error scope (e.g. `Pipeline::new`'s).
            let built = crate::compressor::with_error_scopes(ctx, || {
                let k = Self::build(ctx, params, opts, true)?;
                k.self_test(ctx, params)?;
                Ok(k)
            });
            match built {
                Ok(k) => return Ok(k),
                Err(e) => eprintln!("gzc-gpu: K1 subgroup kernel failed to build or its self-test ({e}); using the fallback K1"),
            }
        }
        Self::build(ctx, params, opts, false)
    }

    fn build(ctx: &GpuContext, params: &MatchParams, opts: ChainsOptions, subgroups: bool) -> anyhow::Result<Self> {
        let env_groups = match std::env::var("GZC_K1_GROUPS") {
            Ok(v) => {
                let g: u32 = v.parse().map_err(|_| anyhow::anyhow!("GZC_K1_GROUPS={v}: not a number"))?;
                anyhow::ensure!(g > 0, "GZC_K1_GROUPS={v}: expected a positive number");
                Some(g)
            }
            Err(_) => None,
        };
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
        let module = if subgroups {
            // naga (wgpu 30) takes subgroup operations from Features::SUBGROUP and rejects the
            // `enable subgroups;` directive.
            let wide = opts.wide_masks || ctx.adapter_info.subgroup_max_size > 32;
            let mut src = format!("{}{}", params_wgsl(params), K1_SG_WGSL)
                .replace("K1_BALLOT_WORD", if wide { "[word]" } else { ".x" });
            if opts.break_subgroup_kernel {
                src = src.replace("firstLeadingBit(lower)", "firstTrailingBit(lower)");
            }
            ctx.shader_trusted("k1_chains_sg", &src)
        } else {
            ctx.shader_trusted("k1_chains", &format!("{}{K1_WGSL}", params_wgsl(params)))
        };
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k1_chains"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self { pipeline, layout, n_hashes: params.n_hashes(), subgroups, opts, env_groups })
    }

    /// Guards the subgroup kernel's assumptions (full, equally sized subgroups of >= 32 lanes
    /// covering the 256-lane workgroup, exact ballots): builds the chains of small-alphabet, text
    /// and texture-like blocks, twice on one head buffer and once with one workgroup building all
    /// of them in a row, and compares them with `gzc_core::reference::chains` — both the decoded
    /// predecessor (`pred_of_word`) and the raw K1 word (predecessor bits plus the `pred_fp`
    /// fingerprint), so a subgroup kernel that gets the predecessor right but the fingerprint
    /// wrong (K2 relies on it to skip candidates without loading bytes) still fails here and falls
    /// back.
    fn self_test(&self, ctx: &GpuContext, params: &MatchParams) -> anyhow::Result<()> {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let alphabet: Vec<u8> = (0..BLOCK_SIZE)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 32) % 3) as u8
            })
            .collect();
        let blocks = [alphabet, gzc_core::synth::text(11, BLOCK_SIZE), gzc_core::synth::dds_like(12, BLOCK_SIZE)];
        let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
        let want: Vec<Vec<Vec<u32>>> = blocks.iter().map(|b| gzc_core::reference::chains(b, params)).collect();
        // The raw K1 word `want` predicts for each block/chain/position: `pred_word(pr, fp)` from
        // common.wgsl, i.e. PRED_POS with the fingerprint OR'd in when there's no predecessor
        // (p < HASHED_POSITIONS), else plain PRED_POS.
        let want_words: Vec<Vec<Vec<u32>>> = blocks
            .iter()
            .zip(&want)
            .map(|(block, chains)| {
                chains
                    .iter()
                    .map(|chain| {
                        (0..BLOCK_SIZE)
                            .map(|p| {
                                if p < HASHED_POSITIONS {
                                    let pr = chain[p];
                                    (if pr == NO_POS { PRED_POS } else { pr }) | pred_fp(block, p)
                                } else {
                                    PRED_POS
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let n = blocks.len() as u32;
        let packed = pack_blocks(&refs);
        let data = ctx.storage_buffer("k1.selftest.data", (packed.len() * 4) as u64, false);
        ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&packed));
        let head = ctx.storage_buffer("k1.selftest.head", head_bytes(n, self.n_hashes), false);
        let pred = ctx.storage_buffer("k1.selftest.pred", pred_bytes(n, self.n_hashes), true);
        let one_group = Self { opts: ChainsOptions { groups: Some(1), ..self.opts }, ..self.shallow_clone() };
        let per_block = self.n_hashes as usize * BLOCK_SIZE;
        for (round, k) in [self, self, &one_group].into_iter().enumerate() {
            let raw = k.run_words(ctx, &data, &head, &pred, n);
            let got_words: Vec<Vec<Vec<u32>>> =
                raw.chunks_exact(per_block).map(|block| block.chunks_exact(BLOCK_SIZE).map(|c| c.to_vec()).collect()).collect();
            let got: Vec<Vec<Vec<u32>>> = got_words
                .iter()
                .map(|block| block.iter().map(|chain| chain.iter().copied().map(pred_of_word).collect()).collect())
                .collect();
            if let Some(b) = (0..blocks.len()).find(|&b| got[b] != want[b]) {
                anyhow::bail!("pred of self-test block {b} differs from the CPU in round {round}");
            }
            if let Some(b) = (0..blocks.len()).find(|&b| got_words[b] != want_words[b]) {
                anyhow::bail!("raw pred word (fingerprint bits) of self-test block {b} differs from the CPU in round {round}");
            }
        }
        Ok(())
    }

    fn shallow_clone(&self) -> Self {
        Self {
            pipeline: self.pipeline.clone(),
            layout: self.layout.clone(),
            n_hashes: self.n_hashes,
            subgroups: self.subgroups,
            opts: self.opts,
            env_groups: self.env_groups,
        }
    }

    /// Chains per block (`MatchParams::n_hashes`) this kernel builds.
    pub fn n_hashes(&self) -> u32 {
        self.n_hashes
    }

    /// True when this is the subgroup kernel, false for the fallback.
    pub fn uses_subgroups(&self) -> bool {
        self.subgroups
    }

    /// Records K1 on `data` into `head`/`pred` for `n_blocks`, submits it and reads `pred` back:
    /// per block, one BLOCK_SIZE-long pred array per chain (`pred_of_word` of K1's words).
    /// `pred` needs COPY_SRC.
    pub fn run(
        &self,
        ctx: &GpuContext,
        data: &wgpu::Buffer,
        head: &wgpu::Buffer,
        pred: &wgpu::Buffer,
        n_blocks: u32,
    ) -> Vec<Vec<Vec<u32>>> {
        let per_block = self.n_hashes as usize * BLOCK_SIZE;
        let all: Vec<u32> = self.run_words(ctx, data, head, pred, n_blocks).into_iter().map(pred_of_word).collect();
        all.chunks_exact(per_block).map(|block| block.chunks_exact(BLOCK_SIZE).map(|c| c.to_vec()).collect()).collect()
    }

    /// `run`, returning K1's raw pred words (predecessor and fingerprint, see `pred_fp`),
    /// layout `[block][chain][pos]`.
    pub fn run_words(
        &self,
        ctx: &GpuContext,
        data: &wgpu::Buffer,
        head: &wgpu::Buffer,
        pred: &wgpu::Buffer,
        n_blocks: u32,
    ) -> Vec<u32> {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("k1") });
        self.record(ctx, &mut enc, data, head, pred, n_blocks);
        ctx.queue.submit([enc.finish()]);
        ctx.read_buffer(pred, 0, self.n_hashes as usize * BLOCK_SIZE * n_blocks as usize)
    }

    /// data: packed blocks; head: at least `head_bytes(n_blocks, n_hashes)`, per-dispatch scratch
    /// (each workgroup clears its table in-kernel; nothing is carried between dispatches); pred:
    /// at least `pred_bytes(n_blocks, n_hashes)`, layout [block][chain][pos] (Dfast: chain 0 long,
    /// 1 short; Single: chain 0 over `hash_width(min_match)`).
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
        let wanted = self.opts.groups.or(self.env_groups).unwrap_or(if self.subgroups { DEFAULT_SG_GROUPS } else { u32::MAX });
        // At most MAX_TAG chains per workgroup (the subgroup kernel's tags).
        let groups = wanted.max(n_tasks.div_ceil(MAX_TAG)).min(n_tasks).min(tables).min(max_wg);
        assert!(n_tasks.div_ceil(groups) <= MAX_TAG, "{n_tasks} chains over {groups} workgroups");
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k1"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: head.as_entire_binding() },
                // Exactly this dispatch's chains: the kernels read n_tasks from its length.
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
    fn tags_fit_a_u32() {
        assert_eq!(MAX_TAG as u64 * (1u64 << LOG2_BLOCK) + (1u64 << LOG2_BLOCK) - 1, u32::MAX as u64);
    }

    #[test]
    fn batch_is_zero_when_one_block_does_not_fit() {
        assert_eq!(max_blocks_per_batch(&limits(256 * 1024, 256 * 1024, 65535), 2), 0);
        assert_eq!(max_blocks_per_batch(&limits(BLOCK_SIZE as u64, BLOCK_SIZE as u64, 65535), 1), 0);
    }
}
