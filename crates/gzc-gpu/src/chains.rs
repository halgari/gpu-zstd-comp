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
use crate::sizing::{TABLE_BYTES, chain_pred_bytes, head_bytes, max_blocks_per_batch_for};
use gzc_core::config::{BLOCK_SIZE, HASHED_POSITIONS, HASH_BITS, LOG2_BLOCK, NO_POS};
use gzc_core::params::{Hashes, SparseChain, MatchParams};

const K1_WGSL: &str = include_str!("shaders/k1_chains.wgsl");
const K1_SG_WGSL: &str = include_str!("shaders/k1_chains_sg.wgsl");

/// Most chains one workgroup of the subgroup kernel may build in a dispatch: its j-th chain stamps
/// head entries `(j + 1) << LOG2_BLOCK | (pos + 1)` in a u32.
pub(crate) const MAX_TAG: u32 = (1u32 << (32 - LOG2_BLOCK)) - 1;

/// Head tables K1 allocates at most (one per workgroup of its persistent grid).
pub(crate) const HEAD_TABLES: u32 = 256;

/// Workgroups (live head tables) of the subgroup kernel by default: 128 x 256 KiB = 32 MiB, sized
/// for the ~32 MB L2 of 8 GB-class target GPUs. The RTX 5090 (96 MB L2) is fastest at 224-256
/// (`GpuOptions::k1_groups`).
pub(crate) const DEFAULT_SG_GROUPS: u32 = 128;

/// Predecessor bits of a K1 pred word (see `pred_fp`); `PRED_NONE` there means none.
pub const PRED_POS: u32 = 0x1_FFFF;
const _: () = assert!(LOG2_BLOCK == 16);

/// The predecessor a K1 pred word holds (`NO_POS` for none), without its fingerprint.
pub fn pred_of_word(w: u32) -> u32 {
    let pr = w & PRED_POS;
    if pr == PRED_POS { NO_POS } else { pr }
}

/// Fingerprint bits K1 stores in the pred word of position `p < HASHED_POSITIONS` of `block`
/// (common.wgsl `pred_fp`): bits 17..24 hash bytes p..p+4, bits 24..32 are byte p + 4.
pub(crate) fn pred_fp(block: &[u8], p: usize) -> u32 {
    let lo = u32::from_le_bytes(block[p..p + 4].try_into().unwrap());
    ((lo.wrapping_mul(0x85EB_CA6B) >> 25) << 17) | ((block[p + 4] as u32) << 24)
}

/// Fingerprint bits K1 stores in the pred words of the `Opt3` h3 chain (chain 1 of `opt` params;
/// common.wgsl `pred_fp3`) for position `p < HASHED_POSITIONS`: bits 17..24 hash bytes p..p+3,
/// bits 24..32 are byte p + 3. A differing hash field means a match shorter than 3 bytes, a
/// differing byte field one of at most 3.
pub(crate) fn pred_fp3(block: &[u8], p: usize) -> u32 {
    let lo = u32::from_le_bytes(block[p..p + 4].try_into().unwrap());
    ((((lo & 0xFF_FFFF).wrapping_mul(0x85EB_CA6B)) >> 25) << 17) | (lo & 0xFF00_0000)
}

/// The fingerprint of `p` in chain `chain`'s pred words under `params`: `pred_fp3` on the `Opt3`
/// h3 chain, `pred_fp` everywhere else.
pub fn chain_fp(params: &MatchParams, chain: usize, block: &[u8], p: usize) -> u32 {
    if params.hashes == Hashes::Opt3 && chain == 1 { pred_fp3(block, p) } else { pred_fp(block, p) }
}

/// `params_wgsl(p)` plus the finder's constants: `KEY_SHIFT = HASH_BITS - p.hash_bits`, so a
/// kernel's key is `hash >> KEY_SHIFT` (== `gzc_core::hash::key`), and `OPT3` (the `Opt3` chains:
/// chain 0 `hash_width(.., 4)`, chain 1 `hash3` with `pred_fp3` fingerprints).
pub(crate) fn finder_wgsl(p: &MatchParams) -> String {
    format!(
        "{}const KEY_SHIFT: u32 = {}u;\nconst OPT3: bool = {};\n{}",
        params_wgsl(p),
        HASH_BITS - p.hash_bits,
        p.hashes == Hashes::Opt3,
        layout_wgsl(p)
    )
}

/// The pred layout and sparse long chains of `p` as WGSL constants (`chain_span`): `N_FULL` full
/// chains, `PRED_PER_BLOCK` words per block, and for k < `N_SPARSE` (at most 3) sparse chain k's
/// key width `SP_W{k}`, stride `SP_S{k}`, walk depth `SP_D{k}`, hashed slots `SP_N{k}` and word
/// offset in the block `SP_OFF{k}` (width, depth, slots and offset 0, stride 1 for absent chains).
pub(crate) fn layout_wgsl(p: &MatchParams) -> String {
    let longs = long_chains(p);
    let full = full_chains(p);
    let mut s = format!(
        "const N_FULL: u32 = {full}u;\nconst N_SPARSE: u32 = {}u;\nconst PRED_PER_BLOCK: u32 = {}u;\n",
        longs.len(),
        pred_words_per_block(p)
    );
    for k in 0..3 {
        let (w, st, d, n, off) = match longs.get(k) {
            Some(c) => (c.width, c.stride, c.depth, long_chain_slots(c), chain_span(p, full + k as u32).0),
            None => (0, 1, 0, 0, 0),
        };
        s += &format!(
            "const SP_W{k}: u32 = {w}u;\nconst SP_S{k}: u32 = {st}u;\nconst SP_D{k}: u32 = {d}u;\n\
             const SP_N{k}: u32 = {n}u;\nconst SP_OFF{k}: u32 = {off}u;\n"
        );
    }
    s + LAYOUT_FNS_WGSL
}

/// Accessors of `layout_wgsl`'s per-chain constants by sparse chain index k < N_SPARSE, and the
/// long chains' hash.
const LAYOUT_FNS_WGSL: &str = "
fn sp_pick(k: u32, a: u32, b: u32, c: u32) -> u32 { return select(select(c, b, k == 1u), a, k == 0u); }
fn sp_width(k: u32) -> u32 { return sp_pick(k, SP_W0, SP_W1, SP_W2); }
fn sp_stride(k: u32) -> u32 { return sp_pick(k, SP_S0, SP_S1, SP_S2); }
fn sp_slots(k: u32) -> u32 { return sp_pick(k, SP_N0, SP_N1, SP_N2); }
fn sp_off(k: u32) -> u32 { return sp_pick(k, SP_OFF0, SP_OFF1, SP_OFF2); }
// The 16-bit long-chain hash of the w (5..=12) bytes at a word-aligned position whose words are
// lo, hi, h2 (== gzc_core::hash::hash_sparse): each word masked to the bytes below p + w.
fn long_hash(lo: u32, hi: u32, h2: u32, w: u32) -> u32 {
    // `& 31u`: no-ops for the widths that use them, but the shifts stay below 32 for any w.
    let mhi = select(0xFFFFFFFFu, (1u << ((8u * (w - 4u)) & 31u)) - 1u, w < 8u);
    var mh2 = 0u;
    if (w > 8u) { mh2 = select(0xFFFFFFFFu, (1u << ((8u * (w - 8u)) & 31u)) - 1u, w < 12u); }
    return (((lo * 0x9E3779B1u) ^ ((hi & mhi) * 0x85EBCA77u) ^ ((h2 & mh2) * 0x27D4EB2Fu)) * 0xC2B2AE3Du) >> 16u;
}
";

/// The sparse long chains of `p` (M6 `OptParams::sparse_chains`, in walk order after h4 and h3),
/// empty for every other preset.
pub(crate) fn long_chains(p: &MatchParams) -> Vec<SparseChain> {
    p.opt.iter().flat_map(|o| o.sparse_chains.into_iter().flatten()).collect()
}

/// Chains of `p` stored at full length (one pred word per position): `n_hashes` minus the sparse
/// long chains.
pub(crate) fn full_chains(p: &MatchParams) -> u32 {
    p.n_hashes() - long_chains(p).len() as u32
}

/// Positions a sparse long chain hashes: `p % stride == 0` and `p < BLOCK_SIZE - 12`
/// (`gzc_core::reference::sparse_chain_preds`), as slots `p / stride`.
pub(crate) fn long_chain_slots(c: &SparseChain) -> u32 {
    (BLOCK_SIZE as u32 - 12).div_ceil(c.stride)
}

/// Where chain `chain` of `p` starts inside a block's pred words, and how many words it has.
/// Layout per block: the full chains (`BLOCK_SIZE` words each, walk order), then each sparse long
/// chain compactly, one word per slot (`BLOCK_SIZE / stride` words; slot s is position
/// `s * stride`; slots from `long_chain_slots` on hold `PRED_NONE`).
pub fn chain_span(p: &MatchParams, chain: u32) -> (u64, u64) {
    let full = full_chains(p);
    if chain < full {
        return (chain as u64 * BLOCK_SIZE as u64, BLOCK_SIZE as u64);
    }
    let mut off = full as u64 * BLOCK_SIZE as u64;
    for (k, c) in long_chains(p).iter().enumerate() {
        let len = (BLOCK_SIZE as u32 / c.stride) as u64;
        if k as u32 + full == chain {
            return (off, len);
        }
        off += len;
    }
    panic!("chain {chain} out of range for {p:?}");
}

/// u32 pred words per block K1 writes for `p`'s chains (`chain_span`).
pub fn pred_words_per_block(p: &MatchParams) -> u64 {
    let (off, len) = chain_span(p, p.n_hashes() - 1);
    off + len
}

/// Whether K1 can build `p`'s sparse long chains: their slots must be word aligned (stride 4 or
/// 8), so a lane's key bytes are three whole data words.
pub(crate) fn long_chains_supported(p: &MatchParams) -> bool {
    long_chains(p).iter().all(|c| c.stride % 4 == 0 && c.width <= 12)
}

/// Expands one block's K1 pred words (`chain_span` layout) into `gzc_core::reference::chains`
/// form: one BLOCK_SIZE-long array per chain, `pred_of_word` decoded, `NO_POS` off a sparse
/// chain's slots.
pub(crate) fn expand_preds(p: &MatchParams, words: &[u32]) -> Vec<Vec<u32>> {
    (0..p.n_hashes())
        .map(|c| {
            let (off, len) = chain_span(p, c);
            let w = &words[off as usize..(off + len) as usize];
            if len == BLOCK_SIZE as u64 {
                return w.iter().copied().map(pred_of_word).collect();
            }
            let stride = BLOCK_SIZE / len as usize;
            let mut out = vec![NO_POS; BLOCK_SIZE];
            for (s, &x) in w.iter().enumerate() {
                out[s * stride] = pred_of_word(x);
            }
            out
        })
        .collect()
}

/// The raw K1 words (`chain_span` layout) of one block: per chain, `common.wgsl`'s `pred_word(pr,
/// fp)` of `gzc_core::reference::chains` and the chain's fingerprint (`chain_fp`) at every hashed
/// position (slot), plain `PRED_POS` past them.
pub fn expected_words(p: &MatchParams, block: &[u8]) -> Vec<u32> {
    let want = gzc_core::reference::chains(block, p);
    let longs = long_chains(p);
    let full = full_chains(p) as usize;
    let mut out = Vec::with_capacity(pred_words_per_block(p) as usize);
    for (c, chain) in want.iter().enumerate() {
        let (stride, hashed) = if c < full {
            (1, HASHED_POSITIONS)
        } else {
            let lc = &longs[c - full];
            (lc.stride as usize, long_chain_slots(lc) as usize)
        };
        for s in 0..BLOCK_SIZE / stride {
            let q = s * stride;
            out.push(if s < hashed {
                let pr = chain[q];
                (if pr == NO_POS { PRED_POS } else { pr }) | chain_fp(p, c, block, q)
            } else {
                PRED_POS
            });
        }
    }
    out
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
    let max_blocks = max_blocks_per_batch_for(&ctx.device.limits(), &kernel.params) as usize;
    anyhow::ensure!(max_blocks > 0, "device limits too small for one K1 block");

    let mut out = Vec::with_capacity(blocks.len());
    for batch in blocks.chunks(max_blocks) {
        let n = batch.len() as u32;
        let packed = pack_blocks(batch);
        let data = ctx.storage_buffer("k1.data", (packed.len() * 4) as u64, false);
        ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&packed));
        let head = ctx.storage_buffer("k1.head", head_bytes(n, nh), false);
        let pred = ctx.storage_buffer("k1.pred", chain_pred_bytes(n, &kernel.params), true);
        out.extend(kernel.run(ctx, &data, &head, &pred, n));
    }
    Ok(out)
}

/// K1 options. `Default` is what `ChainsKernel::new` uses.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChainsOptions {
    /// Workgroups (= live head tables) per dispatch, capped by the tables of the head buffer;
    /// `None` = `GpuOptions::k1_groups` if set, else `DEFAULT_SG_GROUPS` for the subgroup kernel and all
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
    /// The match params whose chains this kernel builds (their pred layout: `chain_span`).
    params: MatchParams,
    subgroups: bool,
    opts: ChainsOptions,
    /// The context's `GpuOptions::k1_groups`, checked at construction (`build`).
    ctx_groups: Option<u32>,
}

impl ChainsKernel {
    /// Builds K1 for the chains of `params` (its `N_HASHES` and `MIN_MATCH` are injected as WGSL
    /// constants): the subgroup kernel when `ctx.subgroups`, the subgroup sizes suit it and its
    /// self-test passes, else the fallback. Errors if `params` is invalid.
    pub fn new(ctx: &GpuContext, params: &MatchParams) -> anyhow::Result<Self> {
        Self::with_options(ctx, params, ChainsOptions::default())
    }

    /// Whether `ctx` may run the subgroup kernel (it then still has to pass its self-test): the
    /// kernel splits its 256-lane tiles into 32-lane chunks, each inside one subgroup, and reads
    /// ballots of up to 128 lanes, so every subgroup size the adapter reports must lie in 32..=128.
    /// Metal reports 4..=64 (its SIMD width is per pipeline), so Apple GPUs take the fallback.
    pub fn subgroup_kernel_possible(ctx: &GpuContext) -> bool {
        let info = &ctx.adapter_info;
        ctx.subgroups && info.subgroup_min_size >= 32 && info.subgroup_max_size <= 128
    }

    /// `new` with explicit options.
    pub fn with_options(ctx: &GpuContext, params: &MatchParams, opts: ChainsOptions) -> anyhow::Result<Self> {
        params.validate().map_err(|e| anyhow::anyhow!("invalid match params {params:?}: {e}"))?;
        anyhow::ensure!(long_chains_supported(params), "K1 builds sparse long chains of stride 4 or 8 only: {params:?}");
        if Self::subgroup_kernel_possible(ctx) {
            // Build + self-test inside their own error scope (like `probe_lanes`), so a wgpu
            // validation error from the subgroup shader/pipeline (not just a wrong self-test
            // result) also falls back here instead of surfacing uncaptured, which wgpu may
            // attribute to a later, unrelated error scope (e.g. `Pipeline::new`'s).
            let built = crate::kernels::with_error_scopes(ctx, || {
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
        let ctx_groups = ctx.opts.k1_groups;
        anyhow::ensure!(ctx_groups != Some(0), "GpuOptions::k1_groups: expected a positive number");
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
            let mut src = format!("{}{}", finder_wgsl(params), K1_SG_WGSL)
                .replace("K1_BALLOT_WORD", if wide { "[word]" } else { ".x" });
            if opts.break_subgroup_kernel {
                src = src.replace("firstLeadingBit(lower)", "firstTrailingBit(lower)");
            }
            ctx.shader("k1_chains_sg", &src)
        } else {
            ctx.shader("k1_chains", &format!("{}{K1_WGSL}", finder_wgsl(params)))
        };
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k1_chains"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: ctx.compilation_options(),
            cache: None,
        });
        Ok(Self { pipeline, layout, n_hashes: params.n_hashes(), params: *params, subgroups, opts, ctx_groups })
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
        // The raw K1 words `want` predicts for each block (`expected_words`, K1's layout).
        let want_words: Vec<Vec<u32>> = blocks.iter().map(|block| expected_words(params, block)).collect();
        let n = blocks.len() as u32;
        let packed = pack_blocks(&refs);
        let data = ctx.storage_buffer("k1.selftest.data", (packed.len() * 4) as u64, false);
        ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&packed));
        let head = ctx.storage_buffer("k1.selftest.head", head_bytes(n, self.n_hashes), false);
        let pred = ctx.storage_buffer("k1.selftest.pred", chain_pred_bytes(n, params), true);
        let one_group = Self { opts: ChainsOptions { groups: Some(1), ..self.opts }, ..self.shallow_clone() };
        let per_block = pred_words_per_block(params) as usize;
        for (round, k) in [self, self, &one_group].into_iter().enumerate() {
            let raw = k.run_words(ctx, &data, &head, &pred, n);
            let got_words: Vec<Vec<u32>> = raw.chunks_exact(per_block).map(|block| block.to_vec()).collect();
            let got: Vec<Vec<Vec<u32>>> = got_words.iter().map(|block| expand_preds(params, block)).collect();
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
            params: self.params,
            subgroups: self.subgroups,
            opts: self.opts,
            ctx_groups: self.ctx_groups,
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
        let per_block = pred_words_per_block(&self.params) as usize;
        let all = self.run_words(ctx, data, head, pred, n_blocks);
        all.chunks_exact(per_block).map(|block| expand_preds(&self.params, block)).collect()
    }

    /// `run`, returning K1's raw pred words (predecessor and fingerprint, see `pred_fp`),
    /// layout `[block][chain][pos]`, sparse long chains compact (`chain_span`).
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
        ctx.read_buffer(pred, 0, pred_words_per_block(&self.params) as usize * n_blocks as usize)
    }

    /// data: packed blocks; head: at least `head_bytes(n_blocks, n_hashes)`, per-dispatch scratch
    /// (each workgroup clears its table in-kernel; nothing is carried between dispatches); pred:
    /// at least `chain_pred_bytes(n_blocks, params)`, layout [block][chain][pos] (Dfast: chain 0
    /// long, 1 short; Single: chain 0 over `hash_width(min_match)`; Opt3: chain 0 h4, 1 h3, then
    /// the sparse long chains, compact: `chain_span`).
    ///
    /// Precondition: `n_blocks <= max_blocks_per_batch_for(&ctx.device.limits(), params)`. That
    /// keeps every buffer within the binding/buffer limits and `n_blocks * pred_words_per_block
    /// <= 2^32` so the shaders' u32 pred indices do not wrap.
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
        let wanted = self.opts.groups.or(self.ctx_groups).unwrap_or(if self.subgroups { DEFAULT_SG_GROUPS } else { u32::MAX });
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
                        size: std::num::NonZeroU64::new(chain_pred_bytes(n_blocks, &self.params)),
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

    #[test]
    fn tags_fit_a_u32() {
        assert_eq!(MAX_TAG as u64 * (1u64 << LOG2_BLOCK) + (1u64 << LOG2_BLOCK) - 1, u32::MAX as u64);
    }
}
