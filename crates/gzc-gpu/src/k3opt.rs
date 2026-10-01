//! K3opt (M5 stage S2): one DP pass of the optimal parse on the GPU (`shaders/k3_opt.wgsl`),
//! byte-identical to `gzc_core::opt::dp_pass_with(.., Engine::Ring)`, then the segmented parse's
//! fix-up (`shaders/k3_fixup.wgsl`).
//!
//! In the pipeline (M5 T5) `compressor::Kernels` runs `OptPasses` as its K3 on the shared
//! `BatchBuffers` (`OptBinds::of_batch`). The harnesses here have their own buffers
//! (`OptBuffers`) and feed K2opt's candidate words from the host (`parses_from_cands`, the
//! counterpart of `compressor::parses_from_best`), so they run on `reference::find_cands` output
//! or on scripted candidates (`opt::cases`). Prices come either from zstd's first-block statistics computed in
//! the kernel's prologue (`PriceSrc::BlockInit`, the oracle's `Seed::BlockInit` pass 0) or from
//! explicit per-block tables (`PriceSrc::Buffer`, e.g. `opt::cases`' tables or a later pass's).
//!
//! Passes (M5 T4): `OptPasses` runs a preset's whole `opt::passes` schedule, one dispatch per
//! pass. Pass 0 is priced by the seed (`PriceSrc::BlockInit`, or `PriceSrc::Prior`: the prior
//! tables plus the cover literals, from the resident candidate words); each cheap pass
//! (optLevel 0, `hist_out`) keeps the candidate words and leaves its output histogram
//! (`opt::Hist` of the fixed-up parse, 377 words per block) in the `prices` buffer from its
//! workgroup epilogue, and the next pass's prologue turns it into price tables
//! (`PriceSrc::Hist`). Only the final pass (at the preset's optLevel) writes its parse and runs
//! the fix-up. `parses_from_passes` / `time_passes` are the host harnesses.
//!
//! Buffers per block: data (BLOCK_SIZE, plus the batch's trailing zero word: `ld32` reads one word
//! past a block), candidate words (`compressor::best_bytes_for`, 8 B per position; after the DP
//! each segment's first words take its raw sequences, as `k3_seg.wgsl` does with `best`), trace
//! (`compressor::trace_bytes`, 8 B per position: K1's `pred` in the pipeline), seqs
//! (`MAX_SEQS_OPT` × 12 B; the DP's series log until the fix-up), counts, the price tables
//! (`PRICE_WORDS` words), and the DP nodes' payload scratch (`scratch_bytes_per_block`, 6336 B at
//! 64 KiB: only the nodes' prices stay in workgroup memory, M5 T3b).
//!
//! Buffer life cycle (the pipeline's order K1 → K2opt → K3opt passes → K5 → K4): the cheap passes
//! overwrite the first `SUM_WORDS` trace words of each segment (so K1's `pred` chains) and the
//! `seqs` series-log words; the final pass overwrites the candidate words (`best`) with its raw
//! sequences. K3opt is therefore the last reader of `pred` and `best`: K2opt reads the chains
//! before the first pass, and K5/K4 read only `data`, `seqs` and `counts`.
//!
//! Precondition: the candidate words come from K2opt (or `reference::find_cands`): every record
//! lies in the block before its position (1 <= offset <= position) with its true common length
//! capped at `SEARCH_CAP`. The kernel does not bounds-check them; only the host harness
//! (`OptBuffers::upload`) validates scripted words.
//!
//! Workgroups: `K3OptConfig::wg` lanes (a power of two, 8..=256; a block's 16 segment lanes may
//! span several workgroups). 16 is the fastest on an RTX 5090 at 64 KiB blocks (M5 T3 and T3b
//! logs in `docs/results/m5-log.md`). Residency: about 5 KB of workgroup memory and ≤ 100
//! registers per wg16 workgroup (T3b), so one wave holds 20 warps/SM (3400 blocks on the 5090);
//! a change that raises either past that splits a 2900-block batch into two waves.
use crate::compressor::{
    BatchBuffers, K3_FIXUP_WGSL, best_bytes_for, counts_bytes, data_bytes, decode_output, seqs_bytes_for, trace_bytes,
};
use crate::context::{GpuContext, pack_blocks, params_wgsl};
use anyhow::{anyhow, ensure};
use gzc_core::config::BLOCK_SIZE;
use gzc_core::opt::{Hist, Prices};
use gzc_core::params::MatchParams;
use gzc_core::reference::CandWords;
use gzc_core::seq::BlockOutput;

const K3_OPT_WGSL: &str = include_str!("shaders/k3_opt.wgsl");

/// Sequences per block of the optimal parse (min match 3): `BLOCK_SIZE / 3 + 1`.
pub use crate::compressor::MAX_SEQS_OPT;

/// Words of one block's price tables in the `prices` buffer: `opt::Prices` as lit[256], ll[36],
/// ml[53], of[32].
pub const PRICE_WORDS: usize = 256 + 36 + 53 + 32;

/// Bytes of the `prices` buffer for `n` blocks (`PRICE_WORDS` words per block, read-write).
pub fn prices_bytes(n: u32) -> u64 {
    n as u64 * PRICE_WORDS as u64 * 4
}

/// Largest batch `parses_from_cands` runs at once.
const BATCH_CAP: usize = 256;

/// Where the DP rings' prices live (the rest of each node is in `OptBuffers::scratch`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingMem {
    /// `var<workgroup>` (needs `ring_bytes(wg)` plus the price tables within the device's
    /// `max_compute_workgroup_storage_size`, which `GpuContext` requests at the adapter's limit).
    Workgroup,
    /// `var<private>`: the fallback for small workgroup storage (the price tables alone must
    /// still fit).
    Private,
}

/// Where the pass's price tables come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PriceSrc {
    /// `opt::Prices::block_init` of each block, computed by the kernel.
    BlockInit,
    /// The per-block tables in `OptBuffers::prices`.
    Buffer,
    /// `opt::seed_prices(.., Seed::Prior)`: the prior LL/ML/OF tables and the block's cover
    /// literals, computed by the kernel from the candidate words (needs `wg % segments == 0`).
    Prior,
    /// `opt::Prices::from_hist` of the per-block `opt::Hist` in `OptBuffers::prices` (the previous
    /// pass's, written by a `hist_out` pass).
    Hist,
}

/// Build options of `K3Opt`.
#[derive(Clone, Copy, Debug)]
pub struct K3OptConfig {
    /// optLevel of the pass: 0 or 2.
    pub level: u8,
    /// Lanes per workgroup (a multiple of the segments per block).
    pub wg: u32,
    /// `None`: `Workgroup` when it fits the device limit, else `Private`.
    pub ring: Option<RingMem>,
    pub prices: PriceSrc,
    /// Build without naga's forced loop bounding (`GpuContext::shader_unbounded_loops`); every
    /// loop has a `Terminates:` note in the kernel.
    pub unbounded: bool,
    /// A cheap pass of `OptPasses`: keep the candidate words, write the pass's `opt::Hist` of
    /// each block to `OptBuffers::prices` (the workgroup epilogue), and skip the fix-up (needs
    /// `wg % segments == 0`).
    pub hist_out: bool,
}

impl Default for K3OptConfig {
    fn default() -> Self {
        Self {
            level: 2,
            wg: 16,
            ring: None,
            prices: PriceSrc::BlockInit,
            unbounded: true,
            hist_out: false,
        }
    }
}

/// Segments per block.
fn n_seg(m: &MatchParams) -> u32 {
    (BLOCK_SIZE >> m.segment_log2) as u32
}

/// Workgroup bytes of `wg` lanes' price rings: 4 B per node, `target_length + 1` nodes per lane.
pub fn ring_bytes(wg: u32, target_length: u32) -> u32 {
    wg * (target_length + 1) * 4
}

/// The ring's `sufficient_len` (`target_length`, at most 4095) of opt params `m`.
fn suff_of(m: &MatchParams) -> u32 {
    m.opt.map_or(32, |o| o.target_length).min(4095)
}

/// Bytes of the DP nodes' payload scratch (`OptBuffers::scratch`) per block: 12 B per node,
/// `target_length + 1` nodes per segment lane.
pub fn scratch_bytes_per_block(m: &MatchParams) -> u64 {
    n_seg(m) as u64 * (suff_of(m) as u64 + 1) * 12
}

/// Workgroup bytes `K3Opt`'s DP entry point needs with its rings in workgroup memory.
pub fn workgroup_bytes(m: &MatchParams, cfg: &K3OptConfig) -> u32 {
    ring_bytes(cfg.wg, suff_of(m)) + table_bytes(m, cfg)
}

/// Workgroup bytes of the DP entry point besides the rings, per pass (M6 A1), exactly what
/// `k3_opt.wgsl` declares for `bpw` blocks per workgroup:
/// - `p_lit`: 128 words of u16 literal-price pairs per block;
/// - `p_tab`: the LL-by-code (36), LL-by-litlen (64), ML (`target_length + 1`) and OF (32)
///   prices as u16 pairs per block;
/// - `hist`: 256 words per block when the pass counts literals (`BlockInit`, `Prior`) or writes
///   its histogram (`hist_out`), else one word;
/// - `hsum`: 5 words per block, one word for `PriceSrc::Buffer`.
///
/// The fix-up's per-segment arrays belong to its own entry point (`main_fixup`, 12 B per
/// segment), which wgpu compiles alone; they are not part of this pipeline.
fn table_bytes(m: &MatchParams, cfg: &K3OptConfig) -> u32 {
    let bpw = (cfg.wg / n_seg(m)).max(1);
    let tab_words = (36 + 64 + (suff_of(m) + 1) + 32).div_ceil(2);
    let hist_used = cfg.hist_out || matches!(cfg.prices, PriceSrc::BlockInit | PriceSrc::Prior);
    let hist = if hist_used { 256 * bpw } else { 1 };
    let hsum = if cfg.prices == PriceSrc::Buffer { 1 } else { 5 * bpw };
    (bpw * (128 + tab_words) + hist + hsum) * 4
}

/// The ring memory `K3Opt::new` uses for `cfg` under a workgroup storage limit of `limit` bytes:
/// `cfg.ring`, or (`None`) `Workgroup` when `workgroup_bytes` fits, else `Private`. Fails cleanly
/// (before any pipeline is created) when a forced `Workgroup` ring does not fit, or when the price
/// tables alone do not.
pub fn ring_for(m: &MatchParams, cfg: &K3OptConfig, limit: u32) -> anyhow::Result<RingMem> {
    let tables = table_bytes(m, cfg);
    ensure!(
        tables <= limit,
        "wg {}: the price tables need {tables} B of workgroup memory > limit {limit}",
        cfg.wg
    );
    let wg_need = workgroup_bytes(m, cfg);
    let ring = cfg.ring.unwrap_or(if wg_need <= limit {
        RingMem::Workgroup
    } else {
        RingMem::Private
    });
    ensure!(
        ring == RingMem::Private || wg_need <= limit,
        "wg {}: the workgroup ring needs {wg_need} B > limit {limit}",
        cfg.wg
    );
    Ok(ring)
}

/// The compiled K3opt pass.
pub struct K3Opt {
    main: wgpu::ComputePipeline,
    fixup: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    pub wg: u32,
    pub ring: RingMem,
    pub prices: PriceSrc,
    pub level: u8,
    pub hist_out: bool,
    /// The opt params the pass was built for (`OptBuffers::new` sizes the scratch by them).
    pub params: MatchParams,
    n_seg: u32,
}

/// K3opt's buffers for `capacity` blocks (the harnesses' own set; the pipeline binds its
/// `BatchBuffers` through `OptBinds::of_batch`).
pub struct OptBuffers {
    pub capacity: u32,
    /// The opt params the buffers were sized for.
    pub params: MatchParams,
    pub data: wgpu::Buffer,
    /// Candidate words `[block][pos][2]`; the pass's raw sequences afterwards.
    pub cands: wgpu::Buffer,
    /// The DP trace `[block][pos][2]`.
    pub trace: wgpu::Buffer,
    pub seqs: wgpu::Buffer,
    pub counts: wgpu::Buffer,
    /// `PRICE_WORDS` words per block: `opt::Prices` (`PriceSrc::Buffer`) or `opt::Hist`
    /// (`PriceSrc::Hist`, `hist_out`), both as lit[256] ll[36] ml[53] of[32].
    pub prices: wgpu::Buffer,
    /// The DP nodes' payload (`scratch_bytes_per_block` per block; K3opt-owned, dead between
    /// passes).
    pub scratch: wgpu::Buffer,
    scratch_per_block: u64,
}

/// The buffers one K3opt dispatch binds (`OptBuffers::binds`, or the pipeline's `BatchBuffers`
/// through `of_batch`), for the first `capacity` blocks.
#[derive(Clone, Copy)]
pub struct OptBinds<'a> {
    pub capacity: u32,
    pub data: &'a wgpu::Buffer,
    pub cands: &'a wgpu::Buffer,
    pub trace: &'a wgpu::Buffer,
    pub seqs: &'a wgpu::Buffer,
    pub counts: &'a wgpu::Buffer,
    pub prices: &'a wgpu::Buffer,
    pub scratch: &'a wgpu::Buffer,
    /// Bytes of `scratch` per block.
    pub scratch_per_block: u64,
}

impl<'a> OptBinds<'a> {
    /// The pipeline's buffers: `best` holds the candidate words, `pred` (dead once K2opt has read
    /// K1's chains) the trace. Errors unless `bufs` was allocated for opt params.
    pub fn of_batch(bufs: &'a BatchBuffers) -> anyhow::Result<Self> {
        let o = bufs.opt.as_ref().ok_or_else(|| anyhow!("BatchBuffers allocated without opt params"))?;
        Ok(Self {
            capacity: bufs.capacity,
            data: &bufs.data,
            cands: &bufs.best,
            trace: &bufs.pred,
            seqs: &bufs.seqs,
            counts: &bufs.counts,
            prices: &o.prices,
            scratch: &o.scratch,
            scratch_per_block: o.scratch_per_block,
        })
    }

    /// Ok when every buffer is large enough for `n` blocks under `m` (checked before each
    /// dispatch: the kernel indexes by block without bounds checks on its own layout).
    fn check(&self, n: u32, m: &MatchParams) -> anyhow::Result<()> {
        ensure!(n >= 1 && n <= self.capacity, "k3opt: {n} blocks, buffers hold {}", self.capacity);
        ensure!(
            self.scratch_per_block >= scratch_bytes_per_block(m),
            "k3opt: scratch sized {} B per block, the params need {} (buffers built for other opt params)",
            self.scratch_per_block,
            scratch_bytes_per_block(m)
        );
        // `ld32` in k3_opt.wgsl reads data[w + 1]: the batch's trailing zero word must exist.
        let need = [
            ("data", self.data, data_bytes(n)),
            ("cands", self.cands, best_bytes_for(n, m)),
            ("trace", self.trace, trace_bytes(n)),
            ("seqs", self.seqs, seqs_bytes_for(n, m)),
            ("counts", self.counts, counts_bytes(n)),
            ("prices", self.prices, prices_bytes(n)),
            ("scratch", self.scratch, n as u64 * scratch_bytes_per_block(m)),
        ];
        for (name, buf, bytes) in need {
            ensure!(buf.size() >= bytes, "k3opt: {name} buffer {} B < {bytes} B for {n} blocks", buf.size());
        }
        Ok(())
    }
}

impl OptBuffers {
    /// Buffers for `capacity` blocks under opt params `m` (the scratch's size depends on the
    /// segments per block and `target_length`). Errors when an allocation fails (out of memory, a
    /// buffer above the device's limits, or a lost device).
    pub fn new(ctx: &GpuContext, m: &MatchParams, capacity: u32) -> anyhow::Result<Self> {
        let scratch_per_block = scratch_bytes_per_block(m);
        let scopes = crate::context::ErrorScopes::push(ctx);
        let bufs = Self {
            capacity,
            params: *m,
            data: ctx.storage_buffer("k3opt.data", data_bytes(capacity), false),
            cands: ctx.storage_buffer("k3opt.cands", best_bytes_for(capacity, m), true),
            trace: ctx.storage_buffer("k3opt.trace", trace_bytes(capacity), false),
            seqs: ctx.storage_buffer("k3opt.seqs", seqs_bytes_for(capacity, m), true),
            counts: ctx.storage_buffer("k3opt.counts", counts_bytes(capacity), true),
            prices: ctx.storage_buffer("k3opt.prices", prices_bytes(capacity), true),
            scratch: ctx.storage_buffer("k3opt.scratch", capacity as u64 * scratch_per_block, false),
            scratch_per_block,
        };
        let bytes = [&bufs.data, &bufs.cands, &bufs.trace, &bufs.seqs, &bufs.counts, &bufs.prices, &bufs.scratch]
            .iter()
            .map(|b| b.size())
            .sum();
        scopes.pop_alloc(&format!("the k3opt buffers ({capacity} blocks)"), bytes)?;
        Ok(bufs)
    }

    /// The buffers as bindings.
    pub fn binds(&self) -> OptBinds<'_> {
        OptBinds {
            capacity: self.capacity,
            data: &self.data,
            cands: &self.cands,
            trace: &self.trace,
            seqs: &self.seqs,
            counts: &self.counts,
            prices: &self.prices,
            scratch: &self.scratch,
            scratch_per_block: self.scratch_per_block,
        }
    }

    /// Uploads blocks, their candidate words and (with `prices`) their price tables.
    pub fn upload(
        &self,
        ctx: &GpuContext,
        blocks: &[&[u8]],
        cands: &[&[CandWords]],
        prices: Option<&[Prices]>,
    ) -> anyhow::Result<()> {
        ensure!(
            blocks.len() == cands.len() && blocks.len() <= self.capacity as usize,
            "bad batch"
        );
        ctx.queue
            .write_buffer(&self.data, 0, bytemuck::cast_slice(&pack_blocks(blocks)));
        for (b, c) in cands.iter().enumerate() {
            ensure!(
                c.len() == BLOCK_SIZE,
                "block {b}: {} candidate words",
                c.len()
            );
            // The kernel trusts the words (no bounds checks on offsets): each record must lie in
            // the block, before its position.
            for (p, w) in c.iter().enumerate() {
                let (a, bb) = gzc_core::reference::unpack_cands(*w);
                for r in [a, bb] {
                    ensure!(
                        r.len == 0 || (r.offset >= 1 && r.offset as usize <= p),
                        "block {b}: bad candidate at {p}: {r:?}"
                    );
                }
            }
            ctx.queue.write_buffer(
                &self.cands,
                b as u64 * best_bytes_for(1, &self.params),
                bytemuck::cast_slice(c),
            );
        }
        if let Some(ps) = prices {
            ensure!(ps.len() == blocks.len(), "one price table per block");
            let mut words = Vec::with_capacity(ps.len() * PRICE_WORDS);
            for p in ps {
                let all = p.lit.iter().chain(&p.ll).chain(&p.ml).chain(&p.of);
                for &x in all {
                    ensure!((0..65536).contains(&x), "price {x} out of range");
                    words.push(x as u32);
                }
            }
            ctx.queue
                .write_buffer(&self.prices, 0, bytemuck::cast_slice(&words));
        }
        Ok(())
    }
}

fn wgsl_array(name: &str, v: &[i32]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
    format!(
        "const {name}: array<i32, {}> = array<i32, {}>({});\n",
        v.len(),
        v.len(),
        items.join(", ")
    )
}

fn wgsl_array_u32(name: &str, v: &[u32]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x}u")).collect();
    format!(
        "const {name}: array<u32, {}> = array<u32, {}>({});\n",
        v.len(),
        v.len(),
        items.join(", ")
    )
}

/// The `codes` and seed constants the kernel needs: block-init and prior-seed LL/ML/OF prices,
/// LL/ML extra bits, and the ML codes of match lengths 3..131.
fn tables_wgsl() -> String {
    use gzc_core::codes::{LL_BITS, ML_BITS, OPT_PRIOR_LL, OPT_PRIOR_ML, OPT_PRIOR_OF, ml_code};
    let bi = Prices::block_init(&vec![0u8; BLOCK_SIZE]);
    let pr = Prices::from_hist(&Hist {
        lit: [0; 256],
        ll: OPT_PRIOR_LL,
        ml: OPT_PRIOR_ML,
        of: OPT_PRIOR_OF,
    });
    let u = |t: &[u8]| t.iter().map(|&x| x as u32).collect::<Vec<u32>>();
    let mlc: Vec<u32> = (0..128u32).map(|b| ml_code(b + 3) as u32).collect();
    [
        wgsl_array("BI_LL", &bi.ll),
        wgsl_array("BI_ML", &bi.ml),
        wgsl_array("BI_OF", &bi.of),
        wgsl_array("PR_LL", &pr.ll),
        wgsl_array("PR_ML", &pr.ml),
        wgsl_array("PR_OF", &pr.of),
        wgsl_array_u32("LL_BITS", &u(&LL_BITS)),
        wgsl_array_u32("ML_BITS", &u(&ML_BITS)),
        wgsl_array_u32("ML_CODE", &mlc),
    ]
    .concat()
}

impl K3Opt {
    /// Builds the pass for opt params `m` (segment size, `target_length`) under `cfg`.
    pub fn new(ctx: &GpuContext, m: &MatchParams, cfg: K3OptConfig) -> anyhow::Result<Self> {
        m.validate()
            .map_err(|e| anyhow!("invalid match params {m:?}: {e}"))?;
        let o = m.opt.ok_or_else(|| anyhow!("K3opt needs opt params"))?;
        ensure!(cfg.level == 0 || cfg.level == 2, "optLevel {}", cfg.level);
        ensure!(BLOCK_SIZE <= 1 << 16, "K3opt: blocks of at most 64 KiB");
        let n_seg = n_seg(m);
        ensure!(
            cfg.wg.is_power_of_two() && (8..=256).contains(&cfg.wg),
            "wg {}: a power of two in 8..=256",
            cfg.wg
        );
        let suff = o.target_length.min(4095);
        ensure!(
            !(cfg.hist_out || cfg.prices == PriceSrc::Prior) || cfg.wg.is_multiple_of(n_seg),
            "wg {}: hist_out and PriceSrc::Prior need a block's {n_seg} segments in one workgroup",
            cfg.wg
        );
        let ring = ring_for(m, &cfg, ctx.device.limits().max_compute_workgroup_storage_size)?;
        let ring_decl = match ring {
            RingMem::Workgroup => {
                "var<workgroup> ring_p: array<i32, RING_N * WG>;\n\
                 fn rix(s: u32) -> u32 { return s * WG + lane_id(); }\n"
            }
            RingMem::Private => {
                "var<private> ring_p: array<i32, RING_N>;\n\
                 fn rix(s: u32) -> u32 { return s; }\n"
            }
        };
        let body = format!(
            "{}const MAX_SEQS: u32 = {MAX_SEQS_OPT}u;\nconst SEG_LOG2: u32 = {}u;\nconst WG: u32 = {}u;\nconst SUFF: u32 = {suff}u;\n\
             const LEVEL: u32 = {}u;\nconst PRICE_MODE: u32 = {}u;\nconst HIST_OUT: bool = {};\n{}{ring_decl}{K3_OPT_WGSL}\n{K3_FIXUP_WGSL}",
            params_wgsl(m),
            m.segment_log2,
            cfg.wg,
            cfg.level,
            match cfg.prices {
                PriceSrc::BlockInit => 0,
                PriceSrc::Buffer => 1,
                PriceSrc::Prior => 2,
                PriceSrc::Hist => 3,
            },
            cfg.hist_out,
            tables_wgsl(),
        );
        let layout = crate::compressor::storage_layout(
            ctx,
            "k3opt",
            &[true, false, false, false, false, false, false],
        );
        let module = if cfg.unbounded {
            ctx.shader_unbounded_loops("k3_opt", &body)
        } else {
            ctx.shader("k3_opt", &body)
        };
        let main =
            crate::compressor::pipeline_from_module(ctx, "k3_opt", &layout, &module, "main_opt");
        let fixup = crate::compressor::pipeline_from_module(
            ctx,
            "k3_opt_fixup",
            &layout,
            &module,
            "main_fixup",
        );
        Ok(Self {
            main,
            fixup,
            layout,
            wg: cfg.wg,
            ring,
            prices: cfg.prices,
            level: cfg.level,
            hist_out: cfg.hist_out,
            params: *m,
            n_seg,
        })
    }

    /// Records the DP pass then (unless `hist_out`) the fix-up on the first `n` blocks of `bufs`,
    /// each in its own compute pass (timestamps: `queries` 0/1 around the DP, 2/3 around the
    /// fix-up). Errors when the buffers are too small for `n` blocks under the pass's params.
    pub fn record(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &OptBuffers,
        n: u32,
        queries: Option<&wgpu::QuerySet>,
    ) -> anyhow::Result<()> {
        self.record_at(ctx, enc, &bufs.binds(), n, queries.map(|q| (q, 0)))
    }

    /// `record` on `bufs` with the timestamps at `queries.1 ..` (DP, then the fix-up).
    pub fn record_at(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &OptBinds,
        n: u32,
        queries: Option<(&wgpu::QuerySet, u32)>,
    ) -> anyhow::Result<()> {
        let ts = |k: u32| {
            queries.map(|(query_set, at)| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(at + 2 * k),
                end_of_pass_write_index: Some(at + 2 * k + 1),
            })
        };
        self.record_with(ctx, enc, bufs, n, ts(0), ts(1))
    }

    /// `record` on `bufs` with the given timestamp writes for the DP's and the fix-up's compute
    /// passes (the fix-up's are unused after a `hist_out` pass).
    pub fn record_with(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &OptBinds,
        n: u32,
        ts_dp: Option<wgpu::ComputePassTimestampWrites>,
        ts_fixup: Option<wgpu::ComputePassTimestampWrites>,
    ) -> anyhow::Result<()> {
        bufs.check(n, &self.params)?;
        let counts = wgpu::BufferBinding {
            buffer: bufs.counts,
            offset: 0,
            size: wgpu::BufferSize::new(counts_bytes(n)),
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k3opt"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs.data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bufs.cands.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs.seqs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(counts),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bufs.trace.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bufs.prices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bufs.scratch.as_entire_binding(),
                },
            ],
        });
        let groups = (n * self.n_seg).div_ceil(self.wg);
        let max_groups = ctx.device.limits().max_compute_workgroups_per_dimension;
        ensure!(groups <= max_groups && n <= max_groups, "k3opt: {groups} workgroups > {max_groups}");
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("k3opt"),
                timestamp_writes: ts_dp,
            });
            pass.set_bind_group(0, &bind, &[]);
            pass.set_pipeline(&self.main);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        if self.hist_out {
            return Ok(());
        }
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k3opt_fixup"),
            timestamp_writes: ts_fixup,
        });
        pass.set_bind_group(0, &bind, &[]);
        pass.set_pipeline(&self.fixup);
        pass.dispatch_workgroups(n, 1, 1);
        Ok(())
    }
}

/// Reads the first `n` blocks' parses out of `bufs` (literals gathered from `blocks`).
pub fn read_parses(
    ctx: &GpuContext,
    bufs: &OptBuffers,
    blocks: &[&[u8]],
) -> anyhow::Result<Vec<BlockOutput>> {
    let n = blocks.len();
    let counts: Vec<u32> = ctx.read_buffer(&bufs.counts, 0, 2 * n);
    let words: Vec<u32> = ctx.read_buffer(&bufs.seqs, 0, n * MAX_SEQS_OPT as usize * 3);
    let mut out = Vec::with_capacity(n);
    for (b, block) in blocks.iter().enumerate() {
        let (n_seq, n_lit) = (counts[2 * b], counts[2 * b + 1]);
        ensure!(
            n_seq <= MAX_SEQS_OPT && n_lit as usize <= BLOCK_SIZE,
            "block {b}: bad counts ({n_seq}, {n_lit})"
        );
        let at = b * MAX_SEQS_OPT as usize * 3;
        let parse = decode_output(block, &words[at..at + 3 * n_seq as usize], n_seq);
        ensure!(
            parse.literals.len() == n_lit as usize,
            "block {b}: K3opt counted {n_lit} literals, the sequences leave {}",
            parse.literals.len()
        );
        out.push(parse);
    }
    Ok(out)
}

/// Runs the pass on `blocks` with their candidate words (and, for `PriceSrc::Buffer`, their
/// price tables), batch by batch, and returns the parses.
pub fn parses_from_cands(
    ctx: &GpuContext,
    k: &K3Opt,
    blocks: &[&[u8]],
    cands: &[&[CandWords]],
    prices: Option<&[Prices]>,
) -> anyhow::Result<Vec<BlockOutput>> {
    ensure!(blocks.len() == cands.len(), "one candidate table per block");
    ensure!(
        prices.is_some() == (k.prices == PriceSrc::Buffer),
        "price tables iff PriceSrc::Buffer"
    );
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    crate::compressor::with_error_scopes(ctx, || {
        let bufs = OptBuffers::new(ctx, &k.params, blocks.len().min(BATCH_CAP) as u32)?;
        let mut out = Vec::with_capacity(blocks.len());
        for (i, chunk) in blocks.chunks(BATCH_CAP).enumerate() {
            let at = i * BATCH_CAP;
            let c = &cands[at..at + chunk.len()];
            bufs.upload(ctx, chunk, c, prices.map(|p| &p[at..at + chunk.len()]))?;
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("k3opt"),
                });
            if ctx.poisoning() {
                // Everything but the uploaded blocks (+ trailing word), candidates and prices.
                let n = chunk.len() as u32;
                ctx.poison_workgroup_memory(&mut enc);
                ctx.poison_from(&mut enc, &bufs.data, data_bytes(n));
                ctx.poison_from(&mut enc, &bufs.cands, best_bytes_for(n, &k.params));
                for b in [&bufs.trace, &bufs.seqs, &bufs.counts, &bufs.scratch] {
                    ctx.poison_from(&mut enc, b, 0);
                }
                let from = if k.prices == PriceSrc::Buffer { prices_bytes(n) } else { 0 };
                ctx.poison_from(&mut enc, &bufs.prices, from);
            }
            k.record(ctx, &mut enc, &bufs, chunk.len() as u32, None)?;
            ctx.queue.submit([enc.finish()]);
            out.extend(read_parses(ctx, &bufs, chunk)?);
        }
        Ok(out)
    })
}

/// GPU time of the DP pass and of the fix-up (ms, median of `reps` runs after one warm-up) on
/// the first `n` blocks of `bufs`. The pass overwrites part of the candidate words with its
/// sequences, so `reupload` runs before every run (outside the timed passes).
pub fn time_pass(
    ctx: &GpuContext,
    k: &K3Opt,
    bufs: &OptBuffers,
    n: u32,
    reps: usize,
    mut reupload: impl FnMut(),
) -> anyhow::Result<(f64, f64)> {
    ensure!(ctx.timestamps, "timestamps unavailable");
    let qs = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("k3opt.ts"),
        ty: wgpu::QueryType::Timestamp,
        count: 4,
    });
    let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("k3opt.ts.resolve"),
        size: 32,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let period = ctx.queue.get_timestamp_period() as f64;
    let mut main = Vec::new();
    let mut fix = Vec::new();
    for r in 0..=reps {
        reupload();
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k3opt.time"),
            });
        k.record(ctx, &mut enc, bufs, n, Some(&qs))?;
        enc.resolve_query_set(&qs, 0..4, &resolve, 0);
        ctx.queue.submit([enc.finish()]);
        let t: Vec<u64> = ctx.read_buffer(&resolve, 0, 4);
        if r > 0 {
            main.push((t[1] - t[0]) as f64 * period / 1e6);
            fix.push(if k.hist_out { 0.0 } else { (t[3] - t[2]) as f64 * period / 1e6 });
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    Ok((med(&mut main), med(&mut fix)))
}

/// The DP passes of an opt preset (`opt::passes`, M5 T4): `o.passes` cheap passes at optLevel 0
/// (`hist_out`: each writes its `opt::Hist` for the next one's prologue and keeps the candidate
/// words), then the final pass at `o.level` with the fix-up. Pass 0 is priced by the seed
/// (`Seed::BlockInit` → `PriceSrc::BlockInit`, `Seed::Prior` → `PriceSrc::Prior`), every later
/// pass by the previous pass's histogram (`PriceSrc::Hist`). One dispatch per pass (plus the
/// fix-up): the candidates, the data and the histograms stay resident in `OptBuffers`.
pub struct OptPasses {
    /// The kernel of each pass, in order (a kernel shared by several passes appears once per pass).
    kernels: Vec<std::sync::Arc<K3Opt>>,
}

impl OptPasses {
    /// Builds the passes of opt params `m`. `base` gives the workgroup size, ring memory and loop
    /// bounding; its `level`, `prices` and `hist_out` are set per pass. `wg` must be a multiple
    /// of the segments per block when there is a cheap pass or the seed is `Prior`.
    pub fn new(ctx: &GpuContext, m: &MatchParams, base: K3OptConfig) -> anyhow::Result<Self> {
        use gzc_core::params::Seed;
        use std::sync::Arc;
        let o = m.opt.ok_or_else(|| anyhow!("OptPasses needs opt params"))?;
        let seed = match o.seed {
            Seed::BlockInit => PriceSrc::BlockInit,
            Seed::Prior => PriceSrc::Prior,
        };
        let cfg = |level: u8, prices: PriceSrc, hist_out: bool| K3OptConfig {
            level,
            prices,
            hist_out,
            ..base
        };
        let mut kernels = Vec::with_capacity(o.passes as usize + 1);
        if o.passes > 0 {
            kernels.push(Arc::new(K3Opt::new(ctx, m, cfg(0, seed, true))?));
            if o.passes > 1 {
                let next = Arc::new(K3Opt::new(ctx, m, cfg(0, PriceSrc::Hist, true))?);
                for _ in 1..o.passes {
                    kernels.push(next.clone());
                }
            }
        }
        let fin_src = if o.passes > 0 { PriceSrc::Hist } else { seed };
        kernels.push(Arc::new(K3Opt::new(ctx, m, cfg(o.level, fin_src, false))?));
        Ok(Self { kernels })
    }

    /// The passes' kernels, in order.
    pub fn kernels(&self) -> impl Iterator<Item = &K3Opt> {
        self.kernels.iter().map(|k| &**k)
    }

    /// The opt params the passes were built for.
    pub fn params(&self) -> &MatchParams {
        &self.kernels[0].params
    }

    /// Number of DP passes (cheap passes + 1).
    pub fn n_passes(&self) -> usize {
        self.kernels.len()
    }

    /// Records every pass on the first `n` blocks of `bufs` (uploaded blocks and candidate
    /// words). Timestamps: pass i's DP at `2i`/`2i + 1`, the final fix-up at `2 * n_passes()` and
    /// `2 * n_passes() + 1`.
    pub fn record(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &OptBuffers,
        n: u32,
        queries: Option<&wgpu::QuerySet>,
    ) -> anyhow::Result<()> {
        let binds = bufs.binds();
        for (i, k) in self.kernels().enumerate() {
            // The final pass's fix-up lands at 2 * i + 2 = 2 * n_passes().
            k.record_at(ctx, enc, &binds, n, queries.map(|q| (q, 2 * i as u32)))?;
        }
        Ok(())
    }

    /// Records every pass on the first `n` blocks of `bufs`, with one timestamp pair spanning
    /// them all (the pipeline's K3 entry): `span`'s beginning write on the first pass's DP and
    /// its end write on the final fix-up.
    pub fn record_span(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &OptBinds,
        n: u32,
        span: Option<wgpu::ComputePassTimestampWrites>,
    ) -> anyhow::Result<()> {
        let last = self.kernels.len() - 1;
        for (i, k) in self.kernels().enumerate() {
            let begin = span.as_ref().filter(|_| i == 0).map(|s| wgpu::ComputePassTimestampWrites {
                query_set: s.query_set,
                beginning_of_pass_write_index: s.beginning_of_pass_write_index,
                end_of_pass_write_index: None,
            });
            let end = span.as_ref().filter(|_| i == last).map(|s| wgpu::ComputePassTimestampWrites {
                query_set: s.query_set,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: s.end_of_pass_write_index,
            });
            // Only the final pass (last) has a fix-up.
            k.record_with(ctx, enc, bufs, n, begin, end)?;
        }
        Ok(())
    }
}

/// Reads the first `n` blocks' histograms that a `hist_out` pass left in `bufs.prices`.
pub fn read_hists(ctx: &GpuContext, bufs: &OptBuffers, n: usize) -> Vec<Hist> {
    let w: Vec<u32> = ctx.read_buffer(&bufs.prices, 0, n * PRICE_WORDS);
    w.chunks(PRICE_WORDS)
        .map(|c| Hist {
            lit: c[..256].try_into().unwrap(),
            ll: c[256..292].try_into().unwrap(),
            ml: c[292..345].try_into().unwrap(),
            of: c[345..].try_into().unwrap(),
        })
        .collect()
}

/// Runs every pass of `p` on `blocks` with their candidate words, batch by batch, and returns the
/// final parses and, with `hists`, each cheap pass's histograms (`[pass][block]`, the GPU's
/// counterpart of `opt::Hist::of_output(&opt::passes(..)[pass].out)`; one submission per pass).
pub fn parses_from_passes(
    ctx: &GpuContext,
    p: &OptPasses,
    blocks: &[&[u8]],
    cands: &[&[CandWords]],
    hists: bool,
) -> anyhow::Result<(Vec<BlockOutput>, Vec<Vec<Hist>>)> {
    ensure!(blocks.len() == cands.len(), "one candidate table per block");
    let mut hs: Vec<Vec<Hist>> = vec![Vec::new(); if hists { p.n_passes() - 1 } else { 0 }];
    if blocks.is_empty() {
        return Ok((Vec::new(), hs));
    }
    crate::compressor::with_error_scopes(ctx, || {
        let bufs = OptBuffers::new(ctx, p.params(), blocks.len().min(BATCH_CAP) as u32)?;
        let mut out = Vec::with_capacity(blocks.len());
        for (i, chunk) in blocks.chunks(BATCH_CAP).enumerate() {
            let at = i * BATCH_CAP;
            bufs.upload(ctx, chunk, &cands[at..at + chunk.len()], None)?;
            let n = chunk.len() as u32;
            if hists {
                for (j, k) in p.kernels().enumerate() {
                    let mut enc = ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("k3opt.pass"),
                        });
                    k.record(ctx, &mut enc, &bufs, n, None)?;
                    ctx.queue.submit([enc.finish()]);
                    if k.hist_out {
                        hs[j].extend(read_hists(ctx, &bufs, chunk.len()));
                    }
                }
            } else {
                let mut enc = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("k3opt.passes"),
                    });
                p.record(ctx, &mut enc, &bufs, n, None)?;
                ctx.queue.submit([enc.finish()]);
            }
            out.extend(read_parses(ctx, &bufs, chunk)?);
        }
        Ok((out, hs))
    })
}

/// GPU time (ms, median of `reps` runs after one warm-up) of each DP pass of `p` and of the final
/// fix-up, on the first `n` blocks of `bufs`: `n_passes() + 1` values, then the span from the
/// first pass's start to the fix-up's end (dispatch gaps included). The final pass overwrites part
/// of the candidate words, so `reupload` runs before every run (outside the timed passes).
pub fn time_passes(
    ctx: &GpuContext,
    p: &OptPasses,
    bufs: &OptBuffers,
    n: u32,
    reps: usize,
    mut reupload: impl FnMut(),
) -> anyhow::Result<Vec<f64>> {
    ensure!(ctx.timestamps, "timestamps unavailable");
    let q = 2 * (p.n_passes() + 1);
    let qs = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("k3opt.passes.ts"),
        ty: wgpu::QueryType::Timestamp,
        count: q as u32,
    });
    let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("k3opt.passes.ts.resolve"),
        size: 8 * q as u64,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let period = ctx.queue.get_timestamp_period() as f64;
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); p.n_passes() + 2];
    for r in 0..=reps {
        reupload();
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k3opt.passes.time"),
            });
        p.record(ctx, &mut enc, bufs, n, Some(&qs))?;
        enc.resolve_query_set(&qs, 0..q as u32, &resolve, 0);
        ctx.queue.submit([enc.finish()]);
        let t: Vec<u64> = ctx.read_buffer(&resolve, 0, q);
        if r > 0 {
            for (i, v) in times.iter_mut().take(p.n_passes() + 1).enumerate() {
                v.push(t[2 * i + 1].saturating_sub(t[2 * i]) as f64 * period / 1e6);
            }
            times[p.n_passes() + 1].push(t[q - 1].saturating_sub(t[0]) as f64 * period / 1e6);
        }
    }
    Ok(times
        .into_iter()
        .map(|mut v| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gzc_core::params::{OPT16, OptParams};

    /// `OptBinds::check`'s error paths: bad `n`, a scratch buffer sized for other opt params,
    /// and a `data` buffer missing the batch's trailing 4-byte word. Each errors rather than
    /// panics (the kernel itself does no bounds checking on these buffers).
    #[test]
    fn check_rejects_bad_buffers() {
        let _gpu = crate::test_support::gpu_test_slot();
        // opt16 only implements at blocks of at most 64 KiB.
        if BLOCK_SIZE > 1 << 16 {
            return;
        }
        // Without poisoning: its padding would make the short buffers below long enough.
        let opts = crate::context::GpuOptions { poison: false, ..crate::context::GpuOptions::default() };
        if crate::context::env_on("GZC_POISON") {
            eprintln!("skipped: poisoning pads every buffer");
            return;
        }
        let ctx = GpuContext::with_gpu_options(opts).expect("GPU required for gzc-gpu tests");
        let n = 4u32;
        let bufs = OptBuffers::new(&ctx, &OPT16, n).unwrap();
        let binds = bufs.binds();

        // (a) n = 0 and n > capacity.
        assert!(binds.check(0, &OPT16).is_err(), "n = 0 must be rejected");
        assert!(binds.check(n + 1, &OPT16).is_err(), "n > capacity must be rejected");
        // Sanity: the buffers themselves are fine for 1..=n blocks.
        assert!(binds.check(n, &OPT16).is_ok());

        // (b) a scratch buffer sized for a different target_length (a smaller one, so its
        // scratch is too small for OPT16's).
        let o = OPT16.opt.unwrap();
        let smaller = MatchParams { opt: Some(OptParams { target_length: 8, ..o }), ..OPT16 };
        assert!(scratch_bytes_per_block(&smaller) < scratch_bytes_per_block(&OPT16));
        let small_bufs = OptBuffers::new(&ctx, &smaller, n).unwrap();
        let e = small_bufs.binds().check(n, &OPT16).unwrap_err();
        assert!(e.to_string().contains("scratch"), "{e}");

        // (c) a `data` buffer missing its trailing 4-byte word (`ld32` reads one word past the
        // last block).
        let short_data = ctx.storage_buffer("test.short_data", data_bytes(n) - 4, false);
        let mut short = binds;
        short.data = &short_data;
        let e = short.check(n, &OPT16).unwrap_err();
        assert!(e.to_string().contains("data"), "{e}");
    }
}
