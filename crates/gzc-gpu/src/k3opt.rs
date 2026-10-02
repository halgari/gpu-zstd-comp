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
//! M6 (`opt16p1`, B3): `inner_gap` (gap3), `relax_lengths` (top-N pruning) and `prior` (the S3
//! prior tables) are compile-time options of the pass kernel (the M5 presets build the same
//! kernels as before); with `drop_max_len > 0` `OptPasses` runs the drop pass (`K3Drop`,
//! `shaders/k3_drop.wgsl`, `opt::drop_pass`) after the final pass's fix-up.
//!
//! Buffers per block: data (BLOCK_SIZE, plus the batch's trailing zero word: `ld32` reads one word
//! past a block), candidate words (`sizing::best_bytes_for`, 8 B per position; after the DP
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
//! capped at `SEARCH_CAP`, and every dead mark (M6 A3) is on a dead position. The kernel does
//! not bounds-check them; only the host harness (`OptBuffers::upload`) validates scripted words.
//!
//! Workgroups: one per block, its 16 lanes the block's 16 segments (the fastest size measured on
//! an RTX 5090 at 64 KiB blocks, M5 T3 and T3b logs in `docs/results/m5-log.md`). The passes are
//! persistent (M6 A4): each workgroup takes blocks in the batch's heavy-first order
//! (`k3_sched.wgsl`) until none is left. Residency (M6 A1, A4): a pass kernel needs at most
//! 4068 B of workgroup memory (`workgroup_bytes`; the final pass 3048 B) and 80 (final) / 85..86
//! (cheap) registers (`vkstats`): 23 cheap-pass workgroups per SM on an RTX 5090 (3910 blocks
//! resident), and a batch beyond that has no second-wave cliff (the loop takes the remaining
//! blocks as slots free; measured +3 % on the cheap passes at 4095 blocks). Check
//! `vkstats` on every pass kernel after a change (`GpuOptions::dump_wgsl` writes the composed modules).
use crate::compressor::{
    BatchBuffers, K3_FIXUP_WGSL, decode_output,
};
use crate::context::{GpuContext, pack_blocks, params_wgsl};
use crate::sizing::{
    best_bytes_for, counts_bytes, data_bytes, prices_bytes, sched_bytes, seqs_bytes_for, trace_bytes,
};
use anyhow::{anyhow, ensure};
use gzc_core::config::BLOCK_SIZE;
use gzc_core::opt::{Hist, Prices};
use gzc_core::params::{MatchParams, PriorTables};
use gzc_core::reference::CandWords;
use gzc_core::seq::BlockOutput;

const K3_OPT_WGSL: &str = include_str!("shaders/k3_opt.wgsl");

// K3opt and K3Drop keep offsets and literal counts in 16-bit fields (k3_opt.wgsl's const_asserts).
const _: () = assert!(BLOCK_SIZE <= 1 << 16, "K3opt/K3Drop need blocks of at most 64 KiB");
const K3_SCHED_WGSL: &str = include_str!("shaders/k3_sched.wgsl");

/// Header words of the `sched` buffer (`k3_sched.wgsl`): the persistent passes' block counter at
/// word 0, then 3 unused words.
pub const SCHED_HDR: u32 = 4;

/// `k3_sched.wgsl`'s weight samples runs of `WEIGHT_RUN` positions every `WEIGHT_STRIDE` positions
/// (M6 A4): one 64-byte burst of candidate words in 33, 1986 positions per block. Over 2900 corpus
/// blocks the sampled count's Spearman correlation with the full count is 0.996. A full scan cost
/// about 1.4 % of opt16's K3 time, and every 33rd position alone (0.998) still 0.3 %: each
/// position is its own memory burst. (Single positions at a power-of-two stride alias with
/// periodic data: 0.79..0.84; the runs cover every phase mod 8, and 264 = 8 * 33 every other.)
pub const WEIGHT_RUN: u32 = 8;
pub const WEIGHT_STRIDE: u32 = 264;

/// Sequences per block of the optimal parse (min match 3): `BLOCK_SIZE / 3 + 1`.
pub use crate::compressor::MAX_SEQS_OPT;

/// Words of one block's price tables in the `prices` buffer: `opt::Prices` as lit[256], ll[36],
/// ml[53], of[32].
pub const PRICE_WORDS: usize = 256 + 36 + 53 + 32;

/// Largest batch `parses_from_cands` runs at once.
const BATCH_CAP: usize = 256;

/// Where the pass's price tables come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PriceSrc {
    /// `opt::Prices::block_init` of each block, computed by the kernel.
    BlockInit,
    /// The per-block tables in `OptBuffers::prices`.
    Buffer,
    /// `opt::seed_prices(.., Seed::Prior)`: the prior LL/ML/OF tables and the block's cover
    /// literals, computed by the kernel from the candidate words.
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
    pub prices: PriceSrc,
    /// Build without naga's forced loop bounding (`GpuContext::shader_unbounded_loops`); every
    /// loop has a `Terminates:` note in the kernel.
    pub unbounded: bool,
    /// A cheap pass of `OptPasses`: keep the candidate words, write the pass's `opt::Hist` of
    /// each block to `OptBuffers::prices` (the workgroup epilogue), and skip the fix-up.
    pub hist_out: bool,
    /// Test only: at most this many workgroups in a dispatch (default: one per block), so that
    /// each workgroup of the persistent loop runs several blocks even in a small batch, where
    /// every workgroup of a one-per-block grid would start at once and take one block.
    #[doc(hidden)]
    pub grid: Option<u32>,
}

impl Default for K3OptConfig {
    fn default() -> Self {
        Self {
            level: 2,
            prices: PriceSrc::BlockInit,
            unbounded: true,
            hist_out: false,
            grid: None,
        }
    }
}

/// Segments per block: the lanes of a K3opt workgroup.
fn n_seg(m: &MatchParams) -> u32 {
    (BLOCK_SIZE >> m.segment_log2) as u32
}

/// Workgroup bytes of a block's price rings under opt params `m`: 4 B per node, `target_length +
/// 1` nodes per segment lane.
pub fn ring_bytes(m: &MatchParams) -> u32 {
    n_seg(m) * (suff_of(m) + 1) * 4
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

/// Workgroup bytes `K3Opt`'s DP entry point needs: the rings and the block's tables.
pub fn workgroup_bytes(m: &MatchParams, cfg: &K3OptConfig) -> u32 {
    ring_bytes(m) + table_bytes(m, cfg)
}

/// Workgroup bytes of the DP entry point besides the rings, per pass (M6 A1), exactly what
/// `k3_opt.wgsl` declares for its one block per workgroup:
/// - `p_lit`: 128 words of u16 literal-price pairs;
/// - `p_tab`: the LL-by-code (36), LL-by-litlen (64) and OF (32) prices as u16 pairs, 66 words;
/// - `p_ml`: the i32 ML prices, `target_length + 1` words;
/// - `hist`: 256 words when the pass counts literals (`BlockInit`, `Prior`) or writes its
///   histogram (`hist_out`), else one word;
/// - `hsum`: 5 words;
/// - `next_item`: one word (the persistent loop's item).
///
/// The fix-up's per-segment arrays belong to its own entry point (`main_fixup`, 12 B per
/// segment), which wgpu compiles alone; they are not part of this pipeline.
fn table_bytes(m: &MatchParams, cfg: &K3OptConfig) -> u32 {
    let hist_used = cfg.hist_out || matches!(cfg.prices, PriceSrc::BlockInit | PriceSrc::Prior);
    let hist = if hist_used { 256 } else { 1 };
    (128 + 66 + suff_of(m) + 1 + hist + 5 + 1) * 4
}

/// Ok when the DP rings and tables of `cfg` (`workgroup_bytes`) fit a workgroup storage limit of
/// `limit` bytes; `K3Opt::new` fails with this error before any pipeline is created. At most
/// 4068 B at `target_length` 32, well under WebGPU's 16 KiB minimum limit, so it fits every
/// conforming device.
pub fn ring_for(m: &MatchParams, cfg: &K3OptConfig, limit: u32) -> anyhow::Result<()> {
    let need = workgroup_bytes(m, cfg);
    ensure!(need <= limit, "the K3opt rings and price tables need {need} B of workgroup memory > limit {limit}");
    Ok(())
}

/// The block schedule's kernels (`k3_sched.wgsl`, M6 A4): each block's weight, then the
/// heavy-first order.
struct Sched {
    weight: wgpu::ComputePipeline,
    rank: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// The compiled K3opt pass.
pub struct K3Opt {
    main: wgpu::ComputePipeline,
    fixup: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// The block order's kernels.
    sched: Sched,
    grid: Option<u32>,
    pub prices: PriceSrc,
    pub level: u8,
    pub hist_out: bool,
    /// The opt params the pass was built for (`OptBuffers::new` sizes the scratch by them).
    pub params: MatchParams,
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
    /// The block schedule (`sched_bytes`, M6 A4).
    pub sched: wgpu::Buffer,
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
    /// The block schedule (`sched_bytes`).
    pub sched: &'a wgpu::Buffer,
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
            sched: &o.sched,
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
            ("sched", self.sched, sched_bytes(n)),
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
            sched: ctx.storage_buffer("k3opt.sched", sched_bytes(capacity), true),
        };
        let bytes = [&bufs.data, &bufs.cands, &bufs.trace, &bufs.seqs, &bufs.counts, &bufs.prices, &bufs.scratch, &bufs.sched]
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
            sched: &self.sched,
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
        check_dead_marks_par(blocks, cands)?;
        for (b, c) in cands.iter().enumerate() {
            ensure!(
                c.len() == BLOCK_SIZE,
                "block {b}: {} candidate words",
                c.len()
            );
            // The kernel trusts the words (no bounds checks on offsets): each record must lie in
            // the block, before its position.
            for (p, w) in c.iter().enumerate() {
                // w1's high half holds DEAD_BIT and nothing else (the kernel reads offB as
                // `w1 & 0xFFFF` and the flag as `w1 & DEAD_BIT`; another bit there would be a
                // format change neither checks for).
                ensure!(
                    w[1] >> 16 & !(gzc_core::reference::DEAD_BIT >> 16) == 0,
                    "block {b}: candidate word 1 at {p} has bits besides DEAD_BIT in its high half: {:#x}",
                    w[1]
                );
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

/// `check_dead_marks` on every block, on up to 16 threads (a timing harness re-uploads between
/// runs, and a long upload lets the GPU clock down).
fn check_dead_marks_par(blocks: &[&[u8]], cands: &[&[CandWords]]) -> anyhow::Result<()> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| -> anyhow::Result<()> {
                    let mut seen = Vec::new();
                    loop {
                        let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let (Some(bl), Some(c)) = (blocks.get(b), cands.get(b)) else { return Ok(()) };
                        ensure!(c.len() == BLOCK_SIZE, "block {b}: {} candidate words", c.len());
                        check_dead_marks(bl, c, &mut seen).map_err(|e| anyhow!("block {b}: {e}"))?;
                    }
                })
            })
            .collect();
        workers.into_iter().try_for_each(|w| w.join().expect("dead-mark check panicked"))
    })
}

/// The kernel also trusts the dead marks (`reference::find_cands`, M6 A3): it skips the search at
/// every marked position. Checks that each marked position `p` is really dead: words `[0,
/// DEAD_BIT]` (no record), `p < PARSE_END`, and no earlier position with its first 3 bytes.
/// `seen` is the caller's scratch (empty, or all zero as this leaves it).
fn check_dead_marks(block: &[u8], c: &[CandWords], seen: &mut Vec<u64>) -> anyhow::Result<()> {
    use gzc_core::config::PARSE_END;
    use gzc_core::reference::{DEAD_BIT, is_dead};
    if !c.iter().any(|w| is_dead(*w)) {
        return Ok(());
    }
    ensure!(block.len() >= PARSE_END + 3, "dead marks on a short block");
    // Seen 3-byte prefixes, a 2^24-bit set (cleared again below, one word per position).
    seen.resize(1 << 18, 0);
    let key = |p: usize| block[p] as usize | (block[p + 1] as usize) << 8 | (block[p + 2] as usize) << 16;
    let mut res = Ok(());
    for (p, &w) in c.iter().enumerate() {
        let dead = is_dead(w);
        if dead && (p >= PARSE_END || w != [0, DEAD_BIT]) {
            res = Err(anyhow!("bad dead word at {p}: {w:?}"));
            break;
        }
        if p < PARSE_END {
            let k = key(p);
            if dead && seen[k >> 6] & 1 << (k & 63) != 0 {
                res = Err(anyhow!("dead position {p} has an earlier 3-byte match"));
                break;
            }
            seen[k >> 6] |= 1 << (k & 63);
        }
    }
    for p in 0..PARSE_END {
        seen[key(p) >> 6] = 0;
    }
    res
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

/// The `codes` and seed constants the kernel needs: block-init and prior-seed LL/ML/OF prices
/// (the `prior` tables, `opt::seed_prices`), LL/ML extra bits, and the ML codes of match lengths
/// 3..131.
fn tables_wgsl(prior: PriorTables) -> String {
    use gzc_core::codes::{
        LL_BITS, ML_BITS, OPT_PRIOR_LL, OPT_PRIOR_ML, OPT_PRIOR_OF, OPT_PRIOR_S3_LL, OPT_PRIOR_S3_ML, OPT_PRIOR_S3_OF, ml_code,
    };
    let bi = Prices::block_init(&vec![0u8; BLOCK_SIZE]);
    let (ll, ml, of) = match prior {
        PriorTables::M5 => (OPT_PRIOR_LL, OPT_PRIOR_ML, OPT_PRIOR_OF),
        PriorTables::S3 => (OPT_PRIOR_S3_LL, OPT_PRIOR_S3_ML, OPT_PRIOR_S3_OF),
    };
    let pr = Prices::from_hist(&Hist { lit: [0; 256], ll, ml, of });
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

const K3_DROP_WGSL: &str = include_str!("shaders/k3_drop.wgsl");

/// The drop pass (M6 B3, `shaders/k3_drop.wgsl`): `opt::drop_pass` on the fixed-up parses of a
/// final K3opt pass (`seqs`, `counts`, and the per-segment trailers the pass left in `best`), in
/// place. One workgroup of 64 lanes per block: the output's histogram and its prices, one lane per
/// 4 KiB segment for the decisions (`opt::drop_decisions`), then the compaction and the
/// off_bases re-encoded against the true reps (`opt::apply_drops`).
pub struct K3Drop {
    pipe: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// The opt params the pass was built for (`drop_max_len > 0`).
    pub params: MatchParams,
    /// Test hook: the drop prices are the `opt::Prices` tables in `prices` (`PRICE_WORDS` per
    /// block), not those of the output's histogram.
    pub prices_in: bool,
}

impl K3Drop {
    /// Builds the drop pass of opt params `m` (`drop_max_len > 0`); `prices_in`: see
    /// `K3Drop::prices_in`.
    pub fn new(ctx: &GpuContext, m: &MatchParams, prices_in: bool) -> anyhow::Result<Self> {
        m.validate().map_err(|e| anyhow!("invalid match params {m:?}: {e}"))?;
        let o = m.opt.ok_or_else(|| anyhow!("K3Drop needs opt params"))?;
        ensure!(o.drop_max_len > 0, "K3Drop: drop_max_len is 0");
        ensure!(n_seg(m) <= 64 && 64 % n_seg(m) == 0, "K3Drop: {} segments per block", n_seg(m));
        let body = format!(
            "{}const MAX_SEQS: u32 = {MAX_SEQS_OPT}u;\nconst SEG_LOG2: u32 = {}u;\nconst DROP_MAX: u32 = {}u;\n\
             const DROP_PRICES_IN: bool = {prices_in};\n{}{K3_DROP_WGSL}\n{K3_FIXUP_WGSL}",
            params_wgsl(m),
            m.segment_log2,
            o.drop_max_len,
            tables_wgsl(o.prior),
        );
        let layout = crate::compressor::storage_layout(ctx, "k3drop", &[true, false, false, false, true]);
        let module = ctx.shader_unbounded_loops("k3_drop", &body);
        let pipe = crate::compressor::pipeline_from_module(ctx, "k3_drop", &layout, &module, "main_drop");
        Ok(Self { pipe, layout, params: *m, prices_in })
    }

    /// Records the drop pass on the first `n` blocks of `bufs` (after the final pass's fix-up), in
    /// its own compute pass with `ts`.
    pub fn record(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &OptBinds,
        n: u32,
        ts: Option<wgpu::ComputePassTimestampWrites>,
    ) -> anyhow::Result<()> {
        bufs.check(n, &self.params)?;
        let max_groups = ctx.device.limits().max_compute_workgroups_per_dimension;
        ensure!(n <= max_groups, "k3drop: {n} blocks > {max_groups} workgroups");
        let counts = wgpu::BufferBinding { buffer: bufs.counts, offset: 0, size: wgpu::BufferSize::new(counts_bytes(n)) };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k3drop"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs.data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs.cands.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bufs.seqs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(counts) },
                wgpu::BindGroupEntry { binding: 4, resource: bufs.prices.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k3drop"), timestamp_writes: ts });
        pass.set_bind_group(0, &bind, &[]);
        pass.set_pipeline(&self.pipe);
        pass.dispatch_workgroups(n, 1, 1);
        Ok(())
    }
}

/// The words a final K3opt pass and its fix-up leave for `K3Drop` when their parse is `out`
/// (`BLOCK_SIZE`-byte segments of `1 << seg_log2`): `best` (`2 * BLOCK_SIZE` words: each
/// segment's raw sequences, last first, with off_bases under the segment's own history from
/// `INITIAL_REPS` (segment 0) or `[0, 0, 0]`, and its 6-word trailer at `SEG_META`), the parse's
/// `seqs` words and its counts. For replaying a scripted drop input (`opt::cases::DropCase`).
/// Every match must lie in one segment.
pub fn final_pass_words(out: &BlockOutput, seg_log2: u32) -> (Vec<u32>, Vec<u32>, [u32; 2]) {
    use gzc_core::seq::{INITIAL_REPS, apply_off_base, off_base_for};
    let seg = 1usize << seg_log2;
    let seg_words = 2 * seg;
    let meta = seg_words - 6;
    let mut segs: Vec<Vec<(usize, u32, u32)>> = vec![Vec::new(); BLOCK_SIZE / seg];
    let (mut r, mut pos) = (INITIAL_REPS, 0usize);
    let mut ml_sum = 0u32;
    for s in &out.sequences {
        let off = apply_off_base(&mut r, s.off_base, s.lit_len);
        let start = pos + s.lit_len as usize;
        pos = start + s.match_len as usize;
        assert_eq!(start / seg, (pos - 1) / seg, "a match crosses a segment");
        segs[start / seg].push((start, s.match_len, off));
        ml_sum += s.match_len;
    }
    let mut best = vec![0u32; 2 * BLOCK_SIZE];
    for (k, v) in segs.iter().enumerate() {
        let mut local = if k == 0 { INITIAL_REPS } else { [0; 3] };
        let mut anchor = k * seg;
        let (mut n, mut ml) = (0u32, 0u32);
        let at = k * seg_words;
        for (i, &(start, m, off)) in v.iter().enumerate() {
            let ll = (start - anchor) as u32;
            let ob = off_base_for(off, ll, &local);
            apply_off_base(&mut local, ob, ll);
            let w = at + 3 * (v.len() - 1 - i);
            best[w..w + 3].copy_from_slice(&[ll, m, ob]);
            anchor = start + m as usize;
            n += 1;
            ml += m;
        }
        best[at + meta..at + meta + 6].copy_from_slice(&[n, anchor as u32, ml, local[0], local[1], local[2]]);
    }
    let seqs: Vec<u32> = out.sequences.iter().flat_map(|s| [s.lit_len, s.match_len, s.off_base]).collect();
    (best, seqs, [out.sequences.len() as u32, BLOCK_SIZE as u32 - ml_sum])
}

/// Runs `d` on scripted inputs: each block's parse `inputs[i]` as a final pass would leave it
/// (`final_pass_words`), with the drop prices `prices[i]` when `d.prices_in`. Returns the parses
/// after the drop pass.
pub fn drops_from_parses(
    ctx: &GpuContext,
    d: &K3Drop,
    blocks: &[&[u8]],
    inputs: &[BlockOutput],
    prices: Option<&[Prices]>,
) -> anyhow::Result<Vec<BlockOutput>> {
    ensure!(blocks.len() == inputs.len() && !blocks.is_empty(), "one input per block");
    ensure!(prices.is_some() == d.prices_in, "price tables iff prices_in");
    let n = blocks.len() as u32;
    crate::compressor::with_error_scopes(ctx, || {
        let bufs = OptBuffers::new(ctx, &d.params, n)?;
        ctx.queue.write_buffer(&bufs.data, 0, bytemuck::cast_slice(&pack_blocks(blocks)));
        let per = best_bytes_for(1, &d.params);
        let mut counts = Vec::with_capacity(2 * blocks.len());
        for (b, out) in inputs.iter().enumerate() {
            let (best, seqs, c) = final_pass_words(out, d.params.segment_log2);
            ctx.queue.write_buffer(&bufs.cands, b as u64 * per, bytemuck::cast_slice(&best));
            ctx.queue.write_buffer(&bufs.seqs, b as u64 * seqs_bytes_for(1, &d.params), bytemuck::cast_slice(&seqs));
            counts.extend_from_slice(&c);
        }
        ctx.queue.write_buffer(&bufs.counts, 0, bytemuck::cast_slice(&counts));
        if let Some(ps) = prices {
            let words: Vec<u32> =
                ps.iter().flat_map(|p| p.lit.iter().chain(&p.ll).chain(&p.ml).chain(&p.of).map(|&x| x as u32)).collect();
            ctx.queue.write_buffer(&bufs.prices, 0, bytemuck::cast_slice(&words));
        }
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("k3drop") });
        if ctx.poisoning() {
            // Everything but the uploaded words (the inputs' seqs words are left as they are).
            ctx.poison_workgroup_memory(&mut enc);
            ctx.poison_from(&mut enc, &bufs.data, data_bytes(n));
            ctx.poison_from(&mut enc, &bufs.cands, best_bytes_for(n, &d.params));
            ctx.poison_from(&mut enc, &bufs.counts, counts_bytes(n));
            for b in [&bufs.trace, &bufs.scratch, &bufs.sched] {
                ctx.poison_from(&mut enc, b, 0);
            }
            ctx.poison_from(&mut enc, &bufs.prices, if d.prices_in { prices_bytes(n) } else { 0 });
        }
        d.record(ctx, &mut enc, &bufs.binds(), n, None)?;
        ctx.queue.submit([enc.finish()]);
        read_parses(ctx, &bufs, blocks)
    })
}

impl K3Opt {
    /// Builds the pass for opt params `m` (segment size, `target_length`) under `cfg`.
    pub fn new(ctx: &GpuContext, m: &MatchParams, cfg: K3OptConfig) -> anyhow::Result<Self> {
        m.validate()
            .map_err(|e| anyhow!("invalid match params {m:?}: {e}"))?;
        let o = m.opt.ok_or_else(|| anyhow!("K3opt needs opt params"))?;
        ensure!(cfg.level == 0 || cfg.level == 2, "optLevel {}", cfg.level);
        let suff = o.target_length.min(4095);
        ring_for(m, &cfg, ctx.device.limits().max_compute_workgroup_storage_size)?;
        let body = format!(
            "{}const MAX_SEQS: u32 = {MAX_SEQS_OPT}u;\nconst SEG_LOG2: u32 = {}u;\nconst SUFF: u32 = {suff}u;\n\
             const LEVEL: u32 = {}u;\nconst PRICE_MODE: u32 = {}u;\nconst HIST_OUT: bool = {};\n\
             const DEAD_BIT: u32 = {}u;\nconst SCHED_HDR: u32 = {SCHED_HDR}u;\n\
             const GAP: u32 = {}u;\nconst RELAX_N: u32 = {}u;\n\
             {}{K3_OPT_WGSL}\n{K3_FIXUP_WGSL}",
            params_wgsl(m),
            m.segment_log2,
            cfg.level,
            match cfg.prices {
                PriceSrc::BlockInit => 0,
                PriceSrc::Buffer => 1,
                PriceSrc::Prior => 2,
                PriceSrc::Hist => 3,
            },
            cfg.hist_out,
            gzc_core::reference::DEAD_BIT,
            o.inner_gap,
            o.relax_lengths.unwrap_or(0),
            tables_wgsl(o.prior),
        );
        let layout = crate::compressor::storage_layout(
            ctx,
            "k3opt",
            &[true, false, false, false, false, false, false, false],
        );
        let module = if cfg.unbounded {
            ctx.shader_unbounded_loops("k3_opt", &body)
        } else {
            ctx.shader("k3_opt", &body)
        };
        let main = crate::compressor::pipeline_from_module(ctx, "k3_opt", &layout, &module, "main_opt_persist");
        let sched = {
            let body = format!(
                "const SCHED_HDR: u32 = {SCHED_HDR}u;\nconst WEIGHT_RUN: u32 = {WEIGHT_RUN}u;\nconst WEIGHT_STRIDE: u32 = {WEIGHT_STRIDE}u;\n{K3_SCHED_WGSL}"
            );
            let layout = crate::compressor::storage_layout(ctx, "k3opt_sched", &[true, true, false]);
            let module = ctx.shader("k3_sched", &body);
            let pipe = |entry: &str| crate::compressor::pipeline_from_module(ctx, "k3_sched", &layout, &module, entry);
            Sched { weight: pipe("main_weight"), rank: pipe("main_rank"), scatter: pipe("main_scatter"), layout }
        };
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
            sched,
            grid: cfg.grid,
            prices: cfg.prices,
            level: cfg.level,
            hist_out: cfg.hist_out,
            params: *m,
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
        self.record_order(ctx, enc, bufs, n, None)?;
        self.record_with(ctx, enc, bufs, n, ts(0), ts(1))
    }

    /// Records the batch's heavy-first block order (`k3_sched.wgsl`) for the first `n` blocks
    /// of `bufs` into `bufs.sched`, from the candidate words in `bufs.cands` (so before a final
    /// pass overwrites them). One compute pass, with `ts`. Every pass on these blocks can then
    /// use it.
    pub fn record_order(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &OptBinds,
        n: u32,
        ts: Option<wgpu::ComputePassTimestampWrites>,
    ) -> anyhow::Result<()> {
        bufs.check(n, &self.params)?;
        let max_groups = ctx.device.limits().max_compute_workgroups_per_dimension;
        ensure!(n <= max_groups, "k3opt: {n} blocks > {max_groups} workgroups");
        let s = &self.sched;
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k3opt_sched"),
            layout: &s.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs.data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs.cands.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sched_binding(bufs, n) },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k3opt_order"), timestamp_writes: ts });
        pass.set_bind_group(0, &bind, &[]);
        let tiles = n.div_ceil(256);
        pass.set_pipeline(&s.weight);
        pass.dispatch_workgroups(n, 1, 1);
        pass.set_pipeline(&s.rank);
        pass.dispatch_workgroups(tiles, tiles, 1);
        pass.set_pipeline(&s.scatter);
        pass.dispatch_workgroups(tiles, 1, 1);
        Ok(())
    }

    /// `record` on `bufs` with the given timestamp writes for the DP's and the fix-up's compute
    /// passes (the fix-up's are unused after a `hist_out` pass). The pass needs the
    /// batch's block order in `bufs.sched` (`record_order`, recorded before it on these blocks), so
    /// this is crate-private: `record_at`, `OptPasses::record` and `OptPasses::record_span` record
    /// the order first.
    pub(crate) fn record_with(
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
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: sched_binding(bufs, n),
                },
            ],
        });
        let max_groups = ctx.device.limits().max_compute_workgroups_per_dimension;
        ensure!(n <= max_groups, "k3opt: {n} workgroups > {max_groups}");
        // The pass's block counter starts at 0 (the order is `record_order`'s). The grid
        // stays one workgroup per block (a07: the same as one per resident slot, which wgpu
        // cannot query); the workgroups that start after the order is used up exit at once.
        enc.clear_buffer(bufs.sched, 0, Some(4 * SCHED_HDR as u64));
        let groups = n.min(self.grid.unwrap_or(u32::MAX).max(1));
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

/// `bufs.sched` bound to exactly `sched_bytes(n)` (the kernels derive `n` from its length).
fn sched_binding<'a>(bufs: &OptBinds<'a>, n: u32) -> wgpu::BindingResource<'a> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: bufs.sched,
        offset: 0,
        size: wgpu::BufferSize::new(sched_bytes(n)),
    })
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
                for b in [&bufs.trace, &bufs.seqs, &bufs.counts, &bufs.scratch, &bufs.sched] {
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

/// The DP passes of an opt preset (`opt::passes`, M5 T4): `o.passes` cheap passes at optLevel 0
/// (`hist_out`: each writes its `opt::Hist` for the next one's prologue and keeps the candidate
/// words), then the final pass at `o.level` with the fix-up. Pass 0 is priced by the seed
/// (`Seed::BlockInit` → `PriceSrc::BlockInit`, `Seed::Prior` → `PriceSrc::Prior`), every later
/// pass by the previous pass's histogram (`PriceSrc::Hist`). One dispatch per pass (plus the
/// fix-up): the candidates, the data and the histograms stay resident in `OptBuffers`. With
/// `drop_max_len > 0` (M6 `opt16p1`) the drop pass (`K3Drop`) follows the fix-up, as
/// `opt::parse` applies `opt::drop_pass` to the final pass's output.
pub struct OptPasses {
    /// The kernel of each pass, in order (a kernel shared by several passes appears once per pass).
    kernels: Vec<std::sync::Arc<K3Opt>>,
    /// The drop pass after the final pass's fix-up (`OptParams::drop_max_len > 0`).
    drop: Option<K3Drop>,
}

impl OptPasses {
    /// Builds the passes of opt params `m`. `base` gives the loop bounding (and the test-only
    /// grid); its `level`, `prices` and `hist_out` are set per pass.
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
        let drop = if o.drop_max_len > 0 { Some(K3Drop::new(ctx, m, false)?) } else { None };
        Ok(Self { kernels, drop })
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

    /// The drop pass after the final pass (`OptParams::drop_max_len > 0`).
    pub fn drop_pass(&self) -> Option<&K3Drop> {
        self.drop.as_ref()
    }

    /// Records the block order (M6 A4, once for all passes: K2opt's candidate words are intact
    /// until the final pass), every pass and the drop pass on the first `n` blocks of `bufs`
    /// (uploaded blocks and candidate words). Timestamps: pass i's DP at `2i`/`2i + 1`, the final
    /// fix-up at `2 * n_passes()` and `2 * n_passes() + 1`, the block order at `2 * n_passes() + 2`
    /// and `2 * n_passes() + 3`, the drop pass at `2 * n_passes() + 4` and `2 * n_passes() + 5`
    /// (an empty compute pass carries them when there is none).
    pub fn record(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &OptBuffers,
        n: u32,
        queries: Option<&wgpu::QuerySet>,
    ) -> anyhow::Result<()> {
        let binds = bufs.binds();
        let at = 2 * self.n_passes() as u32 + 2;
        let ts = queries.map(|query_set| wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(at),
            end_of_pass_write_index: Some(at + 1),
        });
        self.record_order(ctx, enc, &binds, n, ts)?;
        for (i, k) in self.kernels().enumerate() {
            let ts = |j: u32| {
                queries.map(|query_set| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(2 * i as u32 + 2 * j),
                    end_of_pass_write_index: Some(2 * i as u32 + 2 * j + 1),
                })
            };
            // The final pass's fix-up lands at 2 * i + 2 = 2 * n_passes().
            k.record_with(ctx, enc, &binds, n, ts(0), ts(1))?;
        }
        let at = 2 * self.n_passes() as u32 + 4;
        let ts = queries.map(|query_set| wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(at),
            end_of_pass_write_index: Some(at + 1),
        });
        match &self.drop {
            Some(d) => d.record(ctx, enc, &binds, n, ts)?,
            None if ts.is_some() => {
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k3drop"), timestamp_writes: ts });
            }
            None => {}
        }
        Ok(())
    }

    /// The block order of the passes (`K3Opt::record_order`).
    fn record_order(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &OptBinds,
        n: u32,
        ts: Option<wgpu::ComputePassTimestampWrites>,
    ) -> anyhow::Result<()> {
        self.kernels[0].record_order(ctx, enc, bufs, n, ts)
    }

    /// Records the block order, every pass and the drop pass on the first `n` blocks of `bufs`,
    /// with one timestamp pair spanning them all (the pipeline's K3 entry): `span`'s beginning
    /// write on the block order's compute pass and its end write on the drop pass, or without one
    /// on the final fix-up.
    pub fn record_span(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &OptBinds,
        n: u32,
        span: Option<wgpu::ComputePassTimestampWrites>,
    ) -> anyhow::Result<()> {
        let last = self.kernels.len() - 1;
        let begin = span.as_ref().map(|s| wgpu::ComputePassTimestampWrites {
            query_set: s.query_set,
            beginning_of_pass_write_index: s.beginning_of_pass_write_index,
            end_of_pass_write_index: None,
        });
        self.record_order(ctx, enc, bufs, n, begin)?;
        let end = || {
            span.as_ref().map(|s| wgpu::ComputePassTimestampWrites {
                query_set: s.query_set,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: s.end_of_pass_write_index,
            })
        };
        for (i, k) in self.kernels().enumerate() {
            // Only the final pass (last) has a fix-up.
            let fix_end = if i == last && self.drop.is_none() { end() } else { None };
            k.record_with(ctx, enc, bufs, n, None, fix_end)?;
        }
        if let Some(d) = &self.drop {
            d.record(ctx, enc, bufs, n, end())?;
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
/// final parses (after the drop pass, if any: `opt::parse`) and, with `hists`, each cheap pass's
/// histograms (`[pass][block]`, the GPU's counterpart of
/// `opt::Hist::of_output(&opt::passes(..)[pass].out)`; one submission per pass).
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
            if ctx.poisoning() {
                // Everything but the uploaded blocks (+ trailing word) and candidates.
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("k3opt.poison") });
                ctx.poison_workgroup_memory(&mut enc);
                ctx.poison_from(&mut enc, &bufs.data, data_bytes(n));
                ctx.poison_from(&mut enc, &bufs.cands, best_bytes_for(n, p.params()));
                for b in [&bufs.trace, &bufs.seqs, &bufs.counts, &bufs.scratch, &bufs.sched, &bufs.prices] {
                    ctx.poison_from(&mut enc, b, 0);
                }
                ctx.queue.submit([enc.finish()]);
            }
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
                if let Some(d) = &p.drop {
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("k3drop") });
                    d.record(ctx, &mut enc, &bufs.binds(), n, None)?;
                    ctx.queue.submit([enc.finish()]);
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

/// GPU time (ms, median of `reps` runs after one warm-up) of each DP pass of `p`, of the final
/// fix-up, of the block order (M6 A4) and of the drop pass
/// (M6 B3; 0 without one), on the first `n` blocks of `bufs`: `n_passes() + 3` values, then the
/// span from the block order's start to the end of the drop pass, or without one of the fix-up
/// (dispatch gaps included). The final pass overwrites part of the candidate words, so `reupload`
/// runs before every run (outside the timed passes).
pub fn time_passes(
    ctx: &GpuContext,
    p: &OptPasses,
    bufs: &OptBuffers,
    n: u32,
    reps: usize,
    mut reupload: impl FnMut(),
) -> anyhow::Result<Vec<f64>> {
    ensure!(ctx.timestamps, "timestamps unavailable");
    let q = 2 * (p.n_passes() + 3);
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
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); p.n_passes() + 4];
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
            for (i, v) in times.iter_mut().take(p.n_passes() + 3).enumerate() {
                let d = if i == p.n_passes() + 2 && p.drop.is_none() { 0 } else { t[2 * i + 1].saturating_sub(t[2 * i]) };
                v.push(d as f64 * period / 1e6);
            }
            // From the block order's start (2 * n_passes() + 2) to the drop pass's end (q - 1), or
            // the fix-up's (2 * n_passes() + 1).
            let end = if p.drop.is_some() { t[q - 1] } else { t[2 * p.n_passes() + 1] };
            times[p.n_passes() + 3].push(end.saturating_sub(t[2 * p.n_passes() + 2]) as f64 * period / 1e6);
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
        let _gpu = crate::testing::gpu_test_slot();
        // Without poisoning: its padding would make the short buffers below long enough.
        let env = crate::context::GpuOptions::from_env();
        if env.poison {
            eprintln!("skipped: poisoning pads every buffer");
            return;
        }
        let ctx = GpuContext::new(env).expect("GPU required for gzc-gpu tests");
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

    /// `check_dead_marks` (M6 A3) accepts `find_cands`' marks, also twice with the same scratch,
    /// and rejects a mark on a position with an earlier 3-byte match, on a position with a record,
    /// and at `PARSE_END`.
    #[test]
    fn check_dead_marks_rejects_bad_marks() {
        use gzc_core::config::PARSE_END;
        use gzc_core::reference::{DEAD_BIT, chains, find_cands, is_dead};
        let mut block = gzc_core::synth::random(5, BLOCK_SIZE);
        block.copy_within(200..300, 1000);
        let c = find_cands(&block, &chains(&block, &OPT16), &OPT16);
        assert!(c.iter().filter(|w| is_dead(**w)).count() > BLOCK_SIZE / 2);
        let mut seen = Vec::new();
        check_dead_marks(&block, &c, &mut seen).unwrap();
        check_dead_marks(&block, &c, &mut seen).unwrap();
        // 1050 repeats 250's bytes: not dead.
        assert!(!is_dead(c[1050]));
        let mut bad = c.clone();
        bad[1050] = [0, DEAD_BIT];
        assert!(check_dead_marks(&block, &bad, &mut seen).is_err(), "an earlier 3-byte match");
        let mut bad = c.clone();
        bad[1050][1] |= DEAD_BIT;
        assert!(check_dead_marks(&block, &bad, &mut seen).is_err(), "a record");
        let mut bad = c.clone();
        bad[PARSE_END] = [0, DEAD_BIT];
        assert!(check_dead_marks(&block, &bad, &mut seen).is_err(), "at PARSE_END");
        check_dead_marks(&block, &c, &mut seen).unwrap();
    }
}
