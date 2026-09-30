//! K3opt (M5 stage S2): one DP pass of the optimal parse on the GPU (`shaders/k3_opt.wgsl`),
//! byte-identical to `gzc_core::opt::dp_pass_with(.., Engine::Ring)`, then the segmented parse's
//! fix-up (`shaders/k3_fixup.wgsl`).
//!
//! Stand-alone until the pipeline integration (T5): it has its own buffers (`OptBuffers`) and is
//! fed K2opt's candidate words from the host (`parses_from_cands`, the counterpart of
//! `compressor::parses_from_best`), so it runs on `reference::find_cands` output or on scripted
//! candidates (`opt::cases`). Prices come either from zstd's first-block statistics computed in
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
//! Buffers per block: data (BLOCK_SIZE), candidate words (8 B per position; after the DP each
//! segment's first words take its raw sequences, as `k3_seg.wgsl` does with `best`), trace (8 B
//! per position: K1's `pred` in the pipeline), seqs (`MAX_SEQS_OPT` × 12 B; the DP's series log
//! until the fix-up), counts, and the price tables (`PRICE_WORDS` words).
//!
//! Workgroups: `K3OptConfig::wg` lanes (a power of two, 8..=256; a block's 16 segment lanes may
//! span several workgroups). 16 is the fastest on an RTX 5090 at 64 KiB blocks (M5 T3 log in
//! `docs/results/m5-log.md`).
use crate::compressor::{K3_FIXUP_WGSL, counts_bytes, data_bytes, decode_output};
use crate::context::{GpuContext, pack_blocks, params_wgsl};
use anyhow::{anyhow, ensure};
use gzc_core::config::BLOCK_SIZE;
use gzc_core::opt::{Hist, Prices};
use gzc_core::params::MatchParams;
use gzc_core::reference::CandWords;
use gzc_core::seq::BlockOutput;

const K3_OPT_WGSL: &str = include_str!("shaders/k3_opt.wgsl");

/// Sequences per block of the optimal parse (min match 3): `BLOCK_SIZE / 3 + 1`.
pub const MAX_SEQS_OPT: u32 = (BLOCK_SIZE / 3) as u32 + 1;

/// Words of one block's price tables in the `prices` buffer: `opt::Prices` as lit[256], ll[36],
/// ml[53], of[32].
pub const PRICE_WORDS: usize = 256 + 36 + 53 + 32;

/// Largest batch `parses_from_cands` runs at once.
const BATCH_CAP: usize = 256;

/// Where the DP rings live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingMem {
    /// `var<workgroup>` (needs `ring_bytes(wg)` plus the price tables within the device's
    /// `max_compute_workgroup_storage_size`, which `GpuContext` requests at the adapter's limit).
    Workgroup,
    /// `var<private>`: the fallback for small workgroup storage.
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

/// Workgroup bytes of `wg` lanes' rings: 16 B per node, `target_length + 1` nodes per lane.
pub fn ring_bytes(wg: u32, target_length: u32) -> u32 {
    wg * (target_length + 1) * 16
}

/// Workgroup bytes of the price tables (and the prologue's histogram and sums, which also hold a
/// `hist_out` pass's histogram) for `bpw` blocks.
fn price_table_bytes(bpw: u32, target_length: u32) -> u32 {
    bpw * (256 + 64 + 36 + (target_length + 1) + 32 + 256 + 5) * 4
}

/// Workgroup bytes `K3Opt` needs with its rings in workgroup memory.
pub fn workgroup_bytes(m: &MatchParams, cfg: &K3OptConfig) -> u32 {
    let suff = m.opt.map_or(32, |o| o.target_length).min(4095);
    let n_seg = n_seg(m);
    let bpw = (cfg.wg / n_seg).max(1);
    ring_bytes(cfg.wg, suff) + price_table_bytes(bpw, suff) + 3 * 4 * n_seg + 64
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
    n_seg: u32,
}

/// K3opt's buffers for `capacity` blocks.
pub struct OptBuffers {
    pub capacity: u32,
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
}

/// Bytes of the candidate words (and of the trace) per `n` blocks.
pub fn cands_bytes(n: u32) -> u64 {
    n as u64 * BLOCK_SIZE as u64 * 8
}

/// Bytes of the opt `seqs` buffer per `n` blocks.
pub fn seqs_opt_bytes(n: u32) -> u64 {
    n as u64 * MAX_SEQS_OPT as u64 * 12
}

impl OptBuffers {
    pub fn new(ctx: &GpuContext, capacity: u32) -> Self {
        Self {
            capacity,
            data: ctx.storage_buffer("k3opt.data", data_bytes(capacity), false),
            cands: ctx.storage_buffer("k3opt.cands", cands_bytes(capacity), true),
            trace: ctx.storage_buffer("k3opt.trace", cands_bytes(capacity), false),
            seqs: ctx.storage_buffer("k3opt.seqs", seqs_opt_bytes(capacity), true),
            counts: ctx.storage_buffer("k3opt.counts", counts_bytes(capacity), true),
            prices: ctx.storage_buffer(
                "k3opt.prices",
                capacity as u64 * PRICE_WORDS as u64 * 4,
                true,
            ),
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
                b as u64 * cands_bytes(1),
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
        let limit = ctx.device.limits().max_compute_workgroup_storage_size;
        let wg_need = workgroup_bytes(m, &cfg);
        let ring = cfg.ring.unwrap_or(if wg_need <= limit {
            RingMem::Workgroup
        } else {
            RingMem::Private
        });
        ensure!(
            ring == RingMem::Private || wg_need <= limit,
            "workgroup ring needs {wg_need} B > limit {limit}"
        );
        let ring_decl = match ring {
            RingMem::Workgroup => {
                "var<workgroup> ring_p: array<i32, RING_N * WG>;\n\
                 var<workgroup> ring_a: array<u32, RING_N * WG>;\n\
                 var<workgroup> ring_b: array<u32, RING_N * WG>;\n\
                 var<workgroup> ring_c: array<u32, RING_N * WG>;\n\
                 fn rix(s: u32) -> u32 { return s * WG + lane; }\n"
            }
            RingMem::Private => {
                "var<private> ring_p: array<i32, RING_N>;\n\
                 var<private> ring_a: array<u32, RING_N>;\n\
                 var<private> ring_b: array<u32, RING_N>;\n\
                 var<private> ring_c: array<u32, RING_N>;\n\
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
            &[true, false, false, false, false, false],
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
            n_seg,
        })
    }

    /// Records the DP pass then (unless `hist_out`) the fix-up on the first `n` blocks of `bufs`,
    /// each in its own compute pass (timestamps: `queries` 0/1 around the DP, 2/3 around the
    /// fix-up).
    pub fn record(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &OptBuffers,
        n: u32,
        queries: Option<&wgpu::QuerySet>,
    ) {
        self.record_at(ctx, enc, bufs, n, queries.map(|q| (q, 0)));
    }

    /// `record` with the timestamps at `queries.1 ..` (DP, then the fix-up).
    pub fn record_at(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &OptBuffers,
        n: u32,
        queries: Option<(&wgpu::QuerySet, u32)>,
    ) {
        assert!(n >= 1 && n <= bufs.capacity);
        let counts = wgpu::BufferBinding {
            buffer: &bufs.counts,
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
            ],
        });
        let ts = |k: u32| {
            queries.map(|(query_set, at)| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(at + 2 * k),
                end_of_pass_write_index: Some(at + 2 * k + 1),
            })
        };
        let groups = (n * self.n_seg).div_ceil(self.wg);
        assert!(
            groups <= ctx.device.limits().max_compute_workgroups_per_dimension,
            "k3opt: {groups} workgroups"
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("k3opt"),
                timestamp_writes: ts(0),
            });
            pass.set_bind_group(0, &bind, &[]);
            pass.set_pipeline(&self.main);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        if self.hist_out {
            return;
        }
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k3opt_fixup"),
            timestamp_writes: ts(1),
        });
        pass.set_bind_group(0, &bind, &[]);
        pass.set_pipeline(&self.fixup);
        pass.dispatch_workgroups(n, 1, 1);
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
        let bufs = OptBuffers::new(ctx, blocks.len().min(BATCH_CAP) as u32);
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
            k.record(ctx, &mut enc, &bufs, chunk.len() as u32, None);
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
        k.record(ctx, &mut enc, bufs, n, Some(&qs));
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
    ) {
        for (i, k) in self.kernels().enumerate() {
            // The final pass's fix-up lands at 2 * i + 2 = 2 * n_passes().
            k.record_at(ctx, enc, bufs, n, queries.map(|q| (q, 2 * i as u32)));
        }
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
        let bufs = OptBuffers::new(ctx, blocks.len().min(BATCH_CAP) as u32);
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
                    k.record(ctx, &mut enc, &bufs, n, None);
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
                p.record(ctx, &mut enc, &bufs, n, None);
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
        p.record(ctx, &mut enc, bufs, n, Some(&qs));
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
