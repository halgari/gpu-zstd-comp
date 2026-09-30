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
use gzc_core::opt::Prices;
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
}

impl Default for K3OptConfig {
    fn default() -> Self {
        Self {
            level: 2,
            wg: 16,
            ring: None,
            prices: PriceSrc::BlockInit,
            unbounded: true,
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

/// Workgroup bytes of the price tables (and the prologue's histogram) for `bpw` blocks.
fn price_table_bytes(bpw: u32, target_length: u32) -> u32 {
    bpw * (256 + 64 + 36 + (target_length + 1) + 32 + 256 + 1) * 4
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
                false,
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
        let bpw = (cfg.wg / n_seg).max(1);
        let limit = ctx.device.limits().max_compute_workgroup_storage_size;
        let wg_need = ring_bytes(cfg.wg, suff) + price_table_bytes(bpw, suff) + 3 * 4 * n_seg + 64;
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
        let bi = Prices::block_init(&vec![0u8; BLOCK_SIZE]);
        let body = format!(
            "{}const MAX_SEQS: u32 = {MAX_SEQS_OPT}u;\nconst SEG_LOG2: u32 = {}u;\nconst WG: u32 = {}u;\nconst SUFF: u32 = {suff}u;\n\
             const LEVEL: u32 = {}u;\nconst PRICE_MODE: u32 = {}u;\n{}{}{}{ring_decl}{K3_OPT_WGSL}\n{K3_FIXUP_WGSL}",
            params_wgsl(m),
            m.segment_log2,
            cfg.wg,
            cfg.level,
            match cfg.prices {
                PriceSrc::BlockInit => 0,
                PriceSrc::Buffer => 1,
            },
            wgsl_array("BI_LL", &bi.ll),
            wgsl_array("BI_ML", &bi.ml),
            wgsl_array("BI_OF", &bi.of),
        );
        let layout = crate::compressor::storage_layout(
            ctx,
            "k3opt",
            &[true, false, false, false, false, true],
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
            n_seg,
        })
    }

    /// Records the DP pass then the fix-up on the first `n` blocks of `bufs`, each in its own
    /// compute pass (timestamps: `queries` 0/1 around the DP, 2/3 around the fix-up).
    pub fn record(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &OptBuffers,
        n: u32,
        queries: Option<&wgpu::QuerySet>,
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
            queries.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(2 * k),
                end_of_pass_write_index: Some(2 * k + 1),
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
            fix.push((t[3] - t[2]) as f64 * period / 1e6);
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    Ok((med(&mut main), med(&mut fix)))
}
