//! K3opt (M5 T3): one GPU DP pass against the CPU oracle `opt::dp_pass_with(.., Engine::Ring)`,
//! fed `reference::find_cands` words (or `opt::cases`' scripted ones) from the host.
//! M5 T4: every pass of `OptPasses` (seeds, cheap passes with their histograms, final pass)
//! against `opt::passes`.
use gzc_core::block::chunk_file;
use gzc_core::config::BLOCK_SIZE;
use gzc_core::frame::write_frame;
use gzc_core::opt::cases::{opt_test_cases, run_case};
use gzc_core::opt::{Engine, Hist, Prices, dp_pass_with, drop_pass, passes};
use gzc_core::params::{LVL9SEG, MatchParams, OPT14, OPT16, OPT16P1, OptParams, PriorTables, Seed};
use gzc_core::reference::{CandWords, chains, find_cands};
use gzc_core::seq::BlockOutput;
use gzc_gpu::compressor::{GpuParams, Kernels, OptCandKernel, cands_from_blocks, frames_from_parses};
use gzc_gpu::context::GpuContext;
use gzc_gpu::k3opt::{
    K3Drop, K3Opt, K3OptConfig, OptBuffers, OptPasses, PriceSrc, SCHED_HDR, WEIGHT_RUN, WEIGHT_STRIDE,
    drops_from_parses, parses_from_cands, parses_from_passes, ring_for, ring_bytes, scratch_bytes_per_block,
    time_passes, workgroup_bytes,
};

fn cfg(level: u8, prices: PriceSrc) -> K3OptConfig {
    K3OptConfig { level, prices, ..K3OptConfig::default() }
}

fn first_diff(got: &BlockOutput, want: &BlockOutput) -> String {
    let n = got.sequences.len().min(want.sequences.len());
    for i in 0..n {
        if got.sequences[i] != want.sequences[i] {
            return format!(
                "seq {i}: got {:?} want {:?} (of {} / {})",
                got.sequences[i],
                want.sequences[i],
                got.sequences.len(),
                want.sequences.len()
            );
        }
    }
    format!(
        "seq counts {} / {}; literals {} / {}",
        got.sequences.len(),
        want.sequences.len(),
        got.literals.len(),
        want.literals.len()
    )
}

/// The oracle's single pass at `level` with `prices`.
fn oracle(block: &[u8], cands: &[CandWords], prices: &Prices, level: u8) -> BlockOutput {
    oracle_m(&OPT16, block, cands, prices, level)
}

/// `oracle` under opt params `m` (its M6 DP options: gap, relaxation pruning).
fn oracle_m(m: &MatchParams, block: &[u8], cands: &[CandWords], prices: &Prices, level: u8) -> BlockOutput {
    dp_pass_with(block, cands, m, prices, level, m.opt.unwrap().target_length, Engine::Ring)
}

fn synthetic_blocks() -> Vec<(String, Vec<u8>)> {
    gzc_core::synth::test_cases()
        .into_iter()
        .flat_map(|(name, bytes)| {
            chunk_file(&bytes)
                .into_iter()
                .enumerate()
                .map(move |(i, b)| (format!("{name}[{i}]"), b.data))
        })
        .collect()
}

fn cands_of(blocks: &[Vec<u8>]) -> Vec<Vec<CandWords>> {
    cands_of_m(blocks, &OPT16)
}

/// `reference::find_cands` of every block under `m` (its depth and sparse chains).
fn cands_of_m(blocks: &[Vec<u8>], m: &MatchParams) -> Vec<Vec<CandWords>> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Vec<CandWords>>> =
        blocks.iter().map(|_| Default::default()).collect();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(16);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(b) = blocks.get(i) else { break };
                    *out[i].lock().unwrap() = find_cands(b, &chains(b, m), m);
                }
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap()).collect()
}

/// GPU pass vs oracle on `blocks` for every config in `cfgs` (BlockInit prices computed on the
/// GPU; Buffer: the same block-init tables uploaded). Also checks each parse's K4 frame against
/// `write_frame` when `frames` is given.
fn check_blocks(
    ctx: &GpuContext,
    names: &[String],
    blocks: &[Vec<u8>],
    cfgs: &[K3OptConfig],
    frames: Option<&Kernels>,
) {
    let cands = cands_of(blocks);
    let bi: Vec<Prices> = blocks.iter().map(|b| Prices::block_init(b)).collect();
    check_blocks_with(ctx, names, blocks, &cands, &bi, cfgs, frames);
}

/// `check_blocks` with the candidate words and the pass's prices given (`tables`: the oracle's
/// prices, and the uploaded tables of a `PriceSrc::Buffer` config; `Prices::block_init` for a
/// `BlockInit` one).
fn check_blocks_with(
    ctx: &GpuContext,
    names: &[String],
    blocks: &[Vec<u8>],
    cands: &[Vec<CandWords>],
    tables: &[Prices],
    cfgs: &[K3OptConfig],
    frames: Option<&Kernels>,
) {
    check_blocks_m(ctx, &OPT16, names, blocks, cands, tables, cfgs, frames);
}

/// `check_blocks_with` under opt params `m` (the kernels and the oracle's pass).
#[allow(clippy::too_many_arguments)]
fn check_blocks_m(
    ctx: &GpuContext,
    m: &MatchParams,
    names: &[String],
    blocks: &[Vec<u8>],
    cands: &[Vec<CandWords>],
    tables: &[Prices],
    cfgs: &[K3OptConfig],
    frames: Option<&Kernels>,
) {
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let crefs: Vec<&[CandWords]> = cands.iter().map(|c| c.as_slice()).collect();
    let bi = tables;
    for &c in cfgs {
        let k = K3Opt::new(ctx, m, c).expect("K3Opt::new");
        let want: Vec<BlockOutput> = {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let out: Vec<std::sync::Mutex<BlockOutput>> =
                blocks.iter().map(|_| Default::default()).collect();
            std::thread::scope(|s| {
                for _ in 0..16 {
                    s.spawn(|| {
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(b) = blocks.get(i) else { break };
                            *out[i].lock().unwrap() = oracle_m(m, b, &cands[i], &bi[i], c.level);
                        }
                    });
                }
            });
            out.into_iter().map(|m| m.into_inner().unwrap()).collect()
        };
        let prices = (c.prices == PriceSrc::Buffer).then_some(bi);
        let got = parses_from_cands(ctx, &k, &refs, &crefs, prices).expect("parses_from_cands");
        let mut bad = Vec::new();
        for i in 0..blocks.len() {
            if got[i] != want[i] {
                bad.push(format!("{}: {}", names[i], first_diff(&got[i], &want[i])));
            }
        }
        assert!(
            bad.is_empty(),
            "{c:?}: {} of {} blocks differ:\n{}",
            bad.len(),
            blocks.len(),
            bad[..bad.len().min(10)].join("\n")
        );
        if let Some(kf) = frames {
            for (chunk, (bl, g)) in refs.chunks(256).zip(got.chunks(256)).enumerate() {
                let fr = frames_from_parses(ctx, kf, bl, g).expect("frames_from_parses");
                for (j, f) in fr.iter().enumerate() {
                    let i = chunk * 256 + j;
                    assert!(
                        *f == write_frame(&blocks[i], &want[i], kf.frame_options()),
                        "{}: frame differs",
                        names[i]
                    );
                }
            }
        }
        eprintln!("{c:?}: {} blocks equal", blocks.len());
    }
}

/// The workgroup memory of the pass kernels (M5 T3b, M6 A1): the rings and tables fit WebGPU's
/// minimum limit with room to spare, and a limit below them fails cleanly in `ring_for` (before
/// any pipeline is created).
#[test]
fn k3opt_workgroup_memory() {
    let auto = K3OptConfig::default();
    let target = OPT16.opt.unwrap().target_length;
    let need = workgroup_bytes(&OPT16, &auto);
    assert_eq!(ring_bytes(&OPT16), 16 * (target + 1) * 4);
    // M6 A1: the footprint of every pass kernel: at most 4266 B (23..24 resident blocks per SM
    // on an RTX 5090 with 100 KB of shared memory), the final pass (no histogram) 1020 B below
    // its cheap-pass twin. Every kernel within 16384 B (WebGPU's minimum limit).
    for (level, prices, hist_out) in [
        (0, PriceSrc::BlockInit, true),
        (0, PriceSrc::Prior, true),
        (0, PriceSrc::Hist, true),
        (2, PriceSrc::Hist, false),
        (2, PriceSrc::BlockInit, false),
        (2, PriceSrc::Buffer, false),
    ] {
        let c = K3OptConfig { level, prices, hist_out, ..auto };
        let b = workgroup_bytes(&OPT16, &c);
        assert!(b <= 16384, "{c:?}: workgroup footprint {b} B > 16384");
        assert!(b <= 4266, "{c:?}: workgroup footprint {b} B > 4266");
        assert!(ring_for(&OPT16, &c, 16384).is_ok(), "{c:?}");
        if !hist_out && prices == PriceSrc::Hist {
            assert_eq!(b + 1020, workgroup_bytes(&OPT16, &K3OptConfig { hist_out: true, ..c }));
        }
    }
    assert_eq!(need, 4068);
    assert!(ring_for(&OPT16, &auto, need).is_ok());
    let e = ring_for(&OPT16, &auto, need - 1).unwrap_err().to_string();
    assert!(e.contains("workgroup memory") && e.contains("4068"), "{e}");
    let seg = BLOCK_SIZE as u64 >> OPT16.segment_log2;
    assert_eq!(scratch_bytes_per_block(&OPT16), seg * (target as u64 + 1) * 12);
}

/// Every `opt::cases` run: explicit tables as given (or, for the preset runs, the oracle's final
/// pass prices), at the run's level.
#[test]
fn k3opt_matches_opt_cases() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let ctx = GpuContext::new().expect("GPU required");
    /// (name, block, candidate words, params with the pass level, prices, oracle output)
    type Run = (
        String,
        Vec<u8>,
        Vec<CandWords>,
        MatchParams,
        Prices,
        BlockOutput,
    );
    let mut runs: Vec<Run> = Vec::new();
    for c in opt_test_cases() {
        for (i, (params, prices, want)) in c.expect.iter().enumerate() {
            let out = run_case(&c.block, &c.cands, params, prices.as_ref(), Engine::Ring);
            assert_eq!(out.sequences, *want, "{} [{i}]: oracle", c.name);
            let (p, level) = match prices {
                Some(p) => (p.clone(), params.opt.unwrap().level),
                None => {
                    let last = passes(&c.block, &c.cands, params).pop().unwrap();
                    assert_eq!(last.out, out);
                    (last.prices, last.level)
                }
            };
            let pp = MatchParams {
                opt: Some(gzc_core::params::OptParams {
                    level,
                    ..params.opt.unwrap()
                }),
                ..*params
            };
            runs.push((
                format!("{} [{i}]", c.name),
                c.block.clone(),
                c.cands.clone(),
                pp,
                p,
                out,
            ));
        }
    }
    for level in [0u8, 2] {
        let sel: Vec<_> = runs
            .iter()
            .filter(|r| r.3.opt.unwrap().level == level)
            .collect();
        let k = K3Opt::new(&ctx, &OPT16, cfg(level, PriceSrc::Buffer)).expect("K3Opt::new");
        let blocks: Vec<&[u8]> = sel.iter().map(|r| r.1.as_slice()).collect();
        let cands: Vec<&[CandWords]> = sel.iter().map(|r| r.2.as_slice()).collect();
        let prices: Vec<Prices> = sel.iter().map(|r| r.4.clone()).collect();
        let got = parses_from_cands(&ctx, &k, &blocks, &cands, Some(&prices))
            .expect("parses_from_cands");
        for (r, g) in sel.iter().zip(&got) {
            assert!(*g == r.5, "{} level {level}: {}", r.0, first_diff(g, &r.5));
        }
        eprintln!("level {level}: {} case runs equal", sel.len());
    }
}

/// Every synthetic block: the block-init pass at optLevel 2 and 0 (prices computed on the GPU
/// and uploaded), with the persistent loop on 3 workgroups, and with naga's loop bounding; frames
/// through K4/K5.
#[test]
fn k3opt_matches_oracle_synthetic() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let ctx = GpuContext::new().expect("GPU required");
    let all = synthetic_blocks();
    let names: Vec<String> = all.iter().map(|(n, _)| n.clone()).collect();
    let blocks: Vec<Vec<u8>> = all.into_iter().map(|(_, b)| b).collect();
    let kf = Kernels::new(
        &ctx,
        GpuParams {
            matching: LVL9SEG,
            emit_frames: true,
            huffman: true,
        },
    )
    .unwrap();
    let cfgs = [
        cfg(2, PriceSrc::BlockInit),
        cfg(0, PriceSrc::BlockInit),
        cfg(2, PriceSrc::Buffer),
        cfg(0, PriceSrc::Buffer),
        K3OptConfig { grid: Some(3), ..cfg(2, PriceSrc::BlockInit) },
        K3OptConfig { unbounded: false, ..cfg(2, PriceSrc::BlockInit) },
    ];
    check_blocks(&ctx, &names, &blocks, &cfgs[..1], Some(&kf));
    check_blocks(&ctx, &names, &blocks, &cfgs[1..], None);
}

/// Informal (reads the real corpus): 4000 blocks spread over `data/corpus`, block-init pass at
/// optLevel 2 and 0, uploaded tables, the persistent loop on 37 workgroups, frames via K4/K5.
/// `cargo test --release -p gzc-gpu --test k3opt k3opt_corpus -- --ignored --nocapture`
#[test]
#[ignore]
fn k3opt_corpus() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let ctx = GpuContext::new().expect("GPU required");
    let n: usize = std::env::var("GZC_CORPUS_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000);
    let Some(blocks) = gzc_core::testdata::corpus_sample(n) else { return };
    let names: Vec<String> = (0..blocks.len()).map(|i| format!("corpus[{i}]")).collect();
    let kf = Kernels::new(
        &ctx,
        GpuParams {
            matching: LVL9SEG,
            emit_frames: true,
            huffman: true,
        },
    )
    .unwrap();
    check_blocks(&ctx, &names, &blocks, &[cfg(2, PriceSrc::BlockInit)], Some(&kf));
    check_blocks(
        &ctx,
        &names,
        &blocks,
        &[
            cfg(0, PriceSrc::BlockInit),
            cfg(2, PriceSrc::Buffer),
            K3OptConfig { grid: Some(37), ..cfg(2, PriceSrc::BlockInit) },
        ],
        None,
    );
}

// ---- M5 T4: passes ----

/// `f` over `items` on up to 16 threads.
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(usize, &T) -> R + Sync) -> Vec<R> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<R>>> = items.iter().map(|_| Default::default()).collect();
    std::thread::scope(|s| {
        for _ in 0..16 {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(x) = items.get(i) else { break };
                    *out[i].lock().unwrap() = Some(f(i, x));
                }
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}

/// The schedules T4 gates: cheap passes {0, 1, 3} × seeds {BlockInit, Prior}, the final pass at
/// OPT16's optLevel 2. (BlockInit 3 is `opt16`, Prior 1 is `opt14`.)
fn schedules() -> Vec<(String, MatchParams)> {
    let mut v = Vec::new();
    for seed in [Seed::BlockInit, Seed::Prior] {
        for passes in [0u8, 1, 3] {
            let m = MatchParams {
                opt: Some(OptParams {
                    passes,
                    seed,
                    ..OPT16.opt.unwrap()
                }),
                ..OPT16
            };
            v.push((format!("{seed:?} x{passes}"), m));
        }
    }
    assert!(v.iter().any(|(_, m)| *m == OPT16) && v.iter().any(|(_, m)| *m == OPT14));
    v
}

/// Every pass of each schedule in `sched` on the GPU (`OptPasses` built from `base`) against
/// `opt::passes`: each cheap pass's histogram (`Hist::of_output` of the oracle pass's output)
/// and the final parse. Returns the number of blocks whose final pass has `ll[1] < ll[0]` (the
/// optLevel-2 match + 1 literal path is live) over all schedules.
fn check_passes(
    ctx: &GpuContext,
    names: &[String],
    blocks: &[Vec<u8>],
    cands: &[Vec<CandWords>],
    sched: &[(String, MatchParams)],
    base: K3OptConfig,
) -> usize {
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let crefs: Vec<&[CandWords]> = cands.iter().map(|c| c.as_slice()).collect();
    let mut live = 0;
    for (sname, m) in sched {
        let p = OptPasses::new(ctx, m, base).expect("OptPasses::new");
        let want = par_map(blocks, |i, b| passes(b, &cands[i], m));
        let (got, hists) = parses_from_passes(ctx, &p, &refs, &crefs, true).expect("parses_from_passes");
        let mut bad = Vec::new();
        for i in 0..blocks.len() {
            let w = &want[i];
            assert_eq!(w.len(), p.n_passes());
            for (j, h) in hists.iter().enumerate() {
                let hw = Hist::of_output(&w[j].out);
                if h[i] != hw {
                    bad.push(format!(
                        "{}: pass {j} histogram differs:\ngot  {}want {}",
                        names[i],
                        h[i].to_text(),
                        hw.to_text()
                    ));
                }
            }
            let last = w.last().unwrap();
            // opt::parse: the drop pass follows the final DP pass.
            let fin = match m.opt.unwrap().drop_max_len {
                0 => last.out.clone(),
                d => drop_pass(&blocks[i], &last.out, d as u32, m.segment_log2),
            };
            if got[i] != fin {
                bad.push(format!("{}: final parse: {}", names[i], first_diff(&got[i], &fin)));
            }
            live += (last.prices.ll[1] < last.prices.ll[0]) as usize;
        }
        assert!(
            bad.is_empty(),
            "{sname} ({base:?}): {} differences over {} blocks:\n{}",
            bad.len(),
            blocks.len(),
            bad[..bad.len().min(6)].join("\n")
        );
        eprintln!("{sname}: {} blocks equal ({} passes, histograms included)", blocks.len(), p.n_passes());
    }
    live
}

/// The later-pass Buffer-price configs (T3 review): opt16's pass-1 prices (from pass 0's
/// histogram) uploaded as tables, one pass at optLevel 2 and 0. Returns the number of blocks
/// whose tables have `ll[1] < ll[0]`.
fn check_later_pass_tables(ctx: &GpuContext, names: &[String], blocks: &[Vec<u8>], cands: &[Vec<CandWords>]) -> usize {
    let tables: Vec<Prices> = par_map(blocks, |i, b| passes(b, &cands[i], &OPT16).swap_remove(1).prices);
    let live = tables.iter().filter(|p| p.ll[1] < p.ll[0]).count();
    for level in [2u8, 0] {
        check_blocks_with(
            ctx,
            names,
            blocks,
            cands,
            &tables,
            &[K3OptConfig {
                level,
                prices: PriceSrc::Buffer,
                ..K3OptConfig::default()
            }],
            None,
        );
    }
    eprintln!("later-pass tables: {live} of {} blocks have ll[1] < ll[0]", blocks.len());
    live
}

/// Every `opt::cases` block (with its scripted candidates) through every schedule.
#[test]
fn k3opt_passes_opt_cases() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let ctx = GpuContext::new().expect("GPU required");
    let cases = opt_test_cases();
    let names: Vec<String> = cases.iter().map(|c| c.name.clone()).collect();
    let blocks: Vec<Vec<u8>> = cases.iter().map(|c| c.block.clone()).collect();
    let cands: Vec<Vec<CandWords>> = cases.iter().map(|c| c.cands.clone()).collect();
    check_passes(&ctx, &names, &blocks, &cands, &schedules(), K3OptConfig::default());
}

/// Every synthetic block through every schedule, also with the persistent loop on 3 workgroups,
/// and the later-pass Buffer-price configs.
#[test]
fn k3opt_passes_synthetic() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let ctx = GpuContext::new().expect("GPU required");
    let all = synthetic_blocks();
    let names: Vec<String> = all.iter().map(|(n, _)| n.clone()).collect();
    let blocks: Vec<Vec<u8>> = all.into_iter().map(|(_, b)| b).collect();
    let cands = cands_of(&blocks);
    let live = check_passes(&ctx, &names, &blocks, &cands, &schedules(), K3OptConfig::default());
    eprintln!("final passes with ll[1] < ll[0]: {live}");
    // M6 A4: the persistent passes with 3 workgroups (each runs several blocks, so every
    // per-block state must follow the block, not the workgroup).
    let looped = K3OptConfig { grid: Some(3), ..K3OptConfig::default() };
    check_passes(&ctx, &names, &blocks, &cands, &schedules(), looped);
    assert!(check_later_pass_tables(&ctx, &names, &blocks, &cands) > 0, "no block exercises ll_inc1 < 0");
}

/// The persistent passes' block order (M6 A4, `k3_sched.wgsl`): each block's weight (positions
/// whose longest candidate is 3..32, in the sampled runs) and the order, blocks by
/// descending weight with ties by ascending block id. 600 blocks (3 tiles of the rank sort), each
/// synthetic block several times (ties).
#[test]
fn k3opt_block_order() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let ctx = GpuContext::new().expect("GPU required");
    let blocks: Vec<Vec<u8>> = synthetic_blocks().into_iter().map(|(_, b)| b).collect();
    let cands = cands_of(&blocks);
    let n = 600;
    let pick = |i: usize| (i * 7 + i / 5) % blocks.len();
    let refs: Vec<&[u8]> = (0..n).map(|i| blocks[pick(i)].as_slice()).collect();
    let crefs: Vec<&[CandWords]> = (0..n).map(|i| cands[pick(i)].as_slice()).collect();
    let k = K3Opt::new(&ctx, &OPT16, K3OptConfig::default()).expect("K3Opt::new");
    let bufs = OptBuffers::new(&ctx, &OPT16, n as u32).unwrap();
    bufs.upload(&ctx, &refs, &crefs, None).unwrap();
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    k.record_order(&ctx, &mut enc, &bufs.binds(), n as u32, None).unwrap();
    ctx.queue.submit([enc.finish()]);
    let got: Vec<u32> = ctx.read_buffer(&bufs.sched, 0, SCHED_HDR as usize + 3 * n);
    let weight = |c: &[CandWords]| -> u32 {
        c.iter()
            .enumerate()
            .filter(|(p, w)| {
                let (a, b) = gzc_core::reference::unpack_cands(**w);
                (*p as u32 % WEIGHT_STRIDE) < WEIGHT_RUN && (3..=32).contains(&a.len.max(b.len))
            })
            .count() as u32
    };
    let want_w: Vec<u32> = crefs.iter().map(|c| weight(c)).collect();
    let h = SCHED_HDR as usize;
    assert_eq!(&got[h..h + n], &want_w[..], "weights");
    let mut want_o: Vec<u32> = (0..n as u32).collect();
    want_o.sort_by_key(|&i| (std::cmp::Reverse(want_w[i as usize]), i));
    assert_eq!(&got[h + n..h + 2 * n], &want_o[..], "order");
    for (r, &b) in want_o.iter().enumerate() {
        assert_eq!(got[h + 2 * n + b as usize], r as u32, "rank of block {b}");
    }
    let distinct: std::collections::BTreeSet<u32> = want_w.iter().copied().collect();
    assert!(distinct.len() > 3 && distinct.len() < n, "ties and distinct weights both present");
}

/// A block for the rep-length memo's edges (M6 A2/A3): random bytes overwritten with short copies
/// at offsets from `offs` (small, close offsets, so rep histories with two equal offsets are
/// common, e.g. rep0 - 1 == rep2 under ll0), random stretches (dead positions between searches),
/// and in every segment a rep tail ending exactly at the segment's end: a copy at offset o, one
/// mismatching literal, then 9..=32 bytes at offset o up to iend (lim > 8, so the rep's length is
/// lim, and later positions of its series reuse it through the memo capped at their lim). In
/// every other segment the match also really ends at iend (the next byte differs).
fn memo_edge_block(seed: u64, offs: &[usize]) -> Vec<u8> {
    let seg = 1usize << OPT16.segment_log2;
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as usize
    };
    let mut b: Vec<u8> = (0..BLOCK_SIZE).map(|_| rnd() as u8).collect();
    let mut p = 32;
    while p < BLOCK_SIZE {
        if rnd() % 6 == 0 {
            p += 1 + rnd() % 20;
            continue;
        }
        let o = offs[rnd() % offs.len()];
        let len = 3 + rnd() % 10;
        for i in p..(p + len).min(BLOCK_SIZE) {
            b[i] = b[i - o];
        }
        p += len + rnd() % 3;
    }
    for k in 0..BLOCK_SIZE / seg {
        let iend = (k + 1) * seg;
        let t = 9 + (rnd() % 24);
        let o = offs[rnd() % offs.len()] + 16;
        let brk = iend - t - 1;
        for i in brk - 11..brk {
            b[i] = b[i - o];
        }
        b[brk] = b[brk - o] ^ 0x5A;
        for i in iend - t..iend {
            b[i] = b[i - o];
        }
        if k % 2 == 0 && iend < BLOCK_SIZE {
            b[iend] = b[iend - o] ^ 0xA5;
        }
    }
    b
}

/// Counts, over the parse `out` replayed with each segment's own rep history (segment 0 from
/// INITIAL_REPS, segment k > 0 from [0, 0, 0], as the DP sees them): matches whose search saw two
/// valid rep probes with the same offset, and rep matches longer than 8 that end exactly at
/// their segment's end.
fn memo_edge_witnesses(out: &BlockOutput) -> (usize, usize) {
    use gzc_core::seq::{INITIAL_REPS, apply_off_base, off_base_for};
    let seg = 1usize << OPT16.segment_log2;
    let (mut dup, mut tail) = (0, 0);
    let mut reps = INITIAL_REPS;
    let mut local = INITIAL_REPS;
    let (mut pos, mut cur_seg) = (0usize, 0usize);
    for s in &out.sequences {
        let start = pos + s.lit_len as usize;
        let off = apply_off_base(&mut reps, s.off_base, s.lit_len);
        let k = start / seg;
        if k != cur_seg {
            local = [0, 0, 0];
            cur_seg = k;
        }
        let ll = (start - pos.max(k * seg)) as u32;
        let ros = if ll == 0 { [local[1], local[2], local[0].wrapping_sub(1)] } else { local };
        let valid = |r: u32| r >= 1 && r as usize <= start;
        if (0..3).any(|i| (i + 1..3).any(|j| valid(ros[i]) && ros[i] == ros[j])) {
            dup += 1;
        }
        let ob = off_base_for(off, ll, &local);
        apply_off_base(&mut local, ob, ll);
        pos = start + s.match_len as usize;
        if ob <= 3 && s.match_len > 8 && pos == (k + 1) * seg {
            tail += 1;
        }
    }
    (dup, tail)
}

/// The rep-length memo's edges (M6 A2 review, A3): a rep match ending exactly at seg_end() with
/// lim > 8, and searches with two rep probes of the same offset, through single passes (both
/// levels) and the opt16 / opt14 schedules, against the oracle.
#[test]
fn k3opt_memo_edges() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let ctx = GpuContext::new().expect("GPU required");
    let offs: [&[usize]; 3] = [&[3, 4, 5], &[1, 2, 3, 4], &[3, 4, 5, 6, 8, 9, 12, 16]];
    let mut names = Vec::new();
    let mut blocks = Vec::new();
    for seed in 0..6u64 {
        for (j, o) in offs.iter().enumerate() {
            names.push(format!("memo_edge[{seed}, {j}]"));
            blocks.push(memo_edge_block(seed, o));
        }
    }
    let cands = cands_of(&blocks);
    let (mut dup, mut tail) = (0, 0);
    for (b, c) in blocks.iter().zip(&cands) {
        let (d, t) = memo_edge_witnesses(&oracle(b, c, &Prices::block_init(b), 2));
        dup += d;
        tail += t;
    }
    eprintln!("witnesses: {dup} matches after equal-offset rep probes, {tail} rep tails ending at seg_end");
    assert!(dup > 0 && tail > 0, "the blocks miss an edge: {dup} equal-offset, {tail} tails");
    let bi: Vec<Prices> = blocks.iter().map(|b| Prices::block_init(b)).collect();
    let cfgs = [cfg(2, PriceSrc::BlockInit), cfg(0, PriceSrc::BlockInit)];
    check_blocks_with(&ctx, &names, &blocks, &cands, &bi, &cfgs, None);
    let presets = [("opt16".to_string(), OPT16), ("opt14".to_string(), OPT14)];
    check_passes(&ctx, &names, &blocks, &cands, &presets, K3OptConfig::default());
}

/// Informal (reads the real corpus; skips when it is missing): 4000 corpus blocks through every
/// schedule, and the later-pass Buffer-price configs.
/// `cargo test --release -p gzc-gpu --test k3opt k3opt_passes_corpus -- --ignored --nocapture`
#[test]
#[ignore]
fn k3opt_passes_corpus() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let n: usize = std::env::var("GZC_CORPUS_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000);
    let Some(blocks) = gzc_core::testdata::corpus_sample(n) else { return };
    let ctx = GpuContext::new().expect("GPU required");
    eprintln!("subgroups: {}", ctx.subgroups);
    let names: Vec<String> = (0..blocks.len()).map(|i| format!("corpus[{i}]")).collect();
    let cands = cands_of(&blocks);
    let live = check_passes(&ctx, &names, &blocks, &cands, &schedules(), K3OptConfig::default());
    eprintln!("final passes with ll[1] < ll[0]: {live}");
    let presets = [("opt16".to_string(), OPT16), ("opt14".to_string(), OPT14)];
    // M6 A4: the persistent passes on 37 workgroups (about 7 blocks each per 256-block batch).
    let looped = K3OptConfig { grid: Some(37), ..K3OptConfig::default() };
    check_passes(&ctx, &names, &blocks, &cands, &presets, looped);
    assert!(check_later_pass_tables(&ctx, &names, &blocks, &cands) > 0, "no block exercises ll_inc1 < 0");
}

/// Informal: GPU time of every pass of opt14, opt16 and opt16p1 (its single final pass and the
/// drop pass, on its S3 candidates; `GZC_TIMING_PRESETS`, comma-separated names, picks presets),
/// `GZC_CORPUS_BLOCKS`, default 2900 corpus blocks, one batch (the persistent kernel orders
/// blocks heavy-first itself, `k3_sched.wgsl`).
/// `cargo test --release -p gzc-gpu --test k3opt k3opt_passes_timing -- --ignored --nocapture`
#[test]
#[ignore]
fn k3opt_passes_timing() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let n: usize = std::env::var("GZC_CORPUS_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2900);
    let Some(blocks) = gzc_core::testdata::corpus_sample(n) else { return };
    let ctx = GpuContext::new().expect("GPU required");
    let cands = cands_of(&blocks);
    let filter = std::env::var("GZC_TIMING_PRESETS").ok();
    let all = [("opt14", OPT14), ("opt16", OPT16), ("opt16p1", OPT16P1)];
    // In GZC_TIMING_PRESETS' order (for A/B runs that alternate the order).
    let presets: Vec<(&str, MatchParams)> = match &filter {
        Some(f) => f.split(',').filter_map(|x| all.iter().find(|(n, _)| *n == x).copied()).collect(),
        None => all.to_vec(),
    };
    let s3 = presets.iter().any(|(_, m)| m.opt.unwrap().sparse_chains != OPT16.opt.unwrap().sparse_chains);
    let cands_s3 = if s3 { cands_of_m(&blocks, &OPT16P1) } else { Vec::new() };
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let crefs: Vec<&[CandWords]> = cands.iter().map(|c| c.as_slice()).collect();
    let crefs_s3: Vec<&[CandWords]> = cands_s3.iter().map(|c| c.as_slice()).collect();
    let bufs = OptBuffers::new(&ctx, &OPT16, blocks.len() as u32).unwrap();
    let us = |ms: f64| ms * 1000.0 / blocks.len() as f64;
    // GZC_TIMING_WARMUP: one untimed opt16 run first, so the first timed preset does not pay the
    // GPU's clock ramp. (M6 A2: opt14 still shows a bimodal per-process state on an RTX 5090,
    // every pass about 10 % slower with the fix-up at 0.37 instead of 0.27 us/block; compare
    // runs in the same state.)
    if std::env::var("GZC_TIMING_WARMUP").is_ok() {
        let p = OptPasses::new(&ctx, &OPT16, K3OptConfig::default()).expect("OptPasses::new");
        time_passes(&ctx, &p, &bufs, blocks.len() as u32, 5, || {
            bufs.upload(&ctx, &refs, &crefs, None).unwrap();
        })
        .expect("time_passes");
    }
    for (name, m) in presets {
        let p = OptPasses::new(&ctx, &m, K3OptConfig::default()).expect("OptPasses::new");
        let c = if m.opt.unwrap().sparse_chains == OPT16.opt.unwrap().sparse_chains { &crefs } else { &crefs_s3 };
        let mut t = time_passes(&ctx, &p, &bufs, blocks.len() as u32, 5, || {
            bufs.upload(&ctx, &refs, c, None).unwrap();
        })
        .expect("time_passes");
        let span = t.pop().unwrap();
        let per: Vec<String> = t.iter().map(|&ms| format!("{:.2}", us(ms))).collect();
        eprintln!(
            "{name}: {} blocks of {} KiB: us/block per pass (DP..., fixup, order, drop) [{}], total {:.2} us/block ({:.3} ms), span {:.2} us/block",
            blocks.len(),
            BLOCK_SIZE / 1024,
            per.join(", "),
            us(t.iter().sum()),
            t.iter().sum::<f64>(),
            us(span)
        );
    }
}

// ---- M6 B3: opt16p1's DP options (gap3, top-4 pruning, S3 prior, one pass) and the drop pass ----

/// `MatchParams` with `OPT16P1`'s match finder and the opt params `o`.
fn on_p1(o: OptParams) -> MatchParams {
    MatchParams { opt: Some(o), ..OPT16P1 }
}

/// `MatchParams` with `OPT16`'s match finder and the opt params `o`.
fn on_16(o: OptParams) -> MatchParams {
    MatchParams { opt: Some(o), ..OPT16 }
}

/// The M6 DP options one at a time and together, for single passes (`check_blocks_m`; no drop
/// pass there): gap3, top-4 pruning, both, and top-2 (more lengths pruned).
fn m6_dp_params() -> Vec<(String, MatchParams)> {
    let p1 = OPT16P1.opt.unwrap();
    vec![
        ("gap3".into(), on_p1(OptParams { relax_lengths: None, drop_max_len: 0, ..p1 })),
        ("top4".into(), on_p1(OptParams { inner_gap: 8, drop_max_len: 0, ..p1 })),
        ("gap3+top4".into(), on_p1(OptParams { drop_max_len: 0, ..p1 })),
        ("gap3+top2".into(), on_p1(OptParams { relax_lengths: Some(2), drop_max_len: 0, ..p1 })),
    ]
}

/// The M6 schedules over `OPT16P1`'s candidates (`OptPasses`, the drop pass included): opt16p1
/// itself (Prior S3 seed, one optLevel-2 pass with gap3 and top-4, the drop pass), without the
/// drop pass, with a cheap pass first (gap3 and pruning in a `hist_out` pass), with a block-init
/// seed, and with an optLevel-0 final pass.
fn m6_schedules() -> Vec<(String, MatchParams)> {
    let p1 = OPT16P1.opt.unwrap();
    vec![
        ("opt16p1".into(), OPT16P1),
        ("opt16p1 no drop".into(), on_p1(OptParams { drop_max_len: 0, ..p1 })),
        ("opt16p1 x1".into(), on_p1(OptParams { passes: 1, ..p1 })),
        ("opt16p1 BlockInit".into(), on_p1(OptParams { seed: Seed::BlockInit, prior: PriorTables::M5, ..p1 })),
        ("opt16p1 L0 top2".into(), on_p1(OptParams { level: 0, relax_lengths: Some(2), ..p1 })),
    ]
}

/// M6 options on `OPT16`'s candidates (no sparse chains): the drop pass alone, and the S3 prior
/// with a cheap pass.
fn m6_schedules_16() -> Vec<(String, MatchParams)> {
    let o16 = OPT16.opt.unwrap();
    vec![
        ("opt16 + drop".into(), on_16(OptParams { drop_max_len: 6, ..o16 })),
        ("opt16 + S3 prior x1".into(), on_16(OptParams { seed: Seed::Prior, prior: PriorTables::S3, passes: 1, ..o16 })),
    ]
}

/// Single M6 passes (`m6_dp_params`) on `OPT16P1`'s candidates: block-init prices at optLevel 2
/// and 0, and uploaded tables.
fn check_m6_single(ctx: &GpuContext, names: &[String], blocks: &[Vec<u8>], cands: &[Vec<CandWords>]) {
    let bi: Vec<Prices> = blocks.iter().map(|b| Prices::block_init(b)).collect();
    let cfgs = [cfg(2, PriceSrc::BlockInit), cfg(0, PriceSrc::BlockInit), cfg(2, PriceSrc::Buffer)];
    for (name, m) in m6_dp_params() {
        eprintln!("-- {name}");
        check_blocks_m(ctx, &m, names, blocks, cands, &bi, &cfgs, None);
    }
}

/// The oracle frame of every block's `opt::parse` under `m` from `cands`, through `kf`'s frame
/// options.
fn oracle_frames(blocks: &[Vec<u8>], cands: &[Vec<CandWords>], m: &MatchParams, kf: &Kernels) -> Vec<Vec<u8>> {
    let opts = kf.frame_options();
    par_map(blocks, |i, b| write_frame(b, &gzc_core::opt::parse(b, &cands[i], m), opts))
}

/// `OPT16P1`'s passes and drop pass from the candidate words `cands` (the oracle's, or K1/K2opt's
/// on the GPU), frames through K4/K5, against the oracle's frames.
fn check_p1_frames(ctx: &GpuContext, kf: &Kernels, blocks: &[Vec<u8>], cands: &[Vec<CandWords>], want: &[Vec<u8>], what: &str) {
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let crefs: Vec<&[CandWords]> = cands.iter().map(|c| c.as_slice()).collect();
    let p = OptPasses::new(ctx, &OPT16P1, K3OptConfig::default()).expect("OptPasses::new");
    assert!(p.drop_pass().is_some());
    let (got, _) = parses_from_passes(ctx, &p, &refs, &crefs, false).expect("parses_from_passes");
    let mut bad = 0;
    for (chunk, (bl, g)) in refs.chunks(256).zip(got.chunks(256)).enumerate() {
        let fr = frames_from_parses(ctx, kf, bl, g).expect("frames_from_parses");
        for (j, f) in fr.iter().enumerate() {
            bad += (*f != want[chunk * 256 + j]) as usize;
        }
    }
    assert_eq!(bad, 0, "opt16p1 ({what}): {bad} of {} frames differ", blocks.len());
    eprintln!("opt16p1 ({what}): {} frames equal", blocks.len());
}

/// The frames' kernels (K4/K5 only; the parse comes from the K3opt harness).
fn frame_kernels(ctx: &GpuContext) -> Kernels {
    Kernels::new(ctx, GpuParams { matching: LVL9SEG, emit_frames: true, huffman: true }).unwrap()
}

/// `opt::cases::m6_test_cases` (gap3, top-4 pruning) with their tables, at their level; and every `opt::cases` block (with its scripted candidates) through the M6 schedules.
#[test]
fn k3opt_m6_opt_cases() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let ctx = GpuContext::new().expect("GPU required");
    for c in gzc_core::opt::cases::m6_test_cases() {
        for (i, (params, prices, want)) in c.expect.iter().enumerate() {
            let prices = prices.clone().expect("m6 cases carry their tables");
            let out = run_case(&c.block, &c.cands, params, Some(&prices), Engine::Ring);
            assert_eq!(out.sequences, *want, "{} [{i}]: oracle", c.name);
            let level = params.opt.unwrap().level;
            let k = K3Opt::new(&ctx, params, cfg(level, PriceSrc::Buffer)).expect("K3Opt::new");
            let got = parses_from_cands(&ctx, &k, &[&c.block], &[&c.cands], Some(std::slice::from_ref(&prices)))
                .expect("parses_from_cands");
            assert!(got[0] == out, "{} [{i}]: {}", c.name, first_diff(&got[0], &out));
        }
        eprintln!("{}: equal", c.name);
    }
    let cases: Vec<_> = opt_test_cases().into_iter().chain(gzc_core::opt::cases::m6_test_cases()).collect();
    let names: Vec<String> = cases.iter().map(|c| c.name.clone()).collect();
    let blocks: Vec<Vec<u8>> = cases.iter().map(|c| c.block.clone()).collect();
    let cands: Vec<Vec<CandWords>> = cases.iter().map(|c| c.cands.clone()).collect();
    let sched = [m6_schedules(), m6_schedules_16()].concat();
    check_passes(&ctx, &names, &blocks, &cands, &sched, K3OptConfig::default());
}

/// `opt::cases::drop_test_cases` on the drop kernel: each scripted input as a final pass and its
/// fix-up would leave it, with the case's prices (the kernel's test hook) and the expected output,
/// and with the prices of the input's own histogram against `opt::drop_pass`.
#[test]
fn k3drop_matches_drop_cases() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let ctx = GpuContext::new().expect("GPU required");
    let cases = gzc_core::opt::cases::drop_test_cases();
    let blocks: Vec<&[u8]> = cases.iter().map(|c| c.block.as_slice()).collect();
    let inputs: Vec<BlockOutput> = cases.iter().map(|c| c.input.clone()).collect();
    let prices: Vec<Prices> = cases.iter().map(|c| c.prices.clone()).collect();
    let m = on_p1(OptParams { drop_max_len: 6, ..OPT16P1.opt.unwrap() });
    let d = K3Drop::new(&ctx, &m, true).expect("K3Drop::new");
    let got = drops_from_parses(&ctx, &d, &blocks, &inputs, Some(&prices)).expect("drops_from_parses");
    for (c, g) in cases.iter().zip(&got) {
        assert_eq!(c.max_len, 6);
        assert!(g.sequences == c.expect, "{}: got {:?}\n    want {:?}", c.name, g.sequences, c.expect);
    }
    let d = K3Drop::new(&ctx, &m, false).expect("K3Drop::new");
    let got = drops_from_parses(&ctx, &d, &blocks, &inputs, None).expect("drops_from_parses");
    for (c, g) in cases.iter().zip(&got) {
        let want = drop_pass(&c.block, &c.input, 6, m.segment_log2);
        assert!(*g == want, "{} (own prices): {}", c.name, first_diff(g, &want));
    }
    eprintln!("{} drop cases equal", cases.len());
}

/// Every synthetic block: the single M6 passes, the M6 schedules (default and looped
/// persistent), opt16p1's frames from the oracle's candidates and from K1/K2opt's
/// (the same frames), and the memo-edge blocks through the M6 passes (gap3 searches at lim 3..7).
#[test]
fn k3opt_m6_synthetic() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let ctx = GpuContext::new().expect("GPU required");
    let all = synthetic_blocks();
    let names: Vec<String> = all.iter().map(|(n, _)| n.clone()).collect();
    let blocks: Vec<Vec<u8>> = all.into_iter().map(|(_, b)| b).collect();
    let cands = cands_of_m(&blocks, &OPT16P1);
    check_m6_single(&ctx, &names, &blocks, &cands);
    let sched = m6_schedules();
    check_passes(&ctx, &names, &blocks, &cands, &sched, K3OptConfig::default());
    let p1 = [("opt16p1".to_string(), OPT16P1)];
    check_passes(&ctx, &names, &blocks, &cands, &p1, K3OptConfig { grid: Some(3), ..K3OptConfig::default() });
    check_passes(&ctx, &names, &blocks, &cands_of(&blocks), &m6_schedules_16(), K3OptConfig::default());
    let kf = frame_kernels(&ctx);
    let want = oracle_frames(&blocks, &cands, &OPT16P1, &kf);
    check_p1_frames(&ctx, &kf, &blocks, &cands, &want, "oracle candidates");
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let gpu_cands = cands_from_blocks(&ctx, &OptCandKernel::new(&ctx, &OPT16P1).unwrap(), &refs).unwrap();
    assert!(gpu_cands == cands, "K1/K2opt candidates differ from find_cands");
    check_p1_frames(&ctx, &kf, &blocks, &gpu_cands, &want, "K1/K2opt candidates");
    let offs: [&[usize]; 3] = [&[3, 4, 5], &[1, 2, 3, 4], &[3, 4, 5, 6, 8, 9, 12, 16]];
    let (mut mnames, mut mblocks) = (Vec::new(), Vec::new());
    for seed in 0..6u64 {
        for (j, o) in offs.iter().enumerate() {
            mnames.push(format!("memo_edge[{seed}, {j}]"));
            mblocks.push(memo_edge_block(seed, o));
        }
    }
    let mcands = cands_of_m(&mblocks, &OPT16P1);
    check_m6_single(&ctx, &mnames, &mblocks, &mcands);
    check_passes(&ctx, &mnames, &mblocks, &mcands, &sched, K3OptConfig::default());
}

/// Informal (reads the real corpus; skips when it is missing): `GZC_CORPUS_BLOCKS` (default 4000)
/// corpus blocks: the single M6 passes, the M6 schedules, opt16p1 on the looped persistent passes,
/// and opt16p1's frames from the oracle's and from K1/K2opt's candidates.
/// `GZC_CORPUS=… cargo test --release -p gzc-gpu --test k3opt k3opt_m6_corpus -- --ignored --nocapture`
#[test]
#[ignore]
fn k3opt_m6_corpus() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let n: usize = std::env::var("GZC_CORPUS_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(4000);
    let Some(blocks) = gzc_core::testdata::corpus_sample(n) else { return };
    let ctx = GpuContext::new().expect("GPU required");
    eprintln!("subgroups: {}", ctx.subgroups);
    let names: Vec<String> = (0..blocks.len()).map(|i| format!("corpus[{i}]")).collect();
    let cands = cands_of_m(&blocks, &OPT16P1);
    check_m6_single(&ctx, &names, &blocks, &cands);
    check_passes(&ctx, &names, &blocks, &cands, &m6_schedules(), K3OptConfig::default());
    let p1 = [("opt16p1".to_string(), OPT16P1)];
    check_passes(&ctx, &names, &blocks, &cands, &p1, K3OptConfig { grid: Some(37), ..K3OptConfig::default() });
    check_passes(&ctx, &names, &blocks, &cands_of(&blocks), &m6_schedules_16(), K3OptConfig::default());
    let kf = frame_kernels(&ctx);
    let want = oracle_frames(&blocks, &cands, &OPT16P1, &kf);
    check_p1_frames(&ctx, &kf, &blocks, &cands, &want, "oracle candidates");
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let gpu_cands = cands_from_blocks(&ctx, &OptCandKernel::new(&ctx, &OPT16P1).unwrap(), &refs).unwrap();
    let diff = gpu_cands.iter().zip(&cands).filter(|(g, c)| g != c).count();
    assert_eq!(diff, 0, "K1/K2opt candidates differ from find_cands on {diff} blocks");
    check_p1_frames(&ctx, &kf, &blocks, &gpu_cands, &want, "K1/K2opt candidates");
    let dropped: usize = par_map(&blocks, |i, b| {
        let last = passes(b, &cands[i], &OPT16P1).pop().unwrap().out;
        last.sequences.len() - gzc_core::opt::parse(b, &cands[i], &OPT16P1).sequences.len()
    })
    .iter()
    .sum();
    eprintln!("the drop pass removed {dropped} sequences over {} blocks", blocks.len());
}
