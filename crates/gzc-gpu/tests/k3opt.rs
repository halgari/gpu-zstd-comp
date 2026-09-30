//! K3opt (M5 T3): one GPU DP pass against the CPU oracle `opt::dp_pass_with(.., Engine::Ring)`,
//! fed `reference::find_cands` words (or `opt::cases`' scripted ones) from the host.
//! M5 T4: every pass of `OptPasses` (seeds, cheap passes with their histograms, final pass)
//! against `opt::passes`.
use gzc_core::block::chunk_file;
use gzc_core::config::BLOCK_SIZE;
use gzc_core::frame::write_frame;
use gzc_core::opt::cases::{opt_test_cases, run_case};
use gzc_core::opt::{Engine, Hist, Prices, dp_pass_with, passes};
use gzc_core::params::{LVL9, MatchParams, OPT14, OPT16, OptParams, Seed};
use gzc_core::reference::{CandWords, chains, find_cands};
use gzc_core::seq::BlockOutput;
use gzc_gpu::compressor::{GpuParams, Kernels, frames_from_parses};
use gzc_gpu::context::GpuContext;
use gzc_gpu::k3opt::{
    K3Opt, K3OptConfig, OptBuffers, OptPasses, PriceSrc, RingMem, parses_from_cands,
    parses_from_passes, time_pass, time_passes, workgroup_bytes,
};

fn cfg(level: u8, ring: RingMem, prices: PriceSrc) -> K3OptConfig {
    K3OptConfig {
        level,
        ring: Some(ring),
        prices,
        ..K3OptConfig::default()
    }
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
    let o = OPT16.opt.unwrap();
    dp_pass_with(
        block,
        cands,
        &OPT16,
        prices,
        level,
        o.target_length,
        Engine::Ring,
    )
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
                    *out[i].lock().unwrap() = find_cands(b, &chains(b, &OPT16), &OPT16);
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

/// Whether `c` can run here: a forced workgroup ring needs `workgroup_bytes` within the adapter's
/// limit (wg32 at 16 KiB blocks needs about 38 KB); prints a skip message when it cannot.
fn runs_here(ctx: &GpuContext, c: &K3OptConfig) -> bool {
    let need = workgroup_bytes(&OPT16, c);
    let limit = ctx.device.limits().max_compute_workgroup_storage_size;
    let ok = c.ring != Some(RingMem::Workgroup) || need <= limit;
    if !ok {
        eprintln!("{c:?}: skipped, the workgroup ring needs {need} B > limit {limit}");
    }
    ok
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
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let crefs: Vec<&[CandWords]> = cands.iter().map(|c| c.as_slice()).collect();
    let bi = tables;
    for &c in cfgs {
        if !runs_here(ctx, &c) {
            continue;
        }
        let k = K3Opt::new(ctx, &OPT16, c).expect("K3Opt::new");
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
                            *out[i].lock().unwrap() = oracle(b, &cands[i], &bi[i], c.level);
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
            "{c:?} (ring {:?}): {} of {} blocks differ:\n{}",
            k.ring,
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
        eprintln!("{c:?} ring {:?}: {} blocks equal", k.ring, blocks.len());
    }
}

/// Every `opt::cases` run: explicit tables as given (or, for the preset runs, the oracle's final
/// pass prices), at the run's level, on both ring memories.
#[test]
fn k3opt_matches_opt_cases() {
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
    for ring in [RingMem::Workgroup, RingMem::Private] {
        for level in [0u8, 2] {
            let sel: Vec<_> = runs
                .iter()
                .filter(|r| r.3.opt.unwrap().level == level)
                .collect();
            let k =
                K3Opt::new(&ctx, &OPT16, cfg(level, ring, PriceSrc::Buffer)).expect("K3Opt::new");
            let blocks: Vec<&[u8]> = sel.iter().map(|r| r.1.as_slice()).collect();
            let cands: Vec<&[CandWords]> = sel.iter().map(|r| r.2.as_slice()).collect();
            let prices: Vec<Prices> = sel.iter().map(|r| r.4.clone()).collect();
            let got = parses_from_cands(&ctx, &k, &blocks, &cands, Some(&prices))
                .expect("parses_from_cands");
            for (r, g) in sel.iter().zip(&got) {
                assert!(
                    *g == r.5,
                    "{} level {level} {ring:?}: {}",
                    r.0,
                    first_diff(g, &r.5)
                );
            }
            eprintln!("{ring:?} level {level}: {} case runs equal", sel.len());
        }
    }
}

/// Every synthetic block: the block-init pass at optLevel 2 and 0 (prices computed on the GPU
/// and uploaded), on the workgroup ring and the private-memory fallback; frames through K4/K5.
#[test]
fn k3opt_matches_oracle_synthetic() {
    let ctx = GpuContext::new().expect("GPU required");
    let all = synthetic_blocks();
    let names: Vec<String> = all.iter().map(|(n, _)| n.clone()).collect();
    let blocks: Vec<Vec<u8>> = all.into_iter().map(|(_, b)| b).collect();
    let kf = Kernels::new(
        &ctx,
        GpuParams {
            matching: LVL9,
            emit_frames: true,
            huffman: true,
        },
    )
    .unwrap();
    let cfgs = [
        cfg(2, RingMem::Workgroup, PriceSrc::BlockInit),
        cfg(0, RingMem::Workgroup, PriceSrc::BlockInit),
        cfg(2, RingMem::Private, PriceSrc::BlockInit),
        cfg(0, RingMem::Private, PriceSrc::BlockInit),
        cfg(2, RingMem::Workgroup, PriceSrc::Buffer),
        K3OptConfig {
            wg: 32,
            ..cfg(2, RingMem::Workgroup, PriceSrc::BlockInit)
        },
        K3OptConfig {
            wg: 8,
            ..cfg(2, RingMem::Workgroup, PriceSrc::BlockInit)
        },
        K3OptConfig {
            wg: 8,
            ..cfg(0, RingMem::Private, PriceSrc::Buffer)
        },
        K3OptConfig {
            unbounded: false,
            ..cfg(2, RingMem::Workgroup, PriceSrc::BlockInit)
        },
    ];
    check_blocks(&ctx, &names, &blocks, &cfgs[..1], Some(&kf));
    check_blocks(&ctx, &names, &blocks, &cfgs[1..], None);
}

/// Up to `n` corpus blocks sampled uniformly over the whole corpus (`GZC_CORPUS`, default
/// `data/corpus`): every k-th of all (file, block) pairs, files in sorted path order,
/// k = total blocks / n. `None` (after a message) when the corpus directory does not exist.
fn corpus_blocks(n: usize) -> Option<Vec<Vec<u8>>> {
    use std::path::{Path, PathBuf};
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            let ext = p
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            if p.is_dir() {
                walk(&p, out);
            } else if matches!(ext.as_str(), "dds" | "nif") {
                out.push(p);
            }
        }
    }
    let root = std::env::var("GZC_CORPUS")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/corpus").to_string());
    if !Path::new(&root).is_dir() {
        eprintln!("corpus directory {root} not found (set GZC_CORPUS): skipping");
        return None;
    }
    let mut files = Vec::new();
    walk(Path::new(&root), &mut files);
    files.sort();
    let counts: Vec<usize> = files
        .iter()
        .map(|f| (std::fs::metadata(f).unwrap().len() as usize).div_ceil(BLOCK_SIZE))
        .collect();
    let total: usize = counts.iter().sum();
    let step = (total / n.max(1)).max(1);
    let mut blocks = Vec::new();
    let mut first = 0usize;
    for (f, &c) in files.iter().zip(&counts) {
        if blocks.len() >= n {
            break;
        }
        let picked: Vec<usize> = (first.div_ceil(step) * step..first + c)
            .step_by(step)
            .map(|g| g - first)
            .collect();
        if !picked.is_empty() {
            let chunks = chunk_file(&std::fs::read(f).unwrap());
            assert_eq!(chunks.len(), c, "{}", f.display());
            for i in picked {
                blocks.push(chunks[i].data.clone());
            }
        }
        first += c;
    }
    blocks.truncate(n);
    eprintln!(
        "{} corpus blocks (every {step}th of {total}) from {} files",
        blocks.len(),
        files.len()
    );
    Some(blocks)
}

/// Informal (reads the real corpus): 4000 blocks spread over `data/corpus`, block-init pass at
/// optLevel 2 and 0 on the workgroup ring, level 2 on the private ring, frames via K4/K5.
/// `cargo test --release -p gzc-gpu --test k3opt k3opt_corpus -- --ignored --nocapture`
#[test]
#[ignore]
fn k3opt_corpus() {
    let ctx = GpuContext::new().expect("GPU required");
    let n: usize = std::env::var("GZC_CORPUS_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000);
    let Some(blocks) = corpus_blocks(n) else { return };
    let names: Vec<String> = (0..blocks.len()).map(|i| format!("corpus[{i}]")).collect();
    let kf = Kernels::new(
        &ctx,
        GpuParams {
            matching: LVL9,
            emit_frames: true,
            huffman: true,
        },
    )
    .unwrap();
    check_blocks(
        &ctx,
        &names,
        &blocks,
        &[cfg(2, RingMem::Workgroup, PriceSrc::BlockInit)],
        Some(&kf),
    );
    check_blocks(
        &ctx,
        &names,
        &blocks,
        &[
            cfg(0, RingMem::Workgroup, PriceSrc::BlockInit),
            cfg(2, RingMem::Private, PriceSrc::BlockInit),
            K3OptConfig { wg: 32, ..cfg(2, RingMem::Workgroup, PriceSrc::Buffer) },
        ],
        None,
    );
}

/// Informal: GPU time of one K3opt pass per block on corpus blocks (`GZC_CORPUS_BLOCKS`, default
/// 2048, one batch), for the occupancy variants.
/// `cargo test --release -p gzc-gpu --test k3opt k3opt_timing -- --ignored --nocapture`
#[test]
#[ignore]
fn k3opt_timing() {
    let ctx = GpuContext::new().expect("GPU required");
    let n: usize = std::env::var("GZC_CORPUS_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let Some(blocks) = corpus_blocks(n) else { return };
    let cands = cands_of(&blocks);
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let crefs: Vec<&[CandWords]> = cands.iter().map(|c| c.as_slice()).collect();
    let bi: Vec<Prices> = blocks.iter().map(|b| Prices::block_init(b)).collect();
    let bufs = OptBuffers::new(&ctx, &OPT16, blocks.len() as u32);
    // (wg, ring, level, prices, unbounded loops)
    let table: [(u32, RingMem, u8, PriceSrc, bool); 13] = [
        (16, RingMem::Workgroup, 2, PriceSrc::BlockInit, true),
        (16, RingMem::Workgroup, 2, PriceSrc::Buffer, true),
        (16, RingMem::Workgroup, 0, PriceSrc::Buffer, true),
        (32, RingMem::Workgroup, 2, PriceSrc::BlockInit, true),
        (32, RingMem::Workgroup, 2, PriceSrc::Buffer, true),
        (32, RingMem::Workgroup, 0, PriceSrc::Buffer, true),
        (64, RingMem::Workgroup, 2, PriceSrc::Buffer, true),
        (8, RingMem::Workgroup, 2, PriceSrc::Buffer, true),
        (8, RingMem::Private, 2, PriceSrc::Buffer, true),
        (16, RingMem::Private, 2, PriceSrc::Buffer, true),
        (32, RingMem::Private, 2, PriceSrc::Buffer, true),
        (64, RingMem::Private, 2, PriceSrc::Buffer, true),
        (16, RingMem::Workgroup, 2, PriceSrc::Buffer, false),
    ];
    let variants: Vec<(String, K3OptConfig)> = table
        .iter()
        .map(|&(wg, ring, level, prices, unbounded)| {
            let name = format!(
                "wg{wg} ring={} L{level} {}{}",
                if ring == RingMem::Workgroup { "wg" } else { "private" },
                if prices == PriceSrc::Buffer { "buffer" } else { "blockinit" },
                if unbounded { "" } else { " checked" }
            );
            let c = K3OptConfig { wg, ring: Some(ring), level, prices, unbounded, hist_out: false };
            (name, c)
        })
        .collect();
    let filter = std::env::var("GZC_K3OPT_VARIANTS").ok();
    for (name, c) in variants {
        if let Some(f) = &filter
            && !f.split(',').any(|s| name == s)
        {
            continue;
        }
        let k = match K3Opt::new(&ctx, &OPT16, c) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("{name}: {e}");
                continue;
            }
        };
        let prices = (c.prices == PriceSrc::Buffer).then_some(bi.as_slice());
        let (main, fix) = time_pass(&ctx, &k, &bufs, blocks.len() as u32, 5, || {
            bufs.upload(&ctx, &refs, &crefs, prices).unwrap();
        })
        .expect("time_pass");
        let us = |ms: f64| ms * 1000.0 / blocks.len() as f64;
        eprintln!(
            "{name}: {} blocks of {} KiB: DP {main:.3} ms ({:.2} us/block), fixup {fix:.3} ms ({:.2} us/block)",
            blocks.len(),
            BLOCK_SIZE / 1024,
            us(main),
            us(fix)
        );
    }
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
            if got[i] != last.out {
                bad.push(format!("{}: final parse: {}", names[i], first_diff(&got[i], &last.out)));
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
        eprintln!(
            "{sname} wg{} ring {:?}: {} blocks equal ({} passes, histograms included)",
            base.wg,
            p.kernels().last().unwrap().ring,
            blocks.len(),
            p.n_passes()
        );
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
    let ctx = GpuContext::new().expect("GPU required");
    let cases = opt_test_cases();
    let names: Vec<String> = cases.iter().map(|c| c.name.clone()).collect();
    let blocks: Vec<Vec<u8>> = cases.iter().map(|c| c.block.clone()).collect();
    let cands: Vec<Vec<CandWords>> = cases.iter().map(|c| c.cands.clone()).collect();
    check_passes(&ctx, &names, &blocks, &cands, &schedules(), K3OptConfig::default());
}

/// Every synthetic block through every schedule (wg16, the default ring), opt16 / opt14 on the
/// private ring and at wg32, and the later-pass Buffer-price configs.
#[test]
fn k3opt_passes_synthetic() {
    let ctx = GpuContext::new().expect("GPU required");
    let all = synthetic_blocks();
    let names: Vec<String> = all.iter().map(|(n, _)| n.clone()).collect();
    let blocks: Vec<Vec<u8>> = all.into_iter().map(|(_, b)| b).collect();
    let cands = cands_of(&blocks);
    let live = check_passes(&ctx, &names, &blocks, &cands, &schedules(), K3OptConfig::default());
    eprintln!("final passes with ll[1] < ll[0]: {live}");
    let presets = [("opt16".to_string(), OPT16), ("opt14".to_string(), OPT14)];
    check_passes(
        &ctx,
        &names,
        &blocks,
        &cands,
        &presets,
        K3OptConfig {
            ring: Some(RingMem::Private),
            ..K3OptConfig::default()
        },
    );
    let wg32 = K3OptConfig {
        wg: 32,
        ..K3OptConfig::default()
    };
    check_passes(&ctx, &names, &blocks, &cands, &presets, wg32);
    assert!(check_later_pass_tables(&ctx, &names, &blocks, &cands) > 0, "no block exercises ll_inc1 < 0");
}

/// Informal (reads the real corpus; skips when it is missing): 4000 corpus blocks through every
/// schedule, and the later-pass Buffer-price configs.
/// `cargo test --release -p gzc-gpu --test k3opt k3opt_passes_corpus -- --ignored --nocapture`
#[test]
#[ignore]
fn k3opt_passes_corpus() {
    let n: usize = std::env::var("GZC_CORPUS_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000);
    let Some(blocks) = corpus_blocks(n) else { return };
    let ctx = GpuContext::new().expect("GPU required");
    eprintln!("subgroups: {}", ctx.subgroups);
    let names: Vec<String> = (0..blocks.len()).map(|i| format!("corpus[{i}]")).collect();
    let cands = cands_of(&blocks);
    let live = check_passes(&ctx, &names, &blocks, &cands, &schedules(), K3OptConfig::default());
    eprintln!("final passes with ll[1] < ll[0]: {live}");
    assert!(check_later_pass_tables(&ctx, &names, &blocks, &cands) > 0, "no block exercises ll_inc1 < 0");
}

/// Informal: GPU time of every pass of opt14 and opt16 (`GZC_CORPUS_BLOCKS`, default 2900 corpus
/// blocks, one batch; wg16, the default ring).
/// `cargo test --release -p gzc-gpu --test k3opt k3opt_passes_timing -- --ignored --nocapture`
#[test]
#[ignore]
fn k3opt_passes_timing() {
    let n: usize = std::env::var("GZC_CORPUS_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2900);
    let Some(blocks) = corpus_blocks(n) else { return };
    let ctx = GpuContext::new().expect("GPU required");
    let cands = cands_of(&blocks);
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let crefs: Vec<&[CandWords]> = cands.iter().map(|c| c.as_slice()).collect();
    let bufs = OptBuffers::new(&ctx, &OPT16, blocks.len() as u32);
    let us = |ms: f64| ms * 1000.0 / blocks.len() as f64;
    for (name, m) in [("opt14", OPT14), ("opt16", OPT16)] {
        let p = OptPasses::new(&ctx, &m, K3OptConfig::default()).expect("OptPasses::new");
        let mut t = time_passes(&ctx, &p, &bufs, blocks.len() as u32, 5, || {
            bufs.upload(&ctx, &refs, &crefs, None).unwrap();
        })
        .expect("time_passes");
        let span = t.pop().unwrap();
        let per: Vec<String> = t.iter().map(|&ms| format!("{:.2}", us(ms))).collect();
        eprintln!(
            "{name}: {} blocks of {} KiB: us/block per pass (DP..., fixup) [{}], total {:.2} us/block ({:.3} ms), span {:.2} us/block",
            blocks.len(),
            BLOCK_SIZE / 1024,
            per.join(", "),
            us(t.iter().sum()),
            t.iter().sum::<f64>(),
            us(span)
        );
    }
}
