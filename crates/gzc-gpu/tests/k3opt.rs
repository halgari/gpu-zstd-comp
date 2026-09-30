//! K3opt (M5 T3): one GPU DP pass against the CPU oracle `opt::dp_pass_with(.., Engine::Ring)`,
//! fed `reference::find_cands` words (or `opt::cases`' scripted ones) from the host.
use gzc_core::block::chunk_file;
use gzc_core::config::BLOCK_SIZE;
use gzc_core::frame::write_frame;
use gzc_core::opt::cases::{opt_test_cases, run_case};
use gzc_core::opt::{Engine, Prices, dp_pass_with, passes};
use gzc_core::params::{LVL9, MatchParams, OPT16};
use gzc_core::reference::{CandWords, chains, find_cands};
use gzc_core::seq::BlockOutput;
use gzc_gpu::compressor::{GpuParams, Kernels, frames_from_parses};
use gzc_gpu::context::GpuContext;
use gzc_gpu::k3opt::{
    K3Opt, K3OptConfig, OptBuffers, PriceSrc, RingMem, parses_from_cands, time_pass,
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
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let crefs: Vec<&[CandWords]> = cands.iter().map(|c| c.as_slice()).collect();
    let bi: Vec<Prices> = blocks.iter().map(|b| Prices::block_init(b)).collect();
    for &c in cfgs {
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
        let prices = (c.prices == PriceSrc::Buffer).then_some(bi.as_slice());
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

fn corpus_blocks(n: usize) -> Vec<Vec<u8>> {
    use std::path::{Path, PathBuf};
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let mut ents: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        ents.sort();
        for p in ents {
            let e = p
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            if p.is_dir() {
                walk(&p, out);
            } else if e == "dds" || e == "nif" {
                out.push(p);
            }
        }
    }
    let root = std::env::var("GZC_CORPUS")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/corpus").to_string());
    let mut files = Vec::new();
    walk(Path::new(&root), &mut files);
    let mut blocks = Vec::new();
    for f in files {
        blocks.extend(
            chunk_file(&std::fs::read(&f).unwrap())
                .into_iter()
                .map(|b| b.data),
        );
    }
    let stride = (blocks.len() / n).max(1);
    blocks.into_iter().step_by(stride).take(n).collect()
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
    let blocks = corpus_blocks(n);
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
    let blocks = corpus_blocks(n);
    let cands = cands_of(&blocks);
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let crefs: Vec<&[CandWords]> = cands.iter().map(|c| c.as_slice()).collect();
    let bi: Vec<Prices> = blocks.iter().map(|b| Prices::block_init(b)).collect();
    let bufs = OptBuffers::new(&ctx, blocks.len() as u32);
    let variants: Vec<(&str, K3OptConfig)> = vec![
        (
            "wg32 ring=wg L2 blockinit",
            cfg(2, RingMem::Workgroup, PriceSrc::BlockInit),
        ),
        (
            "wg32 ring=wg L2 buffer",
            cfg(2, RingMem::Workgroup, PriceSrc::Buffer),
        ),
        (
            "wg32 ring=wg L0 buffer",
            cfg(0, RingMem::Workgroup, PriceSrc::Buffer),
        ),
        (
            "wg16 ring=wg L2 buffer",
            K3OptConfig {
                wg: 16,
                ..cfg(2, RingMem::Workgroup, PriceSrc::Buffer)
            },
        ),
        (
            "wg64 ring=wg L2 buffer",
            K3OptConfig {
                wg: 64,
                ..cfg(2, RingMem::Workgroup, PriceSrc::Buffer)
            },
        ),
        (
            "wg32 ring=private L2 buffer",
            cfg(2, RingMem::Private, PriceSrc::Buffer),
        ),
        (
            "wg64 ring=private L2 buffer",
            K3OptConfig {
                wg: 64,
                ..cfg(2, RingMem::Private, PriceSrc::Buffer)
            },
        ),
        (
            "wg16 ring=private L2 buffer",
            K3OptConfig {
                wg: 16,
                ..cfg(2, RingMem::Private, PriceSrc::Buffer)
            },
        ),
        (
            "wg8 ring=wg L2 buffer",
            K3OptConfig {
                wg: 8,
                ..cfg(2, RingMem::Workgroup, PriceSrc::Buffer)
            },
        ),
        (
            "wg8 ring=private L2 buffer",
            K3OptConfig {
                wg: 8,
                ..cfg(2, RingMem::Private, PriceSrc::Buffer)
            },
        ),
        (
            "wg16 ring=wg L0 buffer",
            K3OptConfig {
                wg: 16,
                ..cfg(0, RingMem::Workgroup, PriceSrc::Buffer)
            },
        ),
        (
            "wg16 ring=wg L2 blockinit",
            K3OptConfig {
                wg: 16,
                ..cfg(2, RingMem::Workgroup, PriceSrc::BlockInit)
            },
        ),
        (
            "wg32 ring=wg L2 buffer checked",
            K3OptConfig {
                unbounded: false,
                ..cfg(2, RingMem::Workgroup, PriceSrc::Buffer)
            },
        ),
    ];
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
