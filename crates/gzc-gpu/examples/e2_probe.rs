//! speed2 E2 step 1: K2 over a host-built bucket-sorted candidate array vs K1 + K2 over hash chains.
//!
//! `cargo run --release -p gzc-gpu --example e2_probe -- <corpus dir> [stride] [blocks]`
//! Takes every `stride`-th 128 KiB block of the corpus's .dds/.nif files (default 20, up to
//! `blocks` = 2559), builds the sorted arrays on the CPU (`gzc_core::hash::bucket_sort`), checks the
//! GPU window K2 against `gzc_core::reference::find_best`, and times each kernel (median of 5).
use gzc_core::config::{BLOCK_SIZE, HASHED_POSITIONS, PARSE_END};
use gzc_core::hash::bucket_sort;
use gzc_core::params::{LVL9, LVL9D16, LVL9S13, MatchParams};
use gzc_core::reference::{chains, find_best};
use gzc_gpu::chains::{ChainsKernel, finder_wgsl, head_bytes, pred_bytes, pred_fp};
use gzc_gpu::compressor::{BEST_OFF_BITS, decode_best};
use gzc_gpu::context::{GpuContext, pack_blocks};
use std::path::Path;

const K2_WGSL: &str = include_str!("../src/shaders/k2_best.wgsl");
const K2W_WGSL: &str = include_str!("../src/shaders/k2_window.wgsl");

fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk(&p, out);
        } else if matches!(p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(), Some("dds" | "nif")) {
            out.push(p);
        }
    }
}

fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let chunk = items.len().div_ceil(n).max(1);
    std::thread::scope(|s| {
        let hs: Vec<_> = items.chunks(chunk).map(|c| s.spawn(|| c.iter().map(&f).collect::<Vec<R>>())).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

struct Timer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
}

impl Timer {
    fn new(ctx: &GpuContext) -> Self {
        let set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 8 });
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self { set, resolve }
    }
    fn ts(&self, k: u32) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(2 * k),
            end_of_pass_write_index: Some(2 * k + 1),
        })
    }
    /// ms of passes 0..n after `enc` ran.
    fn finish(&self, ctx: &GpuContext, mut enc: wgpu::CommandEncoder, n: u32) -> Vec<f64> {
        enc.resolve_query_set(&self.set, 0..2 * n, &self.resolve, 0);
        ctx.queue.submit([enc.finish()]);
        let v: Vec<u64> = ctx.read_buffer(&self.resolve, 0, 2 * n as usize);
        let period = ctx.queue.get_timestamp_period() as f64;
        (0..n as usize).map(|k| (v[2 * k + 1] - v[2 * k]) as f64 * period / 1e6).collect()
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn k2_pipeline(ctx: &GpuContext, p: &MatchParams, window: bool) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[entry(0, true), entry(1, true), entry(2, false)],
    });
    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
    let src = format!("{}const BEST_OFF_BITS: u32 = {BEST_OFF_BITS}u;\n{K2_WGSL}\n{}", finder_wgsl(p), if window { K2W_WGSL } else { "" });
    let module = ctx.shader("k2", &src);
    let pipe = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pl),
        module: &module,
        entry_point: Some(if window { "main_window" } else { "main" }),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipe, layout)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("corpus dir");
    let stride: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(20);
    let want_blocks: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(2559);
    let mut files = Vec::new();
    walk(Path::new(dir), &mut files);
    let mut blocks: Vec<Vec<u8>> = Vec::new();
    let mut i = 0usize;
    'outer: for f in &files {
        let data = std::fs::read(f)?;
        for c in data.chunks(BLOCK_SIZE) {
            if i % stride == 0 {
                let mut b = vec![0u8; BLOCK_SIZE];
                b[..c.len()].copy_from_slice(c);
                blocks.push(b);
                if blocks.len() == want_blocks {
                    break 'outer;
                }
            }
            i += 1;
        }
    }
    let n = blocks.len() as u32;
    eprintln!("{n} blocks");
    let ctx = GpuContext::new()?;
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let packed = pack_blocks(&refs);
    let data = ctx.storage_buffer("data", (packed.len() * 4) as u64, false);
    ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&packed));
    let pred = ctx.storage_buffer("pred", pred_bytes(n, 1), true);
    let best = ctx.storage_buffer("best", pred_bytes(n, 1), true);
    let rank_src = ctx.storage_buffer("rank", pred_bytes(n, 1), true);
    let head = ctx.storage_buffer("head", head_bytes(n, 1), false);
    let timer = Timer::new(&ctx);
    let bind = |layout: &wgpu::BindGroupLayout| {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pred.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: best.as_entire_binding() },
            ],
        })
    };

    for (name, p) in [("lvl9", LVL9), ("lvl9d16", LVL9D16), ("lvl9s13", LVL9S13), ("s12", MatchParams { hash_bits: 12, ..LVL9S13 }), ("s13d16", MatchParams { depth: 16, ..LVL9S13 }), ("s12d16", MatchParams { hash_bits: 12, depth: 16, ..LVL9S13 })] {
        // CPU reference (checked for every block).
        let want: Vec<Vec<gzc_core::reference::Match>> = par_map(&blocks, |b| find_best(b, &chains(b, &p), &p));
        // Chain path: K1 (GPU) + K2.
        let k1 = ChainsKernel::new(&ctx, &p)?;
        let (k2, k2l) = k2_pipeline(&ctx, &p, false);
        let bg = bind(&k2l);
        let mut t1 = Vec::new();
        let mut t2 = Vec::new();
        for _ in 0..5 {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            k1.record_timed(&ctx, &mut enc, &data, &head, &pred, n, timer.ts(0));
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: timer.ts(1) });
                pass.set_pipeline(&k2);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups((BLOCK_SIZE / 256) as u32, n, 1);
            }
            let ms = timer.finish(&ctx, enc, 2);
            t1.push(ms[0]);
            t2.push(ms[1]);
        }
        let got: Vec<u32> = ctx.read_buffer(&best, 0, n as usize * BLOCK_SIZE);
        let ok = got.chunks_exact(BLOCK_SIZE).zip(&want).all(|(g, w)| decode_best(g) == *w);
        println!("{name:8} chains: K1 {:6.2} ms  K2 {:6.2} ms  (match {ok})", median(t1), median(t2));

        // Window path: host-built sorted array + rank.
        let arrays: Vec<(Vec<u32>, Vec<u32>)> = par_map(&blocks, |b| {
            let (sorted, rank) = bucket_sort(b, &p);
            let words: Vec<u32> = (0..BLOCK_SIZE)
                .map(|s| if s < HASHED_POSITIONS { sorted[s] | pred_fp(b, sorted[s] as usize) } else { 0 })
                .collect();
            let rank: Vec<u32> = (0..BLOCK_SIZE).map(|p| if p < PARSE_END { rank[p] } else { 0 }).collect();
            (words, rank)
        });
        let words: Vec<u32> = arrays.iter().flat_map(|a| a.0.iter().copied()).collect();
        let ranks: Vec<u32> = arrays.iter().flat_map(|a| a.1.iter().copied()).collect();
        ctx.queue.write_buffer(&pred, 0, bytemuck::cast_slice(&words));
        ctx.queue.write_buffer(&rank_src, 0, bytemuck::cast_slice(&ranks));
        let (k2w, k2wl) = k2_pipeline(&ctx, &p, true);
        let bgw = bind(&k2wl);
        let mut tw = Vec::new();
        for _ in 0..5 {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&rank_src, 0, &best, 0, pred_bytes(n, 1));
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: timer.ts(0) });
                pass.set_pipeline(&k2w);
                pass.set_bind_group(0, &bgw, &[]);
                pass.dispatch_workgroups((BLOCK_SIZE / 256) as u32, n, 1);
            }
            tw.push(timer.finish(&ctx, enc, 1)[0]);
        }
        let got: Vec<u32> = ctx.read_buffer(&best, 0, n as usize * BLOCK_SIZE);
        let ok = got.chunks_exact(BLOCK_SIZE).zip(&want).all(|(g, w)| decode_best(g) == *w);
        println!("{name:8} window: K2 {:6.2} ms  (match {ok})", median(tw));

        // GPU sorted K1 + window K2.
        if let Some(k1s) = gzc_gpu::sorted::SortKernel::new(&ctx, &p)? {
            let mut t1 = Vec::new();
            let mut t2 = Vec::new();
            for _ in 0..5 {
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                k1s.record_timed(&ctx, &mut enc, &data, &pred, &best, n, timer.ts(0));
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: timer.ts(1) });
                    pass.set_pipeline(&k2w);
                    pass.set_bind_group(0, &bgw, &[]);
                    pass.dispatch_workgroups((BLOCK_SIZE / 256) as u32, n, 1);
                }
                let ms = timer.finish(&ctx, enc, 2);
                t1.push(ms[0]);
                t2.push(ms[1]);
            }
            let got: Vec<u32> = ctx.read_buffer(&best, 0, n as usize * BLOCK_SIZE);
            let ok = got.chunks_exact(BLOCK_SIZE).zip(&want).all(|(g, w)| decode_best(g) == *w);
            let sw: Vec<u32> = ctx.read_buffer(&pred, 0, n as usize * BLOCK_SIZE);
            let ok_sorted = sw.chunks_exact(BLOCK_SIZE).zip(&words.chunks_exact(BLOCK_SIZE).collect::<Vec<_>>()).all(|(g, w)| g[..HASHED_POSITIONS] == w[..HASHED_POSITIONS]);
            println!("{name:8} sorted: K1 {:6.2} ms  K2 {:6.2} ms  (match {ok}, sorted {ok_sorted})", median(t1), median(t2));
        }
    }
    Ok(())
}
