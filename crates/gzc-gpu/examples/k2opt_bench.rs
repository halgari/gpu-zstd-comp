//! Measurement tool (M5 T2, not part of the test suite): GPU time of K1 + K2opt (`opt16`'s
//! candidates: h4 chain depth 32 + h3 chain depth 4, two words per position) against K1 + K2 of
//! `lvl9`, per block, over the corpus.
//!
//! `cargo run --release -p gzc-gpu --example k2opt_bench -- <corpus dir> [batch] [stride]`
//! Streams every `stride`-th BLOCK_SIZE block (default 1: all) of the corpus's .dds/.nif files in
//! batches of `batch` (default 2048) blocks; per batch, runs each finder `REPS` times and keeps the
//! median of its per-kernel timestamps; reports the sums over the corpus as µs per block.
use gzc_core::config::BLOCK_SIZE;
use gzc_core::params::{LVL3, LVL9, OPT16};
use gzc_gpu::chains::{head_bytes, pred_bytes};
use gzc_gpu::compressor::{BatchBuffers, GpuParams, Kernels, OptCandKernel, best_bytes_for, data_bytes};
use gzc_gpu::context::{GpuContext, pack_blocks};
use std::path::Path;

const REPS: usize = 3;

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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct Timer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
}

impl Timer {
    fn new(ctx: &GpuContext) -> Self {
        let set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 10 });
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 80,
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

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("corpus dir");
    let batch: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2048);
    let stride: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);
    let mut files = Vec::new();
    walk(Path::new(dir), &mut files);

    let ctx = GpuContext::new()?;
    eprintln!("{} (subgroups {})", ctx.adapter_info.name, ctx.subgroups);
    let timer = Timer::new(&ctx);
    let cap = batch as u32;
    // lvl9: the production K1 + K2 (Kernels, K3 recorded too but not counted).
    let lvl9 = Kernels::new(&ctx, GpuParams { matching: LVL9, emit_frames: false, huffman: false })?;
    let lvl9_bufs = BatchBuffers::new(&ctx, cap, false, &LVL9);
    // lvl3 (Dfast): the other two-chain K1, for reference.
    let lvl3 = Kernels::new(&ctx, GpuParams { matching: LVL3, emit_frames: false, huffman: false })?;
    let lvl3_bufs = BatchBuffers::new(&ctx, cap, false, &LVL3);
    // opt16: K1 (Opt3) + K2opt.
    let opt = OptCandKernel::new(&ctx, &OPT16)?;
    let data = ctx.storage_buffer("data", data_bytes(cap), false);
    let head = ctx.storage_buffer("head", head_bytes(cap, 2), false);
    let pred = ctx.storage_buffer("pred", pred_bytes(cap, 2), false);
    let cands = ctx.storage_buffer("cands", best_bytes_for(cap, &OPT16), false);

    // Sums of per-batch medians, ms: lvl9 K1, K2; opt K1, K2opt; lvl3 K1, K2.
    let mut sum = [0f64; 6];
    let mut n_blocks = 0usize;
    let mut pending: Vec<Vec<u8>> = Vec::with_capacity(batch);
    let mut i = 0usize;
    let run = |blocks: &[Vec<u8>], sum: &mut [f64; 6]| {
        let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
        let packed = pack_blocks(&refs);
        let n = blocks.len() as u32;
        ctx.queue.write_buffer(&lvl9_bufs.data, 0, bytemuck::cast_slice(&packed));
        ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&packed));
        ctx.queue.write_buffer(&lvl3_bufs.data, 0, bytemuck::cast_slice(&packed));
        let (mut a, mut b, mut c, mut d, mut e, mut f) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..REPS {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            lvl9.record_timed(&ctx, &mut enc, &lvl9_bufs, n, Some(&timer.set)).unwrap();
            let ms = timer.finish(&ctx, enc, 2);
            a.push(ms[0]);
            b.push(ms[1]);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            opt.record_timed(&ctx, &mut enc, &data, &head, &pred, &cands, n, |k| timer.ts(k));
            let ms = timer.finish(&ctx, enc, 2);
            c.push(ms[0]);
            d.push(ms[1]);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            lvl3.record_timed(&ctx, &mut enc, &lvl3_bufs, n, Some(&timer.set)).unwrap();
            let ms = timer.finish(&ctx, enc, 2);
            e.push(ms[0]);
            f.push(ms[1]);
        }
        for (s, v) in sum.iter_mut().zip([a, b, c, d, e, f]) {
            *s += median(v);
        }
    };
    for f in &files {
        let bytes = std::fs::read(f)?;
        for chunk in bytes.chunks(BLOCK_SIZE) {
            if i.is_multiple_of(stride) {
                let mut blk = vec![0u8; BLOCK_SIZE];
                blk[..chunk.len()].copy_from_slice(chunk);
                pending.push(blk);
                if pending.len() == batch {
                    run(&pending, &mut sum);
                    n_blocks += pending.len();
                    pending.clear();
                }
            }
            i += 1;
        }
    }
    if !pending.is_empty() {
        run(&pending, &mut sum);
        n_blocks += pending.len();
    }
    let us = |ms: f64| ms * 1e3 / n_blocks as f64;
    println!("{n_blocks} blocks of {} KiB, batch {batch}", BLOCK_SIZE / 1024);
    println!("lvl9  K1 {:.3} + K2    {:.3} = {:.3} us/block", us(sum[0]), us(sum[1]), us(sum[0] + sum[1]));
    println!("opt16 K1 {:.3} + K2opt {:.3} = {:.3} us/block", us(sum[2]), us(sum[3]), us(sum[2] + sum[3]));
    println!("lvl3  K1 {:.3} + K2    {:.3} = {:.3} us/block (two-chain K1 reference)", us(sum[4]), us(sum[5]), us(sum[4] + sum[5]));
    Ok(())
}
