//! Measurement tool (M6 B2, not part of the test suite): interleaved A/B GPU time of the candidate
//! stage, K1 + K2opt, of `opt16` (h4 d32 + h3 d4) against `opt16p1` (S3: h4 d8 + h3 d4 + three
//! stride-4 sparse chains d16), per block, on the same corpus blocks.
//!
//! `cargo run --release -p gzc-gpu --example cands_ab -- <corpus dir> [blocks] [rounds] [reps]`
//! Takes `blocks` (default 2900) BLOCK_SIZE blocks sampled uniformly over the corpus's .dds/.nif
//! files (every k-th block, files in sorted path order) as one batch (split if the device limits
//! require). Each round runs A then B, or B then A on odd rounds, `reps` times each (default 5)
//! and keeps each kernel's median; prints every round's µs/block and B/A ratios, then the median
//! ratio over the rounds. `GZC_AB_A` / `GZC_AB_B` name other presets.
use gzc_core::config::BLOCK_SIZE;
use gzc_core::params::{MatchParams, preset};
use gzc_gpu::chains::{chain_pred_bytes, head_bytes};
use gzc_gpu::compressor::{OptCandKernel, best_bytes_for, data_bytes, max_batch_blocks};
use gzc_gpu::context::{GpuContext, pack_blocks};
use std::path::Path;

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
        let set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 4 });
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
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
    /// ms of passes 0 and 1 after `enc` ran.
    fn finish(&self, ctx: &GpuContext, mut enc: wgpu::CommandEncoder) -> [f64; 2] {
        enc.resolve_query_set(&self.set, 0..4, &self.resolve, 0);
        ctx.queue.submit([enc.finish()]);
        let v: Vec<u64> = ctx.read_buffer(&self.resolve, 0, 4);
        let period = ctx.queue.get_timestamp_period() as f64;
        [(v[1] - v[0]) as f64 * period / 1e6, (v[3] - v[2]) as f64 * period / 1e6]
    }
}

struct Side {
    name: String,
    kernel: OptCandKernel,
    head: wgpu::Buffer,
    pred: wgpu::Buffer,
    cands: wgpu::Buffer,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("corpus dir");
    let want: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2900);
    let rounds: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(6);
    let reps: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(5);
    let mut files = Vec::new();
    walk(Path::new(dir), &mut files);
    let counts: Vec<usize> = files.iter().map(|f| (std::fs::metadata(f).unwrap().len() as usize).div_ceil(BLOCK_SIZE)).collect();
    let total: usize = counts.iter().sum();
    let step = (total / want).max(1);
    let mut blocks: Vec<Vec<u8>> = Vec::with_capacity(want);
    let mut first = 0usize;
    for (f, &n) in files.iter().zip(&counts) {
        if blocks.len() >= want {
            break;
        }
        let picked: Vec<usize> = (first.div_ceil(step) * step..first + n).step_by(step).map(|g| g - first).collect();
        if !picked.is_empty() {
            let bytes = std::fs::read(f)?;
            for i in picked {
                let chunk = &bytes[i * BLOCK_SIZE..bytes.len().min((i + 1) * BLOCK_SIZE)];
                let mut blk = vec![0u8; BLOCK_SIZE];
                blk[..chunk.len()].copy_from_slice(chunk);
                blocks.push(blk);
            }
        }
        first += n;
    }
    blocks.truncate(want);

    let ctx = GpuContext::new()?;
    let timer = Timer::new(&ctx);
    let name_a = std::env::var("GZC_AB_A").unwrap_or_else(|_| "opt16".into());
    let name_b = std::env::var("GZC_AB_B").unwrap_or_else(|_| "opt16p1".into());
    let params: Vec<(String, MatchParams)> =
        [name_a, name_b].into_iter().map(|n| (n.clone(), preset(&n).unwrap())).collect();
    let limits = ctx.device.limits();
    let batch = params.iter().map(|(_, m)| max_batch_blocks(&limits, m)).min().unwrap().min(blocks.len() as u32) as usize;
    let data = ctx.storage_buffer("data", data_bytes(batch as u32), false);
    let sides: Vec<Side> = params
        .iter()
        .map(|(name, m)| {
            let cap = batch as u32;
            Side {
                name: name.clone(),
                kernel: OptCandKernel::new(&ctx, m).unwrap(),
                head: ctx.storage_buffer("head", head_bytes(cap, m.n_hashes()), false),
                pred: ctx.storage_buffer("pred", chain_pred_bytes(cap, m), false),
                cands: ctx.storage_buffer("cands", best_bytes_for(cap, m), false),
            }
        })
        .collect();
    eprintln!(
        "{} (subgroups {}, K1 sg {}/{}); {} blocks (every {step}th of {total}), batch {batch}",
        ctx.adapter_info.name,
        ctx.subgroups,
        sides[0].kernel.uses_subgroups(),
        sides[1].kernel.uses_subgroups(),
        blocks.len()
    );
    // Per round: [side][k1, k2] µs/block.
    let mut per_round: Vec<[[f64; 2]; 2]> = Vec::new();
    for round in 0..rounds {
        let order: [usize; 2] = if round % 2 == 0 { [0, 1] } else { [1, 0] };
        let mut ms = [[0f64; 2]; 2];
        for chunk in blocks.chunks(batch) {
            let refs: Vec<&[u8]> = chunk.iter().map(|b| b.as_slice()).collect();
            ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&pack_blocks(&refs)));
            let n = chunk.len() as u32;
            for &s in &order {
                let side = &sides[s];
                let (mut k1, mut k2) = (Vec::new(), Vec::new());
                for _ in 0..reps {
                    let mut enc = ctx.device.create_command_encoder(&Default::default());
                    side.kernel.record_timed(&ctx, &mut enc, &data, &side.head, &side.pred, &side.cands, n, |k| timer.ts(k));
                    let t = timer.finish(&ctx, enc);
                    k1.push(t[0]);
                    k2.push(t[1]);
                }
                ms[s][0] += median(k1);
                ms[s][1] += median(k2);
            }
        }
        let us = ms.map(|x| x.map(|v| v * 1e3 / blocks.len() as f64));
        println!(
            "round {round}: {} K1 {:.3} K2 {:.3} sum {:.3} | {} K1 {:.3} K2 {:.3} sum {:.3} | B/A K1 {:.3} K2 {:.3} sum {:.3}",
            sides[0].name,
            us[0][0],
            us[0][1],
            us[0][0] + us[0][1],
            sides[1].name,
            us[1][0],
            us[1][1],
            us[1][0] + us[1][1],
            us[1][0] / us[0][0],
            us[1][1] / us[0][1],
            (us[1][0] + us[1][1]) / (us[0][0] + us[0][1])
        );
        per_round.push(us);
    }
    let med = |f: &dyn Fn(&[[f64; 2]; 2]) -> f64| median(per_round.iter().map(f).collect());
    println!(
        "median over {rounds} rounds: {} K1 {:.3} K2 {:.3} | {} K1 {:.3} K2 {:.3} | B/A K1 {:.3} K2 {:.3} sum {:.3}",
        sides[0].name,
        med(&|u| u[0][0]),
        med(&|u| u[0][1]),
        sides[1].name,
        med(&|u| u[1][0]),
        med(&|u| u[1][1]),
        med(&|u| u[1][0] / u[0][0]),
        med(&|u| u[1][1] / u[0][1]),
        med(&|u| (u[1][0] + u[1][1]) / (u[0][0] + u[0][1]))
    );
    Ok(())
}
