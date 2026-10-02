//! Host side of the bucket-sorted K1 (speed2 E2, `k1_sort_sg.wgsl`) and its window K2
//! (`k2_window.wgsl`).
//!
//! For Single-hash params with a key of at most `MAX_SORT_KEY_BITS` bits (`lvl9s12`, `lvl9s12seg`,
//! `lvl9s12d16seg`), K1 builds, per block, every hashed position ordered
//! by key and position (`gzc_core::hash::bucket_sort`) into the `pred` buffer: a counting sort in
//! workgroup memory (one 32-lane subgroup per block) ranks the positions into `best` (scratch),
//! then a block-major scatter places them. K2 then walks, for each slot, the entries just below
//! it that share its key, which is its position's hash chain. Both are byte-identical to
//! `gzc_core::reference::find_best`. VRAM is unchanged (the chain kernels' `head` buffer is left
//! unused).
//!
//! K1 has a subgroup version (`k1_sort_sg.wgsl`: ballots and shuffles, subgroups of at least 32
//! lanes) and a workgroup-memory version without subgroups (`k1_sort.wgsl`), picked like the chain
//! K1's two kernels: the subgroup one when the device has suitable subgroups and it passes its
//! self-test. Each needs its `workgroup_bytes` within the device's limit (the context keeps wgpu's
//! default 16 KiB: a 12-bit key fits both versions; a 13-bit key fits only the subgroup
//! version); otherwise (or
//! with `GZC_SORTED=0`) `Kernels` runs the chain kernels, which build the same chains over the
//! same key (byte-identical, just slower: K2 walks the denser chains of the shorter key).
use crate::chains::finder_wgsl;
use crate::context::{GpuContext, pack_blocks};
use gzc_core::config::{BLOCK_SIZE, HASH_BITS, HASHED_POSITIONS};
use gzc_core::params::{Hashes, MatchParams};

const K1_SORT_WGSL: &str = include_str!("shaders/k1_sort.wgsl");
const K1_SORT_SG_WGSL: &str = include_str!("shaders/k1_sort_sg.wgsl");

/// Widest key the sorted K1 handles (2^13 counters: 16 KiB of workgroup memory, wgpu's default
/// limit).
pub const MAX_SORT_KEY_BITS: u32 = 13;

/// Workgroup memory of the sorted K1's counters: 2^hash_bits of 16 bits.
pub fn table_bytes(p: &MatchParams) -> u32 {
    2 << p.hash_bits
}

/// All the workgroup memory of the sorted K1's `main_sg` (`subgroups`: the counters only) or
/// `main` (plus its two 32-word tile arrays, `tkey` and `tbase`). `SortKernel::new` builds a
/// version only when this fits the device's `max_compute_workgroup_storage_size`.
pub fn workgroup_bytes(p: &MatchParams, subgroups: bool) -> u32 {
    table_bytes(p) + if subgroups { 0 } else { 2 * 32 * 4 }
}

/// Whether `p` is a preset the sorted finder serves: a Single hash with a key of at most
/// `MAX_SORT_KEY_BITS` bits.
pub fn sorted_params(p: &MatchParams) -> bool {
    p.hashes == Hashes::Single && p.hash_bits <= MAX_SORT_KEY_BITS
}

/// The sorted K1 pipeline for one `MatchParams`.
pub struct SortKernel {
    pipeline: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    subgroups: bool,
}

impl SortKernel {
    /// Builds and self-tests the kernel; `Ok(None)` when `params` or the device do not suit it
    /// (see the module doc) or `GZC_SORTED=0`. The subgroup version runs when `ctx.subgroups`, the
    /// subgroups have at least 32 lanes and its self-test passes; otherwise the workgroup-memory
    /// version (whose failed build or self-test, reported on stderr, also gives `None`).
    pub fn new(ctx: &GpuContext, params: &MatchParams) -> anyhow::Result<Option<Self>> {
        if !sorted_params(params) || crate::context::env_off("GZC_SORTED") {
            return Ok(None);
        }
        let limit = ctx.device.limits().max_compute_workgroup_storage_size;
        if ctx.subgroups && ctx.adapter_info.subgroup_min_size >= 32 && workgroup_bytes(params, true) <= limit {
            let built = crate::compressor::with_error_scopes(ctx, || {
                let k = Self::build(ctx, params, true);
                k.self_test(ctx, params)?;
                Ok(k)
            });
            match built {
                Ok(k) => return Ok(Some(k)),
                Err(e) => eprintln!("gzc-gpu: sorted K1 subgroup kernel failed to build or its self-test ({e}); using the workgroup-memory one"),
            }
        }
        if workgroup_bytes(params, false) > limit {
            return Ok(None);
        }
        let built = crate::compressor::with_error_scopes(ctx, || {
            let k = Self::build(ctx, params, false);
            k.self_test(ctx, params)?;
            Ok(k)
        });
        match built {
            Ok(k) => Ok(Some(k)),
            Err(e) => {
                eprintln!("gzc-gpu: sorted K1 failed to build or its self-test ({e}); using the chain K1");
                Ok(None)
            }
        }
    }

    /// True when this is the subgroup version (`main_sg`), false for the workgroup-memory one.
    pub fn uses_subgroups(&self) -> bool {
        self.subgroups
    }

    fn build(ctx: &GpuContext, params: &MatchParams, subgroups: bool) -> Self {
        let entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("k1_sort"),
            entries: &[entry(0, true), entry(1, false), entry(2, false)],
        });
        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("k1_sort"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let consts = format!("{}const KEY_BITS: u32 = {}u;\n", finder_wgsl(params), params.hash_bits);
        // naga (wgpu 30) takes subgroup operations from Features::SUBGROUP and rejects the
        // `enable subgroups;` directive.
        let module = if subgroups {
            ctx.shader("k1_sort_sg", &format!("{consts}{K1_SORT_WGSL}\n{K1_SORT_SG_WGSL}"))
        } else {
            ctx.shader("k1_sort", &format!("{consts}{K1_SORT_WGSL}"))
        };
        let make = |entry_point| {
            ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("k1_sort"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: ctx.compilation_options(),
                cache: None,
            })
        };
        let (pipeline, scatter) = (make(if subgroups { "main_sg" } else { "main" }), make("scatter"));
        Self { pipeline, scatter, layout, subgroups }
    }

    /// Sorts small-alphabet, text, texture-like, constant and random blocks and compares every
    /// written slot with `gzc_core::hash::bucket_sort` (position and fingerprint bits).
    fn self_test(&self, ctx: &GpuContext, params: &MatchParams) -> anyhow::Result<()> {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let alphabet: Vec<u8> = (0..BLOCK_SIZE)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 32) % 3) as u8
            })
            .collect();
        let blocks = [
            alphabet,
            gzc_core::synth::text(11, BLOCK_SIZE),
            gzc_core::synth::dds_like(12, BLOCK_SIZE),
            gzc_core::synth::zeros(BLOCK_SIZE),
            gzc_core::synth::random(13, BLOCK_SIZE),
        ];
        let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
        let n = blocks.len() as u32;
        let packed = pack_blocks(&refs);
        let data = ctx.storage_buffer("k1_sort.selftest.data", (packed.len() * 4) as u64, false);
        ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&packed));
        let sorted = ctx.storage_buffer("k1_sort.selftest.sorted", n as u64 * BLOCK_SIZE as u64 * 4, true);
        let rank = ctx.storage_buffer("k1_sort.selftest.rank", n as u64 * BLOCK_SIZE as u64 * 4, false);
        for round in 0..2 {
            let got = self.run(ctx, &data, &sorted, &rank, n);
            for (b, block) in blocks.iter().enumerate() {
                if got[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE][..HASHED_POSITIONS] != sorted_words(block, params)[..HASHED_POSITIONS] {
                    anyhow::bail!("sorted array of self-test block {b} differs from the CPU in round {round}");
                }
            }
        }
        Ok(())
    }

    /// Records K1, submits it and reads the sorted words of `n_blocks` blocks back.
    pub fn run(&self, ctx: &GpuContext, data: &wgpu::Buffer, sorted: &wgpu::Buffer, rank: &wgpu::Buffer, n_blocks: u32) -> Vec<u32> {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("k1_sort") });
        self.record_timed(ctx, &mut enc, data, sorted, rank, n_blocks, None);
        ctx.queue.submit([enc.finish()]);
        ctx.read_buffer(sorted, 0, BLOCK_SIZE * n_blocks as usize)
    }

    /// Records K1 for `n_blocks` blocks of `data` into `sorted` (at least `n_blocks * BLOCK_SIZE`
    /// words, layout `[block][slot]`), using `rank` (as large; `[block][pos]`, scratch: the
    /// pipeline passes `best`, which K2 overwrites) for each position's slot. Both dispatches
    /// run in one compute pass writing `timestamp_writes` (if any).
    #[allow(clippy::too_many_arguments)]
    pub fn record_timed(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        data: &wgpu::Buffer,
        sorted: &wgpu::Buffer,
        rank: &wgpu::Buffer,
        n_blocks: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites>,
    ) {
        if n_blocks == 0 {
            return;
        }
        assert!(sorted.size() >= n_blocks as u64 * BLOCK_SIZE as u64 * 4, "sorted buffer too small");
        assert!(rank.size() >= n_blocks as u64 * BLOCK_SIZE as u64 * 4, "rank buffer too small");
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k1_sort"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sorted.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: rank.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k1_sort"), timestamp_writes });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_blocks, 1, 1);
        pass.set_pipeline(&self.scatter);
        pass.dispatch_workgroups((BLOCK_SIZE / 256) as u32, n_blocks, 1);
    }
}

/// The sorted words K1 writes for `block` (`BLOCK_SIZE` long; slots `HASHED_POSITIONS..` are 0
/// here and not written by K1): slot s holds `q | pred_fp(block, q)` for `q = bucket_sort`'s
/// position in slot s.
pub fn sorted_words(block: &[u8], params: &MatchParams) -> Vec<u32> {
    let (sorted, _) = gzc_core::hash::bucket_sort(block, params);
    (0..BLOCK_SIZE)
        .map(|s| if s < HASHED_POSITIONS { sorted[s] | crate::chains::pred_fp(block, sorted[s] as usize) } else { 0 })
        .collect()
}

const _: () = assert!(MAX_SORT_KEY_BITS <= HASH_BITS);
