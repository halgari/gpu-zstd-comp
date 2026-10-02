//! Host side of the kernels: K1/K2 (match finding), K3 (parse), K5 (Huffman literals) and K4
//! (entropy coding and frame assembly), and the per-batch buffers they run on.
//!
//! K1 (hash chains) → K2 (`find_best`) → K3 (`reference::parse`: greedy, or
//! `lazy::lazy_parse_segmented` when `MatchParams::lazy > 0`) reproduce
//! `gzc_core::reference::compress_block` exactly, for a batch of BLOCK_SIZE blocks. K5 writes each
//! block's literals section into its frame and K4 completes the frame, byte-identical to
//! `gzc_core::frame::write_frame` with `GpuParams::frame_options()`: Huffman literals
//! (`FrameOptions::default()`) with `huffman`, raw literals (still written by K5) without.
//!
//! K3 writes only the sequences and counts; the literals are the block bytes the sequences leave
//! uncovered, so K5 gathers them from `data` and the parse path from the host's copy of the block
//! (`decode_output`).
use crate::chains::{self, ChainsKernel, finder_wgsl, layout_wgsl};
use crate::k3opt::{K3OptConfig, OptBinds, OptPasses};
use crate::context::{ErrorScopes, GpuContext, K3Kernel, params_wgsl};
use crate::error::{Kind, invalid_input, tagged};
use crate::sizing::{BufferSizes, best_bytes, counts_bytes, data_bytes, frame_len_bytes, max_batch_blocks};
use crate::sorted::SortKernel;
use gzc_core::codes::{
    LL_BASE, LL_BITS, LL_DEFAULT_NORM, ML_BASE, ML_BITS, ML_DEFAULT_NORM, OF_DEFAULT_NORM, ll_code, ml_code,
};
use gzc_core::config::BLOCK_SIZE;
use gzc_core::frame::{FrameOptions, frame_header};
use gzc_core::fse::FRAC;
use gzc_core::params::{MatchParams, OPT_H3_DEPTH};
use gzc_core::seq::{BlockOutput, Sequence};

const K2_WGSL: &str = include_str!("shaders/k2_best.wgsl");
const K2_WINDOW_WGSL: &str = include_str!("shaders/k2_window.wgsl");
const K2_OPT_WGSL: &str = include_str!("shaders/k2_opt.wgsl");
/// The repeat-offset history (`r0..r2`) and `off_base_for` / `apply_off_base`, shared by the
/// sequential parse and the segmented fix-up.
const K3_REPS_WGSL: &str = include_str!("shaders/k3_reps.wgsl");
const K3_WGSL: &str = include_str!("shaders/k3_parse.wgsl");
const K3_COOP_WGSL: &str = include_str!("shaders/k3_coop.wgsl");
const K3_SEG_WGSL: &str = include_str!("shaders/k3_seg.wgsl");
/// `main_fixup` and the rep helpers (`K3_REPS_WGSL`), shared by `k3_seg.wgsl` and `k3_opt.wgsl`.
pub(crate) const K3_FIXUP_WGSL: &str =
    concat!(include_str!("shaders/k3_reps.wgsl"), "\n", include_str!("shaders/k3_fixup.wgsl"));
/// K3t: cuts a partial block's parse to its real length (frame path, batches with one only).
const K3_TRUNC_WGSL: &str = include_str!("shaders/k3_trunc.wgsl");
const K4_WGSL: &str = include_str!("shaders/k4_seq_entropy.wgsl");
const K5_WGSL: &str = include_str!("shaders/k5_huffman.wgsl");

/// Bytes reserved per block in the `frames` buffer: a Raw frame (header + 3 + BLOCK_SIZE) is the
/// largest K4 emits.
pub(crate) const FRAME_STRIDE: usize = BLOCK_SIZE + 64;

/// Shortest sequence (`MatchParams::min_seq_len`) any GPU-supported preset emits except the
/// optimal parse (3, `MAX_SEQS_OPT`): the smallest allowed `min_match`. `MAX_SEQS` is sized for it
/// and `check_matching` rejects params that could emit shorter ones than their `max_seqs`.
pub(crate) const MAX_SEQS_MIN_SEQ_LEN: usize = 4;

/// Upper bound on sequences per block: every sequence covers at least `MAX_SEQS_MIN_SEQ_LEN` bytes.
/// 16385 at 64 KiB, below 0x7F00, so K4 never emits the 3-byte nbSeq header.
pub(crate) const MAX_SEQS: u32 = (BLOCK_SIZE / MAX_SEQS_MIN_SEQ_LEN) as u32 + 1;

/// Upper bound on sequences per block of the optimal parse (`MatchParams::opt`, min match 3):
/// `BLOCK_SIZE / 3 + 1` (21846 at 64 KiB). The `seqs` buffer, the parse readback's stride and
/// K3opt/K5/K4's `MAX_SEQS` constant use it for opt params (`max_seqs`); every other preset keeps
/// `MAX_SEQS`.
pub(crate) const MAX_SEQS_OPT: u32 = (BLOCK_SIZE / 3) as u32 + 1;

// K4's 3-byte nbSeq form (nbSeq >= 0x7F00 = 32512) cannot fire: 64 KiB blocks hold at most
// 21846 sequences. K4 still implements it; its header encoding is tested at the CPU level
// (`gzc_core::seqenc::nbseq_header_forms`).
const _: () = assert!(MAX_SEQS < 0x7F00 && MAX_SEQS_OPT < 0x7F00, "blocks never reach the 3-byte nbSeq form");

/// Sequences per block the `seqs` buffer holds under match params `m`: `MAX_SEQS_OPT` for the
/// optimal parse (min match 3), else `MAX_SEQS`.
pub fn max_seqs(m: &MatchParams) -> u32 {
    if m.opt.is_some() { MAX_SEQS_OPT } else { MAX_SEQS }
}

/// Kernel names, in timestamp-query order, as reported in timing breakdowns (`k4_entropy` and
/// `k5_huffman` only run with `GpuParams::emit_frames`; K5, which writes the literals section
/// (Huffman-coded only with `huffman`), is dispatched before K4; see `Kernels::names`).
pub(crate) const KERNEL_NAMES: [&str; 5] = ["k1_chains", "k2_best", "k3_parse", "k4_entropy", "k5_huffman"];

/// `KERNEL_NAMES` when K1/K2 run the bucket-sorted finder (`Kernels::uses_sorted_finder`).
pub(crate) const SORTED_KERNEL_NAMES: [&str; 5] = ["k1_sort", "k2_window", "k3_parse", "k4_entropy", "k5_huffman"];

/// Timestamp queries `Kernels::record_timed` may write: a begin/end pair per kernel.
pub(crate) const KERNEL_QUERIES: u32 = 2 * KERNEL_NAMES.len() as u32;

/// K3t's name in timing breakdowns (`Kernels::record_truncate`). It is not one of `KERNEL_NAMES`:
/// it runs only for a batch that holds a partial block, with timestamp writes its caller places.
pub(crate) const TRUNC_KERNEL_NAME: &str = "k3_trunc";

/// Parameters for the GPU path: the match finder / parse (`params::LVL3` etc., same meaning
/// as for `reference::compress_block`) and which output stages run.
#[derive(Clone, Copy, Debug)]
pub struct GpuParams {
    pub matching: MatchParams,
    /// Also run K4, which turns each block's parse into its complete zstd frame.
    pub emit_frames: bool,
    /// With `emit_frames`: K5 Huffman-codes the literals (`FrameOptions::default()`); without it
    /// the frames keep raw literals (`huffman: false`).
    pub huffman: bool,
}

/// Whether the GPU kernels implement `p`: every valid `MatchParams` (both hash modes; greedy
/// over the whole block; lazy and lazy2 in segments only, `segment_log2 > 0`: an unsegmented lazy
/// parse is refused and runs on the CPU oracle alone; the M5 optimal parse, presets
/// `opt14`/`opt16`, with K2opt and the K3opt passes since M5 T5), and the
/// M6 opt options (preset `opt16p1`, M6 B4): the S3 prior tables, gap3, top-N pruning and the
/// drop pass at every value `validate` allows (K3opt and `K3Drop` compile them in), and sparse
/// chains of stride 4 or 8 (`chains::long_chains_supported`: K1 hashes word-aligned slots).
/// Valid sparse chains of stride 1 or 2 are refused.
pub fn gpu_supports(p: &MatchParams) -> bool {
    p.validate().is_ok() && !unsegmented_lazy(p) && p.opt.is_none_or(|_| chains::long_chains_supported(p))
}

/// A lazy parse over the whole block: valid for the CPU oracle, not implemented on the GPU.
fn unsegmented_lazy(p: &MatchParams) -> bool {
    p.opt.is_none() && p.lazy > 0 && p.segment_log2 == 0
}

/// Ok when `m` is valid, implemented on the GPU and its sequences fit `max_seqs(m)`.
pub(crate) fn check_matching(m: &MatchParams) -> anyhow::Result<()> {
    m.validate().map_err(|e| invalid_input(format!("invalid match params {m:?}: {e}")))?;
    if unsegmented_lazy(m) {
        return Err(tagged(
            Kind::Unsupported,
            format!(
                "match params {m:?} are not implemented on gpu: a lazy parse (lazy {}) needs segments (segment_log2 > 0)",
                m.lazy
            ),
        ));
    }
    if !gpu_supports(m) {
        return Err(tagged(Kind::Unsupported, format!("match params {m:?} are not implemented yet on gpu")));
    }
    let max = max_seqs(m);
    anyhow::ensure!(
        max as usize * m.min_seq_len() as usize >= BLOCK_SIZE,
        "min_seq_len {} needs more than {max} sequences per block",
        m.min_seq_len()
    );
    Ok(())
}

impl GpuParams {
    /// The `write_frame` options the frames equal (checksums stay CPU-only).
    pub fn frame_options(&self) -> FrameOptions {
        FrameOptions { checksum: false, huffman: self.huffman }
    }
}

/// Low bits of a `best[]` word holding the match offset; the (capped) length sits above them.
pub(crate) const BEST_OFF_BITS: u32 = 17;
const _: () = assert!(BLOCK_SIZE <= 1 << BEST_OFF_BITS, "offsets must fit BEST_OFF_BITS");
// MatchParams::validate bounds search_cap to 8..=256.
const _: () = assert!(256 < 1u64 << (32 - BEST_OFF_BITS), "capped lengths must fit above the offset");

/// Per-batch buffers, allocated for `capacity` blocks and reused.
pub struct BatchBuffers {
    pub capacity: u32,
    /// Hash chains per block the head/pred buffers hold (`MatchParams::n_hashes`).
    pub n_hashes: u32,
    /// Packed blocks (`pack_blocks` layout); written by the caller.
    pub data: wgpu::Buffer,
    /// K1 scratch hash-head tables, `[table < chains::HEAD_TABLES][2^HASH_BITS]`
    /// (`sizing::head_bytes`: one table per chain, at most `HEAD_TABLES`).
    pub head: wgpu::Buffer,
    /// K1 predecessor chains, `[block][chain][pos]`.
    pub pred: wgpu::Buffer,
    /// K2 output, `[block][pos]` × `(capped len << BEST_OFF_BITS) | offset`; 0 = none. For `opt`
    /// params K2opt's `[block][pos]` × 2 candidate words (`best_bytes_for`).
    pub best: wgpu::Buffer,
    /// K3 output, `[block][MAX_SEQS]` × (lit_len, match_len, off_base).
    pub seqs: wgpu::Buffer,
    /// K3 output, `[block]` × (n_seq, n_lit).
    pub counts: wgpu::Buffer,
    /// K4 output, `[block][FRAME_STRIDE]` frame bytes (allocated only with `frames: true`).
    pub frames: Option<wgpu::Buffer>,
    /// K4 output, `[block]` frame length in bytes (allocated only with `frames: true`).
    pub frame_len: Option<wgpu::Buffer>,
    /// K3t input, `[block]` real length in bytes (allocated only with `frames: true`; see
    /// `Kernels::record_truncate`).
    pub lens: Option<wgpu::Buffer>,
    /// Optimal parse only (`m.opt`): K3opt's own buffers.
    pub opt: Option<OptScratch>,
}

/// K3opt's own buffers in `BatchBuffers`. Its other bindings are the shared buffers: `best` holds
/// K2opt's candidate words, `pred` (dead once K2opt has read the chains) the DP trace.
pub struct OptScratch {
    /// `k3opt::PRICE_WORDS` words per block: the cheap passes' histograms.
    pub prices: wgpu::Buffer,
    /// The DP nodes' payload (`k3opt::scratch_bytes_per_block` per block).
    pub scratch: wgpu::Buffer,
    /// Bytes of `scratch` per block (for the params it was sized for).
    pub scratch_per_block: u64,
    /// The persistent passes' block schedule (`sizing::sched_bytes`, M6 A4).
    pub sched: wgpu::Buffer,
}

impl BatchBuffers {
    /// Buffers for kernels built with match params `m` (head/pred sized by `m.n_hashes()`).
    /// Panics unless `1 <= capacity <= max_batch_blocks(&ctx.device.limits(), m)`. `frames`
    /// allocates K4's outputs (needed when the kernels emit frames). Errors when an allocation
    /// fails (out of memory, a buffer above the device's limits, or a lost device).
    pub fn new(ctx: &GpuContext, capacity: u32, frames: bool, m: &MatchParams) -> anyhow::Result<Self> {
        Self::with_parts(ctx, capacity, frames, m, None, None)
    }

    /// `new`, taking `data` and (with `frames`) the `frames` / `frame_len` pair from the caller
    /// when given, so only the missing buffers are allocated: `Pipeline` brings a slot's upload
    /// buffer as `data` (direct upload) and buffers shared with the transfer queue as the frame
    /// outputs (transfer readback). The given buffers must be at least `data_bytes(capacity)`,
    /// `frames_bytes(capacity)` and `frame_len_bytes(capacity)` bytes.
    pub(crate) fn with_parts(
        ctx: &GpuContext,
        capacity: u32,
        frames: bool,
        m: &MatchParams,
        data: Option<wgpu::Buffer>,
        frame_bufs: Option<(wgpu::Buffer, wgpu::Buffer)>,
    ) -> anyhow::Result<Self> {
        let max = max_batch_blocks(&ctx.device.limits(), m);
        assert!(capacity >= 1 && capacity <= max, "BatchBuffers capacity {capacity} not in 1..={max}");
        let size = BufferSizes::new(capacity, m);
        let bytes = size.scratch()
            + if data.is_none() { size.data } else { 0 }
            + if frames && frame_bufs.is_none() { size.frames + size.frame_len } else { 0 }
            + if frames { size.lens } else { 0 };
        let scopes = ErrorScopes::push(ctx);
        let n_hashes = m.n_hashes();
        let (frames, frame_len) = match (frames, frame_bufs) {
            (false, _) => (None, None),
            (true, Some((f, l))) => (Some(f), Some(l)),
            (true, None) => (
                Some(ctx.storage_buffer("batch.frames", size.frames, true)),
                Some(ctx.storage_buffer("batch.frame_len", size.frame_len, true)),
            ),
        };
        let lens = frame_len.is_some().then(|| ctx.storage_buffer("batch.lens", size.lens, false));
        let opt = m.opt.is_some().then(|| {
            let scratch_per_block = crate::k3opt::scratch_bytes_per_block(m);
            OptScratch {
                prices: ctx.storage_buffer("batch.opt_prices", size.opt_prices, true),
                scratch: ctx.storage_buffer("batch.opt_scratch", size.opt_scratch, false),
                scratch_per_block,
                sched: ctx.storage_buffer("batch.opt_sched", size.opt_sched, false),
            }
        });
        // K3opt's `ld32` reads one word past a block's last word, so `data` keeps its trailing
        // zero word (`data_bytes` = n * BLOCK_SIZE + 4), also when the caller brings it (the
        // pipeline's direct-upload / zero-copy slots are `data_bytes(batch)` with that word
        // zeroed on every submit).
        let data = data.unwrap_or_else(|| ctx.storage_buffer("batch.data", size.data, false));
        assert!(data.size() >= size.data, "data buffer below data_bytes({capacity})");
        let bufs = Self {
            capacity,
            n_hashes,
            data,
            head: ctx.storage_buffer("batch.head", size.head, false),
            pred: ctx.storage_buffer("batch.pred", size.pred, true),
            best: ctx.storage_buffer("batch.best", size.best, true),
            seqs: ctx.storage_buffer("batch.seqs", size.seqs, true),
            counts: ctx.storage_buffer("batch.counts", size.counts, true),
            frames,
            frame_len,
            lens,
            opt,
        };
        scopes.pop_alloc(&format!("the batch buffers ({capacity} blocks)"), bytes)?;
        Ok(bufs)
    }
}

/// K4 pipeline and its constant tables.
struct EntropyKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// `tab`: code tables, FRAC and the predefined distributions (see `k4_tables`).
    tables: wgpu::Buffer,
}

/// K5 pipeline (no buffers of its own).
struct HuffmanKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// The K1, K2 and K3 pipelines (plus K5 and K4 with `emit_frames`), built once.
pub struct Kernels {
    chains: ChainsKernel,
    /// The bucket-sorted K1 and the window K2 (`sorted::sorted_params` presets, when the device
    /// runs them), used instead of `chains` and `best`.
    sorted: Option<(SortKernel, wgpu::ComputePipeline)>,
    /// K2: `k2_best.wgsl`'s `main`, or K2opt (`k2_opt.wgsl`) for the optimal parse.
    best: wgpu::ComputePipeline,
    best_layout: wgpu::BindGroupLayout,
    /// The sequential K3: the unsegmented greedy parse (None for the segmented and the optimal
    /// parse, whose K3 is `parse_seg` / `opt`).
    parse: Option<wgpu::ComputePipeline>,
    parse_layout: wgpu::BindGroupLayout,
    /// The subgroup-cooperative K3 (`K3Mode::Coop`), used instead of `parse` when present.
    parse_coop: Option<wgpu::ComputePipeline>,
    /// The segmented K3 (`MatchParams::segment_log2 > 0`), used instead of both when present.
    parse_seg: Option<SegParse>,
    /// How the unsegmented greedy K3 runs; None when K3 is `parse_seg` or `opt`.
    k3_mode: Option<K3Mode>,
    /// The optimal parse (`MatchParams::opt`, M5): the K3opt passes (`k3opt::OptPasses`), run as
    /// K3 instead of every parse above, on the shared buffers (`OptBinds::of_batch`).
    opt: Option<OptPasses>,
    entropy: Option<EntropyKernel>,
    huffman: Option<HuffmanKernel>,
    /// K3t (`k3_trunc.wgsl`) and its layout, with `emit_frames`.
    trunc: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    params: GpuParams,
}

/// K3 for a segmented parse (`k3_seg.wgsl`): one lane per segment, then one workgroup per block
/// concatenating the segments and encoding the offsets. Its `best` binding is read-write: the
/// segment lanes keep their raw sequences in their own part of `best` (K3 is its last reader).
struct SegParse {
    seg: wgpu::ComputePipeline,
    fixup: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Segments per block.
    n_seg: u32,
}

// k3_seg.wgsl keeps each segment's sequences and trailer in that segment's own `best` words:
// every block needs BLOCK_SIZE words there.
const _: () = assert!(best_bytes(1) >= 4 * BLOCK_SIZE as u64, "k3_seg needs BLOCK_SIZE best words per block");

/// Lanes per workgroup of `k3_seg.wgsl`'s `main_seg` (32: 4 % faster K3 than 64 or 128 on an
/// RTX 5090 at 64 KiB blocks).
const K3_SEG_WG: u32 = 32;
/// Positions `k3_seg.wgsl`'s literal scan tests per step (8: K3 40 % faster than with 1; 4 and
/// 12-16 are slower).
const K3_SEG_SCAN: u32 = 8;

/// K4's `tab` buffer contents and the WGSL constants locating each table in it. Every value
/// comes from gzc_core, so the GPU mirrors the CPU tables exactly.
/// Bytes of K4's constant table buffer.
pub(crate) fn k4_tables_bytes() -> u64 {
    k4_tables(MAX_SEQS).0.len() as u64 * 4
}

/// Frame-header options: the header does not depend on `huffman`.
const HEADER_OPTIONS: FrameOptions = FrameOptions { checksum: false, huffman: true };

/// `max_seqs`: the `seqs` stride per block (`max_seqs(m)`), injected as K4/K5's `MAX_SEQS`.
fn k4_tables(max_seqs: u32) -> (Vec<u32>, String) {
    let mut tab: Vec<u32> = Vec::new();
    let mut consts = String::new();
    let mut add = |name: &str, values: Vec<u32>| {
        consts += &format!("const TAB_{name}: u32 = {}u;\n", tab.len());
        tab.extend(values);
    };
    add("LL_CODE", (0..64).map(|l| ll_code(l) as u32).collect());
    add("ML_CODE", (0..128).map(|b| ml_code(b + 3) as u32).collect());
    add("LL_BITS", LL_BITS.iter().map(|&b| b as u32).collect());
    add("ML_BITS", ML_BITS.iter().map(|&b| b as u32).collect());
    add("LL_BASE", LL_BASE.to_vec());
    add("ML_BASE", ML_BASE.to_vec());
    add("FRAC", FRAC.iter().map(|&f| f as u32).collect());
    let norm = |n: &[i16]| n.iter().map(|&v| v as i32 as u32).collect();
    add("LL_NORM", norm(&LL_DEFAULT_NORM));
    add("OF_NORM", norm(&OF_DEFAULT_NORM));
    add("ML_NORM", norm(&ML_DEFAULT_NORM));

    let hdr = frame_header(HEADER_OPTIONS);
    // K4 writes frame words [0, 4) byte by byte: header + block header + a literals header of up
    // to 3 bytes must fit in 16 bytes. Its `hdr_byte` rebuilds the header of any content size
    // from this one: magic, descriptor, 2-byte content size (`frame_header_for`).
    assert_eq!(hdr.len(), 7, "frame header layout K4 expects");
    assert_eq!(&hdr[5..], &((BLOCK_SIZE - 256) as u16).to_le_bytes(), "2-byte FCS");
    let mut hw = [0u8; 12];
    hw[..hdr.len()].copy_from_slice(&hdr);
    let w = |i: usize| u32::from_le_bytes(hw[4 * i..4 * i + 4].try_into().unwrap());
    consts += &format!(
        "const HDR_LEN: u32 = {}u;\nconst HDR_W0: u32 = 0x{:08X}u;\nconst HDR_W1: u32 = 0x{:08X}u;\nconst HDR_W2: u32 = 0x{:08X}u;\n",
        hdr.len(),
        w(0),
        w(1),
        w(2)
    );
    consts += &format!("const MAX_SEQS: u32 = {max_seqs}u;\nconst FRAME_WORDS: u32 = {}u;\n", FRAME_STRIDE / 4);
    (tab, consts)
}

pub(crate) fn storage_layout(ctx: &GpuContext, label: &str, read_only: &[bool]) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = read_only
        .iter()
        .enumerate()
        .map(|(i, &read_only)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &entries })
}

fn compute_pipeline(ctx: &GpuContext, label: &str, layout: &wgpu::BindGroupLayout, body: &str) -> wgpu::ComputePipeline {
    // K2, K4 and K5: loops terminate and indices stay in bounds for any input (`shader_trusted`).
    pipeline_from_module(ctx, label, layout, &ctx.shader_trusted(label, body), "main")
}

pub(crate) fn pipeline_from_module(
    ctx: &GpuContext,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: ctx.compilation_options(),
        cache: None,
    })
}

/// How K3 runs the unsegmented greedy parse (speed phase S3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum K3Mode {
    /// `k3_parse.wgsl`: one lane per block.
    Seq,
    /// `k3_coop.wgsl`: one workgroup of `w` lanes, a single subgroup, per block (needs
    /// `Features::SUBGROUP`).
    Coop { w: u32 },
}

/// Ballot bits (x, y) of `w` active lanes 0..w.
fn lane_mask(w: u32) -> (u32, u32) {
    match w {
        64 => (u32::MAX, u32::MAX),
        32 => (u32::MAX, 0),
        _ => ((1 << w) - 1, 0),
    }
}

/// The K3 mode on `ctx`. Devices without subgroups use `Seq`.
/// Otherwise `Coop` with W = the adapter's minimum subgroup size (clamped to 8..=64, a power of
/// two), provided a one-time probe confirms that a workgroup of W lanes is one subgroup with lane
/// ids 0..W-1 (else `Seq`; `GpuOptions::subgroups` off makes the context subgroup-less).
/// Overrides: `GpuOptions::k3_kernel` (`Coop` errors when unavailable) and
/// `GpuOptions::k3_width` (4, 8, 16, 32 or 64, at most the minimum subgroup size). The output
/// never depends on them.
pub(crate) fn k3_mode(ctx: &GpuContext) -> anyhow::Result<K3Mode> {
    let force_coop = ctx.opts.k3_kernel == Some(K3Kernel::Coop);
    if ctx.opts.k3_kernel == Some(K3Kernel::Seq) {
        return Ok(K3Mode::Seq);
    }
    let min = ctx.adapter_info.subgroup_min_size;
    if !ctx.subgroups {
        anyhow::ensure!(!force_coop, "k3_kernel Coop (GZC_K3_MODE=coop): the device has no subgroup support");
        return Ok(K3Mode::Seq);
    }
    let w = match ctx.opts.k3_width {
        Some(w) => {
            anyhow::ensure!(
                w.is_power_of_two() && (4..=64).contains(&w) && w <= min,
                "k3_width {w} (GZC_K3_W): expected a power of two in 4..=64 and at most the minimum subgroup size {min}"
            );
            w
        }
        None => {
            let w = min.clamp(8, 64);
            1 << (31 - w.leading_zeros())
        }
    };
    if !probe_lanes(ctx, w)? {
        anyhow::ensure!(!force_coop, "k3_kernel Coop (GZC_K3_MODE=coop): subgroup lane probe failed for W = {w}");
        eprintln!("gzc: subgroup lane probe failed for W = {w}; using the sequential K3");
        return Ok(K3Mode::Seq);
    }
    Ok(K3Mode::Coop { w })
}

/// Dispatches one `@workgroup_size(w)` workgroup that records each lane's local index,
/// subgroup lane id, subgroup size and `subgroupBallot(true)`; true when the `w` lanes are one
/// subgroup with lane id == local index, size >= w and exactly the w-lane ballot (what
/// `k3_coop.wgsl` assumes).
pub fn probe_lanes(ctx: &GpuContext, w: u32) -> anyhow::Result<bool> {
    let src = format!(
        "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
         @compute @workgroup_size({w})\n\
         fn main(@builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) sid: u32,\n\
                 @builtin(subgroup_size) sz: u32) {{\n\
             let m = subgroupBallot(true);\n\
             let o = lid * 5u;\n\
             out[o] = lid; out[o + 1u] = sid; out[o + 2u] = sz; out[o + 3u] = m.x; out[o + 4u] = m.y;\n\
         }}\n"
    );
    with_error_scopes(ctx, || {
        let module = ctx.wgsl_module("k3_probe", &src, wgpu::ShaderRuntimeChecks::checked());
        let layout = storage_layout(ctx, "k3_probe", &[false]);
        let pipeline = pipeline_from_module(ctx, "k3_probe", &layout, &module, "main");
        let buf = ctx.storage_buffer("k3_probe", 5 * 4 * w as u64, true);
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k3_probe"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
        });
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("k3_probe") });
        {
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k3_probe"), timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit([enc.finish()]);
        let v: Vec<u32> = ctx.read_buffer(&buf, 0, 5 * w as usize);
        let (mx, my) = lane_mask(w);
        Ok(v.chunks(5).enumerate().all(|(i, l)| l[0] == i as u32 && l[1] == i as u32 && l[2] >= w && l[3] == mx && l[4] == my))
    })
}

impl Kernels {
    /// Builds the pipelines for `params`, with its `MatchParams` injected as WGSL constants.
    /// Errors if the params are invalid or not implemented on the GPU (`gpu_supports`).
    pub fn new(ctx: &GpuContext, params: GpuParams) -> anyhow::Result<Self> {
        let m = params.matching;
        check_matching(&m)?;
        let match_consts = params_wgsl(&m);
        let best_consts = format!("{match_consts}const BEST_OFF_BITS: u32 = {BEST_OFF_BITS}u;\n");
        let is_opt = m.opt.is_some();
        let best_layout = storage_layout(ctx, "k2", &[true, true, false]);
        let best = if is_opt {
            k2_opt_pipeline(ctx, &m, &best_layout)
        } else {
            compute_pipeline(ctx, "k2_best", &best_layout, &format!("{best_consts}{K2_WGSL}"))
        };
        let parse_layout = storage_layout(ctx, "k3", &[true, true, false, false]);
        // The unsegmented parse (greedy: `check_matching` refused an unsegmented lazy one).
        let unsegmented = !is_opt && m.segment_log2 == 0;
        let k3_body = format!("{best_consts}const MAX_SEQS: u32 = {MAX_SEQS}u;\n{K3_REPS_WGSL}\n{K3_WGSL}");
        // Both K3 modules are built without naga's forced loop bounding (a per-iteration counter
        // naga adds so the driver may not assume termination; it costs 33 % of the greedy K3's
        // time on an RTX 5090). Every K3 loop provably ends, whatever best[] holds:
        // - k3_coop.wgsl: each loop has a `// Terminates:` note (variant and bound); the only
        //   subtraction that could wrap, push_lits' end - start, is guarded.
        // - k3_parse.wgsl: the parse loop advances p below PARSE_END (a store by >= 1 byte, a
        //   skip by step >= 1); match_len's n grows to max <= BLOCK_SIZE - p (p < BLOCK_SIZE);
        //   push_lits has no loop.
        // Bounds checks stay on. A new K3 loop must come with the same argument, or use
        // `ctx.shader` instead.
        let parse = unsegmented.then(|| {
            pipeline_from_module(ctx, "k3_parse", &parse_layout, &ctx.shader_unbounded_loops("k3_parse", &k3_body), "main")
        });
        let k3_mode = if unsegmented { Some(k3_mode(ctx)?) } else { None };
        let parse_coop = match k3_mode {
            None | Some(K3Mode::Seq) => None,
            Some(K3Mode::Coop { w }) => {
                let (mx, my) = lane_mask(w);
                // The greedy rep test's second word: its first min_match - 4 bytes (4..=8).
                let rep_hi = ((1u64 << (8 * (m.min_match - 4))) - 1) as u32;
                // Test-only: `GpuOptions::k3_force_fallback` makes every workgroup take the
                // in-kernel sequential fallback (the path a failed lane-layout guard takes).
                let force_fallback = ctx.opts.k3_force_fallback;
                let body = format!(
                    "const W: u32 = {w}u;\nconst W_MASK_X: u32 = {mx}u;\nconst W_MASK_Y: u32 = {my}u;\n\
                     const REP_HI_MASK: u32 = {rep_hi}u;\nconst K3_FORCE_FALLBACK: bool = {force_fallback};\n\
                     {k3_body}\n{K3_COOP_WGSL}"
                );
                // Subgroup built-ins need Features::SUBGROUP on the device (naga 30 rejects
                // `enable subgroups;`).
                let module = ctx.shader_unbounded_loops("k3_coop", &body);
                Some(pipeline_from_module(ctx, "k3_coop", &parse_layout, &module, "main_coop"))
            }
        };
        // Segmented parse, also without the forced loop bounding: its parse loops advance ip
        // below the segment's lim (a store by >= 1 byte, a skip by exactly 1, the deferral by
        // 1-2, the immediate loop by ml >= 4); match_len's n grows to max <= lim - p; the
        // catch-up's start falls toward anchor; the fixup loops count up to NSEG and to the
        // segments' sequence counts.
        let parse_seg = (m.segment_log2 > 0 && !is_opt).then(|| {
            let seg_log2 = m.segment_log2;
            let layout = storage_layout(ctx, "k3_seg", &[true, false, false, false]);
            let body = format!(
                "{best_consts}const MAX_SEQS: u32 = {MAX_SEQS}u;\nconst SEG_LOG2: u32 = {}u;\nconst SEG_WG: u32 = {K3_SEG_WG}u;\nconst SCAN_W: u32 = {K3_SEG_SCAN}u;\n{K3_SEG_WGSL}\n{K3_FIXUP_WGSL}",
                seg_log2
            );
            let module = ctx.shader_unbounded_loops("k3_seg", &body);
            SegParse {
                seg: pipeline_from_module(ctx, "k3_seg", &layout, &module, "main_seg"),
                fixup: pipeline_from_module(ctx, "k3_seg_fixup", &layout, &module, "main_fixup"),
                layout,
                n_seg: (BLOCK_SIZE >> seg_log2) as u32,
            }
        });
        // The optimal parse: K3opt's passes, one workgroup of 16 lanes per block (a block's
        // segments in one workgroup, which the Prior seed and the cheap passes' histograms need).
        let opt = if is_opt { Some(OptPasses::new(ctx, &m, K3OptConfig::default())?) } else { None };
        let (tab, consts) = k4_tables(max_seqs(&m));
        let huffman = params.emit_frames && params.huffman;
        let consts = format!("{consts}const HUFFMAN: bool = {huffman};\n");
        let entropy = params.emit_frames.then(|| {
            let layout = storage_layout(ctx, "k4", &[true, true, true, true, false, false]);
            let pipeline = compute_pipeline(ctx, "k4_entropy", &layout, &format!("{consts}{K4_WGSL}"));
            let tables = ctx.storage_buffer("k4.tables", (tab.len() * 4) as u64, false);
            ctx.queue.write_buffer(&tables, 0, bytemuck::cast_slice(&tab));
            EntropyKernel { pipeline, layout, tables }
        });
        // K5 writes every literals section (Raw ones too), so it runs whenever K4 does.
        let huffman_kernel = params.emit_frames.then(|| {
            let layout = storage_layout(ctx, "k5", &[true, true, true, false, false]);
            let pipeline = compute_pipeline(ctx, "k5_huffman", &layout, &format!("{consts}{K5_WGSL}"));
            HuffmanKernel { pipeline, layout }
        });
        let trunc = params.emit_frames.then(|| {
            let layout = storage_layout(ctx, "k3t", &[true, true, false, false]);
            let body = format!("const MAX_SEQS: u32 = {}u;\n{K3_TRUNC_WGSL}", max_seqs(&m));
            (pipeline_from_module(ctx, "k3_trunc", &layout, &ctx.shader("k3_trunc", &body), "main"), layout)
        });
        let params = GpuParams { huffman, ..params };
        let chains = ChainsKernel::new(ctx, &m)?;
        let sorted = SortKernel::new(ctx, &m)?.map(|k1| {
            let body = format!("{}const BEST_OFF_BITS: u32 = {BEST_OFF_BITS}u;\n{K2_WGSL}\n{K2_WINDOW_WGSL}", finder_wgsl(&m));
            // Loops: the staging loop steps by 256 below WIN, the walk is j <= DEPTH, and
            // match_len_capped is bounded by SEARCH_CAP; win/wkey indices DEPTH + lid - j are in
            // 0..WIN (see `GpuContext::shader_trusted` and the E3 report).
            // Premise for the data-dependent indices: K1 (k1_sort / k1_sort_sg, run before this
            // in the same submission) wrote each block's `pred` slots 0..HASHED_POSITIONS as a
            // permutation of the positions 0..HASHED_POSITIONS (plus fingerprint bits above
            // PRED_POS). So every p = w & PRED_POS is < HASHED_POSITIONS <= BLOCK_SIZE, and
            // `best[sb + p]` and the byte loads at p and at q < p stay inside the block. The
            // sorted K1's self-test (against `gzc_core::hash::bucket_sort`, a permutation) and the
            // differential tests check it; a `pred` not written by the sorted K1 (e.g. chain-K1
            // output) breaks it, so K2 window must only ever run right after the sorted K1.
            let module = ctx.shader_trusted("k2_window", &body);
            (k1, pipeline_from_module(ctx, "k2_window", &best_layout, &module, "main_window"))
        });
        Ok(Self {
            chains,
            sorted,
            best,
            best_layout,
            parse,
            parse_layout,
            parse_coop,
            parse_seg,
            k3_mode,
            opt,
            entropy,
            huffman: huffman_kernel,
            trunc,
            params,
        })
    }

    /// True for the optimal parse (K2opt + the K3opt passes).
    pub fn is_opt(&self) -> bool {
        self.opt.is_some()
    }

    /// Bytes of the buffers the kernels own (K4's constant tables when emitting frames).
    pub fn own_buffer_bytes(&self) -> u64 {
        self.entropy.as_ref().map_or(0, |k4| k4.tables.size())
    }

    /// The match params the kernels were built for.
    pub fn matching(&self) -> MatchParams {
        self.params.matching
    }

    /// True when K1/K2 run the bucket-sorted finder (`sorted`), false for the hash chains.
    pub fn uses_sorted_finder(&self) -> bool {
        self.sorted.is_some()
    }

    /// How the unsegmented greedy K3 runs (see `k3_mode`); None when K3 is the segmented or the
    /// optimal parse.
    pub fn k3_mode(&self) -> Option<K3Mode> {
        self.k3_mode
    }

    /// True when K4 runs (built with `GpuParams::emit_frames`).
    pub fn emits_frames(&self) -> bool {
        self.entropy.is_some()
    }

    /// The `write_frame` options the emitted frames equal (Huffman literals iff K5 runs).
    pub fn frame_options(&self) -> FrameOptions {
        self.params.frame_options()
    }

    /// Names of the kernels `record_timed` runs, in timestamp order.
    pub fn names(&self) -> &'static [&'static str] {
        let names = if self.sorted.is_some() { &SORTED_KERNEL_NAMES } else { &KERNEL_NAMES };
        &names[..if self.emits_frames() { 5 } else { 3 }]
    }

    /// Upload is done by the caller (queue.write_buffer into bufs.data). Records K1→K2→K3, and
    /// (K5→) K4 when emitting frames (then `bufs` must have been allocated with frames).
    ///
    /// Precondition: `n_blocks <= bufs.capacity` (asserted); `BatchBuffers::new` guarantees
    /// `capacity <= max_batch_blocks`, which keeps every dispatch and u32 index in range. Errors
    /// when the optimal parse's buffers do not fit its passes (`OptBinds::check`).
    pub fn record(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &BatchBuffers,
        n_blocks: u32,
    ) -> anyhow::Result<()> {
        self.record_timed(ctx, enc, bufs, n_blocks, None)
    }

    /// `record`, with each kernel in its own compute pass writing begin/end timestamps into
    /// `queries` (at least `KERNEL_QUERIES` entries): K1 at 0/1, K2 at 2/3, K3 at 4/5, K4 at 6/7,
    /// K5 at 8/9.
    pub fn record_timed(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &BatchBuffers,
        n_blocks: u32,
        queries: Option<&wgpu::QuerySet>,
    ) -> anyhow::Result<()> {
        assert!(n_blocks <= bufs.capacity, "n_blocks {n_blocks} > capacity {}", bufs.capacity);
        assert_eq!(bufs.n_hashes, self.chains.n_hashes(), "BatchBuffers allocated for other match params");
        if n_blocks == 0 {
            return Ok(());
        }
        self.record_front(ctx, enc, bufs, n_blocks, queries)?;
        if self.emits_frames() {
            self.record_entropy(ctx, enc, bufs, n_blocks, queries);
        }
        Ok(())
    }

    /// K1, K2 and K3 of `record_timed` (the pipeline's transfer readback submits K5/K4
    /// separately). `1 <= n_blocks <= bufs.capacity`.
    pub(crate) fn record_front(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &BatchBuffers,
        n_blocks: u32,
        queries: Option<&wgpu::QuerySet>,
    ) -> anyhow::Result<()> {
        debug_assert!(n_blocks >= 1 && n_blocks <= bufs.capacity, "record_front: n_blocks {n_blocks} not in 1..={}", bufs.capacity);
        // `bufs.opt` (`OptScratch`, K3opt's prices/scratch) exists iff `bufs` was allocated for
        // an opt preset; reject a mismatch with these kernels before recording any dispatch,
        // rather than let K3opt bind a scratch buffer that was never allocated (or skip one that
        // was).
        anyhow::ensure!(
            bufs.opt.is_some() == self.is_opt(),
            "BatchBuffers allocated for {} match params, but these Kernels are {}opt",
            if bufs.opt.is_some() { "opt" } else { "non-opt" },
            if self.is_opt() { "" } else { "not " }
        );
        let ts = |k: u32| {
            queries.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(2 * k),
                end_of_pass_write_index: Some(2 * k + 1),
            })
        };
        poison_front(ctx, enc, bufs, n_blocks);
        self.record_best(ctx, enc, bufs, n_blocks, queries);
        self.record_parse(ctx, enc, bufs, n_blocks, ts(2))
    }

    /// Records K1 then K2 (timestamps as in `record_timed`) for the first `n_blocks` blocks of
    /// `bufs.data`, leaving the matches in `bufs.best`. `1 <= n_blocks <= bufs.capacity`.
    pub(crate) fn record_best(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &BatchBuffers,
        n_blocks: u32,
        queries: Option<&wgpu::QuerySet>,
    ) {
        let ts = |k: u32| {
            queries.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(2 * k),
                end_of_pass_write_index: Some(2 * k + 1),
            })
        };
        debug_assert!(n_blocks >= 1 && n_blocks <= bufs.capacity);
        let best = match &self.sorted {
            Some((k1, k2)) => {
                k1.record_timed(ctx, enc, &bufs.data, &bufs.pred, &bufs.best, n_blocks, ts(0));
                k2
            }
            None => {
                self.chains.record_timed(ctx, enc, &bufs.data, &bufs.head, &bufs.pred, n_blocks, ts(0));
                &self.best
            }
        };

        let k2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k2"),
            layout: &self.best_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs.data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs.pred.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bufs.best.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k2"), timestamp_writes: ts(1) });
        pass.set_pipeline(best);
        pass.set_bind_group(0, &k2, &[]);
        pass.dispatch_workgroups((BLOCK_SIZE / 256) as u32, n_blocks, 1);
    }

    /// Records K3 alone on whatever blocks (`data`) and matches (`best`) `bufs` holds for its
    /// first `n_blocks` blocks. `n_blocks` is at least 1 and at most `bufs.capacity`. For the
    /// optimal parse K3 is the block order, every K3opt pass, the fix-up and (opt16p1) the drop
    /// pass, `timestamp_writes` spanning them all; it errors when `bufs` does not fit them.
    pub(crate) fn record_parse(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &BatchBuffers,
        n_blocks: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites>,
    ) -> anyhow::Result<()> {
        debug_assert!(n_blocks >= 1 && n_blocks <= bufs.capacity);
        if let Some(passes) = &self.opt {
            // K1 wrote `pred`, K2opt read it and wrote the candidates into `best`: the passes use
            // `pred` as their trace and `best` as their candidates (then raw sequences).
            return passes.record_span(ctx, enc, &OptBinds::of_batch(bufs)?, n_blocks, timestamp_writes);
        }
        // K3 derives the batch size from the bound length of `counts`.
        let counts = wgpu::BufferBinding { buffer: &bufs.counts, offset: 0, size: wgpu::BufferSize::new(counts_bytes(n_blocks)) };
        if let Some(sp) = &self.parse_seg {
            let k3 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("k3_seg"),
                layout: &sp.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: bufs.data.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: bufs.best.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: bufs.seqs.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(counts) },
                ],
            });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k3"), timestamp_writes });
            pass.set_bind_group(0, &k3, &[]);
            pass.set_pipeline(&sp.seg);
            let groups = (n_blocks * sp.n_seg).div_ceil(K3_SEG_WG);
            assert!(
                groups <= ctx.device.limits().max_compute_workgroups_per_dimension,
                "k3_seg: {groups} workgroups exceed the device's per-dimension limit"
            );
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_pipeline(&sp.fixup);
            pass.dispatch_workgroups(n_blocks, 1, 1);
            return Ok(());
        }
        let k3 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k3"),
            layout: &self.parse_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs.data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs.best.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bufs.seqs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(counts) },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k3"), timestamp_writes });
        let seq = self.parse.as_ref().expect("the sequential K3 exists for an unsegmented parse");
        pass.set_pipeline(self.parse_coop.as_ref().unwrap_or(seq));
        pass.set_bind_group(0, &k3, &[]);
        pass.dispatch_workgroups(n_blocks, 1, 1);
        Ok(())
    }

    /// Records K5 then K4, on whatever parse (`seqs`, `counts`) and blocks (`data`) `bufs` holds
    /// for its first `n_blocks` blocks; timestamps as in `record_timed`. Panics unless built with
    /// `emit_frames` and `bufs` has frames and `n_blocks <= bufs.capacity`.
    pub fn record_entropy(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &BatchBuffers,
        n_blocks: u32,
        queries: Option<&wgpu::QuerySet>,
    ) {
        self.record_entropy_from(ctx, enc, bufs, n_blocks, queries, &bufs.data);
    }

    /// K3t: cuts the parse of every partial block of the batch to its real length, from
    /// `bufs.lens` (one u32 per block, BLOCK_SIZE for a full block, written by the caller before
    /// this runs). Recorded between K3 (`record_front`) and `record_entropy`, and only for a batch
    /// holding a partial block (it leaves full blocks alone), in its own compute pass with
    /// `timestamp_writes`. `1 <= n_blocks <= bufs.capacity`.
    pub(crate) fn record_truncate(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &BatchBuffers,
        n_blocks: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites>,
    ) {
        let (pipeline, layout) = self.trunc.as_ref().expect("Kernels built without emit_frames");
        let lens = bufs.lens.as_ref().expect("BatchBuffers allocated without frames");
        assert!(n_blocks >= 1 && n_blocks <= bufs.capacity, "n_blocks {n_blocks} not in 1..={}", bufs.capacity);
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k3t"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs.data.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: lens,
                        offset: 0,
                        size: wgpu::BufferSize::new(n_blocks as u64 * 4),
                    }),
                },
                wgpu::BindGroupEntry { binding: 2, resource: bufs.seqs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bufs.counts.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k3t"), timestamp_writes });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_blocks, 1, 1);
    }

    /// `record_entropy` with K5 gathering the literals from `lit_src` (`data`'s layout) instead of
    /// `bufs.data`; K4 still reads `bufs.data` (block-level RLE check, Raw fallback).
    pub(crate) fn record_entropy_from(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bufs: &BatchBuffers,
        n_blocks: u32,
        queries: Option<&wgpu::QuerySet>,
        lit_src: &wgpu::Buffer,
    ) {
        let ts = |k: u32| {
            queries.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(2 * k),
                end_of_pass_write_index: Some(2 * k + 1),
            })
        };
        assert!(n_blocks <= bufs.capacity, "n_blocks {n_blocks} > capacity {}", bufs.capacity);
        if n_blocks == 0 {
            return;
        }
        let k4 = self.entropy.as_ref().expect("Kernels built without emit_frames");
        let (Some(frames), Some(frame_len)) = (&bufs.frames, &bufs.frame_len) else {
            panic!("K4 needs BatchBuffers allocated with frames");
        };
        if ctx.poisoning() {
            // K5/K4's outputs, and their inputs past the batch's trailing zero word.
            ctx.poison_workgroup_memory(enc);
            ctx.poison_from(enc, frames, 0);
            ctx.poison_from(enc, frame_len, 0);
            ctx.poison_from(enc, &bufs.data, data_bytes(n_blocks));
            ctx.poison_from(enc, lit_src, data_bytes(n_blocks));
        }
        // K5 and K4 derive the batch size from the bound length of `frame_len`.
        let frame_len =
            wgpu::BufferBinding { buffer: frame_len, offset: 0, size: wgpu::BufferSize::new(frame_len_bytes(n_blocks)) };
        if let Some(k5) = &self.huffman {
            let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("k5"),
                layout: &k5.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: lit_src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: bufs.seqs.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: bufs.counts.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: frames.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Buffer(frame_len.clone()) },
                ],
            });
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k5"), timestamp_writes: ts(4) });
            pass.set_pipeline(&k5.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(n_blocks, 1, 1);
        }
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k4"),
            layout: &k4.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs.data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs.seqs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bufs.counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: k4.tables.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: frames.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Buffer(frame_len) },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k4"), timestamp_writes: ts(3) });
        pass.set_pipeline(&k4.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_blocks, 1, 1);
    }
}

/// When poisoning (`crate::poison`): garbage into workgroup memory and into every buffer K1–K3
/// (and K3opt) write, and into `data` past the batch's `n_blocks` blocks and trailing zero word.
fn poison_front(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, bufs: &BatchBuffers, n_blocks: u32) {
    if !ctx.poisoning() {
        return;
    }
    ctx.poison_workgroup_memory(enc);
    ctx.poison_from(enc, &bufs.data, data_bytes(n_blocks));
    for b in [&bufs.head, &bufs.pred, &bufs.best, &bufs.seqs, &bufs.counts] {
        ctx.poison_from(enc, b, 0);
    }
    if let Some(o) = &bufs.opt {
        ctx.poison_from(enc, &o.prices, 0);
        ctx.poison_from(enc, &o.scratch, 0);
        ctx.poison_from(enc, &o.sched, 0);
    }
}

/// Decodes one block's readback into a BlockOutput: `seq_words` holds at least `3 * n_seq` words
/// (lit_len, match_len, off_base) and `block` is the block K3 parsed. The literals are gathered
/// from the block (`gather_literals`).
pub(crate) fn decode_output(block: &[u8], seq_words: &[u32], n_seq: u32) -> BlockOutput {
    let sequences: Vec<Sequence> = seq_words[..3 * n_seq as usize]
        .chunks_exact(3)
        .map(|s| Sequence { lit_len: s[0], match_len: s[1], off_base: s[2] })
        .collect();
    let literals = gather_literals(block, &sequences);
    BlockOutput { sequences, literals }
}

/// The literal stream of `sequences` over `block`: sequence i's `lit_len` bytes start at the sum
/// of `lit_len + match_len` over the sequences before it, and the bytes after the last sequence
/// end the stream. A run that would leave the block (only a parse of a scripted best[] word that
/// claims a match past the block end can have one) is cut at the block's end.
pub fn gather_literals(block: &[u8], sequences: &[Sequence]) -> Vec<u8> {
    let mut literals = Vec::new();
    let mut anchor = 0usize;
    let span = |a: usize, len: usize| a.min(block.len())..a.saturating_add(len).min(block.len());
    for s in sequences {
        literals.extend_from_slice(&block[span(anchor, s.lit_len as usize)]);
        anchor += (s.lit_len + s.match_len) as usize;
    }
    literals.extend_from_slice(&block[span(anchor, block.len())]);
    literals
}

/// Runs `f` inside wgpu out-of-memory and validation error scopes, turning either (or a device
/// lost meanwhile) into `Err`.
/// `pub(crate)` so other modules that build+probe a pipeline outside `Kernels` (`chains`'
/// subgroup-kernel self-test) can catch a wgpu validation error at its source, instead of letting
/// it surface uncaptured (which wgpu may attribute to a later, unrelated error scope).
pub(crate) fn with_error_scopes<T>(ctx: &GpuContext, f: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
    let scopes = ErrorScopes::push(ctx);
    let result = f();
    scopes.pop()?;
    result
}

/// K2opt (`k2_opt.wgsl`'s `main_opt`) for opt params `m`, on a (data, pred, cands) layout.
pub(crate) fn k2_opt_pipeline(ctx: &GpuContext, m: &MatchParams, layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let body = format!(
        "{}{}const BEST_OFF_BITS: u32 = {BEST_OFF_BITS}u;\nconst H3_DEPTH: u32 = {OPT_H3_DEPTH}u;\nconst DEAD_BIT: u32 = {}u;\n{K2_WGSL}\n{K2_OPT_WGSL}",
        params_wgsl(m),
        layout_wgsl(m),
        gzc_core::reference::DEAD_BIT,
    );
    // Loops: every iteration of the merged walk spends one step of at least one live chain (so
    // at most the sum of the chains' depths, `reference::cand_depths`: h4, h3 and each sparse
    // chain's, 32 for opt16 and 60 for opt16p1) and match_len_capped is
    // bounded by SEARCH_CAP; indices as in K2 (pred words hold positions below HASHED_POSITIONS,
    // sparse chains' slot positions below SPARSE_END, from K1 in the same submission).
    let module = ctx.shader_trusted("k2_opt", &body);
    pipeline_from_module(ctx, "k2_opt", layout, &module, "main_opt")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sizing::{
        best_bytes_for, best_words, chain_pred_bytes, head_bytes, pred_bytes_for, scratch_bytes, seqs_bytes_for,
        slot_bytes, trace_bytes,
    };
    use gzc_core::fixtures::{LVL9, RUNG1, RUNG2};
    use gzc_core::params::{LVL3, LVL9S12SEG, LVL9SEG};

    const MIB: u64 = 1 << 20;

    fn limits(binding: u64, buffer: u64, wg: u32) -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffer_binding_size: binding,
            max_buffer_size: buffer,
            max_compute_workgroups_per_dimension: wg,
            ..wgpu::Limits::default()
        }
    }

    /// Every batch buffer for `n` blocks of `m` fits `limit`.
    fn fits(n: u32, m: &MatchParams, limit: u64) -> bool {
        let s = BufferSizes::new(n, m);
        [s.data, s.head, s.pred, s.best, s.seqs, s.counts, s.frames, s.opt_prices, s.opt_scratch, s.opt_sched]
        .iter()
        .all(|&b| b <= limit)
    }

    #[test]
    fn max_seqs_bounds_minimal_sequences() {
        assert_eq!(MAX_SEQS as usize, BLOCK_SIZE / 4 + 1);
        assert_eq!(MAX_SEQS_OPT as usize, BLOCK_SIZE / 3 + 1);
        assert_eq!(MAX_SEQS_OPT, 21846);
        for (name, p) in gzc_core::params::PRESETS {
            if gpu_supports(&p) {
                assert!(max_seqs(&p) as usize * p.min_seq_len() as usize >= BLOCK_SIZE, "{name}");
                check_matching(&p).unwrap_or_else(|e| panic!("{name}: {e}"));
            }
            // Existing presets keep MAX_SEQS; only the optimal parse (min match 3) needs more.
            assert_eq!(max_seqs(&p), if p.opt.is_some() { MAX_SEQS_OPT } else { MAX_SEQS }, "{name}");
            assert_eq!(seqs_bytes_for(7, &p), 7 * max_seqs(&p) as u64 * 12, "{name}");
        }
        assert_eq!(seqs_bytes_for(7, &LVL9SEG), 7 * MAX_SEQS as u64 * 12);
    }

    #[test]
    fn gpu_supports_all_presets() {
        use gzc_core::params::{OPT16, OPT16P1, OptParams, PriorTables, Seed, SparseChain};
        for (name, p) in gzc_core::params::PRESETS {
            // M5 T5: the optimal-parse presets too, and since M6 B4 opt16p1.
            assert!(gpu_supports(&p), "{name}");
            check_matching(&p).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
        let p1 = OPT16P1.opt.unwrap();
        let with = |o: OptParams| MatchParams { opt: Some(o), ..OPT16P1 };
        let sparse = |stride: u32| {
            with(OptParams { sparse_chains: [Some(SparseChain { width: 8, stride, depth: 16 }), None, None], ..p1 })
        };
        // Each M6 option alone on opt16, and at other valid values: accepted.
        let o16 = OPT16.opt.unwrap();
        for o in [
            OptParams { inner_gap: 3, ..o16 },
            OptParams { relax_lengths: Some(1), ..o16 },
            OptParams { relax_lengths: Some(32), ..o16 },
            OptParams { drop_max_len: 3, ..o16 },
            OptParams { drop_max_len: 32, ..o16 },
            OptParams { seed: Seed::Prior, prior: PriorTables::S3, ..o16 },
        ] {
            assert!(gpu_supports(&MatchParams { opt: Some(o), ..OPT16 }), "{o:?}");
        }
        assert!(gpu_supports(&with(OptParams { passes: 2, seed: Seed::BlockInit, prior: PriorTables::M5, ..p1 })));
        assert!(gpu_supports(&sparse(4)) && gpu_supports(&sparse(8)));
        // Valid, but K1 cannot hash slots that are not word aligned: refused.
        for stride in [1, 2] {
            let m = sparse(stride);
            assert!(m.validate().is_ok() && !gpu_supports(&m), "stride {stride}");
            assert!(check_matching(&m).unwrap_err().to_string().contains("not implemented"), "stride {stride}");
        }
        // Invalid M6 values stay refused.
        assert!(!gpu_supports(&with(OptParams { inner_gap: 4, ..p1 })));
        assert!(!gpu_supports(&with(OptParams { drop_max_len: 2, ..p1 })));
        assert!(!gpu_supports(&with(OptParams { seed: Seed::BlockInit, ..p1 })), "S3 prior without the Prior seed");
        assert!(gpu_supports(&MatchParams { depth: 4, ..LVL3 }));
        assert!(gpu_supports(&MatchParams { min_match: 8, depth: 64, ..RUNG1 }));
        // Lazy parses run on the GPU in segments only: the unsegmented ones (valid, and what the
        // CPU oracle's hand-built lazy cases use) are refused.
        for lazy in [RUNG2, LVL9, MatchParams { min_match: 6, ..RUNG2 }, MatchParams { hash_bits: 12, ..LVL9 }] {
            assert!(lazy.validate().is_ok() && !gpu_supports(&lazy), "{lazy:?}");
            let e = check_matching(&lazy).unwrap_err().to_string();
            assert!(e.contains("not implemented") && e.contains("segment"), "{e}");
            for segment_log2 in [10, 12, 16] {
                assert!(gpu_supports(&MatchParams { segment_log2, ..lazy }), "{lazy:?} in 2^{segment_log2} segments");
            }
        }
        let bad = MatchParams { lazy: 3, ..LVL9 };
        assert!(!gpu_supports(&bad));
        assert!(check_matching(&bad).unwrap_err().to_string().contains("lazy 3"));
    }

    /// `record` (via `record_front`) rejects `BatchBuffers` whose `opt` scratch does not match
    /// these `Kernels`' opt-ness, in both directions, before recording any dispatch.
    #[test]
    fn record_front_rejects_mismatched_opt_buffers() {
        let _gpu = crate::testing::gpu_test_slot();
        let ctx = crate::testing::gpu();
        let gp = |matching| GpuParams { matching, emit_frames: false, huffman: false };
        let opt16 = gzc_core::params::OPT16;

        let non_opt_kernels = Kernels::new(&ctx, gp(LVL3)).unwrap();
        let opt_bufs = BatchBuffers::new(&ctx, 4, false, &opt16).unwrap();
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let e = non_opt_kernels.record(&ctx, &mut enc, &opt_bufs, 4).unwrap_err();
        assert!(e.to_string().contains("opt"), "{e}");

        let opt_kernels = Kernels::new(&ctx, gp(opt16)).unwrap();
        let non_opt_bufs = BatchBuffers::new(&ctx, 4, false, &LVL3).unwrap();
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let e = opt_kernels.record(&ctx, &mut enc, &non_opt_bufs, 4).unwrap_err();
        assert!(e.to_string().contains("opt"), "{e}");
    }

    #[test]
    fn batch_fits_every_buffer() {
        for m in [LVL3, RUNG1, LVL9SEG, LVL9S12SEG] {
            for limit in [4 * MIB, 128 * MIB, 1 << 31, (1 << 32) - 4] {
                let n = max_batch_blocks(&limits(limit, u64::MAX, 65535), &m);
                assert!(n > 0 && fits(n, &m, limit) && !fits(n + 1, &m, limit), "{m:?} limit {limit}: n {n}");
                // max_buffer_size is honoured the same way as the binding size.
                assert_eq!(max_batch_blocks(&limits(u64::MAX, limit, 65535), &m), n);
            }
        }
        let at_128m = |m: &MatchParams| max_batch_blocks(&limits(128 * MIB, 128 * MIB, 65535), m);
        // pred 512 KiB per block with two chains; one chain: pred/best 256 KiB.
        assert_eq!((at_128m(&LVL3), at_128m(&RUNG1)), (256, 512));
    }

    /// The optimal parse's `best` buffer holds two candidate words per position (8 B), which
    /// `max_batch_blocks`, `BatchBuffers` and `scratch_bytes` (so `vram_bytes`) all count.
    #[test]
    fn opt_batch_counts_two_candidate_words() {
        use gzc_core::params::{OPT14, OPT16};
        assert_eq!((best_words(&LVL9SEG), best_words(&OPT16), best_words(&OPT14)), (1, 2, 2));
        assert_eq!(best_bytes_for(3, &OPT16), 3 * 8 * BLOCK_SIZE as u64);
        // Against lvl3 (also two chains): the second candidate word, the larger seqs, and K3opt's
        // prices (377 words) and scratch per block. The trace reuses pred (8 B per position).
        let k3opt = crate::sizing::prices_bytes(10)
            + 10 * crate::k3opt::scratch_bytes_per_block(&OPT16)
            + crate::sizing::sched_bytes(10);
        let opt_bytes = |m: &MatchParams| {
            let s = BufferSizes::new(10, m);
            s.opt_prices + s.opt_scratch + s.opt_sched
        };
        assert_eq!(pred_bytes_for(10, &OPT16), trace_bytes(10));
        assert_eq!(pred_bytes_for(10, &OPT16), chain_pred_bytes(10, &LVL3));
        assert_eq!(opt_bytes(&OPT16), k3opt);
        assert_eq!((opt_bytes(&LVL3), crate::sizing::prices_bytes(1)), (0, 377 * 4));
        assert_eq!(
            scratch_bytes(10, &OPT16) - scratch_bytes(10, &LVL3),
            best_bytes(10) + seqs_bytes_for(10, &OPT16) - seqs_bytes_for(10, &LVL3) + k3opt
        );
        assert_eq!(crate::k3opt::scratch_bytes_per_block(&OPT16), 6336);
        for limit in [4 * MIB, 128 * MIB, 1 << 31, (1 << 32) - 4] {
            let n = max_batch_blocks(&limits(limit, u64::MAX, 65535), &OPT16);
            assert!(n > 0 && fits(n, &OPT16, limit) && !fits(n + 1, &OPT16, limit), "limit {limit}: n {n}");
        }
        let n = max_batch_blocks(&limits(u64::MAX, u64::MAX, u32::MAX), &OPT16) as u64;
        assert!(n > 0 && n * 2 * BLOCK_SIZE as u64 <= 1 << 32, "cands / trace word index");
        assert!(n * 3 * MAX_SEQS_OPT as u64 <= 1 << 32, "seqs word index");
    }

    /// M6 `opt16p1`: K1's pred buffer holds h4 and h3 at full length plus the three stride-4 sparse
    /// chains compactly (11 B per position, against opt16's 8 B, which K3opt's trace also needs),
    /// counted by `pred_bytes_for` and so by `scratch_bytes` (`vram_bytes`) and `max_batch_blocks`.
    #[test]
    fn opt16p1_pred_counts_sparse_chains() {
        use gzc_core::params::{OPT16, OPT16P1};
        assert_eq!(OPT16P1.n_hashes(), 5);
        assert_eq!(pred_bytes_for(10, &OPT16P1), 10 * 11 * BLOCK_SIZE as u64);
        assert_eq!(pred_bytes_for(10, &OPT16P1), chain_pred_bytes(10, &OPT16P1));
        assert!(pred_bytes_for(10, &OPT16P1) > trace_bytes(10));
        // Five chains share K1's head tables, still at most HEAD_TABLES.
        assert_eq!(head_bytes(10, OPT16P1.n_hashes()), 50 * (1 << 18));
        assert_eq!(head_bytes(1000, 5), head_bytes(1000, 2));
        assert_eq!(
            scratch_bytes(10, &OPT16P1) - scratch_bytes(10, &OPT16),
            head_bytes(10, 5) - head_bytes(10, 2) + 10 * 3 * BLOCK_SIZE as u64
        );
        for limit in [4 * MIB, 128 * MIB, 1 << 31, (1 << 32) - 4] {
            let n = max_batch_blocks(&limits(limit, u64::MAX, 65535), &OPT16P1);
            assert!(n > 0 && fits(n, &OPT16P1, limit) && !fits(n + 1, &OPT16P1, limit), "limit {limit}: n {n}");
        }
        // With 2 GiB storage bindings (an RTX 5090 under wgpu), the 704 KiB of pred per block
        // bound the batch below the 6 GiB budget's 3125 (M6 B4: `--batch max` is 2978).
        assert_eq!(max_batch_blocks(&limits(1 << 31, u64::MAX, 65535), &OPT16P1), 2978);
        let n = max_batch_blocks(&limits(u64::MAX, u64::MAX, u32::MAX), &OPT16P1) as u64;
        assert!(n > 0 && n * 11 * BLOCK_SIZE as u64 / 4 <= 1 << 32, "pred word index");
    }

    /// The batch buffers for `opt16p1` allocate exactly `scratch_bytes` + `slot_bytes`.
    #[test]
    fn opt16p1_batch_buffers_allocate_scratch_bytes() {
        let _gpu = crate::testing::gpu_test_slot();
        let m = gzc_core::params::OPT16P1;
        let ctx = crate::testing::gpu();
        let b = BatchBuffers::new(&ctx, 7, true, &m).unwrap();
        let o = b.opt.as_ref().unwrap();
        let frames = [b.frames.as_ref().unwrap(), b.frame_len.as_ref().unwrap(), b.lens.as_ref().unwrap()];
        let sizes = [&b.data, &b.head, &b.pred, &b.best, &b.seqs, &b.counts, frames[0], frames[1], frames[2]]
            .iter()
            .chain([&o.prices, &o.scratch, &o.sched].iter())
            .map(|x| x.size())
            .sum::<u64>();
        assert_eq!(sizes, scratch_bytes(7, &m) + slot_bytes(7, true));
        assert_eq!(b.pred.size(), 7 * 11 * BLOCK_SIZE as u64);
    }

    #[test]
    fn batch_respects_workgroup_cap() {
        for m in [LVL3, RUNG1, LVL9SEG, LVL9S12SEG] {
            assert_eq!(max_batch_blocks(&limits(128 * MIB, 128 * MIB, 3), &m), 3);
        }
    }

    #[test]
    fn batch_keeps_u32_indices_in_range() {
        for m in [LVL3, RUNG1, LVL9SEG, LVL9S12SEG] {
            let nh = m.n_hashes() as u64;
            let n = max_batch_blocks(&limits(u64::MAX, u64::MAX, u32::MAX), &m) as u64;
            assert!(n > 0);
            // Largest word index of each buffer indexed by the kernels.
            assert!(n * BLOCK_SIZE as u64 <= 1 << 32, "best");
            assert!(n * nh * BLOCK_SIZE as u64 <= 1 << 32, "pred");
            assert!(n * MAX_SEQS as u64 * 3 <= 1 << 32, "seqs");
            assert!((n * nh).min(chains::HEAD_TABLES as u64) << gzc_core::config::HASH_BITS <= 1 << 32, "head");
        }
    }

    #[test]
    fn batch_is_zero_when_one_block_does_not_fit() {
        for m in [LVL3, RUNG1, LVL9SEG, LVL9S12SEG] {
            assert_eq!(max_batch_blocks(&limits(BLOCK_SIZE as u64, u64::MAX, 65535), &m), 0);
        }
    }

    #[test]
    fn decode_output_gathers_literals() {
        let seqs = [3, 5, 7, 0, 6, 1, 99, 99, 99];
        let block = b"abcXXXXXYYYYYYdefg";
        let out = decode_output(block, &seqs, 2);
        assert_eq!(
            out.sequences,
            vec![
                Sequence { lit_len: 3, match_len: 5, off_base: 7 },
                Sequence { lit_len: 0, match_len: 6, off_base: 1 }
            ]
        );
        assert_eq!(out.literals, b"abcdefg");
        assert_eq!(decode_output(b"", &[], 0), BlockOutput::default());
        assert_eq!(decode_output(b"xyz", &[], 0).literals, b"xyz");
        // A match that ends exactly at the block's end leaves no trailing literals; one past it
        // (a scripted best[] only) is cut, not a panic.
        assert_eq!(decode_output(b"ab0123", &[2, 4, 1], 1).literals, b"ab");
        assert_eq!(decode_output(b"ab0123", &[2, 9, 1, 3, 4, 1], 2).literals, b"ab");
    }
}
