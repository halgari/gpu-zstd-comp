//! Host side of the K2 (best match), K3 (parse), K5 (Huffman literals) and K4 (entropy + frame
//! assembly) kernels; compress_batch / compress_frames entry points.
//!
//! K1 (hash chains) → K2 (`find_best`) → K3 (`reference::parse`: greedy, or `lazy::lazy_parse`
//! when `MatchParams::lazy > 0`) reproduce
//! `gzc_core::reference::compress_block` exactly, for a batch of BLOCK_SIZE blocks. K5 writes each
//! block's literals section into its frame and K4 completes the frame, byte-identical to
//! `gzc_core::frame::write_frame` with `GpuParams::frame_options()`: Huffman literals
//! (`FrameOptions::default()`) with `huffman`, raw literals (still written by K5) without.
//!
//! K3 writes only the sequences and counts; the literals are the block bytes the sequences leave
//! uncovered, so K5 gathers them from `data` and the parse path from the host's copy of the block
//! (`decode_output`).
use crate::chains::{self, ChainsKernel, finder_wgsl, head_bytes, pred_bytes};
use crate::k3opt::{K3OptConfig, OptBinds, OptPasses};
use crate::context::{ErrorScopes, GpuContext, pack_blocks, params_wgsl};
use crate::sorted::SortKernel;
use anyhow::{Context as _, anyhow};
use gzc_core::codes::{
    LL_BASE, LL_BITS, LL_DEFAULT_NORM, ML_BASE, ML_BITS, ML_DEFAULT_NORM, OF_DEFAULT_NORM, ll_code, ml_code,
};
use gzc_core::config::BLOCK_SIZE;
use gzc_core::frame::{FrameOptions, frame_header};
use gzc_core::fse::FRAC;
use gzc_core::params::{MatchParams, OPT_H3_DEPTH};
use gzc_core::reference::{CandWords, Match};
use gzc_core::seq::{BlockOutput, Sequence};

const K2_WGSL: &str = include_str!("shaders/k2_best.wgsl");
const K2_WINDOW_WGSL: &str = include_str!("shaders/k2_window.wgsl");
const K2_OPT_WGSL: &str = include_str!("shaders/k2_opt.wgsl");
const K3_WGSL: &str = include_str!("shaders/k3_parse.wgsl");
const K3_LAZY_WGSL: &str = include_str!("shaders/k3_lazy.wgsl");
const K3_COOP_WGSL: &str = include_str!("shaders/k3_coop.wgsl");
const K3_SEG_WGSL: &str = include_str!("shaders/k3_seg.wgsl");
/// `main_fixup` and the rep helpers, shared by `k3_seg.wgsl` and `k3_opt.wgsl`.
pub(crate) const K3_FIXUP_WGSL: &str = include_str!("shaders/k3_fixup.wgsl");
const K4_WGSL: &str = include_str!("shaders/k4_seq_entropy.wgsl");
const K5_WGSL: &str = include_str!("shaders/k5_huffman.wgsl");

/// Bytes reserved per block in the `frames` buffer: a Raw frame (header + 3 + BLOCK_SIZE) is the
/// largest K4 emits.
pub const FRAME_STRIDE: usize = BLOCK_SIZE + 64;

/// Shortest sequence (`MatchParams::min_seq_len`) any GPU-supported preset emits except the
/// optimal parse (3, `MAX_SEQS_OPT`): the smallest allowed `min_match`. `MAX_SEQS` is sized for it
/// and `check_matching` rejects params that could emit shorter ones than their `max_seqs`.
pub const MAX_SEQS_MIN_SEQ_LEN: usize = 4;

/// Upper bound on sequences per block: every sequence covers at least `MAX_SEQS_MIN_SEQ_LEN` bytes.
/// At 128K this is 32769 > 0x7F00, so K4's 3-byte nbSeq header is reachable.
pub const MAX_SEQS: u32 = (BLOCK_SIZE / MAX_SEQS_MIN_SEQ_LEN) as u32 + 1;

/// Upper bound on sequences per block of the optimal parse (`MatchParams::opt`, min match 3):
/// `BLOCK_SIZE / 3 + 1` (21846 at 64 KiB). The `seqs` buffer, the parse readback's stride and
/// K3opt/K5/K4's `MAX_SEQS` constant use it for opt params (`max_seqs`); every other preset keeps
/// `MAX_SEQS`.
pub const MAX_SEQS_OPT: u32 = (BLOCK_SIZE / 3) as u32 + 1;

// K4's 3-byte nbSeq form (nbSeq >= 0x7F00 = 32512) cannot fire for the optimal parse at the block
// sizes it runs at (<= 64 KiB, `gpu_supports`): 21846 < 32512. The form itself is tested at the
// CPU level (`gzc_core::seqenc` tests) and on the GPU at 128 KiB (`differential.rs`).
#[cfg(not(feature = "block-128k"))]
const _: () = assert!(MAX_SEQS_OPT < 0x7F00, "opt blocks never reach the 3-byte nbSeq form");

/// Sequences per block the `seqs` buffer holds under match params `m`: `MAX_SEQS_OPT` for the
/// optimal parse (min match 3), else `MAX_SEQS`.
pub fn max_seqs(m: &MatchParams) -> u32 {
    if m.opt.is_some() { MAX_SEQS_OPT } else { MAX_SEQS }
}

/// Largest batch `compress_batch`/`compress_frames` allocate buffers for, even when the device
/// limits would allow more. At 128K blocks, worst case (dfast's two hash chains, `emit_frames`:
/// `data_bytes` + `chains::head_bytes`/`pred_bytes` + `best_bytes` + `seqs_bytes` +
/// `counts_bytes` + `frames_bytes` + `frame_len_bytes`, with `head_bytes` capped at
/// `chains::HEAD_TABLES` tables) is ~2.4 MiB per block, ~304 MiB at this cap. These one-shot
/// paths serve the tests, several of
/// which run at once (each with its own device): 128 keeps their 300-block batches split into
/// full and partial batches (as 256 did) at half the memory.
const COMPRESS_BATCH_CAP: u32 = 128;

/// Kernel names, in timestamp-query order, as reported in timing breakdowns (`k4_entropy` and
/// `k5_huffman` only run with `GpuParams::emit_frames`; K5, which writes the literals section
/// (Huffman-coded only with `huffman`), is dispatched before K4; see `Kernels::names`).
pub const KERNEL_NAMES: [&str; 5] = ["k1_chains", "k2_best", "k3_parse", "k4_entropy", "k5_huffman"];

/// `KERNEL_NAMES` when K1/K2 run the bucket-sorted finder (`Kernels::uses_sorted_finder`).
pub const SORTED_KERNEL_NAMES: [&str; 5] = ["k1_sort", "k2_window", "k3_parse", "k4_entropy", "k5_huffman"];

/// Timestamp queries `Kernels::record_timed` may write: a begin/end pair per kernel.
pub const KERNEL_QUERIES: u32 = 2 * KERNEL_NAMES.len() as u32;

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

/// Whether the GPU kernels implement `p`: every valid `MatchParams` (both hash modes, greedy,
/// lazy and lazy2 since M4 Task 5; the M5 optimal parse, presets `opt14`/`opt16`, with K2opt and
/// the K3opt passes since M5 T5, at blocks of at most 64 KiB: K3opt's 16-bit offsets).
/// The M6 opt options (sparse chains, gap3, top-4 pruning, the drop pass, the S3 prior: preset
/// `opt16p1`) are not on the GPU yet, so only M5-shaped `OptParams` are accepted.
pub fn gpu_supports(p: &MatchParams) -> bool {
    p.validate().is_ok() && p.opt.is_none_or(|o| o.is_m5() && BLOCK_SIZE <= 1 << 16)
}

/// Ok when `m` is valid, implemented on the GPU and its sequences fit `max_seqs(m)`.
pub fn check_matching(m: &MatchParams) -> anyhow::Result<()> {
    m.validate().map_err(|e| anyhow!("invalid match params {m:?}: {e}"))?;
    anyhow::ensure!(gpu_supports(m), "match params {m:?} are not implemented yet on gpu");
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

/// Bytes of the packed `data` buffer for `n_blocks` (blocks plus one trailing zero word).
pub fn data_bytes(n_blocks: u32) -> u64 {
    n_blocks as u64 * BLOCK_SIZE as u64 + 4
}

/// Low bits of a `best[]` word holding the match offset; the (capped) length sits above them.
pub const BEST_OFF_BITS: u32 = 17;
const _: () = assert!(BLOCK_SIZE <= 1 << BEST_OFF_BITS, "offsets must fit BEST_OFF_BITS");
// MatchParams::validate bounds search_cap to 8..=256.
const _: () = assert!(256 < 1u64 << (32 - BEST_OFF_BITS), "capped lengths must fit above the offset");

/// Bytes of the `best` buffer: `[block][pos]` × one u32, `(capped len << BEST_OFF_BITS) | offset`
/// (0 = no match).
pub const fn best_bytes(n_blocks: u32) -> u64 {
    n_blocks as u64 * BLOCK_SIZE as u64 * 4
}

/// u32 words per position of the `best` buffer under match params `m`: 1 (K2's best match), or 2
/// for the optimal parse (`m.opt`: K2opt's two candidate words, `reference::CandWords`).
pub fn best_words(m: &MatchParams) -> u32 {
    if m.opt.is_some() { 2 } else { 1 }
}

/// Bytes of the `best` buffer under match params `m`: `best_bytes`, times `best_words(m)`.
pub fn best_bytes_for(n_blocks: u32, m: &MatchParams) -> u64 {
    best_bytes(n_blocks) * best_words(m) as u64
}

/// Bytes of the `seqs` buffer: `[block][MAX_SEQS]` × (lit_len, match_len, off_base) u32.
pub fn seqs_bytes(n_blocks: u32) -> u64 {
    n_blocks as u64 * MAX_SEQS as u64 * 12
}

/// Bytes of the `seqs` buffer under match params `m`: `[block][max_seqs(m)]` × 3 u32.
pub fn seqs_bytes_for(n_blocks: u32, m: &MatchParams) -> u64 {
    n_blocks as u64 * max_seqs(m) as u64 * 12
}

/// Bytes of the `pred` buffer under match params `m`: K1's chains (`chains::pred_bytes`), and for
/// the optimal parse at least K3opt's DP trace, which reuses the buffer once K2opt has read the
/// chains (`trace_bytes`). Opt3 has two chains, so the two are equal.
pub fn pred_bytes_for(n_blocks: u32, m: &MatchParams) -> u64 {
    let chains = pred_bytes(n_blocks, m.n_hashes());
    if m.opt.is_some() { chains.max(trace_bytes(n_blocks)) } else { chains }
}

/// Bytes of K3opt's DP trace for `n_blocks`: `[block][pos][2]` u32 (`tbase = 2 * b * BLOCK_SIZE`
/// in `k3_opt.wgsl`), 8 B per position.
pub fn trace_bytes(n_blocks: u32) -> u64 {
    n_blocks as u64 * BLOCK_SIZE as u64 * 8
}

/// Bytes of K3opt's own buffers for `n_blocks` under match params `m` (0 without `opt`): the
/// price tables / histograms (`k3opt::PRICE_WORDS` words per block) and the DP nodes' payload
/// scratch (`k3opt::scratch_bytes_per_block`).
pub fn opt_bytes(n_blocks: u32, m: &MatchParams) -> u64 {
    if m.opt.is_none() {
        return 0;
    }
    n_blocks as u64 * (crate::k3opt::prices_bytes(1) + crate::k3opt::scratch_bytes_per_block(m))
}

/// Bytes of the `counts` buffer: `[block]` × (n_seq, n_lit) u32.
pub fn counts_bytes(n_blocks: u32) -> u64 {
    n_blocks as u64 * 8
}

/// Bytes of the `frames` buffer: `[block][FRAME_STRIDE]` frame bytes.
pub fn frames_bytes(n_blocks: u32) -> u64 {
    n_blocks as u64 * FRAME_STRIDE as u64
}

/// Bytes of the `frame_len` buffer: `[block]` u32.
pub fn frame_len_bytes(n_blocks: u32) -> u64 {
    n_blocks as u64 * 4
}

/// Largest `n_blocks` one `Kernels::record` call (and one `BatchBuffers`) for match params `m`
/// may take under `limits`: K1's bound (`chains::max_blocks_per_batch` for `m.n_hashes()`: data,
/// head and pred buffers, the workgroups-per-dimension limit used by K1's x and K2's y dispatch,
/// u32 head/pred indices) further limited so the best, seqs, counts and frames buffers each
/// fit one storage binding and one buffer and their u32 word indices (at most BLOCK_SIZE words
/// per block, in `best`, times `best_words(m)`; `seqs` has 3*MAX_SEQS < BLOCK_SIZE) cannot wrap.
/// 0 if one block doesn't fit.
pub fn max_batch_blocks(limits: &wgpu::Limits, m: &MatchParams) -> u32 {
    let limit = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
    let by_k1 = chains::max_blocks_per_batch(limits, m.n_hashes()) as u64;
    let mut per_block =
        vec![best_bytes_for(1, m), pred_bytes_for(1, m), seqs_bytes_for(1, m), counts_bytes(1), frames_bytes(1)];
    if m.opt.is_some() {
        per_block.push(crate::k3opt::prices_bytes(1));
        per_block.push(crate::k3opt::scratch_bytes_per_block(m));
    }
    let by_buffers = per_block.into_iter().map(|per_block| limit / per_block).min().unwrap();
    let by_index = (1u64 << 32) / (BLOCK_SIZE as u64 * best_words(m) as u64);
    by_k1.min(by_buffers).min(by_index) as u32
}

/// Per-batch buffers, allocated for `capacity` blocks and reused.
pub struct BatchBuffers {
    pub capacity: u32,
    /// Hash chains per block the head/pred buffers hold (`MatchParams::n_hashes`).
    pub n_hashes: u32,
    /// Packed blocks (`pack_blocks` layout); written by the caller.
    pub data: wgpu::Buffer,
    /// K1 scratch hash-head tables, `[table < chains::HEAD_TABLES][2^HASH_BITS]`
    /// (`chains::head_bytes`: one table per chain, at most `HEAD_TABLES`).
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
        let bytes = scratch_bytes(capacity, m)
            + if data.is_none() { data_bytes(capacity) } else { 0 }
            + if frames && frame_bufs.is_none() { frames_bytes(capacity) + frame_len_bytes(capacity) } else { 0 };
        let scopes = ErrorScopes::push(ctx);
        let n_hashes = m.n_hashes();
        let (frames, frame_len) = match (frames, frame_bufs) {
            (false, _) => (None, None),
            (true, Some((f, l))) => (Some(f), Some(l)),
            (true, None) => (
                Some(ctx.storage_buffer("batch.frames", frames_bytes(capacity), true)),
                Some(ctx.storage_buffer("batch.frame_len", frame_len_bytes(capacity), true)),
            ),
        };
        let opt = m.opt.is_some().then(|| {
            let scratch_per_block = crate::k3opt::scratch_bytes_per_block(m);
            OptScratch {
                prices: ctx.storage_buffer("batch.opt_prices", crate::k3opt::prices_bytes(capacity), true),
                scratch: ctx.storage_buffer("batch.opt_scratch", capacity as u64 * scratch_per_block, false),
                scratch_per_block,
            }
        });
        // K3opt's `ld32` reads one word past a block's last word, so `data` keeps its trailing
        // zero word (`data_bytes` = n * BLOCK_SIZE + 4), also when the caller brings it (the
        // pipeline's direct-upload / zero-copy slots are `data_bytes(batch)` with that word
        // zeroed on every submit).
        let data = data.unwrap_or_else(|| ctx.storage_buffer("batch.data", data_bytes(capacity), false));
        assert!(data.size() >= data_bytes(capacity), "data buffer below data_bytes({capacity})");
        let bufs = Self {
            capacity,
            n_hashes,
            data,
            head: ctx.storage_buffer("batch.head", head_bytes(capacity, n_hashes), false),
            pred: ctx.storage_buffer("batch.pred", pred_bytes_for(capacity, m), true),
            best: ctx.storage_buffer("batch.best", best_bytes_for(capacity, m), true),
            seqs: ctx.storage_buffer("batch.seqs", seqs_bytes_for(capacity, m), true),
            counts: ctx.storage_buffer("batch.counts", counts_bytes(capacity), true),
            frames,
            frame_len,
            opt,
        };
        scopes.pop_alloc(&format!("the batch buffers ({capacity} blocks)"), bytes)?;
        Ok(bufs)
    }
}

/// Bytes of the scratch buffers (head, pred, best, seqs, counts, and K3opt's prices and scratch
/// for the optimal parse) for `n_blocks` under match params `m`: one head/pred chain per
/// `m.n_hashes()` (`pred` at least the opt trace, `pred_bytes_for`), `seqs` of `max_seqs(m)`.
pub fn scratch_bytes(n_blocks: u32, m: &MatchParams) -> u64 {
    head_bytes(n_blocks, m.n_hashes())
        + pred_bytes_for(n_blocks, m)
        + best_bytes_for(n_blocks, m)
        + seqs_bytes_for(n_blocks, m)
        + counts_bytes(n_blocks)
        + opt_bytes(n_blocks, m)
}

/// Bytes of the remaining `BatchBuffers` (data, plus frames and frame_len with `frames`) for
/// `n_blocks`; the pipeline allocates them once, shared by all its slots.
pub fn slot_bytes(n_blocks: u32, frames: bool) -> u64 {
    data_bytes(n_blocks) + if frames { frames_bytes(n_blocks) + frame_len_bytes(n_blocks) } else { 0 }
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
    /// The sequential K3 (None for the optimal parse, whose K3 is `opt`).
    parse: Option<wgpu::ComputePipeline>,
    parse_layout: wgpu::BindGroupLayout,
    /// The subgroup-cooperative K3 (`K3Mode::Coop`), used instead of `parse` when present.
    parse_coop: Option<wgpu::ComputePipeline>,
    /// The segmented K3 (`MatchParams::segment_log2 > 0`), used instead of both when present.
    parse_seg: Option<SegParse>,
    k3_mode: K3Mode,
    /// The optimal parse (`MatchParams::opt`, M5): the K3opt passes (`k3opt::OptPasses`), run as
    /// K3 instead of every parse above, on the shared buffers (`OptBinds::of_batch`).
    opt: Option<OptPasses>,
    entropy: Option<EntropyKernel>,
    huffman: Option<HuffmanKernel>,
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
/// Positions `k3_seg.wgsl`'s literal scan tests per step (8: K3 12.1 -> 7.1 ms at 128 KiB blocks
/// against 1; 4 and 12-16 are slower).
const K3_SEG_SCAN: u32 = 8;

/// K4's `tab` buffer contents and the WGSL constants locating each table in it. Every value
/// comes from gzc_core, so the GPU mirrors the CPU tables exactly.
/// Bytes of K4's constant table buffer.
pub fn k4_tables_bytes() -> u64 {
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
    // to 3 bytes must fit in 16 bytes.
    assert!(hdr.len() <= 10, "frame header longer than K4 expects");
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

/// How K3 runs the lazy / lazy2 parse (speed phase S3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum K3Mode {
    /// `k3_parse.wgsl` + `k3_lazy.wgsl`: one lane per block.
    Seq,
    /// `k3_coop.wgsl`: one subgroup of `w` lanes per block (needs `Features::SUBGROUP`), `bpw`
    /// blocks per workgroup.
    Coop { w: u32, bpw: u32 },
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
/// ids 0..W-1 (else `Seq`; `GZC_NO_SUBGROUPS` makes the context subgroup-less, see
/// `GpuContext::new`). Overrides: `GZC_K3_MODE=seq|coop` (coop errors when unavailable),
/// `GZC_K3_W=4|8|16|32|64` (at most the minimum subgroup size) and `GZC_K3_BPW=1|2` (blocks per
/// workgroup; 2 needs minimum == maximum subgroup size == W). The output never depends on them.
pub fn k3_mode(ctx: &GpuContext) -> anyhow::Result<K3Mode> {
    let forced = std::env::var("GZC_K3_MODE").ok();
    match forced.as_deref() {
        None | Some("") | Some("coop") | Some("seq") => {}
        Some(v) => anyhow::bail!("GZC_K3_MODE={v}: expected seq or coop"),
    }
    let force_coop = forced.as_deref() == Some("coop");
    if forced.as_deref() == Some("seq") {
        return Ok(K3Mode::Seq);
    }
    let min = ctx.adapter_info.subgroup_min_size;
    if !ctx.subgroups {
        anyhow::ensure!(!force_coop, "GZC_K3_MODE=coop: the device has no subgroup support");
        return Ok(K3Mode::Seq);
    }
    let w = match std::env::var("GZC_K3_W") {
        Ok(v) => {
            let w: u32 = v.parse().map_err(|_| anyhow!("GZC_K3_W={v}: not a number"))?;
            anyhow::ensure!(
                w.is_power_of_two() && (4..=64).contains(&w) && w <= min,
                "GZC_K3_W={w}: expected a power of two in 4..=64 and at most the minimum subgroup size {min}"
            );
            w
        }
        Err(_) => {
            let w = min.clamp(8, 64);
            1 << (31 - w.leading_zeros())
        }
    };
    let bpw = match std::env::var("GZC_K3_BPW").as_deref() {
        Ok("2") => {
            let max = ctx.adapter_info.subgroup_max_size;
            anyhow::ensure!(min == max && w == min, "GZC_K3_BPW=2 needs W == min == max subgroup size ({w}, {min}, {max})");
            2
        }
        Ok("1") | Err(_) => 1,
        Ok(v) => anyhow::bail!("GZC_K3_BPW={v}: expected 1 or 2"),
    };
    if !probe_lanes(ctx, w, bpw)? {
        anyhow::ensure!(!force_coop, "GZC_K3_MODE=coop: subgroup lane probe failed for W = {w}, BPW = {bpw}");
        eprintln!("gzc: subgroup lane probe failed for W = {w}, BPW = {bpw}; using the sequential K3");
        return Ok(K3Mode::Seq);
    }
    Ok(K3Mode::Coop { w, bpw })
}

/// Dispatches one `@workgroup_size(w * bpw)` workgroup that records each lane's local index,
/// subgroup lane id, subgroup size and `subgroupBallot(true)`; true when every group of `w`
/// lanes is one subgroup with lane id == local index % w, size >= w (== w when bpw > 1) and
/// exactly the w-lane ballot (what `k3_coop.wgsl` assumes).
pub fn probe_lanes(ctx: &GpuContext, w: u32, bpw: u32) -> anyhow::Result<bool> {
    let n = w * bpw;
    let src = format!(
        "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
         @compute @workgroup_size({n})\n\
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
        let buf = ctx.storage_buffer("k3_probe", 5 * 4 * n as u64, true);
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
        let v: Vec<u32> = ctx.read_buffer(&buf, 0, 5 * n as usize);
        let (mx, my) = lane_mask(w);
        Ok(v.chunks(5).enumerate().all(|(i, l)| {
            let size_ok = if bpw > 1 { l[2] == w } else { l[2] >= w };
            l[0] == i as u32 && l[1] == i as u32 % w && size_ok && l[3] == mx && l[4] == my
        }))
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
        // The greedy (`LAZY == 0`) and lazy entry are selected by the injected LAZY constant.
        let k3_body = format!("{best_consts}const MAX_SEQS: u32 = {MAX_SEQS}u;\n{K3_WGSL}\n{K3_LAZY_WGSL}");
        // Both K3 modules are built without naga's forced loop bounding (a per-iteration counter
        // naga adds so the driver may not assume termination; it costs 4 % (lazy2, cooperative),
        // 4 % (lazy2, sequential) and 33 % (greedy) of K3 time on an RTX 5090). Every K3 loop
        // provably ends, whatever best[] holds:
        // - k3_coop.wgsl: each loop has a `// Terminates:` note (variant and bound); the only
        //   subtraction that could wrap, push_lits' end - start, is guarded.
        // - k3_parse.wgsl / k3_lazy.wgsl: the parse loops advance p / ip below PARSE_END (a store
        //   by >= 1 byte, a skip by step >= 1, the deferral by 1-2, the immediate loop by ml >= 4);
        //   match_len's n grows to max <= BLOCK_SIZE - p (p < BLOCK_SIZE); the catch-up's start
        //   falls toward anchor; push_lits has no loop.
        // Bounds checks stay on. A new K3 loop must come with the same argument, or use
        // `ctx.shader` instead.
        let parse = (!is_opt).then(|| {
            pipeline_from_module(ctx, "k3_parse", &parse_layout, &ctx.shader_unbounded_loops("k3_parse", &k3_body), "main")
        });
        let k3_mode = k3_mode(ctx)?;
        let parse_coop = match k3_mode {
            K3Mode::Seq => None,
            K3Mode::Coop { .. } if is_opt => None,
            K3Mode::Coop { w, bpw } => {
                let (mx, my) = lane_mask(w);
                // The greedy rep test's second word: its first min_match - 4 bytes (4..=8).
                let rep_hi = ((1u64 << (8 * (m.min_match - 4))) - 1) as u32;
                // Test-only: GZC_K3_FORCE_FALLBACK=1 makes every workgroup take the in-kernel
                // sequential fallback (the path a failed lane-layout guard takes).
                let force_fallback = std::env::var("GZC_K3_FORCE_FALLBACK").is_ok_and(|v| v == "1");
                let body = format!(
                    "const W: u32 = {w}u;\nconst BPW: u32 = {bpw}u;\nconst W_MASK_X: u32 = {mx}u;\nconst W_MASK_Y: u32 = {my}u;\n\
                     const REP_HI_MASK: u32 = {rep_hi}u;\nconst K3_FORCE_FALLBACK: bool = {force_fallback};\n\
                     {k3_body}\n{K3_COOP_WGSL}"
                );
                // Subgroup built-ins need Features::SUBGROUP on the device (naga 30 rejects
                // `enable subgroups;`).
                let module = ctx.shader_unbounded_loops("k3_coop", &body);
                Some(pipeline_from_module(ctx, "k3_coop", &parse_layout, &module, "main_coop"))
            }
        };
        // Segmented parse: loops as in k3_lazy.wgsl, bounded by the segment's lim instead of
        // BLOCK_SIZE (match_len's n grows to max <= lim - p), the skip by exactly 1; the fixup
        // loops count up to NSEG and to the segments' sequence counts.
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
        // The optimal parse: K3opt's passes with the default build (wg16, the ring in workgroup
        // memory when it fits): wg16 keeps a block's segments in one workgroup (wg % segments ==
        // 0 at 16..64 KiB), which the Prior seed and the cheap passes' histograms need.
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

    /// How K3 runs (see `k3_mode`).
    pub fn k3_mode(&self) -> K3Mode {
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

    /// K1 alone, or K2 alone on the chains already in `bufs.pred` (the E3 overlap probe).
    #[cfg(test)]
    pub(crate) fn record_k1_or_k2(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, bufs: &BatchBuffers, n: u32, k2: bool) {
        assert!(self.sorted.is_none(), "record_k1_or_k2 runs the hash-chain K1/K2 only, not the bucket-sorted finder");
        if !k2 {
            self.chains.record_timed(ctx, enc, &bufs.data, &bufs.head, &bufs.pred, n, None);
            return;
        }
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k2"),
            layout: &self.best_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs.data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs.pred.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bufs.best.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k2"), timestamp_writes: None });
        pass.set_pipeline(&self.best);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((BLOCK_SIZE / 256) as u32, n, 1);
    }

    /// Records K3 alone on whatever blocks (`data`) and matches (`best`) `bufs` holds for its
    /// first `n_blocks` blocks. `n_blocks` is at least 1 and at most `bufs.capacity`. For the
    /// optimal parse K3 is every K3opt pass plus the fix-up, `timestamp_writes` spanning them all;
    /// it errors when `bufs` does not fit them.
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
        let seq = self.parse.as_ref().expect("the sequential K3 exists without opt");
        pass.set_pipeline(self.parse_coop.as_ref().unwrap_or(seq));
        pass.set_bind_group(0, &k3, &[]);
        let per_workgroup = match self.k3_mode {
            K3Mode::Coop { bpw, .. } => bpw,
            K3Mode::Seq => 1,
        };
        pass.dispatch_workgroups(n_blocks.div_ceil(per_workgroup), 1, 1);
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

    /// `record_entropy` with K5 gathering the literals from `lit_src` (`data`'s layout) instead of
    /// `bufs.data`; K4 still reads `bufs.data` (block-level RLE check, Raw fallback).
    fn record_entropy_from(
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
    }
}

/// Decodes one block's readback into a BlockOutput: `seq_words` holds at least `3 * n_seq` words
/// (lit_len, match_len, off_base) and `block` is the block K3 parsed. The literals are gathered
/// from the block (`gather_literals`).
pub fn decode_output(block: &[u8], seq_words: &[u32], n_seq: u32) -> BlockOutput {
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

/// Convenience synchronous path used by tests: upload, run, read back, decode into BlockOutputs.
/// Splits `blocks` (each BLOCK_SIZE bytes) into batches that fit the device limits.
/// wgpu validation and out-of-memory errors while recording/submitting become `Err`.
pub fn compress_batch(ctx: &GpuContext, kernels: &Kernels, blocks: &[&[u8]]) -> anyhow::Result<Vec<BlockOutput>> {
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    let m = kernels.matching();
    let max = max_batch_blocks(&ctx.device.limits(), &m).min(COMPRESS_BATCH_CAP) as usize;
    anyhow::ensure!(max > 0, "device limits too small for one block");

    let scopes = ErrorScopes::push(ctx);
    let bufs = BatchBuffers::new(ctx, blocks.len().min(max) as u32, kernels.emits_frames(), &m)?;
    let mut out = Vec::with_capacity(blocks.len());
    let mut result = Ok(());
    for batch in blocks.chunks(max) {
        let n = batch.len() as u32;
        ctx.queue.write_buffer(&bufs.data, 0, bytemuck::cast_slice(&pack_blocks(batch)));
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("compress_batch") });
        result = kernels.record(ctx, &mut enc, &bufs, n);
        if result.is_err() {
            break;
        }
        ctx.queue.submit([enc.finish()]);
        result = read_outputs(ctx, &bufs, batch, &m, &mut out);
        if result.is_err() {
            break;
        }
    }
    scopes.pop()?;
    result?;
    Ok(out)
}

/// Like `compress_batch`, but returns each block's complete zstd frame as produced by K4.
/// Errors unless `kernels` was built with `emit_frames`.
pub fn compress_frames(ctx: &GpuContext, kernels: &Kernels, blocks: &[&[u8]]) -> anyhow::Result<Vec<Vec<u8>>> {
    anyhow::ensure!(kernels.emits_frames(), "compress_frames needs Kernels built with emit_frames");
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    let m = kernels.matching();
    let max = max_batch_blocks(&ctx.device.limits(), &m).min(COMPRESS_BATCH_CAP) as usize;
    anyhow::ensure!(max > 0, "device limits too small for one block");

    let scopes = ErrorScopes::push(ctx);
    let bufs = BatchBuffers::new(ctx, blocks.len().min(max) as u32, true, &m)?;
    let (frames, frame_len) = (bufs.frames.as_ref().unwrap(), bufs.frame_len.as_ref().unwrap());
    let mut out = Vec::with_capacity(blocks.len());
    let mut result = Ok(());
    for batch in blocks.chunks(max) {
        let n = batch.len() as u32;
        ctx.queue.write_buffer(&bufs.data, 0, bytemuck::cast_slice(&pack_blocks(batch)));
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("compress_frames") });
        result = kernels.record(ctx, &mut enc, &bufs, n);
        if result.is_err() {
            break;
        }
        ctx.queue.submit([enc.finish()]);
        result = read_regions(ctx, &[(frame_len, 0, n as u64), (frames, 0, frames_bytes(n) / 4)]).and_then(|words| {
            let (lens, frames) = words.split_at(n as usize);
            let bytes: &[u8] = bytemuck::cast_slice(frames);
            for (b, &len) in lens.iter().enumerate() {
                out.push(frame_bytes(bytes, b, len)?.to_vec());
            }
            Ok(())
        });
        if result.is_err() {
            break;
        }
    }
    scopes.pop()?;
    result?;
    Ok(out)
}

/// Runs K5 and K4 alone on caller-supplied parses: frame `i` encodes `blocks[i]` (BLOCK_SIZE
/// bytes) with `parses[i]` as its parse, which must cover the block exactly (as `write_frame`
/// requires). Lets tests drive K4/K5 with scripted parses the match finder would never produce.
/// K5 gathers the literals from a separate source holding each parse's literals at their block
/// positions (zeros elsewhere), so a test may pair a parse with a block whose bytes differ from
/// the parse's literals, as `write_frame` allows. At most `max_batch_blocks` blocks (one batch).
pub fn frames_from_parses(
    ctx: &GpuContext,
    kernels: &Kernels,
    blocks: &[&[u8]],
    parses: &[BlockOutput],
) -> anyhow::Result<Vec<Vec<u8>>> {
    anyhow::ensure!(kernels.emits_frames(), "frames_from_parses needs Kernels built with emit_frames");
    anyhow::ensure!(blocks.len() == parses.len(), "one parse per block");
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    let n = blocks.len() as u32;
    let m = kernels.matching();
    anyhow::ensure!(n <= max_batch_blocks(&ctx.device.limits(), &m), "too many blocks for one batch");
    let scopes = ErrorScopes::push(ctx);
    let bufs = BatchBuffers::new(ctx, n, true, &m)?;
    ctx.queue.write_buffer(&bufs.data, 0, bytemuck::cast_slice(&pack_blocks(blocks)));
    let mut lit_blocks = vec![vec![0u8; BLOCK_SIZE]; blocks.len()];
    for (b, p) in parses.iter().enumerate() {
        anyhow::ensure!(p.sequences.len() <= max_seqs(&m) as usize && p.literals.len() <= BLOCK_SIZE, "parse {b} too large");
        // Place the literals at their runs' block positions (see `gather_literals`).
        let mut runs = Vec::with_capacity(p.sequences.len() + 1);
        let mut anchor = 0usize;
        for s in &p.sequences {
            // K4 is built without index clamps (`GpuContext::shader_trusted`); its code tables
            // assume zstd's sequence invariants, which K3 guarantees but a scripted parse may not:
            // match_len >= MINMATCH (3) and 1 <= off_base <= BLOCK_SIZE + 3 (an offset within the
            // block, or a repcode). Lengths are summed in usize (no u32 wrap) and the coverage
            // check below bounds each of them by BLOCK_SIZE.
            anyhow::ensure!(s.match_len >= 3, "parse {b}: match_len {} below 3", s.match_len);
            let ob_ok = s.off_base >= 1 && s.off_base as usize <= BLOCK_SIZE + 3;
            anyhow::ensure!(ob_ok, "parse {b}: off_base {} out of range", s.off_base);
            runs.push((anchor, s.lit_len as usize));
            anchor = anchor.saturating_add(s.lit_len as usize).saturating_add(s.match_len as usize);
        }
        runs.push((anchor, BLOCK_SIZE.saturating_sub(anchor)));
        let mut used = 0usize;
        for (at, len) in runs {
            let ok = at.checked_add(len).is_some_and(|e| e <= BLOCK_SIZE) && used + len <= p.literals.len();
            anyhow::ensure!(ok, "parse {b} does not cover its block");
            lit_blocks[b][at..at + len].copy_from_slice(&p.literals[used..used + len]);
            used += len;
        }
        anyhow::ensure!(used == p.literals.len(), "parse {b} does not cover its block");
        let seqs: Vec<u32> = p.sequences.iter().flat_map(|s| [s.lit_len, s.match_len, s.off_base]).collect();
        let b = b as u64;
        if !seqs.is_empty() {
            ctx.queue.write_buffer(&bufs.seqs, b * seqs_bytes_for(1, &m), bytemuck::cast_slice(&seqs));
        }
        let counts = [p.sequences.len() as u32, p.literals.len() as u32];
        ctx.queue.write_buffer(&bufs.counts, b * counts_bytes(1), bytemuck::cast_slice(&counts));
    }
    let lit_refs: Vec<&[u8]> = lit_blocks.iter().map(|v| v.as_slice()).collect();
    let lit_src = ctx.storage_buffer("frames_from_parses.lit_src", data_bytes(n), false);
    ctx.queue.write_buffer(&lit_src, 0, bytemuck::cast_slice(&pack_blocks(&lit_refs)));
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frames_from_parses") });
    kernels.record_entropy_from(ctx, &mut enc, &bufs, n, None, &lit_src);
    ctx.queue.submit([enc.finish()]);
    let (frames, frame_len) = (bufs.frames.as_ref().unwrap(), bufs.frame_len.as_ref().unwrap());
    let result = read_regions(ctx, &[(frame_len, 0, n as u64), (frames, 0, frames_bytes(n) / 4)]).and_then(|words| {
        let (lens, frames) = words.split_at(n as usize);
        let bytes: &[u8] = bytemuck::cast_slice(frames);
        lens.iter().enumerate().map(|(b, &len)| Ok(frame_bytes(bytes, b, len)?.to_vec())).collect()
    });
    scopes.pop()?;
    result
}

/// Uploads `blocks` and one caller-supplied `best[]` table per block (K2's output layout; entries
/// past the table's end are zero, i.e. no match), then records K3 (and, with `frames`, K5/K4)
/// into a new encoder and submits it. At most `max_batch_blocks` blocks (one batch).
fn submit_from_best(
    ctx: &GpuContext,
    kernels: &Kernels,
    blocks: &[&[u8]],
    bests: &[Vec<Match>],
    frames: bool,
    allow_past_end: bool,
) -> anyhow::Result<BatchBuffers> {
    anyhow::ensure!(blocks.len() == bests.len(), "one best[] table per block");
    anyhow::ensure!(!blocks.is_empty(), "no blocks");
    let n = blocks.len() as u32;
    let m = kernels.matching();
    anyhow::ensure!(m.opt.is_none(), "best[] tables drive K2's parses; the optimal parse takes candidate words (k3opt)");
    anyhow::ensure!(n <= max_batch_blocks(&ctx.device.limits(), &m), "too many blocks for one batch");
    let bufs = BatchBuffers::new(ctx, n, frames, &m)?;
    ctx.queue.write_buffer(&bufs.data, 0, bytemuck::cast_slice(&pack_blocks(blocks)));
    for (b, best) in bests.iter().enumerate() {
        anyhow::ensure!(
            (gzc_core::config::PARSE_END..=BLOCK_SIZE).contains(&best.len()),
            "best[] {b} has {} entries, not in PARSE_END..=BLOCK_SIZE",
            best.len()
        );
        // Every non-empty (len > 0) scripted entry must be a match the K3 kernels could actually
        // have produced: a live source before `ip` and a match that stays inside the block.
        // K3 does not itself bounds-check `best[]` (it trusts K2's output), so a bad scripted
        // entry from a test would otherwise read/write out of bounds on the GPU.
        for (ip, entry) in best.iter().enumerate() {
            if entry.len == 0 {
                continue;
            }
            anyhow::ensure!(entry.offset > 0, "best[] {b}[{ip}] has len {} but offset 0", entry.len);
            anyhow::ensure!(
                entry.offset as usize <= ip,
                "best[] {b}[{ip}] offset {} is past ip {ip}",
                entry.offset
            );
            anyhow::ensure!(
                allow_past_end || ip + entry.len as usize <= BLOCK_SIZE,
                "best[] {b}[{ip}] ip {ip} + len {} > BLOCK_SIZE {BLOCK_SIZE}",
                entry.len
            );
            anyhow::ensure!(
                entry.len <= m.search_cap,
                "best[] {b}[{ip}] len {} > search_cap {}",
                entry.len,
                m.search_cap
            );
        }
        let words = encode_best(best)?;
        ctx.queue.write_buffer(&bufs.best, b as u64 * best_bytes(1), bytemuck::cast_slice(&words));
    }
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("from_best") });
    if ctx.poisoning() {
        // Everything but the uploaded blocks (+ trailing word) and best[] tables.
        ctx.poison_workgroup_memory(&mut enc);
        ctx.poison_from(&mut enc, &bufs.data, data_bytes(n));
        ctx.poison_from(&mut enc, &bufs.best, best_bytes(n));
        ctx.poison_from(&mut enc, &bufs.seqs, 0);
        ctx.poison_from(&mut enc, &bufs.counts, 0);
    }
    kernels.record_parse(ctx, &mut enc, &bufs, n, None)?;
    if frames {
        kernels.record_entropy(ctx, &mut enc, &bufs, n, None);
    }
    ctx.queue.submit([enc.finish()]);
    Ok(bufs)
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

/// Uploads blocks plus caller-supplied best[] tables, runs K3 (+K5+K4), returns frames. For
/// branch-targeted tests: `bests[i]` (at least `PARSE_END` entries, missing ones are "no match")
/// replaces K1/K2's output for `blocks[i]`, so the parse sees exactly the scripted matches. Errors
/// unless `kernels` was built with `emit_frames`. At most `max_batch_blocks` blocks (one batch).
pub fn frames_from_best(
    ctx: &GpuContext,
    kernels: &Kernels,
    blocks: &[&[u8]],
    bests: &[Vec<Match>],
) -> anyhow::Result<Vec<Vec<u8>>> {
    anyhow::ensure!(kernels.emits_frames(), "frames_from_best needs Kernels built with emit_frames");
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    with_error_scopes(ctx, || {
        let bufs = submit_from_best(ctx, kernels, blocks, bests, true, false)?;
        let n = blocks.len() as u32;
        let (frames, frame_len) = (bufs.frames.as_ref().unwrap(), bufs.frame_len.as_ref().unwrap());
        let words = read_regions(ctx, &[(frame_len, 0, n as u64), (frames, 0, frames_bytes(n) / 4)])?;
        let (lens, frames) = words.split_at(n as usize);
        let bytes: &[u8] = bytemuck::cast_slice(frames);
        lens.iter().enumerate().map(|(b, &len)| Ok(frame_bytes(bytes, b, len)?.to_vec())).collect()
    })
}

/// `frames_from_best`, returning K3's parse of each block instead of its frame (works with or
/// without `emit_frames`). Lets tests compare the parse itself, which a Raw frame would hide.
pub fn parses_from_best(
    ctx: &GpuContext,
    kernels: &Kernels,
    blocks: &[&[u8]],
    bests: &[Vec<Match>],
) -> anyhow::Result<Vec<BlockOutput>> {
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    with_error_scopes(ctx, || {
        let bufs = submit_from_best(ctx, kernels, blocks, bests, false, false)?;
        let mut out = Vec::with_capacity(blocks.len());
        read_outputs(ctx, &bufs, blocks, &kernels.matching(), &mut out)?;
        Ok(out)
    })
}

/// Test-only: `parses_from_best` without the check that a scripted match ends inside the block
/// (K2 never writes such an entry). Lets tests check that K3 still terminates when an anchor
/// lands past BLOCK_SIZE; the parse it returns (or the error from its garbage counts) has no
/// CPU reference.
#[doc(hidden)]
pub fn parses_from_best_unchecked(
    ctx: &GpuContext,
    kernels: &Kernels,
    blocks: &[&[u8]],
    bests: &[Vec<Match>],
) -> anyhow::Result<Vec<BlockOutput>> {
    with_error_scopes(ctx, || {
        let bufs = submit_from_best(ctx, kernels, blocks, bests, false, true)?;
        let mut out = Vec::with_capacity(blocks.len());
        read_outputs(ctx, &bufs, blocks, &kernels.matching(), &mut out)?;
        Ok(out)
    })
}

/// Runs K1 and K2 on `blocks` (one batch, at most `max_batch_blocks`) and returns each block's
/// `best[]` table (BLOCK_SIZE entries, K2's layout decoded), for tests that check K2 against
/// `reference::find_best` directly.
pub fn best_from_blocks(ctx: &GpuContext, kernels: &Kernels, blocks: &[&[u8]]) -> anyhow::Result<Vec<Vec<Match>>> {
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    let n = blocks.len() as u32;
    let m = kernels.matching();
    anyhow::ensure!(m.opt.is_none(), "best_from_blocks reads K2's best[] words (opt: cands_from_blocks)");
    anyhow::ensure!(n <= max_batch_blocks(&ctx.device.limits(), &m), "too many blocks for one batch");
    with_error_scopes(ctx, || {
        let bufs = BatchBuffers::new(ctx, n, false, &m)?;
        ctx.queue.write_buffer(&bufs.data, 0, bytemuck::cast_slice(&pack_blocks(blocks)));
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("best_from_blocks") });
        kernels.record_best(ctx, &mut enc, &bufs, n, None);
        ctx.queue.submit([enc.finish()]);
        let words = read_regions(ctx, &[(&bufs.best, 0, best_bytes(n) / 4)])?;
        Ok(words.chunks(best_bytes(1) as usize / 4).map(decode_best).collect())
    })
}

/// K1 (the `Opt3` h4 + h3 chains) and K2opt (`k2_opt.wgsl`, == `reference::find_cands`): the
/// optimal parse's candidate words, two per position, in the `best` buffer
/// (`best_bytes_for(n, m)`, layout `[block][pos][2]`, `reference::CandWords`).
pub struct OptCandKernel {
    chains: ChainsKernel,
    cands: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    matching: MatchParams,
}

impl OptCandKernel {
    /// Builds K1 and K2opt for `m`. Errors unless `m` is valid with `opt` set. Independent of
    /// `gpu_supports` (it builds only the candidate stage, not the rest of the opt pipeline:
    /// K3opt's passes, K5, K4), so tests and benches can run the candidate stage alone.
    pub fn new(ctx: &GpuContext, m: &MatchParams) -> anyhow::Result<Self> {
        m.validate().map_err(|e| anyhow!("invalid match params {m:?}: {e}"))?;
        anyhow::ensure!(m.opt.is_some(), "OptCandKernel needs opt params, got {m:?}");
        let chains = ChainsKernel::new(ctx, m)?;
        let layout = storage_layout(ctx, "k2opt", &[true, true, false]);
        let cands = k2_opt_pipeline(ctx, m, &layout);
        Ok(Self { chains, cands, layout, matching: *m })
    }

    /// The match params the kernels were built for.
    pub fn matching(&self) -> MatchParams {
        self.matching
    }

    /// True when K1 is the subgroup kernel.
    pub fn uses_subgroups(&self) -> bool {
        self.chains.uses_subgroups()
    }

    /// Records K1 then K2opt for the first `n_blocks` blocks of `data`: K1 into `head`/`pred`
    /// (`chains::head_bytes`/`pred_bytes` for 2 chains), the candidates into `cands` (at least
    /// `best_bytes_for(n_blocks, m)`). `ts(0)` / `ts(1)` are K1's / K2opt's timestamp writes.
    /// `n_blocks <= max_batch_blocks(.., m)`.
    #[allow(clippy::too_many_arguments)]
    pub fn record_timed<'q>(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        data: &wgpu::Buffer,
        head: &wgpu::Buffer,
        pred: &wgpu::Buffer,
        cands: &wgpu::Buffer,
        n_blocks: u32,
        ts: impl Fn(u32) -> Option<wgpu::ComputePassTimestampWrites<'q>>,
    ) {
        if n_blocks == 0 {
            return;
        }
        assert!(cands.size() >= best_bytes_for(n_blocks, &self.matching), "cands buffer too small");
        self.chains.record_timed(ctx, enc, data, head, pred, n_blocks, ts(0));
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k2opt"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pred.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cands.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("k2opt"), timestamp_writes: ts(1) });
        pass.set_pipeline(&self.cands);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((BLOCK_SIZE / 256) as u32, n_blocks, 1);
    }
}

/// K2opt (`k2_opt.wgsl`'s `main_opt`) for opt params `m`, on a (data, pred, cands) layout.
fn k2_opt_pipeline(ctx: &GpuContext, m: &MatchParams, layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let body = format!(
        "{}const BEST_OFF_BITS: u32 = {BEST_OFF_BITS}u;\nconst H3_DEPTH: u32 = {OPT_H3_DEPTH}u;\n{K2_WGSL}\n{K2_OPT_WGSL}",
        params_wgsl(m)
    );
    // Loops: the merged walk decrements a depth counter every iteration (DEPTH + H3_DEPTH at
    // most) and match_len_capped is bounded by SEARCH_CAP; indices as in K2 (pred words hold
    // positions below HASHED_POSITIONS, from K1 in the same submission).
    let module = ctx.shader_trusted("k2_opt", &body);
    pipeline_from_module(ctx, "k2_opt", layout, &module, "main_opt")
}

/// Runs K1 + K2opt on `blocks` (in batches of at most `max_batch_blocks`) and returns each
/// block's candidate words (BLOCK_SIZE entries), for tests that check K2opt against
/// `reference::find_cands` directly.
pub fn cands_from_blocks(ctx: &GpuContext, kernel: &OptCandKernel, blocks: &[&[u8]]) -> anyhow::Result<Vec<Vec<CandWords>>> {
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    let m = kernel.matching();
    let max = max_batch_blocks(&ctx.device.limits(), &m).min(COMPRESS_BATCH_CAP) as usize;
    anyhow::ensure!(max > 0, "device limits too small for one block");
    with_error_scopes(ctx, || {
        let cap = blocks.len().min(max) as u32;
        let nh = m.n_hashes();
        let data = ctx.storage_buffer("cands.data", data_bytes(cap), false);
        let head = ctx.storage_buffer("cands.head", head_bytes(cap, nh), false);
        let pred = ctx.storage_buffer("cands.pred", pred_bytes(cap, nh), false);
        let cands = ctx.storage_buffer("cands.cands", best_bytes_for(cap, &m), true);
        let mut out = Vec::with_capacity(blocks.len());
        for batch in blocks.chunks(max) {
            let n = batch.len() as u32;
            ctx.queue.write_buffer(&data, 0, bytemuck::cast_slice(&pack_blocks(batch)));
            let mut enc =
                ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cands_from_blocks") });
            kernel.record_timed(ctx, &mut enc, &data, &head, &pred, &cands, n, |_| None);
            ctx.queue.submit([enc.finish()]);
            let words = read_regions(ctx, &[(&cands, 0, best_bytes_for(n, &m) / 4)])?;
            out.extend(words.chunks_exact(2 * BLOCK_SIZE).map(|b| b.chunks_exact(2).map(|w| [w[0], w[1]]).collect::<Vec<_>>()));
        }
        Ok(out)
    })
}

/// Decodes one block's `best[]` words (K2's layout) into matches.
pub fn decode_best(words: &[u32]) -> Vec<Match> {
    let mask = (1 << BEST_OFF_BITS) - 1;
    words.iter().map(|&w| Match { offset: w & mask, len: w >> BEST_OFF_BITS }).collect()
}

/// Encodes one block's matches into `best[]` words (K2's layout), padded with "no match" to
/// BLOCK_SIZE entries. Entries with len 0 encode as 0 (no match), whatever their offset.
/// Every entry must have `offset < BLOCK_SIZE` and `len <= 256` (checked).
pub fn encode_best(best: &[Match]) -> anyhow::Result<Vec<u32>> {
    let mut words = Vec::with_capacity(BLOCK_SIZE);
    for (i, m) in best.iter().enumerate() {
        anyhow::ensure!((m.offset as usize) < BLOCK_SIZE && m.len <= 256, "best[{i}] {m:?} does not fit a best[] word");
        words.push(if m.len == 0 { 0 } else { (m.len << BEST_OFF_BITS) | m.offset });
    }
    words.resize(BLOCK_SIZE, 0);
    Ok(words)
}

/// Block `b`'s frame out of a fixed-stride frames region (`FRAME_STRIDE` bytes per block),
/// checking K4's reported length.
pub fn frame_bytes(frames: &[u8], b: usize, len: u32) -> anyhow::Result<&[u8]> {
    anyhow::ensure!(len > 0 && len as usize <= FRAME_STRIDE, "block {b}: bad frame length {len}");
    let start = b * FRAME_STRIDE;
    Ok(&frames[start..start + len as usize])
}

/// Reads counts, then each block's used seqs region, and decodes them (literals gathered from
/// `blocks`, the batch's blocks) into `out`.
fn read_outputs(
    ctx: &GpuContext,
    bufs: &BatchBuffers,
    blocks: &[&[u8]],
    m: &MatchParams,
    out: &mut Vec<BlockOutput>,
) -> anyhow::Result<()> {
    let n = blocks.len() as u32;
    let counts: Vec<u32> = read_regions(ctx, &[(&bufs.counts, 0, 2 * n as u64)])?;
    let mut regions = Vec::with_capacity(n as usize);
    let max = max_seqs(m);
    for b in 0..n as u64 {
        let (n_seq, n_lit) = (counts[2 * b as usize], counts[2 * b as usize + 1]);
        anyhow::ensure!(n_seq <= max && n_lit as usize <= BLOCK_SIZE, "block {b}: bad counts ({n_seq}, {n_lit})");
        regions.push((&bufs.seqs, seqs_bytes_for(1, m) * b, 3 * n_seq as u64));
    }
    let words = read_regions(ctx, &regions)?;
    let mut at = 0usize;
    for (b, block) in blocks.iter().enumerate() {
        let (n_seq, n_lit) = (counts[2 * b], counts[2 * b + 1]);
        let seq_end = at + 3 * n_seq as usize;
        let parse = decode_output(block, &words[at..seq_end], n_seq);
        let got = parse.literals.len();
        anyhow::ensure!(got == n_lit as usize, "block {b}: K3 counted {n_lit} literals, the sequences leave {got}");
        out.push(parse);
        at = seq_end;
    }
    Ok(())
}

/// Copies `(buffer, byte offset, word count)` regions into one staging buffer, waits, and
/// returns their words concatenated in order.
fn read_regions(ctx: &GpuContext, regions: &[(&wgpu::Buffer, u64, u64)]) -> anyhow::Result<Vec<u32>> {
    let total: u64 = regions.iter().map(|r| r.2 * 4).sum();
    if total == 0 {
        return Ok(Vec::new());
    }
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("compress_batch.readback"),
        size: total,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
    let mut dst = 0u64;
    for &(buf, offset, words) in regions {
        if words > 0 {
            enc.copy_buffer_to_buffer(buf, offset, &staging, dst, words * 4);
            dst += words * 4;
        }
    }
    ctx.queue.submit([enc.finish()]);

    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    ctx.wait_callback(&rx, None, ctx.poll_only())?.context("map readback buffer")?;
    let words = {
        let view = staging.get_mapped_range(..).context("mapped range")?;
        bytemuck::pod_collect_to_vec(&view[..])
    };
    staging.unmap();
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gzc_core::params::{LVL3, LVL9, RUNG1, RUNG2};

    const MIB: u64 = 1 << 20;

    fn limits(binding: u64, buffer: u64, wg: u32) -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffer_binding_size: binding,
            max_buffer_size: buffer,
            max_compute_workgroups_per_dimension: wg,
            ..wgpu::Limits::default()
        }
    }

    /// Every batch buffer for `n` blocks with `nh` chains fits `limit`.
    fn fits(n: u32, nh: u32, limit: u64) -> bool {
        [
            data_bytes(n),
            head_bytes(n, nh),
            pred_bytes(n, nh),
            best_bytes(n),
            seqs_bytes(n),
            counts_bytes(n),
            frames_bytes(n),
        ]
        .iter()
        .all(|&b| b <= limit)
    }

    #[test]
    fn max_seqs_bounds_minimal_sequences() {
        assert_eq!(MAX_SEQS as usize, BLOCK_SIZE / 4 + 1);
        assert_eq!(MAX_SEQS_OPT as usize, BLOCK_SIZE / 3 + 1);
        #[cfg(feature = "block-64k")]
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
        assert_eq!(seqs_bytes_for(7, &LVL9), seqs_bytes(7));
    }

    #[test]
    fn gpu_supports_all_presets() {
        for (name, p) in gzc_core::params::PRESETS {
            // M5 T5: the optimal-parse presets too (at blocks of at most 64 KiB); M6's opt16p1
            // not until its K3/drop kernels land.
            let m5_opt = p.opt.is_none_or(|o| o.is_m5() && BLOCK_SIZE <= 1 << 16);
            assert_eq!(gpu_supports(&p), m5_opt, "{name}");
            if gpu_supports(&p) {
                check_matching(&p).unwrap_or_else(|e| panic!("{name}: {e}"));
            }
        }
        assert!(gpu_supports(&MatchParams { depth: 4, ..LVL3 }));
        assert!(gpu_supports(&MatchParams { min_match: 8, depth: 64, ..RUNG1 }));
        assert!(gpu_supports(&MatchParams { min_match: 6, ..RUNG2 }));
        let bad = MatchParams { lazy: 3, ..LVL9 };
        assert!(!gpu_supports(&bad));
        assert!(check_matching(&bad).unwrap_err().to_string().contains("lazy 3"));
    }

    /// `record` (via `record_front`) rejects `BatchBuffers` whose `opt` scratch does not match
    /// these `Kernels`' opt-ness, in both directions, before recording any dispatch.
    #[test]
    fn record_front_rejects_mismatched_opt_buffers() {
        let _gpu = crate::test_support::gpu_test_slot();
        // opt14/opt16 only implement at blocks of at most 64 KiB.
        if BLOCK_SIZE > 1 << 16 {
            return;
        }
        let ctx = GpuContext::new().expect("GPU required for gzc-gpu tests");
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
        for m in [LVL3, RUNG1, RUNG2, LVL9] {
            let nh = m.n_hashes();
            for limit in [4 * MIB, 128 * MIB, 1 << 31, (1 << 32) - 4] {
                let n = max_batch_blocks(&limits(limit, u64::MAX, 65535), &m);
                assert!(n > 0 && fits(n, nh, limit) && !fits(n + 1, nh, limit), "{m:?} limit {limit}: n {n}");
                // max_buffer_size is honoured the same way as the binding size.
                assert_eq!(max_batch_blocks(&limits(u64::MAX, limit, 65535), &m), n);
            }
        }
        let at_128m = |m: &MatchParams| max_batch_blocks(&limits(128 * MIB, 128 * MIB, 65535), m);
        // 128K: pred 1 MiB per block with two chains; one chain: pred/best 512 KiB.
        #[cfg(feature = "block-128k")]
        assert_eq!((at_128m(&LVL3), at_128m(&RUNG1)), (128, 256));
        // 64K (default) and 32K: pred 512 / 256 KiB per block with two chains.
        #[cfg(feature = "block-64k")]
        assert_eq!((at_128m(&LVL3), at_128m(&RUNG1)), (256, 512));
        #[cfg(feature = "block-32k")]
        assert_eq!((at_128m(&LVL3), at_128m(&RUNG1)), (512, 1024));
        // 16K: pred 128 KiB per block with two chains, pred/best 64 KiB with one (head is capped at
        // chains::HEAD_TABLES tables).
        #[cfg(feature = "block-16k")]
        assert_eq!((at_128m(&LVL3), at_128m(&RUNG1)), (1024, 2048));
    }

    /// The optimal parse's `best` buffer holds two candidate words per position (8 B), which
    /// `max_batch_blocks`, `BatchBuffers` and `scratch_bytes` (so `vram_bytes`) all count.
    #[test]
    fn opt_batch_counts_two_candidate_words() {
        use gzc_core::params::{OPT14, OPT16};
        assert_eq!((best_words(&LVL9), best_words(&OPT16), best_words(&OPT14)), (1, 2, 2));
        assert_eq!(best_bytes_for(3, &OPT16), 3 * 8 * BLOCK_SIZE as u64);
        // Against lvl3 (also two chains): the second candidate word, the larger seqs, and K3opt's
        // prices (377 words) and scratch per block. The trace reuses pred (8 B per position).
        let k3opt = crate::k3opt::prices_bytes(10) + 10 * crate::k3opt::scratch_bytes_per_block(&OPT16);
        assert_eq!(pred_bytes_for(10, &OPT16), trace_bytes(10));
        assert_eq!(pred_bytes_for(10, &OPT16), pred_bytes(10, 2));
        assert_eq!(opt_bytes(10, &OPT16), k3opt);
        assert_eq!((opt_bytes(10, &LVL3), crate::k3opt::prices_bytes(1)), (0, 377 * 4));
        assert_eq!(
            scratch_bytes(10, &OPT16) - scratch_bytes(10, &LVL3),
            best_bytes(10) + seqs_bytes_for(10, &OPT16) - seqs_bytes(10) + k3opt
        );
        #[cfg(feature = "block-64k")]
        assert_eq!(crate::k3opt::scratch_bytes_per_block(&OPT16), 6336);
        for limit in [4 * MIB, 128 * MIB, 1 << 31, (1 << 32) - 4] {
            let n = max_batch_blocks(&limits(limit, u64::MAX, 65535), &OPT16);
            let fits = |n: u32| {
                fits(n, 2, limit)
                    && [best_bytes_for(n, &OPT16), seqs_bytes_for(n, &OPT16), opt_bytes(n, &OPT16)]
                        .iter()
                        .all(|&b| b <= limit)
            };
            assert!(n > 0 && fits(n) && !fits(n + 1), "limit {limit}: n {n}");
        }
        let n = max_batch_blocks(&limits(u64::MAX, u64::MAX, u32::MAX), &OPT16) as u64;
        assert!(n > 0 && n * 2 * BLOCK_SIZE as u64 <= 1 << 32, "cands / trace word index");
        assert!(n * 3 * MAX_SEQS_OPT as u64 <= 1 << 32, "seqs word index");
    }

    #[test]
    fn batch_respects_workgroup_cap() {
        for m in [LVL3, RUNG1, RUNG2, LVL9] {
            assert_eq!(max_batch_blocks(&limits(128 * MIB, 128 * MIB, 3), &m), 3);
        }
    }

    #[test]
    fn batch_keeps_u32_indices_in_range() {
        for m in [LVL3, RUNG1, RUNG2, LVL9] {
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
        for m in [LVL3, RUNG1, RUNG2, LVL9] {
            assert_eq!(max_batch_blocks(&limits(BLOCK_SIZE as u64, u64::MAX, 65535), &m), 0);
        }
    }

    #[test]
    fn best_words_round_trip_at_the_bounds() {
        let last = BLOCK_SIZE as u32 - 1;
        let best = [
            Match { offset: 1, len: 4 },
            Match { offset: last, len: 256 },
            Match { offset: last, len: 1 },
            Match::default(),
            Match { offset: 7, len: 0 },
        ];
        let words = encode_best(&best).unwrap();
        assert_eq!(words.len(), BLOCK_SIZE);
        assert_eq!(words[4], 0, "len 0 is no match whatever the offset");
        let back = decode_best(&words);
        assert_eq!(back[..4], best[..4]);
        assert!(back[4..].iter().all(|m| *m == Match::default()));
        assert!(encode_best(&[Match { offset: BLOCK_SIZE as u32, len: 4 }]).is_err());
        assert!(encode_best(&[Match { offset: 1, len: 257 }]).is_err());
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
