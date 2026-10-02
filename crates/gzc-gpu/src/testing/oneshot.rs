//! One-shot harnesses for the tests: upload a batch, run some of the kernels, read the result
//! back. Each call allocates its own buffers and waits for the GPU. The streaming `Pipeline` is
//! the production path.
use anyhow::{Context as _, anyhow};
use gzc_core::config::BLOCK_SIZE;
use gzc_core::params::MatchParams;
use gzc_core::reference::{CandWords, Match};
use gzc_core::seq::BlockOutput;

use crate::chains::ChainsKernel;
use crate::context::{ErrorScopes, GpuContext, pack_blocks};
use crate::kernels::{
    BEST_OFF_BITS, BatchBuffers, FRAME_STRIDE, Kernels, decode_output, k2_opt_pipeline, max_seqs, storage_layout,
    with_error_scopes,
};
use crate::sizing::{
    best_bytes, best_bytes_for, chain_pred_bytes, counts_bytes, data_bytes, frames_bytes, head_bytes, max_batch_blocks,
    seqs_bytes_for,
};

/// Largest batch `compress_batch`/`compress_frames` allocate buffers for, even when the device
/// limits would allow more. The worst case is dfast's two hash chains with `emit_frames`: every
/// `sizing::BufferSizes` buffer, with `head` capped at `chains::HEAD_TABLES` tables, comes to
/// about 1.6 MiB per block, 200 MiB at this cap. These one-shot paths serve the tests, several
/// of which run at once, each with its own device. 128 splits their 300-block batches into full
/// and partial batches.
const COMPRESS_BATCH_CAP: u32 = 128;

/// Convenience synchronous path used by tests: upload, run, read back, decode into BlockOutputs.
/// Splits `blocks` (each BLOCK_SIZE bytes) into batches that fit the device limits.
/// wgpu validation and out-of-memory errors while recording/submitting become `Err`.
pub fn compress_batch(ctx: &GpuContext, kernels: &Kernels, blocks: &[&[u8]]) -> anyhow::Result<Vec<BlockOutput>> {
    crate::pipeline::check_blocks(blocks)?;
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
    crate::pipeline::check_blocks(blocks)?;
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
        // Every non-empty (len > 0) scripted entry must be a match K2 could have produced: a
        // live source before `ip` and a match that stays inside the block. K3 trusts K2's output
        // and does not bounds-check `best[]`, so a bad scripted entry would read or write out of
        // bounds on the GPU.
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
    /// (`sizing::head_bytes` for `m.n_hashes()` chains, `chain_pred_bytes`), the candidates into `cands` (at least
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
        let pred = ctx.storage_buffer("cands.pred", chain_pred_bytes(cap, &m), false);
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
    use crate::kernels::GpuParams;
    use gzc_core::params::LVL3;

    /// The one-shot paths return an error, not a panic, on a block that is not BLOCK_SIZE bytes.
    #[test]
    fn one_shot_paths_reject_wrong_size_blocks() {
        let _gpu = crate::testing::gpu_test_slot();
        let ctx = crate::testing::gpu();
        let kernels = Kernels::new(&ctx, GpuParams { matching: LVL3, emit_frames: true, huffman: true }).unwrap();
        let full = vec![0u8; BLOCK_SIZE];
        for bad in [vec![0u8; 3], vec![0u8; BLOCK_SIZE + 1]] {
            let blocks = [full.as_slice(), bad.as_slice()];
            let e = compress_frames(&ctx, &kernels, &blocks).unwrap_err().to_string();
            assert!(e.contains("block 1"), "{e}");
            let e = compress_batch(&ctx, &kernels, &blocks).unwrap_err().to_string();
            assert!(e.contains("block 1"), "{e}");
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
}
