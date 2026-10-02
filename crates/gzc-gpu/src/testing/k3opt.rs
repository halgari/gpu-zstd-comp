//! K3opt on its own: the passes' kernels with buffers of their own (`OptBuffers`) and candidate
//! words uploaded from the host, so the tests can run a pass on `reference::find_cands` output or
//! on scripted candidates, read back each pass's parse and histogram, and time the passes.
use anyhow::{anyhow, ensure};
use gzc_core::config::BLOCK_SIZE;
use gzc_core::opt::{Hist, Prices};
use gzc_core::params::MatchParams;
use gzc_core::reference::CandWords;
use gzc_core::seq::BlockOutput;

use crate::context::{GpuContext, pack_blocks};
use crate::k3opt::{OptBinds, PRICE_WORDS};
pub use crate::k3opt::{
    K3Drop, K3Opt, K3OptConfig, OptPasses, PriceSrc, SCHED_HDR, WEIGHT_RUN, WEIGHT_STRIDE, ring_bytes, check_ring_fits,
    scratch_bytes_per_block, workgroup_bytes,
};
use crate::kernels::{MAX_SEQS_OPT, decode_output, with_error_scopes};
use crate::sizing::{
    best_bytes_for, counts_bytes, data_bytes, prices_bytes, sched_bytes, seqs_bytes_for, trace_bytes,
};

/// Largest batch `parses_from_cands` runs at once.
const BATCH_CAP: usize = 256;

/// K3opt's buffers for `capacity` blocks (the harnesses' own set; the pipeline binds its
/// `BatchBuffers` through `OptBinds::of_batch`).
pub struct OptBuffers {
    pub capacity: u32,
    /// The opt params the buffers were sized for.
    pub params: MatchParams,
    pub data: wgpu::Buffer,
    /// Candidate words `[block][pos][2]`; the pass's raw sequences afterwards.
    pub cands: wgpu::Buffer,
    /// The DP trace `[block][pos][2]`.
    pub trace: wgpu::Buffer,
    pub seqs: wgpu::Buffer,
    pub counts: wgpu::Buffer,
    /// `PRICE_WORDS` words per block: `opt::Prices` (`PriceSrc::Buffer`) or `opt::Hist`
    /// (`PriceSrc::Hist`, `hist_out`), both as `lit[256]` `ll[36]` `ml[53]` `of[32]`.
    pub prices: wgpu::Buffer,
    /// The DP nodes' payload (`scratch_bytes_per_block` per block; K3opt-owned, dead between
    /// passes).
    pub scratch: wgpu::Buffer,
    scratch_per_block: u64,
    /// The block schedule (`sched_bytes`, M6 A4).
    pub sched: wgpu::Buffer,
}

impl OptBuffers {
    /// Buffers for `capacity` blocks under opt params `m` (the scratch's size depends on the
    /// segments per block and `target_length`). Errors when an allocation fails (out of memory, a
    /// buffer above the device's limits, or a lost device).
    pub fn new(ctx: &GpuContext, m: &MatchParams, capacity: u32) -> anyhow::Result<Self> {
        let scratch_per_block = scratch_bytes_per_block(m);
        let scopes = crate::context::ErrorScopes::push(ctx);
        let bufs = Self {
            capacity,
            params: *m,
            data: ctx.storage_buffer("k3opt.data", data_bytes(capacity), false),
            cands: ctx.storage_buffer("k3opt.cands", best_bytes_for(capacity, m), true),
            trace: ctx.storage_buffer("k3opt.trace", trace_bytes(capacity), false),
            seqs: ctx.storage_buffer("k3opt.seqs", seqs_bytes_for(capacity, m), true),
            counts: ctx.storage_buffer("k3opt.counts", counts_bytes(capacity), true),
            prices: ctx.storage_buffer("k3opt.prices", prices_bytes(capacity), true),
            scratch: ctx.storage_buffer("k3opt.scratch", capacity as u64 * scratch_per_block, false),
            scratch_per_block,
            sched: ctx.storage_buffer("k3opt.sched", sched_bytes(capacity), true),
        };
        let bytes = [&bufs.data, &bufs.cands, &bufs.trace, &bufs.seqs, &bufs.counts, &bufs.prices, &bufs.scratch, &bufs.sched]
            .iter()
            .map(|b| b.size())
            .sum();
        scopes.pop_alloc(&format!("the k3opt buffers ({capacity} blocks)"), bytes)?;
        Ok(bufs)
    }

    /// The buffers as bindings.
    pub fn binds(&self) -> OptBinds<'_> {
        OptBinds {
            capacity: self.capacity,
            data: &self.data,
            cands: &self.cands,
            trace: &self.trace,
            seqs: &self.seqs,
            counts: &self.counts,
            prices: &self.prices,
            scratch: &self.scratch,
            scratch_per_block: self.scratch_per_block,
            sched: &self.sched,
        }
    }

    /// Uploads blocks, their candidate words and (with `prices`) their price tables.
    pub fn upload(
        &self,
        ctx: &GpuContext,
        blocks: &[&[u8]],
        cands: &[&[CandWords]],
        prices: Option<&[Prices]>,
    ) -> anyhow::Result<()> {
        ensure!(
            blocks.len() == cands.len() && blocks.len() <= self.capacity as usize,
            "bad batch"
        );
        ctx.queue
            .write_buffer(&self.data, 0, bytemuck::cast_slice(&pack_blocks(blocks)));
        check_dead_marks_par(blocks, cands)?;
        for (b, c) in cands.iter().enumerate() {
            ensure!(
                c.len() == BLOCK_SIZE,
                "block {b}: {} candidate words",
                c.len()
            );
            // The kernel trusts the words (no bounds checks on offsets): each record must lie in
            // the block, before its position.
            for (p, w) in c.iter().enumerate() {
                // w1's high half holds DEAD_BIT and nothing else (the kernel reads offB as
                // `w1 & 0xFFFF` and the flag as `w1 & DEAD_BIT`; another bit there would be a
                // format change neither checks for).
                ensure!(
                    w[1] >> 16 & !(gzc_core::reference::DEAD_BIT >> 16) == 0,
                    "block {b}: candidate word 1 at {p} has bits besides DEAD_BIT in its high half: {:#x}",
                    w[1]
                );
                let (a, bb) = gzc_core::reference::unpack_cands(*w);
                for r in [a, bb] {
                    ensure!(
                        r.len == 0 || (r.offset >= 1 && r.offset as usize <= p),
                        "block {b}: bad candidate at {p}: {r:?}"
                    );
                }
            }
            ctx.queue.write_buffer(
                &self.cands,
                b as u64 * best_bytes_for(1, &self.params),
                bytemuck::cast_slice(c),
            );
        }
        if let Some(ps) = prices {
            ensure!(ps.len() == blocks.len(), "one price table per block");
            let mut words = Vec::with_capacity(ps.len() * PRICE_WORDS);
            for p in ps {
                let all = p.lit.iter().chain(&p.ll).chain(&p.ml).chain(&p.of);
                for &x in all {
                    ensure!((0..65536).contains(&x), "price {x} out of range");
                    words.push(x as u32);
                }
            }
            ctx.queue
                .write_buffer(&self.prices, 0, bytemuck::cast_slice(&words));
        }
        Ok(())
    }
}

/// `check_dead_marks` on every block, on up to 16 threads (a timing harness re-uploads between
/// runs, and a long upload lets the GPU clock down).
fn check_dead_marks_par(blocks: &[&[u8]], cands: &[&[CandWords]]) -> anyhow::Result<()> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| -> anyhow::Result<()> {
                    let mut seen = Vec::new();
                    loop {
                        let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let (Some(bl), Some(c)) = (blocks.get(b), cands.get(b)) else { return Ok(()) };
                        ensure!(c.len() == BLOCK_SIZE, "block {b}: {} candidate words", c.len());
                        check_dead_marks(bl, c, &mut seen).map_err(|e| anyhow!("block {b}: {e}"))?;
                    }
                })
            })
            .collect();
        workers.into_iter().try_for_each(|w| w.join().expect("dead-mark check panicked"))
    })
}

/// The kernel also trusts the dead marks (`reference::find_cands`, M6 A3): it skips the search at
/// every marked position. Checks that each marked position `p` is really dead: words `[0,
/// DEAD_BIT]` (no record), `p < PARSE_END`, and no earlier position with its first 3 bytes.
/// `seen` is the caller's scratch (empty, or all zero as this leaves it).
fn check_dead_marks(block: &[u8], c: &[CandWords], seen: &mut Vec<u64>) -> anyhow::Result<()> {
    use gzc_core::config::PARSE_END;
    use gzc_core::reference::{DEAD_BIT, is_dead};
    if !c.iter().any(|w| is_dead(*w)) {
        return Ok(());
    }
    ensure!(block.len() >= PARSE_END + 3, "dead marks on a short block");
    // Seen 3-byte prefixes, a 2^24-bit set (cleared again below, one word per position).
    seen.resize(1 << 18, 0);
    let key = |p: usize| block[p] as usize | (block[p + 1] as usize) << 8 | (block[p + 2] as usize) << 16;
    let mut res = Ok(());
    for (p, &w) in c.iter().enumerate() {
        let dead = is_dead(w);
        if dead && (p >= PARSE_END || w != [0, DEAD_BIT]) {
            res = Err(anyhow!("bad dead word at {p}: {w:?}"));
            break;
        }
        if p < PARSE_END {
            let k = key(p);
            if dead && seen[k >> 6] & 1 << (k & 63) != 0 {
                res = Err(anyhow!("dead position {p} has an earlier 3-byte match"));
                break;
            }
            seen[k >> 6] |= 1 << (k & 63);
        }
    }
    for p in 0..PARSE_END {
        seen[key(p) >> 6] = 0;
    }
    res
}

/// The words a final K3opt pass and its fix-up leave for `K3Drop` when their parse is `out`
/// (`BLOCK_SIZE`-byte segments of `1 << seg_log2`): `best` (`2 * BLOCK_SIZE` words: each
/// segment's raw sequences, last first, with off_bases under the segment's own history from
/// `INITIAL_REPS` (segment 0) or `[0, 0, 0]`, and its 6-word trailer at `SEG_META`), the parse's
/// `seqs` words and its counts. For replaying a scripted drop input (`opt::cases::DropCase`).
/// Every match must lie in one segment.
pub(crate) fn final_pass_words(out: &BlockOutput, seg_log2: u32) -> (Vec<u32>, Vec<u32>, [u32; 2]) {
    use gzc_core::seq::{INITIAL_REPS, apply_off_base, off_base_for};
    let seg = 1usize << seg_log2;
    let seg_words = 2 * seg;
    let meta = seg_words - 6;
    let mut segs: Vec<Vec<(usize, u32, u32)>> = vec![Vec::new(); BLOCK_SIZE / seg];
    let (mut r, mut pos) = (INITIAL_REPS, 0usize);
    let mut ml_sum = 0u32;
    for s in &out.sequences {
        let off = apply_off_base(&mut r, s.off_base, s.lit_len);
        let start = pos + s.lit_len as usize;
        pos = start + s.match_len as usize;
        assert_eq!(start / seg, (pos - 1) / seg, "a match crosses a segment");
        segs[start / seg].push((start, s.match_len, off));
        ml_sum += s.match_len;
    }
    let mut best = vec![0u32; 2 * BLOCK_SIZE];
    for (k, v) in segs.iter().enumerate() {
        let mut local = if k == 0 { INITIAL_REPS } else { [0; 3] };
        let mut anchor = k * seg;
        let (mut n, mut ml) = (0u32, 0u32);
        let at = k * seg_words;
        for (i, &(start, m, off)) in v.iter().enumerate() {
            let ll = (start - anchor) as u32;
            let ob = off_base_for(off, ll, &local);
            apply_off_base(&mut local, ob, ll);
            let w = at + 3 * (v.len() - 1 - i);
            best[w..w + 3].copy_from_slice(&[ll, m, ob]);
            anchor = start + m as usize;
            n += 1;
            ml += m;
        }
        best[at + meta..at + meta + 6].copy_from_slice(&[n, anchor as u32, ml, local[0], local[1], local[2]]);
    }
    let seqs: Vec<u32> = out.sequences.iter().flat_map(|s| [s.lit_len, s.match_len, s.off_base]).collect();
    (best, seqs, [out.sequences.len() as u32, BLOCK_SIZE as u32 - ml_sum])
}

/// Runs `d` on scripted inputs: each block's parse `inputs[i]` as a final pass would leave it
/// (`final_pass_words`), with the drop prices `prices[i]` when `d.prices_in`. Returns the parses
/// after the drop pass.
pub fn drops_from_parses(
    ctx: &GpuContext,
    d: &K3Drop,
    blocks: &[&[u8]],
    inputs: &[BlockOutput],
    prices: Option<&[Prices]>,
) -> anyhow::Result<Vec<BlockOutput>> {
    ensure!(blocks.len() == inputs.len() && !blocks.is_empty(), "one input per block");
    ensure!(prices.is_some() == d.prices_in, "price tables iff prices_in");
    let n = blocks.len() as u32;
    with_error_scopes(ctx, || {
        let bufs = OptBuffers::new(ctx, &d.params, n)?;
        ctx.queue.write_buffer(&bufs.data, 0, bytemuck::cast_slice(&pack_blocks(blocks)));
        let per = best_bytes_for(1, &d.params);
        let mut counts = Vec::with_capacity(2 * blocks.len());
        for (b, out) in inputs.iter().enumerate() {
            let (best, seqs, c) = final_pass_words(out, d.params.segment_log2);
            ctx.queue.write_buffer(&bufs.cands, b as u64 * per, bytemuck::cast_slice(&best));
            ctx.queue.write_buffer(&bufs.seqs, b as u64 * seqs_bytes_for(1, &d.params), bytemuck::cast_slice(&seqs));
            counts.extend_from_slice(&c);
        }
        ctx.queue.write_buffer(&bufs.counts, 0, bytemuck::cast_slice(&counts));
        if let Some(ps) = prices {
            let words: Vec<u32> =
                ps.iter().flat_map(|p| p.lit.iter().chain(&p.ll).chain(&p.ml).chain(&p.of).map(|&x| x as u32)).collect();
            ctx.queue.write_buffer(&bufs.prices, 0, bytemuck::cast_slice(&words));
        }
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("k3drop") });
        if ctx.poisoning() {
            // Everything but the uploaded words (the inputs' seqs words are left as they are).
            ctx.poison_workgroup_memory(&mut enc);
            ctx.poison_from(&mut enc, &bufs.data, data_bytes(n));
            ctx.poison_from(&mut enc, &bufs.cands, best_bytes_for(n, &d.params));
            ctx.poison_from(&mut enc, &bufs.counts, counts_bytes(n));
            for b in [&bufs.trace, &bufs.scratch, &bufs.sched] {
                ctx.poison_from(&mut enc, b, 0);
            }
            ctx.poison_from(&mut enc, &bufs.prices, if d.prices_in { prices_bytes(n) } else { 0 });
        }
        d.record(ctx, &mut enc, &bufs.binds(), n, None)?;
        ctx.queue.submit([enc.finish()]);
        read_parses(ctx, &bufs, blocks)
    })
}

/// Reads the first `n` blocks' parses out of `bufs` (literals gathered from `blocks`).
pub(crate) fn read_parses(
    ctx: &GpuContext,
    bufs: &OptBuffers,
    blocks: &[&[u8]],
) -> anyhow::Result<Vec<BlockOutput>> {
    let n = blocks.len();
    let counts: Vec<u32> = ctx.read_buffer(&bufs.counts, 0, 2 * n);
    let words: Vec<u32> = ctx.read_buffer(&bufs.seqs, 0, n * MAX_SEQS_OPT as usize * 3);
    let mut out = Vec::with_capacity(n);
    for (b, block) in blocks.iter().enumerate() {
        let (n_seq, n_lit) = (counts[2 * b], counts[2 * b + 1]);
        ensure!(
            n_seq <= MAX_SEQS_OPT && n_lit as usize <= BLOCK_SIZE,
            "block {b}: bad counts ({n_seq}, {n_lit})"
        );
        let at = b * MAX_SEQS_OPT as usize * 3;
        let parse = decode_output(block, &words[at..at + 3 * n_seq as usize], n_seq);
        ensure!(
            parse.literals.len() == n_lit as usize,
            "block {b}: K3opt counted {n_lit} literals, the sequences leave {}",
            parse.literals.len()
        );
        out.push(parse);
    }
    Ok(out)
}

/// Runs the pass on `blocks` with their candidate words (and, for `PriceSrc::Buffer`, their
/// price tables), batch by batch, and returns the parses.
pub fn parses_from_cands(
    ctx: &GpuContext,
    k: &K3Opt,
    blocks: &[&[u8]],
    cands: &[&[CandWords]],
    prices: Option<&[Prices]>,
) -> anyhow::Result<Vec<BlockOutput>> {
    ensure!(blocks.len() == cands.len(), "one candidate table per block");
    ensure!(
        prices.is_some() == (k.prices == PriceSrc::Buffer),
        "price tables iff PriceSrc::Buffer"
    );
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    with_error_scopes(ctx, || {
        let bufs = OptBuffers::new(ctx, &k.params, blocks.len().min(BATCH_CAP) as u32)?;
        let mut out = Vec::with_capacity(blocks.len());
        for (i, chunk) in blocks.chunks(BATCH_CAP).enumerate() {
            let at = i * BATCH_CAP;
            let c = &cands[at..at + chunk.len()];
            bufs.upload(ctx, chunk, c, prices.map(|p| &p[at..at + chunk.len()]))?;
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("k3opt"),
                });
            if ctx.poisoning() {
                // Everything but the uploaded blocks (+ trailing word), candidates and prices.
                let n = chunk.len() as u32;
                ctx.poison_workgroup_memory(&mut enc);
                ctx.poison_from(&mut enc, &bufs.data, data_bytes(n));
                ctx.poison_from(&mut enc, &bufs.cands, best_bytes_for(n, &k.params));
                for b in [&bufs.trace, &bufs.seqs, &bufs.counts, &bufs.scratch, &bufs.sched] {
                    ctx.poison_from(&mut enc, b, 0);
                }
                let from = if k.prices == PriceSrc::Buffer { prices_bytes(n) } else { 0 };
                ctx.poison_from(&mut enc, &bufs.prices, from);
            }
            k.record(ctx, &mut enc, &bufs.binds(), chunk.len() as u32, None)?;
            ctx.queue.submit([enc.finish()]);
            out.extend(read_parses(ctx, &bufs, chunk)?);
        }
        Ok(out)
    })
}

/// Reads the first `n` blocks' histograms that a `hist_out` pass left in `bufs.prices`.
pub(crate) fn read_hists(ctx: &GpuContext, bufs: &OptBuffers, n: usize) -> Vec<Hist> {
    let w: Vec<u32> = ctx.read_buffer(&bufs.prices, 0, n * PRICE_WORDS);
    w.chunks(PRICE_WORDS)
        .map(|c| Hist {
            lit: c[..256].try_into().unwrap(),
            ll: c[256..292].try_into().unwrap(),
            ml: c[292..345].try_into().unwrap(),
            of: c[345..].try_into().unwrap(),
        })
        .collect()
}

/// Runs every pass of `p` on `blocks` with their candidate words, batch by batch, and returns the
/// final parses (after the drop pass, if any: `opt::parse`) and, with `hists`, each cheap pass's
/// histograms (`[pass][block]`, the GPU's counterpart of
/// `opt::Hist::of_output(&opt::passes(..)[pass].out)`; one submission per pass).
pub fn parses_from_passes(
    ctx: &GpuContext,
    p: &OptPasses,
    blocks: &[&[u8]],
    cands: &[&[CandWords]],
    hists: bool,
) -> anyhow::Result<(Vec<BlockOutput>, Vec<Vec<Hist>>)> {
    ensure!(blocks.len() == cands.len(), "one candidate table per block");
    let mut hs: Vec<Vec<Hist>> = vec![Vec::new(); if hists { p.n_passes() - 1 } else { 0 }];
    if blocks.is_empty() {
        return Ok((Vec::new(), hs));
    }
    with_error_scopes(ctx, || {
        let bufs = OptBuffers::new(ctx, p.params(), blocks.len().min(BATCH_CAP) as u32)?;
        let mut out = Vec::with_capacity(blocks.len());
        for (i, chunk) in blocks.chunks(BATCH_CAP).enumerate() {
            let at = i * BATCH_CAP;
            bufs.upload(ctx, chunk, &cands[at..at + chunk.len()], None)?;
            let n = chunk.len() as u32;
            if ctx.poisoning() {
                // Everything but the uploaded blocks (+ trailing word) and candidates.
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("k3opt.poison") });
                ctx.poison_workgroup_memory(&mut enc);
                ctx.poison_from(&mut enc, &bufs.data, data_bytes(n));
                ctx.poison_from(&mut enc, &bufs.cands, best_bytes_for(n, p.params()));
                for b in [&bufs.trace, &bufs.seqs, &bufs.counts, &bufs.scratch, &bufs.sched, &bufs.prices] {
                    ctx.poison_from(&mut enc, b, 0);
                }
                ctx.queue.submit([enc.finish()]);
            }
            if hists {
                for (j, k) in p.kernels().enumerate() {
                    let mut enc = ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("k3opt.pass"),
                        });
                    k.record(ctx, &mut enc, &bufs.binds(), n, None)?;
                    ctx.queue.submit([enc.finish()]);
                    if k.hist_out {
                        hs[j].extend(read_hists(ctx, &bufs, chunk.len()));
                    }
                }
                if let Some(d) = &p.drop {
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("k3drop") });
                    d.record(ctx, &mut enc, &bufs.binds(), n, None)?;
                    ctx.queue.submit([enc.finish()]);
                }
            } else {
                let mut enc = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("k3opt.passes"),
                    });
                p.record(ctx, &mut enc, &bufs.binds(), n, None)?;
                ctx.queue.submit([enc.finish()]);
            }
            out.extend(read_parses(ctx, &bufs, chunk)?);
        }
        Ok((out, hs))
    })
}

/// GPU time (ms, median of `reps` runs after one warm-up) of each DP pass of `p`, of the final
/// fix-up, of the block order (M6 A4) and of the drop pass
/// (M6 B3; 0 without one), on the first `n` blocks of `bufs`: `n_passes() + 3` values, then the
/// span from the block order's start to the end of the drop pass, or without one of the fix-up
/// (dispatch gaps included). The final pass overwrites part of the candidate words, so `reupload`
/// runs before every run (outside the timed passes).
pub fn time_passes(
    ctx: &GpuContext,
    p: &OptPasses,
    bufs: &OptBuffers,
    n: u32,
    reps: usize,
    mut reupload: impl FnMut(),
) -> anyhow::Result<Vec<f64>> {
    ensure!(ctx.timestamps, "timestamps unavailable");
    let q = 2 * (p.n_passes() + 3);
    let qs = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("k3opt.passes.ts"),
        ty: wgpu::QueryType::Timestamp,
        count: q as u32,
    });
    let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("k3opt.passes.ts.resolve"),
        size: 8 * q as u64,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let period = ctx.queue.get_timestamp_period() as f64;
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); p.n_passes() + 4];
    for r in 0..=reps {
        reupload();
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k3opt.passes.time"),
            });
        p.record(ctx, &mut enc, &bufs.binds(), n, Some(&qs))?;
        enc.resolve_query_set(&qs, 0..q as u32, &resolve, 0);
        ctx.queue.submit([enc.finish()]);
        let t: Vec<u64> = ctx.read_buffer(&resolve, 0, q);
        if r > 0 {
            for (i, v) in times.iter_mut().take(p.n_passes() + 3).enumerate() {
                let d = if i == p.n_passes() + 2 && p.drop.is_none() { 0 } else { t[2 * i + 1].saturating_sub(t[2 * i]) };
                v.push(d as f64 * period / 1e6);
            }
            // From the block order's start (2 * n_passes() + 2) to the drop pass's end (q - 1), or
            // the fix-up's (2 * n_passes() + 1).
            let end = if p.drop.is_some() { t[q - 1] } else { t[2 * p.n_passes() + 1] };
            times[p.n_passes() + 3].push(end.saturating_sub(t[2 * p.n_passes() + 2]) as f64 * period / 1e6);
        }
    }
    Ok(times
        .into_iter()
        .map(|mut v| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gzc_core::params::{OPT16, OptParams};

    /// `OptBinds::check`'s error paths: bad `n`, a scratch buffer sized for other opt params,
    /// and a `data` buffer missing the batch's trailing 4-byte word. Each errors rather than
    /// panics (the kernel itself does no bounds checking on these buffers).
    #[test]
    fn check_rejects_bad_buffers() {
        let _gpu = crate::testing::gpu_test_slot();
        // Without poisoning: its padding would make the short buffers below long enough.
        let env = crate::context::GpuOptions::from_env();
        if env.poison {
            eprintln!("skipped: poisoning pads every buffer");
            return;
        }
        let ctx = GpuContext::new(env).expect("GPU required for gzc-gpu tests");
        let n = 4u32;
        let bufs = OptBuffers::new(&ctx, &OPT16, n).unwrap();
        let binds = bufs.binds();

        // (a) n = 0 and n > capacity.
        assert!(binds.check(0, &OPT16).is_err(), "n = 0 must be rejected");
        assert!(binds.check(n + 1, &OPT16).is_err(), "n > capacity must be rejected");
        // Sanity: the buffers themselves are fine for 1..=n blocks.
        assert!(binds.check(n, &OPT16).is_ok());

        // (b) a scratch buffer sized for a different target_length (a smaller one, so its
        // scratch is too small for OPT16's).
        let o = OPT16.opt.unwrap();
        let smaller = MatchParams { opt: Some(OptParams { target_length: 8, ..o }), ..OPT16 };
        assert!(scratch_bytes_per_block(&smaller) < scratch_bytes_per_block(&OPT16));
        let small_bufs = OptBuffers::new(&ctx, &smaller, n).unwrap();
        let e = small_bufs.binds().check(n, &OPT16).unwrap_err();
        assert!(e.to_string().contains("scratch"), "{e}");

        // (c) a `data` buffer missing its trailing 4-byte word (`ld32` reads one word past the
        // last block).
        let short_data = ctx.storage_buffer("test.short_data", data_bytes(n) - 4, false);
        let mut short = binds;
        short.data = &short_data;
        let e = short.check(n, &OPT16).unwrap_err();
        assert!(e.to_string().contains("data"), "{e}");
    }

    /// `check_dead_marks` (M6 A3) accepts `find_cands`' marks, also twice with the same scratch,
    /// and rejects a mark on a position with an earlier 3-byte match, on a position with a record,
    /// and at `PARSE_END`.
    #[test]
    fn check_dead_marks_rejects_bad_marks() {
        use gzc_core::config::PARSE_END;
        use gzc_core::reference::{DEAD_BIT, chains, find_cands, is_dead};
        let mut block = gzc_core::synth::random(5, BLOCK_SIZE);
        block.copy_within(200..300, 1000);
        let c = find_cands(&block, &chains(&block, &OPT16), &OPT16);
        assert!(c.iter().filter(|w| is_dead(**w)).count() > BLOCK_SIZE / 2);
        let mut seen = Vec::new();
        check_dead_marks(&block, &c, &mut seen).unwrap();
        check_dead_marks(&block, &c, &mut seen).unwrap();
        // 1050 repeats 250's bytes: not dead.
        assert!(!is_dead(c[1050]));
        let mut bad = c.clone();
        bad[1050] = [0, DEAD_BIT];
        assert!(check_dead_marks(&block, &bad, &mut seen).is_err(), "an earlier 3-byte match");
        let mut bad = c.clone();
        bad[1050][1] |= DEAD_BIT;
        assert!(check_dead_marks(&block, &bad, &mut seen).is_err(), "a record");
        let mut bad = c.clone();
        bad[PARSE_END] = [0, DEAD_BIT];
        assert!(check_dead_marks(&block, &bad, &mut seen).is_err(), "at PARSE_END");
        check_dead_marks(&block, &c, &mut seen).unwrap();
    }
}
