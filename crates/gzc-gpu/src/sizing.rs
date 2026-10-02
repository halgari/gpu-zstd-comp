//! Byte sizes of the per-batch GPU buffers and the batch limits derived from them.
//!
//! Every buffer-size helper lives here. `BufferSizes` lists each buffer of one `BatchBuffers`
//! once; `BatchBuffers`, `scratch_bytes`, `max_batch_blocks` and (through `scratch_bytes`)
//! `pipeline::vram_bytes` all derive from it, so a new buffer is added in one place.
use gzc_core::config::{BLOCK_SIZE, HASH_BITS};
use gzc_core::params::MatchParams;

use crate::chains::{HEAD_TABLES, pred_words_per_block};
use crate::compressor::{FRAME_STRIDE, max_seqs};
use crate::k3opt::{PRICE_WORDS, SCHED_HDR, scratch_bytes_per_block};

/// Bytes of the packed `data` buffer for `n_blocks` (blocks plus one trailing zero word).
pub fn data_bytes(n_blocks: u32) -> u64 {
    n_blocks as u64 * BLOCK_SIZE as u64 + 4
}

/// Bytes of one head table (2^HASH_BITS u32 entries).
pub(crate) const TABLE_BYTES: u64 = (1u64 << HASH_BITS) * 4;

/// Bytes of the `head` buffer K1 needs for `n_blocks` with `n_hashes` chains per block
/// (`MatchParams::n_hashes`): one table per chain, at most `HEAD_TABLES`.
pub fn head_bytes(n_blocks: u32, n_hashes: u32) -> u64 {
    (n_blocks as u64 * n_hashes as u64).min(HEAD_TABLES as u64) * TABLE_BYTES
}

/// Bytes of the `pred` buffer K1 writes for `n_blocks` under `p` (`chains::pred_words_per_block`):
/// `n_blocks * p.n_hashes() * BLOCK_SIZE` words without sparse long chains.
pub fn chain_pred_bytes(n_blocks: u32, p: &MatchParams) -> u64 {
    n_blocks as u64 * pred_words_per_block(p) * 4
}

/// Bytes of a one-word-per-position `best` buffer: `[block][pos]` × one u32,
/// `(capped len << BEST_OFF_BITS) | offset` (0 = no match).
pub(crate) const fn best_bytes(n_blocks: u32) -> u64 {
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

/// Bytes of the `seqs` buffer under match params `m`: `[block][max_seqs(m)]` × 3 u32.
pub fn seqs_bytes_for(n_blocks: u32, m: &MatchParams) -> u64 {
    n_blocks as u64 * max_seqs(m) as u64 * 12
}

/// Bytes of the `pred` buffer under match params `m`: K1's chains (`chain_pred_bytes`),
/// and for the optimal parse at least K3opt's DP trace, which reuses the buffer once K2opt has
/// read the chains (`trace_bytes`). Opt3 has two full chains, so the two are equal; M6's sparse
/// long chains add `BLOCK_SIZE / stride` words per block each (S3: 11 B per position).
pub fn pred_bytes_for(n_blocks: u32, m: &MatchParams) -> u64 {
    let chains = chain_pred_bytes(n_blocks, m);
    if m.opt.is_some() { chains.max(trace_bytes(n_blocks)) } else { chains }
}

/// Bytes of K3opt's DP trace for `n_blocks`: `[block][pos][2]` u32 (`tbase = 2 * b * BLOCK_SIZE`
/// in `k3_opt.wgsl`), 8 B per position.
pub fn trace_bytes(n_blocks: u32) -> u64 {
    n_blocks as u64 * BLOCK_SIZE as u64 * 8
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

/// Bytes of K3opt's `prices` buffer for `n` blocks (`k3opt::PRICE_WORDS` words per block).
pub fn prices_bytes(n: u32) -> u64 {
    n as u64 * PRICE_WORDS as u64 * 4
}

/// Bytes of K3opt's `sched` buffer for `n` blocks (M6 A4): the header, then each block's weight,
/// the heavy-first block order and each block's rank in it (4 B per block each).
pub fn sched_bytes(n: u32) -> u64 {
    (SCHED_HDR as u64 + 3 * n as u64) * 4
}

/// The byte size of every buffer of one `BatchBuffers` for `n_blocks` under match params `m`.
/// The K3opt buffers are 0 without `m.opt`; `frames` and `frame_len` are allocated only on the
/// frame path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferSizes {
    pub data: u64,
    pub head: u64,
    pub pred: u64,
    pub best: u64,
    pub seqs: u64,
    pub counts: u64,
    pub frames: u64,
    pub frame_len: u64,
    /// The blocks' real lengths (K3t's input), one u32 per block.
    pub lens: u64,
    pub opt_prices: u64,
    pub opt_scratch: u64,
    pub opt_sched: u64,
}

impl BufferSizes {
    pub fn new(n_blocks: u32, m: &MatchParams) -> Self {
        let opt = m.opt.is_some();
        Self {
            data: data_bytes(n_blocks),
            head: head_bytes(n_blocks, m.n_hashes()),
            pred: pred_bytes_for(n_blocks, m),
            best: best_bytes_for(n_blocks, m),
            seqs: seqs_bytes_for(n_blocks, m),
            counts: counts_bytes(n_blocks),
            frames: frames_bytes(n_blocks),
            frame_len: frame_len_bytes(n_blocks),
            lens: frame_len_bytes(n_blocks),
            opt_prices: if opt { prices_bytes(n_blocks) } else { 0 },
            opt_scratch: if opt { n_blocks as u64 * scratch_bytes_per_block(m) } else { 0 },
            opt_sched: if opt { sched_bytes(n_blocks) } else { 0 },
        }
    }

    /// The scratch buffers: head, pred, best, seqs, counts and K3opt's own.
    pub fn scratch(&self) -> u64 {
        self.head + self.pred + self.best + self.seqs + self.counts + self.opt_prices + self.opt_scratch + self.opt_sched
    }

    /// The buffers `Pipeline` allocates once and shares between its slots: data, plus frames,
    /// frame_len and lens with `frames`.
    pub fn shared(&self, frames: bool) -> u64 {
        self.data + if frames { self.frames + self.frame_len + self.lens } else { 0 }
    }

    /// Every buffer that K1's bound (`max_blocks_per_batch_for`, which covers data and head)
    /// does not already cover, each of which must fit one storage binding.
    fn bound_by_max_batch(&self) -> [u64; 8] {
        [self.best, self.pred, self.seqs, self.counts, self.frames, self.opt_prices, self.opt_scratch, self.opt_sched]
    }
}

/// Bytes of the scratch buffers for `n_blocks` under match params `m` (`BufferSizes::scratch`).
pub fn scratch_bytes(n_blocks: u32, m: &MatchParams) -> u64 {
    BufferSizes::new(n_blocks, m).scratch()
}

/// Bytes of the remaining `BatchBuffers` (data, plus frames, frame_len and lens with `frames`)
/// for `n_blocks`; the pipeline allocates them once, shared by all its slots
/// (`BufferSizes::shared`).
pub fn slot_bytes(n_blocks: u32, frames: bool) -> u64 {
    data_bytes(n_blocks) + if frames { frames_bytes(n_blocks) + 2 * frame_len_bytes(n_blocks) } else { 0 }
}

/// Largest `n_blocks` one `ChainsKernel::record` call may take under `limits` for the chains of
/// `p`: the data (n*BLOCK_SIZE + 4 bytes), head (`p.n_hashes()` tables per block) and pred
/// (`pred_words_per_block` words) buffers each fit one storage binding and one buffer, and the
/// kernels' u32 pred indices cannot wrap (head indices `table << HASH_BITS` stay below 2^24). The
/// block count is also kept within `max_compute_workgroups_per_dimension`, which K2 dispatches
/// over. 0 if one block doesn't fit.
pub fn max_blocks_per_batch_for(limits: &wgpu::Limits, p: &MatchParams) -> u32 {
    let limit = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
    let nh = p.n_hashes() as u64;
    let pred_words = pred_words_per_block(p);
    let by_data = limit.saturating_sub(4) / BLOCK_SIZE as u64;
    // head_bytes(n) <= limit: always once HEAD_TABLES tables fit, else n * nh tables must.
    let by_head = if HEAD_TABLES as u64 * TABLE_BYTES <= limit { u64::MAX } else { limit / TABLE_BYTES / nh };
    let by_buffers = by_data.min(by_head).min(limit / (pred_words * 4));
    // The kernels' u32 pred word indices (b * PRED_PER_BLOCK + ..) cannot wrap; without sparse
    // chains this is (b * n_hashes + chain) * BLOCK_SIZE.
    let by_index = (1u64 << 32) / pred_words;
    by_buffers.min(by_index).min(limits.max_compute_workgroups_per_dimension as u64) as u32
}

/// Largest `n_blocks` one `Kernels::record` call (and one `BatchBuffers`) for match params `m`
/// may take under `limits`: K1's bound (`max_blocks_per_batch_for(limits, m)`: data, head
/// and pred buffers, the pred words counting `m`'s sparse chains, the workgroups-per-dimension
/// limit used by K1's x and K2's y dispatch, u32 head/pred indices) further limited so the best,
/// pred (`pred_bytes_for`, which K3opt reuses as its trace), seqs, counts and frames buffers, and
/// for opt params K3opt's prices, scratch and sched buffers, each fit one storage binding and one
/// buffer, and their u32 word indices (at most BLOCK_SIZE words per block, in `best`, times
/// `best_words(m)`; `seqs` has 3*MAX_SEQS < BLOCK_SIZE) cannot wrap. 0 if one block doesn't fit.
pub fn max_batch_blocks(limits: &wgpu::Limits, m: &MatchParams) -> u32 {
    let limit = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
    let by_k1 = max_blocks_per_batch_for(limits, m) as u64;
    let by_buffers = BufferSizes::new(1, m)
        .bound_by_max_batch()
        .into_iter()
        .filter(|&per_block| per_block > 0)
        .map(|per_block| limit / per_block)
        .min()
        .unwrap();
    let by_index = (1u64 << 32) / (BLOCK_SIZE as u64 * best_words(m) as u64);
    by_k1.min(by_buffers).min(by_index) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use gzc_core::config::LOG2_BLOCK;
    use gzc_core::params::{LVL3, RUNG1};

    const MIB: u64 = 1 << 20;

    fn limits(binding: u64, buffer: u64, wg: u32) -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffer_binding_size: binding,
            max_buffer_size: buffer,
            max_compute_workgroups_per_dimension: wg,
            ..wgpu::Limits::default()
        }
    }

    /// K1's buffers for `n` blocks of `p` fit `limit`.
    fn k1_fits(n: u32, p: &MatchParams, limit: u64) -> bool {
        head_bytes(n, p.n_hashes()) <= limit && chain_pred_bytes(n, p) <= limit && data_bytes(n) <= limit
    }

    #[test]
    fn k1_batch_fits_every_buffer_at_default_limits() {
        // LVL3: two full chains; RUNG1: one.
        let n2 = max_blocks_per_batch_for(&limits(128 * MIB, 256 * MIB, 65535), &LVL3);
        let n1 = max_blocks_per_batch_for(&limits(128 * MIB, 256 * MIB, 65535), &RUNG1);
        // pred-bound: 512 KiB per block (256 KiB with one chain); head is at most 64 MiB.
        assert_eq!((n2, n1), (256, 512));
        assert!(k1_fits(n2, &LVL3, 128 * MIB) && !k1_fits(n2 + 1, &LVL3, 128 * MIB));
        assert!(k1_fits(n1, &RUNG1, 128 * MIB) && !k1_fits(n1 + 1, &RUNG1, 128 * MIB));
    }

    #[test]
    fn k1_batch_respects_buffer_size_and_workgroup_cap() {
        for p in [RUNG1, LVL3] {
            assert_eq!(max_blocks_per_batch_for(&limits(128 * MIB, 128 * MIB, 3), &p), 3);
            let n = max_blocks_per_batch_for(&limits(u64::MAX, 4 * MIB, 65535), &p);
            assert!(n > 0 && k1_fits(n, &p, 4 * MIB) && !k1_fits(n + 1, &p, 4 * MIB));
        }
    }

    #[test]
    fn k1_batch_keeps_u32_indices_in_range() {
        // With unlimited buffers the cap keeps (b*n_hashes+chain) * BLOCK_SIZE in u32.
        for p in [RUNG1, LVL3] {
            let n = max_blocks_per_batch_for(&limits(u64::MAX, u64::MAX, u32::MAX), &p);
            assert_eq!(n as u64 * p.n_hashes() as u64, 1u64 << (32 - LOG2_BLOCK));
        }
        assert!((HEAD_TABLES as u64) << HASH_BITS <= 1 << 32);
    }

    #[test]
    fn head_holds_one_table_per_chain_up_to_head_tables() {
        assert_eq!(head_bytes(3, 2), 6 * TABLE_BYTES);
        assert_eq!(head_bytes(HEAD_TABLES, 1), HEAD_TABLES as u64 * TABLE_BYTES);
        assert_eq!(head_bytes(1638, 1), head_bytes(HEAD_TABLES, 1));
        assert_eq!(head_bytes(1365, 2), head_bytes(HEAD_TABLES, 1));
        // A limit below the full head bounds the chains to the tables that fit.
        let n = max_blocks_per_batch_for(&limits(u64::MAX, 40 * TABLE_BYTES, 65535), &LVL3);
        assert!(k1_fits(n, &LVL3, 40 * TABLE_BYTES) && !k1_fits(n + 1, &LVL3, 40 * TABLE_BYTES));
    }

    #[test]
    fn k1_batch_is_zero_when_one_block_does_not_fit() {
        assert_eq!(max_blocks_per_batch_for(&limits(256 * 1024, 256 * 1024, 65535), &LVL3), 0);
        assert_eq!(max_blocks_per_batch_for(&limits(BLOCK_SIZE as u64, BLOCK_SIZE as u64, 65535), &RUNG1), 0);
    }

    /// Without sparse chains K1's pred is one BLOCK_SIZE-long u32 array per chain.
    #[test]
    fn chain_pred_bytes_without_sparse_chains() {
        assert_eq!(chain_pred_bytes(10, &LVL3), 10 * 2 * BLOCK_SIZE as u64 * 4);
        assert_eq!(chain_pred_bytes(10, &RUNG1), 10 * BLOCK_SIZE as u64 * 4);
    }

    /// `scratch` and `shared` add up every buffer exactly once.
    #[test]
    fn buffer_sizes_partition_every_buffer() {
        for (_, m) in gzc_core::params::PRESETS {
            let s = BufferSizes::new(7, &m);
            let all = s.data + s.head + s.pred + s.best + s.seqs + s.counts + s.frames + s.frame_len + s.lens
                + s.opt_prices + s.opt_scratch + s.opt_sched;
            assert_eq!(s.scratch() + s.shared(true), all);
            assert_eq!(s.shared(true), slot_bytes(7, true));
            assert_eq!(s.shared(false), slot_bytes(7, false));
            assert_eq!(s.scratch(), scratch_bytes(7, &m));
        }
    }
}
