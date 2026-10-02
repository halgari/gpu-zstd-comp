// K3 for the segmented lazy / lazy2 parse (MatchParams::segment_log2 > 0, preset lvl9seg):
// == gzc_core::lazy::lazy_parse_segmented. Two entry points, dispatched back to back in K3's pass:
//
// main_seg (one invocation per segment of SEG = 1 << SEG_LOG2 bytes, NSEG per block): the
//   sequential lazy parse (== lazy::lazy_core) over one lazy::Seg::segment:
//   segment 0 starts at ip 1 with INITIAL_REPS, segment k > 0 at ip = anchor = k*SEG with all-zero
//   reps; matches are clamped to the segment end `lim` (best[] matches past it are cut, and
//   dropped below MIN_MATCH); the loop runs while ip < pend (lim - 4, PARSE_END for the last
//   segment); no skip acceleration (step 1). It writes its sequences (lit_len, match_len,
//   off_base under its own rep history) and a 6-word trailer (n_seq, final anchor, sum of
//   match_len, final r0 r1 r2) into its OWN segment of the block's best[] words, which only this
//   invocation reads, and which K3 is the last to read:
//     best[b*BLOCK_SIZE + k*SEG + 3*i ..]        = (ll, ml, off_base) of sequence i
//     best[b*BLOCK_SIZE + k*SEG + SEG - 6 ..]    = the trailer
//   Sequence i is written once the parse's anchor has passed k*SEG + 4*(i + 1) (each sequence
//   covers at least 4 bytes: ml >= 4), and every later best[] read is at ip >= anchor, so the
//   parse never reads a word it overwrote: 3*i + 2 < 4*(i + 1). At most SEG/4 sequences fit a
//   segment, so the raw words end before the trailer (const_assert below).
// main_fixup: k3_fixup.wgsl (appended), one workgroup per block; see there.
//
// No subgroup operations: this is also the no-subgroup path for the preset. Consts injected by
// the host: MIN_MATCH, SEARCH_CAP, LAZY, BEST_OFF_BITS, MAX_SEQS, SEG_LOG2.
//
// Built without naga's loop bounding (GpuContext::shader_unbounded_loops; bounds checks stay on).
// Terminates, whatever best[] holds:
// - main_seg's parse loop raises ip on every pass and ends at ilimit: a scan without a hit adds
//   SCAN_W, a position without a match adds 1, and a stored match moves ip to its end, at least
//   4 bytes past the position the pass started at.
// - The deferral loop raises ip by 1 or 2 before each `continue` and breaks at ilimit.
// - The catch-up lowers `start` to `anchor`. The immediate-repcode loop raises ip by ml >= 4,
//   to ilimit.
// - match_len runs with p < lim <= BLOCK_SIZE and cap lim - p (common.wgsl). scan's loops are
//   counted.
// - main_fixup: see k3_fixup.wgsl.
// A loop added here needs its own argument.

const SEG: u32 = 1u << SEG_LOG2;
const NSEG: u32 = BLOCK_SIZE >> SEG_LOG2;
// This kernel keeps each segment's raw sequences in its own SEG words of `best` (k3_fixup.wgsl).
const SEG_WORDS: u32 = SEG;
const SEG_META: u32 = SEG_WORDS - 6u;
const RAW_REVERSED: bool = false;
const_assert 3u * (SEG / 4u) <= SEG_META;
const_assert NSEG * (SEG / 4u) <= MAX_SEQS;

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> best: array<u32>;
@group(0) @binding(2) var<storage, read_write> seqs: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;

fn best_len_of(w: u32) -> u32 { return w >> BEST_OFF_BITS; }
fn best_off_of(w: u32) -> u32 { return w & ((1u << BEST_OFF_BITS) - 1u); }

// zstd ZSTD_highbit32 (x > 0).
fn highbit(x: u32) -> i32 {
    return i32(firstLeadingBit(x));
}

// == lazy.rs rep_len with iend = lim.
fn rep_len(base: u32, p: u32, off: u32, lim: u32) -> u32 {
    if (off == 0u || off > p) { return 0u; }
    let l = match_len(base, p, p - off, lim - p);
    return select(0u, l, l >= 4u);
}

// == lazy.rs search_max with Seg::clamp: (matchLength, offBase) of best[ip], matchLength 0 when
// there is no match of at least MIN_MATCH inside [ip, lim).
fn search_max(base: u32, bbase: u32, ip: u32, lim: u32) -> vec2<u32> {
    let w = best[bbase + ip];
    let bl = best_len_of(w);
    if (bl < MIN_MATCH) { return vec2<u32>(0u, 0u); }
    let off = best_off_of(w);
    var len = bl;
    if (bl == SEARCH_CAP) {
        len = match_len(base, ip, ip - off, lim - ip);
    }
    // Cut at lim; a capped entry's lim-bounded extension can also end below MIN_MATCH.
    len = min(len, lim - ip);
    if (len < MIN_MATCH) { return vec2<u32>(0u, 0u); }
    return vec2<u32>(len, off + 3u);
}

// Bytes [p, p + 4) of the 8 bytes (lo, hi) starting at p - j, for j in 0..4.
fn window4(lo: u32, hi: u32, j: u32) -> u32 {
    if (j == 0u) { return lo; }
    return (lo >> (8u * j)) | (hi << (32u - 8u * j));
}

// Literal scan, SCAN_W positions per step: the first j < SCAN_W such that position ip + j could
// start a match at the loop top (best[ip + j] has at least MIN_MATCH bytes, or the 4 bytes at
// ip + j + 1 repeat at offset_1 == rep_len(ip + j + 1, offset_1) > 0), considering only
// positions below ilimit; SCAN_W when there is none. Every position it skips is one where the
// loop top would find no match and step by 1. For ip + j < ilimit, ip + j + 1 + 4 <= lim, so a
// 4-byte compare equals rep_len's >= 4 test. best[] is read only below ilimit (this segment's
// words). The data loads for positions at or past ilimit (masked out) can read up to 4 bytes past
// the block (ip <= PARSE_END - 1 = BLOCK_SIZE - 9), i.e. the next block or pack_blocks' trailing
// zero word: still inside the buffer.
const_assert SCAN_W == 4u || SCAN_W == 8u;
fn scan(base: u32, bbase: u32, ip: u32, ilimit: u32, lim: u32, off1: u32) -> u32 {
    var hit = 0u;
    for (var j = 0u; j < SCAN_W; j += 1u) {
        if (ip + j < ilimit && best_len_of(best[bbase + ip + j]) >= MIN_MATCH) { hit |= 1u << j; }
    }
    let p0 = ip + 1u;
    if (off1 != 0u) {
        if (off1 <= p0) {
            let q0 = p0 - off1;
            for (var h = 0u; h < SCAN_W; h += 4u) {
                let x0 = load_u32_at(base, p0 + h);
                let x1 = load_u32_at(base, p0 + h + 4u);
                let y0 = load_u32_at(base, q0 + h);
                let y1 = load_u32_at(base, q0 + h + 4u);
                for (var j = 0u; j < 4u; j += 1u) {
                    if (window4(x0, x1, j) == window4(y0, y1, j)) { hit |= 1u << (h + j); }
                }
            }
        } else {
            for (var j = 0u; j < SCAN_W; j += 1u) {
                if (ip + j < ilimit && rep_len(base, p0 + j, off1, lim) > 0u) { hit |= 1u << j; }
            }
        }
    }
    let n_valid = min(SCAN_W, ilimit - ip);
    let valid = select((1u << n_valid) - 1u, 0xFFFFFFFFu, n_valid >= 32u);
    return min(countTrailingZeros(hit & valid), SCAN_W);
}

// == lazy.rs store: sequence n_seq (ll, ml, off_base under the parse's own rep history) into the
// segment's best[] words at wbase, updating that history. Returns the new count.
fn store_raw(wbase: u32, n_seq: u32, ll: u32, offset: u32, ml: u32) -> u32 {
    let ob = off_base_for(offset, ll);
    apply_off_base(ob, ll);
    let s = wbase + n_seq * 3u;
    best[s] = ll;
    best[s + 1u] = ml;
    best[s + 2u] = ob;
    return n_seq + 1u;
}

@compute @workgroup_size(SEG_WG)
fn main_seg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x / NSEG;
    let k = gid.x % NSEG;
    if (b >= arrayLength(&counts) / 2u) { return; }
    let base = block_base(b);
    let bbase = b * BLOCK_SIZE;
    let s = k * SEG;
    let wbase = bbase + s;
    let last = k + 1u == NSEG;
    let lim = select(s + SEG, BLOCK_SIZE, last);
    let ilimit = select(lim - 4u, PARSE_END, last);

    var anchor = s;
    var ip = s;
    if (k == 0u) {
        // zstd: ip += (dictAndPrefixLength == 0); INITIAL_REPS [1, 4, 8].
        ip = 1u;
        r0 = 1u;
        r1 = 4u;
        r2 = 8u;
    } else {
        r0 = 0u;
        r1 = 0u;
        r2 = 0u;
    }
    // maxRep = ip: offsets above it start disabled (segment 0: offset_1 = 1, offset_2 = 0).
    var offset_1 = select(0u, r0, r0 <= ip);
    var offset_2 = select(0u, r1, r1 <= ip);
    var n_seq = 0u;
    var ml_sum = 0u;

    while (ip < ilimit) {
        // Skip positions that cannot start a match, SCAN_W at a time.
        let j = scan(base, bbase, ip, ilimit, lim, offset_1);
        ip += j;
        if (j == SCAN_W) { continue; }

        var match_length = 0u;
        var off_base = 1u; // REPCODE1_TO_OFFBASE
        var start = ip + 1u;

        // check repCode at ip+1
        let l = rep_len(base, ip + 1u, offset_1, lim);
        if (l > 0u) {
            match_length = l;
        }

        // first search (depth 0)
        let m0 = search_max(base, bbase, ip, lim);
        if (m0.x > 0u && m0.x > match_length) {
            match_length = m0.x;
            start = ip;
            off_base = m0.y;
        }

        if (match_length < 4u) {
            ip += 1u; // no skip acceleration in a segmented parse
            continue;
        }

        // let's try to find a better solution
        loop {
            if (ip + 1u >= ilimit) { break; }
            ip += 1u;
            // search depth 1: repcode at ip, x3 rule
            let ml_rep = rep_len(base, ip, offset_1, lim);
            if (ml_rep >= 4u) {
                let gain2 = i32(ml_rep * 3u);
                let gain1 = i32(match_length * 3u) - highbit(off_base) + 1;
                if (gain2 > gain1) {
                    match_length = ml_rep;
                    off_base = 1u;
                    start = ip;
                }
            }
            let m1 = search_max(base, bbase, ip, lim);
            if (m1.x > 0u) {
                let gain2 = i32(m1.x * 4u) - highbit(m1.y);
                let gain1 = i32(match_length * 4u) - highbit(off_base) + 4;
                if (m1.x >= 4u && gain2 > gain1) {
                    match_length = m1.x;
                    off_base = m1.y;
                    start = ip;
                    continue; // search a better one
                }
            }

            // let's find an even better one
            if (LAZY == 2u && ip + 1u < ilimit) {
                ip += 1u;
                // search depth 2: repcode at ip, x4 rule
                let ml_rep2 = rep_len(base, ip, offset_1, lim);
                if (ml_rep2 >= 4u) {
                    let gain2 = i32(ml_rep2 * 4u);
                    let gain1 = i32(match_length * 4u) - highbit(off_base) + 1;
                    if (gain2 > gain1) {
                        match_length = ml_rep2;
                        off_base = 1u;
                        start = ip;
                    }
                }
                let m2 = search_max(base, bbase, ip, lim);
                if (m2.x > 0u) {
                    let gain2 = i32(m2.x * 4u) - highbit(m2.y);
                    let gain1 = i32(match_length * 4u) - highbit(off_base) + 7;
                    if (m2.x >= 4u && gain2 > gain1) {
                        match_length = m2.x;
                        off_base = m2.y;
                        start = ip;
                        continue;
                    }
                }
            }
            break; // nothing found: store previous solution
        }

        if (off_base > 3u) {
            // catch up: start > anchor && start - offset > prefixLowest (block position 0)
            let off = off_base - 3u;
            while (start > anchor && start > off && load_byte(base, start - 1u) == load_byte(base, start - 1u - off)) {
                start -= 1u;
                match_length += 1u;
            }
            n_seq = store_raw(wbase, n_seq, start - anchor, off, match_length);
            // deviation (dedup store): follow the decoder's reps, not zstd's offset_2 = offset_1.
            offset_1 = r0;
            offset_2 = r1;
        } else {
            // repcode 1 at ip+1 or later (ll > 0): history unchanged.
            n_seq = store_raw(wbase, n_seq, start - anchor, offset_1, match_length);
        }
        ml_sum += match_length;
        anchor = start + match_length;
        ip = anchor;

        // check immediate repcode: offset_2 at ip, stored as (ll 0, repcode 1) = zstd's swap.
        while (ip <= ilimit && offset_2 > 0u) {
            let ml = rep_len(base, ip, offset_2, lim);
            if (ml == 0u) { break; }
            let t = offset_2;
            offset_2 = offset_1;
            offset_1 = t;
            n_seq = store_raw(wbase, n_seq, 0u, offset_1, ml);
            ml_sum += ml;
            ip += ml;
            anchor = ip;
        }
    }
    best[wbase + SEG_META] = n_seq;
    best[wbase + SEG_META + 1u] = anchor;
    best[wbase + SEG_META + 2u] = ml_sum;
    best[wbase + SEG_META + 3u] = r0;
    best[wbase + SEG_META + 4u] = r1;
    best[wbase + SEG_META + 5u] = r2;
}

