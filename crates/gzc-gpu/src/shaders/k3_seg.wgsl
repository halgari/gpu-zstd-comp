// K3 for the segmented lazy / lazy2 parse (MatchParams::segment_log2 > 0, preset lvl9seg):
// == gzc_core::lazy::lazy_parse_segmented. Two entry points, dispatched back to back in K3's pass:
//
// main_seg (one invocation per segment of SEG = 1 << SEG_LOG2 bytes, NSEG per block): the
//   sequential lazy parse of k3_lazy.wgsl (== lazy::lazy_core) over one lazy::Seg::segment:
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
// main_fixup (one workgroup per block): concatenates the segments' raw sequences into the block's
//   `seqs` in the usual layout (all lanes copy lit_len / match_len; the first sequence of a
//   segment also gets the literals the previous segments left after their last match), and
//   assigns every off_base from the true decoder rep history (== lazy::encode_raw): lane 0 walks
//   the segments in order, re-encoding each only until the true history meets the segment
//   parse's own (which started empty); the rest of its off_bases are already right.
//   counts = (n_seq, n_lit) as in k3_parse.wgsl.
//
// No subgroup operations: this is also the no-subgroup path for the preset. Consts injected by
// the host: MIN_MATCH, SEARCH_CAP, LAZY, BEST_OFF_BITS, MAX_SEQS, SEG_LOG2.

const SEG: u32 = 1u << SEG_LOG2;
const NSEG: u32 = BLOCK_SIZE >> SEG_LOG2;
const SEG_META: u32 = SEG - 6u;
const_assert 3u * (SEG / 4u) <= SEG_META;
const_assert NSEG * (SEG / 4u) <= MAX_SEQS;
const FIXUP_WG: u32 = 64u;

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> best: array<u32>;
@group(0) @binding(2) var<storage, read_write> seqs: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;

fn best_len_of(w: u32) -> u32 { return w >> BEST_OFF_BITS; }
fn best_off_of(w: u32) -> u32 { return w & ((1u << BEST_OFF_BITS) - 1u); }

// Repeat-offset history (gzc_core::seq::Reps).
var<private> r0: u32;
var<private> r1: u32;
var<private> r2: u32;

// == gzc_core::seq::off_base_for
fn off_base_for(offset: u32, lit_len: u32) -> u32 {
    if (lit_len > 0u) {
        if (offset == r0) { return 1u; }
        if (offset == r1) { return 2u; }
        if (offset == r2) { return 3u; }
    } else {
        if (offset == r1) { return 1u; }
        if (offset == r2) { return 2u; }
        if (r0 > 1u && offset == r0 - 1u) { return 3u; }
    }
    return offset + 3u;
}

// == gzc_core::seq::apply_off_base (repeat-history update only).
fn apply_off_base(off_base: u32, lit_len: u32) {
    if (off_base > 3u) {
        r2 = r1;
        r1 = r0;
        r0 = off_base - 3u;
        return;
    }
    let idx = off_base - 1u + select(0u, 1u, lit_len == 0u);
    var off: u32;
    switch (idx) {
        case 0u: { off = r0; }
        case 1u: { off = r1; }
        case 2u: { off = r2; }
        default: { off = r0 - 1u; } // wrapping, as in the reference
    }
    if (idx > 0u) {
        if (idx > 1u) { r2 = r1; }
        r1 = r0;
        r0 = off;
    }
}

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
    if (ip + len > lim) {
        len = lim - ip;
        if (len < MIN_MATCH) { return vec2<u32>(0u, 0u); }
    }
    return vec2<u32>(len, off + 3u);
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

@compute @workgroup_size(64)
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

var<workgroup> seg_n: array<u32, NSEG>;
// Index of the segment's first sequence in the block's `seqs`.
var<workgroup> seg_first: array<u32, NSEG>;
// Literals before the segment's first sequence that precede the segment start.
var<workgroup> seg_carry: array<u32, NSEG>;

// off_base_for / apply_off_base on an explicit history value (reps.x = r0 ...).
fn ob_for(reps: vec3<u32>, offset: u32, lit_len: u32) -> u32 {
    r0 = reps.x;
    r1 = reps.y;
    r2 = reps.z;
    return off_base_for(offset, lit_len);
}
fn applied(reps: vec3<u32>, off_base: u32, lit_len: u32) -> vec3<u32> {
    r0 = reps.x;
    r1 = reps.y;
    r2 = reps.z;
    apply_off_base(off_base, lit_len);
    return vec3<u32>(r0, r1, r2);
}

// The offset `off_base` stands for under history `reps` (== gzc_core::seq::apply_off_base's).
fn offset_of(reps: vec3<u32>, off_base: u32, lit_len: u32) -> u32 {
    if (off_base > 3u) { return off_base - 3u; }
    switch (off_base - 1u + select(0u, 1u, lit_len == 0u)) {
        case 0u: { return reps.x; }
        case 1u: { return reps.y; }
        case 2u: { return reps.z; }
        default: { return reps.x - 1u; }
    }
}

@compute @workgroup_size(64)
fn main_fixup(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let b = wid.x;
    if (b >= arrayLength(&counts) / 2u) { return; }
    let bbase = b * BLOCK_SIZE;
    let sbase = b * MAX_SEQS * 3u;

    if (t == 0u) {
        var first = 0u;
        var prev_end = 0u;
        var ml_sum = 0u;
        for (var k = 0u; k < NSEG; k += 1u) {
            let m = bbase + k * SEG + SEG_META;
            let n = best[m];
            seg_n[k] = n;
            seg_first[k] = first;
            seg_carry[k] = k * SEG - prev_end;
            if (n > 0u) {
                prev_end = best[m + 1u];
            }
            first += n;
            ml_sum += best[m + 2u];
        }
        counts[b * 2u] = first;
        counts[b * 2u + 1u] = BLOCK_SIZE - ml_sum;
    }
    workgroupBarrier();

    // Every lane: the segments' sequences, lanes striding, with the parse's own off_bases.
    for (var k = 0u; k < NSEG; k += 1u) {
        let n = seg_n[k];
        let src = bbase + k * SEG;
        let dst = sbase + seg_first[k] * 3u;
        let carry = seg_carry[k];
        for (var i = t; i < n; i += FIXUP_WG) {
            seqs[dst + i * 3u] = best[src + i * 3u] + select(0u, carry, i == 0u);
            seqs[dst + i * 3u + 1u] = best[src + i * 3u + 1u];
            seqs[dst + i * 3u + 2u] = best[src + i * 3u + 2u];
        }
    }
    storageBarrier();
    workgroupBarrier();

    // Lane 0, segments in order: re-encode each segment from the true incoming history until it
    // equals the parse's own history (`spec`): from there on the parse's off_bases are the true
    // ones and its end history is the true end history. == lazy::encode_raw overall. A segment
    // k > 0 starts from the empty history, so its first sequence is explicit whatever its lit_len
    // (the carry cannot change it); every later lit_len is the parse's own.
    if (t == 0u) {
        var reps = vec3<u32>(1u, 4u, 8u);
        for (var k = 0u; k < NSEG; k += 1u) {
            let n = seg_n[k];
            let src = bbase + k * SEG;
            let dst = sbase + seg_first[k] * 3u;
            let carry = seg_carry[k];
            var spec = select(vec3<u32>(0u), vec3<u32>(1u, 4u, 8u), k == 0u);
            var i = 0u;
            loop {
                if (i >= n || all(reps == spec)) { break; }
                let ll = best[src + i * 3u] + select(0u, carry, i == 0u);
                let spec_ob = best[src + i * 3u + 2u];
                let offset = offset_of(spec, spec_ob, ll);
                spec = applied(spec, spec_ob, ll);
                let ob = ob_for(reps, offset, ll);
                reps = applied(reps, ob, ll);
                seqs[dst + i * 3u + 2u] = ob;
                i += 1u;
            }
            if (i < n) {
                let m = src + SEG_META + 3u;
                reps = vec3<u32>(best[m], best[m + 1u], best[m + 2u]);
            }
        }
    }
}
