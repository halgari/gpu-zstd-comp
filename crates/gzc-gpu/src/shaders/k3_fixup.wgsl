// The per-block fix-up of a segmented parse (K3 of lvl9seg, k3_seg.wgsl, and of the optimal
// parse, k3_opt.wgsl), plus the repeat-offset helpers both use. Appended to the kernel's body.
// The kernel defines SEG, NSEG, SEG_WORDS (the `best` words each segment keeps its raw
// sequences and 6-word trailer in: block b's segment k at best[(b * NSEG + k) * SEG_WORDS ..],
// trailer at SEG_META = SEG_WORDS - 6), MAX_SEQS and the bindings best / seqs / counts.
//
// main_fixup (one workgroup per block): concatenates the segments' raw sequences into the block's
//   `seqs` in the usual layout (all lanes copy lit_len / match_len; the first sequence of a
//   segment also gets the literals the previous segments left after their last match), and
//   assigns every off_base from the true decoder rep history (== lazy::encode_raw): lane 0 walks
//   the segments in order, re-encoding each only until the true history meets the segment
//   parse's own (which started empty); the rest of its off_bases are already right. This needs
//   every raw off_base to be off_base_for(offset, lit_len) under the segment's own history.
//   counts = (n_seq, n_lit) as in k3_parse.wgsl.

const FIXUP_WG: u32 = 64u;

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
    let bbase = b * NSEG * SEG_WORDS;
    let sbase = b * MAX_SEQS * 3u;

    if (t == 0u) {
        var first = 0u;
        var prev_end = 0u;
        var ml_sum = 0u;
        for (var k = 0u; k < NSEG; k += 1u) {
            let m = bbase + k * SEG_WORDS + SEG_META;
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
        let src = bbase + k * SEG_WORDS;
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
            let src = bbase + k * SEG_WORDS;
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
