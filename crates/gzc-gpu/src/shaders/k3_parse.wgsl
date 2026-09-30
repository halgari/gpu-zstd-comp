// K3: parse, one thread per block, sequential. LAZY == 0: the greedy parse below
// (== gzc_core::reference::greedy_parse); LAZY 1/2: `lazy_parse` in k3_lazy.wgsl (appended by the
// host; == gzc_core::lazy::lazy_parse). The injected LAZY constant selects the entry.
// Dispatch (n_blocks, 1, 1) with workgroup size 1: each block gets its own subgroup, so the
// long, data-dependent per-block loops never diverge against each other (2.5-6x faster than
// 64-wide workgroups on an RTX 5090). The host binds exactly n_blocks * 2 words of `counts`,
// so arrayLength(&counts) / 2 is the batch size (the bounds check below is defensive).
// Outputs per block b:
//   seqs[(b*MAX_SEQS + i)*3 ..] = (lit_len, match_len, off_base) for i < n_seq
//   lits[b*BLOCK_SIZE/4 ..]     = literal bytes packed little-endian
//   counts[b*2 ..]              = (n_seq, n_lit)
// MAX_SEQS is prepended by the host; MIN_MATCH, SEARCH_CAP and LAZY come from the injected
// MatchParams.

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> best: array<u32>;
@group(0) @binding(2) var<storage, read_write> seqs: array<u32>;
@group(0) @binding(3) var<storage, read_write> lits: array<u32>;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;

// Repeat-offset history (gzc_core::seq::Reps).
var<private> r0: u32;
var<private> r1: u32;
var<private> r2: u32;

// Literal byte accumulator: `acc` holds `acc_n` pending bytes; full words go to lits[lit_w].
var<private> acc: u32;
var<private> acc_n: u32;
var<private> lit_w: u32;
var<private> n_lit: u32;

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

// Appends block bytes [start, end) to the literal stream.
fn push_lits(base: u32, start: u32, end: u32) {
    for (var i = start; i < end; i++) {
        acc |= load_byte(base, i) << (acc_n * 8u);
        acc_n += 1u;
        if (acc_n == 4u) {
            lits[lit_w] = acc;
            lit_w += 1u;
            acc = 0u;
            acc_n = 0u;
        }
    }
    n_lit += end - start;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= arrayLength(&counts) / 2u) { return; }
    let base = block_base(b);
    let sbase = b * MAX_SEQS * 3u;
    let bbase = b * BLOCK_SIZE * 2u;

    r0 = 1u;
    r1 = 4u;
    r2 = 8u;
    acc = 0u;
    acc_n = 0u;
    lit_w = b * (BLOCK_SIZE / 4u);
    n_lit = 0u;

    // Each parse pushes every literal including the trailing ones and returns n_seq.
    var n_seq: u32;
    if (LAZY == 0u) {
        n_seq = greedy_parse(base, sbase, bbase);
    } else {
        n_seq = lazy_parse(base, sbase, bbase);
    }
    if (acc_n > 0u) {
        lits[lit_w] = acc;
    }
    counts[b * 2u] = n_seq;
    counts[b * 2u + 1u] = n_lit;
}

// == gzc_core::reference::greedy_parse.
fn greedy_parse(base: u32, sbase: u32, bbase: u32) -> u32 {
    var n_seq = 0u;
    var p = 0u;
    var anchor = 0u;
    while (p < PARSE_END) {
        var off = 0u;
        var len = 0u;
        if (p > anchor && p >= r0) {
            let l = match_len(base, p, p - r0, 0xFFFFFFFFu);
            if (l >= MIN_MATCH) {
                off = r0;
                len = l;
            }
        }
        if (len == 0u) {
            let bl = best[bbase + p * 2u + 1u];
            if (bl >= MIN_MATCH) {
                off = best[bbase + p * 2u];
                len = bl;
                if (bl == SEARCH_CAP) {
                    // K2 stopped comparing at the cap: extend to the full length.
                    len = match_len(base, p, p - off, 0xFFFFFFFFu);
                }
            }
        }
        if (len == 0u) {
            p += 1u + ((p - anchor) >> 8u);
            continue;
        }
        // emit(p, off, len)
        let ll = p - anchor;
        let ob = off_base_for(off, ll);
        apply_off_base(ob, ll);
        push_lits(base, anchor, p);
        let s = sbase + n_seq * 3u;
        seqs[s] = ll;
        seqs[s + 1u] = len;
        seqs[s + 2u] = ob;
        n_seq += 1u;
        p += len;
        anchor = p;
    }
    push_lits(base, anchor, BLOCK_SIZE);
    return n_seq;
}
