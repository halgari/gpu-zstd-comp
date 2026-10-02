// K3: parse, one thread per block, sequential. LAZY == 0: the greedy parse below
// (== gzc_core::reference::greedy_parse); LAZY 1/2: `lazy_parse` in k3_lazy.wgsl (appended by the
// host; == gzc_core::lazy::lazy_parse). The injected LAZY constant selects the entry.
// Dispatch (n_blocks, 1, 1) with workgroup size 1: each block gets its own subgroup, so the
// long, data-dependent per-block loops never diverge against each other (2.5-6x faster than
// 64-wide workgroups on an RTX 5090). The host binds exactly n_blocks * 2 words of `counts`,
// so arrayLength(&counts) / 2 is the batch size (the bounds check below is defensive).
// Outputs per block b:
//   seqs[(b*MAX_SEQS + i)*3 ..] = (lit_len, match_len, off_base) for i < n_seq
//   counts[b*2 ..]              = (n_seq, n_lit)
// The literals themselves are not written: they are the block bytes the sequences leave
// uncovered (literal run i = block[anchor_i .. anchor_i + lit_len_i), anchor_i = the sum of
// lit_len + match_len over the sequences before i, then block[anchor_n_seq .. BLOCK_SIZE)), and
// K5 / the host gather them from there (speed phase S4).
// MAX_SEQS and BEST_OFF_BITS are prepended by the host; MIN_MATCH, SEARCH_CAP and LAZY come
// from the injected MatchParams. best[b*BLOCK_SIZE + p] = (capped len << BEST_OFF_BITS) | offset
// (K2's layout; 0 = no match).

fn best_len_of(w: u32) -> u32 { return w >> BEST_OFF_BITS; }
fn best_off_of(w: u32) -> u32 { return w & ((1u << BEST_OFF_BITS) - 1u); }

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> best: array<u32>;
@group(0) @binding(2) var<storage, read_write> seqs: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;

// Literals of the parse so far.
var<private> n_lit: u32;

// Counts block bytes [start, end) as literals (none when start >= end: a final push from an
// anchor past BLOCK_SIZE, only reachable with a best[] word that claims a match past the block end).
fn push_lits(start: u32, end: u32) {
    if (end > start) { n_lit += end - start; }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= arrayLength(&counts) / 2u) { return; }
    let base = block_base(b);
    let sbase = b * MAX_SEQS * 3u;
    let bbase = b * BLOCK_SIZE;

    r0 = 1u;
    r1 = 4u;
    r2 = 8u;
    n_lit = 0u;

    // Each parse counts every literal including the trailing ones and returns n_seq.
    var n_seq: u32;
    if (LAZY == 0u) {
        n_seq = greedy_parse(base, sbase, bbase);
    } else {
        n_seq = lazy_parse(base, sbase, bbase);
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
            let w = best[bbase + p];
            let bl = best_len_of(w);
            if (bl >= MIN_MATCH) {
                off = best_off_of(w);
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
        push_lits(anchor, p);
        let s = sbase + n_seq * 3u;
        seqs[s] = ll;
        seqs[s + 1u] = len;
        seqs[s + 2u] = ob;
        n_seq += 1u;
        p += len;
        anchor = p;
    }
    push_lits(anchor, BLOCK_SIZE);
    return n_seq;
}
