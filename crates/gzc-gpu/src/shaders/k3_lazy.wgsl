// K3 lazy / lazy2 parse (LAZY 1 or 2): a statement-by-statement mirror of
// gzc_core::lazy::lazy_parse, itself a port of libzstd 1.5.7 ZSTD_compressBlock_lazy_generic
// (noDict, depth = LAZY). Appended by the host after k3_parse.wgsl, whose bindings, rep history
// (r0/r1/r2), off_base_for / apply_off_base and literal count (push_lits) it shares; the
// seqs / counts layout is the greedy parse's. Names follow zstd (ip, anchor, start,
// offBase, matchLength, offset_1, offset_2, ilimit = PARSE_END). Every `// deviation:` of the CPU
// oracle is mirrored here (see lazy.rs for the full notes):
//   - deferral only while ip + 1 < PARSE_END (best[] is valid below PARSE_END only);
//   - no lazySkipping (K2 searched every position);
//   - after an explicit store offset_1 / offset_2 are reloaded from the decoder's reps (r0, r1),
//     so a dedup store (offset == r0, ll > 0, stored as repcode 1) keeps the old r1.
// Gains are i32; highbit via firstLeadingBit. Nothing reads past BLOCK_SIZE (match_len bounds).

// zstd ZSTD_highbit32 (x > 0).
fn highbit(x: u32) -> i32 {
    return i32(firstLeadingBit(x));
}

// == lazy.rs rep_len: repeat-offset match length at p, or 0 when shorter than 4 bytes or `off`
// is unusable (0, or beyond p: the guard keeps p - off from wrapping).
fn rep_len(base: u32, p: u32, off: u32) -> u32 {
    if (off == 0u || off > p) { return 0u; }
    let l = match_len(base, p, p - off, 0xFFFFFFFFu);
    return select(0u, l, l >= 4u);
}

// == lazy.rs search_max: (matchLength, offBase) of best[ip], matchLength 0 when best[ip] has no
// match of at least MIN_MATCH. A capped length is extended to the true match length.
fn search_max(base: u32, bbase: u32, ip: u32) -> vec2<u32> {
    let w = best[bbase + ip];
    let bl = best_len_of(w);
    if (bl < MIN_MATCH) { return vec2<u32>(0u, 0u); }
    let off = best_off_of(w);
    var len = bl;
    if (bl == SEARCH_CAP) {
        len = match_len(base, ip, ip - off, 0xFFFFFFFFu);
    }
    return vec2<u32>(len, off + 3u);
}

// == lazy.rs store: one sequence of `ll` literals from `anchor` then `ml` bytes at `offset`, with
// the decoder's rep bookkeeping. Returns the new sequence count.
fn store_seq(base: u32, sbase: u32, n_seq: u32, anchor: u32, ll: u32, offset: u32, ml: u32) -> u32 {
    let ob = off_base_for(offset, ll);
    apply_off_base(ob, ll);
    push_lits(anchor, anchor + ll);
    let s = sbase + n_seq * 3u;
    seqs[s] = ll;
    seqs[s + 1u] = ml;
    seqs[s + 2u] = ob;
    return n_seq + 1u;
}

fn lazy_parse(base: u32, sbase: u32, bbase: u32) -> u32 {
    var n_seq = 0u;
    var anchor = 0u;
    // zstd: ip += (dictAndPrefixLength == 0).
    var ip = 1u;
    // maxRep = ip = 1: INITIAL_REPS [1, 4, 8] give offset_1 = 1, offset_2 = 0 (disabled).
    var offset_1 = select(0u, r0, r0 <= 1u);
    var offset_2 = select(0u, r1, r1 <= 1u);

    while (ip < PARSE_END) {
        var match_length = 0u;
        var off_base = 1u; // REPCODE1_TO_OFFBASE
        var start = ip + 1u;

        // check repCode at ip+1
        let l = rep_len(base, ip + 1u, offset_1);
        if (l > 0u) {
            match_length = l;
        }

        // first search (depth 0)
        let m0 = search_max(base, bbase, ip);
        if (m0.x > 0u && m0.x > match_length) {
            match_length = m0.x;
            start = ip;
            off_base = m0.y;
        }

        if (match_length < 4u) {
            // jump faster over incompressible sections (kSearchStrength 8)
            ip += ((ip - anchor) >> 8u) + 1u;
            continue;
        }

        // let's try to find a better solution
        loop {
            if (ip + 1u >= PARSE_END) { break; }
            ip += 1u;
            // search depth 1: repcode at ip, x3 rule
            let ml_rep = rep_len(base, ip, offset_1);
            if (ml_rep >= 4u) {
                let gain2 = i32(ml_rep * 3u);
                let gain1 = i32(match_length * 3u) - highbit(off_base) + 1;
                if (gain2 > gain1) {
                    match_length = ml_rep;
                    off_base = 1u;
                    start = ip;
                }
            }
            let m1 = search_max(base, bbase, ip);
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
            if (LAZY == 2u && ip + 1u < PARSE_END) {
                ip += 1u;
                // search depth 2: repcode at ip, x4 rule
                let ml_rep2 = rep_len(base, ip, offset_1);
                if (ml_rep2 >= 4u) {
                    let gain2 = i32(ml_rep2 * 4u);
                    let gain1 = i32(match_length * 4u) - highbit(off_base) + 1;
                    if (gain2 > gain1) {
                        match_length = ml_rep2;
                        off_base = 1u;
                        start = ip;
                    }
                }
                let m2 = search_max(base, bbase, ip);
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
            n_seq = store_seq(base, sbase, n_seq, anchor, start - anchor, off, match_length);
            // deviation (dedup store): follow the decoder's reps, not zstd's offset_2 = offset_1.
            offset_1 = r0;
            offset_2 = r1;
        } else {
            // repcode 1 at ip+1 or later (ll > 0): stored as repcode 1, history unchanged.
            n_seq = store_seq(base, sbase, n_seq, anchor, start - anchor, offset_1, match_length);
        }
        anchor = start + match_length;
        ip = anchor;

        // check immediate repcode: offset_2 at ip, stored as (ll 0, repcode 1) = zstd's swap.
        while (ip <= PARSE_END && offset_2 > 0u) {
            let ml = rep_len(base, ip, offset_2);
            if (ml == 0u) { break; }
            let t = offset_2;
            offset_2 = offset_1;
            offset_1 = t;
            n_seq = store_seq(base, sbase, n_seq, anchor, 0u, offset_1, ml);
            ip += ml;
            anchor = ip;
        }
    }
    // last literals
    push_lits(anchor, BLOCK_SIZE);
    return n_seq;
}
