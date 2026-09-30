// K3 lazy / lazy2 parse, subgroup-cooperative (speed phase S3, design .superpowers/speed/s3-design.md).
// Appended by the host after k3_parse.wgsl and k3_lazy.wgsl (whose bindings, rep history, literal
// accumulator, off_base_for / apply_off_base, highbit and sequential lazy_parse it reuses), and
// compiled (on a device with Features::SUBGROUP) with `const W: u32` (the workgroup size: the adapter's minimum
// subgroup size, or GZC_K3_W) and W_MASK_X / W_MASK_Y (the ballot of W lanes). Entry point
// `main_coop`, one block per workgroup of W lanes, which form one (possibly partial) subgroup.
//
// Every lane holds the same parse state (r0/r1/r2, ip, anchor, ...). Per-lane values reach
// control flow only through subgroupBallot / subgroupShuffle / subgroupAll, so all branch and loop
// conditions are uniform and every collective runs with all W lanes active (no barriers, no
// workgroup memory). Stores happen on lane 0 only, or on disjoint addresses per lane. Each
// cooperative primitive returns exactly what its sequential counterpart returns, for any W >= 1,
// so the output never depends on W (see the equivalence notes per function).
//
// If the lane layout is not the assumed one (subgroup smaller than W, lane ids not equal to the
// local index, ballot not exactly W bits), lane 0 runs the sequential lazy_parse instead: the
// output is still exact, only slower.

// Position of the first set bit of a ballot, or W when none. Bits >= W are zero (inactive lanes).
fn first_lane(m: vec4<u32>) -> u32 {
    if (m.x != 0u) { return countTrailingZeros(m.x); }
    if (W > 32u && m.y != 0u) { return 32u + countTrailingZeros(m.y); }
    return W;
}

// load_u32_at without its branch (its `if (sh == 0u)` would be per-lane control flow).
fn load_u32_nb(base: u32, byte_off: u32) -> u32 {
    let w = base + (byte_off >> 2u);
    let sh = (byte_off & 3u) * 8u;
    let hi = data[w + 1u] << ((32u - sh) & 31u);
    return (data[w] >> sh) | select(hi, 0u, sh == 0u);
}

// == match_len(base, p, q, cap) (common.wgsl), q < p. Lane k compares the word at n + 4k; the
// lanes the sequential loop would still compare (n + 4k + 4 <= max) are a prefix of the lanes, so
// the first lane that mismatches or is invalid is where the sequential word loop returns or
// stops. The byte tail is the sequential one. Invalid lanes load at p / q (in the block).
fn coop_match_len(base: u32, p: u32, q: u32, cap: u32, k: u32) -> u32 {
    let max = min(BLOCK_SIZE - p, cap);
    var n = 0u;
    // The sequential loop's first word, uniformly (same address in every lane): most calls are
    // rep probes that end here, and then cost what the sequential probe costs.
    if (4u <= max) {
        let x0 = load_u32_at(base, p) ^ load_u32_at(base, q);
        if (x0 != 0u) { return countTrailingZeros(x0) >> 3u; }
        n = 4u;
    }
    loop {
        let o = n + 4u * k;
        let valid = o + 4u <= max;
        let oc = select(0u, o, valid);
        var x = load_u32_nb(base, p + oc) ^ load_u32_nb(base, q + oc);
        x = select(0xFFFFFFFFu, x, valid);
        let f = first_lane(subgroupBallot(x != 0u));
        if (f < W) {
            if (n + 4u * f + 4u <= max) {
                let xf = subgroupShuffle(x, f);
                return n + 4u * f + (countTrailingZeros(xf) >> 3u);
            }
            n += 4u * f;
            break;
        }
        n += 4u * W;
    }
    loop {
        if (n >= max || load_byte(base, p + n) != load_byte(base, q + n)) { break; }
        n += 1u;
    }
    return n;
}

// == rep_len (k3_lazy.wgsl).
fn coop_rep_len(base: u32, p: u32, off: u32, k: u32) -> u32 {
    if (off == 0u || off > p) { return 0u; }
    let l = coop_match_len(base, p, p - off, 0xFFFFFFFFu, k);
    return select(0u, l, l >= 4u);
}

// == search_max (k3_lazy.wgsl) for the best[] word `w` at ip.
fn coop_search_max_w(base: u32, w: u32, ip: u32, k: u32) -> vec2<u32> {
    let bl = best_len_of(w);
    if (bl < MIN_MATCH) { return vec2<u32>(0u, 0u); }
    let off = best_off_of(w);
    var len = bl;
    if (bl == SEARCH_CAP) {
        len = coop_match_len(base, ip, ip - off, 0xFFFFFFFFu, k);
    }
    return vec2<u32>(len, off + 3u);
}

// Number of bytes the lazy catch-up `while (start > anchor && start > off && byte[start-1] ==
// byte[start-1-off]) start--` moves `start0` back. Lane k tests the predicate the sequential loop
// tests at iteration moved + k; the loop runs for the leading run of true predicates.
fn coop_catch_up(base: u32, start0: u32, anchor: u32, off: u32, k: u32) -> u32 {
    var moved = 0u;
    loop {
        let s = start0 - moved;
        let sk = s - min(k, s);
        let bound_ok = (k < s) && (sk > anchor) && (sk > off);
        let a = select(0u, sk - 1u, bound_ok);
        let c = select(0u, sk - 1u - off, bound_ok);
        let ok = bound_ok && (load_byte(base, a) == load_byte(base, c));
        let cnt = first_lane(subgroupBallot(!ok));
        moved += cnt;
        if (cnt < W) { break; }
    }
    return moved;
}

// == push_lits (k3_parse.wgsl): every lane keeps the same accumulator, lane 0 stores.
fn coop_push_lits(base: u32, start: u32, end: u32, k: u32) {
    for (var i = start; i < end; i++) {
        acc |= load_byte(base, i) << (acc_n * 8u);
        acc_n += 1u;
        if (acc_n == 4u) {
            if (k == 0u) { lits[lit_w] = acc; }
            lit_w += 1u;
            acc = 0u;
            acc_n = 0u;
        }
    }
    n_lit += end - start;
}

// == store_seq (k3_lazy.wgsl), seqs written by lane 0.
fn coop_store_seq(base: u32, sbase: u32, n_seq: u32, anchor: u32, ll: u32, offset: u32, ml: u32, k: u32) -> u32 {
    let ob = off_base_for(offset, ll);
    apply_off_base(ob, ll);
    coop_push_lits(base, anchor, anchor + ll, k);
    if (k == 0u) {
        let s = sbase + n_seq * 3u;
        seqs[s] = ll;
        seqs[s + 1u] = ml;
        seqs[s + 2u] = ob;
    }
    return n_seq + 1u;
}

// == lazy_parse (k3_lazy.wgsl) with every match_len cooperative.
fn coop_lazy_parse(base: u32, sbase: u32, bbase: u32, k: u32) -> u32 {
    var n_seq = 0u;
    var anchor = 0u;
    var ip = 1u;
    var offset_1 = select(0u, r0, r0 <= 1u);
    var offset_2 = select(0u, r1, r1 <= 1u);

    while (ip < PARSE_END) {
        var match_length = 0u;
        var off_base = 1u;
        var start = ip + 1u;

        let l = coop_rep_len(base, ip + 1u, offset_1, k);
        if (l > 0u) {
            match_length = l;
        }
        let m0 = coop_search_max_w(base, best[bbase + ip], ip, k);
        if (m0.x > 0u && m0.x > match_length) {
            match_length = m0.x;
            start = ip;
            off_base = m0.y;
        }
        if (match_length < 4u) {
            ip += ((ip - anchor) >> 8u) + 1u;
            continue;
        }

        loop {
            if (ip + 1u >= PARSE_END) { break; }
            ip += 1u;
            let ml_rep = coop_rep_len(base, ip, offset_1, k);
            if (ml_rep >= 4u) {
                let gain2 = i32(ml_rep * 3u);
                let gain1 = i32(match_length * 3u) - highbit(off_base) + 1;
                if (gain2 > gain1) {
                    match_length = ml_rep;
                    off_base = 1u;
                    start = ip;
                }
            }
            let m1 = coop_search_max_w(base, best[bbase + ip], ip, k);
            if (m1.x > 0u) {
                let gain2 = i32(m1.x * 4u) - highbit(m1.y);
                let gain1 = i32(match_length * 4u) - highbit(off_base) + 4;
                if (m1.x >= 4u && gain2 > gain1) {
                    match_length = m1.x;
                    off_base = m1.y;
                    start = ip;
                    continue;
                }
            }
            if (LAZY == 2u && ip + 1u < PARSE_END) {
                ip += 1u;
                let ml_rep2 = coop_rep_len(base, ip, offset_1, k);
                if (ml_rep2 >= 4u) {
                    let gain2 = i32(ml_rep2 * 4u);
                    let gain1 = i32(match_length * 4u) - highbit(off_base) + 1;
                    if (gain2 > gain1) {
                        match_length = ml_rep2;
                        off_base = 1u;
                        start = ip;
                    }
                }
                let m2 = coop_search_max_w(base, best[bbase + ip], ip, k);
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
            break;
        }

        if (off_base > 3u) {
            let off = off_base - 3u;
            let moved = coop_catch_up(base, start, anchor, off, k);
            start -= moved;
            match_length += moved;
            n_seq = coop_store_seq(base, sbase, n_seq, anchor, start - anchor, off, match_length, k);
            offset_1 = r0;
            offset_2 = r1;
        } else {
            n_seq = coop_store_seq(base, sbase, n_seq, anchor, start - anchor, offset_1, match_length, k);
        }
        anchor = start + match_length;
        ip = anchor;

        while (ip <= PARSE_END && offset_2 > 0u) {
            let ml = coop_rep_len(base, ip, offset_2, k);
            if (ml == 0u) { break; }
            let t = offset_2;
            offset_2 = offset_1;
            offset_1 = t;
            n_seq = coop_store_seq(base, sbase, n_seq, anchor, 0u, offset_1, ml, k);
            ip += ml;
            anchor = ip;
        }
    }
    coop_push_lits(base, anchor, BLOCK_SIZE, k);
    return n_seq;
}

@compute @workgroup_size(W)
fn main_coop(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let b = wid.x;
    if (b >= arrayLength(&counts) / 2u) { return; }
    let base = block_base(b);
    let sbase = b * MAX_SEQS * 3u;
    let bbase = b * BLOCK_SIZE;
    let k = sid;

    r0 = 1u;
    r1 = 4u;
    r2 = 8u;
    acc = 0u;
    acc_n = 0u;
    lit_w = b * (BLOCK_SIZE / 4u);
    n_lit = 0u;

    // Lane-layout guard: the workgroup is one subgroup whose lane ids are 0..W-1.
    let m = subgroupBallot(true);
    let lanes_ok = sg_size >= W && sid == lid && m.x == W_MASK_X && m.y == W_MASK_Y;
    var n_seq = 0u;
    if (subgroupAll(lanes_ok)) {
        n_seq = coop_lazy_parse(base, sbase, bbase, k);
    } else {
        // Unexpected lane layout: the exact sequential parse on one lane.
        if (lid != 0u) { return; }
        n_seq = lazy_parse(base, sbase, bbase);
    }
    if (lid == 0u) {
        if (acc_n > 0u) {
            lits[lit_w] = acc;
        }
        counts[b * 2u] = n_seq;
        counts[b * 2u + 1u] = n_lit;
    }
}
