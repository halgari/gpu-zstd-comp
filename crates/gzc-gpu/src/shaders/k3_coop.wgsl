// K3 greedy / lazy / lazy2 parse, subgroup-cooperative (speed phase S3, design .superpowers/speed/s3-design.md).
// Appended by the host after k3_parse.wgsl and k3_lazy.wgsl (whose bindings, rep history, literal
// count, off_base_for / apply_off_base, highbit and sequential lazy_parse it reuses), and
// compiled (on a device with Features::SUBGROUP) with `const W: u32` (the workgroup size: the adapter's minimum
// subgroup size, or GZC_K3_W) and W_MASK_X / W_MASK_Y (the ballot of W lanes). Entry point
// `main_coop`, one block per W lanes, which form one (possibly partial) subgroup; BPW (1 or 2)
// blocks per workgroup.
//
// Every lane holds the same parse state (r0/r1/r2, ip, anchor, ...). Per-lane values reach
// control flow only through subgroupBallot / subgroupShuffle / subgroupAll, so all branch and loop
// conditions are uniform and every collective runs with all W lanes active (no barriers, no
// workgroup memory). Per-lane booleans combine with `&` / `|`, never `&&` / `||`: naga lowers
// those to an `if` on the left operand, a lane-dependent branch right before the ballot that
// would need the lanes reconverged after it (VK_KHR_shader_maximal_reconvergence is not
// enabled; .superpowers/m6-research/subgroup-audit.md). The only per-lane branches are lane 0's
// stores. Stores happen on lane 0 only, or on disjoint addresses per lane. Each
// cooperative primitive returns exactly what its sequential counterpart returns, for any W >= 1,
// so the output never depends on W (see the equivalence notes per function).
//
// Termination (the host builds this module without naga's forced loop bounding): every loop below
// carries a `// Terminates:` note naming a variable that strictly increases (or decreases) each
// iteration and the bound that ends it. None of the bounds relies on best[] being sane: a best[]
// word may claim a match past the block end (K2 never writes one, tests can), and the loops still
// end (anchors past BLOCK_SIZE end the parse; push_lits counts nothing for them).
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
// mode: MATCH_PROBE first compares the first word uniformly (same address in every lane): most
// rep probes end there, and then cost what the sequential probe costs. MATCH_FIRST_EQ: the caller
// knows the first 4 bytes match and 4 <= max (the compare starts at 4). MATCH_WIDE: all words
// cooperatively from 0 (capped extensions, which are long).
const NO_DIFF: u32 = 0xFFFFFFFFu;
const MATCH_PROBE: u32 = 0u;
const MATCH_FIRST_EQ: u32 = 1u;
const MATCH_WIDE: u32 = 2u;

fn coop_match_len(base: u32, p: u32, q: u32, cap: u32, k: u32, mode: u32) -> u32 {
    let max = min(BLOCK_SIZE - p, cap);
    var n = 0u;
    if (mode == MATCH_FIRST_EQ) {
        n = 4u;
    } else if (mode == MATCH_PROBE && 4u <= max) {
        let x0 = load_u32_at(base, p) ^ load_u32_at(base, q);
        if (x0 != 0u) { return countTrailingZeros(x0) >> 3u; }
        n = 4u;
    }
    // Lane k's candidate is 4k + (index of its first differing byte), or NO_DIFF; an invalid lane
    // counts as differing at its byte 0. Candidates grow with k (4k + b < 4(k + 1)), so the
    // minimum is the first lane that differs or is invalid; which of the two it is follows from
    // its lane index m / 4.
    // Terminates: p < BLOCK_SIZE at every call (callers keep ip <= PARSE_END), so max <= BLOCK_SIZE;
    // n grows by 4W per full step, and a step whose lanes reach n + 4k + 4 > max has an invalid lane,
    // so m != NO_DIFF and the loop returns or breaks.
    loop {
        let o = n + 4u * k;
        let valid = o + 4u <= max;
        let oc = select(0u, o, valid);
        var x = load_u32_nb(base, p + oc) ^ load_u32_nb(base, q + oc);
        x = select(0xFFFFFFFFu, x, valid);
        let m = subgroupMin(select(NO_DIFF, 4u * k + (countTrailingZeros(x) >> 3u), x != 0u));
        if (m != NO_DIFF) {
            if (n + (m & ~3u) + 4u <= max) {
                return n + m;
            }
            n += m;
            break;
        }
        n += 4u * W;
    }
    // Terminates: n increases by 1 up to max (at most 3 iterations after the word loop).
    loop {
        if (n >= max || load_byte(base, p + n) != load_byte(base, q + n)) { break; }
        n += 1u;
    }
    return n;
}

// == rep_len (k3_lazy.wgsl).
fn coop_rep_len(base: u32, p: u32, off: u32, k: u32) -> u32 {
    if (off == 0u || off > p) { return 0u; }
    let l = coop_match_len(base, p, p - off, 0xFFFFFFFFu, k, MATCH_PROBE);
    return select(0u, l, l >= 4u);
}

// == rep_len(p, off) when its first 4 bytes are known to match (then off is usable, the result
// is >= 4, and BLOCK_SIZE - p >= 8 at every call site).
fn coop_rep_len_hit(base: u32, p: u32, off: u32, k: u32) -> u32 {
    return coop_match_len(base, p, p - off, 0xFFFFFFFFu, k, MATCH_FIRST_EQ);
}

// == search_max (k3_lazy.wgsl) for the best[] word `w` at ip.
fn coop_search_max_w(base: u32, w: u32, ip: u32, k: u32) -> vec2<u32> {
    let bl = best_len_of(w);
    if (bl < MIN_MATCH) { return vec2<u32>(0u, 0u); }
    let off = best_off_of(w);
    var len = bl;
    if (bl == SEARCH_CAP) {
        len = coop_match_len(base, ip, ip - off, 0xFFFFFFFFu, k, MATCH_WIDE);
    }
    return vec2<u32>(len, off + 3u);
}

// Number of bytes the lazy catch-up `while (start > anchor && start > off && byte[start-1] ==
// byte[start-1-off]) start--` moves `start0` back. Lane k tests the predicate the sequential loop
// tests at iteration moved + k; the loop runs for the leading run of true predicates.
fn coop_catch_up(base: u32, start0: u32, anchor: u32, off: u32, k: u32) -> u32 {
    var moved = 0u;
    // Terminates: a full step (cnt == W) needs lane W - 1 in bounds, i.e. start0 - moved - (W - 1)
    // > anchor >= 0, so moved grows by W while staying below start0; any other step breaks.
    loop {
        let s = start0 - moved;
        let sk = s - min(k, s);
        let bound_ok = (k < s) & (sk > anchor) & (sk > off);
        let a = select(0u, sk - 1u, bound_ok);
        let c = select(0u, sk - 1u - off, bound_ok);
        let ok = bound_ok & (load_byte(base, a) == load_byte(base, c));
        let cnt = first_lane(subgroupBallot(!ok));
        moved += cnt;
        if (cnt < W) { break; }
    }
    return moved;
}

// == store_seq (k3_lazy.wgsl), seqs written by lane 0.
fn coop_store_seq(base: u32, sbase: u32, n_seq: u32, anchor: u32, ll: u32, offset: u32, ml: u32, k: u32) -> u32 {
    let ob = off_base_for(offset, ll);
    apply_off_base(ob, ll);
    push_lits(anchor, anchor + ll);
    if (k == 0u) {
        let s = sbase + n_seq * 3u;
        seqs[s] = ll;
        seqs[s + 1u] = ml;
        seqs[s + 2u] = ob;
    }
    return n_seq + 1u;
}

// Deferral window: lane k holds best[win_b + k] and whether the rep probe at win_b + k + 1 matches
// its first 4 bytes (== rep_len(win_b + k + 1, offset_1) > 0), for lanes k < win_n. Position P is
// covered when 1 <= P - win_b < win_n. A literal scan with step 1 leaves exactly this in its lanes
// (win_b = the scan's start), so the deferral after it usually needs no loads; otherwise the
// window is filled from P - 1. offset_1 cannot change between the scan and the end of the deferral
// loop. Filled positions are clamped below PARSE_END (never visited at or past it).
var<private> win_b: u32;
var<private> win_n: u32;
var<private> win_bw: u32;
var<private> win_rep4: u32;

fn win_fill(base: u32, bbase: u32, p: u32, off1: u32, k: u32) {
    win_b = p - 1u;
    win_n = W;
    win_bw = best[bbase + min(p - 1u + k, PARSE_END - 1u)];
    let rp = min(p + k, PARSE_END - 1u);
    let usable = (off1 != 0u) & (off1 <= rp);
    let src = select(rp, rp - off1, usable);
    win_rep4 = select(0u, 1u, usable & (load_u32_nb(base, rp) == load_u32_nb(base, src)));
}

// (best[p], rep4 at p) from the window, refilled when p is not covered. p > win_b.
fn win_at(base: u32, bbase: u32, p: u32, off1: u32, k: u32) -> vec2<u32> {
    if (p - win_b == 0u || p - win_b >= win_n) { win_fill(base, bbase, p, off1, k); }
    let i = p - win_b;
    return vec2<u32>(subgroupShuffle(win_bw, i), subgroupShuffle(win_rep4, i - 1u));
}

// == lazy_parse (k3_lazy.wgsl): cooperative literal scan, match_len and catch-up.
fn coop_lazy_parse(base: u32, sbase: u32, bbase: u32, k: u32) -> u32 {
    var n_seq = 0u;
    var anchor = 0u;
    var ip = 1u;
    var offset_1 = select(0u, r0, r0 <= 1u);
    var offset_2 = select(0u, r1, r1 <= 1u);

    // Terminates: every iteration ends with a larger ip, bounded by PARSE_END. A miss adds
    // n_valid * step >= 1 (lane 0 is always valid: ip < PARSE_END); the fallback adds step >= 1; a
    // store sets ip = anchor = start + match_length >= (the hit position) + 4, and the immediate
    // loop only adds to it.
    while (ip < PARSE_END) {
        // Literal scan: the sequential loop body `continue`s (skips ip) exactly when neither the
        // rep probe at ip + 1 (rep_len > 0 <=> the first 4 bytes match, as BLOCK_SIZE - (ip + 1)
        // >= 8) nor best[ip] (len >= MIN_MATCH) has a match.
        // Lane k tests the k-th element of the skip sequence from ip, ip + k * step, while it is
        // below PARSE_END and still in the same step regime; both limits are monotone in k, so the
        // valid lanes are a prefix. ip moves to the first hit, else to the first element past the
        // valid lanes, which is the next element of the sequence either way.
        let step = ((ip - anchor) >> 8u) + 1u;
        let cand = ip + k * step;
        let valid = (((cand - anchor) >> 8u) + 1u == step) & (cand < PARSE_END);
        let c = select(ip, cand, valid);
        let bw = best[bbase + c];
        let rp = c + 1u;
        let usable = (offset_1 != 0u) & (offset_1 <= rp);
        let src = select(rp, rp - offset_1, usable);
        let rep4 = usable & (load_u32_nb(base, rp) == load_u32_nb(base, src));
        let hit = valid & ((best_len_of(bw) >= MIN_MATCH) | rep4);
        let h = first_lane(subgroupBallot(hit));
        let n_valid = first_lane(subgroupBallot(!valid));
        if (h == W) {
            ip += n_valid * step;
            continue;
        }
        // The scan's lanes are the deferral window when it visited consecutive positions.
        win_b = ip;
        win_n = select(0u, n_valid, step == 1u);
        win_bw = bw;
        win_rep4 = select(0u, 1u, rep4);
        ip += h * step;
        let ip_bw = subgroupShuffle(bw, h);
        let ip_rep4 = subgroupShuffle(select(0u, 1u, rep4), h) != 0u;

        // The sequential body at ip.
        var match_length = 0u;
        var off_base = 1u;
        var start = ip + 1u;

        if (ip_rep4) {
            match_length = coop_rep_len_hit(base, ip + 1u, offset_1, k);
        }
        let m0 = coop_search_max_w(base, ip_bw, ip, k);
        if (m0.x > 0u && m0.x > match_length) {
            match_length = m0.x;
            start = ip;
            off_base = m0.y;
        }
        if (match_length < 4u) {
            // Only when a capped best[ip] extends to fewer than 4 bytes, which K2 never produces
            // (a scripted best[] in a test can).
            ip += step;
            continue;
        }

        // Terminates: ip increases by 1 or 2 per iteration and the loop breaks at ip + 1 >= PARSE_END.
        loop {
            if (ip + 1u >= PARSE_END) { break; }
            ip += 1u;
            let d1 = win_at(base, bbase, ip, offset_1, k);
            var ml_rep = 0u;
            if (d1.y != 0u) { ml_rep = coop_rep_len_hit(base, ip, offset_1, k); }
            if (ml_rep >= 4u) {
                let gain2 = i32(ml_rep * 3u);
                let gain1 = i32(match_length * 3u) - highbit(off_base) + 1;
                if (gain2 > gain1) {
                    match_length = ml_rep;
                    off_base = 1u;
                    start = ip;
                }
            }
            let m1 = coop_search_max_w(base, d1.x, ip, k);
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
                let d2 = win_at(base, bbase, ip, offset_1, k);
                var ml_rep2 = 0u;
                if (d2.y != 0u) { ml_rep2 = coop_rep_len_hit(base, ip, offset_1, k); }
                if (ml_rep2 >= 4u) {
                    let gain2 = i32(ml_rep2 * 4u);
                    let gain1 = i32(match_length * 4u) - highbit(off_base) + 1;
                    if (gain2 > gain1) {
                        match_length = ml_rep2;
                        off_base = 1u;
                        start = ip;
                    }
                }
                let m2 = coop_search_max_w(base, d2.x, ip, k);
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

        // Terminates: ml >= 4 on every continuing iteration, so ip grows until ip > PARSE_END.
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
    push_lits(anchor, BLOCK_SIZE);
    return n_seq;
}

// == greedy_parse (k3_parse.wgsl) with the cooperative literal scan and match_len.
// The sequential loop skips p exactly when neither the rep test (p > anchor, p >= r0 and
// match_len(p, p - r0) >= MIN_MATCH, i.e. its first MIN_MATCH bytes match: BLOCK_SIZE - p > 8)
// nor best[p] (len >= MIN_MATCH) holds; the scan is the lazy one's with that predicate at p.
// REP_HI_MASK (host-injected) masks the second word to its MIN_MATCH - 4 bytes.
fn coop_greedy_parse(base: u32, sbase: u32, bbase: u32, k: u32) -> u32 {
    var n_seq = 0u;
    var p = 0u;
    var anchor = 0u;
    // Terminates: p grows each iteration (a miss by n_valid * step >= 1, a store by len >= 1, the
    // fallback by step >= 1) and the loop ends at PARSE_END.
    while (p < PARSE_END) {
        let step = ((p - anchor) >> 8u) + 1u;
        let cand = p + k * step;
        let valid = (((cand - anchor) >> 8u) + 1u == step) & (cand < PARSE_END);
        let c = select(p, cand, valid);
        let bw = best[bbase + c];
        let usable = (c > anchor) & (c >= r0);
        let src = select(c, c - r0, usable);
        let x0 = load_u32_nb(base, c) ^ load_u32_nb(base, src);
        let x1 = (load_u32_nb(base, c + 4u) ^ load_u32_nb(base, src + 4u)) & REP_HI_MASK;
        let rep_ok = usable & (x0 == 0u) & (x1 == 0u);
        let hit = valid & (rep_ok | (best_len_of(bw) >= MIN_MATCH));
        let h = first_lane(subgroupBallot(hit));
        if (h == W) {
            p += first_lane(subgroupBallot(!valid)) * step;
            continue;
        }
        p += h * step;
        let p_bw = subgroupShuffle(bw, h);
        let p_rep = subgroupShuffle(select(0u, 1u, rep_ok), h) != 0u;

        var off = 0u;
        var len = 0u;
        if (p_rep) {
            off = r0;
            len = coop_match_len(base, p, p - r0, 0xFFFFFFFFu, k, MATCH_FIRST_EQ);
        } else {
            off = best_off_of(p_bw);
            len = best_len_of(p_bw);
            if (len == SEARCH_CAP) {
                // K2 stopped comparing at the cap: extend to the full length.
                len = coop_match_len(base, p, p - off, 0xFFFFFFFFu, k, MATCH_WIDE);
            }
        }
        if (len == 0u) {
            // Only when a capped best[p] extends to nothing (a scripted best[] in a test).
            p += step;
            continue;
        }
        n_seq = coop_store_seq(base, sbase, n_seq, anchor, p - anchor, off, len, k);
        p += len;
        anchor = p;
    }
    push_lits(anchor, BLOCK_SIZE);
    return n_seq;
}

@compute @workgroup_size(W * BPW)
fn main_coop(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    // BPW blocks per workgroup, one per subgroup of W lanes (BPW > 1 only when every subgroup has
    // exactly W lanes). li: the lane within this block's W lanes.
    let li = lid % W;
    let b = wid.x * BPW + lid / W;
    // No early return for lanes past the last block: the layout guard's ballot must see every
    // lane of the subgroup (a return here is subgroup-uniform only under the host's BPW rule).
    // They take neither branch below (no loads, no stores).
    let in_range = b < arrayLength(&counts) / 2u;
    let base = block_base(b);
    let sbase = b * MAX_SEQS * 3u;
    let bbase = b * BLOCK_SIZE;
    let k = sid;

    r0 = 1u;
    r1 = 4u;
    r2 = 8u;
    n_lit = 0u;

    // Lane-layout guard: this block's W lanes are one subgroup whose lane ids are 0..W-1.
    let m = subgroupBallot(true);
    // K3_FORCE_FALLBACK (host-injected, test-only) takes the sequential branch below on purpose.
    let lanes_ok = !K3_FORCE_FALLBACK & (sg_size >= W) & (sid == li) & (m.x == W_MASK_X) & (m.y == W_MASK_Y);
    var n_seq = 0u;
    let coop = subgroupAll(lanes_ok & in_range);
    if (coop) {
        if (LAZY == 0u) {
            n_seq = coop_greedy_parse(base, sbase, bbase, k);
        } else {
            n_seq = coop_lazy_parse(base, sbase, bbase, k);
        }
    } else if (in_range & (li == 0u)) {
        // Unexpected lane layout: the exact sequential parse on one lane.
        if (LAZY == 0u) {
            n_seq = greedy_parse(base, sbase, bbase);
        } else {
            n_seq = lazy_parse(base, sbase, bbase);
        }
    }
    if (in_range & (li == 0u)) {
        counts[b * 2u] = n_seq;
        counts[b * 2u + 1u] = n_lit;
    }
}
