// K3 greedy parse, subgroup-cooperative. Appended by the host after k3_reps.wgsl and
// k3_parse.wgsl (whose bindings, rep history, literal count, off_base_for / apply_off_base and
// sequential greedy_parse it reuses), and compiled on a device with Features::SUBGROUP with
// `const W: u32` (the workgroup size: the adapter's minimum subgroup size, or
// GpuOptions::k3_width) and W_MASK_X / W_MASK_Y (the ballot of W lanes). Entry point
// `main_coop`, one block per workgroup of W lanes, which form one (possibly partial) subgroup.
//
// Every lane holds the same parse state (r0/r1/r2, ip, anchor, ...). Per-lane values reach
// control flow only through subgroupBallot / subgroupShuffle / subgroupAll, so all branch and loop
// conditions are uniform and every collective runs with all W lanes active (no barriers, no
// workgroup memory). Per-lane booleans combine with `&` / `|`, never `&&` / `||`: naga lowers
// those to an `if` on the left operand, a lane-dependent branch right before the ballot that
// would need the lanes reconverged after it (VK_KHR_shader_maximal_reconvergence is not
// enabled). The only per-lane branches are lane 0's stores, which feed no subgroup operation.
// Stores happen on lane 0 only, or on disjoint addresses per lane.
// docs/design/m6/subgroup-audit.md lists every subgroup call with its argument. Each
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
// local index, ballot not exactly W bits), lane 0 runs the sequential greedy_parse instead: the
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
// mode: MATCH_FIRST_EQ: the caller knows the first 4 bytes match and 4 <= max (the compare starts
// at 4). MATCH_WIDE: all words cooperatively from 0 (capped extensions, which are long).
const NO_DIFF: u32 = 0xFFFFFFFFu;
const MATCH_FIRST_EQ: u32 = 1u;
const MATCH_WIDE: u32 = 2u;

fn coop_match_len(base: u32, p: u32, q: u32, cap: u32, k: u32, mode: u32) -> u32 {
    let max = min(BLOCK_SIZE - p, cap);
    var n = 0u;
    if (mode == MATCH_FIRST_EQ) {
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

// One sequence of `ll` literals from `anchor` then `ml` bytes at `offset`, with the decoder's rep
// bookkeeping (greedy_parse's emit); seqs written by lane 0. Returns the new sequence count.
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

// == greedy_parse (k3_parse.wgsl) with the cooperative literal scan and match_len.
// The sequential loop skips p exactly when neither the rep test (p > anchor, p >= r0 and
// match_len(p, p - r0) >= MIN_MATCH, i.e. its first MIN_MATCH bytes match: BLOCK_SIZE - p > 8)
// nor best[p] (len >= MIN_MATCH) holds.
// Lane k tests the k-th element of the skip sequence from p, p + k * step, while it is below
// PARSE_END and still in the same step regime; both limits are monotone in k, so the valid lanes
// are a prefix. p moves to the first hit, else to the first element past the valid lanes, which is
// the next element of the sequence either way.
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

@compute @workgroup_size(W)
fn main_coop(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    // One block per workgroup. li: the lane within this block's W lanes.
    let li = lid;
    let b = wid.x;
    // No early return for a workgroup past the last block: its lanes take neither branch below
    // (no loads, no stores).
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
        n_seq = coop_greedy_parse(base, sbase, bbase, k);
    } else if (in_range & (li == 0u)) {
        // Unexpected lane layout: the exact sequential parse on one lane.
        n_seq = greedy_parse(base, sbase, bbase);
    }
    if (in_range & (li == 0u)) {
        counts[b * 2u] = n_seq;
        counts[b * 2u + 1u] = n_lit;
    }
}
