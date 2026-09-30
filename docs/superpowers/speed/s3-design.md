# S3 — Warp/subgroup-cooperative K3 lazy2 parse: implementation design

Status: design only (nothing implemented, nothing benchmarked). Normative references: `crates/gzc-core/src/lazy.rs`
(the oracle, including every `// deviation:` note), `crates/gzc-gpu/src/shaders/k3_lazy.wgsl` (the current
single-lane mirror), plan `docs/superpowers/plans/2026-09-30-speed-phase.md` (constraints), ideas-fable §4/§5.
Facts about naga/wgpu below were checked against the vendored `naga-30.0.1` and `wgpu-types-30.0.1` sources.

## 0. Decisions in one screen

| Question | Decision |
|---|---|
| Execution model | One block per workgroup, `@workgroup_size(W)`, **W = `adapter_info.subgroup_min_size` clamped to [8, 64]** (a compile-time constant injected by the host), so a workgroup is always one (possibly partial) subgroup whatever wave size the driver picks. All W lanes hold identical parse state; per-lane values reach control flow only through `subgroupBallot` / `subgroupShuffle` / `subgroupAll`. No workgroup memory, no barriers. |
| Subgroup-size independence | Results never depend on W: every cooperative primitive computes the same value as its sequential counterpart for any W ≥ 1 (proved per primitive in §3). W only changes speed. Ballot masks handle W = 64 via `.x` and `.y`. A one-time host probe checks the lane-id/ballot assumptions for the chosen W. |
| Fallback | No `Features::SUBGROUP` (or probe failure, or `GZC_K3_MODE=seq`): the **existing** `k3_lazy.wgsl` sequential kernel (`@workgroup_size(1)`), unchanged. A workgroup-memory cooperative variant is documented (§5.4) but not recommended for this experiment. |
| What becomes cooperative | (a) literal-run scan over the deterministic skip sequence, W candidates per step; (b) every `match_len` (capped-match extension, `rep_len`), 4·W bytes per step; (c) catch-up, W bytes per step; (d) the immediate `offset_2` loop (via b); (e) literal output as lane-parallel word stores with a replicated 0–3 byte accumulator, isolated so S4 can delete it. |
| Interface to S1 | `search_max(ip)` reads **one** packed word `best[bbase + ip]` and decodes it with two host-injected helpers `best_len(w)` / `best_off(w)`; the design never assumes field widths (`search_cap` may be up to 256). If S3 lands before S1, the helpers read today's two-word layout. |
| Expected | K3 61 ms/batch → ~15–20 ms on the 5090 (3–4× on a typical DDS block, ≥ 20× on flat blocks, which likely set today's wall time); 4060 projection ~100–120 ms → ~35–50 ms, ~25 ms with two blocks per workgroup (§7). |
| Stages | A: replicate the sequential parse on W lanes with cooperative `match_len` only (small diff, largest max-block win). B: cooperative scan. C: cooperative literal copy (or S4). D: deferral lookahead window. E: 2 blocks/workgroup (occupancy on the 4060). Each stage byte-identical and measured. |

## 1. Scope and interfaces

### 1.1 Inputs/outputs (unchanged)
Bindings, `seqs`/`lits`/`counts` layouts, `MAX_SEQS`, the injected `MatchParams` constants (`MIN_MATCH`, `SEARCH_CAP`,
`LAZY`) and the dispatch `dispatch_workgroups(n_blocks, 1, 1)` stay as in `k3_parse.wgsl`. The host additionally
injects `const W: u32` and, for the parse-path tests, `const K3_GUARD: bool`.

### 1.2 Abstract `best[]` access (S1 coupling)
```wgsl
// Injected by the host to match the best[] layout in use. S1 (packed, one u32 per position):
fn best_word(bbase: u32, ip: u32) -> u32 { return best[bbase + ip]; }      // bbase = b * BLOCK_SIZE
fn best_len(w: u32) -> u32 { return w & LEN_MASK; }                       // capped length, 0 = no match
fn best_off(w: u32) -> u32 { return w >> LEN_BITS; }
// Today's layout (if S3 lands first): bbase = b * BLOCK_SIZE * 2, best_word = ip*2 pair folded into
// (off << 16 | len) by best_word itself, or simply two loads. Only best_word/best_len/best_off change.
```
K3 relies on exactly the two properties the oracle relies on: `best_len(w) < MIN_MATCH` means "no match", and
`best_len(w) == SEARCH_CAP` means "extend with `match_len`". Words at `ip >= PARSE_END` are never read (§4).

### 1.3 S4 coupling
Literal emission is one function, `coop_push_lits(start, end)`, plus the replicated accumulator state
(`acc`, `acc_n`, `lit_w`). S4 deletes the function body except `n_lit += end - start`. Nothing else in the parse
touches `lits`.

## 2. Execution model

### 2.1 Why one subgroup per workgroup with W = subgroup_min_size
- Vulkan forms subgroups inside a workgroup; a workgroup of W ≤ subgroupSize invocations is one (partial)
  subgroup. Choosing W = the adapter's *minimum* subgroup size makes this true for every wave size the driver
  may pick per pipeline (AMD RDNA reports min 32 / max 64 and picks per pipeline; NVIDIA 32/32; Intel 8–16 /
  16–32). This is what makes the "results do not depend on subgroup size" rule hold by construction rather than
  by trusting a probe of a *different* pipeline.
- The alternative "workgroup 32, subgroup ops only if `subgroup_size >= 32`" fails on Intel (min 8) and on any
  device where the K3 pipeline is compiled with a smaller wave than the probe; the alternative "workgroup memory
  + `workgroupBarrier`" runs into naga's uniformity analysis (§5.4) and costs a barrier per collective.
- Cost of a small W: W = 8 gives 4× less cooperative width than 32 but the same results; Intel is not a target.
  On AMD wave64 with W = 32, half the wave idles: acceptable for this experiment (stage E can raise W to 64 when
  `subgroup_min_size == subgroup_max_size == 64`, e.g. GCN).

### 2.2 Occupancy on a 24-SM card (RTX 4060, Ada; RX 7600 similar within a factor)
- Residency is bounded by *workgroups per SM* (Ada: 24; Ampere: 16; Blackwell consumer: ≥ 24), not by threads:
  one block per workgroup → 24 SMs × 24 = **576 blocks resident** for a 1638-block batch ≈ 3 waves. This is the
  same limit the current `@workgroup_size(1)` kernel has (it uses 24 *threads* per SM); the cooperative kernel
  fills each resident warp instead of one lane of it. On the 5090 (170 SMs) the whole batch is a single wave
  either way, so 5090 K3 wall time ≈ the slowest block's time, while 4060 K3 time ≈ 3 × mean-block time + tail.
- Registers: ~16 replicated state words + ~12 primitive temporaries + 2 window words (stage D) ≈ 40 regs/lane;
  Ada allows 85 regs/lane at 24 warps/SM and 42 at 48 warps/SM. Workgroup memory: 0 bytes (the 16–32 KiB budget
  is not touched).
- Stage E (2 blocks per workgroup, `@workgroup_size(2*W)`, block = `wid.x * 2 + subgroup_id`): doubles residency
  to 48 warps/SM on Ada (≈ 1152 blocks) for ~2× on the 4060 where the SM's issue slots are idle. Only valid when
  `subgroup_min_size == subgroup_max_size` (so the two subgroups are exactly W each); otherwise BPW = 1.

### 2.3 Lane identity
`k = subgroup_invocation_id`. The host probe (§6.2) verifies that for a `@workgroup_size(W)` dispatch every
lane has `subgroup_invocation_id == local_invocation_index`, `subgroup_size >= W`, and `subgroupBallot(true)`
has exactly bits 0..W-1 set. Ballot bit i ↔ subgroup_invocation_id i (SPIR-V guarantee).

## 3. Cooperative primitives (with equivalence arguments)

Notation: [U] uniform (identical in every lane), [L] per-lane, ⟂ collective op. `ctz = countTrailingZeros`.
Design rule R1: **no branch or loop condition ever depends on an [L] value**; per-lane predicates go through
`select` or into a ballot. The only exception (R1'): `if (k == 0u) { store }` / `if (k < n) { store }` blocks that
contain only stores (the pattern K4/K5 already use before barriers). Rule R2: every load address in a lane is
in-bounds even when that lane's result is unused (clamp with `select`), so no lane reads outside its block.

```wgsl
// Position of the first set bit of a ballot, or W when none. Bits >= W are zero (inactive lanes).
fn first_lane(m: vec4<u32>) -> u32 {
    if (m.x != 0u) { return ctz(m.x); }              // [U]: m is uniform (ballot result)
    if (W > 32u && m.y != 0u) { return 32u + ctz(m.y); }
    return W;
}
// Branch-free unaligned load (load_u32_at has an `if (sh == 0u)`; keep it out of per-lane paths).
fn load_u32_nb(base: u32, byte_off: u32) -> u32 {
    let w = base + (byte_off >> 2u);
    let sh = (byte_off & 3u) * 8u;
    let hi = data[w + 1u] << ((32u - sh) & 31u);
    return (data[w] >> sh) | select(hi, 0u, sh == 0u);
}
```

### 3.1 `coop_match_len(base, p, q, cap)` ≡ `match_len` (common.wgsl), q < p
```wgsl
fn coop_match_len(base: u32, p: u32, q: u32, cap: u32, k: u32) -> u32 {
    let max = min(BLOCK_SIZE - p, cap);                       // [U]
    var n = 0u;                                               // [U]
    loop {
        let o = n + 4u * k;                                   // [L]
        let valid = o + 4u <= max;                            // [L], monotone: valid(k) ⇒ valid(k-1)
        let oc = select(0u, o, valid);                        // R2: in-bounds address for invalid lanes (max >= 4 at every call site, §4)
        var x = load_u32_nb(base, p + oc) ^ load_u32_nb(base, q + oc);
        x = select(0xFFFFFFFFu, x, valid);                    // invalid lane = "differs at byte 0"
        let f = first_lane(subgroupBallot(x != 0u));          // ⟂ → [U]
        if (f < W) {
            if (n + 4u * f + 4u <= max) {                     // lane f was valid: a real mismatch
                let xf = subgroupShuffle(x, f);               // ⟂ → [U]
                return n + 4u * f + (ctz(xf) >> 3u);
            }
            n += 4u * f;                                      // lane f is the first word past max: word loop ends here
            break;
        }
        n += 4u * W;                                          // all W words matched and were valid
    }
    loop {                                                    // byte tail, [U], <= 3 iterations
        if (n >= max || load_byte(base, p + n) != load_byte(base, q + n)) { break; }
        n += 1u;
    }
    return n;
}
```
Equivalence. The sequential loop examines words at n = 0, 4, 8, … while `n + 4 <= max` and returns at the first
differing word. The cooperative loop examines the same words in the same order (lane order = offset order),
because validity is monotone in k: the lanes below the first invalid lane are exactly the words the sequential
loop would still examine. A mismatch in a valid lane returns the same `n + ctz/8`; if no valid lane mismatches,
the word loop stops at the same `n` (the first offset with `n + 4 > max`), and the byte tail is verbatim.
Independent of W. Note the alignment `(p + oc) & 3 == p & 3` is the same in every lane and every step, so
adjacent lanes load adjacent words: two coalesced 128-byte lines per side per step at W = 32.
Optional (stage A+): a call site that knows the first `SEARCH_CAP` bytes match could start at `n = SEARCH_CAP`;
this is byte-identical only if `best_len == SEARCH_CAP` implies the first SEARCH_CAP bytes really match, which
holds for K2 output but not for arbitrary scripted `best[]` in tests. Keep `n = 0` unless a host-side check of
scripted tables is added.

### 3.2 `coop_rep_len(base, p, off)` ≡ `rep_len`
```wgsl
fn coop_rep_len(base: u32, p: u32, off: u32, k: u32) -> u32 {
    if (off == 0u || off > p) { return 0u; }                  // [U]
    let l = coop_match_len(base, p, p - off, 0xFFFFFFFFu, k);
    return select(0u, l, l >= 4u);
}
```
A miss (first 4 bytes differ) costs one step: the same two loads the sequential probe pays.

### 3.3 `coop_catch_up(base, start, anchor, off)` ≡ the catch-up `while`
Sequential: `while (start > anchor && start > off && byte[start-1] == byte[start-1-off]) { start--; ml++; }`.
```wgsl
fn coop_catch_up(base: u32, start0: u32, anchor: u32, off: u32, k: u32) -> u32 {   // bytes moved back
    var moved = 0u;                                           // [U]
    loop {
        let s = start0 - moved;                               // the oracle's `start` before lane k's step is s - k
        let sk = s - min(k, s);                               // [L], clamped
        let bound_ok = (k < s) && (sk > anchor) && (sk > off);          // [L], monotone in k
        let a = select(0u, sk - 1u, bound_ok);                          // R2
        let c = select(0u, sk - 1u - off, bound_ok);
        let ok = bound_ok && (load_byte(base, a) == load_byte(base, c)); // [L]
        let cnt = first_lane(subgroupBallot(!ok));            // ⟂ → [U]: length of the leading run of ok lanes
        moved += cnt;
        if (cnt < W) { break; }
    }
    return moved;
}
```
Equivalence: iteration j of the sequential loop tests the predicate at `start = start0 - j`; lane k of the
current step tests it at `start0 - moved - k`. The loop runs exactly as many iterations as the leading run of
true predicates, which is `ctz(ballot(!ok))` per step, continued while a full step succeeds. Independent of W.

### 3.4 `coop_push_lits(base, start, end)` ≡ `push_lits` (stage C; deleted by S4)
State `acc`, `acc_n` (0..3 pending bytes), `lit_w` (next word index), `n_lit` stay replicated [U].
```
1. head [U]: while (acc_n != 0 && start < end): acc |= byte(start) << 8*acc_n; acc_n++; start++;
              if acc_n == 4: if (k == 0) lits[lit_w] = acc; lit_w++; acc = 0; acc_n = 0        (<= 3 byte loads)
2. middle: n_words = (end - start) >> 2  [U];  for (t = 0; t < ceil(n_words / W); t++) {          // [U] trip count
              j = t*W + k;  jc = min(j, max(n_words,1)-1);                                       // R2 clamp
              v = load_u32_nb(base, start + 4*jc);                                               // (start+4j)+4 <= end <= BLOCK_SIZE
              if (j < n_words) { lits[lit_w + j] = v; }  }                                       // R1' store guard
           lit_w += n_words; start += 4*n_words
3. tail [U]: acc |= remaining 0..3 bytes (byte loads), acc_n += ...
4. n_lit += (end - start_original)
```
Equivalence: the byte stream written is identical (same bytes, same word packing); only which lane issues each
store changes. No lane ever reads `lits`, so lane-to-lane store ordering is irrelevant; the final `acc` flush in
`main` stays on lane 0.

### 3.5 Scan step (stage B): first position of the skip sequence where "something happens"
For the oracle, an iteration at `ip` with anchor `a` `continue`s (skips) iff `rep_len(ip+1, offset_1) == 0`
and `search_max(ip)` is None, i.e. iff **not** `H(ip) := rep4(ip+1, offset_1) || best_len(best_word(ip)) >= MIN_MATCH`,
where `rep4(p, off) := off != 0 && off <= p && u32(p) == u32(p-off)` (`match_len >= 4`, since `BLOCK_SIZE - p >= 8`
for p ≤ PARSE_END). Justification: `rep_len` is 0 or ≥ 4, `search_max` is None or ≥ MIN_MATCH ≥ 4, and the
skip test is `match_length < 4`; so H(ip) ⇔ the body proceeds to the deferral loop. (Uncapped extension can
only lengthen a match, so it never changes H.)
The skip sequence from `ip` with step regime `s1 = ((ip - a) >> 8) + 1` is `cand_k = ip + k·s1` as long as
`cand_k` stays in the regime (`((cand_k - a) >> 8) + 1 == s1`) and `cand_k < PARSE_END`; both conditions are
monotone in k, and `cand_{f}` for the first failing lane f is still the next element of the sequence (it is
`cand_{f-1} + s1`, and `cand_{f-1}` is in regime s1). Lane 0 is always valid.
```wgsl
// Returns the new ip [U]: either the first hit (then H(ip) holds) or the first candidate past the window
// (which is the next element of the skip sequence, possibly >= PARSE_END). Also returns lane h's probe results.
struct Scan { ip: u32, hit: bool, bw: u32, rep4: bool }
fn scan_step(base, bbase, ip, anchor, offset_1, k) -> Scan {
    let s1 = ((ip - anchor) >> 8u) + 1u;                       // [U]
    let cand = ip + k * s1;                                    // [L]
    let valid = (((cand - anchor) >> 8u) + 1u == s1) && (cand < PARSE_END);   // [L], monotone
    let c = select(ip, cand, valid);                           // R2: < PARSE_END
    let bw = best_word(bbase, c);                              // [L] (coalesced when s1 == 1)
    let rp = c + 1u;                                           // <= PARSE_END, rp + 4 <= BLOCK_SIZE
    let usable = offset_1 != 0u && offset_1 <= rp;
    let src = select(rp, rp - offset_1, usable);               // R2
    let rep4 = usable && (load_u32_nb(base, rp) == load_u32_nb(base, src));
    let hit = valid && (best_len(bw) >= MIN_MATCH || rep4);
    let h = first_lane(subgroupBallot(hit));                   // ⟂ [U]
    let f = first_lane(subgroupBallot(!valid));                // ⟂ [U] = number of valid lanes (>= 1)
    if (h == W) { return Scan(ip + f * s1, false, 0u, false); }         // f == W ⇒ ip + W*s1 is also the next element
    return Scan(ip + h * s1, true, subgroupShuffle(bw, h), subgroupShuffle(u32(rep4), h) != 0u);   // ⟂
}
```
Equivalence: every candidate below h is one the oracle visits and skips (¬H there, same step), and `cand_h` is
the first the oracle does not skip. If there is no hit, the oracle would visit all valid candidates, skip them,
and arrive at `cand_f` (or exit the outer loop if `cand_f >= PARSE_END`). Independent of W and of how many
steps it takes.
Tunable `SCAN_C` (candidates per lane, interleaved `cand = ip + (t·W + k)·s1`, first hit via `subgroupMin`) is
not needed for DDS (mean literal run ≪ 4·W bytes); keep C = 1.

## 4. The parse: control flow (pseudocode; every line maps to lazy.rs)

All variables below are [U] unless marked. Collectives happen only inside the primitives of §3 and the two
shuffles of `scan_step`. `PARSE_END`, gains, `highbit`, `off_base_for`, `apply_off_base` and the dedup-store rule
are verbatim from `k3_lazy.wgsl`.

```
main(wid, lid, k = subgroup_invocation_id, sg_size = subgroup_size):
  b = wid.x;  if (b >= arrayLength(&counts) / 2) return                      // per-workgroup uniform
  if (K3_GUARD) { ok = sg_size >= W && k == lid && ballot_bits(subgroupBallot(true)) == W;
                  if (!subgroupAll(ok)) { if (k == 0) counts[2b] = 0xFFFFFFFF; return } }   // §6.2
  base, sbase, bbase as today; r0=1 r1=4 r2=8; acc=0 acc_n=0 lit_w=b*BLOCK_SIZE/4 n_lit=0
  n_seq = (LAZY == 0) ? greedy_parse_coop(...) : lazy_parse_coop(...)
  if (k == 0) { if (acc_n > 0) lits[lit_w] = acc; counts[2b] = n_seq; counts[2b+1] = n_lit }

lazy_parse_coop(base, sbase, bbase, k) -> u32:
  n_seq = 0; anchor = 0; ip = 1
  offset_1 = select(0, r0, r0 <= 1); offset_2 = select(0, r1, r1 <= 1)
  while (ip < PARSE_END):                                                    // oracle: while ip < ilimit
    // ---- (a) literal scan: replaces the oracle's skip iterations ----
    sc = scan_step(base, bbase, ip, anchor, offset_1, k)
    ip = sc.ip
    if (!sc.hit) { continue; }                     // ip is the next skip-sequence element; loop condition re-checks PARSE_END
    // ---- the oracle's body at an ip with H(ip): match_length >= 4 is guaranteed ----
    match_length = 0; off_base = 1 (REPCODE1_TO_OFFBASE); start = ip + 1
    // check repCode at ip+1
    l = select(0, coop_rep_len(base, ip + 1, offset_1, k), sc.rep4)       // == rep_len: 0 when the first 4 bytes differ / off unusable
    if (l > 0) match_length = l
    // first search (depth 0)
    m0 = search_max_w(base, sc.bw, ip, k)                                   // (len, off_base); len 0 if best_len < MIN_MATCH; coop extension if == SEARCH_CAP
    if (m0.len > 0 && m0.len > match_length) { match_length = m0.len; start = ip; off_base = m0.ob }
    // (match_length < 4 is impossible here; the oracle's `ip += step; continue` branch is the scan)
    // ---- deferral loop: verbatim ----
    loop:
      if (ip + 1 >= PARSE_END) break
      ip += 1
      ml_rep = coop_rep_len(base, ip, offset_1, k)                          // (stage D: rep4 from the window first)
      if (ml_rep >= 4) { gain2 = ml_rep*3; gain1 = match_length*3 - highbit(off_base) + 1; if (gain2 > gain1) { match_length = ml_rep; off_base = 1; start = ip } }
      m1 = search_max(base, bbase, ip, k)                                   // uniform broadcast load of best_word(ip) + coop extension if capped
      if (m1.len > 0) { gain2 = m1.len*4 - highbit(m1.ob); gain1 = match_length*4 - highbit(off_base) + 4;
                        if (m1.len >= 4 && gain2 > gain1) { match_length = m1.len; off_base = m1.ob; start = ip; continue } }
      if (LAZY == 2 && ip + 1 < PARSE_END):
        ip += 1
        ml_rep2 = coop_rep_len(base, ip, offset_1, k)
        if (ml_rep2 >= 4) { gain2 = ml_rep2*4; gain1 = match_length*4 - highbit(off_base) + 1; if (gain2 > gain1) { match_length = ml_rep2; off_base = 1; start = ip } }
        m2 = search_max(base, bbase, ip, k)
        if (m2.len > 0) { gain2 = m2.len*4 - highbit(m2.ob); gain1 = match_length*4 - highbit(off_base) + 7;
                          if (m2.len >= 4 && gain2 > gain1) { match_length = m2.len; off_base = m2.ob; start = ip; continue } }
      break
    // ---- store ----
    if (off_base > 3):
      off = off_base - 3
      moved = coop_catch_up(base, start, anchor, off, k);  start -= moved;  match_length += moved      // (c)
      n_seq = store_seq(anchor, start - anchor, off, match_length)          // off_base_for/apply_off_base/coop_push_lits/lane-0 seqs write
      offset_1 = r0; offset_2 = r1                                          // deviation (dedup store): decoder's reps
    else:
      n_seq = store_seq(anchor, start - anchor, offset_1, match_length)     // repcode 1, ll > 0, history unchanged
    anchor = start + match_length; ip = anchor
    // ---- (d) immediate repcode: verbatim, rep_len → coop_rep_len ----
    while (ip <= PARSE_END && offset_2 > 0):
      ml = coop_rep_len(base, ip, offset_2, k)
      if (ml == 0) break
      swap(offset_1, offset_2)
      n_seq = store_seq(anchor, 0, offset_1, ml)
      ip += ml; anchor = ip
  coop_push_lits(base, anchor, BLOCK_SIZE)                                  // last literals
  return n_seq

search_max_w(base, bw, ip, k):  bl = best_len(bw); if (bl < MIN_MATCH) return (0, 0)
                                off = best_off(bw); len = bl
                                if (bl == SEARCH_CAP) len = coop_match_len(base, ip, ip - off, 0xFFFFFFFF, k)   // (b)
                                return (len, off + 3)
search_max(base, bbase, ip, k) = search_max_w(base, best_word(bbase, ip), ip, k)      // ip < PARSE_END at every call (loop guards)
store_seq(anchor, ll, offset, ml): ob = off_base_for(offset, ll); apply_off_base(ob, ll); coop_push_lits(base, anchor, anchor + ll)
                                   if (k == 0) { seqs[sbase + n_seq*3 + 0..2] = (ll, ml, ob) }   // R1'
                                   return n_seq + 1
```
Bounds used by R2 (`max >= 4` in `coop_match_len`): `search_max` calls have `ip < PARSE_END` so
`BLOCK_SIZE - ip > 8`; `coop_rep_len` calls have `p <= PARSE_END` (`ip + 1` in the body, `ip` in the deferral loop
with `ip < PARSE_END`, `ip <= PARSE_END` in the immediate loop), so `BLOCK_SIZE - p >= 8`; catch-up byte loads are
always inside the block. `best[]` is read only at `c < PARSE_END` (scan) and `ip < PARSE_END` (deferral).

Greedy path (`LAZY == 0`, lvl3/rung1): same shape — scan predicate `H_g(p) = (p > anchor && p >= r0 &&
rep_ge(p, r0, MIN_MATCH)) || best_len >= MIN_MATCH`, where `rep_ge` compares `ceil(MIN_MATCH/4)` words
(MIN_MATCH ≤ 8, so at most two, the second masked; `p + 8 <= BLOCK_SIZE` for `p < PARSE_END`); note the greedy
rep test needs `>= MIN_MATCH`, not `>= 4`. Emission uses `coop_match_len` for the rep length and extension.
Do it after the lazy path is measured; it is cheap once the primitives exist.

Stage D (deferral window): at entry to the deferral loop (after `start`/`match_length` are set), lane k loads
`bw_k = best_word(w0 + k)` (clamped to `< PARSE_END`) and `rep4_k = rep4(w0 + k, offset_1)` with `w0 = ip + 1`.
`offset_1` and `anchor` cannot change inside the loop, so the window is valid for the whole loop; the loop reads
`rep4 = shuffle(rep4_k, ip - w0)` and `bw = shuffle(bw_k, ip - w0)` and only runs `coop_rep_len` when rep4 is
true and the extension when capped. When `ip - w0 >= W` (rare chained `continue`s), refill with `w0 = ip`. This
removes ~2 round trips per sequence; equivalence is trivial (same values, different transport).

## 5. Uniformity analysis (naga 30 and hardware)

5.1 What naga checks. `valid/analyzer.rs`: `UniformityRequirements` has only `WORK_GROUP_BARRIER`,
`DERIVATIVE`, `IMPLICIT_LEVEL`. `Statement::SubgroupBallot / SubgroupCollectiveOperation / SubgroupGather`
carry **no** requirement, so naga never rejects a subgroup op for control-flow non-uniformity. `ControlBarrier`
/ `MemoryBarrier` (`workgroupBarrier`, `storageBarrier`, `subgroupBarrier`) do, and wgpu-core validates with
`ValidationFlags::all()`, so **the cooperative kernel must contain no barrier of any kind** (it does not need
one: no workgroup memory). Ballot/collective *results* are marked non-uniform by naga
(`E::SubgroupBallotResult → non_uniform_result: Some`), as are locals, `var<private>` and read-write storage;
this only matters if someone later adds a barrier under a branch on such a value.

5.2 What hardware needs. Every collective must execute with all W lanes active. This design guarantees it
*dynamically*: all state is replicated and every branch/loop condition is [U] by rule R1 (§3). The only per-lane
control flow is the R1' store blocks, which contain no collectives and reconverge at their merge block, the
same pattern K4/K5 already rely on ahead of barriers on this driver stack. Because control flow is dynamically
uniform, no lane ever diverges, so the design does not depend on `VK_KHR_shader_maximal_reconvergence` or on
ITS reconvergence heuristics.

5.3 Loop shapes. Loops are written `loop { if (uniform_cond) { break; } ... }` with collectives at the loop-body
top level or inside `if (uniform)`; `continue` jumps are on [U] values. WGSL `while`/`for` lower to the same
structure, so either spelling is fine in the subgroup path.

5.4 If a workgroup-memory fallback is ever written (not recommended now). naga's analysis treats every local as
non-uniform, so `if (off_base > 3u) { workgroupBarrier(); }` is rejected (`NonUniformControlFlow`) even though it
is uniform in practice; `if (cond) { break; }` before a barrier is accepted (a `Break` sets no disruptor, only
`Return`/`Kill` under a non-uniform branch do), and a *function* containing a barrier called under a
non-uniform `if` is rejected too (`process_call` propagates requirements). Two ways out: (i) predicate-as-data,
running each cooperative primitive unconditionally with a "count = 0" predicate, barriers at loop-body top
level; (ii) launder each branch value through `workgroupUniformLoad` (its result is uniform for naga) at the
cost of one barrier per decision. Also `arrayLength(&counts)` early-return must precede all barriers as in K5.

## 6. Host-side changes (`context.rs`, `compressor.rs`)

6.1 Feature. `GpuContext::new` requests `Features::SUBGROUP` when `adapter.features()` has it (shared with S2;
one `ctx.subgroups: bool` field). No `SUBGROUP_BARRIER` needed. `ctx.adapter_info.subgroup_min_size` /
`subgroup_max_size` (`wgpu::AdapterInfo`) give W.

6.2 Probe (once per `Kernels::new`, ~40 lines). Build a tiny `@workgroup_size(W)` compute shader that writes,
per lane, `(local_invocation_index, subgroup_invocation_id, subgroup_size, ballot.x, ballot.y)` to a small
buffer; dispatch one workgroup; read back; require `subgroup_invocation_id == local_invocation_index` for all
lanes, `subgroup_size >= W`, and the ballot equal to the W-bit mask. On failure log and use the sequential
kernel. W selection: `W = min(64, max(8, subgroup_min_size))` rounded down to a power of two; overrides
`GZC_K3_W=8|16|32|64` (must be ≤ `subgroup_min_size`, for the W-independence tests) and `GZC_K3_MODE=seq|coop`.
Stage E adds `GZC_K3_BPW=2` (only when `subgroup_min_size == subgroup_max_size`).

6.3 Pipelines. `Kernels` holds `parse_seq` (today's `k3_parse.wgsl + k3_lazy.wgsl`, always built) and
`parse_coop: Option<_>` (new `k3_parse_coop.wgsl` with the §3 primitives + §4 parse, built with
`enable subgroups;` and `const W`, `const K3_GUARD`). `record_parse` picks one; same bind group layout and
dispatch. `KERNEL_NAMES` unchanged. `K3_GUARD = true` for `parses_from_best`/`frames_from_best` kernels and for
`--verify`; the sentinel `counts[2b] = 0xFFFFFFFF` is caught by `read_outputs`'s existing `bad counts` check
(and by the frame comparison in `--verify`).

## 7. Cost model and expected speedup

7.1 Where the 61 ms go today (a model, not a measurement; the first task of the experiment is to measure it,
see 7.4). The sequential kernel is a chain of dependent round trips (RT) of ~0.3–0.6 µs (L2 hits ~0.3 µs;
`best[]` is streamed from DRAM at 8 B/position, ~0.5 µs but sector-amortised). Per block:
- Scan: one RT per visited literal position (best word + two rep words, independent). Runs shorter than 256
  bytes are visited byte by byte. For 1.34× DDS, the literal share is roughly 55–70 % of the block
  (≈ 70–90 K positions), in runs with a mean of ~10–15 bytes ⇒ ~70–90 K RTs ≈ 25–45 ms. **This is the mean
  block's dominant cost.**
- Deferral: ~2–3 visits per sequence (~6–7 K sequences ⇒ ~15–20 K RTs ≈ 5–10 ms), each an independent rep probe
  + best load; extensions only when capped (mean ml ≈ 6–8 on DDS, so rarely).
- Flat / periodic regions (alpha planes, mip tails, zero padding): a run of R bytes costs up to five uncapped
  extensions of ~R bytes (`search_max(ip)`, `rep_len(ip+1)`, `search_max(ip+1)`, `rep_len(ip+2)`,
  `search_max(ip+2)`), i.e. ~1.25·R word compares of 2 loads each; R = 64 K ⇒ ~20 K iterations ≈ 5–10 ms per
  run, and a whole zero block ≈ 40–60 ms. On the 5090 the batch is one wave, so **the slowest block sets K3's
  wall time**; the 61 ms is consistent with such blocks, which is why stage A (cooperative extension only) is
  the first thing to measure.
- Catch-up and `push_lits`: byte loops, but their loads are independent (throughput ~10–20 ns/byte) ⇒ ~1–3 ms.

7.2 After S3 (per block, W = 32). Scan: one RT (plus ~40 instructions of ballots/shuffles) per ≤ 32 candidates
⇒ ~3–5 K RTs instead of 70–90 K. Deferral: ~1 RT per visited position (stage B), ~0.3 with the window
(stage D). Extension/rep/catch-up: 128 B (resp. 32 B) per RT ⇒ a 64 K run costs ~500 RTs instead of 20 K,
a zero block ~2 ms instead of ~50. Per sequence the floor is ~4–5 dependent RTs (rep probe at ip+1, deferral
probes, catch-up, immediate probe; literal copy is 1 RT per 128 B and disappears with S4). Typical DDS block:
~6.5 K × ~5 RT + ~4 K scan RT ≈ 35 K RTs ≈ 12–18 ms vs ~40–50 ms today ⇒ **~3× on the mean block, ≥ 20× on
flat blocks; K3 61 → ~15–20 ms on the 5090** (wall time then set by the mean/tail, no longer by flat outliers).
Instruction issue is not a limit: ~10 warps/SM on the 5090, 24 on the 4060, each mostly waiting on memory.

7.3 4060 projection (scaling model of ideas-fable §0). The parse is latency-bound and a 4060's L2 latency and
per-warp behaviour are similar, so per-block warp time falls by the same ~3× (mean) / 20× (flat). Residency
576 blocks ⇒ ~3 waves: today ≈ 3 × mean(40 ms) + tail ≈ 100–130 ms; after S3 ≈ 3 × 15 + tail ≈ 40–55 ms;
stage E (48 warps/SM) ≈ 25–35 ms. The gain carries over undiminished because it cuts per-warp latency chains,
not aggregate bandwidth (K3 bandwidth is < 5 GB/s per batch either way).

7.4 Measure before building (cheap, CPU-only, ~30 lines in a test or bench binary): instrument `lazy_parse`
(or a copy in the bench crate) to count per block: scan candidates, deferral visits, extension bytes,
rep-probe bytes, catch-up bytes, literal bytes; run it over `corpus_blocks_match_cpu_per_preset`'s 300-block
sample and report mean, p99 and max. This decides whether the 61 ms is a max-block (flat) or a mean-block
(scan) phenomenon and therefore whether stage A alone already moves the 5090 number. The same counters give
the per-stage expected RT reduction to compare against the K3 timestamps.

## 8. Implementation stages (each: byte-identical, tests green, timestamps recorded in `speed-log.md`)

A. `k3_parse_coop.wgsl`: entry with `@workgroup_size(W)`, replicated state, R1'/R2 rules, `coop_match_len`,
   `coop_rep_len`, `coop_catch_up`; the parse body is the current sequential code with `match_len` calls
   replaced and stores on lane 0; `push_lits` still the byte loop (uniform, every lane computes `acc`, lane 0
   stores). Host: feature, probe, pipeline selection, env overrides, guard. Differential suite at W = 8/16/32
   and seq. Measure: this alone should collapse flat-block outliers.
B. `scan_step` and the restructured outer loop (§4). New scan tests (§9). Measure.
C. `coop_push_lits` (skip if S4 is scheduled right after; then S4 deletes it).
D. Deferral window. Measure.
E. `BPW = 2` (`@workgroup_size(2*W)`, `subgroup_id`), only for `min == max` devices; measure on the 5090 (expect
   little) and record the 4060 projection.
Then the greedy path (lvl3/rung1) with the same primitives, if stages A–B paid off.

## 9. Test plan

9.1 Existing coverage (must pass for `coop` at every feasible W and for `seq`, at both block sizes, all four
presets): `k3_lazy_hand_built_best_matches_cpu` (the 21 `lazy::cases` replayed via `parses_from_best` and
`frames_from_best`, batched and one by one: gain ties, `continue` after wins, catch-up bounds, dedup-store
rule, immediate loop incl. at PARSE_END, PARSE_END guards with poisoned `best[PARSE_END..]`, depth-0 tie,
min_match 6), `gpu_matches_reference`, `gpu_matches_reference_depth4`, `gpu_matches_reference_lazy_variants`,
`block_ends_with_adjacent_neighbours` (zeros / period3 / exact_block neighbours: flat-run extensions to the
block end), `batch_of_300_mixed`, `gpu_frames_identical_{no_,}huffman`, `gpu_frames_batch_of_300_mixed`,
`stream_frames_match_cpu_non_lvl3_presets`, `corpus_blocks_match_cpu_per_preset` (ignored; run it with the
corpus), plus one `--verify` lvl9 run per stage and the lvl3 anchor test. Run the suite with `GZC_K3_W` = 8, 16,
32 (and 64 on a wave64 device) and with `GZC_K3_MODE=seq`; outputs must be identical across all of them.

9.2 New targeted tests (new file `crates/gzc-gpu/tests/k3_coop.rs`; expected values come from `lazy_parse` /
`compress_block`, so gzc-core is not edited; scripted `best[]` via `parses_from_best`, which already validates
that scripted entries are in-block). Parametrise every case over W ∈ {8, 16, 32} where the position depends on
W (the test reads W from the same selection function the host uses).
- T1 skip-sequence fidelity: random block (no natural matches), one planted 8-byte explicit match at P, for
  P ∈ {255, 256, 257, 258, 259, 510, 511, 512, 513, 514, 515, 516, 767–771, 1023–1028} (regime boundaries: with
  anchor 0 the oracle visits 1..256, 258, 260, …, 512, 515, 518, …) and P ∈ {W−1, W, W+1, 2W−1, 2W, 2W+1,
  256+2(W−1), 256+2W, 256+2W+2} (lane-window boundaries), and the same with a rep hit instead of an explicit
  match (an early match sets `offset_1`; plant a 4-byte repeat at `cand + 1` for visited and for skipped cands).
  GPU == oracle in every case (found iff P is in the sequence). Variant with `MIN_MATCH = 6` (RUNG2-based) and
  planted lengths 4 and 5 (must be skipped) and 6 (found).
- T2 runs ending at PARSE_END: all-literal random block (`n_seq = 0`, `n_lit = BLOCK_SIZE`); planted matches at
  PARSE_END−1 and PARSE_END−2 with poisoned `best[PARSE_END..]`; scan windows straddling PARSE_END (anchor set
  by a match ending at PARSE_END − W − 3, − W − 1, − W, − 1); a rep repeat exactly at PARSE_END after a match
  ending at PARSE_END−1 (probe at `cand + 1 = PARSE_END`).
- T3 extension geometry: capped hits (`best_len = SEARCH_CAP`, real run) for every `(p & 3, q & 3)` and true
  lengths ∈ {SEARCH_CAP, 4W−1, 4W, 4W+1, 4W+3, 8W, 8W+5, BLOCK_SIZE−p, BLOCK_SIZE−p−1, −2, −3} (the last four
  exercise the "first invalid lane" exit and the byte tail).
- T4 rep lengths across windows: `offset_1` repeats at ip+1 of lengths {4, 5, 4W−1, 4W, 4W+1}; immediate
  `offset_2` chains of 3 repeats with lengths {8, 4W, to-block-end}; the immediate loop starting at PARSE_END.
- T5 catch-up: backward runs of {1, W−1, W, W+1, 2W+3} bytes; bounded by `anchor` at exactly lane W−1 and lane
  W; bounded by `start > off` (source at 0); a run with a mismatch in the middle (bytes match at s−1, s−2,
  differ at s−3, match again at s−4..s−8: the count must be 2).
- T6 literal packing (until S4): sequences with `lit_len` ∈ {0, 1, 2, 3, 4, 5, 4W−1, 4W, 4W+1, 8W+3} at every
  accumulator phase (arrange previous `lit_len`s so `acc_n` ∈ 0..3); `BlockOutput.literals` must match.
- T7 W-independence: T1–T6 and 9.1 at each W and `seq`; assert byte-equal frames across configurations.
- T8 flat and periodic blocks at both block sizes (zeros, period 1/2/3/4/8/16 at every phase, a zero block with
  one byte flipped at BLOCK_SIZE−5..−1) — long extensions crossing every lane boundary and the byte tail at
  the block end.
- T9 probe unit test: the §6.2 probe returns the expected lane map on the dev GPU; a forced-failure path
  (simulate by asking W > subgroup_min_size) selects the sequential kernel.

9.3 Mutation plan (each mutant must make at least one named test fail; run before declaring the stage done):
| Mutant | Detected by |
|---|---|
| scan: drop the regime check from `valid` (step s1 for all lanes) | T1 (P = 257 found by the mutant, skipped by the oracle) |
| scan: drop `cand < PARSE_END` | `lazy_near_parse_end` (poisoned entries), T2 |
| scan: `ip += W*s1` instead of `f*s1` on a miss | T1 (P = 258: mutant visits 257, 259, … and misses it) |
| scan: hit uses `best_len >= 4` instead of `>= MIN_MATCH` | T1 min_match-6 variant |
| scan: rep probe at `cand` instead of `cand + 1` | T1 rep variant |
| `coop_match_len`: return on the first invalid lane without the byte tail | T3 (BLOCK_SIZE−p−1..−3), T8 |
| `coop_match_len`: `valid = o + 4 < max` | T3 (BLOCK_SIZE−p exact) |
| `first_lane`: ignore `.y` | only observable at W = 64 (AMD wave64); note in the log if untestable |
| `coop_catch_up`: count `popcount(ok)` instead of the leading run | T5 (mismatch in the middle) |
| `coop_catch_up`: `sk >= anchor` | `catch_up_stops_at_anchor`, T5 |
| `coop_push_lits`: wrong accumulator phase / off-by-one in `n_words` | T6 |
| immediate loop `ip < PARSE_END` | `immediate_offset2_at_parse_end` |
| depth-2 guard `ip + 1 <= PARSE_END` | `lazy_near_parse_end/no_b` |
| dedup store: `offset_2 = offset_1` (zstd's rule) | `dedup_store_enables_old_rep1_immediate` |
| store on all lanes instead of lane 0 (benign race) | not detectable by output; keep as a review item |
| guard: skip the `subgroupAll` | T9 |

## 10. Risks and fallback

- **Subgroup convergence on some driver** (a collective executed with lanes missing). Mitigated by R1/R1'
  (dynamically uniform control flow, no reliance on reconvergence), the probe, the `K3_GUARD` sentinel in test
  builds, and the full differential suite; residual risk is a driver bug, in which case `GZC_K3_MODE=seq` is the
  escape hatch and the experiment is recorded as reverted for that device.
- **Wave size differs between probe and K3 pipeline** (AMD picks per pipeline): eliminated by W = min size (one
  partial subgroup for any wave ≥ W). Stage E's `BPW = 2` is gated on `min == max`.
- **naga rejections**: none expected (no barriers). If an implementer adds `workgroupBarrier` anywhere, naga
  will reject it under any local-dependent branch (§5.4); do not.
- **Register spills** from inlining every primitive into the parse (each has its own loop): check
  `wgpu` validation output / NSight for spills; if the coop kernel spills, out-line the primitives via a
  `switch`-free helper and reduce replicated temporaries. Even spilled, the kernel is memory-latency bound.
- **Performance risk**: if the 61 ms is mean-bound (scan) rather than max-bound, stage A alone shows little;
  stage B is where the gain is. If neither moves K3 by > 3 %, revert to `seq` and record; the primitives are
  still reusable by a later megakernel (ideas-fable §7).
- **S1 interlock**: `best_word/best_len/best_off` are the only coupling; land whichever first and adjust the
  three helpers. The scan's coalescing benefits from S1's one-word layout (stride 4 B instead of 8 B).
- **S4 interlock**: `coop_push_lits` is dead code once S4 lands; stage C is optional if S4 follows within days.
- **Fallback plan**: (1) device without `SUBGROUP` or failing probe → sequential kernel, automatically;
  (2) correctness problem found late → `GZC_K3_MODE=seq` default with the coop pipeline behind an opt-in flag
  while it is fixed; (3) speed disappointing → keep stage A only (cooperative `match_len`, smallest diff, kills
  the flat-block outliers) and revert B–D.
