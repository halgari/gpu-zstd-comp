// K3opt: one DP pass of the M5 optimal parse (presets opt14 / opt16), == gzc_core::opt::dp_pass_with
// on Engine::Ring (the normative GPU spec, documented on opt::Engine), statement by statement.
// Two entry points, dispatched back to back:
//
// main_opt (one invocation per 4 KiB segment, NSEG per block; WG lanes per workgroup, BPW blocks
//   per workgroup, or a block across NSEG / WG workgroups): a prologue builds the block's static
//   price tables in workgroup memory (PRICE_MODE 0: zstd's first-block statistics,
//   opt::Prices::block_init, the literal histogram counted by the workgroup's lanes of the block;
//   PRICE_MODE 1: the tables in `prices`, opt::Prices verbatim, 377 words per block: lit[256]
//   ll[36] ml[53] of[32]). Then each lane runs opt::Dp::segment_ring over its segment: segment 0
//   from ip 1 with INITIAL_REPS, segment k > 0 from ip = anchor = k*SEG with reps [0, 0, 0];
//   iend = the segment end, ilimit = iend - 8.
//   Phase 1, the DP:
//   - The lane loop is flattened (one trip per series start probe or series position, no
//     `continue`) so that the lanes' get_all_matches calls run together; each trip issues its
//     position's loads (data, candidate words) first.
//   - The DP nodes live in a ring of RING_N = sufficient_len + 1 slots per lane (position pos of
//     the series in slot pos % RING_N). Only the prices (ring_p) are in workgroup memory, or in
//     private memory (the fallback when the adapter's workgroup storage is too small), both
//     declared by the host (rix); the rest of each node is in the global scratch `scr` (M5 T3b:
//     workgroup memory bounds K3opt's residency, and 4 B per node lets a 2900-block batch run in
//     one wave on an RTX 5090).
//   - `trace` (the dead K1 pred buffer, 2 words per position) gets (mlen | litlen << 8, offBase)
//     of every node when it becomes final; the backward trace reads it, never the ring. The entry
//     at iend (a node at the segment end) is never read and not written (it would be the next
//     segment's first position).
//   - The end of a series (commit_ring) only updates the parse state (end_series: ip, anchor and
//     reps follow from the last stretch alone) and logs the series in `seqs` (free until
//     main_fixup); no lane waits for another's backward trace inside the DP loop.
//   Phase 2: the logged series' backward traces, last series first (emit), write the
//   segment's sequences in reverse order (RAW_REVERSED) into its own words of `best` (the
//   candidate words, 2 per position, no longer read): sequence i from the segment's end at
//   best[wbase() + 3*i ..], trailer (n_seq, final anchor, sum of match_len, final reps) at
//   best[wbase() + SEG_META ..], for k3_fixup.wgsl's main_fixup (at most SEG / 3 sequences).
// main_fixup (k3_fixup.wgsl): concatenation, literal carry and off_base re-encoding against the
//   block's true reps (== lazy::encode_raw), one workgroup per block.
//
// Candidate words (reference::CandWords, K2opt): best[2*(b*BLOCK_SIZE + p)] = offA | lenA << 16 |
// lenB << 24, best[.. + 1] = offB, plus DEAD_BIT at a dead position; lengths capped at SEARCH_CAP
// (a stored SEARCH_CAP is extended).
//
// Precondition (K2opt's output satisfies it): every candidate record lies in the block before its
// position, and its length is the true common length capped at SEARCH_CAP. The kernel trusts the
// words (no bounds checks on offsets); only the host test harness (k3opt.rs `OptBuffers::upload`)
// validates scripted ones, the pipeline runs it right after K2opt.
//
// Passes (M5 T4, opt::passes): pass n + 1 is priced from pass n's own output histogram
// (opt::Hist::of_output of the fixed-up block parse: literal bytes, and each sequence's LL code
// with the literal carry, ML code, and OF code of the off_base under the block's true decoder
// reps). A HIST_OUT pass (the cheap ones) keeps the candidate words: its raw sequences go to the
// segment's series-log words of `seqs` instead of `best`, and a workgroup epilogue (hist_epilogue)
// histograms them: every lane counts its own sequences and literal bytes, then the block's
// segment-0 lane applies what main_fixup would change (the literal carry of each segment's first
// sequence, and the OF codes of the sequences it re-encodes against the true reps), and the
// block's lanes store the Hist (377 words: lit[256] ll[36] ml[53] of[32]) at `prices`[b * 377].
// The next pass's prologue (PRICE_MODE 3) turns it into price tables (opt::Prices::from_hist).
// No fix-up runs after a HIST_OUT pass. A HIST_OUT pass needs each block's segments in one
// workgroup (LPB == NSEG).
//
// PRICE_MODE: 0 block-init (opt::Prices::block_init), 1 explicit tables in `prices`, 2 the prior
// seed (codes::OPT_PRIOR_* tables plus opt::cover_literals of the block's candidate words; needs
// LPB == NSEG), 3 opt::Prices::from_hist of the Hist in `prices`.
//
// Consts injected by the host: MIN_MATCH, SEARCH_CAP, MAX_SEQS, SEG_LOG2, WG, SUFF
// (sufficient_len), LEVEL (optLevel 0 | 2), PRICE_MODE, HIST_OUT, BI_LL / BI_ML / BI_OF
// (block-init LL / ML / OF prices), PR_LL / PR_ML / PR_OF (prior-seed LL / ML / OF prices),
// LL_BITS / ML_BITS / ML_CODE (codes.rs), DEAD_BIT (reference::DEAD_BIT), SCHED_HDR
// (k3_sched.wgsl), and the price ring's declaration (ring_p, rix).
//
// Persistent passes (M6 A4, a07): with WG == NSEG the host runs the entry point main_opt_persist
// instead of main_opt. The grid is the same (one workgroup per block), but each workgroup loops:
// it takes the next item of the batch's heavy-first block order (k3_sched.wgsl) from a global
// counter and runs prologue, DP and epilogue for that block, until the order is used up. An
// NVIDIA Vulkan dispatch of this kernel larger than one wave runs as synchronous waves (each
// pays its slowest block); the loop lets a free slot take the next block at once, heaviest first.
// All per-block state is keyed by the block (gsg), not by the workgroup.

const SEG: u32 = 1u << SEG_LOG2;
const NSEG: u32 = BLOCK_SIZE >> SEG_LOG2;
const SEG_WORDS: u32 = 2u * SEG;
const SEG_META: u32 = SEG_WORDS - 6u;
const RAW_REVERSED: bool = true;
// Lanes of one block in a workgroup, and blocks per workgroup (WG < NSEG: a block spans
// NSEG / WG workgroups, each with its own copy of the block's price tables).
const LPB: u32 = min(WG, NSEG);
const BPW: u32 = WG / LPB;
const RING_N: u32 = SUFF + 1u;
const MAXP: i32 = 1 << 30;
const OPT_NUM: u32 = 4096u;
const MATCH_FEE: i32 = 51;
const PRICE_WORDS: u32 = 377u;
// seqs words per segment for the series log: at most SEG / 3 series with a match (each covers at
// least one 3-byte match, and series are disjoint), 3 words each.
const LOG_WORDS: u32 = (3u * MAX_SEQS) / NSEG;
// HIST_OUT: the segment's raw sequences, in order, end at word lbase() + 3 * LOG_SEQS (written last
// first while phase 2 still reads the series log from the front: raw sequence j from the end
// lands at slot LOG_SEQS - 1 - j, which holds a series entry already read, since each unread
// series still has at least one sequence to emit and a segment has at most SEG / 3).
const LOG_SEQS: u32 = LOG_WORDS / 3u;
const_assert 3u * (SEG / 3u) <= LOG_WORDS;
const_assert SEG / 3u <= LOG_SEQS;
const HIST_WORDS: u32 = 377u;
// HIST_OUT counts in the workgroup's `hist` words, no extra workgroup memory (K3opt's residency
// is bound by it: 870 more bytes per workgroup cost 33 % at wg16, M5 T4 log): word e of local
// block pbf() holds literal byte e's count in its low 17 bits (at most BLOCK_SIZE <= 65536) and
// code e's count in its high 15 bits (at most MAX_SEQS < 32768), codes numbered as in the Hist:
// LL 0..36, ML 36..89, OF 89..121. Counts never go negative, so the packed atomics are exact.
const LIT_BITS: u32 = 17u;
const LIT_MASK: u32 = (1u << LIT_BITS) - 1u;
const CODE_ONE: u32 = 1u << LIT_BITS;
const H_LL: u32 = 0u;
const H_ML: u32 = 36u;
const H_OF: u32 = 89u;
const_assert BLOCK_SIZE < (1u << LIT_BITS);
const_assert MAX_SEQS < (1u << (32u - LIT_BITS));
// HIST_OUT: the segment summary (sequences, final anchor, final reps) for hist_epilogue, in the
// segment's first trace words (dead after its phase 2).
const SUM_WORDS: u32 = 5u;
// The summary's first trace word for segment k of block b (the segment's first position's trace
// words: written by dp's HIST_OUT tail, read by hist_epilogue).
fn summ_base(b: u32, k: u32) -> u32 { return 2u * (b * BLOCK_SIZE + k * SEG); }
const_assert BLOCK_SIZE <= 65536u;
const_assert WG % NSEG == 0u || NSEG % WG == 0u;
// Dead positions (M6 A3, a09; reference::find_cands): DEAD_BIT in a candidate word w1 marks p as
// dead: no earlier position of the block shares p's first 3 bytes, so no candidate and no rep
// reaches MIN_MATCH there and get_all_matches(p) is empty under every rep state. The DP skips
// such positions' searches: outside a series in a tight loop before the trip (st_ip += 1 per
// dead position), inside one the position only gets its literal extension. The rep-length memo
// stays exact across a skipped search (its lengths hold at any later position of the segment,
// see mem). (A run length per position, capped at K2opt's 256-position tiles, measured no faster
// here and cost K2opt two workgroup barriers.) DEAD_BIT is injected by the host
// (reference::DEAD_BIT), so the kernel and the oracle cannot drift apart.
// Literal-only positions of a series folded into one trip (dp): measured on the optLevel-0
// passes only (a09: the final pass is faster without; it has fewer `+128` skips).
const FOLD: bool = LEVEL == 0u;
const_assert 3u * (SEG / 3u) <= SEG_META;
const_assert NSEG * (SEG / 3u) <= MAX_SEQS;
// ML code == mlen - 3 for every priced length (mlen <= SUFF <= 34).
const_assert SUFF <= 34u;
const_assert MIN_MATCH == 3u;

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> best: array<u32>;
@group(0) @binding(2) var<storage, read_write> seqs: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> trace: array<u32>;
// PRICE_MODE 1: opt::Prices per block; PRICE_MODE 3 / HIST_OUT: opt::Hist per block (the
// previous pass's, replaced by this pass's). PRICE_WORDS == HIST_WORDS words per block.
@group(0) @binding(5) var<storage, read_write> prices: array<u32>;
// The DP nodes' payload (reps, litlen, mlen, offBase): 3 consecutive words per node, RING_N nodes
// per lane (see six; lanes interleaved per slot measured 4 % slower, 16-byte nodes 0.5 %). Only
// the price stays in ring_p.
@group(0) @binding(6) var<storage, read_write> scr: array<u32>;
// The batch's block schedule (k3_sched.wgsl; main_opt_persist only): the counter at word 0, the
// order at SCHED_HDR + n.
@group(0) @binding(7) var<storage, read_write> sched: array<atomic<u32>>;

// zstd LL_Code for lit_len < 64 (codes::ll_code).
const LL_CODE: array<u32, 64> = array<u32, 64>(
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u,
    16u, 16u, 17u, 17u, 18u, 18u, 19u, 19u, 20u, 20u, 20u, 20u, 21u, 21u, 21u, 21u,
    22u, 22u, 22u, 22u, 22u, 22u, 22u, 22u, 23u, 23u, 23u, 23u, 23u, 23u, 23u, 23u,
    24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u);

// Price tables of the workgroup's blocks (local block pbf() at pbf() * size). Every price is in
// 0..65536 (opt::Prices; a PRICE_MODE 1 table is checked on upload), so the literal, LL and OF
// prices are stored as u16 pairs (entry 2w in the low half of word w) in u32 words. The ML
// prices, read four at a time in the relaxation loop, stay i32: unpacking them there measured
// 0.8 % slower than the 66 B they save per block (M6 A1; 264 B saved per block overall).
// Literal prices by byte value.
var<workgroup> p_lit: array<u32, 128u * BPW>;
// The LL and OF tables, one packed array (tab): LL price by code (litlen >= 64: code
// highbit(litlen) + 19) at T_LLC, ll_price(litlen) for litlen < 64 at T_LLS, OF price by code
// at T_OF.
const T_LLC: u32 = 0u;
const T_LLS: u32 = 36u;
const T_OF: u32 = 100u;
const TAB_N: u32 = 132u;
const TAB_W: u32 = TAB_N / 2u;
var<workgroup> p_tab: array<u32, TAB_W * BPW>;
// ML price by match length 0..=SUFF (3..=SUFF used).
var<workgroup> p_ml: array<i32, RING_N * BPW>;
// The prologue's literal frequencies (PRICE_MODE 0 / 2); with HIST_OUT, then the pass's packed
// histogram (see LIT_BITS). A pass that uses neither declares one word (M6 A1: the final pass
// carried 1 KiB it never touched; workgroup memory bounds K3opt's residency).
// The one-word placeholders (`hist` without HIST_USED, `hsum` under PRICE_MODE 1) must never be
// indexed: every access to `hist` must stay under HIST_USED, every access to `hsum` under
// PRICE_MODE != 1 (WGSL clamps or discards an out-of-range workgroup index, so a stray access
// would not fault, it would silently alias word 0).
const HIST_USED: bool = HIST_OUT || PRICE_MODE == 0u || PRICE_MODE == 2u;
var<workgroup> hist: array<atomic<u32>, select(1u, 256u * BPW, HIST_USED)>;
// Per block: the table sums (lit, ll, ml, of) and, for PRICE_MODE 2, the uncovered bytes
// (unused by PRICE_MODE 1).
var<workgroup> hsum: array<atomic<u32>, select(1u, 5u * BPW, PRICE_MODE != 1u)>;

// Lane constants (M6 A1 register diet): everything per lane derives from gsg, the lane's global
// segment index (workgroup_id * WG + local_invocation_index == block * NSEG + segment), instead
// of being held in registers across the DP.
var<private> gsg: u32;
// The lane in its workgroup (local_invocation_index).
fn lane_id() -> u32 { return gsg % WG; }
// The lane's local block in the workgroup (0 when a workgroup holds at most one block).
fn pbf() -> u32 { return select(lane_id() / LPB, 0u, BPW == 1u); }
// The lane's block and its segment in the block.
fn block_id() -> u32 { return gsg / NSEG; }
fn seg_id() -> u32 { return gsg % NSEG; }
// Word bases: the block's data, its candidate words and trace (2 words per position), the
// segment's words of `best` (its raw sequences and trailer), its payload ring in `scr`, and its
// series log in `seqs`.
fn dbase() -> u32 { return block_base(block_id()); }
fn cbase() -> u32 { return 2u * block_id() * BLOCK_SIZE; }
fn wbase() -> u32 { return SEG_WORDS * gsg; }
fn sbase() -> u32 { return gsg * (3u * RING_N); }
fn lbase() -> u32 { return block_id() * (3u * MAX_SEQS) + seg_id() * LOG_WORDS; }
// The segment's end (iend: positions [k * SEG, iend)).
fn seg_end() -> u32 { return (seg_id() + 1u) * SEG; }

// Records of the last get_all_matches: offBase | length << 17 (offBase <= 65538 < 2^17, length
// <= SEG < 2^15), strictly increasing length.
var<private> m_rec: array<u32, 5>;
var<private> m_n: u32;
const REC_SHIFT: u32 = 17u;
const REC_OB: u32 = (1u << REC_SHIFT) - 1u;
const_assert BLOCK_SIZE + 3u <= REC_OB;
const_assert SEG < (1u << (32u - REC_SHIFT));

// Segment state (opt::SegState) and output counters.
var<private> st_ip: u32;
var<private> st_anchor: u32;
var<private> st_rep: vec3<u32>;
var<private> n_seq: u32;
var<private> ml_sum: u32;
// Phase 1 series log: seqs[lbase() + 3*i ..] (free until main_fixup).
var<private> n_series: u32;
// Rep-length memo (M6 A2, a01 P1): the lane's last search position mem_p and, for each of its
// three rep probes, offset | length << 16 with the probe's exact length (its common prefix capped
// at that position's lim), or 0 when unknown. For the same offset, the capped common prefix at
// mem_p + d is exactly L - d whenever L >= d + 3 (lim falls by d too), so a later search reuses
// it instead of extending. The lane's search positions strictly increase.
var<private> mem_p: u32;
var<private> mem: vec3<u32>;

// load_u32_at without its alignment branch (M5 T3b: a per-lane branch diverges): both words
// are loaded, the high one masked when aligned. data[w + 1] is in bounds because every `data`
// binding ends with one zero word after the batch's last block (`compressor::data_bytes` =
// n * BLOCK_SIZE + 4, asserted by `OptBinds::check`; the pipeline's copy, direct-upload and
// zero-copy slots all keep and zero it).
fn ld32(base: u32, byte_off: u32) -> u32 {
    let w = base + (byte_off >> 2u);
    let sh = (byte_off & 3u) * 8u;
    let hi = data[w + 1u] << ((32u - sh) & 31u);
    return (data[w] >> sh) | select(hi, 0u, sh == 0u);
}

// match_len on ld32.
fn match_len_nb(base: u32, p: u32, q: u32, cap: u32) -> u32 {
    let max = min(BLOCK_SIZE - p, cap);
    var n = 0u;
    // Terminates: n rises by 4 to max.
    loop {
        if (n + 4u > max) { break; }
        let x = ld32(base, p + n) ^ ld32(base, q + n);
        if (x != 0u) { return n + (countTrailingZeros(x) >> 3u); }
        n += 4u;
    }
    // Terminates: n rises to max.
    loop {
        if (n >= max || load_byte(base, p + n) != load_byte(base, q + n)) { break; }
        n += 1u;
    }
    return n;
}

// opt::frac_weight
fn frac_weight(raw: u32) -> u32 {
    let stat = raw + 1u;
    let hb = firstLeadingBit(stat);
    return hb * 256u + ((stat << 8u) >> hb);
}

// Entry i of the lane's block's packed LL / OF table (see T_LLC).
fn tab(i: u32) -> i32 {
    return i32((p_tab[pbf() * TAB_W + (i >> 1u)] >> ((i & 1u) * 16u)) & 0xFFFFu);
}

// Branch-free: both tables read, the code index clamped (litlen <= 65536: code <= 35).
fn ll_price(litlen: u32) -> i32 {
    let a = tab(T_LLS + min(litlen, 63u));
    let b = tab(T_LLC + min(firstLeadingBit(litlen | 64u) + 19u, 35u));
    return select(b, a, litlen < 64u);
}

// ll_price(1) - ll_price(0) (re-read at use rather than kept live across the DP).
fn ll_inc1() -> i32 {
    return tab(T_LLS + 1u) - tab(T_LLS);
}

// Literal price of byte value `c`.
fn lit_cost(c: u32) -> i32 {
    return i32((p_lit[pbf() * 128u + (c >> 1u)] >> ((c & 1u) * 16u)) & 0xFFFFu);
}

fn of_price(ob: u32) -> i32 {
    return tab(T_OF + firstLeadingBit(ob));
}

fn ml_price(mlen: u32) -> i32 {
    return p_ml[pbf() * RING_N + mlen];
}

// seq::apply_off_base on a history value: the reps after offBase `ob` with `ll` literals.
// Branch-free: idx 0: r; 1: (r1, r0, r2); 2: (r2, r0, r1); 3: (r0 - 1, r0, r1);
// ob > 3: (ob - 3, r0, r1).
fn rep_after(r: vec3<u32>, ob: u32, ll: u32) -> vec3<u32> {
    let idx = ob - 1u + select(0u, 1u, ll == 0u);
    let rep = ob <= 3u;
    let keep = rep && idx == 0u;
    let first = select(select(select(r.x - 1u, r.z, idx == 2u), r.y, idx == 1u), ob - 3u, !rep);
    let third = select(select(r.y, r.z, rep && idx == 1u), r.z, keep);
    return vec3<u32>(select(first, r.x, keep), select(r.x, r.y, keep), third);
}

// opt::new_rep (ZSTD_newRep).
fn new_rep(r: vec3<u32>, ob: u32, ll0: bool) -> vec3<u32> {
    return rep_after(r, ob, select(1u, 0u, ll0));
}

// A DP node (opt::Node). ring_p = price; payload words (scr, six): rep0 | rep1 << 16,
// rep2 | litlen << 16, mlen | offBase << 8 (mlen <= SUFF, reps < 65536).
struct Node {
    price: i32,
    r: vec3<u32>,
    litlen: u32,
    mlen: u32,
    ob: u32,
}

fn slot(pos: u32) -> u32 { return pos % RING_N; }

// The payload words of the lane's slot s: scr[six(s)], scr[six(s) + SF], scr[six(s) + 2 SF].
const SF: u32 = 1u;
fn six(s: u32) -> u32 { return sbase() + 3u * s; }

fn ld(pos: u32) -> Node {
    let s = slot(pos);
    let j = six(s);
    let a = scr[j];
    let b = scr[j + SF];
    let c = scr[j + 2u * SF];
    return Node(ring_p[rix(s)], vec3<u32>(a & 0xFFFFu, a >> 16u, b & 0xFFFFu), b >> 16u, c & 0xFFu, c >> 8u);
}

fn ld_price(pos: u32) -> i32 {
    return ring_p[rix(slot(pos))];
}

fn st(pos: u32, n: Node) {
    let s = slot(pos);
    let j = six(s);
    ring_p[rix(s)] = n.price;
    scr[j] = n.r.x | (n.r.y << 16u);
    scr[j + SF] = n.r.z | (n.litlen << 16u);
    scr[j + 2u * SF] = n.mlen | (n.ob << 8u);
}

// A relaxation's match node at slot s: price, rep0 | rep1 << 16, rep2 (litlen 0), mlen | ob << 8.
fn st_match(s: u32, price: i32, ra: u32, rz: u32, mo: u32) {
    let j = six(s);
    ring_p[rix(s)] = price;
    scr[j] = ra;
    scr[j + SF] = rz;
    scr[j + 2u * SF] = mo;
}

// The relaxation's fill: price MAX, litlen 1, the rest left as it was.
fn fill(pos: u32) {
    let s = slot(pos);
    let j = six(s) + SF;
    ring_p[rix(s)] = MAXP;
    scr[j] = (scr[j] & 0xFFFFu) | (1u << 16u);
}

fn push_match(ob: u32, len: u32) {
    m_rec[m_n] = ob | (len << REC_SHIFT);
    m_n += 1u;
}

// The memo's exact length for offset ro at p (> mem_p), or 0 (see mem).
fn memo_len(p: u32, ro: u32) -> u32 {
    let d = p - mem_p;
    var r = 0u;
    for (var i = 0u; i < 3u; i += 1u) {
        let e = mem[i];
        let l = e >> 16u;
        r = select(r, l - d, (e & 0xFFFFu) == ro && l >= d + 3u);
    }
    return r;
}

// One rep probe of get_all_matches (probe j): offBase `ob`, source offset `ro` (valid: 1 <= ro
// <= p), `x` and `y` the 4 bytes at p and p - ro, `h` its memo length (0: unknown). Records it
// when its length (the common prefix within `lim`) is at least MIN_MATCH and beats `bestl`; true
// when the search ends there.
fn rep_probe(j: u32, p: u32, lim: u32, ob: u32, ro: u32, valid: bool, x: u32, y: u32, h: u32, bestl: ptr<function, u32>) -> bool {
    let d = x ^ y;
    if (!valid || (d & 0xFFFFFFu) != 0u) { return false; }
    // lim >= 8 at every searched position (p <= ilimit = iend - 8).
    var rl = h;
    if (h == 0u) {
        rl = 3u;
        if (d == 0u) { rl = 4u + match_len_nb(dbase(), p + 4u, p - ro + 4u, lim - 4u); }
    }
    mem[j] = ro | (rl << 16u);
    if (rl <= *bestl) { return false; }
    *bestl = rl;
    push_match(ob, rl);
    return rl > SUFF || rl == lim;
}

// opt::Dp::get_all_matches: records at p (<= ilimit) for reps `r` and `ll0`, lengths clamped
// to iend - p. `x` holds the 4 bytes at p and (w0, w1) the candidate words of p, loaded by the
// caller; the three rep sources are loaded together before any is compared.
fn get_all_matches(p: u32, r: vec3<u32>, ll0: bool, iend: u32, x: u32, w0: u32, w1: u32) {
    m_n = 0u;
    let lim = iend - p;
    var bestl = MIN_MATCH - 1u;
    // Rep indices ll0 .. ll0 + 2 (index 3 = rep0 - 1), offBase 1 .. 3.
    let ro0 = select(r.x, r.y, ll0);
    let ro1 = select(r.y, r.z, ll0);
    let ro2 = select(r.z, r.x - 1u, ll0);
    let v0 = ro0 - 1u < p;
    let v1 = ro1 - 1u < p;
    let v2 = ro2 - 1u < p;
    let y0 = ld32(dbase(), p - select(0u, ro0, v0));
    let y1 = ld32(dbase(), p - select(0u, ro1, v1));
    let y2 = ld32(dbase(), p - select(0u, ro2, v2));
    let h0 = select(0u, memo_len(p, ro0), v0);
    let h1 = select(0u, memo_len(p, ro1), v1);
    let h2 = select(0u, memo_len(p, ro2), v2);
    mem_p = p;
    mem = vec3<u32>(0u);
    if (rep_probe(0u, p, lim, 1u, ro0, v0, x, y0, h0, &bestl)) { return; }
    if (rep_probe(1u, p, lim, 2u, ro1, v1, x, y1, h1, &bestl)) { return; }
    if (rep_probe(2u, p, lim, 3u, ro2, v2, x, y2, h2, &bestl)) { return; }
    for (var j = 0u; j < 2u; j += 1u) {
        let off = select(w1 & 0xFFFFu, w0 & 0xFFFFu, j == 0u);
        let len = select(w0 >> 24u, (w0 >> 16u) & 0xFFu, j == 0u);
        if (len == 0u) { continue; }
        var l = min(len, lim);
        if (len == SEARCH_CAP) { l = match_len_nb(dbase(), p, p - off, lim); }
        if (l > bestl) {
            bestl = l;
            push_match(off + 3u, l);
            if (l == lim) { break; }
        }
    }
}

// == codes::ll_code / codes::ml_code.
fn ll_code(ll: u32) -> u32 {
    if (ll > 63u) { return firstLeadingBit(ll) + 19u; }
    return LL_CODE[ll];
}
fn ml_code(ml: u32) -> u32 {
    let b = ml - 3u;
    if (b > 127u) { return firstLeadingBit(b) + 36u; }
    return ML_CODE[b];
}

// HIST_OUT: counts the literal bytes at [a, e) of the lane's block.
fn hist_lits(a: u32, e: u32) {
    // Terminates: q rises to e.
    for (var q = a; q < e; q += 1u) {
        atomicAdd(&hist[pbf() * 256u + load_byte(dbase(), q)], 1u);
    }
}

// HIST_OUT: counts one raw sequence (its match at block position `m`) under the segment's own
// history: its literal bytes and its LL / ML / OF codes (hist_epilogue fixes the rest).
fn hist_seq(m: u32, ll: u32, ml: u32, ob: u32) {
    hist_lits(m - ll, m);
    atomicAdd(&hist[pbf() * 256u + H_LL + ll_code(ll)], CODE_ONE);
    atomicAdd(&hist[pbf() * 256u + H_ML + ml_code(ml)], CODE_ONE);
    atomicAdd(&hist[pbf() * 256u + H_OF + firstLeadingBit(ob)], CODE_ONE);
}

// HIST_OUT: moves one count of local block pbf()'s code from index i to j (hist_epilogue's fixes).
fn hist_move(i: u32, j: u32) {
    if (i != j) {
        atomicSub(&hist[pbf() * 256u + i], CODE_ONE);
        atomicAdd(&hist[pbf() * 256u + j], CODE_ONE);
    }
}

fn trace_put(p: u32, n: Node) {
    let t = cbase() + 2u * p;
    trace[t] = n.mlen | (n.litlen << 8u);
    trace[t + 1u] = n.ob;
}

// Phase 1 end of a series (opt::Dp::commit_ring's effect on the parse state): `last` ends the
// series started at `sip`, at series position `last_pos`. The stored sequences cover the literals
// before the series and every stretch up to the end of the last match, so the anchor moves to
// sip + last_pos - last.litlen and ip to sip + last_pos. A series with a match is logged for
// phase 2; its trace entries (positions sip ..= sip + last_pos - 1) are never written again.
fn end_series(last: Node, sip: u32, last_pos: u32) {
    st_ip = sip + last_pos;
    if (last.mlen == 0u) { return; }
    st_rep = last.r;
    st_anchor = st_ip - last.litlen;
    let l = lbase() + 3u * n_series;
    seqs[l] = sip | (last.mlen << 16u);
    seqs[l + 1u] = last_pos | (last.litlen << 16u);
    seqs[l + 2u] = last.ob;
    n_series += 1u;
}

// Phase 2 (opt::Dp::commit_ring's output): the backward traces of the logged series, last series
// first, each appending its sequences last first to the segment's words of `best`, so the
// segment's raw sequences end up in reverse order (RAW_REVERSED).
//
// One flat loop, one sequence per iteration (M6 A2, a01 P5): a series' header is read when the
// previous series is done, so the warp runs the maximum over its lanes of n_seq iterations, not
// the sum over series of the longest series' trace. Reads and writes keep the order of the
// series-by-series form (which the HIST_OUT in-place log relies on, see LOG_SEQS).
//
// The DP's offBases are stored as they are: each is seq::off_base_for(offset, lit_len) under the
// DP's own history, as main_fixup needs. A rep record (index i, zstd's ll0 numbering) is only
// kept when no earlier rep index holds the same offset (that one gives the same length first),
// which is off_base_for's order; an explicit record whose offset equals a probed rep has that
// rep's length (candidate lengths are true capped lengths, extended at the cap, both clamped to
// iend) and so is never recorded after it; the last rep index needs rep0 > 1 in both. So the
// DP's history is the canonical one and the trailer's reps are st_rep.
//
// HIST_OUT: the sequences go to the segment's series-log words instead (see LOG_SEQS), and each
// is counted in the pass's histograms with its literal bytes (hist_seq).
fn emit() {
    var i = n_series;
    var have = false;
    var sip = 0u;
    var sp = 0u;
    var mlen = 0u;
    var ob = 0u;
    // Terminates: every iteration emits one sequence or takes the next series (i falls to 0);
    // inside a series sp falls by nl + nm >= 3 per sequence (a node ending a match has mlen >= 3),
    // and the series start (sp = 0) has nm = 0.
    loop {
        if (!have) {
            if (i == 0u) { break; }
            i -= 1u;
            let l = lbase() + 3u * i;
            let a = seqs[l];
            let c = seqs[l + 1u];
            sip = a & 0xFFFFu;
            mlen = a >> 16u;
            sp = (c & 0xFFFFu) - mlen - (c >> 16u);
            ob = seqs[l + 2u];
            have = true;
        }
        let t = cbase() + 2u * (sip + sp);
        let t0 = trace[t];
        let nm = t0 & 0xFFu;
        let nl = t0 >> 8u;
        if (HIST_OUT) {
            let o = lbase() + 3u * (LOG_SEQS - 1u - n_seq);
            seqs[o] = nl;
            seqs[o + 1u] = mlen;
            seqs[o + 2u] = ob;
            hist_seq(sip + sp, nl, mlen, ob);
        } else {
            let o = wbase() + 3u * n_seq;
            best[o] = nl;
            best[o + 1u] = mlen;
            best[o + 2u] = ob;
        }
        n_seq += 1u;
        ml_sum += mlen;
        if (nm == 0u || sp < nl + nm) {
            have = false;
        } else {
            mlen = nm;
            ob = trace[t + 1u];
            sp -= nl + nm;
        }
    }
}

// Literal price from the frequency `f` and the base weight `lb` (opt::Prices::from_freqs).
fn lit_from(f: u32, lb: u32) -> i32 {
    return i32(lb - min(frac_weight(f), lb - 256u));
}

// opt::Prices::from_hist's frequency of a count.
fn seen(c: u32) -> u32 {
    return c + select(0u, 1u, c > 0u);
}

// The block's literal byte counts into hist (LPB lanes striding over its words).
fn count_block(blk: u32, kl: u32, b: u32) {
    let w0 = block_base(b);
    for (var w = kl; w < BLOCK_SIZE / 4u; w += LPB) {
        let v = data[w0 + w];
        atomicAdd(&hist[blk * 256u + (v & 0xFFu)], 1u);
        atomicAdd(&hist[blk * 256u + ((v >> 8u) & 0xFFu)], 1u);
        atomicAdd(&hist[blk * 256u + ((v >> 16u) & 0xFFu)], 1u);
        atomicAdd(&hist[blk * 256u + (v >> 24u)], 1u);
    }
}

// PRICE_MODE 2 (opt::cover_literals): the bytes of lane kl's chunk of the block that no
// candidate covers into hist, and their number into hsum[blk * 5 + 4]. Position p is covered
// when some p' <= p has p' + lenB(p') > p; lenB <= SEARCH_CAP, so the chunk's running maximum
// starts from the SEARCH_CAP positions before it.
fn cover_chunk(blk: u32, kl: u32, b: u32) {
    const C: u32 = BLOCK_SIZE / LPB;
    let cb = 2u * b * BLOCK_SIZE;
    let bb = block_base(b);
    let p0 = kl * C;
    var reach = 0u;
    // Terminates: p rises to p0.
    for (var p = p0 - min(p0, SEARCH_CAP); p < p0; p += 1u) {
        reach = max(reach, p + (best[cb + 2u * p] >> 24u));
    }
    var n = 0u;
    // Terminates: p rises to p0 + C.
    for (var p = p0; p < p0 + C; p += 1u) {
        reach = max(reach, p + (best[cb + 2u * p] >> 24u));
        if (reach <= p) {
            atomicAdd(&hist[blk * 256u + load_byte(bb, p)], 1u);
            n += 1u;
        }
    }
    atomicAdd(&hsum[blk * 5u + 4u], n);
}

// The LL / ML / OF prices under PRICE_MODE, each the value the i32 tables held before M6 A1 (in
// 0..65536): `q` the block's words in `prices` (PRICE_MODE 1 / 3) and `bases` the LL / ML / OF
// base weights (PRICE_MODE 3: frac_weight of the table sums).
// Entry i < TAB_N of the packed LL / OF table (see T_LLC).
fn tab_entry(i: u32, q: u32, bases: vec3<u32>) -> u32 {
    if (i < T_OF) {
        // LL: by code (i < T_LLS), or by litlen i - T_LLS < 64.
        let c = select(LL_CODE[min(i - min(i, T_LLS), 63u)], i, i < T_LLS);
        if (PRICE_MODE == 0u) { return u32(BI_LL[c]); }
        if (PRICE_MODE == 2u) { return u32(PR_LL[c]); }
        if (PRICE_MODE == 1u) { return prices[q + 256u + c]; }
        return LL_BITS[c] * 256u + bases.x - frac_weight(seen(prices[q + 256u + c]));
    }
    let c = i - T_OF;
    if (PRICE_MODE == 0u) { return u32(BI_OF[c]); }
    if (PRICE_MODE == 2u) { return u32(PR_OF[c]); }
    if (PRICE_MODE == 1u) { return prices[q + 345u + c]; }
    return c * 256u + bases.z - frac_weight(seen(prices[q + 345u + c]));
}

// The ML price of match length m <= SUFF (lengths below MIN_MATCH take code 0's, never read).
fn ml_entry(m: u32, q: u32, bases: vec3<u32>) -> u32 {
    let c = max(m, 3u) - 3u;
    if (PRICE_MODE == 0u) { return u32(BI_ML[c]); }
    if (PRICE_MODE == 2u) { return u32(PR_ML[c]); }
    if (PRICE_MODE == 1u) { return prices[q + 292u + c]; }
    return ML_BITS[c] * 256u + bases.y - frac_weight(seen(prices[q + 292u + c]));
}

// Local block blk's LL / OF (packed) and ML tables, written in place by the block's lanes (kl
// striding). Nothing is staged, so the private-ring fallback needs no workgroup scratch.
fn build_tables(blk: u32, kl: u32, q: u32, bases: vec3<u32>) {
    for (var w = kl; w < TAB_W; w += LPB) {
        let lo = tab_entry(2u * w, q, bases);
        let hi = tab_entry(2u * w + 1u, q, bases);
        p_tab[blk * TAB_W + w] = (lo & 0xFFFFu) | (hi << 16u);
    }
    for (var m = kl; m < RING_N; m += LPB) {
        p_ml[blk * RING_N + m] = i32(ml_entry(m, q, bases));
    }
}

// The workgroup's price tables (see the header). Every lane calls it (it has barriers).
fn prologue(valid: bool, b: u32) {
    let lane = lane_id();
    let blk = lane / LPB;
    let kl = lane % LPB;
    if (PRICE_MODE == 0u || PRICE_MODE == 2u) {
        for (var i = lane; i < 256u * BPW; i += WG) { atomicStore(&hist[i], 0u); }
        for (var i = lane; i < 5u * BPW; i += WG) { atomicStore(&hsum[i], 0u); }
        workgroupBarrier();
        if (valid) {
            if (PRICE_MODE == 0u) { count_block(blk, kl, b); } else { cover_chunk(blk, kl, b); }
        }
        workgroupBarrier();
        // Every byte covered: the whole block's histogram.
        if (PRICE_MODE == 2u && valid && atomicLoad(&hsum[blk * 5u + 4u]) == 0u) {
            count_block(blk, kl, b);
        }
        workgroupBarrier();
        for (var e = kl; e < 256u; e += LPB) {
            let c = atomicLoad(&hist[blk * 256u + e]);
            var f = seen(c);
            if (PRICE_MODE == 0u) { f = select(0u, 1u, c > 0u) + (c >> 8u); }
            atomicStore(&hist[blk * 256u + e], f);
            atomicAdd(&hsum[blk * 5u], f);
        }
        workgroupBarrier();
        let lb = frac_weight(max(atomicLoad(&hsum[blk * 5u]), 1u));
        for (var e = kl; e < 128u; e += LPB) {
            let lo = lit_from(atomicLoad(&hist[blk * 256u + 2u * e]), lb);
            let hi = lit_from(atomicLoad(&hist[blk * 256u + 2u * e + 1u]), lb);
            p_lit[blk * 128u + e] = u32(lo) | (u32(hi) << 16u);
        }
        build_tables(blk, kl, 0u, vec3<u32>(0u));
    } else if (PRICE_MODE == 1u) {
        if (valid) {
            let q = b * PRICE_WORDS;
            for (var e = kl; e < 128u; e += LPB) { p_lit[blk * 128u + e] = prices[q + 2u * e] | (prices[q + 2u * e + 1u] << 16u); }
            build_tables(blk, kl, q, vec3<u32>(0u));
        }
    } else {
        // PRICE_MODE 3: opt::Prices::from_hist of the block's Hist (lit, ll, ml, of).
        for (var i = lane; i < 5u * BPW; i += WG) { atomicStore(&hsum[i], 0u); }
        workgroupBarrier();
        let q = b * HIST_WORDS;
        if (valid) {
            var sl = 0u;
            var sll = 0u;
            var sml = 0u;
            var sof = 0u;
            for (var e = kl; e < 256u; e += LPB) { sl += seen(prices[q + e]); }
            for (var c = kl; c < 36u; c += LPB) { sll += seen(prices[q + 256u + c]); }
            for (var c = kl; c < 53u; c += LPB) { sml += seen(prices[q + 292u + c]); }
            for (var c = kl; c < 32u; c += LPB) { sof += seen(prices[q + 345u + c]); }
            atomicAdd(&hsum[blk * 5u], sl);
            atomicAdd(&hsum[blk * 5u + 1u], sll);
            atomicAdd(&hsum[blk * 5u + 2u], sml);
            atomicAdd(&hsum[blk * 5u + 3u], sof);
        }
        workgroupBarrier();
        if (valid) {
            let lb = frac_weight(max(atomicLoad(&hsum[blk * 5u]), 1u));
            let llb = frac_weight(max(atomicLoad(&hsum[blk * 5u + 1u]), 1u));
            let mlb = frac_weight(max(atomicLoad(&hsum[blk * 5u + 2u]), 1u));
            let ofb = frac_weight(max(atomicLoad(&hsum[blk * 5u + 3u]), 1u));
            for (var e = kl; e < 128u; e += LPB) {
                let lo = lit_from(seen(prices[q + 2u * e]), lb);
                let hi = lit_from(seen(prices[q + 2u * e + 1u]), lb);
                p_lit[blk * 128u + e] = u32(lo) | (u32(hi) << 16u);
            }
            build_tables(blk, kl, q, vec3<u32>(llb, mlb, ofb));
        }
    }
    workgroupBarrier();
    if (HIST_OUT) {
        for (var i = lane; i < 256u * BPW; i += WG) { atomicStore(&hist[i], 0u); }
        workgroupBarrier();
    }
}

// HIST_OUT: the pass's Hist of block b (lane of segment k), after every lane's phase 2. Each lane
// has counted its own raw sequences and literal bytes (in `hist`, packed, see LIT_BITS) and left
// its segment summary in its first trace words (SUM_WORDS); the segment-0 lane then applies main_fixup's
// changes (the literal carry of each segment's first sequence, and the off_bases it re-encodes
// against the true reps, walking the segments in order exactly as main_fixup does), and the
// block's lanes store the Hist at prices[b * HIST_WORDS]. Every lane calls it (barriers).
fn hist_epilogue(valid: bool, b: u32, k: u32) {
    storageBarrier();
    workgroupBarrier();
    if (valid && k == 0u) {
        var reps = vec3<u32>(1u, 4u, 8u);
        var prev_end = 0u;
        for (var kk = 0u; kk < NSEG; kk += 1u) {
            let sm = summ_base(b, kk);
            let n = trace[sm];
            let carry = kk * SEG - prev_end;
            if (n > 0u) { prev_end = trace[sm + 1u]; }
            let src = b * (3u * MAX_SEQS) + kk * LOG_WORDS + 3u * (LOG_SEQS - n);
            if (n > 0u && carry > 0u) {
                let own = seqs[src];
                hist_move(H_LL + ll_code(own), H_LL + ll_code(own + carry));
            }
            var spec = select(vec3<u32>(0u), vec3<u32>(1u, 4u, 8u), kk == 0u);
            var i = 0u;
            // Terminates: i rises to n, bounded by LOG_SEQS (a segment logs at most SEG / 3 <=
            // LOG_SEQS sequences, so the bound only guards against a corrupt summary word).
            loop {
                if (i >= min(n, LOG_SEQS) || all(reps == spec)) { break; }
                let r = src + 3u * i;
                let ll = seqs[r] + select(0u, carry, i == 0u);
                let spec_ob = seqs[r + 2u];
                let offset = offset_of(spec, spec_ob, ll);
                spec = applied(spec, spec_ob, ll);
                let ob = ob_for(reps, offset, ll);
                reps = applied(reps, ob, ll);
                hist_move(H_OF + firstLeadingBit(spec_ob), H_OF + firstLeadingBit(ob));
                i += 1u;
            }
            if (i < n) { reps = vec3<u32>(trace[sm + 2u], trace[sm + 3u], trace[sm + 4u]); }
        }
    }
    workgroupBarrier();
    if (valid) {
        let q = b * HIST_WORDS;
        for (var e = k; e < HIST_WORDS; e += NSEG) {
            if (e < 256u) {
                prices[q + e] = atomicLoad(&hist[pbf() * 256u + e]) & LIT_MASK;
            } else {
                prices[q + e] = atomicLoad(&hist[pbf() * 256u + e - 256u]) >> LIT_BITS;
            }
        }
    }
}

@compute @workgroup_size(WG)
fn main_opt(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    gsg = wid.x * WG + lid;
    run_block();
}

// The lane's block (gsg): prologue, DP and (HIST_OUT) epilogue. Every lane of the workgroup calls
// it (barriers).
fn run_block() {
    prologue(in_batch(), block_id());
    if (in_batch()) { dp(); }
    // The epilogue's arguments are recomputed from gsg, not kept live across the DP (M6 A1).
    if (HIST_OUT) { hist_epilogue(in_batch(), block_id(), seg_id()); }
}

// The persistent entry point (see the header; WG == NSEG, one block per workgroup and item).
// The item is broadcast through workgroup memory: workgroupUniformLoad's barrier also keeps a
// lane from starting the next block's prologue (which rewrites the workgroup's tables) while
// another lane still runs this block's DP.
var<workgroup> next_item: u32;
@compute @workgroup_size(WG)
fn main_opt_persist(@builtin(local_invocation_index) lid: u32) {
    let n = arrayLength(&counts) / 2u;
    // Terminates: every trip takes a new counter value, and the trip with one >= n ends the loop.
    loop {
        if (lid == 0u) { next_item = atomicAdd(&sched[0], 1u); }
        let item = workgroupUniformLoad(&next_item);
        if (item >= n) { break; }
        let b = atomicLoad(&sched[SCHED_HDR + n + item]);
        // b < n < 2^30, so b >> 30 is 0. It keeps the lane opaque to the compiler, so the lane
        // constants are recomputed from gsg in each trip instead of being hoisted out of the
        // loop and held in registers (a07: 92 -> 138 registers without it).
        gsg = b * WG + lid + (b >> 30u);
        run_block();
    }
}
const_assert !PERSIST || WG == NSEG;

// Whether the lane's block is in the batch.
fn in_batch() -> bool {
    return block_id() < arrayLength(&counts) / 2u;
}

// The DP pass of the lane's segment (phases 1 and 2).
fn dp() {
    let s = seg_id() * SEG;
    // iend = seg_end(), ilimit = seg_end() - 8 (recomputed at use, M6 A1).
    st_anchor = s;
    if (s == 0u) {
        st_ip = 1u;
        st_rep = vec3<u32>(1u, 4u, 8u);
    } else {
        st_ip = s;
        st_rep = vec3<u32>(0u);
    }
    n_seq = 0u;
    ml_sum = 0u;
    n_series = 0u;
    mem_p = 0u;
    mem = vec3<u32>(0u);

    var in_series = false;
    var sip = 0u;       // the series' ip
    var cur = 0u;
    var last_pos = 0u;
    // Terminates: every trip either advances st_ip (outside a series: by 1, or a commit moves it
    // past the series start) or cur (inside one, cur <= last_pos <= iend - sip, then a commit).
    loop {
        if (!in_series) {
            // Dead positions (M6 A3, a09; see DEAD_BIT): a series start probe there finds
            // nothing and only moves st_ip on by 1, so a dead stretch is skipped here, without
            // a trip per position. Terminates: st_ip rises by 1 per step.
            loop {
                if (st_ip >= seg_end() - 8u) { break; }
                if ((best[cbase() + 2u * st_ip + 1u] & DEAD_BIT) == 0u) { break; }
                st_ip += 1u;
            }
        }
        if (!in_series && st_ip >= seg_end() - 8u) { break; }
        // The trip's position and its loads, issued before anything depends on them: the 4 bytes
        // at p (x), the byte before it, and p's candidate words (p is clamped to the segment: a
        // series may reach iend, where nothing is searched and x is unused).
        // (vars: a FOLD trip moves on to the next series position.)
        var p = select(st_ip, sip + cur, in_series);
        var pc = min(p, seg_end() - 1u);
        var x = ld32(dbase(), pc);
        var xprev = load_byte(dbase(), p - 1u);
        var w0 = best[cbase() + 2u * pc];
        var w1 = best[cbase() + 2u * pc + 1u];

        // Part 1: what to search, if anything.
        var grep = vec3<u32>(0u);
        var gll0 = false;
        var search = false;
        var n = Node(0, vec3<u32>(0u), 0u, 0u, 0u);
        var advance = false;
        var finish = false;
        if (!in_series) {
            grep = st_rep;
            gll0 = st_ip == st_anchor;
            search = true;
        } else {
            // FOLD (M6 A3, a09): a position that only needs its literal extension (advance) is
            // followed by the next one in the same trip, with the same statements, while cur <
            // last_pos (so the next trip would have been that position, in the series).
            // Terminates: cur rises by 1 per fold, to last_pos, which only a LEVEL >= 1 pass
            // raises here (FOLD is LEVEL 0 only).
            loop {
                let inr = p;
                let prev = ld(cur - 1u);
                let litlen = prev.litlen + 1u;
                let price = prev.price + lit_cost(xprev) + (ll_price(litlen) - ll_price(litlen - 1u));
                n = ld(cur);
                if (price <= n.price) {
                    let pm = n;
                    n = prev;
                    n.litlen = litlen;
                    n.price = price;
                    st(cur, n);
                    if (LEVEL >= 1u && pm.litlen == 0u && ll_inc1() < 0 && inr < seg_end()) {
                        let next_lit = lit_cost(x & 0xFFu);
                        let with1 = pm.price + next_lit + ll_inc1();
                        let with_more = price + next_lit + (ll_price(litlen + 1u) - ll_price(litlen));
                        var next_price = MAXP;
                        if (cur < last_pos) { next_price = ld_price(cur + 1u); }
                        if (with1 < with_more && with1 < next_price) {
                            var q = pm;
                            q.litlen = 1u;
                            q.price = with1;
                            st(cur + 1u, q);
                            last_pos = max(last_pos, cur + 1u);
                        }
                    }
                }
                if (inr < seg_end()) { trace_put(inr, n); }
                if (inr > seg_end() - 8u) {
                    advance = true;
                } else if (cur == last_pos) {
                    finish = true;
                } else if (LEVEL == 0u && ld_price(cur + 1u) <= n.price + 128) {
                    advance = true;
                } else if ((w1 & DEAD_BIT) != 0u) {
                    // A dead position (inr <= ilimit, so w1 is inr's): get_all_matches would
                    // find nothing, and the trip would only advance.
                    advance = true;
                } else {
                    grep = n.r;
                    gll0 = n.litlen == 0u;
                    search = true;
                }
                if (FOLD && advance && cur < last_pos) {
                    advance = false;
                    cur += 1u;
                    p = sip + cur;
                    pc = min(p, seg_end() - 1u);
                    x = ld32(dbase(), pc);
                    xprev = load_byte(dbase(), p - 1u);
                    w0 = best[cbase() + 2u * pc];
                    w1 = best[cbase() + 2u * pc + 1u];
                    continue;
                }
                break;
            }
        }

        // The lanes' match searches, together.
        if (search) {
            get_all_matches(p, grep, gll0, seg_end(), x, w0, w1);
        }

        // Part 2 (no `continue`: every lane reaches the end of the trip, so the lanes stay
        // together for the next trip's get_all_matches). Seeding and relaxation are one loop (M5
        // T3b): a series start is a relaxation from a virtual node n0 at cur 0 with last_pos 0.
        // - The records' target sets are disjoint (record i covers mlen in (len_{i-1}, len_i]),
        //   so every target is compared against the ring as it was before this step, whatever the
        //   order: target pos improves iff pos > lp0 || price < its ring price (a target above
        //   lp0 is MAXP from the oracle's fill, or fresh). The optLevel-0 early abort stays per
        //   record (lengths descending, nothing below the record's first non-improving length).
        // - The oracle's fill of the slots in (lp0, cur + longest] leaves, besides the targets
        //   (all written), only cur + 1 and cur + 2; a fresh series' slots 1 and 2 get litlen 1
        //   (the oracle: litlen + 1, + 2), which nothing reads: price MAXP always loses to the
        //   visit's literal extension, which overwrites the node, and `pm.litlen != 0` holds.
        if (search) {
            var src = n;
            var c0 = cur;
            var lp0 = last_pos;
            if (!in_series) {
                let litlen = st_ip - st_anchor;
                src = Node(ll_price(litlen), st_rep, litlen, 0u, 0u);
                c0 = 0u;
                lp0 = 0u;
            }
            if (m_n == 0u) {
                if (in_series) { advance = true; } else { st_ip += 1u; }
            } else {
                let max_ob = m_rec[m_n - 1u] & REC_OB;
                let longest = m_rec[m_n - 1u] >> REC_SHIFT;
                if (!in_series) {
                    st(0u, src);
                    trace_put(st_ip, src);
                }
                if (longest > SUFF || (in_series && (cur + longest >= OPT_NUM || p + longest >= seg_end()))) {
                    // large match -> immediate encoding
                    last_pos = c0 + longest;
                    end_series(Node(0, new_rep(src.r, max_ob, gll0), 0u, longest, max_ob), select(st_ip, sip, in_series), last_pos);
                    in_series = false;
                } else {
                    // Records last first, each record's lengths descending, 4 per step (a
                    // record's lengths start at len_i >= lo, since lengths strictly increase).
                    // Terminates: mi falls to 0; mlen falls by 4 per step, staying >= lo >= 3.
                    let base_price = src.price + tab(T_LLS);
                    for (var mi = m_n; mi > 0u; mi -= 1u) {
                        let rec = m_rec[mi - 1u];
                        let ob = rec & REC_OB;
                        let mp = base_price + of_price(ob) + MATCH_FEE;
                        let mrep = new_rep(src.r, ob, gll0);
                        let ra = mrep.x | (mrep.y << 16u);
                        let obs = ob << 8u;
                        let lo = select(MIN_MATCH, (m_rec[max(mi, 2u) - 2u] >> REC_SHIFT) + 1u, mi > 1u);
                        var mlen = rec >> REC_SHIFT;
                        loop {
                            let v1 = mlen - 1u >= lo;
                            let v2 = mlen - 2u >= lo;
                            let v3 = mlen - 3u >= lo;
                            let s0 = slot(c0 + mlen);
                            let s1 = slot(c0 + mlen - 1u);
                            let s2 = slot(c0 + mlen - 2u);
                            let s3 = slot(c0 + mlen - 3u);
                            let pr0 = mp + ml_price(mlen);
                            let pr1 = mp + ml_price(mlen - 1u);
                            let pr2 = mp + ml_price(mlen - 2u);
                            let pr3 = mp + ml_price(mlen - 3u);
                            let k0 = c0 + mlen > lp0 || pr0 < ring_p[rix(s0)];
                            var k1 = v1 && (c0 + mlen - 1u > lp0 || pr1 < ring_p[rix(s1)]);
                            var k2 = v2 && (c0 + mlen - 2u > lp0 || pr2 < ring_p[rix(s2)]);
                            var k3 = v3 && (c0 + mlen - 3u > lp0 || pr3 < ring_p[rix(s3)]);
                            if (LEVEL == 0u) {
                                // early update abort: nothing below the record's first failure
                                k1 = k1 && k0;
                                k2 = k2 && k1;
                                k3 = k3 && k2;
                            }
                            // (!v3 implies mlen < lo + 3.)
                            let stop = mlen < lo + 4u || (LEVEL == 0u && !k3);
                            if (k0) { st_match(s0, pr0, ra, mrep.z, mlen | obs); }
                            if (k1) { st_match(s1, pr1, ra, mrep.z, (mlen - 1u) | obs); }
                            if (k2) { st_match(s2, pr2, ra, mrep.z, (mlen - 2u) | obs); }
                            if (k3) { st_match(s3, pr3, ra, mrep.z, (mlen - 3u) | obs); }
                            if (stop) { break; }
                            mlen -= 4u;
                        }
                    }
                    // The fill's slots no record reaches (only c0 + 1, c0 + 2).
                    if (c0 + longest > lp0) {
                        // Terminates: q rises to c0 + MIN_MATCH.
                        for (var q = lp0 + 1u; q < c0 + MIN_MATCH; q += 1u) { fill(q); }
                        last_pos = c0 + longest;
                    }
                    if (in_series) {
                        advance = true;
                    } else {
                        sip = st_ip;
                        cur = 1u;
                        in_series = true;
                    }
                }
            }
        }
        if (advance) {
            cur += 1u;
            finish = cur > last_pos;
        }
        if (finish) {
            end_series(ld(last_pos), sip, last_pos);
            in_series = false;
        }
    }
    emit();
    if (HIST_OUT) {
        // The literals after the segment's last match; the summary for hist_epilogue. The
        // candidate words stay (no trailer in `best`).
        hist_lits(st_anchor, seg_end());
        let sm = summ_base(block_id(), seg_id());
        trace[sm] = n_seq;
        trace[sm + 1u] = st_anchor;
        trace[sm + 2u] = st_rep.x;
        trace[sm + 3u] = st_rep.y;
        trace[sm + 4u] = st_rep.z;
        return;
    }
    best[wbase() + SEG_META] = n_seq;
    best[wbase() + SEG_META + 1u] = st_anchor;
    best[wbase() + SEG_META + 2u] = ml_sum;
    best[wbase() + SEG_META + 3u] = st_rep.x;
    best[wbase() + SEG_META + 4u] = st_rep.y;
    best[wbase() + SEG_META + 5u] = st_rep.z;
}
