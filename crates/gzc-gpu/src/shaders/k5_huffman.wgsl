// K5: the literals section. One workgroup of 256 threads per block turns the block's literals
// into its literals section, byte-identical to gzc_core::huffman::write_literals_section (HUFFMAN)
// or to a Raw section (!HUFFMAN):
//   RLE (>= 2 literals, all equal), else Compressed (>= 64 literals, a describable Huffman table,
//   and strictly smaller than Raw), else Raw.
// K5 writes the section straight into the block's frame at byte SEC (right after the frame
// header and the 3-byte block header) and stores the section's length in frame_len[b]. K4 then
// reads frame_len[b], writes the sequences section after the literals section and patches the
// frame / block headers into the bytes below SEC (whatever K5 leaves there).
//
// Literals are gathered from the block (K3 only writes seqs and counts): literal
// run j < n_seq is data[anchor_j .. anchor_j + ll_j), anchor_j = sum over i < j of ll_i + ml_i,
// and run n_seq is data[anchor_n_seq .. BLOCK_SIZE). A workgroup prefix sum over G = ceil(n_seq /
// 256) sequences per thread leaves in ix_lit / ix_src the literal index and block byte where each
// thread's group of runs starts (2 KiB of workgroup memory whatever n_seq is). A thread seeks
// literal i by a binary search over ix_lit and a walk over at most G runs, then streams bytes
// forward (next_byte) or backward (prev_byte) run by run. Every literal pass covers a contiguous
// range of literal indices per thread, so each thread seeks once per pass.
//
// Threads cooperate on the histogram, the (count, symbol) rank sort, the stream encode and the Raw
// copy; thread 0 builds the table (two-queue Huffman merge, HUF_setMaxHeight, canonical codes),
// writes the weights description (direct, or FSE with two interleaved states) into workgroup
// memory and lays out the section.
//
// Stream encode: 1 stream (< 256 literals, 256 threads) or 4 (64 threads each). Within a stream
// the literals are split into contiguous chunks; symbols are coded last -> first, so the thread
// holding the last chunk writes first. A segmented prefix sum of the chunks' bit counts (taken in
// that reversed order) gives each thread its bit offset; each thread then packs its codes into
// words, atomicOr-ing the words it may share with a neighbour (its first and its last) and
// storing the words only it covers. Thread 0 adds the header, the jump table and the end marks.
//
// The host binds exactly n_blocks words of `frame_len`, so arrayLength(&frame_len) is the batch
// size. Prepended by the host: HDR_LEN, FRAME_WORDS, MAX_SEQS, HUFFMAN (and K4's other constants).
//
// Built without naga's loop bounding and index clamps (GpuContext::shader_trusted), so the
// kernel must terminate and stay in bounds for every input it can be given. It relies on K3's
// output, which a harness that scripts a parse checks first (testing::frames_from_parses):
// - n_seq <= MAX_SEQS, and the sequences' lit_len + match_len sum to at most BLOCK_SIZE;
// - counts holds the literal count n those sequences leave, so n <= BLOCK_SIZE.
// A loop or index added here needs its own argument.
//
// Terminates:
// - seek: the binary search shrinks [lo, hi], then the walk raises cj to n_seq. next_byte
//   raises cj to n_seq and prev_byte lowers it to 0.
// - build_index and the stream scan double d up to WG.
// - set_max_height: `while (tp[n] > tgt)` and the `while (tp[n] == tgt)` after it stop at
//   n >= 0, because tp[0], the most frequent symbol's depth, is at most 8: a full binary tree of
//   at most 256 leaves has a leaf that shallow, and depths do not fall as counts fall.
//   `while (total_cost > 0)` lowers total_cost by 1 << (nb_dec - 1) >= 1 per pass, and
//   `while (total_cost < 0)` raises it by 1. The nb_dec loops move nb_dec by 1 inside
//   1..=HUF_MAX_BITS + 2.
// - normalize and ncount: as fse_normalize and ncount in k4_seq_entropy.wgsl.
// - dput removes 8 bits per pass. describe's `while (i > 0u)` lowers i by 2 from an even start.
// - Every other loop is counted, to a constant, to np <= 256, or to a literal or word count.
//
// Bounds, workgroup and function arrays:
// - hist, ctab, leaves [256]: a byte value, or a rank below the number of present symbols.
// - tw, tp [512]: a node id below 2 * np - 1 <= 511.
// - set_max_height indexes tp and rank_last [14] exactly as gzc_core::huffman::set_max_height
//   does, statement for statement. nb_dec starts at most 8 (total_cost is below the number of
//   clamped leaves) and ends at most HUF_MAX_BITS + 2 = 13. The remaining indices rest on the
//   invariants of zstd's HUF_setMaxHeight, which both ports share: a symbol shorter than
//   HUF_MAX_BITS exists while the cost is not zero, so rank_last[nb_dec] names a symbol when a
//   code is lengthened, and n stays >= 0. The Rust port asserts the first and is bounds-checked,
//   so an input that broke them would fail on the oracle.
// - nb_per_rank, val_per_rank, wc, wnorm, wdfs, wdnb [12]: a code length or weight <= 11
//   (set_max_height caps the lengths at HUF_MAX_BITS).
// - wcumul [13]: a weight + 1. wst, wspread [64]: a cell of the weights' table, log <= 6.
// - desc [40]: dbyte checks DESC_BYTES; the copy out reads bytes below L_DESC <= 128.
// - ix_lit, ix_src [257]: lid + 1, or a search index <= WG.
// - scan [WG]: lid, or lid - d inside the thread's own stream. lay [16]: constants <= 13.
//
// Bounds, storage buffers:
// - data: the cursor stays on literal bytes, which lie inside the block (the facts above), and
//   next_word / prev_word load 4 bytes only inside one run.
// - frames: every section K5 writes is at most a Raw section, SEC + 3 + n bytes, which fits
//   FRAME_WORDS.
// - seqs, counts and frame_len are indexed inside block b's own words, and b is checked
//   against the batch size.

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> seqs: array<u32>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> frames: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> frame_len: array<u32>;

const WG: u32 = 256u;
const SEC: u32 = HDR_LEN + 3u; // frame byte of the literals section
const HUF_MAX_BITS: u32 = 11u;
const MIN_HUF_LITERALS: u32 = 64u;
const WEIGHTS_MAX_LOG: u32 = 6u;
const NO_SYMBOL: u32 = 0xF0F0F0F0u;
const DESC_BYTES: u32 = 160u;

// Decisions broadcast by thread 0.
const GO: u32 = 0u;
const RLE: u32 = 1u;
const RAW: u32 = 2u;

var<workgroup> hist: array<atomic<u32>, 256>;
var<workgroup> n_present: atomic<u32>;
// Present symbols in (count, symbol) ascending order.
var<workgroup> leaves: array<u32, 256>;
// Huffman tree (thread 0): node weights, then depths.
var<workgroup> tw: array<u32, 512>;
// Huffman tree (thread 0): parents, then the code lengths in count-descending order.
var<workgroup> tp: array<u32, 512>;
// Per symbol: code | nb_bits << 16 (0 for absent symbols).
var<workgroup> ctab: array<u32, 256>;
var<workgroup> scan: array<u32, WG>;
// Table description bytes (at most 128 are ever used; writes past DESC_BYTES are dropped).
var<workgroup> desc: array<u32, 40>;
// FSE table for the weights (log <= 6, 12 symbols).
var<workgroup> wnorm: array<i32, 12>;
var<workgroup> wcumul: array<u32, 13>;
var<workgroup> wst: array<u32, 64>;
var<workgroup> wspread: array<u32, 64>;
var<workgroup> wdfs: array<i32, 12>;
var<workgroup> wdnb: array<u32, 12>;
var<workgroup> flag: u32;
// Section layout (thread 0): see L_* below.
var<workgroup> lay: array<u32, 16>;
const L_HDR: u32 = 0u;        // section header bytes
const L_DESC: u32 = 1u;       // description bytes
const L_TOTAL: u32 = 2u;      // section bytes
const L_START: u32 = 3u;      // + k: frame byte of stream k
const L_BITS: u32 = 7u;       // + k: code bits of stream k
const L_MAX_BITS: u32 = 11u;
const L_MAX_SYMBOL: u32 = 12u;
const L_FIRST: u32 = 13u;     // literal 0

fn highbit(v: u32) -> u32 { return firstLeadingBit(v); }

var<private> fbase: u32;

// ---- literal gather: the literal stream as runs of block bytes (see the top) ----

// Per thread t (and t = WG: the trailing run): literal index and block byte of the first run of
// sequences [t * G, (t + 1) * G).
var<workgroup> ix_lit: array<u32, 257>;
var<workgroup> ix_src: array<u32, 257>;

var<private> dbase: u32;   // word index of the block in data
var<private> sbase: u32;   // word index of the block's sequences in seqs
var<private> n_seq: u32;
var<private> grp: u32;     // G

// Cursor: run cj (sequence cj's literals, or the trailing run when cj == n_seq) spans block bytes
// [c_start, c_end); c_pos is the next byte forward (c_pos - 1 the next one backward).
var<private> cj: u32;
var<private> c_start: u32;
var<private> c_end: u32;
var<private> c_pos: u32;

fn seq_ll(j: u32) -> u32 { return seqs[sbase + 3u * j]; }
fn seq_ml(j: u32) -> u32 { return seqs[sbase + 3u * j + 1u]; }

// Sets run cj starting at block byte `start` (the trailing run ends at BLOCK_SIZE).
fn set_run(start: u32) {
    c_start = start;
    var len = 0u;
    if (cj < n_seq) {
        len = seq_ll(cj);
    } else if (start < BLOCK_SIZE) {
        len = BLOCK_SIZE - start;
    }
    c_end = start + len;
}

// Places the cursor on literal i (< the block's literal count): c_pos is its block byte.
fn seek(i: u32) {
    // Largest t with ix_lit[t] <= i (ix_lit[0] = 0).
    var lo = 0u;
    var hi = WG;
    while (lo < hi) {
        let mid = (lo + hi + 1u) >> 1u;
        if (ix_lit[mid] <= i) { lo = mid; } else { hi = mid - 1u; }
    }
    cj = min(lo * grp, n_seq);
    var at = ix_lit[lo];
    set_run(ix_src[lo]);
    // At most G runs: literal i lies before ix_lit[lo + 1].
    while (cj < n_seq && i >= at + (c_end - c_start)) {
        at += c_end - c_start;
        let next = c_end + seq_ml(cj);
        cj += 1u;
        set_run(next);
    }
    c_pos = c_start + (i - at);
}

// The byte at the cursor; moves the cursor forward past it (into later runs as needed).
fn next_byte() -> u32 {
    while (c_pos >= c_end && cj < n_seq) {
        let next = c_end + seq_ml(cj);
        cj += 1u;
        set_run(next);
        c_pos = c_start;
    }
    let v = load_byte(dbase, c_pos);
    c_pos += 1u;
    return v;
}

// The next 4 bytes (at least 4 literals must follow the cursor), packed little-endian; moves the
// cursor past them. One unaligned word load when they lie in one run.
fn next_word() -> u32 {
    if (c_pos + 4u <= c_end) {
        let v = load_u32_at(dbase, c_pos);
        c_pos += 4u;
        return v;
    }
    var v = 0u;
    for (var k = 0u; k < 4u; k++) { v |= next_byte() << (8u * k); }
    return v;
}

// The byte before the cursor; moves the cursor back onto it (into earlier runs as needed).
fn prev_byte() -> u32 {
    while (c_pos <= c_start && cj > 0u) {
        cj -= 1u;
        c_end = c_start - seq_ml(cj);
        c_start = c_end - seq_ll(cj);
        c_pos = c_end;
    }
    c_pos -= 1u;
    return load_byte(dbase, c_pos);
}

// The 4 bytes before the cursor (at least 4 literals must precede it), packed little-endian (the
// earliest in the low byte); moves the cursor back onto the earliest. One unaligned word load when
// they lie in one run.
fn prev_word() -> u32 {
    if (c_pos >= c_start + 4u) {
        c_pos -= 4u;
        return load_u32_at(dbase, c_pos);
    }
    var v = 0u;
    for (var k = 4u; k > 0u; k--) { v |= prev_byte() << (8u * (k - 1u)); }
    return v;
}

// Builds ix_lit / ix_src (every thread; ends with a barrier).
fn build_index(lid: u32) {
    grp = (n_seq + WG - 1u) / WG;
    let j0 = min(lid * grp, n_seq);
    let j1 = min(j0 + grp, n_seq);
    var sl = 0u;
    var ss = 0u;
    for (var j = j0; j < j1; j++) {
        let ll = seq_ll(j);
        sl += ll;
        ss += ll + seq_ml(j);
    }
    if (lid == 0u) {
        ix_lit[0] = 0u;
        ix_src[0] = 0u;
    }
    ix_lit[lid + 1u] = sl;
    ix_src[lid + 1u] = ss;
    workgroupBarrier();
    // Inclusive Hillis-Steele scan of entries 1..=WG.
    for (var d = 1u; d < WG; d <<= 1u) {
        var vl = ix_lit[lid + 1u];
        var vs = ix_src[lid + 1u];
        if (lid >= d) {
            vl += ix_lit[lid + 1u - d];
            vs += ix_src[lid + 1u - d];
        }
        workgroupBarrier();
        ix_lit[lid + 1u] = vl;
        ix_src[lid + 1u] = vs;
        workgroupBarrier();
    }
}

// The Raw literals section (header + the n literals) at SEC, written by every thread (a
// contiguous range of frame words each); frame_len[b] = its length.
fn write_raw(b: u32, n: u32, lid: u32) {
    let hdr_len = select(select(3u, 2u, n < 4096u), 1u, n < 32u);
    let hdr = select(select(0xCu | (n << 4u), 0x4u | (n << 4u), n < 4096u), n << 3u, n < 32u);
    let lit_start = SEC + hdr_len;
    let end = lit_start + n;
    let w0 = SEC / 4u;
    let nw = (end + 3u) / 4u - w0;
    let per = (nw + WG - 1u) / WG;
    let wa = w0 + min(lid * per, nw);
    let we = w0 + min(lid * per + per, nw);
    if (wa < we) {
        let first = max(4u * wa, lit_start);
        if (first < end) { seek(first - lit_start); }
        for (var w = wa; w < we; w++) {
            if (4u * w >= lit_start && 4u * w + 4u <= end) {
                atomicStore(&frames[fbase + w], next_word());
                continue;
            }
            var v = 0u;
            for (var k = 0u; k < 4u; k++) {
                let p = 4u * w + k;
                var x = 0u;
                if (p >= lit_start && p < end) {
                    x = next_byte();
                } else if (p >= SEC && p < lit_start) {
                    x = (hdr >> (8u * (p - SEC))) & 0xFFu;
                }
                v |= x << (8u * k);
            }
            atomicStore(&frames[fbase + w], v);
        }
    }
    if (lid == 0u) { frame_len[b] = hdr_len + n; }
}

fn nb_bits(s: u32) -> u32 { return ctab[s] >> 16u; }

// OR byte `v` into frame byte `p` of this block.
fn or_byte(p: u32, v: u32) {
    atomicOr(&frames[fbase + (p >> 2u)], (v & 0xFFu) << ((p & 3u) * 8u));
}

// Zero frame words covering bytes [SEC, SEC + len).
fn zero_section(first: u32, len: u32, step: u32) {
    for (var w = SEC / 4u + first; w < (SEC + len + 3u) / 4u; w += step) {
        atomicStore(&frames[fbase + w], 0u);
    }
}

// ---- Huffman table (thread 0; == gzc_core::huffman::build_table) ----

// Count of the symbol at position j of the count-descending order.
fn node_count(np: u32, j: u32) -> u32 { return atomicLoad(&hist[leaves[np - 1u - j]]); }

// == set_max_height(nodes, 11) over (node_count(j), tp[j]), j < np.
fn set_max_height(np: u32) {
    let tgt = HUF_MAX_BITS;
    let last = np - 1u;
    let largest = tp[last];
    if (largest <= tgt) { return; }
    // largest - tgt < 31: at most 65536 literals bound the tree depth to about 23.
    let base_cost = 1i << (largest - tgt);
    var total_cost = 0i;
    var n = i32(last);
    while (tp[n] > tgt) {
        total_cost += base_cost - (1i << (largest - tp[n]));
        tp[n] = tgt;
        n -= 1;
    }
    while (tp[n] == tgt) { n -= 1; }
    total_cost >>= largest - tgt;

    var rank_last: array<u32, 14>;
    for (var k = 0u; k < 14u; k++) { rank_last[k] = NO_SYMBOL; }
    var current = tgt;
    for (var pos = n; pos >= 0; pos--) {
        let bits = tp[pos];
        if (bits >= current) { continue; }
        current = bits;
        rank_last[tgt - current] = u32(pos);
    }

    while (total_cost > 0) {
        var nb_dec = highbit(u32(total_cost)) + 1u;
        while (nb_dec > 1u) {
            let high = rank_last[nb_dec];
            let low = rank_last[nb_dec - 1u];
            if (high == NO_SYMBOL) {
                nb_dec -= 1u;
                continue;
            }
            if (low == NO_SYMBOL) { break; }
            if (node_count(np, high) <= 2u * node_count(np, low)) { break; }
            nb_dec -= 1u;
        }
        while (nb_dec <= HUF_MAX_BITS + 1u && rank_last[nb_dec] == NO_SYMBOL) { nb_dec += 1u; }
        let pos = rank_last[nb_dec];
        total_cost -= 1i << (nb_dec - 1u);
        tp[pos] += 1u;
        if (rank_last[nb_dec - 1u] == NO_SYMBOL) { rank_last[nb_dec - 1u] = pos; }
        if (pos == 0u) {
            rank_last[nb_dec] = NO_SYMBOL;
        } else {
            rank_last[nb_dec] = pos - 1u;
            if (tp[pos - 1u] != tgt - nb_dec) { rank_last[nb_dec] = NO_SYMBOL; }
        }
    }
    while (total_cost < 0) {
        if (rank_last[1] == NO_SYMBOL) {
            while (tp[n] == tgt) { n -= 1; }
            tp[n + 1] -= 1u;
            rank_last[1] = u32(n + 1);
            total_cost += 1;
            continue;
        }
        tp[rank_last[1] + 1u] -= 1u;
        rank_last[1] += 1u;
        total_cost += 1;
    }
}

// Two-queue merge over leaves (np >= 2), limit, canonical codes into ctab; sets L_MAX_BITS and
// L_MAX_SYMBOL.
fn build_table(np: u32) {
    for (var i = 0u; i < np; i++) { tw[i] = atomicLoad(&hist[leaves[i]]); }
    var next_leaf = 0u;
    var next_int = np;
    var len = np;
    for (var it = 1u; it < np; it++) {
        var pick: array<u32, 2>;
        for (var k = 0u; k < 2u; k++) {
            if (next_leaf < np && (next_int == len || tw[next_leaf] <= tw[next_int])) {
                pick[k] = next_leaf;
                next_leaf += 1u;
            } else {
                pick[k] = next_int;
                next_int += 1u;
            }
        }
        tp[pick[0]] = len;
        tp[pick[1]] = len;
        tw[len] = tw[pick[0]] + tw[pick[1]];
        len += 1u;
    }
    // Depths, walking ids down from the root (parents are created after their children).
    tw[2u * np - 2u] = 0u;
    for (var id = 2u * np - 2u; id > 0u; id--) { tw[id - 1u] = tw[tp[id - 1u]] + 1u; }
    // Count-descending order: position j holds leaf np - 1 - j.
    for (var j = 0u; j < np; j++) { tp[j] = tw[np - 1u - j]; }
    set_max_height(np);

    for (var s = 0u; s < 256u; s++) { ctab[s] = 0u; }
    for (var j = 0u; j < np; j++) { ctab[leaves[np - 1u - j]] = tp[j] << 16u; }

    // == from_lengths
    var nb_per_rank: array<u32, 12>;
    var max_bits = 0u;
    var max_symbol = 0u;
    for (var s = 0u; s < 256u; s++) {
        let b = nb_bits(s);
        if (b > 0u) {
            nb_per_rank[b] += 1u;
            max_bits = max(max_bits, b);
            max_symbol = s;
        }
    }
    var val_per_rank: array<u32, 12>;
    var mn = 0u;
    for (var b = max_bits; b >= 1u; b--) {
        val_per_rank[b] = mn;
        mn = (mn + nb_per_rank[b]) >> 1u;
    }
    for (var s = 0u; s < 256u; s++) {
        let b = nb_bits(s);
        if (b > 0u) {
            ctab[s] = val_per_rank[b] | (b << 16u);
            val_per_rank[b] += 1u;
        }
    }
    lay[L_MAX_BITS] = max_bits;
    lay[L_MAX_SYMBOL] = max_symbol;
}

fn weight(s: u32) -> u32 {
    let b = nb_bits(s);
    if (b == 0u) { return 0u; }
    return lay[L_MAX_BITS] + 1u - b;
}

// ---- description bytes (thread 0): a byte-oriented bit writer into `desc` ----

var<private> dacc: u32;
var<private> dcnt: u32;
var<private> dpos: u32;

fn dbyte(p: u32, v: u32) {
    if (p < DESC_BYTES) { desc[p >> 2u] |= (v & 0xFFu) << ((p & 3u) * 8u); }
}

// Append the low n bits of v (n <= 24; fewer than 8 bits are pending between calls).
fn dput(v: u32, n: u32) {
    if (n == 0u) { return; }
    dacc |= (v & ((1u << n) - 1u)) << dcnt;
    dcnt += n;
    while (dcnt >= 8u) {
        dbyte(dpos, dacc);
        dpos += 1u;
        dacc >>= 8u;
        dcnt -= 8u;
    }
}

// == gzc_core::fse::choose_table_log
fn choose_table_log(total: u32, nb_used: u32, mlog: u32) -> u32 {
    let hb = highbit(total);
    let by_total = clamp(select(0u, hb - 2u, hb >= 2u), 5u, mlog);
    return max(by_total, highbit(nb_used) + 1u);
}

// == gzc_core::fse::normalize of wc[0..len] (total wtotal) into wnorm.
fn normalize(wc: ptr<function, array<u32, 12>>, len: u32, total: u32, tl: u32) {
    let size = 1u << tl;
    var sum = 0u;
    var largest = 0u;
    for (var s = 0u; s < len; s++) {
        let c = (*wc)[s];
        if (c == 0u) {
            wnorm[s] = 0;
            continue;
        }
        let n = max(1u, c * size / total);
        wnorm[s] = i32(n);
        sum += n;
        if (c > (*wc)[largest]) { largest = s; }
    }
    if (sum < size) { wnorm[largest] += i32(size - sum); }
    while (sum > size) {
        var best = 0xFFFFFFFFu;
        for (var s = 0u; s < len; s++) {
            let n = wnorm[s];
            if (n > 1 && (best == 0xFFFFFFFFu || n > wnorm[best])) { best = s; }
        }
        wnorm[best] -= 1;
        sum -= 1u;
    }
}

// == gzc_core::fse::write_ncount of wnorm[0..len], through dput.
fn ncount(len: u32, tl: u32) {
    let table_size = 1i << tl;
    var bit_stream = tl - 5u;
    var bit_count = 4u;
    var remaining = table_size + 1;
    var threshold = table_size;
    var nb_bits = tl + 1u;
    var symbol = 0u;
    var previous_is0 = false;
    while (symbol < len && remaining > 1) {
        if (previous_is0) {
            var start = symbol;
            while (symbol < len && wnorm[symbol] == 0) { symbol++; }
            while (symbol >= start + 24u) {
                start += 24u;
                bit_stream += 0xFFFFu << bit_count;
                dput(bit_stream & 0xFFFFu, 16u);
                bit_stream >>= 16u;
            }
            while (symbol >= start + 3u) {
                start += 3u;
                bit_stream += 3u << bit_count;
                bit_count += 2u;
            }
            bit_stream += (symbol - start) << bit_count;
            bit_count += 2u;
            if (bit_count > 16u) {
                dput(bit_stream & 0xFFFFu, 16u);
                bit_stream >>= 16u;
                bit_count -= 16u;
            }
        }
        var count = wnorm[symbol];
        symbol++;
        let mx = (2 * threshold - 1) - remaining;
        remaining -= abs(count);
        count += 1;
        if (count >= threshold) { count += mx; }
        bit_stream += u32(count) << bit_count;
        bit_count += nb_bits;
        bit_count -= select(0u, 1u, count < mx);
        previous_is0 = count == 1;
        while (remaining < threshold) {
            nb_bits -= 1u;
            threshold >>= 1u;
        }
        if (bit_count > 16u) {
            dput(bit_stream & 0xFFFFu, 16u);
            bit_stream >>= 16u;
            bit_count -= 16u;
        }
    }
    dput(bit_stream, ((bit_count + 7u) / 8u) * 8u);
}

// == FseCTable::from_normalized for wnorm[0..len] (no -1 entries: normalize never makes them).
fn build_fse(len: u32, tl: u32) {
    let size = 1u << tl;
    let mask = size - 1u;
    let stp = (size >> 1u) + (size >> 3u) + 3u;
    wcumul[0] = 0u;
    for (var s = 0u; s < len; s++) { wcumul[s + 1u] = wcumul[s] + u32(wnorm[s]); }
    var pos = 0u;
    for (var s = 0u; s < len; s++) {
        for (var i = 0; i < wnorm[s]; i++) {
            wspread[pos] = s;
            pos = (pos + stp) & mask;
        }
    }
    for (var u = 0u; u < size; u++) {
        let s = wspread[u];
        wst[wcumul[s]] = size + u;
        wcumul[s] += 1u;
    }
    var total = 0;
    for (var s = 0u; s < len; s++) {
        let n = wnorm[s];
        if (n == 0) {
            wdfs[s] = 0;
            wdnb[s] = ((tl + 1u) << 16u) - size;
        } else if (n == 1) {
            wdfs[s] = total - 1;
            wdnb[s] = (tl << 16u) - size;
            total += 1;
        } else {
            let mbo = tl - highbit(u32(n) - 1u);
            wdfs[s] = total - n;
            wdnb[s] = (mbo << 16u) - (u32(n) << mbo);
            total += n;
        }
    }
}

// == FseState::init
fn fse_init(sym: u32) -> u32 {
    let dnb = wdnb[sym];
    let nb = (dnb + (1u << 15u)) >> 16u;
    let v = (nb << 16u) - dnb;
    return wst[u32(i32(v >> nb) + wdfs[sym])];
}

// == FseState::encode; returns the next state.
fn fse_encode(state: u32, sym: u32) -> u32 {
    let nb = (state + wdnb[sym]) >> 16u;
    dput(state, nb);
    return wst[u32(i32(state >> nb) + wdfs[sym])];
}

// == table_description into desc; sets L_DESC. False if the table cannot be described.
fn describe() -> bool {
    for (var i = 0u; i < 40u; i++) { desc[i] = 0u; }
    dacc = 0u;
    dcnt = 0u;
    dpos = 0u;
    let nw = lay[L_MAX_SYMBOL]; // weights of symbols 0 .. max_symbol - 1
    if (nw <= 128u) {
        dput(127u + nw, 8u);
        for (var s = 0u; s < nw; s += 2u) {
            var lo = 0u;
            if (s + 1u < nw) { lo = weight(s + 1u); }
            dput((weight(s) << 4u) | lo, 8u);
        }
        lay[L_DESC] = dpos;
        return true;
    }
    // == compress_weights (nw > 128, so more than 2 weights)
    var wc: array<u32, 12>;
    for (var s = 0u; s < nw; s++) { wc[weight(s)] += 1u; }
    var max_count = 0u;
    var max_w = 0u;
    var used = 0u;
    for (var w = 0u; w < 12u; w++) {
        max_count = max(max_count, wc[w]);
        if (wc[w] > 0u) {
            max_w = w;
            used += 1u;
        }
    }
    if (max_count == nw || max_count == 1u) { return false; }
    let tl = choose_table_log(nw, used, WEIGHTS_MAX_LOG);
    let len = max_w + 1u;
    normalize(&wc, len, nw, tl);
    dpos = 1u; // byte 0: the compressed size
    ncount(len, tl);
    build_fse(len, tl);
    var i = nw;
    var s1: u32;
    var s2: u32;
    if (nw % 2u == 1u) {
        s1 = fse_init(weight(nw - 1u));
        s2 = fse_init(weight(nw - 2u));
        s1 = fse_encode(s1, weight(nw - 3u));
        i -= 3u;
    } else {
        s2 = fse_init(weight(nw - 1u));
        s1 = fse_init(weight(nw - 2u));
        i -= 2u;
    }
    while (i > 0u) {
        s2 = fse_encode(s2, weight(i - 1u));
        s1 = fse_encode(s1, weight(i - 2u));
        i -= 2u;
    }
    dput(s2, tl);
    dput(s1, tl);
    dput(1u, 1u); // end mark
    if (dcnt > 0u) {
        dbyte(dpos, dacc);
        dpos += 1u;
    }
    let c_len = dpos - 1u;
    if (c_len >= 128u) { return false; }
    dbyte(0u, c_len);
    lay[L_DESC] = dpos;
    return true;
}

// ---- stream encode (every thread): codes packed into frame words at absolute bit positions ----

var<private> acc: u32;
var<private> cnt: u32;
var<private> wpos: u32;
var<private> first_w: u32;

fn emit(w: u32, v: u32) {
    // Only a thread's first and last words can hold another writer's bits.
    if (w == first_w) {
        atomicOr(&frames[fbase + w], v);
    } else {
        atomicStore(&frames[fbase + w], v);
    }
}

// Append `n` <= 11 bits `v` (v < 2^n).
fn put(v: u32, n: u32) {
    acc |= v << cnt;
    let total = cnt + n;
    if (total >= 32u) {
        emit(wpos, acc);
        wpos += 1u;
        acc = v >> (32u - cnt); // cnt > 21 here
        cnt = total - 32u;
    } else {
        cnt = total;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.x;
    if (b >= arrayLength(&frame_len)) { return; }
    fbase = b * FRAME_WORDS;
    dbase = block_base(b);
    sbase = b * MAX_SEQS * 3u;
    n_seq = counts[2u * b];
    let n = counts[2u * b + 1u];

    build_index(lid);
    if (!HUFFMAN) {
        write_raw(b, n, lid);
        return;
    }

    // ---- histogram (a contiguous range of literals per thread) ----
    atomicStore(&hist[lid], 0u);
    if (lid == 0u) { atomicStore(&n_present, 0u); }
    workgroupBarrier();
    {
        let per = (n + WG - 1u) / WG;
        let ha = min(lid * per, n);
        let he = min(ha + per, n);
        if (ha < he) { seek(ha); }
        var i = ha;
        for (; i + 4u <= he; i += 4u) {
            let w = next_word();
            for (var k = 0u; k < 4u; k++) { atomicAdd(&hist[(w >> (8u * k)) & 0xFFu], 1u); }
        }
        for (; i < he; i++) { atomicAdd(&hist[next_byte()], 1u); }
    }
    workgroupBarrier();

    // ---- RLE / too short ----
    if (lid == 0u) {
        flag = GO;
        var first = 0u;
        if (n > 0u) {
            seek(0u);
            first = next_byte();
        }
        if (n >= 2u && atomicLoad(&hist[first]) == n) {
            flag = RLE;
        } else if (n < MIN_HUF_LITERALS) {
            flag = RAW;
        }
        lay[L_FIRST] = first;
    }
    let f1 = workgroupUniformLoad(&flag);
    if (f1 == RLE) {
        if (lid == 0u) {
            // == write_raw_rle_header(1, n) + the byte
            var hdr_len = 3u;
            var hdr = 1u | 0xCu | (n << 4u);
            if (n < 32u) {
                hdr_len = 1u;
                hdr = 1u | (n << 3u);
            } else if (n < 4096u) {
                hdr_len = 2u;
                hdr = 1u | 0x4u | (n << 4u);
            }
            zero_section(0u, hdr_len + 1u, 1u);
            for (var i = 0u; i < hdr_len; i++) { or_byte(SEC + i, hdr >> (8u * i)); }
            or_byte(SEC + hdr_len, lay[L_FIRST]);
            frame_len[b] = hdr_len + 1u;
        }
        return;
    }
    if (f1 == RAW) {
        write_raw(b, n, lid);
        return;
    }

    // ---- rank sort: present symbols by (count, symbol) ----
    let c = atomicLoad(&hist[lid]);
    if (c > 0u) {
        var rank = 0u;
        for (var s = 0u; s < 256u; s++) {
            let d = atomicLoad(&hist[s]);
            if (d > 0u && (d < c || (d == c && s < lid))) { rank += 1u; }
        }
        leaves[rank] = lid;
        atomicAdd(&n_present, 1u);
    }
    workgroupBarrier();

    // ---- table + description (thread 0) ----
    if (lid == 0u) {
        let np = atomicLoad(&n_present);
        flag = RAW;
        if (np >= 2u) {
            build_table(np);
            if (describe()) { flag = GO; }
        }
    }
    if (workgroupUniformLoad(&flag) == RAW) {
        write_raw(b, n, lid);
        return;
    }

    // ---- chunk bit counts and the segmented prefix sum ----
    let single = n < 256u;
    let n_streams = select(4u, 1u, single);
    let per = WG / n_streams;           // threads per stream
    let k = lid / per;                  // stream
    let r = lid % per;                  // rank in the stream, in coding order
    let seg = select((n + 3u) / 4u, n, single);
    let s0 = k * seg;
    let s1 = min(s0 + seg, n);
    let chunk = (s1 - s0 + per - 1u) / per;
    let j = per - 1u - r;               // chunk index in literal order
    let a = min(s0 + j * chunk, s1);
    let e = min(a + chunk, s1);
    // The cursor stays just past literal e - 1 for the encode below.
    var bits = 0u;
    if (a < e) { seek(a); }
    var i = a;
    for (; i + 4u <= e; i += 4u) {
        let w = next_word();
        for (var k = 0u; k < 4u; k++) { bits += nb_bits((w >> (8u * k)) & 0xFFu); }
    }
    for (; i < e; i++) { bits += nb_bits(next_byte()); }
    scan[lid] = bits;
    workgroupBarrier();
    for (var d = 1u; d < WG; d <<= 1u) {
        var v = scan[lid];
        if (r >= d) { v += scan[lid - d]; }
        workgroupBarrier();
        scan[lid] = v;
        workgroupBarrier();
    }

    // ---- layout and the Compressed-vs-Raw choice (thread 0) ----
    if (lid == 0u) {
        let desc_len = lay[L_DESC];
        var payload = desc_len + select(6u, 0u, single);
        for (var q = 0u; q < n_streams; q++) {
            let sb = scan[q * per + per - 1u];
            lay[L_BITS + q] = sb;
            payload += (sb + 8u) / 8u; // + end mark, rounded up to bytes
        }
        let size = max(n, payload);
        let hdr_len = select(select(5u, 4u, size < 16384u), 3u, size < 1024u);
        let total = hdr_len + payload;
        let raw_len = n + 1u + select(0u, 1u, n >= 32u) + select(0u, 1u, n >= 4096u);
        if (total >= raw_len) {
            flag = RAW;
        } else {
            lay[L_HDR] = hdr_len;
            lay[L_TOTAL] = total;
            var at = SEC + hdr_len + payload;
            for (var q = n_streams; q > 0u; q--) {
                at -= (lay[L_BITS + q - 1u] + 8u) / 8u;
                lay[L_START + q - 1u] = at;
            }
        }
    }
    if (workgroupUniformLoad(&flag) == RAW) {
        write_raw(b, n, lid);
        return;
    }

    // ---- write the section ----
    zero_section(lid, lay[L_TOTAL], WG);
    storageBarrier();

    // Codes of this thread's chunk, last literal first.
    let pos = 8u * lay[L_START + k] + scan[lid] - bits;
    acc = 0u;
    cnt = pos & 31u;
    wpos = pos >> 5u;
    first_w = wpos;
    var ie = e;
    for (; ie >= a + 4u; ie -= 4u) {
        let w = prev_word();
        for (var k = 4u; k > 0u; k--) {
            let t = ctab[(w >> (8u * (k - 1u))) & 0xFFu];
            put(t & 0xFFFFu, t >> 16u);
        }
    }
    for (; ie > a; ie--) {
        let t = ctab[prev_byte()];
        put(t & 0xFFFFu, t >> 16u);
    }
    if (cnt > 0u) { atomicOr(&frames[fbase + wpos], acc); }

    let hdr_len = lay[L_HDR];
    let desc_len = lay[L_DESC];
    for (var i = lid; i < desc_len; i += WG) {
        or_byte(SEC + hdr_len + i, desc[i >> 2u] >> ((i & 3u) * 8u));
    }
    if (lid == 0u) {
        // == compressed_section's header
        let comp = lay[L_TOTAL] - hdr_len;
        var h = 0u;
        if (hdr_len == 3u) {
            h = 2u | (select(1u, 0u, single) << 2u) | (n << 4u) | (comp << 14u);
        } else if (hdr_len == 4u) {
            h = 2u | (2u << 2u) | (n << 4u) | (comp << 18u);
        } else {
            h = 2u | (3u << 2u) | (n << 4u) | (comp << 22u);
            or_byte(SEC + 4u, comp >> 10u);
        }
        for (var i = 0u; i < min(hdr_len, 4u); i++) { or_byte(SEC + i, h >> (8u * i)); }
        // Jump table: sizes of streams 0..2.
        if (!single) {
            for (var q = 0u; q < 3u; q++) {
                let sz = lay[L_START + q + 1u] - lay[L_START + q];
                or_byte(SEC + hdr_len + desc_len + 2u * q, sz);
                or_byte(SEC + hdr_len + desc_len + 2u * q + 1u, sz >> 8u);
            }
        }
        // End marks.
        for (var q = 0u; q < n_streams; q++) {
            let p = 8u * lay[L_START + q] + lay[L_BITS + q];
            atomicOr(&frames[fbase + (p >> 5u)], 1u << (p & 31u));
        }
        frame_len[b] = lay[L_TOTAL];
    }
}
