// K4: sequence entropy coding + block assembly. One workgroup of 64 threads per block turns K3's
// parse into the block's complete zstd frame, byte-identical to
// gzc_core::frame::write_frame(block, parse, FrameOptions { checksum: false, huffman: HUFFMAN }):
//   frame header, then one block that is
//   - RLE (every byte equal), else
//   - Compressed: literals section + sequences section (gzc_core::seqenc::
//     write_sequences_section_auto: per-stream predefined / RLE / computed FSE table), if the
//     content is smaller than BLOCK_SIZE, else
//   - Raw.
// K5 has already written the literals section (Raw, RLE or Compressed) into the frame at byte
// HDR_LEN + 3, and frame_len[b] holds its length on entry: K4 keeps K5's bytes, writing only the
// headers below it and the sequences section after it.
// Threads cooperate on the RLE check, the code histograms, the FSE table builds (in workgroup
// memory) and the raw copy; thread 0 makes the mode decisions and writes the table descriptions.
// The backward sequence encode runs in chunks: only the FSE state transitions are sequential
// (one thread per stream), the bit counts and the bit placement are parallel.
//
// The host binds exactly n_blocks words of `frame_len`, so arrayLength(&frame_len) is the batch
// size. Outputs per block b:
//   frames[b*FRAME_WORDS ..]  frame bytes, packed little-endian (bytes past frame_len are junk)
//   frame_len[b]              frame length in bytes
// Prepended by the host: MAX_SEQS, FRAME_WORDS, HDR_LEN / HDR_W0..2 (frame header bytes) and the
// TAB_* offsets into `tab` (code tables, FRAC, predefined distributions from
// gzc_core).

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> seqs: array<u32>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read> tab: array<u32>;
@group(0) @binding(4) var<storage, read_write> frames: array<u32>;
@group(0) @binding(5) var<storage, read_write> frame_len: array<u32>;

const WG: u32 = 64u;
// Histogram / norm / symbol-transform offsets per stream (LL 36 codes, OF 32, ML 53).
const H_LL: u32 = 0u;
const H_OF: u32 = 36u;
const H_ML: u32 = 68u;
const H_ALL: u32 = 121u;
// State-table offsets per stream (LL/ML up to log 9, OF up to log 8).
const S_LL: u32 = 0u;
const S_OF: u32 = 512u;
const S_ML: u32 = 768u;
const S_ALL: u32 = 1280u;

const MODE_PREDEFINED: u32 = 0u;
const MODE_RLE: u32 = 1u;
const MODE_COMPRESSED: u32 = 2u;

var<workgroup> hist: array<atomic<u32>, H_ALL>;
var<workgroup> norm: array<i32, H_ALL>;
var<workgroup> tt_dfs: array<i32, H_ALL>;
var<workgroup> tt_dnb: array<u32, H_ALL>;
var<workgroup> st: array<u32, S_ALL>;
var<workgroup> cumul: array<u32, 54>;
var<workgroup> not_rle: atomic<u32>;
var<workgroup> rle_flag: u32;
var<workgroup> raw_flag: u32;
var<workgroup> section_wg: u32;

// ---- chunked backward sequence encode ----
// Sequences are encoded in chunks of C, last chunk first. Per chunk:
//   1. all threads: the chunk's codes into sbuf[r] (r = rank in coding order, i.e. sequence
//      hi - 1 - r), packed LL | OF << 8 | ML << 16; thread t handles ranks [t * PER, t * PER + PER)
//      in steps 1 and 3 and holds their sequences in registers, loaded one chunk ahead;
//   2. threads 0..2 (one per stream k: LL, OF, ML): the FSE state chain, recording for each
//      sequence the bits its step emits, value | nbits << 16, into sbuf[(k + 1) * C + r];
//   3. all threads: per-sequence bit counts, a workgroup prefix sum, and the bits placed with
//      atomicOr into the staging words `stg` (bit 0 of stg[0] = frame bit pos_wg & ~31);
//   4. all threads: the complete staging words out to the frame; the partial last word carries
//      over as the next chunk's stg[0].
const C: u32 = 256u;
const PER: u32 = C / WG;
// Longest sequence: 27 state bits + 16 + 16 + 17 extra bits = 76 (offset codes stay below 18
// for BLOCK_SIZE <= 128K); + one word for the carry offset + one of slack.
const STG: u32 = (C * 76u + 31u) / 32u + 2u;
// sbuf: codes [0, C), emitted state bits [C, 4C).
var<workgroup> sbuf: array<u32, 1024>;
var<workgroup> stg: array<atomic<u32>, STG>;
// wg_scan's per-thread values. Scalars: a store to one component of a workgroup vector is a
// read-modify-write of the whole vector on Apple GPUs (Metal), so threads storing neighbouring
// components of one vec4 lost each other's values there (short, corrupt frames on an M4 Pro;
// `GZC_EMULATE_VEC_RMW` reproduces it elsewhere).
var<workgroup> scan: array<u32, WG>;
var<workgroup> nseq_wg: u32;
var<workgroup> pos_wg: u32;
var<workgroup> carry_wg: u32;
// pos_wg once the sequences cannot fit: the block goes out Raw.
const STOP: u32 = 0xFFFFFFFFu;
var<workgroup> fin: array<u32, 3>;
// build_table: the spread, one symbol per byte (tables up to 512 cells); visit starts per symbol;
// the tables to build (per stream 0, or len | log << 8); the number of -1 symbols.
var<workgroup> sp: array<atomic<u32>, 128>;
var<workgroup> vcum: array<u32, 54>;
var<workgroup> bld: array<u32, 3>;
var<workgroup> nlow_wg: u32;

// ---- stream parameters (k: 0 = LL, 1 = OF, 2 = ML) ----

fn h_off(k: u32) -> u32 { return select(select(H_ML, H_OF, k == 1u), H_LL, k == 0u); }
fn s_off(k: u32) -> u32 { return select(select(S_ML, S_OF, k == 1u), S_LL, k == 0u); }
fn alphabet(k: u32) -> u32 { return select(select(53u, 32u, k == 1u), 36u, k == 0u); }
fn max_log(k: u32) -> u32 { return select(9u, 8u, k == 1u); }
fn def_off(k: u32) -> u32 { return select(select(TAB_ML_NORM, TAB_OF_NORM, k == 1u), TAB_LL_NORM, k == 0u); }
fn def_len(k: u32) -> u32 { return select(select(53u, 29u, k == 1u), 36u, k == 0u); }
fn def_log(k: u32) -> u32 { return select(6u, 5u, k == 1u); }
fn def_norm(k: u32, s: u32) -> i32 { return bitcast<i32>(tab[def_off(k) + s]); }

fn highbit(v: u32) -> u32 { return firstLeadingBit(v); }

// == gzc_core::codes::{ll_code, ml_code, of_code}
fn ll_code(ll: u32) -> u32 {
    if (ll > 63u) { return highbit(ll) + 19u; }
    return tab[TAB_LL_CODE + ll];
}
fn ml_code(ml: u32) -> u32 {
    let b = ml - 3u;
    if (b > 127u) { return highbit(b) + 36u; }
    return tab[TAB_ML_CODE + b];
}

// ---- output: frame words of block `fbase`, written by one thread at a time ----

var<private> fbase: u32;

// Bit writer (thread 0): `acc` holds `cnt` < 32 pending bits of frame word `wpos`; bytes before
// the sequences section's start in its first word are zero bits (patched afterwards).
var<private> acc: u32;
var<private> cnt: u32;
var<private> wpos: u32;

fn store_word(w: u32, v: u32) {
    if (w < FRAME_WORDS) { frames[fbase + w] = v; }
}

// Append the low `n` bits of `v` (n <= 24).
fn put(v: u32, n: u32) {
    if (n == 0u) { return; }
    let m = v & ((1u << n) - 1u);
    acc |= m << cnt;
    let total = cnt + n;
    if (total >= 32u) {
        store_word(wpos, acc);
        wpos += 1u;
        acc = m >> (32u - cnt); // cnt > 0 here since n <= 24
        cnt = total - 32u;
    } else {
        cnt = total;
    }
}

// Byte position just past the written bits, rounded up.
fn put_end() -> u32 { return wpos * 4u + (cnt + 7u) / 8u; }

// Store the pending partial word.
fn put_flush() {
    if (cnt > 0u) { store_word(wpos, acc); }
}

// ---- FSE helpers (== gzc_core::fse) ----

fn log2_x256(x: u32) -> u32 {
    let hb = highbit(x);
    return 256u * hb + tab[TAB_FRAC + (((x << 8u) >> hb) & 255u)];
}

// == choose_table_log
fn choose_table_log(total: u32, nb_used: u32, mlog: u32) -> u32 {
    let hb = highbit(total);
    let by_total = clamp(select(0u, hb - 2u, hb >= 2u), 5u, mlog);
    return max(by_total, highbit(nb_used) + 1u);
}

// == normalize, into norm[h .. h + len] (len = last used code + 1).
fn fse_normalize(h: u32, len: u32, total: u32, tl: u32) {
    let size = 1u << tl;
    var sum = 0u;
    var largest = 0u;
    for (var s = 0u; s < len; s++) {
        let c = atomicLoad(&hist[h + s]);
        if (c == 0u) {
            norm[h + s] = 0;
            continue;
        }
        let n = max(1u, c * size / total);
        norm[h + s] = i32(n);
        sum += n;
        if (c > atomicLoad(&hist[h + largest])) { largest = s; }
    }
    if (sum < size) {
        norm[h + largest] += i32(size - sum);
    }
    while (sum > size) {
        var best = 0xFFFFFFFFu;
        for (var s = 0u; s < len; s++) {
            let n = norm[h + s];
            if (n > 1 && (best == 0xFFFFFFFFu || n > norm[h + best])) { best = s; }
        }
        norm[h + best] -= 1;
        sum -= 1u;
    }
}

// == cost_x256 of the histogram at h against a norm read through `k`'s default (dflt) or norm[h..].
fn cost(k: u32, len: u32, tl: u32, dflt: bool) -> u32 {
    let h = h_off(k);
    let full = 256u * tl;
    var c_sum = 0u;
    for (var s = 0u; s < len; s++) {
        let c = atomicLoad(&hist[h + s]);
        if (c == 0u) { continue; }
        var n: i32;
        if (dflt) { n = def_norm(k, s); } else { n = norm[h + s]; }
        c_sum += c * (full - log2_x256(u32(abs(n))));
    }
    return c_sum;
}

// == write_ncount for norm[h .. h + len]; appends through `put` when `emit`, returns the byte count.
fn ncount(h: u32, len: u32, tl: u32, emit: bool) -> u32 {
    var bytes = 0u;
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
            while (symbol < len && norm[h + symbol] == 0) { symbol++; }
            while (symbol >= start + 24u) {
                start += 24u;
                bit_stream += 0xFFFFu << bit_count;
                if (emit) { put(bit_stream & 0xFFFFu, 16u); }
                bytes += 2u;
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
                if (emit) { put(bit_stream & 0xFFFFu, 16u); }
                bytes += 2u;
                bit_stream >>= 16u;
                bit_count -= 16u;
            }
        }
        var count = norm[h + symbol];
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
            if (emit) { put(bit_stream & 0xFFFFu, 16u); }
            bytes += 2u;
            bit_stream >>= 16u;
            bit_count -= 16u;
        }
    }
    let tail = (bit_count + 7u) / 8u;
    if (emit) { put(bit_stream, tail * 8u); }
    return bytes + tail;
}

// Exclusive prefix sum of v over the workgroup's threads (x) and the total (y); all threads call
// it. A barrier must separate two calls.
fn wg_scan(lid: u32, v: u32) -> vec2<u32> {
    scan[lid] = v;
    workgroupBarrier();
    var pre = 0u;
    var tot = 0u;
    for (var i = 0u; i < WG; i++) {
        let q = scan[i];
        pre += select(0u, q, i < lid);
        tot += q;
    }
    return vec2<u32>(pre, tot);
}

// == FseCTable::from_normalized for stream k (len symbols, table log tl), from norm[h_off(k) ..]
// into st[s_off(k) ..] and tt_*[h_off(k) ..]. All threads call it (uniformly):
//   - thread 0: cumul[s] (first state-table slot of s, a -1 symbol counting 1), vcum[s] (first
//     spread visit of s, -1 symbols not visiting), and the -1 symbols in the top cells;
//   - the spread: visit j of the sequential walk lands on the j-th t in 0, 1, .. whose
//     (t * step) & mask is <= high (the walk is that sequence with the top cells skipped), so a
//     thread takes a range of t, counts its visits and places them after a prefix sum (none is
//     needed without -1 symbols: visit j is t = j);
//   - thread s: symbol s's transform and its state-table slots, its cells in increasing order.
// The spread holds one symbol per byte in `sp`.
fn build_table(k: u32, len: u32, tl: u32, lid: u32) {
    let h = h_off(k);
    let so = s_off(k);
    let size = 1u << tl;
    let mask = size - 1u;
    let stp = (size >> 1u) + (size >> 3u) + 3u;
    for (var i = lid; i < size / 4u; i += WG) { atomicStore(&sp[i], 0u); }
    workgroupBarrier();
    if (lid == 0u) {
        var c = 0u;
        var v = 0u;
        var high = size - 1u;
        for (var s = 0u; s < len; s++) {
            let n = norm[h + s];
            cumul[s] = c;
            vcum[s] = v;
            if (n == -1) {
                atomicOr(&sp[high >> 2u], s << ((high & 3u) * 8u));
                high -= 1u;
                c += 1u;
            } else {
                c += u32(n);
                v += u32(n);
            }
        }
        vcum[len] = v;
        nlow_wg = size - 1u - high;
    }
    let nlow = workgroupUniformLoad(&nlow_wg);
    let high = size - 1u - nlow;
    let per = (size + WG - 1u) / WG;
    let t0 = min(lid * per, size);
    let t1 = min(t0 + per, size);
    var j = t0;
    if (nlow > 0u) {
        var nv = 0u;
        for (var t = t0; t < t1; t++) {
            if (((t * stp) & mask) <= high) { nv += 1u; }
        }
        j = wg_scan(lid, nv).x;
    }
    let visits = size - nlow;
    if (j < visits && t0 < t1) {
        // Symbol of visit j: the first s with vcum[s + 1] > j.
        var lo = 0u;
        var hi = len - 1u;
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            if (vcum[mid + 1u] > j) { hi = mid; } else { lo = mid + 1u; }
        }
        var s = lo;
        for (var t = t0; t < t1; t++) {
            let u = (t * stp) & mask;
            if (u > high) { continue; }
            while (vcum[s + 1u] <= j) { s++; }
            atomicOr(&sp[u >> 2u], s << ((u & 3u) * 8u));
            j++;
        }
    }
    workgroupBarrier();
    if (lid < len) {
        let s = lid;
        let n = norm[h + s];
        let c0 = cumul[s];
        if (n == 0) {
            tt_dfs[h + s] = 0;
            tt_dnb[h + s] = ((tl + 1u) << 16u) - size;
        } else if (n == -1 || n == 1) {
            tt_dfs[h + s] = i32(c0) - 1;
            tt_dnb[h + s] = (tl << 16u) - size;
        } else {
            let mbo = tl - highbit(u32(n) - 1u);
            tt_dfs[h + s] = i32(c0) - n;
            tt_dnb[h + s] = (mbo << 16u) - (u32(n) << mbo);
        }
        if (n == -1) {
            // The -1 symbols before s (c0 - vcum[s]) took the cells above s's.
            st[so + c0] = size + size - 1u - (c0 - vcum[s]);
        } else if (n > 0) {
            var c = c0;
            let c1 = c0 + u32(n);
            for (var w = 0u; w < size / 4u && c < c1; w++) {
                let word = atomicLoad(&sp[w]);
                for (var i = 0u; i < 4u; i++) {
                    if (((word >> (8u * i)) & 0xFFu) == s) {
                        st[so + c] = size + 4u * w + i;
                        c++;
                    }
                }
            }
        }
    }
    workgroupBarrier();
}

// == FseCTable::rle: one state, no bits for `sym`.
fn build_rle(k: u32, sym: u32) {
    st[s_off(k)] = 0u;
    tt_dfs[h_off(k) + sym] = 0;
    tt_dnb[h_off(k) + sym] = 0u;
}

// == FseState::init
fn fse_init(k: u32, sym: u32) -> u32 {
    let dnb = tt_dnb[h_off(k) + sym];
    let nb = (dnb + (1u << 15u)) >> 16u;
    let v = (nb << 16u) - dnb;
    return st[s_off(k) + u32(i32(v >> nb) + tt_dfs[h_off(k) + sym])];
}

// A sequence's codes, packed LL | OF << 8 | ML << 16.
fn seq_codes(ll: u32, ml: u32, ob: u32) -> u32 { return ll_code(ll) | (highbit(ob) << 8u) | (ml_code(ml) << 16u); }

// Staging writer (phase 3): the same bit writer as `put`, OR-ing words into stg (a thread's
// first and last words may be shared with its neighbours).
fn put_s(v: u32, n: u32) {
    if (n == 0u) { return; }
    let m = v & ((1u << n) - 1u);
    acc |= m << cnt;
    let total = cnt + n;
    if (total >= 32u) {
        atomicOr(&stg[wpos], acc);
        wpos += 1u;
        acc = m >> (32u - cnt);
        cnt = total - 32u;
    } else {
        cnt = total;
    }
}

// ---- frame prefix bytes ----

fn hdr_byte(k: u32) -> u32 {
    let w = select(select(HDR_W2, HDR_W1, k < 8u), HDR_W0, k < 4u);
    return (w >> ((k & 3u) * 8u)) & 0xFFu;
}

// Byte k of a frame whose block header is `bh` (3 bytes), followed by `extra_len` bytes of `extra`,
// then block bytes from word `src` of data.
fn prefix_byte(k: u32, bh: u32, extra: u32, extra_len: u32, src: u32) -> u32 {
    if (k < HDR_LEN) { return hdr_byte(k); }
    let j = k - HDR_LEN;
    if (j < 3u) { return (bh >> (j * 8u)) & 0xFFu; }
    if (j < 3u + extra_len) { return (extra >> ((j - 3u) * 8u)) & 0xFFu; }
    let i = j - 3u - extra_len;
    return (data[src + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xFFu;
}

fn prefix_word(w: u32, bh: u32, extra: u32, extra_len: u32, src: u32) -> u32 {
    var v = 0u;
    for (var i = 0u; i < 4u; i++) {
        v |= prefix_byte(4u * w + i, bh, extra, extra_len, src) << (8u * i);
    }
    return v;
}

// Frame word w with its bytes below the literals section (frame + block header, block header
// `bh`) set, keeping the bytes K5 wrote from HDR_LEN + 3 on.
fn prefix_word_over_section(w: u32, bh: u32) -> u32 {
    var v = frames[fbase + w];
    for (var i = 0u; i < 4u; i++) {
        let k = 4u * w + i;
        if (k < HDR_LEN + 3u) {
            v = (v & ~(0xFFu << (8u * i))) | (prefix_byte(k, bh, 0u, 0u, 0u) << (8u * i));
        }
    }
    return v;
}

// Funnel-shifted word of data words from `src`, starting at byte offset i (bytes i .. i+3).
fn data_word(src: u32, i: u32) -> u32 {
    let a = src + (i >> 2u);
    let sh = (i & 3u) * 8u;
    if (sh == 0u) { return data[a]; }
    return (data[a] >> sh) | (data[a + 1u] << (32u - sh));
}

// Words [0, 4) are written by thread 0 byte by byte; the header is at most 15 bytes long.
const PREFIX_WORDS: u32 = 4u;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.x;
    if (b >= arrayLength(&frame_len)) { return; }
    let base = block_base(b);
    fbase = b * FRAME_WORDS;
    let n_seq = counts[2u * b];
    let sbase = b * MAX_SEQS * 3u;
    // K5's literals section length. Thread 0 alone reads frame_len[b] (it alone overwrites it
    // later) and shares it through workgroup memory: a per-invocation storage read would race
    // with that write, since workgroupBarrier does not order storage accesses.
    if (lid == 0u) { section_wg = frame_len[b]; }
    let section = workgroupUniformLoad(&section_wg);

    // ---- histograms and the RLE-block check ----
    for (var i = lid; i < H_ALL; i += WG) { atomicStore(&hist[i], 0u); }
    if (lid < 3u) { bld[lid] = 0u; }
    if (lid == 0u) { atomicStore(&not_rle, 0u); }
    workgroupBarrier();
    for (var i = lid; i < n_seq; i += WG) {
        let s = sbase + i * 3u;
        atomicAdd(&hist[H_LL + ll_code(seqs[s])], 1u);
        atomicAdd(&hist[H_ML + ml_code(seqs[s + 1u])], 1u);
        atomicAdd(&hist[H_OF + highbit(seqs[s + 2u])], 1u);
    }
    // The RLE check runs in rounds of 8 words per thread and stops after the first round that
    // finds a byte differing from the first (most blocks: the first round).
    let first = data[base] & 0xFFu;
    let first4 = first * 0x01010101u;
    var w0 = 0u;
    loop {
        var diff = 0u;
        for (var q = 0u; q < 8u; q++) {
            let w = w0 + q * WG + lid;
            if (w < BLOCK_SIZE / 4u) { diff |= data[base + w] ^ first4; }
        }
        if (diff != 0u) { atomicOr(&not_rle, 1u); }
        workgroupBarrier();
        if (lid == 0u) { rle_flag = atomicLoad(&not_rle); }
        w0 += 8u * WG;
        if (workgroupUniformLoad(&rle_flag) != 0u || w0 >= BLOCK_SIZE / 4u) { break; }
    }
    workgroupBarrier();
    if (lid == 0u) { rle_flag = atomicLoad(&not_rle); }
    if (workgroupUniformLoad(&rle_flag) == 0u) {
        if (lid == 0u) {
            let bh = 1u | (1u << 1u) | (BLOCK_SIZE << 3u);
            for (var w = 0u; w < PREFIX_WORDS; w++) {
                frames[fbase + w] = prefix_word(w, bh, first, 1u, 0u);
            }
            frame_len[b] = HDR_LEN + 4u;
        }
        return;
    }

    // frame byte of the sequences section
    let seq_start = HDR_LEN + 3u + section;

    // ---- sequences section: header and tables (thread 0) ----
    var tlog: array<u32, 3>;
    if (lid == 0u) {
        acc = 0u;
        cnt = (seq_start & 3u) * 8u;
        wpos = seq_start / 4u;
        if (cnt > 0u) {
            // Keep the literals-section bytes of the first sequences word.
            acc = frames[fbase + wpos] & ((1u << cnt) - 1u);
        }
        if (n_seq < 128u) {
            put(n_seq, 8u);
        } else if (n_seq < 0x7F00u) {
            put((n_seq >> 8u) + 0x80u, 8u);
            put(n_seq & 0xFFu, 8u);
        } else {
            put(0xFFu, 8u);
            put(n_seq - 0x7F00u, 16u);
        }
        if (n_seq > 0u) {
            // Mode choice per stream (== StreamTable::choose).
            var mode: array<u32, 3>;
            var len: array<u32, 3>;
            var sym: array<u32, 3>;
            for (var k = 0u; k < 3u; k++) {
                let h = h_off(k);
                var used = 0u;
                var first_used = 0u;
                var last_used = 0u;
                var predefined_ok = true;
                for (var s = 0u; s < alphabet(k); s++) {
                    if (atomicLoad(&hist[h + s]) == 0u) { continue; }
                    if (used == 0u) { first_used = s; }
                    used += 1u;
                    last_used = s;
                    if (s >= def_len(k) || def_norm(k, s) == 0) { predefined_ok = false; }
                }
                len[k] = last_used + 1u;
                sym[k] = first_used;
                if (used == 1u && n_seq > 2u) {
                    mode[k] = MODE_RLE;
                    continue;
                }
                let lg = choose_table_log(n_seq, used, max_log(k));
                fse_normalize(h, len[k], n_seq, lg);
                tlog[k] = lg;
                mode[k] = MODE_COMPRESSED;
                if (predefined_ok) {
                    let computed_cost = cost(k, len[k], lg, false) + 256u * 8u * ncount(h, len[k], lg, false);
                    if (computed_cost >= cost(k, len[k], def_log(k), true)) {
                        mode[k] = MODE_PREDEFINED;
                    }
                }
            }
            put((mode[0] << 6u) | (mode[1] << 4u) | (mode[2] << 2u), 8u);
            for (var k = 0u; k < 3u; k++) {
                if (mode[k] == MODE_RLE) {
                    put(sym[k], 8u);
                    build_rle(k, sym[k]);
                    tlog[k] = 0u;
                    bld[k] = 0u;
                } else if (mode[k] == MODE_COMPRESSED) {
                    ncount(h_off(k), len[k], tlog[k], true);
                    bld[k] = len[k] | (tlog[k] << 8u);
                } else {
                    tlog[k] = def_log(k);
                    for (var s = 0u; s < def_len(k); s++) { norm[h_off(k) + s] = def_norm(k, s); }
                    bld[k] = def_len(k) | (tlog[k] << 8u);
                }
            }
        }
        pos_wg = wpos * 32u + cnt;
        carry_wg = acc;
        nseq_wg = n_seq;
    }

    // ---- FSE tables (all threads) ----
    for (var k = 0u; k < 3u; k++) {
        let bk = workgroupUniformLoad(&bld[k]);
        if (bk != 0u) { build_table(k, bk & 0xFFu, bk >> 8u, lid); }
    }

    // ---- backward sequence encode (== write_sequences_section_with), chunked ----
    let nseq = workgroupUniformLoad(&nseq_wg);
    for (var j = lid; j < STG; j += WG) { atomicStore(&stg[j], 0u); }
    workgroupBarrier();
    if (lid == 0u) { atomicStore(&stg[0], carry_wg); }
    var hi = nseq;
    var chain = 0u; // threads 0..2: the FSE state of stream lid
    // Thread t's sequences (ranks [t * PER, t * PER + PER)) of the current chunk, loaded one chunk
    // ahead so that the loads' latency hides behind the previous chunk's work.
    var q_ll: array<u32, PER>;
    var q_ml: array<u32, PER>;
    var q_ob: array<u32, PER>;
    for (var q = 0u; q < PER; q++) {
        let r = lid * PER + q;
        if (r < min(hi, C)) {
            let s = sbase + (hi - 1u - r) * 3u;
            q_ll[q] = seqs[s];
            q_ml[q] = seqs[s + 1u];
            q_ob[q] = seqs[s + 2u];
        }
    }
    loop {
        let pos = workgroupUniformLoad(&pos_wg);
        if (hi == 0u || pos == STOP) { break; }
        let lo = select(0u, hi - C, hi > C);
        let n = hi - lo;
        // 1. codes
        for (var q = 0u; q < PER; q++) {
            let r = lid * PER + q;
            if (r < n) { sbuf[r] = seq_codes(q_ll[q], q_ml[q], q_ob[q]); }
        }
        workgroupBarrier();
        // The next chunk's sequences.
        var x_ll: array<u32, PER>;
        var x_ml: array<u32, PER>;
        var x_ob: array<u32, PER>;
        for (var q = 0u; q < PER; q++) {
            let r = lid * PER + q;
            if (r < min(lo, C)) {
                let s = sbase + (lo - 1u - r) * 3u;
                x_ll[q] = seqs[s];
                x_ml[q] = seqs[s + 1u];
                x_ob[q] = seqs[s + 2u];
            }
        }
        // 2. state chains (== FseState::init / encode)
        if (lid < 3u) {
            let k = lid;
            let h = h_off(k);
            let so = s_off(k);
            let sh = 8u * k;
            let eo = (k + 1u) * C;
            var r = 0u;
            if (hi == nseq) {
                // The last sequence initializes the states and emits no state bits.
                chain = fse_init(k, (sbuf[0] >> sh) & 0xFFu);
                sbuf[eo] = 0u;
                r = 1u;
            }
            // Four steps at a time: the table loads, which do not depend on the state, go first.
            for (; r + 4u <= n; r += 4u) {
                let s0 = h + ((sbuf[r] >> sh) & 0xFFu);
                let s1 = h + ((sbuf[r + 1u] >> sh) & 0xFFu);
                let s2 = h + ((sbuf[r + 2u] >> sh) & 0xFFu);
                let s3 = h + ((sbuf[r + 3u] >> sh) & 0xFFu);
                let d0 = tt_dnb[s0];
                let d1 = tt_dnb[s1];
                let d2 = tt_dnb[s2];
                let d3 = tt_dnb[s3];
                let f0 = i32(so) + tt_dfs[s0];
                let f1 = i32(so) + tt_dfs[s1];
                let f2 = i32(so) + tt_dfs[s2];
                let f3 = i32(so) + tt_dfs[s3];
                var nb = (chain + d0) >> 16u;
                sbuf[eo + r] = (chain & ((1u << nb) - 1u)) | (nb << 16u);
                chain = st[u32(i32(chain >> nb) + f0)];
                nb = (chain + d1) >> 16u;
                sbuf[eo + r + 1u] = (chain & ((1u << nb) - 1u)) | (nb << 16u);
                chain = st[u32(i32(chain >> nb) + f1)];
                nb = (chain + d2) >> 16u;
                sbuf[eo + r + 2u] = (chain & ((1u << nb) - 1u)) | (nb << 16u);
                chain = st[u32(i32(chain >> nb) + f2)];
                nb = (chain + d3) >> 16u;
                sbuf[eo + r + 3u] = (chain & ((1u << nb) - 1u)) | (nb << 16u);
                chain = st[u32(i32(chain >> nb) + f3)];
            }
            for (; r < n; r++) {
                let sym = (sbuf[r] >> sh) & 0xFFu;
                let nb = (chain + tt_dnb[h + sym]) >> 16u;
                sbuf[eo + r] = (chain & ((1u << nb) - 1u)) | (nb << 16u);
                chain = st[so + u32(i32(chain >> nb) + tt_dfs[h + sym])];
            }
            fin[k] = chain;
        }
        workgroupBarrier();
        // 3. bit counts, prefix sum, placement. Thread t owns ranks [t * PER, t * PER + PER).
        let r0 = lid * PER;
        let r1 = min(r0 + PER, n);
        var bits = 0u;
        for (var r = r0; r < r1; r++) {
            let c = sbuf[r];
            let llc = c & 0xFFu;
            let ofc = (c >> 8u) & 0xFFu;
            let mlc = c >> 16u;
            bits += (sbuf[C + r] >> 16u) + (sbuf[2u * C + r] >> 16u) + (sbuf[3u * C + r] >> 16u)
                + tab[TAB_LL_BITS + llc] + tab[TAB_ML_BITS + mlc] + ofc;
        }
        let sc = wg_scan(lid, bits);
        let p = (pos & 31u) + sc.x;
        acc = 0u;
        cnt = p & 31u;
        wpos = p >> 5u;
        for (var q = 0u; q < PER; q++) {
            let r = r0 + q;
            if (r >= r1) { break; }
            let c = sbuf[r];
            let llc = c & 0xFFu;
            let ofc = (c >> 8u) & 0xFFu;
            let mlc = c >> 16u;
            let e_ll = sbuf[C + r];
            let e_of = sbuf[2u * C + r];
            let e_ml = sbuf[3u * C + r];
            put_s(e_of & 0xFFFFu, e_of >> 16u);
            put_s(e_ml & 0xFFFFu, e_ml >> 16u);
            put_s(e_ll & 0xFFFFu, e_ll >> 16u);
            put_s(q_ll[q] - tab[TAB_LL_BASE + llc], tab[TAB_LL_BITS + llc]);
            put_s(q_ml[q] - tab[TAB_ML_BASE + mlc], tab[TAB_ML_BITS + mlc]);
            put_s(q_ob[q] - (1u << ofc), ofc);
        }
        if (cnt > 0u) { atomicOr(&stg[wpos], acc); }
        workgroupBarrier();
        // 4. complete words out; the partial last one carries over.
        let total = sc.y;
        let nf = ((pos & 31u) + total) >> 5u;
        let w0 = pos >> 5u;
        for (var j = lid; j < nf; j += WG) { store_word(w0 + j, atomicLoad(&stg[j])); }
        if (lid == 0u) {
            carry_wg = atomicLoad(&stg[nf]);
            pos_wg = pos + total;
            // Past a Raw block's size: the block goes out Raw, stop encoding.
            if ((pos + total) / 8u >= HDR_LEN + 3u + BLOCK_SIZE) { pos_wg = STOP; }
        }
        workgroupBarrier();
        // Only words [0, nf] were written.
        for (var j = lid; j <= nf; j += WG) { atomicStore(&stg[j], select(0u, carry_wg, j == 0u)); }
        q_ll = x_ll;
        q_ml = x_ml;
        q_ob = x_ob;
        hi = lo;
    }
    storageBarrier();

    // ---- final states, end mark, block header (thread 0) ----
    if (lid == 0u) {
        let pos = pos_wg;
        acc = carry_wg;
        cnt = pos & 31u;
        wpos = pos >> 5u;
        if (n_seq > 0u) {
            put(fin[2], tlog[2]);
            put(fin[1], tlog[1]);
            put(fin[0], tlog[0]);
            put(1u, 1u); // end mark
        }
        let end = put_end();
        put_flush();
        let content = end - (HDR_LEN + 3u);
        if (pos == STOP || wpos >= FRAME_WORDS || content >= BLOCK_SIZE) {
            raw_flag = 1u;
        } else {
            raw_flag = 0u;
            let bh = 1u | (2u << 1u) | (content << 3u);
            let first_seq_word = seq_start / 4u;
            for (var w = 0u; w < min(PREFIX_WORDS, first_seq_word); w++) {
                frames[fbase + w] = prefix_word_over_section(w, bh);
            }
            // The first sequences-section word starts with prefix / literal bytes.
            let keep = (seq_start & 3u) * 8u;
            if (keep > 0u) {
                var low = prefix_word_over_section(first_seq_word, bh);
                low &= (1u << keep) - 1u;
                frames[fbase + first_seq_word] = (frames[fbase + first_seq_word] & ~((1u << keep) - 1u)) | low;
            }
            frame_len[b] = end;
        }
    }

    // ---- raw fallback: header + the block verbatim, over whatever was written above ----
    storageBarrier();
    if (workgroupUniformLoad(&raw_flag) == 1u) {
        let raw_start = HDR_LEN + 3u;
        let raw_words = (raw_start + BLOCK_SIZE + 3u) / 4u;
        for (var w = PREFIX_WORDS + lid; w < raw_words; w += WG) {
            frames[fbase + w] = data_word(base, 4u * w - raw_start);
        }
        if (lid == 0u) {
            let bh = 1u | (BLOCK_SIZE << 3u);
            for (var w = 0u; w < PREFIX_WORDS; w++) {
                frames[fbase + w] = prefix_word(w, bh, 0u, 0u, base);
            }
            frame_len[b] = raw_start + BLOCK_SIZE;
        }
    }
}
