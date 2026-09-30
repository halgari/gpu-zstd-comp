// K4: sequence entropy coding + block assembly. One workgroup of 64 threads per block turns K3's
// parse into the block's complete zstd frame, byte-identical to
// gzc_core::frame::write_frame(block, parse, FrameOptions { checksum: false, huffman: HUFFMAN }):
//   frame header, then one block that is
//   - RLE (every byte equal), else
//   - Compressed: literals section + sequences section (gzc_core::seqenc::
//     write_sequences_section_auto: per-stream predefined / RLE / computed FSE table), if the
//     content is smaller than BLOCK_SIZE, else
//   - Raw.
// The literals section is Raw (copied here from `lits`) unless HUFFMAN and K5 already wrote an RLE
// or Compressed section into the frame at byte HDR_LEN + 3: then frame_len[b] holds its length on
// entry (RAW_SECTION otherwise) and K4 keeps K5's bytes, writing only the headers below it and the
// sequences section after it.
// Threads cooperate on the RLE check, the code histograms, the literal copy and the raw copy;
// thread 0 makes the mode decisions, writes the table descriptions, builds the FSE tables in
// workgroup memory and runs the (inherently sequential) backward sequence encode.
//
// The host binds exactly n_blocks words of `frame_len`, so arrayLength(&frame_len) is the batch
// size. Outputs per block b:
//   frames[b*FRAME_WORDS ..]  frame bytes, packed little-endian (bytes past frame_len are junk)
//   frame_len[b]              frame length in bytes
// Prepended by the host: MAX_SEQS, FRAME_WORDS, HDR_LEN / HDR_W0..2 (frame header bytes), HUFFMAN,
// RAW_SECTION and the TAB_* offsets into `tab` (code tables, FRAC, predefined distributions from
// gzc_core).

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> seqs: array<u32>;
@group(0) @binding(2) var<storage, read> lits: array<u32>;
@group(0) @binding(3) var<storage, read> counts: array<u32>;
@group(0) @binding(4) var<storage, read> tab: array<u32>;
@group(0) @binding(5) var<storage, read_write> frames: array<u32>;
@group(0) @binding(6) var<storage, read_write> frame_len: array<u32>;

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
var<workgroup> spread: array<u32, 512>;
var<workgroup> cumul: array<u32, 54>;
var<workgroup> not_rle: atomic<u32>;
var<workgroup> rle_flag: u32;
var<workgroup> raw_flag: u32;

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

// == FseCTable::from_normalized for stream k, from norm[h_off(k) ..] (dflt: k's predefined
// distribution), into st[s_off(k) ..] and tt_*[h_off(k) ..].
fn build_table(k: u32, len: u32, tl: u32, dflt: bool) {
    let h = h_off(k);
    let so = s_off(k);
    let size = 1u << tl;
    let mask = size - 1u;
    let stp = (size >> 1u) + (size >> 3u) + 3u;
    var high = size - 1u;
    cumul[0] = 0u;
    for (var s = 0u; s < len; s++) {
        var n: i32;
        if (dflt) { n = def_norm(k, s); } else { n = norm[h + s]; }
        if (dflt) { norm[h + s] = n; }
        if (n == -1) {
            cumul[s + 1u] = cumul[s] + 1u;
            spread[high] = s;
            high -= 1u;
        } else {
            cumul[s + 1u] = cumul[s] + u32(n);
        }
    }
    var pos = 0u;
    for (var s = 0u; s < len; s++) {
        let n = norm[h + s];
        for (var i = 0; i < n; i++) {
            spread[pos] = s;
            pos = (pos + stp) & mask;
            while (pos > high) { pos = (pos + stp) & mask; }
        }
    }
    for (var u = 0u; u < size; u++) {
        let s = spread[u];
        st[so + cumul[s]] = size + u;
        cumul[s] += 1u;
    }
    var total = 0;
    for (var s = 0u; s < len; s++) {
        let n = norm[h + s];
        if (n == 0) {
            tt_dfs[h + s] = 0;
            tt_dnb[h + s] = ((tl + 1u) << 16u) - size;
        } else if (n == -1 || n == 1) {
            tt_dfs[h + s] = total - 1;
            tt_dnb[h + s] = (tl << 16u) - size;
            total += 1;
        } else {
            let mbo = tl - highbit(u32(n) - 1u);
            tt_dfs[h + s] = total - n;
            tt_dnb[h + s] = (mbo << 16u) - (u32(n) << mbo);
            total += n;
        }
    }
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

// == FseState::encode; returns the next state.
fn fse_encode(k: u32, state: u32, sym: u32) -> u32 {
    let nb = (state + tt_dnb[h_off(k) + sym]) >> 16u;
    put(state, nb);
    return st[s_off(k) + u32(i32(state >> nb) + tt_dfs[h_off(k) + sym])];
}

// Extra bits of one sequence in decoder order (LL, ML, OF).
fn put_extras(ll: u32, llc: u32, ml: u32, mlc: u32, ob: u32, ofc: u32) {
    put(ll - tab[TAB_LL_BASE + llc], tab[TAB_LL_BITS + llc]);
    put(ml - tab[TAB_ML_BASE + mlc], tab[TAB_ML_BITS + mlc]);
    put(ob - (1u << ofc), ofc);
}

// ---- frame prefix bytes ----

fn hdr_byte(k: u32) -> u32 {
    let w = select(select(HDR_W2, HDR_W1, k < 8u), HDR_W0, k < 4u);
    return (w >> ((k & 3u) * 8u)) & 0xFFu;
}

// Byte k of a frame whose block header is `bh` (3 bytes), followed by `extra_len` bytes of `extra`
// (the literals header), then bytes of the source at word `src` (literals or block data).
fn prefix_byte(k: u32, bh: u32, extra: u32, extra_len: u32, src: u32, from_data: bool) -> u32 {
    if (k < HDR_LEN) { return hdr_byte(k); }
    let j = k - HDR_LEN;
    if (j < 3u) { return (bh >> (j * 8u)) & 0xFFu; }
    if (j < 3u + extra_len) { return (extra >> ((j - 3u) * 8u)) & 0xFFu; }
    let i = j - 3u - extra_len;
    var w: u32;
    if (from_data) { w = data[src + (i >> 2u)]; } else { w = lits[src + (i >> 2u)]; }
    return (w >> ((i & 3u) * 8u)) & 0xFFu;
}

fn prefix_word(w: u32, bh: u32, extra: u32, extra_len: u32, src: u32, from_data: bool) -> u32 {
    var v = 0u;
    for (var i = 0u; i < 4u; i++) {
        v |= prefix_byte(4u * w + i, bh, extra, extra_len, src, from_data) << (8u * i);
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
            v = (v & ~(0xFFu << (8u * i))) | (prefix_byte(k, bh, 0u, 0u, 0u, false) << (8u * i));
        }
    }
    return v;
}

// Funnel-shifted word of `src` words starting at byte offset i (bytes i .. i+3).
fn lit_word(src: u32, i: u32) -> u32 {
    let a = src + (i >> 2u);
    let sh = (i & 3u) * 8u;
    if (sh == 0u) { return lits[a]; }
    return (lits[a] >> sh) | (lits[a + 1u] << (32u - sh));
}
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
    let n_lit = counts[2u * b + 1u];
    let sbase = b * MAX_SEQS * 3u;
    let lbase = b * (BLOCK_SIZE / 4u);
    // K5's literals section length, read before thread 0 overwrites frame_len[b].
    var section = RAW_SECTION;
    if (HUFFMAN) { section = frame_len[b]; }
    let in_frame = section != RAW_SECTION;

    // ---- histograms and the RLE-block check ----
    for (var i = lid; i < H_ALL; i += WG) { atomicStore(&hist[i], 0u); }
    if (lid == 0u) { atomicStore(&not_rle, 0u); }
    workgroupBarrier();
    let first = data[base] & 0xFFu;
    let first4 = first * 0x01010101u;
    var diff = 0u;
    for (var w = lid; w < BLOCK_SIZE / 4u; w += WG) { diff |= data[base + w] ^ first4; }
    if (diff != 0u) { atomicOr(&not_rle, 1u); }
    for (var i = lid; i < n_seq; i += WG) {
        let s = sbase + i * 3u;
        atomicAdd(&hist[H_LL + ll_code(seqs[s])], 1u);
        atomicAdd(&hist[H_ML + ml_code(seqs[s + 1u])], 1u);
        atomicAdd(&hist[H_OF + highbit(seqs[s + 2u])], 1u);
    }
    workgroupBarrier();
    if (lid == 0u) { rle_flag = atomicLoad(&not_rle); }
    if (workgroupUniformLoad(&rle_flag) == 0u) {
        if (lid == 0u) {
            let bh = 1u | (1u << 1u) | (BLOCK_SIZE << 3u);
            for (var w = 0u; w < PREFIX_WORDS; w++) {
                frames[fbase + w] = prefix_word(w, bh, first, 1u, 0u, false);
            }
            frame_len[b] = HDR_LEN + 4u;
        }
        return;
    }

    // ---- raw literals section: copied cooperatively (words fully below the sequences section) ----
    let lit_hdr_len = select(select(3u, 2u, n_lit < 4096u), 1u, n_lit < 32u);
    let lit_hdr = select(select(0xCu | (n_lit << 4u), 0x4u | (n_lit << 4u), n_lit < 4096u), n_lit << 3u, n_lit < 32u);
    let lit_start = HDR_LEN + 3u + lit_hdr_len; // frame byte of literal 0
    // frame byte of the sequences section
    let seq_start = select(lit_start + n_lit, HDR_LEN + 3u + section, in_frame);
    if (!in_frame) {
        for (var w = PREFIX_WORDS + lid; w < seq_start / 4u; w += WG) {
            frames[fbase + w] = lit_word(lbase, 4u * w - lit_start);
        }
    }

    // ---- sequences section (thread 0) ----
    if (lid == 0u) {
        acc = 0u;
        cnt = (seq_start & 3u) * 8u;
        wpos = seq_start / 4u;
        if (in_frame && cnt > 0u) {
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
            var tlog: array<u32, 3>;
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
                } else if (mode[k] == MODE_COMPRESSED) {
                    ncount(h_off(k), len[k], tlog[k], true);
                    build_table(k, len[k], tlog[k], false);
                } else {
                    tlog[k] = def_log(k);
                    build_table(k, def_len(k), tlog[k], true);
                }
                if (mode[k] == MODE_RLE) { tlog[k] = 0u; }
            }

            // Backward encode (== write_sequences_section_with).
            var s = sbase + (n_seq - 1u) * 3u;
            var ll = seqs[s];
            var ml = seqs[s + 1u];
            var ob = seqs[s + 2u];
            var llc = ll_code(ll);
            var mlc = ml_code(ml);
            var ofc = highbit(ob);
            var ml_state = fse_init(2u, mlc);
            var of_state = fse_init(1u, ofc);
            var ll_state = fse_init(0u, llc);
            put_extras(ll, llc, ml, mlc, ob, ofc);
            for (var i = n_seq - 1u; i > 0u; i--) {
                if (wpos >= FRAME_WORDS) { break; } // cannot fit: the block goes out Raw
                s -= 3u;
                ll = seqs[s];
                ml = seqs[s + 1u];
                ob = seqs[s + 2u];
                llc = ll_code(ll);
                mlc = ml_code(ml);
                ofc = highbit(ob);
                of_state = fse_encode(1u, of_state, ofc);
                ml_state = fse_encode(2u, ml_state, mlc);
                ll_state = fse_encode(0u, ll_state, llc);
                put_extras(ll, llc, ml, mlc, ob, ofc);
            }
            put(ml_state, tlog[2]);
            put(of_state, tlog[1]);
            put(ll_state, tlog[0]);
            put(1u, 1u); // end mark
        }
        let end = put_end();
        put_flush();
        let content = end - (HDR_LEN + 3u);
        if (wpos >= FRAME_WORDS || content >= BLOCK_SIZE) {
            raw_flag = 1u;
        } else {
            raw_flag = 0u;
            let bh = 1u | (2u << 1u) | (content << 3u);
            let first_seq_word = seq_start / 4u;
            for (var w = 0u; w < min(PREFIX_WORDS, first_seq_word); w++) {
                if (in_frame) {
                    frames[fbase + w] = prefix_word_over_section(w, bh);
                } else {
                    frames[fbase + w] = prefix_word(w, bh, lit_hdr, lit_hdr_len, lbase, false);
                }
            }
            // The first sequences-section word starts with prefix / literal bytes.
            let keep = (seq_start & 3u) * 8u;
            if (keep > 0u) {
                var low: u32;
                if (in_frame) {
                    low = prefix_word_over_section(first_seq_word, bh);
                } else {
                    low = prefix_word(first_seq_word, bh, lit_hdr, lit_hdr_len, lbase, false);
                }
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
                frames[fbase + w] = prefix_word(w, bh, 0u, 0u, base, true);
            }
            frame_len[b] = raw_start + BLOCK_SIZE;
        }
    }
}
