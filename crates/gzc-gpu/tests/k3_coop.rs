//! Targeted tests for the subgroup-cooperative K3 (speed phase S3, `k3_coop.wgsl`): scripted
//! `best[]` tables that put matches, rep repeats, catch-up runs and literal runs on the step-regime
//! and lane-window boundaries of the cooperative primitives. Expected values come from
//! `gzc_core::lazy::lazy_parse` / `reference::compress_block`. The same tests run against the
//! sequential K3 with `GZC_K3_MODE=seq`, and at other widths with `GZC_K3_W=8|16` (the boundaries
//! for W = 8, 16 and 32 are always all included).
use gzc_core::config::{BLOCK_SIZE, PARSE_END};
use gzc_core::lazy::lazy_parse;
use gzc_core::params::{LVL3, LVL9, MatchParams, RUNG1, RUNG2};
use gzc_core::reference::{Match, compress_block, greedy_parse};
use gzc_core::seq::BlockOutput;
use gzc_gpu::compressor::{
    GpuParams, K3Mode, Kernels, compress_batch, parses_from_best, parses_from_best_unchecked, probe_lanes,
};
use gzc_gpu::context::GpuContext;

const WIDTHS: [usize; 3] = [8, 16, 32];
const CAP: u32 = gzc_core::config::MATCH_SEARCH_CAP as u32;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// A random block with a scripted best[] (no matches until planted).
struct Case {
    name: String,
    block: Vec<u8>,
    best: Vec<Match>,
}

impl Case {
    fn random(name: impl Into<String>, seed: u64) -> Self {
        let mut r = Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x5DEECE66D);
        let block = (0..BLOCK_SIZE).map(|_| r.next() as u8).collect();
        Case { name: name.into(), block, best: vec![Match::default(); BLOCK_SIZE] }
    }

    /// block[dst..dst+len] = block[dst-off..], byte by byte (overlapping copies repeat), then the
    /// byte after it made to differ from its source.
    fn repeat(&mut self, dst: usize, off: usize, len: usize) -> &mut Self {
        for i in dst..dst + len {
            self.block[i] = self.block[i - off];
        }
        self.differ(dst + len, off)
    }

    /// Makes block[p] != block[p - off] (when p is in the block).
    fn differ(&mut self, p: usize, off: usize) -> &mut Self {
        if p < BLOCK_SIZE && p >= off && self.block[p] == self.block[p - off] {
            self.block[p] ^= 0x5A;
        }
        self
    }

    /// A real repeat of `len` bytes at p from p - off, announced in best[p] as length `blen`.
    fn explicit(&mut self, p: usize, off: usize, len: usize, blen: u32) -> &mut Self {
        self.repeat(p, off, len);
        self.best[p] = Match { offset: off as u32, len: blen };
        self
    }

    /// Explicit matches every 100 bytes from 200 up to about `end` (the last one ends 100 to 216
    /// bytes before it), so the parse's anchor keeps up and the scan reaches `end` with step 1.
    fn chain_to(&mut self, end: usize) -> &mut Self {
        let mut p = 200;
        while p + 100 + 16 < end {
            self.explicit(p, 150, 8, 8);
            p += 100;
        }
        self
    }

    fn done(&mut self) -> Case {
        Case { name: std::mem::take(&mut self.name), block: std::mem::take(&mut self.block), best: std::mem::take(&mut self.best) }
    }
}

fn setup(m: MatchParams) -> (GpuContext, Kernels) {
    let ctx = GpuContext::new().expect("GPU required");
    let kernels = Kernels::new(&ctx, GpuParams { matching: m, emit_frames: false, huffman: false }).expect("Kernels::new");
    eprintln!("K3 mode {:?} for {m:?}", kernels.k3_mode());
    (ctx, kernels)
}

fn first_diff(got: &BlockOutput, want: &BlockOutput) -> String {
    if let Some(i) = got.sequences.iter().zip(&want.sequences).position(|(a, b)| a != b) {
        return format!("sequence {i}: gpu {:?} cpu {:?}", got.sequences[i], want.sequences[i]);
    }
    if got.sequences.len() != want.sequences.len() {
        return format!("sequence count gpu {} cpu {}", got.sequences.len(), want.sequences.len());
    }
    let i = got.literals.iter().zip(&want.literals).position(|(a, b)| a != b);
    format!("literals: first diff {i:?}, lengths gpu {} cpu {}", got.literals.len(), want.literals.len())
}

/// K3 on the scripted tables (in batches) must equal lazy_parse (greedy_parse for lazy 0) for
/// every case.
fn check(m: MatchParams, cases: &[Case]) {
    let (ctx, kernels) = setup(m);
    for chunk in cases.chunks(128) {
        let blocks: Vec<&[u8]> = chunk.iter().map(|c| c.block.as_slice()).collect();
        let bests: Vec<Vec<Match>> = chunk.iter().map(|c| c.best.clone()).collect();
        let got = parses_from_best(&ctx, &kernels, &blocks, &bests).expect("parses_from_best");
        for (c, got) in chunk.iter().zip(&got) {
            let want = if m.lazy == 0 { greedy_parse(&c.block, &c.best, &m) } else { lazy_parse(&c.block, &c.best, &m) };
            assert!(*got == want, "{} (lazy {}): K3 != CPU parse; {}", c.name, m.lazy, first_diff(got, &want));
        }
    }
    eprintln!("{} cases equal", cases.len());
}

fn dedup(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v.dedup();
    v
}

/// Skip-sequence positions from anchor `a` around the step-regime boundaries (steps 1..4) and
/// the lane-window boundaries for every W.
fn scan_positions(a: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (255..=259).chain(510..=516).chain(767..=771).chain(1023..=1028).collect();
    for w in WIDTHS {
        v.extend([w - 1, w, w + 1, 2 * w - 1, 2 * w, 2 * w + 1, 256 + 2 * (w - 1), 256 + 2 * w, 256 + 2 * w + 2]);
        v.extend([512 + 3 * w - 3, 512 + 3 * w, 768 + 4 * w, 768 + 4 * w + 1]);
    }
    dedup(v.into_iter().map(|x| a + x).collect())
}

/// T1: one planted explicit match (found iff it is on the skip sequence), and a planted 4-byte
/// rep repeat at cand + 1 after a first match set offset_1.
#[test]
fn scan_finds_exactly_the_skip_sequence() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let mut cases = Vec::new();
    for p in scan_positions(0) {
        cases.push(Case::random(format!("explicit@{p}"), p as u64).explicit(p, (17 + p % 5).min(p), 8, 8).done());
    }
    for p in scan_positions(28) {
        // Match at 20 (offset 13) ends at 28 = anchor, offset_1 = 13; rep repeat probed at p + 1.
        let mut c = Case::random(format!("rep@{p}+1"), 1000 + p as u64);
        c.explicit(20, 13, 8, 8).repeat(p + 1, 13, 4);
        cases.push(c.done());
    }
    check(LVL9, &cases);
    check(RUNG2, &cases);
    check(RUNG1, &cases);
    check(LVL3, &cases);
}

/// A hit found by a scan with step 2 or 3 (the scan lanes are not consecutive positions and must
/// not serve as the deferral window), then a longer match at P + 1 or P + 2 the deferral takes.
#[test]
fn deferral_after_a_wide_step_scan() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let mut cases = Vec::new();
    for p in [258usize, 260, 300, 330, 512, 515, 600, 702] {
        for d in [1usize, 2] {
            let mut c = Case::random(format!("hit@{p} better@+{d}"), (p * 10 + d) as u64);
            c.explicit(p, 50, 5, 5).explicit(p + d, 90, 24, 24);
            cases.push(c.done());
        }
    }
    check(LVL9, &cases);
    check(RUNG2, &cases);
    check(RUNG1, &cases);
    check(LVL3, &cases);
}

/// best[] entries shorter than their real repeat: the repeat goes on at the new anchor, where the
/// greedy parse must not test the rep offset (it only does for p > anchor).
#[test]
fn match_shorter_than_its_repeat() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let mut cases = Vec::new();
    for blen in [4u32, 5, 6, 8, 12] {
        for real in [blen as usize + 4, blen as usize + 5, 40] {
            let mut c = Case::random(format!("best len {blen}, repeat {real}"), (blen as u64) * 100 + real as u64);
            c.repeat(500, 37, real);
            c.best[500] = Match { offset: 37, len: blen };
            cases.push(c.done());
        }
    }
    check(LVL9, &cases);
    check(RUNG2, &cases);
    check(RUNG1, &cases);
    check(LVL3, &cases);
}

/// A scan that restarts mid-regime: a capped best[] entry on the skip sequence whose real repeat
/// is shorter than 4 bytes (0 or 2) sends the parse on by one step without a store, so the next
/// scan window starts at an odd place and the step-2 -> step-3 boundary (512 from anchor 0) falls
/// inside a window. Planted matches before and after the boundary must be found exactly when the
/// oracle visits them.
#[test]
fn scan_restarts_mid_regime() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let mut cases = Vec::new();
    for restart in [300usize, 302, 450, 480, 494, 496, 508, 510] {
        for real in [0usize, 2] {
            for target in [511usize, 512, 513, 514, 515, 516, 518, 521] {
                let mut c = Case::random(format!("restart@{restart} real {real} target@{target}"), (restart * 1000 + real * 100 + target) as u64);
                c.repeat(restart, 70, real);
                c.best[restart] = Match { offset: 70, len: CAP };
                c.explicit(target, 45, 8, 8);
                cases.push((real, c.done()));
            }
        }
    }
    let all: Vec<Case> = cases.iter().map(|(_, c)| Case { name: c.name.clone(), block: c.block.clone(), best: c.best.clone() }).collect();
    check(LVL9, &all);
    check(RUNG2, &all);
    // The greedy oracle emits a capped entry's extension whatever its length, and never advances
    // on a 0-byte one (outside its domain; the GPU kernels skip it): greedy gets the 2-byte cases,
    // which it stores as a match (so the restart there is not mid-regime, only the lazy one is).
    let two: Vec<Case> = cases.into_iter().filter(|(real, _)| *real == 2).map(|(_, c)| c).collect();
    check(RUNG1, &two);
    check(LVL3, &two);
}

/// A best[] entry that claims a match past the block end (K2 never writes one) puts the anchor
/// past BLOCK_SIZE. K3 must still terminate (no GPU hang; the device stays usable). Both K3s
/// count no trailing literals then (since S4 the sequential one no longer wraps its count).
#[test]
fn anchor_past_block_end_terminates() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let mut cases = Vec::new();
    for (ip, len) in [(PARSE_END - 1, 40u32), (PARSE_END - 30, 63), (BLOCK_SIZE - 60, 62)] {
        let mut c = Case::random(format!("match at {ip} len {len}"), ip as u64);
        c.chain_to(ip).best[ip] = Match { offset: 20, len };
        cases.push(c.done());
    }
    for m in [LVL9, RUNG2, RUNG1, LVL3] {
        let (ctx, kernels) = setup(m);
        let blocks: Vec<&[u8]> = cases.iter().map(|c| c.block.as_slice()).collect();
        let bests: Vec<Vec<Match>> = cases.iter().map(|c| c.best.clone()).collect();
        let got = parses_from_best_unchecked(&ctx, &kernels, &blocks, &bests);
        let got = got.unwrap_or_else(|e| panic!("{:?} K3 on a match past the block end: {e}", kernels.k3_mode()));
        for (c, got) in cases.iter().zip(&got) {
            let last = got.sequences.last().unwrap_or_else(|| panic!("{}: no sequence", c.name));
            let covered: usize = got.sequences.iter().map(|q| (q.lit_len + q.match_len) as usize).sum();
            assert!(covered > BLOCK_SIZE, "{}: the last match should run past the block end", c.name);
            assert_eq!(got.literals.len(), got.sequences.iter().map(|q| q.lit_len as usize).sum::<usize>(), "{}", c.name);
            assert!(last.match_len >= 4, "{}", c.name);
        }
        // The device survived: a normal scripted case still parses exactly.
        let normal = [Case::random("after", 1).explicit(1000, 33, 8, 8).done()];
        let blocks: Vec<&[u8]> = normal.iter().map(|c| c.block.as_slice()).collect();
        let bests: Vec<Vec<Match>> = normal.iter().map(|c| c.best.clone()).collect();
        let after = parses_from_best(&ctx, &kernels, &blocks, &bests).expect("device still usable");
        let want = if m.lazy == 0 { greedy_parse(&normal[0].block, &normal[0].best, &m) } else { lazy_parse(&normal[0].block, &normal[0].best, &m) };
        assert!(after[0] == want);
    }
}

/// Literal words must stay inside their block's region: a block that is literals only (its region
/// completely full) next to blocks whose first literal word is written early (a byte, then a run
/// to the end: one fast long match). A store one word past a block's literals would land on the
/// neighbour's first word.
#[test]
fn full_literal_region_next_to_other_blocks() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let mut blocks: Vec<(String, Vec<u8>)> = Vec::new();
    let mut r = Lcg(99);
    for i in 0..8 {
        let lits: Vec<u8> = (0..BLOCK_SIZE).map(|_| r.next() as u8).collect();
        blocks.push((format!("all literals {i}"), lits));
        let mut run = vec![(i * 37 + 5) as u8; BLOCK_SIZE];
        for (j, x) in run.iter_mut().take(1 + i % 5).enumerate() {
            *x = (j * 91 + i) as u8;
        }
        blocks.push((format!("short head + run {i}"), run));
    }
    for m in [LVL9, RUNG2, RUNG1, LVL3] {
        let (ctx, kernels) = setup(m);
        let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
        let got = compress_batch(&ctx, &kernels, &refs).expect("compress_batch");
        for ((name, b), got) in blocks.iter().zip(&got) {
            let want = compress_block(b, m);
            assert!(*got == want, "{name}: GPU != compress_block; {}", first_diff(got, &want));
        }
    }
}

/// T1, MIN_MATCH 6: planted best lengths 4 and 5 are no match, 6 is.
#[test]
fn scan_respects_min_match() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let mut cases = Vec::new();
    for p in scan_positions(0).into_iter().step_by(3) {
        for len in [4u32, 5, 6] {
            cases.push(Case::random(format!("len{len}@{p}"), p as u64 * 7 + len as u64).explicit(p, 9.min(p), len as usize, len).done());
        }
    }
    check(MatchParams { min_match: 6, ..RUNG2 }, &cases);
    check(MatchParams { min_match: 6, ..RUNG1 }, &cases);
    check(MatchParams { min_match: 8, ..RUNG1 }, &cases);
}

/// T2: scans and matches at PARSE_END, with poisoned best[] entries at and past it.
#[test]
fn scan_and_matches_at_parse_end() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let poison = |c: &mut Case| {
        for p in PARSE_END..BLOCK_SIZE - 4 {
            c.best[p] = Match { offset: 1, len: 4 };
        }
    };
    let mut cases = vec![Case::random("all literals", 1).done()];
    for d in [1usize, 2, 3] {
        let mut c = Case::random(format!("explicit@PARSE_END-{d}"), 10 + d as u64);
        c.chain_to(PARSE_END - d).explicit(PARSE_END - d, 33, 8.min(BLOCK_SIZE - PARSE_END + d), 8);
        poison(&mut c);
        cases.push(c.done());
    }
    for w in WIDTHS {
        for back in [w + 3, w + 1, w, 1] {
            for tail in [false, true] {
                let e = PARSE_END - back;
                let mut c = Case::random(format!("anchor PARSE_END-{back} tail {tail}"), (w * 100 + back) as u64);
                c.chain_to(e - 8).explicit(e - 8, 21, 8, 8);
                if tail {
                    c.explicit(PARSE_END - 2, 7, 8, 8);
                }
                poison(&mut c);
                cases.push(c.done());
            }
        }
    }
    // A match ending at PARSE_END - 1, then a rep1 repeat probed at PARSE_END.
    let mut c = Case::random("rep at PARSE_END", 77);
    c.chain_to(PARSE_END - 9).explicit(PARSE_END - 9, 40, 8, 8).repeat(PARSE_END, 40, 4);
    cases.push(c.done());
    // A match ending exactly at PARSE_END, then offset_2 repeats there (immediate loop at PARSE_END).
    let mut c = Case::random("immediate at PARSE_END", 78);
    c.chain_to(PARSE_END - 200).explicit(PARSE_END - 190, 300, 10, 10);
    c.explicit(PARSE_END - 10, 50, 10, 10).repeat(PARSE_END, 300, 8);
    cases.push(c.done());
    check(LVL9, &cases);
    check(RUNG2, &cases);
    check(RUNG1, &cases);
    check(LVL3, &cases);
}

/// T3: capped best[] entries extended to their true length, for every (p & 3, q & 3), lengths
/// around the lane-window multiples and up to (and just short of) the block end.
#[test]
fn capped_extension_geometry() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let mut lens: Vec<usize> = vec![2, 5, CAP as usize, 300, 1000];
    for w in WIDTHS {
        lens.extend([4 * w - 1, 4 * w, 4 * w + 1, 4 * w + 3, 8 * w, 8 * w + 5]);
    }
    let lens = dedup(lens);
    let mut cases = Vec::new();
    for pa in 0..4 {
        for qa in 0..4 {
            let p = 4096 + pa;
            let q = 2048 + qa;
            for &len in &lens {
                cases.push(Case::random(format!("p&3={pa} q&3={qa} len {len}"), (pa * 4 + qa) as u64 * 1000 + len as u64)
                    .explicit(p, p - q, len, CAP)
                    .done());
            }
            let p = BLOCK_SIZE - 300 + pa;
            let q = BLOCK_SIZE - 3000 + qa;
            for short in 0..4 {
                let len = BLOCK_SIZE - p - short;
                cases.push(Case::random(format!("to end-{short} p&3={pa} q&3={qa}"), 50_000 + (pa * 16 + qa * 4 + short) as u64)
                    .explicit(p, p - q, len, CAP)
                    .done());
            }
        }
    }
    check(LVL9, &cases);
    check(RUNG1, &cases);
}

/// T4: rep1 repeats at ip + 1 of lengths around the lane windows; immediate offset_2 chains.
#[test]
fn rep_lengths_and_immediate_chains() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let mut lens: Vec<usize> = vec![4, 5, 64, 200];
    for w in WIDTHS {
        lens.extend([4 * w - 1, 4 * w, 4 * w + 1]);
    }
    let lens = dedup(lens);
    let mut cases = Vec::new();
    for &len in &lens {
        // Match at 100 (offset 50) -> offset_1 = 50; rep repeat at 301.
        let mut c = Case::random(format!("rep1 len {len}"), len as u64);
        c.explicit(100, 50, 10, 10).repeat(301, 50, len);
        cases.push(c.done());
    }
    for w in WIDTHS {
        for (l1, l2) in [(8, 8), (4 * w, 8), (8, 4 * w), (4 * w, 4 * w + 1)] {
            // offset_1 = 70, offset_2 = 300 after the second match; then three immediate repeats
            // alternating 300, 70, 300 (each hit swaps them), the last one to the block end.
            let mut c = Case::random(format!("immediate chain w{w} {l1} {l2}"), (w * 1000 + l1 * 10 + l2) as u64);
            c.explicit(1000, 300, 10, 10).explicit(2000, 70, 12, 12);
            let e = 2012;
            c.repeat(e, 300, l1).repeat(e + l1, 70, l2);
            let e3 = e + l1 + l2;
            c.repeat(e3, 300, BLOCK_SIZE - e3);
            cases.push(c.done());
        }
    }
    check(LVL9, &cases);
    check(RUNG2, &cases);
    check(RUNG1, &cases);
    check(LVL3, &cases);
}

/// T5: catch-up runs across the lane windows, bounded by the anchor, by position 0 of the
/// source, and stopped by a mismatch in the middle.
#[test]
fn catch_up_bounds() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let mut cases = Vec::new();
    for w in WIDTHS {
        for b in [1, w - 1, w, w + 1, 2 * w + 3] {
            // best[p] announces 8 bytes; the real repeat starts b bytes earlier.
            let p = 5000;
            let mut c = Case::random(format!("catch-up {b} (w{w})"), (w * 100 + b) as u64);
            c.repeat(p - b, 700, b + 8).differ(p - b - 1, 700);
            c.best[p] = Match { offset: 700, len: 8 };
            cases.push(c.done());
        }
        // An earlier match ending m bytes before p (the anchor) bounds a run that would go on.
        for m in [w - 1, w, w + 1] {
            let p = 5000;
            let mut c = Case::random(format!("catch-up to anchor -{m} (w{w})"), (w * 10_000 + m) as u64);
            c.explicit(p - m - 9, 400, 9, 9);
            c.repeat(p - m, 700, m + 8);
            let a = p - m - 1;
            c.block[a] = c.block[a - 700];
            c.best[p] = Match { offset: 700, len: 8 };
            cases.push(c.done());
        }
    }
    // Bounded by start > off: the source reaches position 0.
    for (p, off) in [(40usize, 30usize), (40, 39), (70, 40)] {
        let mut c = Case::random(format!("catch-up to source 0 p {p} off {off}"), (p * 100 + off) as u64);
        c.repeat(off, off, p - off + 8);
        c.best[p] = Match { offset: off as u32, len: 8 };
        cases.push(c.done());
    }
    // Matches at s-1, s-2, mismatch at s-3, matches at s-4..s-8: moves exactly 2.
    let mut c = Case::random("catch-up stops at a mismatch", 999);
    let p = 3000;
    c.repeat(p - 8, 500, 16);
    c.block[p - 3] = c.block[p - 3 - 500] ^ 0x11;
    c.best[p] = Match { offset: 500, len: 8 };
    cases.push(c.done());
    check(LVL9, &cases);
    check(RUNG2, &cases);
    check(RUNG1, &cases);
    check(LVL3, &cases);
}

/// T6: literal runs of every length class at every accumulator phase.
#[test]
fn literal_packing() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let mut gaps: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 7];
    for w in WIDTHS {
        gaps.extend([4 * w - 1, 4 * w, 4 * w + 1, 8 * w + 3]);
    }
    let gaps = dedup(gaps);
    let mut cases = Vec::new();
    for lead in 0..4 {
        for order in 0..3 {
            let mut c = Case::random(format!("lits lead {lead} order {order}"), (lead * 10 + order) as u64);
            let mut g = gaps.clone();
            match order {
                1 => g.reverse(),
                2 => g.rotate_left(gaps.len() / 2),
                _ => {}
            }
            let mut pos = 1 + lead;
            for (i, gap) in g.iter().cycle().take(3 * g.len()).enumerate() {
                pos += gap;
                let off = 40 + (i * 7) % 300;
                if pos + 6 >= PARSE_END || pos < off {
                    break;
                }
                c.explicit(pos, off, 6 + i % 3, (6 + i % 3) as u32);
                pos += 6 + i % 3;
            }
            cases.push(c.done());
        }
    }
    check(LVL9, &cases);
    check(RUNG2, &cases);
    check(RUNG1, &cases);
    check(LVL3, &cases);
}

/// T8: flat and periodic blocks through K1/K2/K3 (long extensions to the block end, every lane
/// boundary, the byte tail at the end).
#[test]
fn flat_and_periodic_blocks() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let mut blocks: Vec<(String, Vec<u8>)> = vec![("zeros".into(), vec![0; BLOCK_SIZE])];
    let mut r = Lcg(7);
    for period in [1usize, 2, 3, 4, 8, 16] {
        let pat: Vec<u8> = (0..period).map(|_| r.next() as u8).collect();
        for phase in 0..period.min(4) {
            let mut b = vec![0u8; BLOCK_SIZE];
            for (i, x) in b.iter_mut().enumerate() {
                *x = if i < phase { r.next() as u8 } else { pat[(i - phase) % period] };
            }
            blocks.push((format!("period {period} phase {phase}"), b));
        }
    }
    for d in 1..=5 {
        let mut b = vec![0u8; BLOCK_SIZE];
        b[BLOCK_SIZE - d] = 1;
        blocks.push((format!("zeros, byte at end-{d}"), b));
    }
    for m in [LVL9, RUNG2, RUNG1, LVL3] {
        let (ctx, kernels) = setup(m);
        let refs: Vec<&[u8]> = blocks.iter().map(|(_, b)| b.as_slice()).collect();
        let got = compress_batch(&ctx, &kernels, &refs).expect("compress_batch");
        for ((name, b), got) in blocks.iter().zip(&got) {
            let want = compress_block(b, m);
            assert!(*got == want, "{name}: GPU != compress_block; {}", first_diff(got, &want));
        }
    }
}

/// T9: the lane probe accepts every W up to the minimum subgroup size and rejects a workgroup
/// of two subgroups; every preset uses the cooperative K3 by default when subgroups exist.
#[test]
fn probe_and_mode_selection() {
    let _gpu = gzc_gpu::test_support::gpu_test_slot();
    let ctx = GpuContext::new().expect("GPU required");
    if !ctx.subgroups {
        eprintln!("no subgroup support: sequential K3 only");
        return;
    }
    let min = ctx.adapter_info.subgroup_min_size;
    for w in [4u32, 8, 16, 32, 64].into_iter().filter(|&w| w <= min) {
        assert!(probe_lanes(&ctx, w, 1).unwrap(), "probe failed for W = {w} (min subgroup size {min})");
    }
    if 2 * ctx.adapter_info.subgroup_max_size <= 64 {
        let w = 2 * ctx.adapter_info.subgroup_max_size;
        assert!(!probe_lanes(&ctx, w, 1).unwrap(), "probe accepted two subgroups (W = {w})");
    }
    // BPW = 2 (stage E, opt-in) is only offered when every subgroup has exactly W lanes.
    let max = ctx.adapter_info.subgroup_max_size;
    if min == max && (4..=32).contains(&min) {
        assert!(probe_lanes(&ctx, min, 2).unwrap(), "probe failed for W = {min}, 2 blocks per workgroup");
    }
    let kernels = Kernels::new(&ctx, GpuParams { matching: LVL9, emit_frames: false, huffman: false }).unwrap();
    // The default W is the minimum subgroup size clamped to 8..=64; on a device with smaller
    // subgroups (a W of 8 spans two of them) the probe fails and K3 stays sequential.
    let default_w = {
        let w = min.clamp(8, 64);
        1 << (31 - w.leading_zeros())
    };
    let forced_w = std::env::var("GZC_K3_W").ok().map(|v| v.parse::<u32>().unwrap());
    let bpw = if std::env::var("GZC_K3_BPW").as_deref() == Ok("2") { 2 } else { 1 };
    let w = forced_w.unwrap_or(default_w);
    match std::env::var("GZC_K3_MODE").as_deref() {
        Ok("seq") => assert_eq!(kernels.k3_mode(), K3Mode::Seq),
        _ if probe_lanes(&ctx, w, bpw).unwrap() => assert_eq!(kernels.k3_mode(), K3Mode::Coop { w, bpw }),
        _ => assert_eq!(kernels.k3_mode(), K3Mode::Seq),
    }
    let greedy = Kernels::new(&ctx, GpuParams { matching: LVL3, emit_frames: false, huffman: false }).unwrap();
    assert_eq!(greedy.k3_mode(), kernels.k3_mode());
}
