// a12-metal microbenchmark for Apple GPUs. Build and run on the Mac:
//   swiftc -O applebench.swift -o applebench && ./applebench
// Part A: integer op throughput (ops per lane-cycle relative to IADD) for the ops our kernels lean on.
// Part B: K2-style chain walk + match_len over 64 KiB blocks, packed words (today's layout, funnel
//         shifts) vs a word-per-position "expanded" layout (no shifts); outputs must match.
import Foundation
import Metal

let src = """
#include <metal_stdlib>
using namespace metal;

// ---------- Part A ----------
#define ITERS 2048u
#define ALU(NAME, OP) \\
kernel void NAME(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]], \\
                 uint tid [[thread_position_in_grid]]) { \\
  uint a = in[tid & 1023u], b = in[(tid + 1u) & 1023u], c = in[(tid + 2u) & 1023u], d = in[(tid + 3u) & 1023u]; \\
  uint y = in[(tid + 5u) & 1023u] | 1u; uint s = in[(tid + 7u) & 1023u] & 31u; \\
  for (uint i = 0u; i < ITERS; i++) { \\
    a = OP(a, y, s); b = OP(b, y, s); c = OP(c, y, s); d = OP(d, y, s); \\
    a = OP(a, y, s); b = OP(b, y, s); c = OP(c, y, s); d = OP(d, y, s); \\
    s = (s + 1u) & 31u; \\
  } \\
  out[tid] = a ^ b ^ c ^ d; }

inline uint op_iadd(uint x, uint y, uint s)        { return (x + s) ^ y; }
inline uint op_shl_dyn(uint x, uint y, uint s)     { return (x << s) ^ y; }
inline uint op_shr_dyn(uint x, uint y, uint s)     { return (x >> s) ^ y; }
inline uint op_shr_const(uint x, uint y, uint s)   { return (x >> 16u) ^ y ^ s; }
inline uint op_and16(uint x, uint y, uint s)       { return (x & 0xFFFFu) ^ y ^ s; }
// naga's spelling of K2's funnel (two shifts on the high word)
inline uint op_funnel_naga(uint x, uint y, uint s) { return (x >> s) | ((y << (31u - s)) << 1u); }
// a C spelling the compiler might map to one extract instruction
inline uint op_funnel_c(uint x, uint y, uint s)    { return s == 0u ? x : ((x >> s) | (y << (32u - s))); }
inline uint op_funnel_64(uint x, uint y, uint s)   { return uint(((ulong(y) << 32) | ulong(x)) >> s); }
inline uint op_extract(uint x, uint y, uint s)     { return extract_bits(x, s & 15u, 8u) ^ y; }
inline uint op_ctz(uint x, uint y, uint s)         { return (x ^ y) + ctz(x | 1u); }
inline uint op_clz(uint x, uint y, uint s)         { return (x ^ y) + clz(x | 1u); }
inline uint op_popc(uint x, uint y, uint s)        { return (x ^ y) + popcount(x); }
inline uint op_imul(uint x, uint y, uint s)        { return (x * y) ^ s; }
inline uint op_mulhi(uint x, uint y, uint s)       { return mulhi(x, y) ^ s; }
inline uint op_mod33(uint x, uint y, uint s)       { return (x % 33u) + y + s; }
inline uint op_dec33(uint x, uint y, uint s)       { uint t = x + y + s; return t == 0u ? 32u : t - 1u; }
inline uint op_min(uint x, uint y, uint s)         { return min(x, y) + s; }
inline uint op_u16(uint x, uint y, uint s)         { return uint(ushort(x) + ushort(y)) ^ s; }

ALU(k_iadd, op_iadd)
ALU(k_shl_dyn, op_shl_dyn)
ALU(k_shr_dyn, op_shr_dyn)
ALU(k_shr_const, op_shr_const)
ALU(k_and16, op_and16)
ALU(k_funnel_naga, op_funnel_naga)
ALU(k_funnel_c, op_funnel_c)
ALU(k_funnel_64, op_funnel_64)
ALU(k_extract, op_extract)
ALU(k_ctz, op_ctz)
ALU(k_clz, op_clz)
ALU(k_popc, op_popc)
ALU(k_imul, op_imul)
ALU(k_mulhi, op_mulhi)
ALU(k_mod33, op_mod33)
ALU(k_dec33, op_dec33)
ALU(k_min, op_min)
ALU(k_u16, op_u16)

// ---------- Part B ----------
constant uint BS = 65536u;
constant uint CAP = 64u;
constant uint DEPTH = 32u;
constant uint NONE = 0xFFFFFFFFu;

inline uint funnel(uint lo, uint hi, uint sh) { return (lo >> sh) | ((hi << (31u - sh)) << 1u); }
inline uint ld_packed(device const uint* d, uint b, uint off) {
  uint w = b * (BS / 4u) + (off >> 2u); return funnel(d[w], d[w + 1u], (off & 3u) * 8u);
}
inline uint ld_exp(device const uint* d, uint b, uint off) { return d[b * BS + off]; }

#define K2(NAME, LD) \\
kernel void NAME(device const uint* data [[buffer(0)]], device const uint* pred [[buffer(1)]], \\
                 device uint* best [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) { \\
  uint p = gid.x, b = gid.y, o = b * BS + p; \\
  if (p >= BS - 8u) { best[o] = 0u; return; } \\
  uint mx = min(BS - p, CAP); \\
  uint p0 = LD(data, b, p), p1 = LD(data, b, p + 4u); \\
  uint bl = 0u, bq = 0u, q = pred[o]; \\
  for (uint dd = 0u; dd < DEPTH && q != NONE; dd++) { \\
    uint qn = pred[b * BS + q]; \\
    uint n; uint x = p0 ^ LD(data, b, q); \\
    if (x != 0u) { n = ctz(x) >> 3u; } \\
    else { x = p1 ^ LD(data, b, q + 4u); \\
      if (x != 0u) { n = 4u + (ctz(x) >> 3u); } \\
      else { n = 8u; \\
        while (n < mx) { x = LD(data, b, p + n) ^ LD(data, b, q + n); uint left = mx - n; \\
          if (left < 4u) { x &= (1u << (left * 8u)) - 1u; } \\
          if (x != 0u) { n += ctz(x) >> 3u; break; } n += 4u; } \\
        n = min(n, mx); } } \\
    if (n > bl) { bl = n; bq = q; if (n == mx) { break; } } \\
    q = qn; \\
  } \\
  best[o] = (bl << 16u) | (p - bq); }

K2(k2_packed, ld_packed)
K2(k2_expanded, ld_exp)
"""

guard let dev = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
print("device: \(dev.name)")
let queue = dev.makeCommandQueue()!
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: nil) } catch { fatalError("compile: \(error)") }
func pipe(_ n: String) -> MTLComputePipelineState {
  let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!)
  return p
}
func run(_ p: MTLComputePipelineState, _ bufs: [MTLBuffer], _ grid: MTLSize, _ tg: Int) -> Double {
  let cb = queue.makeCommandBuffer()!
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(p)
  for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
  e.dispatchThreads(grid, threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
  e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  return cb.gpuEndTime - cb.gpuStartTime
}

// ---------- Part A ----------
let nThreads = 1 << 20
var seed: UInt32 = 12345
func rnd() -> UInt32 { seed = seed &* 1664525 &+ 1013904223; return seed }
let inA = dev.makeBuffer(bytes: (0..<1024).map { _ in rnd() }, length: 4096, options: .storageModeShared)!
let outA = dev.makeBuffer(length: nThreads * 4, options: .storageModeShared)!
let ops = ["iadd", "shl_dyn", "shr_dyn", "shr_const", "and16", "funnel_naga", "funnel_c", "funnel_64", "extract",
           "ctz", "clz", "popc", "imul", "mulhi", "mod33", "dec33", "min", "u16"]
let pipesA = ops.map { pipe("k_" + $0) }
for p in pipesA { _ = run(p, [inA, outA], MTLSize(width: nThreads, height: 1, depth: 1), 256) } // warm-up
var bestA = [Double](repeating: 1e9, count: ops.count)
for _ in 0..<5 { for (i, p) in pipesA.enumerated() {
  bestA[i] = min(bestA[i], run(p, [inA, outA], MTLSize(width: nThreads, height: 1, depth: 1), 256)) } }
let opsPerRun = Double(nThreads) * 2048.0 * 8.0
print("\nPart A: lane-ops/s (each op incl. one xor/add); ratio = time / time(iadd)")
for (i, n) in ops.enumerated() {
  print(String(format: "  %-12@ %8.1f Gop/s  x%5.2f", n as NSString, opsPerRun / bestA[i] / 1e9, bestA[i] / bestA[0]))
}

// ---------- Part B ----------
let nBlk = 512, BS = 65536
var bytes = [UInt8](repeating: 0, count: nBlk * BS + 64)
for b in 0..<nBlk {
  var pos = 0
  let base = b * BS
  while pos < BS {
    let r = rnd()
    if pos < 64 || r % 4 != 0 {
      let n = Int(r >> 8) % 8 + 1
      for _ in 0..<n where pos < BS { bytes[base + pos] = UInt8(truncatingIfNeeded: (rnd() >> 16) % 32 + 65); pos += 1 }
    } else {
      let len = Int(r >> 8) % 37 + 4
      let off = Int(r >> 16) % min(pos, 4096) + 1
      for _ in 0..<len where pos < BS { bytes[base + pos] = bytes[base + pos - off]; pos += 1 }
    }
  }
}
func le32(_ i: Int) -> UInt32 {
  let b0 = UInt32(bytes[i]), b1 = UInt32(bytes[i + 1]) << 8
  let b2 = UInt32(bytes[i + 2]) << 16, b3 = UInt32(bytes[i + 3]) << 24
  return b0 | b1 | b2 | b3
}
let nWords = nBlk * BS / 4 + 4
var packed = [UInt32](repeating: 0, count: nWords)
for w in 0..<(nBlk * BS / 4) { packed[w] = le32(4 * w) }
var expanded = [UInt32](repeating: 0, count: nBlk * BS + 8)
for i in 0..<(nBlk * BS) { expanded[i] = le32(i) }
var pred = [UInt32](repeating: 0xFFFF_FFFF, count: nBlk * BS)
var head = [UInt32](repeating: 0xFFFF_FFFF, count: 1 << 16)
for b in 0..<nBlk {
  for i in 0..<head.count { head[i] = 0xFFFF_FFFF }
  for p in 0..<(BS - 8) {
    let h = Int((le32(b * BS + p) &* 2654435761) >> 16)
    pred[b * BS + p] = head[h]; head[h] = UInt32(p)
  }
}
let bPacked = dev.makeBuffer(bytes: packed, length: packed.count * 4, options: .storageModeShared)!
let bExp = dev.makeBuffer(bytes: expanded, length: expanded.count * 4, options: .storageModeShared)!
let bPred = dev.makeBuffer(bytes: pred, length: pred.count * 4, options: .storageModeShared)!
let out1 = dev.makeBuffer(length: nBlk * BS * 4, options: .storageModeShared)!
let out2 = dev.makeBuffer(length: nBlk * BS * 4, options: .storageModeShared)!
let kp = pipe("k2_packed"), ke = pipe("k2_expanded")
let grid = MTLSize(width: BS, height: nBlk, depth: 1)
_ = run(kp, [bPacked, bPred, out1], grid, 256); _ = run(ke, [bExp, bPred, out2], grid, 256)
var tp = 1e9, te = 1e9
for _ in 0..<5 {
  tp = min(tp, run(kp, [bPacked, bPred, out1], grid, 256))
  te = min(te, run(ke, [bExp, bPred, out2], grid, 256))
}
let same = memcmp(out1.contents(), out2.contents(), nBlk * BS * 4) == 0
print("\nPart B (\(nBlk) blocks of 64 KiB, depth 32, cap 64): outputs identical: \(same)")
print(String(format: "  packed   %7.2f us/block", tp / Double(nBlk) * 1e6))
print(String(format: "  expanded %7.2f us/block  (x%.3f of packed)", te / Double(nBlk) * 1e6, te / tp))
