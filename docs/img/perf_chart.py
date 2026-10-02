#!/usr/bin/env python3
"""Writes docs/img/perf.svg: compression ratio against throughput for libzstd on CPUs and for
gpu-zstd-comp's presets on GPUs. Full corpus (6.49 GB of Skyrim DDS/NIF), 64 KiB independent
blocks. Numbers are copied from docs/results/. No dependencies: run `python3 perf_chart.py`."""

import math

# (label, colour, dashed, [(ratio, MB/s, point label)])
SERIES = [
    ("libzstd, Ryzen 9 9950X3D, 32 threads", "#7f7f7f", True,
     [(1.246, 18046, "L1"), (1.262, 9360, "L3"), (1.335, 2265, "L6"), (1.338, 1747, "L9"),
      (1.36827, 590, "L14"), (1.37100, 501, "L16")]),
    ("libzstd, Ryzen 9 9950X3D, 8 threads", "#bdbdbd", True,
     [(1.246, 7092, "L1"), (1.262, 3841, "L3"), (1.335, 964, "L6"), (1.338, 731, "L9"),
      (1.368, 224, "L14"), (1.371, 190, "L16")]),
    ("libzstd, Apple M4 Pro, 12 threads", "#c49c94", True,
     [(1.246, 9427, "L1"), (1.262, 4821, "L3"), (1.335, 1161, "L6"), (1.338, 692, "L9"),
      (1.368, 295, "L14"), (1.371, 259, "L16")]),
    ("GPU, RTX 5090", "#1f77b4", False,
     [(1.264, 7447, "lvl3"), (1.339, 10396, "lvl9s12seg"),
      (1.37067, 2936, "opt14"), (1.37232, 3416, "opt16p1")]),
    ("GPU, GTX 1660 Super", "#2ca02c", False,
     [(1.339, 562, "lvl9s12seg"), (1.37067, 127, "opt14"), (1.37232, 140, "opt16p1")]),
    ("GPU, Apple M4 Pro", "#d62728", False,
     [(1.264, 491, "lvl3"), (1.339, 436, "lvl9s12seg"), (1.37067, 229, "opt14"),
      (1.37232, 239, "opt16p1")]),
]

W, H = 900, 560
L, R, T, B = 80, 300, 40, 60          # plot margins (legend on the right)
X0, X1 = 1.24, 1.38                    # ratio axis
Y0, Y1 = 50, 30000                     # MB/s axis (log)


def px(r):
    return L + (r - X0) / (X1 - X0) * (W - L - R)


def py(v):
    return T + (math.log10(Y1) - math.log10(v)) / (math.log10(Y1) - math.log10(Y0)) * (H - T - B)


out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       'font-family="Helvetica, Arial, sans-serif" font-size="12">',
       f'<rect width="{W}" height="{H}" fill="white"/>']

# grid and axes
for v in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
    y = py(v)
    out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#eeeeee"/>')
    out.append(f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" fill="#444">{v:,}</text>')
for i in range(8):
    r = X0 + i * 0.02
    x = px(r)
    out.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H-B}" stroke="#eeeeee"/>')
    out.append(f'<text x="{x:.1f}" y="{H-B+18}" text-anchor="middle" fill="#444">{r:.2f}</text>')
out.append(f'<rect x="{L}" y="{T}" width="{W-L-R}" height="{H-T-B}" fill="none" stroke="#999"/>')
out.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-18}" text-anchor="middle" fill="#222">'
           'compression ratio (higher is smaller output)</text>')
out.append(f'<text x="18" y="{(T+H-B)/2:.0f}" text-anchor="middle" fill="#222" '
           f'transform="rotate(-90 18 {(T+H-B)/2:.0f})">throughput, MB/s (log scale)</text>')

# download-speed reference lines
for v, name in [(1250, "10 Gbit/s"), (125, "1 Gbit/s")]:
    y = py(v)
    out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#ff7f0e" '
               'stroke-width="1.5" stroke-dasharray="6 4"/>')
    out.append(f'<text x="{L+6}" y="{y-5:.1f}" fill="#ff7f0e">{name}</text>')

# series
for name, col, dashed, pts in SERIES:
    pts = sorted(pts)
    path = " ".join(f"{'M' if i == 0 else 'L'}{px(r):.1f},{py(v):.1f}" for i, (r, v, _) in enumerate(pts))
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    out.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2"{dash}/>')
    for r, v, lab in pts:
        out.append(f'<circle cx="{px(r):.1f}" cy="{py(v):.1f}" r="3.5" fill="{col}"/>')
        if not dashed or name.endswith("32 threads"):
            below = (lab in ("opt14", "lvl3") or (lab == "lvl9s12seg" and col == "#d62728")) and not dashed
            anchor = "end" if lab == "opt14" else "start"
            dx = -6 if lab == "opt14" else 5
            dy = 14 if below else -6
            out.append(f'<text x="{px(r)+dx:.1f}" y="{py(v)+dy:.1f}" text-anchor="{anchor}" '
                       f'fill="{col}" font-size="10">{lab}</text>')

# legend
ly = T + 10
for name, col, dashed, _ in SERIES:
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    out.append(f'<line x1="{W-R+20}" y1="{ly}" x2="{W-R+48}" y2="{ly}" stroke="{col}" stroke-width="2"{dash}/>')
    out.append(f'<text x="{W-R+54}" y="{ly+4}" fill="#222">{name}</text>')
    ly += 22
out.append(f'<text x="{W-R+20}" y="{ly+14}" fill="#666" font-size="11">Full corpus, 6.49 GB,</text>')
out.append(f'<text x="{W-R+20}" y="{ly+29}" fill="#666" font-size="11">independent 64 KiB blocks.</text>')
out.append(f'<text x="{W-R+20}" y="{ly+44}" fill="#666" font-size="11">GPU numbers are end to end.</text>')
out.append("</svg>")

with open(__file__.replace("perf_chart.py", "perf.svg"), "w") as f:
    f.write("\n".join(out) + "\n")
