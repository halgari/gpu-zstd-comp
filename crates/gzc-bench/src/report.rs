//! Table, JSON and HTML report generation for benchmark results.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::corpus::Kind;
use crate::result::RunResult;

const GBIT_1: f64 = 125.0;
const GBIT_10: f64 = 1250.0;

pub fn print_table(results: &[RunResult]) {
    println!(
        "{:<14} {:<10} {:>8} {:>10} {:>12} {:>10}",
        "engine", "config", "threads", "ratio", "MB/s", "seconds"
    );
    for r in results {
        let threads = r.threads.map(|t| t.to_string()).unwrap_or_else(|| "-".to_string());
        println!(
            "{:<14} {:<10} {:>8} {:>10.3} {:>12.1} {:>10.3}",
            r.engine,
            r.config,
            threads,
            r.ratio(),
            r.mb_per_s(),
            r.seconds
        );
    }
}

pub fn write_json(results: &[RunResult], dir: &Path) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("results.json");
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write(&path, json)?;
    Ok(path)
}

pub fn write_html(results: &[RunResult], dir: &Path) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("report.html");
    std::fs::write(&path, render_html(results))?;
    Ok(path)
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Dds => "DDS",
        Kind::Nif => "NIF",
        Kind::Other => "Other",
    }
}

/// Fixed categorical palette (light-mode hexes; dark-mode values are set via
/// CSS custom properties, see `palette_style`), assigned to series in a
/// fixed order — never cycled per re-render for the same input.
const SERIES_SLOTS: usize = 8;

fn series_color_var(slot: usize) -> String {
    format!("var(--series-{})", (slot % SERIES_SLOTS) + 1)
}

fn palette_style() -> &'static str {
    r#"
:root, .gzc-report {
  color-scheme: light;
  --page: #f9f9f7;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --series-4: #eda100;
  --series-5: #e87ba4;
  --series-6: #008300;
  --series-7: #4a3aa7;
  --series-8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --series-4: #c98500;
    --series-5: #d55181;
    --series-6: #008300;
    --series-7: #9085e9;
    --series-8: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d;
  --surface-1: #1a1a19;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted: #898781;
  --grid: #2c2c2a;
  --axis: #383835;
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
  --series-4: #c98500;
  --series-5: #d55181;
  --series-6: #008300;
  --series-7: #9085e9;
  --series-8: #e66767;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
.gzc-report { max-width: 1040px; margin: 0 auto; padding: 24px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; color: var(--text-secondary); margin: 32px 0 8px; }
.subtitle { color: var(--text-secondary); font-size: 13px; margin: 0 0 24px; }
.headline {
  background: var(--surface-1);
  border: 1px solid var(--grid);
  border-radius: 8px;
  padding: 14px 18px;
  font-size: 14px;
  margin-bottom: 20px;
}
.headline b { color: var(--text-primary); }
.chart-wrap { background: var(--surface-1); border: 1px solid var(--grid); border-radius: 8px; padding: 12px; }
svg text { fill: var(--text-secondary); font-size: 11px; }
svg .axis-label { fill: var(--text-muted); }
svg .rule-label { fill: var(--text-secondary); font-weight: 600; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 8px; }
th, td {
  padding: 6px 10px;
  text-align: right;
  border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
th { color: var(--text-muted); font-weight: 600; }
"#
}

struct Series<'a> {
    name: String,
    slot: usize,
    points: Vec<(f64, f64, &'a RunResult)>, // (mb_per_s, ratio, run)
}

fn render_html(results: &[RunResult]) -> String {
    // Group cpu-libzstd runs by thread count, ascending; sort each series by
    // throughput so a polyline reads left to right.
    let mut by_threads: BTreeMap<usize, Vec<&RunResult>> = BTreeMap::new();
    for r in results.iter().filter(|r| r.engine == "cpu-libzstd") {
        if let Some(t) = r.threads {
            by_threads.entry(t).or_default().push(r);
        }
    }
    for v in by_threads.values_mut() {
        v.sort_by(|a, b| a.mb_per_s().partial_cmp(&b.mb_per_s()).unwrap());
    }

    let mut series: Vec<Series> = Vec::new();
    for (slot, (threads, runs)) in by_threads.iter().enumerate() {
        series.push(Series {
            name: format!("{threads} threads"),
            slot,
            points: runs.iter().map(|r| (r.mb_per_s(), r.ratio(), *r)).collect(),
        });
    }

    // Dashed linear-scaling projection: the 1-thread series' points at 8x
    // throughput, same ratio.
    let projection: Option<Series> = by_threads.get(&1).map(|runs| Series {
        name: "1 thread x8 (projected)".to_string(),
        slot: 0,
        points: runs.iter().map(|r| (r.mb_per_s() * 8.0, r.ratio(), *r)).collect(),
    });

    // Non-cpu-libzstd runs (cpu-ref, gpu) plotted as labelled points.
    let mut other_slot = series.len();
    let other_points: Vec<(String, usize, f64, f64, &RunResult)> = results
        .iter()
        .filter(|r| r.engine != "cpu-libzstd")
        .map(|r| {
            let slot = other_slot;
            other_slot += 1;
            (format!("{} ({})", r.engine, r.config), slot, r.mb_per_s(), r.ratio(), r)
        })
        .collect();

    // Domain.
    let mut xs: Vec<f64> = vec![GBIT_1, GBIT_10];
    let mut ys: Vec<f64> = vec![1.0];
    for s in series.iter().chain(projection.iter()) {
        for &(x, y, _) in &s.points {
            if x > 0.0 {
                xs.push(x);
            }
            ys.push(y);
        }
    }
    for &(_, _, x, y, _) in &other_points {
        if x > 0.0 {
            xs.push(x);
        }
        ys.push(y);
    }

    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min).max(1.0);
    let x_max = xs.iter().cloned().fold(0.0, f64::max).max(x_min * 1.1);
    let y_max = ys.iter().cloned().fold(1.0, f64::max) * 1.15;
    let y_min = 1.0;

    let width = 960.0;
    let height = 460.0;
    let margin_left = 60.0;
    let margin_right = 190.0;
    let margin_top = 20.0;
    let margin_bottom = 46.0;
    let plot_w = width - margin_left - margin_right;
    let plot_h = height - margin_top - margin_bottom;

    let x_log_min = (x_min / 1.15).max(1.0).log10();
    let x_log_max = (x_max * 1.15).log10();
    let x_pos = |mb: f64| -> f64 {
        let mb = mb.max(1.0);
        margin_left + (mb.log10() - x_log_min) / (x_log_max - x_log_min) * plot_w
    };
    let y_pos = |ratio: f64| -> f64 {
        margin_top + plot_h - (ratio - y_min) / (y_max - y_min).max(1e-9) * plot_h
    };

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg viewBox="0 0 {width} {height}" role="img" aria-label="Compression ratio vs throughput" xmlns="http://www.w3.org/2000/svg">"#
    ));
    svg.push_str(&format!(
        r#"<rect x="0" y="0" width="{width}" height="{height}" fill="var(--surface-1)"/>"#
    ));

    // Y gridlines + tick labels (ratio, round steps).
    let y_step = (((y_max - y_min) / 5.0).ceil()).max(1.0);
    let mut y_tick = y_min;
    while y_tick <= y_max {
        let py = y_pos(y_tick);
        svg.push_str(&format!(
            r#"<line x1="{:.1}" y1="{py:.1}" x2="{:.1}" y2="{py:.1}" stroke="var(--grid)" stroke-width="1"/>"#,
            margin_left,
            width - margin_right
        ));
        svg.push_str(&format!(
            r#"<text class="axis-label" x="{:.1}" y="{:.1}" text-anchor="end" dominant-baseline="middle">{:.1}x</text>"#,
            margin_left - 8.0,
            py,
            y_tick
        ));
        y_tick += y_step;
    }

    // X axis log ticks at powers of ten within range.
    let mut p = 10f64.powi(x_log_min.floor() as i32);
    while p <= 10f64.powi(x_log_max.ceil() as i32) {
        if p >= x_min / 2.0 && p <= x_max * 2.0 {
            let px = x_pos(p);
            if px >= margin_left && px <= width - margin_right {
                svg.push_str(&format!(
                    r#"<line x1="{px:.1}" y1="{:.1}" x2="{px:.1}" y2="{:.1}" stroke="var(--grid)" stroke-width="1"/>"#,
                    margin_top,
                    margin_top + plot_h
                ));
                let label = if p >= 1000.0 { format!("{:.0}k", p / 1000.0) } else { format!("{p:.0}") };
                svg.push_str(&format!(
                    r#"<text class="axis-label" x="{px:.1}" y="{:.1}" text-anchor="middle">{label}</text>"#,
                    margin_top + plot_h + 16.0
                ));
            }
        }
        p *= 10.0;
    }

    // Vertical rules at 1 Gbit / 10 Gbit.
    for (x, label) in [(GBIT_1, "1 Gbit"), (GBIT_10, "10 Gbit")] {
        let px = x_pos(x);
        svg.push_str(&format!(
            r#"<line x1="{px:.1}" y1="{:.1}" x2="{px:.1}" y2="{:.1}" stroke="var(--axis)" stroke-width="1.5" stroke-dasharray="4 3"/>"#,
            margin_top,
            margin_top + plot_h
        ));
        svg.push_str(&format!(
            r#"<text class="rule-label" x="{px:.1}" y="{:.1}" text-anchor="middle">{label}</text>"#,
            margin_top - 6.0
        ));
    }

    // Axis lines.
    svg.push_str(&format!(
        r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="var(--axis)" stroke-width="1"/>"#,
        margin_left,
        margin_top + plot_h,
        width - margin_right,
        margin_top + plot_h
    ));
    svg.push_str(&format!(
        r#"<text class="axis-label" x="{:.1}" y="{:.1}" text-anchor="middle">throughput, MB/s (log scale)</text>"#,
        margin_left + plot_w / 2.0,
        height - 6.0
    ));

    // Dashed projection line (drawn first, under the real series).
    if let Some(proj) = &projection {
        if proj.points.len() > 1 {
            let pts: Vec<String> =
                proj.points.iter().map(|&(x, y, _)| format!("{:.1},{:.1}", x_pos(x), y_pos(y))).collect();
            svg.push_str(&format!(
                r#"<polyline points="{}" fill="none" stroke="{}" stroke-width="2" stroke-dasharray="6 4" opacity="0.65"/>"#,
                pts.join(" "),
                series_color_var(proj.slot)
            ));
        }
    }

    // Thread-count series.
    for s in &series {
        let emphasized = s.name == "8 threads";
        let stroke_w = if emphasized { 3.0 } else { 2.0 };
        let color = series_color_var(s.slot);
        if s.points.len() > 1 {
            let pts: Vec<String> =
                s.points.iter().map(|&(x, y, _)| format!("{:.1},{:.1}", x_pos(x), y_pos(y))).collect();
            svg.push_str(&format!(
                r#"<polyline points="{}" fill="none" stroke="{color}" stroke-width="{stroke_w}"/>"#,
                pts.join(" ")
            ));
        }
        for &(x, y, r) in &s.points {
            let (px, py) = (x_pos(x), y_pos(y));
            svg.push_str(&format!(
                r#"<circle cx="{px:.1}" cy="{py:.1}" r="5" fill="{color}" stroke="var(--surface-1)" stroke-width="2"><title>{}</title></circle>"#,
                escape(&format!(
                    "{} {} @ {} threads: {:.2}x, {:.1} MB/s",
                    r.engine,
                    r.config,
                    s.name.replace(" threads", ""),
                    y,
                    x
                ))
            ));
        }
    }

    // GPU / ref points.
    for (label, slot, x, y, r) in &other_points {
        let color = series_color_var(*slot);
        let (px, py) = (x_pos(*x), y_pos(*y));
        svg.push_str(&format!(
            r#"<rect x="{:.1}" y="{:.1}" width="10" height="10" fill="{color}" stroke="var(--surface-1)" stroke-width="2"><title>{}</title></rect>"#,
            px - 5.0,
            py - 5.0,
            escape(&format!("{label}: {:.2}x, {:.1} MB/s", y, x))
        ));
        svg.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" text-anchor="start">{}</text>"#,
            px + 9.0,
            py - 8.0,
            escape(label)
        ));
        let _ = r;
    }

    // Legend.
    let mut legend_y = margin_top + 4.0;
    let legend_x = width - margin_right + 16.0;
    for s in &series {
        let emphasized = s.name == "8 threads";
        svg.push_str(&format!(
            r#"<line x1="{legend_x:.1}" y1="{legend_y:.1}" x2="{:.1}" y2="{legend_y:.1}" stroke="{}" stroke-width="{}"/>"#,
            legend_x + 20.0,
            series_color_var(s.slot),
            if emphasized { 3.0 } else { 2.0 }
        ));
        svg.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" dominant-baseline="middle">{}</text>"#,
            legend_x + 26.0,
            legend_y,
            escape(&s.name)
        ));
        legend_y += 18.0;
    }
    if let Some(proj) = &projection {
        svg.push_str(&format!(
            r#"<line x1="{legend_x:.1}" y1="{legend_y:.1}" x2="{:.1}" y2="{legend_y:.1}" stroke="{}" stroke-width="2" stroke-dasharray="6 4" opacity="0.65"/>"#,
            legend_x + 20.0,
            series_color_var(proj.slot)
        ));
        svg.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" dominant-baseline="middle">1 thread x8 (proj.)</text>"#,
            legend_x + 26.0,
            legend_y
        ));
        legend_y += 18.0;
    }
    for (label, slot, ..) in &other_points {
        svg.push_str(&format!(
            r#"<rect x="{legend_x:.1}" y="{:.1}" width="10" height="10" fill="{}"/>"#,
            legend_y - 5.0,
            series_color_var(*slot)
        ));
        svg.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" dominant-baseline="middle">{}</text>"#,
            legend_x + 16.0,
            legend_y,
            escape(label)
        ));
        legend_y += 18.0;
    }

    svg.push_str("</svg>");

    // Headline: best ratio with mb_per_s >= 10 Gbit for cpu-libzstd@8 threads, vs best for gpu.
    let best_cpu8 = results
        .iter()
        .filter(|r| r.engine == "cpu-libzstd" && r.threads == Some(8) && r.mb_per_s() >= GBIT_10)
        .max_by(|a, b| a.ratio().partial_cmp(&b.ratio()).unwrap());
    let best_gpu = results
        .iter()
        .filter(|r| r.engine == "gpu" && r.mb_per_s() >= GBIT_10)
        .max_by(|a, b| a.ratio().partial_cmp(&b.ratio()).unwrap());

    let headline = match (best_cpu8, best_gpu) {
        (Some(c), Some(g)) => format!(
            "At >= 10 Gbit (1250 MB/s): best cpu-libzstd (8 threads) ratio is <b>{:.2}x</b> ({}); best gpu ratio is <b>{:.2}x</b> ({}).",
            c.ratio(), escape(&c.config), g.ratio(), escape(&g.config)
        ),
        (Some(c), None) => format!(
            "At >= 10 Gbit (1250 MB/s): best cpu-libzstd (8 threads) ratio is <b>{:.2}x</b> ({}); no gpu run reached 10 Gbit yet.",
            c.ratio(), escape(&c.config)
        ),
        (None, Some(g)) => format!(
            "No cpu-libzstd (8 threads) run reached 10 Gbit; best gpu ratio at >= 10 Gbit is <b>{:.2}x</b> ({}).",
            g.ratio(), escape(&g.config)
        ),
        (None, None) => "No cpu-libzstd (8 threads) or gpu run reached 10 Gbit (1250 MB/s) yet.".to_string(),
    };

    // Results table.
    let mut results_rows = String::new();
    for r in results {
        results_rows.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3}x</td><td>{:.1}</td><td>{:.3}</td></tr>",
            escape(&r.engine),
            escape(&r.config),
            r.threads.map(|t| t.to_string()).unwrap_or_else(|| "-".to_string()),
            r.ratio(),
            r.mb_per_s(),
            r.seconds
        ));
    }

    // Per-kind table: DDS/NIF/Other ratio per run.
    let mut kind_rows = String::new();
    for r in results {
        for k in &r.per_kind {
            let ratio = if k.compressed_bytes > 0 {
                k.real_bytes as f64 / k.compressed_bytes as f64
            } else {
                0.0
            };
            kind_rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{ratio:.3}x</td></tr>",
                escape(&r.engine),
                escape(&r.config),
                r.threads.map(|t| t.to_string()).unwrap_or_else(|| "-".to_string()),
                kind_name(k.kind)
            ));
        }
    }

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gzc-bench report</title>
<style>{style}</style>
</head>
<body>
<div class="gzc-report">
<h1>gzc-bench: compression ratio vs throughput</h1>
<p class="subtitle">x = throughput (MB/s, log scale), y = compression ratio. 8-thread cpu-libzstd line is emphasized; dashed line is the 1-thread result linearly projected to 8x throughput.</p>
<div class="headline">{headline}</div>
<div class="chart-wrap">{svg}</div>

<h2>All runs</h2>
<table>
<thead><tr><th>Engine</th><th>Config</th><th>Threads</th><th>Ratio</th><th>MB/s</th><th>Seconds</th></tr></thead>
<tbody>{results_rows}</tbody>
</table>

<h2>Per-kind ratio</h2>
<table>
<thead><tr><th>Engine</th><th>Config</th><th>Threads</th><th>Kind</th><th>Ratio</th></tr></thead>
<tbody>{kind_rows}</tbody>
</table>
</div>
</body>
</html>
"#,
        style = palette_style(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::KindStat;

    fn sample_results() -> Vec<RunResult> {
        vec![
            RunResult {
                engine: "cpu-libzstd".into(),
                config: "L3".into(),
                threads: Some(1),
                real_bytes: 1_000_000,
                compressed_bytes: 400_000,
                seconds: 0.008,
                per_kind: vec![KindStat { kind: Kind::Dds, real_bytes: 1_000_000, compressed_bytes: 400_000 }],
                kernel_ms: vec![],
            },
            RunResult {
                engine: "cpu-libzstd".into(),
                config: "L3".into(),
                threads: Some(8),
                real_bytes: 1_000_000,
                compressed_bytes: 400_000,
                seconds: 0.0007,
                per_kind: vec![KindStat { kind: Kind::Dds, real_bytes: 1_000_000, compressed_bytes: 400_000 }],
                kernel_ms: vec![],
            },
            RunResult {
                engine: "gpu".into(),
                config: "lvl3-greedy".into(),
                threads: Some(4),
                real_bytes: 1_000_000,
                compressed_bytes: 350_000,
                seconds: 0.0006,
                per_kind: vec![KindStat { kind: Kind::Dds, real_bytes: 1_000_000, compressed_bytes: 350_000 }],
                kernel_ms: vec![("match".into(), 0.1)],
            },
        ]
    }

    #[test]
    fn html_report_contains_svg_and_bandwidth_labels() {
        let dir = std::env::temp_dir()
            .join(format!("gzc-bench-report-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let results = sample_results();
        let path = write_html(&results, &dir).unwrap();
        let html = std::fs::read_to_string(&path).unwrap();

        assert!(html.contains("<svg"));
        assert!(html.contains("1 Gbit"));
        assert!(html.contains("10 Gbit"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn json_report_round_trips() {
        let dir = std::env::temp_dir()
            .join(format!("gzc-bench-report-json-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let results = sample_results();
        let path = write_json(&results, &dir).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed.len(), 3);

        std::fs::remove_dir_all(&dir).ok();
    }
}
