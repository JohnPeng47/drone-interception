from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import wandb


DEFAULT_RUN_PATH = "swagcorp/drone-interception/r0s4vgry"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export W&B run history and local scalar charts.")
    parser.add_argument("--run-path", default=DEFAULT_RUN_PATH, help="W&B run path: entity/project/run_id.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=1000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    run = api.run(args.run_path)
    rows = list(run.scan_history(page_size=int(args.page_size)))
    history_source = "scan_history"
    if not rows:
        history = run.history(samples=10000)
        rows = dataframe_records(history)
        history_source = "history"
    rows.sort(key=_row_order_key)

    summary = dict(run.summary)
    config = dict(run.config)
    metadata = {
        "run_path": args.run_path,
        "run_id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "entity": run.entity,
        "project": run.project,
        "history_source": history_source,
        "history_rows": len(rows),
        "numeric_metrics": numeric_metric_names(rows),
    }

    write_json(args.out_dir / "metadata.json", metadata)
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "config.json", config)
    write_jsonl(args.out_dir / "history.jsonl", rows)
    write_csv(args.out_dir / "history.csv", rows)
    (args.out_dir / "charts.html").write_text(render_html(metadata, rows), encoding="utf-8")
    print(args.out_dir)
    return 0


def _row_order_key(row: dict[str, Any]) -> tuple[float, float]:
    return (_number(row.get("_step")) or 0.0, _number(row.get("global_step")) or 0.0)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def dataframe_records(frame: Any) -> list[dict[str, Any]]:
    if frame is None:
        return []
    records = frame.to_dict(orient="records")
    return [{key: clean_value(value) for key, value in row.items()} for row in records]


def clean_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar_csv_value(row.get(key)) for key in keys})


def scalar_csv_value(value: Any) -> Any:
    if isinstance(value, (str, int, float)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, default=str)


def numeric_metric_names(rows: list[dict[str, Any]]) -> list[str]:
    x_keys = {"_step", "global_step"}
    names: list[str] = []
    for key in sorted({key for row in rows for key in row} - x_keys):
        points = series_points(rows, key)
        if len(points) >= 2:
            names.append(key)
    return names


def series_points(rows: list[dict[str, Any]], key: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index, row in enumerate(rows):
        y = _number(row.get(key))
        if y is None:
            continue
        x = _number(row.get("global_step"))
        if x is None:
            x = _number(row.get("_step"))
        if x is None:
            x = float(index)
        points.append((x, y))
    return points


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    return None


def render_html(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    metric_names = numeric_metric_names(rows)
    charts = "\n".join(render_chart(name, series_points(rows, name)) for name in metric_names)
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            f"<title>{escape(str(metadata['name']))} W&B Charts</title>",
            "<style>",
            CSS,
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            f"<h1>{escape(str(metadata['name']))}</h1>",
            f"<p class=\"path\"><a href=\"{escape(str(metadata['url']))}\">{escape(str(metadata['run_path']))}</a></p>",
            "<section class=\"cards\">",
            card("History rows", str(metadata["history_rows"])),
            card("Numeric charts", str(len(metric_names))),
            card("State", str(metadata["state"])),
            "</section>",
            "<section>",
            "<h2>Charts</h2>",
            "<div class=\"grid\">",
            charts,
            "</div>",
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def card(label: str, value: str) -> str:
    return f"<div class=\"card\"><div class=\"label\">{escape(label)}</div><div class=\"value\">{escape(value)}</div></div>"


def render_chart(name: str, points: list[tuple[float, float]]) -> str:
    return f"<article class=\"chart-card\"><h3>{escape(name)}</h3>{svg_line_chart(points, name)}</article>"


def svg_line_chart(points: list[tuple[float, float]], name: str) -> str:
    width, height = 720, 240
    left, right, top, bottom = 68, 20, 18, 44
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if math.isclose(min_x, max_x):
        min_x -= 1.0
        max_x += 1.0
    if math.isclose(min_y, max_y):
        pad = max(abs(min_y) * 0.05, 1.0)
        min_y -= pad
        max_y += pad
    else:
        pad = (max_y - min_y) * 0.08
        min_y -= pad
        max_y += pad

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + (max_y - y) / (max_y - min_y) * plot_h

    grid = []
    for i in range(4):
        frac = i / 3
        y = top + frac * plot_h
        value = max_y - frac * (max_y - min_y)
        grid.append(f"<line class=\"grid-line\" x1=\"{left}\" y1=\"{y:.2f}\" x2=\"{width-right}\" y2=\"{y:.2f}\" />")
        grid.append(f"<text class=\"tick\" x=\"{left-8}\" y=\"{y+4:.2f}\" text-anchor=\"end\">{format_value(value)}</text>")
    for i in range(4):
        frac = i / 3
        x = left + frac * plot_w
        value = min_x + frac * (max_x - min_x)
        grid.append(f"<line class=\"grid-line\" x1=\"{x:.2f}\" y1=\"{top}\" x2=\"{x:.2f}\" y2=\"{height-bottom}\" />")
        grid.append(f"<text class=\"tick\" x=\"{x:.2f}\" y=\"{height-bottom+22}\" text-anchor=\"middle\">{format_step(value)}</text>")

    path = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
    circles = "\n".join(
        f"<circle cx=\"{sx(x):.2f}\" cy=\"{sy(y):.2f}\" r=\"2.5\"><title>step {format_step(x)}: {y:.6g}</title></circle>"
        for x, y in points
    )
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(name)} over global step">
  <rect class="plot-bg" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" />
  {''.join(grid)}
  <line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" />
  <line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" />
  <polyline class="line" points="{path}" />
  <g>{circles}</g>
</svg>
"""


def format_step(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def format_value(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2g}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2g}K"
    return f"{value:.3g}"


def escape(value: str) -> str:
    return html.escape(value, quote=True)


CSS = """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #fff;
  --ink: #1f2933;
  --muted: #667085;
  --line: #1f7a8c;
  --border: #d9dee7;
  --grid: #e7eaf0;
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  max-width: 1480px;
  margin: 0 auto;
  padding: 28px 24px 48px;
}
h1 {
  font-size: 28px;
  margin: 0 0 6px;
}
h2 {
  font-size: 18px;
  margin: 0 0 16px;
}
h3 {
  font-size: 13px;
  margin: 0 0 10px;
  overflow-wrap: anywhere;
}
a, .path {
  color: var(--muted);
}
section {
  margin: 18px 0;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.card, .chart-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
}
.card {
  padding: 14px;
}
.label {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}
.value {
  font-size: 22px;
  font-weight: 700;
  margin-top: 6px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
  gap: 14px;
}
.chart-card {
  padding: 14px;
  min-width: 0;
}
.chart {
  width: 100%;
  height: auto;
  display: block;
}
.plot-bg {
  fill: #fbfcfe;
}
.grid-line {
  stroke: var(--grid);
  stroke-width: 1;
}
.axis {
  stroke: #98a2b3;
  stroke-width: 1.4;
}
.line {
  fill: none;
  stroke: var(--line);
  stroke-width: 2.2;
  stroke-linejoin: round;
  stroke-linecap: round;
}
circle {
  fill: var(--line);
}
.tick {
  fill: var(--muted);
  font-size: 10px;
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
