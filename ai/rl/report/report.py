from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = ROOT / "ai" / "rl" / "runs"
REPORTS_ROOT = ROOT / "ai" / "rl" / "report" / "reports"
REPORT_INDEX = ROOT / "ai" / "rl" / "report" / "index.html"
DEFAULT_SNAPSHOT_SET = "stationary_target_512"


@dataclass(frozen=True)
class Series:
    name: str
    points: list[tuple[float, float]]
    note: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML RL reports for ai/rl/runs directories.")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run directory, run-date directory, or path relative to ai/rl/runs.",
    )
    parser.add_argument("--out-dir", type=Path, default=REPORTS_ROOT, help="Report output root.")
    parser.add_argument("--snapshot-set", default=DEFAULT_SNAPSHOT_SET, help="Preferred snapshots/<name> directory.")
    args = parser.parse_args()

    written = []
    for run_dir in resolve_run_dirs(args.paths):
        report_dir = args.out_dir / safe_name(run_dir.name)
        written.append(generate_report(run_dir, report_dir, snapshot_set=args.snapshot_set))
    index_path = write_reports_index(args.out_dir)
    print(json.dumps({"index": str(index_path), "reports": [str(path) for path in written]}, indent=2, sort_keys=True))


def generate_report(run_dir: Path, out_path: Path, *, snapshot_set: str | None = DEFAULT_SNAPSHOT_SET) -> Path:
    out_path = Path(out_path)
    report_path = out_path if out_path.suffix.lower() == ".html" else out_path / "index.html"
    assets_dir = report_path.parent / "assets"
    train_rows, train_meta = load_train_log(run_dir)
    if not train_rows:
        train_rows, train_meta = load_wandb_history(run_dir)
    eval_rows = load_snapshot_evals(run_dir, snapshot_set=snapshot_set)
    checkpoints = checkpoint_paths(run_dir)

    loss_series, loss_note = build_loss_series(train_rows, train_meta)
    capture_series = Series(
        "capture_percent",
        [(row["global_step"], row["catch_rate"] * 100.0) for row in eval_rows if "catch_rate" in row],
        "Snapshot evaluation capture percentage.",
    )
    min_distance_series = Series(
        "min_distance_p50_m",
        [(row["global_step"], row["min_distance_p50_m"]) for row in eval_rows if "min_distance_p50_m" in row],
        "Snapshot evaluation median minimum distance.",
    )
    out_of_bounds_series = Series(
        "out_of_bounds",
        [(row["global_step"], row["out_of_bounds"]) for row in eval_rows if "out_of_bounds" in row],
        "Out-of-bounds terminal episodes in each snapshot evaluation.",
    )

    missing = []
    if not train_rows:
        missing.append("No JSON training metrics found in logs/train.log, so loss-over-steps cannot be plotted.")
    elif not loss_series.points:
        missing.append("Training metrics exist, but no usable loss components were found.")
    if not eval_rows:
        missing.append(
            "No snapshots/**/snapshot_eval.json files found, so Snapshot Trends cannot be plotted. "
            "Run snapshot eval across checkpoints to generate these summaries."
        )
    else:
        if not capture_series.points:
            missing.append("Snapshot eval files exist, but summary.catch_rate was not found for Capture %.")
        if not min_distance_series.points:
            missing.append("Snapshot eval files exist, but summary.min_distance_p50_m was not found for Median Min Distance.")
        if not out_of_bounds_series.points:
            missing.append("Snapshot eval files exist, but summary.terminal_counts.oob was not found for Out-of-Bounds.")
        if len(eval_rows) < len(checkpoints):
            missing.append(
                f"Found snapshot eval summaries for {len(eval_rows)} of {len(checkpoints)} checkpoint files; "
                "Snapshot Trends only include evaluated checkpoints."
            )

    chart_assets = write_chart_assets(
        assets_dir,
        loss_series=loss_series,
        capture_series=capture_series,
        min_distance_series=min_distance_series,
        out_of_bounds_series=out_of_bounds_series,
    )
    html_text = render_html(
        run_dir=run_dir,
        report_path=report_path,
        checkpoints=checkpoints,
        train_rows=train_rows,
        eval_rows=eval_rows,
        loss_series=loss_series,
        loss_note=loss_note,
        capture_series=capture_series,
        min_distance_series=min_distance_series,
        out_of_bounds_series=out_of_bounds_series,
        chart_assets=chart_assets,
        missing=missing,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def write_reports_index(reports_root: Path = REPORTS_ROOT, index_path: Path = REPORT_INDEX) -> Path:
    reports_root = Path(reports_root)
    index_path = Path(index_path)
    reports = sorted(
        path
        for path in reports_root.glob("*/index.html")
        if path.is_file()
    )
    rows = []
    for report in reports:
        report_dir = report.parent
        updated = report.stat().st_mtime
        rows.append(
            {
                "name": report_dir.name,
                "href": report.relative_to(index_path.parent).as_posix(),
                "updated": updated,
            }
        )
    rows.sort(key=lambda row: row["name"].lower())

    links = "\n".join(
        (
            "<li>"
            f"<a href=\"{escape(row['href'])}\">{escape(row['name'])}</a>"
            "</li>"
        )
        for row in rows
    )
    if not links:
        links = "<li class=\"empty\">No reports found.</li>"
    html_text = "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            "<title>RL Reports</title>",
            "<style>",
            INDEX_CSS,
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<h1>RL Reports</h1>",
            "<p>Generated from report folders under <code>reports/</code>.</p>",
            f"<ul>{links}</ul>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(html_text, encoding="utf-8")
    return index_path


def resolve_run_dir(path: Path) -> Path:
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend((Path.cwd() / path, RUNS_ROOT / path))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            try:
                resolved.relative_to(RUNS_ROOT.resolve())
            except ValueError as exc:
                raise ValueError(f"run directory must be inside {RUNS_ROOT}: {resolved}") from exc
            return resolved
    raise FileNotFoundError(f"run directory not found under {RUNS_ROOT}: {path}")


def resolve_run_dirs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = resolve_run_or_parent(path)
        child_runs = sorted(child for child in resolved.iterdir() if child.is_dir() and is_run_dir(child))
        candidates = child_runs if child_runs else ([resolved] if is_run_dir(resolved) else [])
        for candidate in candidates:
            key = candidate.resolve()
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
    if not out:
        raise FileNotFoundError("no run directories found")
    return out


def resolve_run_or_parent(path: Path) -> Path:
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend((Path.cwd() / path, RUNS_ROOT / path))
    runs_root = RUNS_ROOT.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_dir():
            continue
        try:
            resolved.relative_to(runs_root)
        except ValueError as exc:
            raise ValueError(f"path must be inside {RUNS_ROOT}: {resolved}") from exc
        return resolved
    raise FileNotFoundError(f"path not found under {RUNS_ROOT}: {path}")


def is_run_dir(path: Path) -> bool:
    return bool(
        any(path.glob("*.pt"))
        or any((path / "checkpoints").glob("**/*.pt"))
        or (path / "logs" / "train.log").exists()
        or (path / "wandb_export" / "history.csv").exists()
        or (path / "snapshots").is_dir()
    )


def load_train_log(run_dir: Path) -> tuple[list[dict[str, float]], dict[str, Any]]:
    log_path = run_dir / "logs" / "train.log"
    rows: list[dict[str, float]] = []
    meta: dict[str, Any] = {}
    if not log_path.exists():
        return rows, meta

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "start" and isinstance(payload.get("args"), dict):
            meta["args"] = payload["args"]
        if "global_step" not in payload:
            continue
        numeric = {
            key: float(value)
            for key, value in payload.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        if numeric:
            rows.append(numeric)
    rows.sort(key=lambda row: row.get("global_step", 0.0))
    return rows, meta


def load_wandb_history(run_dir: Path) -> tuple[list[dict[str, float]], dict[str, Any]]:
    history_path = run_dir / "wandb_export" / "history.csv"
    rows: list[dict[str, float]] = []
    if not history_path.exists():
        return rows, {}
    with history_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for payload in csv.DictReader(handle):
            numeric: dict[str, float] = {}
            for key, value in payload.items():
                if value is None or value == "":
                    continue
                number = _number(value)
                if number is not None:
                    numeric[key] = number
            step = numeric.get("global_step")
            if step is None:
                step = numeric.get("global_step_or_env_step") or numeric.get("_step")
                if step is not None:
                    numeric["global_step"] = step
            if "global_step" in numeric:
                rows.append(numeric)
    rows.sort(key=lambda row: row.get("global_step", 0.0))
    return rows, {"source": str(history_path)}


def load_snapshot_evals(run_dir: Path, snapshot_set: str | None = DEFAULT_SNAPSHOT_SET) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    root = run_dir / "snapshots"
    if snapshot_set and (root / snapshot_set).is_dir():
        search_root = root / snapshot_set
    else:
        search_root = root
    for path in sorted(search_root.glob("**/snapshot_eval.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        info = payload.get("checkpoint_info", {}) if isinstance(payload, dict) else {}
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        step = _number(info.get("global_step"))
        if step is None:
            step = _number(path.parent.name)
        if step is None:
            continue
        row: dict[str, float] = {"global_step": step}
        catch_rate = _number(summary.get("catch_rate"))
        min_distance = _number(summary.get("min_distance_p50_m"))
        episodes = _number(summary.get("episodes"))
        terminal_counts = summary.get("terminal_counts", {})
        out_of_bounds = _number(terminal_counts.get("oob")) if isinstance(terminal_counts, dict) else None
        if catch_rate is not None:
            row["catch_rate"] = catch_rate
        if min_distance is not None:
            row["min_distance_p50_m"] = min_distance
        if episodes is not None:
            row["episodes"] = episodes
        if out_of_bounds is not None:
            row["out_of_bounds"] = out_of_bounds
        rows.append(row)
    rows.sort(key=lambda row: row["global_step"])
    return rows


def checkpoint_paths(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.glob("**/*.pt") if path.name != "latest.pt")


def build_loss_series(rows: list[dict[str, float]], meta: dict[str, Any]) -> tuple[Series, str]:
    points: list[tuple[float, float]] = []
    args = meta.get("args", {}) if isinstance(meta.get("args"), dict) else {}
    vf_coef = _number(args.get("vf_coef"))
    ent_coef = _number(args.get("ent_coef"))
    note = "Estimated PPO objective from logged components: policy_loss + vf_coef * value_loss - ent_coef * entropy."
    if vf_coef is None:
        vf_coef = 1.0
        note += " vf_coef was missing; used 1.0."
    if ent_coef is None:
        ent_coef = 0.0
        note += " ent_coef was missing; used 0.0."

    for row in rows:
        step = row.get("global_step")
        policy = row.get("loss/policy_loss")
        value = row.get("loss/value_loss")
        entropy = row.get("loss/entropy")
        if step is None or policy is None or value is None or entropy is None:
            continue
        loss = policy + vf_coef * value - ent_coef * entropy
        if math.isfinite(loss):
            points.append((step, loss))

    if points:
        return Series("estimated_ppo_loss", points, note), note

    fallback_key = None
    if rows:
        fallback_key = next((key for key in sorted(rows[0].keys()) if key.startswith("loss/")), None)
    if fallback_key is None:
        return Series("estimated_ppo_loss", []), note
    fallback = [
        (row["global_step"], row[fallback_key])
        for row in rows
        if "global_step" in row and fallback_key in row and math.isfinite(row[fallback_key])
    ]
    return Series(fallback_key, fallback, f"Total PPO objective could not be reconstructed; plotted {fallback_key}."), note


def render_html(
    *,
    run_dir: Path,
    report_path: Path,
    checkpoints: list[Path],
    train_rows: list[dict[str, float]],
    eval_rows: list[dict[str, float]],
    loss_series: Series,
    loss_note: str,
    capture_series: Series,
    min_distance_series: Series,
    out_of_bounds_series: Series,
    chart_assets: dict[str, Path],
    missing: list[str],
) -> str:
    title = f"RL Run Report: {run_dir.name}"
    cards = [
        ("Training metric rows", str(len(train_rows))),
        ("Snapshot evals", str(len(eval_rows))),
        ("Checkpoints", str(len(checkpoints))),
    ]
    if train_rows:
        cards.append(("First train step", format_step(train_rows[0]["global_step"])))
        cards.append(("Last train step", format_step(train_rows[-1]["global_step"])))
    if eval_rows:
        cards.append(("Last eval step", format_step(eval_rows[-1]["global_step"])))
        if "catch_rate" in eval_rows[-1]:
            cards.append(("Latest capture %", f"{eval_rows[-1]['catch_rate'] * 100.0:.1f}%"))
        if "min_distance_p50_m" in eval_rows[-1]:
            cards.append(("Latest median min distance", f"{eval_rows[-1]['min_distance_p50_m']:.3f} m"))
        if "out_of_bounds" in eval_rows[-1]:
            cards.append(("Latest out-of-bounds", str(int(eval_rows[-1]["out_of_bounds"]))))
            cards.append(("Total out-of-bounds", str(int(sum(row.get("out_of_bounds", 0.0) for row in eval_rows)))))

    body = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{escape(title)}</title>",
        "<style>",
        CSS,
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{escape(title)}</h1>",
        f"<p class=\"path\">{escape(str(run_dir))}</p>",
        render_cards(cards),
    ]
    if missing:
        body.append("<section><h2>Missing Data</h2><ul class=\"missing\">")
        body.extend(f"<li>{escape(item)}</li>" for item in missing)
        body.append("</ul></section>")

    body.extend([
        render_chart_section(
            "Loss Over Training Steps",
            loss_series,
            y_label="estimated loss",
            empty="No loss series available.",
            note=loss_series.note or loss_note,
            image_path=chart_assets.get(loss_series.name),
            report_path=report_path,
        ),
        render_snapshot_trends_section(
            capture_series=capture_series,
            min_distance_series=min_distance_series,
            out_of_bounds_series=out_of_bounds_series,
            chart_assets=chart_assets,
            report_path=report_path,
        ),
        render_table_section(train_rows, eval_rows),
        "</main>",
        "</body>",
        "</html>",
    ])
    return "\n".join(body)


def render_cards(cards: list[tuple[str, str]]) -> str:
    inner = "\n".join(
        f"<div class=\"card\"><div class=\"card-label\">{escape(label)}</div><div class=\"card-value\">{escape(value)}</div></div>"
        for label, value in cards
    )
    return f"<section class=\"cards\">{inner}</section>"


def render_chart_section(
    title: str,
    series: Series,
    *,
    y_label: str,
    empty: str,
    note: str = "",
    y_min: float | None = None,
    y_max: float | None = None,
    image_path: Path | None = None,
    report_path: Path | None = None,
) -> str:
    if not series.points:
        chart = f"<div class=\"empty\">{escape(empty)}</div>"
    elif image_path is not None and report_path is not None:
        chart = render_image(image_path, report_path, y_label)
    else:
        chart = svg_line_chart(series.points, y_label=y_label, y_min=y_min, y_max=y_max)
    note_html = f"<p class=\"note\">{escape(note)}</p>" if note else ""
    return f"<section><h2>{escape(title)}</h2>{note_html}{chart}</section>"


def render_snapshot_trends_section(
    *,
    capture_series: Series,
    min_distance_series: Series,
    out_of_bounds_series: Series,
    chart_assets: dict[str, Path],
    report_path: Path,
) -> str:
    charts = [
        render_inline_chart(
            "Median Min Distance",
            min_distance_series,
            y_label="meters",
            empty="No median min-distance series available.",
            note=min_distance_series.note,
            image_path=chart_assets.get(min_distance_series.name),
            report_path=report_path,
        ),
        render_inline_chart(
            "Capture %",
            capture_series,
            y_label="capture %",
            empty="No capture percentage series available.",
            note=capture_series.note,
            y_min=0.0,
            y_max=100.0,
            image_path=chart_assets.get(capture_series.name),
            report_path=report_path,
        ),
        render_inline_chart(
            "Out-of-Bounds v. Snapshots",
            out_of_bounds_series,
            y_label="episodes",
            empty="No out-of-bounds series available.",
            note=out_of_bounds_series.note,
            y_min=0.0,
            image_path=chart_assets.get(out_of_bounds_series.name),
            report_path=report_path,
        ),
    ]
    return f"<section><h2>Snapshot Trends</h2><div class=\"chart-grid\">{''.join(charts)}</div></section>"


def render_inline_chart(
    title: str,
    series: Series,
    *,
    y_label: str,
    empty: str,
    note: str = "",
    y_min: float | None = None,
    y_max: float | None = None,
    image_path: Path | None = None,
    report_path: Path | None = None,
) -> str:
    if not series.points:
        chart = f"<div class=\"empty\">{escape(empty)}</div>"
    elif image_path is not None and report_path is not None:
        chart = render_image(image_path, report_path, y_label)
    else:
        chart = svg_line_chart(series.points, y_label=y_label, y_min=y_min, y_max=y_max)
    note_html = f"<p class=\"note\">{escape(note)}</p>" if note else ""
    return f"<div class=\"chart-panel\"><h3>{escape(title)}</h3>{note_html}{chart}</div>"


def write_chart_assets(
    assets_dir: Path,
    *,
    loss_series: Series,
    capture_series: Series,
    min_distance_series: Series,
    out_of_bounds_series: Series,
) -> dict[str, Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    specs = [
        (loss_series, "loss_over_training_steps.png", "training steps", "estimated loss", None, None),
        (min_distance_series, "median_min_distance.png", "training steps", "meters", None, None),
        (capture_series, "capture_percent.png", "training steps", "capture %", 0.0, 100.0),
        (out_of_bounds_series, "out_of_bounds_vs_snapshots.png", "training steps", "episodes", 0.0, None),
    ]
    for series, filename, x_label, y_label, y_min, y_max in specs:
        if not series.points:
            continue
        path = assets_dir / filename
        plot_line_chart(path, series.points, title=series.name, x_label=x_label, y_label=y_label, y_min=y_min, y_max=y_max)
        written[series.name] = path
    return written


def plot_line_chart(
    path: Path,
    points: list[tuple[float, float]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    fig, ax = plt.subplots(figsize=(9.2, 3.8), constrained_layout=True)
    ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.0, color="#1f7a8c")
    ax.set_title(title.replace("_", " "))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_min is not None or y_max is not None:
        current_min, current_max = ax.get_ylim()
        ax.set_ylim(current_min if y_min is None else y_min, current_max if y_max is None else y_max)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs)
    ax.set_xticklabels([format_step(value) for value in xs], rotation=30, ha="right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_image(path: Path, report_path: Path, alt: str) -> str:
    src = path.relative_to(report_path.parent).as_posix()
    return f"<img class=\"chart-img\" src=\"{escape(src)}\" alt=\"{escape(alt)}\">"


def svg_line_chart(
    points: list[tuple[float, float]],
    *,
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    width, height = 920, 320
    left, right, top, bottom = 72, 24, 22, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y = min(ys) if y_min is None else y_min
    max_y = max(ys) if y_max is None else y_max
    if math.isclose(min_x, max_x):
        min_x -= 1.0
        max_x += 1.0
    if math.isclose(min_y, max_y):
        pad = max(abs(min_y) * 0.05, 1.0)
        min_y -= pad
        max_y += pad
    else:
        pad = (max_y - min_y) * 0.08
        if y_min is None:
            min_y -= pad
        if y_max is None:
            max_y += pad

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + (max_y - y) / (max_y - min_y) * plot_h

    polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
    circles = "\n".join(
        (
            f"<circle cx=\"{sx(x):.2f}\" cy=\"{sy(y):.2f}\" r=\"3\">"
            f"<title>step {format_step(x)}: {y:.6g}</title></circle>"
        )
        for x, y in points
    )
    grid = []
    for i in range(5):
        frac = i / 4
        y = top + frac * plot_h
        value = max_y - frac * (max_y - min_y)
        grid.append(f"<line class=\"grid\" x1=\"{left}\" y1=\"{y:.2f}\" x2=\"{width-right}\" y2=\"{y:.2f}\" />")
        grid.append(f"<text class=\"tick\" x=\"{left-10}\" y=\"{y+4:.2f}\" text-anchor=\"end\">{value:.3g}</text>")
    for i in range(5):
        frac = i / 4
        x = left + frac * plot_w
        value = min_x + frac * (max_x - min_x)
        grid.append(f"<line class=\"grid\" x1=\"{x:.2f}\" y1=\"{top}\" x2=\"{x:.2f}\" y2=\"{height-bottom}\" />")
        grid.append(f"<text class=\"tick\" x=\"{x:.2f}\" y=\"{height-bottom+24}\" text-anchor=\"middle\">{format_step(value)}</text>")

    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(y_label)} over training steps">
  <rect class="plot-bg" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" />
  {''.join(grid)}
  <line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" />
  <line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" />
  <polyline class="line" points="{polyline}" />
  <g class="points">{circles}</g>
  <text class="axis-label" x="{width/2}" y="{height-8}" text-anchor="middle">training steps</text>
  <text class="axis-label" transform="translate(16 {height/2}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>
</svg>
"""


def render_table_section(train_rows: list[dict[str, float]], eval_rows: list[dict[str, float]]) -> str:
    latest_train = train_rows[-1] if train_rows else {}
    latest_eval = eval_rows[-1] if eval_rows else {}
    rows = []
    for key in ("global_step", "estimated_ppo_loss", "loss/policy_loss", "loss/value_loss", "loss/entropy", "min_distance_m"):
        value = latest_train.get(key)
        if value is not None:
            rows.append(("latest train", key, value))
    for key in ("global_step", "catch_rate", "min_distance_p50_m", "out_of_bounds", "episodes"):
        value = latest_eval.get(key)
        if value is not None:
            if key == "catch_rate":
                rows.append(("latest eval", "capture_percent", value * 100.0))
            else:
                rows.append(("latest eval", key, value))
    if not rows:
        return ""
    trs = "\n".join(
        f"<tr><td>{escape(source)}</td><td>{escape(key)}</td><td>{value:.6g}</td></tr>"
        for source, key, value in rows
    )
    return f"<section><h2>Latest Values</h2><table><thead><tr><th>Source</th><th>Metric</th><th>Value</th></tr></thead><tbody>{trs}</tbody></table></section>"


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str):
        try:
            out = float(value)
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None


def format_step(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(int(value))


def escape(value: str) -> str:
    return html.escape(value, quote=True)


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return name.strip("._") or "run"


CSS = """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --ink: #1f2933;
  --muted: #667085;
  --line: #1f7a8c;
  --grid: #e5e7eb;
  --border: #d9dee7;
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  max-width: 1080px;
  margin: 0 auto;
  padding: 32px 24px 48px;
}
h1 {
  font-size: 28px;
  margin: 0 0 6px;
}
h2 {
  font-size: 18px;
  margin: 0 0 14px;
}
h3 {
  font-size: 15px;
  margin: 0 0 10px;
}
.path, .note {
  color: var(--muted);
  margin: 0 0 20px;
}
.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 18px;
}
.chart-panel {
  min-width: 0;
}
.chart-panel .note {
  margin-bottom: 10px;
}
section {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 18px;
  margin: 18px 0;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px;
  border: 0;
  background: transparent;
  padding: 0;
}
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
}
.card-label {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}
.card-value {
  font-size: 22px;
  font-weight: 700;
  margin-top: 6px;
}
.missing {
  margin: 0;
  padding-left: 20px;
}
.empty {
  color: var(--muted);
  border: 1px dashed var(--border);
  border-radius: 8px;
  padding: 32px;
  text-align: center;
}
.chart {
  width: 100%;
  height: auto;
  display: block;
}
.chart-img {
  width: 100%;
  height: auto;
  display: block;
  border: 1px solid var(--border);
  border-radius: 8px;
}
.plot-bg {
  fill: #fbfcfe;
}
.grid {
  stroke: var(--grid);
  stroke-width: 1;
}
.axis {
  stroke: #98a2b3;
  stroke-width: 1.5;
}
.line {
  fill: none;
  stroke: var(--line);
  stroke-width: 2.5;
  stroke-linejoin: round;
  stroke-linecap: round;
}
.points circle {
  fill: var(--line);
}
.tick {
  fill: var(--muted);
  font-size: 11px;
}
.axis-label {
  fill: var(--muted);
  font-size: 12px;
}
table {
  border-collapse: collapse;
  width: 100%;
}
th, td {
  border-bottom: 1px solid var(--border);
  padding: 8px;
  text-align: left;
}
th {
  color: var(--muted);
  font-weight: 600;
}
"""


INDEX_CSS = """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --ink: #1f2933;
  --muted: #667085;
  --border: #d9dee7;
  --link: #1f7a8c;
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  max-width: 820px;
  margin: 0 auto;
  padding: 32px 24px 48px;
}
h1 {
  font-size: 28px;
  margin: 0 0 6px;
}
p {
  color: var(--muted);
  margin: 0 0 20px;
}
ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 10px;
}
li {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
}
a {
  color: var(--link);
  display: block;
  font-weight: 700;
  padding: 14px 16px;
  text-decoration: none;
}
a:hover {
  text-decoration: underline;
}
code {
  color: var(--ink);
}
.empty {
  color: var(--muted);
  padding: 14px 16px;
}
"""


if __name__ == "__main__":
    main()
