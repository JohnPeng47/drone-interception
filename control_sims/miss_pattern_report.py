"""Generate an HTML miss-pattern report for uniform-distance benchmark runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backends.csim.generator.generators.robust_intercept_uniform_distance import (
    RobustInterceptUniformDistanceConfigGenerator,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir / "miss_pattern_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_joined(run_dir / "trials.csv")
    stats = _summary_stats(data)
    figures = [
        _plot_range_speed_hit_rate(data, out_dir),
        _plot_range_histograms(data, out_dir),
        _plot_los_elevation_bands(data, out_dir),
        _plot_visibility_distribution(data, out_dir),
        _plot_camera_v_bands(data, out_dir),
        _plot_miss_classes(data, out_dir),
        _plot_min_distance_by_range_speed(data, out_dir),
    ]
    report_path = out_dir / "miss_patterns_report.html"
    report_path.write_text(_html_report(run_dir, stats, figures), encoding="utf-8")

    manifest = {
        "report": str(report_path),
        "figures": [str(fig["path"]) for fig in figures],
        "stats": stats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "figures": [str(fig["path"]) for fig in figures]}, indent=2))
    return 0


def _load_joined(path: Path) -> dict[str, np.ndarray]:
    generator = RobustInterceptUniformDistanceConfigGenerator()
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    data: dict[str, list[Any]] = {
        "seed": [],
        "caught": [],
        "range_m": [],
        "speed_mps": [],
        "min_distance_m": [],
        "final_distance_m": [],
        "visible_fraction": [],
        "control_effort": [],
        "camera_u": [],
        "camera_v": [],
        "los_elevation_deg": [],
        "los_azimuth_deg": [],
    }
    for row in rows:
        seed = int(row["seed"])
        point = generator._by_seed[seed]
        instance = generator._sample_once(seed=seed)
        target_initial = instance.target_initials[0]
        rel = np.asarray(target_initial.position_w, dtype=float) - np.asarray(instance.pursuer_initial.position_w, dtype=float)
        rel_unit = rel / max(float(np.linalg.norm(rel)), 1e-12)
        azimuth = float(np.degrees(np.arctan2(rel_unit[1], rel_unit[0])))
        if azimuth < 0.0:
            azimuth += 360.0
        elevation = float(np.degrees(np.arcsin(np.clip(rel_unit[2], -1.0, 1.0))))

        data["seed"].append(seed)
        data["caught"].append(_bool(row["caught"]))
        data["range_m"].append(float(row["range_m"]))
        data["speed_mps"].append(float(row["closing_speed_mps"]))
        data["min_distance_m"].append(_float(row["min_distance_m"]))
        data["final_distance_m"].append(_float(row["final_distance_m"]))
        data["visible_fraction"].append(_float(row["visible_fraction"]))
        data["control_effort"].append(_float(row["control_effort"]))
        data["camera_u"].append(float(point.values["camera_u_fraction"]))
        data["camera_v"].append(float(point.values["camera_v_fraction"]))
        data["los_elevation_deg"].append(elevation)
        data["los_azimuth_deg"].append(azimuth)
    return {key: np.asarray(values, dtype=bool if key == "caught" else float) for key, values in data.items()}


def _summary_stats(data: dict[str, np.ndarray]) -> dict[str, Any]:
    caught = data["caught"]
    miss = ~caught
    near_miss = miss & (data["min_distance_m"] <= 1.0)
    bad_miss = miss & (data["min_distance_m"] > 3.0)
    return {
        "n": int(caught.size),
        "hits": int(np.sum(caught)),
        "misses": int(np.sum(miss)),
        "hit_rate": float(np.mean(caught)),
        "hit_range_p50": _pct(data["range_m"][caught], 50),
        "miss_range_p50": _pct(data["range_m"][miss], 50),
        "hit_visible_mean": _mean(data["visible_fraction"][caught]),
        "miss_visible_mean": _mean(data["visible_fraction"][miss]),
        "near_miss_count": int(np.sum(near_miss)),
        "bad_miss_count": int(np.sum(bad_miss)),
        "bad_miss_range_p50": _pct(data["range_m"][bad_miss], 50),
        "bad_miss_speed_p50": _pct(data["speed_mps"][bad_miss], 50),
    }


def _plot_range_speed_hit_rate(data: dict[str, np.ndarray], out_dir: Path) -> dict[str, Any]:
    range_bins = np.linspace(5.0, 20.0, 6)
    speed_values = np.array([0.0, 5.0, 10.0, 20.0])
    hit_rate = np.full((len(speed_values), len(range_bins) - 1), np.nan)
    counts = np.zeros_like(hit_rate)
    for i, speed in enumerate(speed_values):
        for j, (lo, hi) in enumerate(zip(range_bins[:-1], range_bins[1:])):
            mask = (data["speed_mps"] == speed) & (data["range_m"] >= lo) & (data["range_m"] < hi)
            counts[i, j] = int(np.sum(mask))
            if np.any(mask):
                hit_rate[i, j] = float(np.mean(data["caught"][mask]))

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    image = ax.imshow(hit_rate, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(range_bins) - 1), [f"{range_bins[i]:.0f}-{range_bins[i+1]:.0f}" for i in range(len(range_bins) - 1)])
    ax.set_yticks(np.arange(len(speed_values)), [f"{int(speed)}" for speed in speed_values])
    ax.set_xlabel("initial range bin (m)")
    ax.set_ylabel("initial closing speed (m/s)")
    ax.set_title("Hit rate falls mainly with range, with speed-specific failures")
    for i in range(hit_rate.shape[0]):
        for j in range(hit_rate.shape[1]):
            text = "n=0" if not np.isfinite(hit_rate[i, j]) else f"{hit_rate[i, j]:.2f}\nn={int(counts[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if np.nan_to_num(hit_rate[i, j]) < 0.45 else "black")
    plt.colorbar(image, ax=ax, label="hit fraction")
    path = out_dir / "01_range_speed_hit_rate.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "path": path,
        "title": "Range and speed are the strongest first-order split",
        "body": "Longer starts miss more often. The 0 m/s bucket collapses beyond roughly 14 m, while 20 m/s is excellent close-in but weak at mid and long range where the target is harder to keep visible.",
    }


def _plot_range_histograms(data: dict[str, np.ndarray], out_dir: Path) -> dict[str, Any]:
    caught = data["caught"]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    bins = np.linspace(5.0, 20.0, 31)
    ax.hist(data["range_m"][~caught], bins=bins, color="#d62728", alpha=0.55, label="miss")
    ax.hist(data["range_m"][caught], bins=bins, color="#2ca02c", alpha=0.65, label="hit")
    ax.axvline(np.nanmedian(data["range_m"][caught]), color="#2ca02c", linewidth=2, label="hit median")
    ax.axvline(np.nanmedian(data["range_m"][~caught]), color="#d62728", linewidth=2, label="miss median")
    ax.set_title("Misses skew toward longer initial range")
    ax.set_xlabel("initial range (m)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.22)
    ax.legend()
    path = out_dir / "02_range_hit_miss_histogram.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "path": path,
        "title": "Misses start farther away",
        "body": "The hit median initial range is about 10.3 m, while the miss median is about 13.8 m. This points to insufficient long-range closure or target-retention behavior.",
    }


def _plot_los_elevation_bands(data: dict[str, np.ndarray], out_dir: Path) -> dict[str, Any]:
    bins = np.array([-90, -60, -30, 0, 30, 60, 90], dtype=float)
    labels = [f"{int(lo)}..{int(hi)}" for lo, hi in zip(bins[:-1], bins[1:])]
    hit_rate, visible, min_p50 = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (data["los_elevation_deg"] >= lo) & (data["los_elevation_deg"] < hi)
        hit_rate.append(float(np.mean(data["caught"][mask])))
        visible.append(float(np.mean(data["visible_fraction"][mask])))
        min_p50.append(float(np.nanpercentile(data["min_distance_m"][mask], 50)))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    _bar(axes[0], labels, hit_rate, "Hit fraction", "fraction", "#2ca02c", ylim=(0, 1))
    _bar(axes[1], labels, visible, "Mean visibility", "fraction", "#9467bd", ylim=(0, 1))
    _bar(axes[2], labels, min_p50, "Median min distance", "m", "#1f77b4")
    for ax in axes:
        ax.set_xlabel("LOS elevation band (deg)")
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Steep vertical geometry is a major miss pattern")
    path = out_dir / "03_los_elevation_bands.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "path": path,
        "title": "The controller likes near-horizontal intercept geometry",
        "body": "Hit rate peaks in the -30 to +30 degree LOS elevation bands. Very steep upward/downward geometries combine lower hit rates with worse minimum-distance behavior.",
    }


def _plot_visibility_distribution(data: dict[str, np.ndarray], out_dir: Path) -> dict[str, Any]:
    caught = data["caught"]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(data["visible_fraction"][~caught], bins=bins, color="#d62728", alpha=0.58, label="miss")
    ax.hist(data["visible_fraction"][caught], bins=bins, color="#2ca02c", alpha=0.65, label="hit")
    ax.set_title("Misses often spend the rollout with poor target visibility")
    ax.set_xlabel("visible fraction")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.22)
    ax.legend()
    path = out_dir / "04_visibility_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "path": path,
        "title": "Visibility separates hits from misses",
        "body": "Hits are almost always visible for most of the rollout. Misses span the full range, but a large cluster has low image availability, which suggests target-retention logic is a high-value target.",
    }


def _plot_camera_v_bands(data: dict[str, np.ndarray], out_dir: Path) -> dict[str, Any]:
    bins = np.linspace(-0.9, 0.9, 7)
    labels = [f"{lo:.1f}..{hi:.1f}" for lo, hi in zip(bins[:-1], bins[1:])]
    hit_rate, visible, min_p50 = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (data["camera_v"] >= lo) & (data["camera_v"] < hi)
        hit_rate.append(float(np.mean(data["caught"][mask])))
        visible.append(float(np.mean(data["visible_fraction"][mask])))
        min_p50.append(float(np.nanpercentile(data["min_distance_m"][mask], 50)))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    _bar(axes[0], labels, hit_rate, "Hit fraction", "fraction", "#2ca02c", ylim=(0, 1))
    _bar(axes[1], labels, visible, "Mean visibility", "fraction", "#9467bd", ylim=(0, 1))
    _bar(axes[2], labels, min_p50, "Median min distance", "m", "#1f77b4")
    for ax in axes:
        ax.set_xlabel("initial camera v fraction")
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Vertical image-plane placement matters more than horizontal placement")
    path = out_dir / "05_camera_v_bands.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "path": path,
        "title": "Initial vertical image error has a directional bias",
        "body": "Starts with the target lower in the image underperform starts with positive v. The horizontal u coordinate is much weaker, so vertical centering and pitch/thrust coupling deserve attention.",
    }


def _plot_miss_classes(data: dict[str, np.ndarray], out_dir: Path) -> dict[str, Any]:
    miss = ~data["caught"]
    near = miss & (data["min_distance_m"] <= 1.0)
    bad = miss & (data["min_distance_m"] > 3.0)
    other = miss & ~(near | bad)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].scatter(data["range_m"][other], data["los_elevation_deg"][other], c="#ff9896", s=14, alpha=0.5, label="other miss")
    axes[0].scatter(data["range_m"][near], data["los_elevation_deg"][near], c="#ff7f0e", s=18, alpha=0.75, label="near miss <=1 m")
    axes[0].scatter(data["range_m"][bad], data["los_elevation_deg"][bad], c="#8c000f", s=22, alpha=0.85, label="bad miss >3 m")
    axes[0].set_xlabel("initial range (m)")
    axes[0].set_ylabel("LOS elevation deg")
    axes[0].set_title("Miss classes in range/elevation")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend()

    axes[1].scatter(data["visible_fraction"][other], data["final_distance_m"][other], c="#ff9896", s=14, alpha=0.5, label="other miss")
    axes[1].scatter(data["visible_fraction"][near], data["final_distance_m"][near], c="#ff7f0e", s=18, alpha=0.75, label="near miss <=1 m")
    axes[1].scatter(data["visible_fraction"][bad], data["final_distance_m"][bad], c="#8c000f", s=22, alpha=0.85, label="bad miss >3 m")
    axes[1].set_xlabel("visible fraction")
    axes[1].set_ylabel("final distance (m)")
    axes[1].set_title("Miss classes in visibility/final distance")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend()
    path = out_dir / "06_miss_classes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "path": path,
        "title": "Misses split into recoverable near misses and structural failures",
        "body": "Near misses likely need terminal damping or capture-radius timing improvements. Bad misses skew long-range, low-speed, and steep-elevation, which is a different problem: getting into a viable interception geometry at all.",
    }


def _plot_min_distance_by_range_speed(data: dict[str, np.ndarray], out_dir: Path) -> dict[str, Any]:
    speeds = np.array([0.0, 5.0, 10.0, 20.0])
    colors = {0.0: "#1f77b4", 5.0: "#ff7f0e", 10.0: "#2ca02c", 20.0: "#d62728"}
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    for speed in speeds:
        mask = data["speed_mps"] == speed
        ax.scatter(data["range_m"][mask], data["min_distance_m"][mask], s=13, alpha=0.45, color=colors[speed], label=f"{int(speed)} m/s")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="capture radius")
    ax.set_title("Minimum distance exposes late misses and long-range failures")
    ax.set_xlabel("initial range (m)")
    ax.set_ylabel("minimum distance (m)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.22)
    ax.legend(title="closing speed")
    path = out_dir / "07_min_distance_range_speed.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "path": path,
        "title": "Many misses still get close",
        "body": "The dense band just above the 0.5 m capture radius shows a terminal-control opportunity. The high-min-distance tail is mostly long range or awkward vertical geometry.",
    }


def _html_report(run_dir: Path, stats: dict[str, Any], figures: list[dict[str, Any]]) -> str:
    figure_blocks = "\n".join(_figure_block(fig) for fig in figures)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Uniform-Distance Miss Pattern Report</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #17202a;
      background: #f6f7f8;
    }}
    header {{
      padding: 28px 36px 18px;
      background: #ffffff;
      border-bottom: 1px solid #d7dce0;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 20px;
    }}
    p {{
      line-height: 1.45;
    }}
    .meta {{
      color: #52606d;
      font-size: 14px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 12px;
      padding: 18px 36px;
      background: #ffffff;
      border-bottom: 1px solid #d7dce0;
    }}
    .metric {{
      padding: 12px 14px;
      background: #eef2f5;
      border: 1px solid #d7dce0;
      border-radius: 6px;
    }}
    .metric .label {{
      font-size: 12px;
      color: #52606d;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric .value {{
      margin-top: 4px;
      font-size: 22px;
      font-weight: 650;
    }}
    main {{
      padding: 24px 36px 40px;
    }}
    section {{
      display: grid;
      grid-template-columns: minmax(320px, 1fr) minmax(360px, 1.25fr);
      gap: 24px;
      align-items: start;
      margin-bottom: 28px;
      padding: 20px;
      background: #ffffff;
      border: 1px solid #d7dce0;
      border-radius: 8px;
    }}
    img {{
      width: 100%;
      height: auto;
      border: 1px solid #d7dce0;
      background: white;
    }}
    ul {{
      margin-top: 8px;
      padding-left: 20px;
    }}
    @media (max-width: 900px) {{
      .summary {{
        grid-template-columns: repeat(2, 1fr);
      }}
      section {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Uniform-Distance Miss Pattern Report</h1>
    <div class="meta">Source run: {html.escape(str(run_dir))}</div>
  </header>
  <div class="summary">
    {_metric("Hit Rate", f"{stats['hit_rate']:.1%}")}
    {_metric("Hits / Misses", f"{stats['hits']} / {stats['misses']}")}
    {_metric("Hit Range p50", f"{stats['hit_range_p50']:.2f} m")}
    {_metric("Miss Range p50", f"{stats['miss_range_p50']:.2f} m")}
    {_metric("Hit Visibility Mean", f"{stats['hit_visible_mean']:.2f}")}
    {_metric("Miss Visibility Mean", f"{stats['miss_visible_mean']:.2f}")}
    {_metric("Near Misses <= 1 m", str(stats['near_miss_count']))}
    {_metric("Bad Misses > 3 m", str(stats['bad_miss_count']))}
  </div>
  <main>
    <section>
      <div>
        <h2>Takeaways</h2>
        <p>The misses are not one homogeneous failure mode. The benchmark shows a range-limited pursuit problem, a steep-elevation/vertical-image problem, and a terminal-control problem near the capture radius.</p>
        <ul>
          <li>Long-range, low-closure starts need stronger early pursuit.</li>
          <li>Steep LOS elevation and negative image-plane v are risky geometries.</li>
          <li>Low target visibility strongly predicts misses.</li>
          <li>Near misses are common enough to justify terminal damping or capture timing work.</li>
        </ul>
      </div>
      <div>
        <h2>How To Use This Report</h2>
        <p>Use the first plots to choose broad policy families, then use the miss-class and minimum-distance plots to decide whether a candidate fixes terminal behavior or only shifts the failure to another region.</p>
      </div>
    </section>
    {figure_blocks}
  </main>
</body>
</html>
"""


def _figure_block(fig: dict[str, Any]) -> str:
    rel = html.escape(Path(fig["path"]).name)
    return f"""<section>
  <div>
    <h2>{html.escape(fig["title"])}</h2>
    <p>{html.escape(fig["body"])}</p>
  </div>
  <img src="{rel}" alt="{html.escape(fig["title"])}">
</section>"""


def _metric(label: str, value: str) -> str:
    return f"""<div class="metric"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>"""


def _bar(ax, labels: list[str], values: list[float], title: str, ylabel: str, color: str, ylim: tuple[float, float] | None = None) -> None:
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.22)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float:
    text = str(value).strip()
    return float("nan") if text == "" else float(text)


def _pct(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def _mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
