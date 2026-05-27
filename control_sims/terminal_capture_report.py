"""Analyze near-miss terminal capture failures for uniform-distance benchmarks."""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backends.csim.generator.generators.robust_intercept_uniform_distance import (
    RobustInterceptUniformDistanceConfigGenerator,
)
from control_sims.beihang_minimal_sim.config import (
    CameraConfig,
    TargetConfig,
    TrialConfig,
    VehicleConfig,
)
from control_sims.beihang_minimal_sim.replay import run_trial


CAPTURE_RADIUS_M = 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--near-max-m", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir / "terminal_capture_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "near_miss_terminal_metrics.csv"
    if csv_path.exists() and not args.recompute:
        terminal_rows = _load_terminal_rows(csv_path)
    else:
        generator = RobustInterceptUniformDistanceConfigGenerator()
        benchmark_rows = _load_rows(run_dir / "trials.csv")
        near_rows = [
            row for row in benchmark_rows
            if not _bool(row["caught"])
            and CAPTURE_RADIUS_M < _float(row["min_distance_m"]) <= float(args.near_max_m)
        ]
        terminal_rows = [_analyze_seed(generator, int(row["seed"])) for row in near_rows]
        _write_csv(csv_path, terminal_rows)
    png_path = out_dir / "terminal_capture_near_miss_dashboard.png"
    _plot_dashboard(terminal_rows, png_path, near_max_m=float(args.near_max_m))
    panel_paths = _plot_panels(terminal_rows, out_dir / "figures")
    html_path = out_dir / "terminal_capture_report.html"
    summary = _summary(terminal_rows)
    html_path.write_text(
        _html_report(
            run_dir=run_dir,
            panel_paths=panel_paths,
            csv_path=csv_path,
            near_max_m=float(args.near_max_m),
            summary=summary,
        ),
        encoding="utf-8",
    )
    manifest = {
        "near_max_m": float(args.near_max_m),
        "near_miss_count": len(terminal_rows),
        "metrics_csv": str(csv_path),
        "dashboard_png": str(png_path),
        "html_report": str(html_path),
        "panel_pngs": {panel["id"]: str(panel["path"]) for panel in panel_paths},
        "summary": summary,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _analyze_seed(generator: RobustInterceptUniformDistanceConfigGenerator, seed: int) -> dict[str, Any]:
    point = generator._by_seed[seed]
    instance = generator._sample_once(seed=seed)
    config = _minimal_config(instance)
    metrics, samples = run_trial(config)
    if metrics is None or not samples:
        raise RuntimeError(f"seed {seed} produced no minimal-sim samples")

    distances = np.array([
        np.linalg.norm(sample.target.position_w - sample.vehicle.position_w)
        for sample in samples
    ], dtype=float)
    idx = int(np.nanargmin(distances))
    sample = samples[idx]
    rel = np.asarray(sample.target.position_w, dtype=float) - np.asarray(sample.vehicle.position_w, dtype=float)
    rel_vel = np.asarray(sample.target.velocity_w, dtype=float) - np.asarray(sample.vehicle.velocity_w, dtype=float)
    dist = float(distances[idx])
    dist_rate = float(np.dot(rel, rel_vel) / max(dist, 1e-12))
    closing_rate = -dist_rate
    image_error = (
        float("nan") if not sample.feature.detected or sample.feature.uv_norm is None
        else float(np.linalg.norm(sample.feature.uv_norm))
    )
    thrust_axis_w = np.asarray(sample.vehicle.rotation_wb, dtype=float)[:, 2]
    rel_unit = rel / max(dist, 1e-12)
    thrust_alignment = float(np.dot(thrust_axis_w, rel_unit))
    rel_elevation = float(np.degrees(np.arcsin(np.clip(rel_unit[2], -1.0, 1.0))))
    horizontal_distance = float(np.linalg.norm(rel[:2]))
    vertical_error = float(rel[2])
    thrust_n = float(sample.command.thrust_n)
    thrust_fraction = thrust_n / max(float(config.vehicle.max_thrust_n), 1e-12)

    visible_window = _window_fraction(samples, idx, key="visible", radius=20)
    image_error_window = _window_mean(samples, idx, key="image_error", radius=20)
    closing_rate_before = _distance_rate_at(samples, max(0, idx - 20))
    closing_rate_after = _distance_rate_at(samples, min(len(samples) - 1, idx + 20))

    return {
        "seed": seed,
        "range_m": float(point.values["range_m"]),
        "closing_speed_mps": float(point.values["closing_speed_mps"]),
        "camera_u": float(point.values["camera_u_fraction"]),
        "camera_v": float(point.values["camera_v_fraction"]),
        "t_min_s": float(sample.t),
        "min_distance_m": dist,
        "capture_gap_m": dist - CAPTURE_RADIUS_M,
        "final_distance_m": float(metrics.distance_m),
        "visible_fraction": _visible_fraction(samples),
        "visible_window_fraction": visible_window,
        "image_error_at_min": image_error,
        "image_error_window_mean": image_error_window,
        "closing_rate_at_min_mps": closing_rate,
        "closing_rate_before_mps": closing_rate_before,
        "closing_rate_after_mps": closing_rate_after,
        "rel_elevation_deg_at_min": rel_elevation,
        "horizontal_distance_m_at_min": horizontal_distance,
        "vertical_error_m_at_min": vertical_error,
        "thrust_n_at_min": thrust_n,
        "thrust_fraction_at_min": thrust_fraction,
        "body_rate_norm_at_min": float(np.linalg.norm(sample.command.body_rates_b)),
        "thrust_axis_target_alignment": thrust_alignment,
    }


def _minimal_config(instance) -> TrialConfig:
    target = instance.config.targets[0]
    target_initial = instance.target_initials[0]
    camera = instance.config.cameras[0]
    return TrialConfig(
        dt=float(instance.config.options.backend_dt),
        duration_s=float(instance.config.options.duration_s),
        capture_radius_m=float(instance.config.intercept_radius_m),
        arena_min_w=(-100.0, -100.0, -100.0),
        arena_max_w=(100.0, 100.0, 100.0),
        vehicle=VehicleConfig(
            mass_kg=float(instance.config.pursuer.mass_kg),
            max_thrust_n=float(instance.config.max_thrust_n),
            max_body_rate_rad_s=float(instance.config.max_rate_rps),
            initial_position_w=tuple(float(x) for x in instance.pursuer_initial.position_w),
            initial_velocity_w=tuple(float(x) for x in instance.pursuer_initial.velocity_w),
            initial_quat_xyzw=tuple(float(x) for x in instance.pursuer_initial.quat_xyzw),
        ),
        target=TargetConfig(
            radius_m=float(target.radius_m),
            initial_position_w=tuple(float(x) for x in target_initial.position_w),
            base_velocity_w=tuple(float(x) for x in target_initial.velocity_w),
            weave_amplitude_m=(0.0, 0.0),
            weave_frequency_hz=(0.0, 0.0),
        ),
        camera=CameraConfig(
            body_to_camera=tuple(tuple(float(v) for v in row) for row in camera.body_to_camera),
            max_uv_norm=max(
                float(np.tan(float(camera.intrinsics.hfov_rad) / 2.0)),
                float(np.tan(float(camera.intrinsics.vfov_rad) / 2.0)),
            ),
            min_depth_m=0.1,
        ),
    )


def _plot_dashboard(rows: list[dict[str, Any]], path: Path, *, near_max_m: float) -> None:
    data = {key: np.array([row[key] for row in rows], dtype=float) for key in rows[0] if key != "seed"}
    speed = data["closing_speed_mps"]
    colors = _speed_colors(speed)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    fig.suptitle(
        f"Terminal capture analysis: missed trials with {CAPTURE_RADIUS_M:.1f} m < min distance <= {near_max_m:.1f} m",
        fontsize=18,
    )

    ax = axes[0, 0]
    ax.scatter(data["range_m"], data["capture_gap_m"], c=colors, s=26, alpha=0.78)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="capture boundary")
    ax.set_title("Capture gap vs initial range")
    ax.set_xlabel("initial range (m)")
    ax.set_ylabel("min distance - capture radius (m)")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    bins = np.linspace(0.0, 3.0, 31)
    for speed_value, color in _SPEED_COLORS.items():
        mask = speed == speed_value
        if np.any(mask):
            ax.hist(data["t_min_s"][mask], bins=bins, color=color, alpha=0.45, label=f"{int(speed_value)} m/s")
    ax.set_title("Closest approach timing")
    ax.set_xlabel("time of minimum distance (s)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.legend(title="initial speed")

    ax = axes[0, 2]
    ax.scatter(data["closing_rate_at_min_mps"], data["capture_gap_m"], c=colors, s=26, alpha=0.78)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Radial closure at closest approach")
    ax.set_xlabel("closing rate at min distance (m/s)")
    ax.set_ylabel("capture gap (m)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.scatter(data["rel_elevation_deg_at_min"], data["capture_gap_m"], c=colors, s=26, alpha=0.78)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.axvspan(-30, 30, color="#2ca02c", alpha=0.10, label="high-hit elevation band")
    ax.set_title("Terminal elevation and underactuation")
    ax.set_xlabel("relative elevation at closest approach (deg)")
    ax.set_ylabel("capture gap (m)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.scatter(data["horizontal_distance_m_at_min"], data["vertical_error_m_at_min"], c=data["capture_gap_m"], cmap="magma", s=30, alpha=0.82)
    circle = plt.Circle((0.0, 0.0), CAPTURE_RADIUS_M, fill=False, color="black", linestyle="--", linewidth=1.2)
    ax.add_patch(circle)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Terminal miss decomposition")
    ax.set_xlabel("horizontal separation at min (m)")
    ax.set_ylabel("vertical error at min (m)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 2]
    ax.scatter(data["thrust_axis_target_alignment"], data["capture_gap_m"], c=colors, s=26, alpha=0.78)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Thrust-axis alignment at closest approach")
    ax.set_xlabel("dot(body z, target direction)")
    ax.set_ylabel("capture gap (m)")
    ax.grid(True, alpha=0.25)

    ax = axes[2, 0]
    ax.scatter(data["visible_window_fraction"], data["capture_gap_m"], c=colors, s=26, alpha=0.78)
    ax.set_title("Local visibility near closest approach")
    ax.set_xlabel("visible fraction in +/-20 samples")
    ax.set_ylabel("capture gap (m)")
    ax.grid(True, alpha=0.25)

    ax = axes[2, 1]
    ax.scatter(data["image_error_window_mean"], data["capture_gap_m"], c=colors, s=26, alpha=0.78)
    ax.set_title("Image centering near closest approach")
    ax.set_xlabel("mean image error in +/-20 samples")
    ax.set_ylabel("capture gap (m)")
    ax.grid(True, alpha=0.25)

    ax = axes[2, 2]
    ax.scatter(data["thrust_fraction_at_min"], data["body_rate_norm_at_min"], c=colors, s=26, alpha=0.78)
    ax.set_title("Command usage at closest approach")
    ax.set_xlabel("thrust / max thrust")
    ax.set_ylabel("body-rate norm (rad/s)")
    ax.grid(True, alpha=0.25)
    _legend_for_speeds(ax)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_panels(rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {key: np.array([row[key] for row in rows], dtype=float) for key in rows[0] if key != "seed"}
    speed = data["closing_speed_mps"]
    colors = _speed_colors(speed)
    panels: list[dict[str, Any]] = []

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.scatter(data["range_m"], data["capture_gap_m"], c=colors, s=28, alpha=0.78)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Capture gap vs initial range")
    ax.set_xlabel("initial range (m)")
    ax.set_ylabel("min distance - capture radius (m)")
    ax.grid(True, alpha=0.25)
    _legend_for_speeds(ax)
    panels.append(_save_panel(
        fig,
        out_dir / "01_capture_gap_vs_range.png",
        "capture_gap_vs_range",
        "Capture gap vs initial range",
        "The gap is measured above the 0.5 m capture radius. Most near misses are genuinely close, but the longer-range starts leave less terminal margin.",
    ))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    bins = np.linspace(0.0, 3.0, 31)
    for speed_value, color in _SPEED_COLORS.items():
        mask = speed == speed_value
        if np.any(mask):
            ax.hist(data["t_min_s"][mask], bins=bins, color=color, alpha=0.45, label=f"{int(speed_value)} m/s")
    ax.set_title("Closest approach timing")
    ax.set_xlabel("time of minimum distance (s)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.legend(title="initial speed")
    panels.append(_save_panel(
        fig,
        out_dir / "02_closest_approach_timing.png",
        "closest_approach_timing",
        "Closest approach timing",
        "The misses cluster around the terminal phase rather than timing out late. This supports treating them as final-capture failures.",
    ))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.scatter(data["closing_rate_at_min_mps"], data["capture_gap_m"], c=colors, s=28, alpha=0.78)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Radial closure at closest approach")
    ax.set_xlabel("closing rate at min distance (m/s)")
    ax.set_ylabel("capture gap (m)")
    ax.grid(True, alpha=0.25)
    panels.append(_save_panel(
        fig,
        out_dir / "03_radial_closure.png",
        "radial_closure",
        "Radial closure at closest approach",
        "The median closure rate is essentially zero, which is the signature of grazing the capture sphere instead of driving through it.",
    ))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.scatter(data["rel_elevation_deg_at_min"], data["capture_gap_m"], c=colors, s=28, alpha=0.78)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.axvspan(-30, 30, color="#2ca02c", alpha=0.10, label="high-hit elevation band")
    ax.set_title("Terminal elevation and underactuation")
    ax.set_xlabel("relative elevation at closest approach (deg)")
    ax.set_ylabel("capture gap (m)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    panels.append(_save_panel(
        fig,
        out_dir / "04_terminal_elevation.png",
        "terminal_elevation",
        "Terminal elevation and underactuation",
        "The elevation band matters because a quadcopter cannot command arbitrary force toward the target. Final geometries outside the favorable band ask the thrust axis and body rates to solve a harder alignment problem.",
    ))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    scatter = ax.scatter(
        data["horizontal_distance_m_at_min"],
        data["vertical_error_m_at_min"],
        c=data["capture_gap_m"],
        cmap="magma",
        s=32,
        alpha=0.82,
    )
    circle = plt.Circle((0.0, 0.0), CAPTURE_RADIUS_M, fill=False, color="black", linestyle="--", linewidth=1.2)
    ax.add_patch(circle)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Terminal miss decomposition")
    ax.set_xlabel("horizontal separation at min (m)")
    ax.set_ylabel("vertical error at min (m)")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="capture gap (m)")
    panels.append(_save_panel(
        fig,
        out_dir / "05_miss_decomposition.png",
        "miss_decomposition",
        "Terminal miss decomposition",
        "This separates sideways miss distance from vertical miss distance. Vertical offsets are especially important here because the vehicle only accelerates by reorienting its thrust axis.",
    ))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.scatter(data["thrust_axis_target_alignment"], data["capture_gap_m"], c=colors, s=28, alpha=0.78)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Thrust-axis alignment at closest approach")
    ax.set_xlabel("dot(body z, target direction)")
    ax.set_ylabel("capture gap (m)")
    ax.grid(True, alpha=0.25)
    panels.append(_save_panel(
        fig,
        out_dir / "06_thrust_axis_alignment.png",
        "thrust_axis_alignment",
        "Thrust-axis alignment",
        "Poor alignment means the controller reaches closest approach without putting the available acceleration authority in a useful terminal direction.",
    ))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.scatter(data["visible_window_fraction"], data["capture_gap_m"], c=colors, s=28, alpha=0.78)
    ax.set_title("Local visibility near closest approach")
    ax.set_xlabel("visible fraction in +/-20 samples")
    ax.set_ylabel("capture gap (m)")
    ax.grid(True, alpha=0.25)
    panels.append(_save_panel(
        fig,
        out_dir / "07_local_visibility.png",
        "local_visibility",
        "Local visibility near closest approach",
        "Many near misses have little target visibility in the local terminal window. Those cases need target retention or a less aggressive attitude schedule, not just stronger pursuit.",
    ))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.scatter(data["image_error_window_mean"], data["capture_gap_m"], c=colors, s=28, alpha=0.78)
    ax.set_title("Image centering near closest approach")
    ax.set_xlabel("mean image error in +/-20 samples")
    ax.set_ylabel("capture gap (m)")
    ax.grid(True, alpha=0.25)
    panels.append(_save_panel(
        fig,
        out_dir / "08_image_centering.png",
        "image_centering",
        "Image centering near closest approach",
        "High image error in the final window suggests the target is near the edge of the camera frame, which weakens the terminal correction loop.",
    ))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.scatter(data["thrust_fraction_at_min"], data["body_rate_norm_at_min"], c=colors, s=28, alpha=0.78)
    ax.set_title("Command usage at closest approach")
    ax.set_xlabel("thrust / max thrust")
    ax.set_ylabel("body-rate norm (rad/s)")
    ax.grid(True, alpha=0.25)
    _legend_for_speeds(ax)
    panels.append(_save_panel(
        fig,
        out_dir / "09_command_usage.png",
        "command_usage",
        "Command usage at closest approach",
        "This shows whether the terminal failure coincides with command saturation. The body-rate concentration is a useful signal for whether the policy is authority-limited or geometry-limited.",
    ))

    return panels


def _save_panel(fig, path: Path, panel_id: str, title: str, interpretation: str) -> dict[str, Any]:
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "id": panel_id,
        "title": title,
        "path": path,
        "interpretation": interpretation,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    data = {key: np.array([row[key] for row in rows], dtype=float) for key in rows[0] if key != "seed"}
    return {
        "n": len(rows),
        "capture_gap_p50_m": _pct(data["capture_gap_m"], 50),
        "capture_gap_p90_m": _pct(data["capture_gap_m"], 90),
        "t_min_p50_s": _pct(data["t_min_s"], 50),
        "closing_rate_at_min_p50_mps": _pct(data["closing_rate_at_min_mps"], 50),
        "visible_window_fraction_mean": float(np.nanmean(data["visible_window_fraction"])),
        "image_error_window_mean": float(np.nanmean(data["image_error_window_mean"])),
        "thrust_fraction_at_min_p50": _pct(data["thrust_fraction_at_min"], 50),
        "body_rate_norm_at_min_p50": _pct(data["body_rate_norm_at_min"], 50),
    }


def _html_report(
    *,
    run_dir: Path,
    panel_paths: list[dict[str, Any]],
    csv_path: Path,
    near_max_m: float,
    summary: dict[str, Any],
) -> str:
    panel_sections = "\n".join(_html_panel(panel) for panel in panel_paths)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Terminal Capture Near-Miss Report</title>
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
      margin-bottom: 28px;
      padding: 20px;
      background: #ffffff;
      border: 1px solid #d7dce0;
      border-radius: 8px;
    }}
    .split {{
      display: grid;
      grid-template-columns: minmax(320px, 0.75fr) minmax(480px, 1.25fr);
      gap: 24px;
      align-items: start;
    }}
    .framing {{
      max-width: 980px;
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
    code {{
      background: #eef2f5;
      padding: 2px 4px;
      border-radius: 4px;
    }}
    @media (max-width: 1000px) {{
      .summary {{
        grid-template-columns: repeat(2, 1fr);
      }}
      .split {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Terminal Capture Near-Miss Report</h1>
    <div class="meta">Source run: {html.escape(str(run_dir))}</div>
    <div class="meta">Scope: missed trials with {CAPTURE_RADIUS_M:.1f} m &lt; min distance &lt;= {near_max_m:.1f} m</div>
  </header>
  <div class="summary">
    {_metric("Near Miss Count", f"{summary['n']}")}
    {_metric("Capture Gap p50", f"{summary['capture_gap_p50_m']:.3f} m")}
    {_metric("Capture Gap p90", f"{summary['capture_gap_p90_m']:.3f} m")}
    {_metric("Closest Approach p50", f"{summary['t_min_p50_s']:.2f} s")}
    {_metric("Closing Rate p50", f"{summary['closing_rate_at_min_p50_mps']:.3f} m/s")}
    {_metric("Local Visibility Mean", f"{summary['visible_window_fraction_mean']:.2f}")}
    {_metric("Image Error Mean", f"{summary['image_error_window_mean']:.2f}")}
    {_metric("Thrust Fraction p50", f"{summary['thrust_fraction_at_min_p50']:.2f}")}
  </div>
  <main>
    <section class="framing">
        <h2>Terminal Problem Framing</h2>
        <p>This isolates trials that almost solved the task but did not cross the {CAPTURE_RADIUS_M:.1f} m capture radius. These are not broad acquisition failures; they are terminal geometry, timing, and underactuation failures.</p>
        <ul>
          <li>The median miss gap is small, so many failures are plausibly fixable with terminal behavior changes.</li>
          <li>The median radial closing rate at closest approach is near zero, indicating grazing or sliding past the target rather than decisive terminal closure.</li>
          <li>Local visibility near closest approach is low, so the controller often lacks image feedback exactly when it needs terminal correction.</li>
          <li>The elevation plots should be read through the quadcopter underactuation lens: useful acceleration is generated through body-z thrust, not arbitrary force toward the target.</li>
        </ul>
    </section>
    {panel_sections}
    <section class="framing">
      <h2>Combined Read</h2>
      <p>The high-hit elevation band from the previous report roughly corresponds to geometries where the vehicle can solve the interception without asking the underactuated thrust axis to do something impossible at the end. In the near-miss set, many closest-approach samples still have vertical separation and poor thrust-axis alignment. That means a policy change should not only push harder toward the target; it should shape the approach so the final target direction is compatible with available thrust and body-rate authority.</p>
      <p>The most actionable split is between <strong>terminal grazing</strong> and <strong>terminal loss of visual correction</strong>. Grazing cases have tiny capture gaps and near-zero radial closure at closest approach. Visibility-loss cases have weak local feature availability or high image error near closest approach. Those likely need different fixes: damping/lead for the first, target-retention and pitch scheduling for the second.</p>
    </section>
    <section>
      <h2>Generated Artifacts</h2>
      <ul>
        <li>Per-figure PNGs: <code>figures/*.png</code></li>
        <li>Per-seed terminal metrics CSV: <code>{html.escape(csv_path.name)}</code></li>
      </ul>
    </section>
  </main>
</body>
</html>
"""


def _html_panel(panel: dict[str, Any]) -> str:
    path = Path(panel["path"])
    src = html.escape(str(Path("figures") / path.name))
    title = html.escape(str(panel["title"]))
    interpretation = html.escape(str(panel["interpretation"]))
    return f"""    <section class="split">
      <div>
        <h2>{title}</h2>
        <p>{interpretation}</p>
      </div>
      <div>
        <img src="{src}" alt="{title}">
      </div>
    </section>"""


def _metric(label: str, value: str) -> str:
    return f"""<div class="metric"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>"""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_terminal_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _load_rows(path):
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if key == "seed":
                parsed[key] = int(value)
            else:
                parsed[key] = _float(value)
        rows.append(parsed)
    return rows


def _visible_fraction(samples) -> float:
    return sum(1 for sample in samples if sample.feature.detected) / max(len(samples), 1)


def _window_fraction(samples, idx: int, *, key: str, radius: int) -> float:
    window = samples[max(0, idx - radius): min(len(samples), idx + radius + 1)]
    if key != "visible":
        raise ValueError(key)
    return sum(1 for sample in window if sample.feature.detected) / max(len(window), 1)


def _window_mean(samples, idx: int, *, key: str, radius: int) -> float:
    window = samples[max(0, idx - radius): min(len(samples), idx + radius + 1)]
    values = []
    for sample in window:
        if key == "image_error":
            if sample.feature.detected and sample.feature.uv_norm is not None:
                values.append(float(np.linalg.norm(sample.feature.uv_norm)))
        else:
            raise ValueError(key)
    return float(np.mean(values)) if values else float("nan")


def _distance_rate_at(samples, idx: int) -> float:
    sample = samples[idx]
    rel = np.asarray(sample.target.position_w, dtype=float) - np.asarray(sample.vehicle.position_w, dtype=float)
    rel_vel = np.asarray(sample.target.velocity_w, dtype=float) - np.asarray(sample.vehicle.velocity_w, dtype=float)
    dist = float(np.linalg.norm(rel))
    return -float(np.dot(rel, rel_vel) / max(dist, 1e-12))


_SPEED_COLORS = {
    0.0: "#1f77b4",
    5.0: "#ff7f0e",
    10.0: "#2ca02c",
    20.0: "#d62728",
}


def _speed_colors(speed: np.ndarray) -> list[str]:
    return [_SPEED_COLORS.get(float(value), "#555555") for value in speed]


def _legend_for_speeds(ax) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=f"{int(speed)} m/s")
        for speed, color in _SPEED_COLORS.items()
    ]
    ax.legend(handles=handles, title="initial speed")


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float:
    text = str(value).strip()
    return float("nan") if text == "" else float(text)


def _pct(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if values.size else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
