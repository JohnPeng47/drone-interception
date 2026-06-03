#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
import argparse
from pathlib import Path
from typing import Any

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.generators.robust_intercept import (
    _camera_ray_from_fov_fraction,
    _camera_rotation_from_forward,
    _spherical_deg,
    _unit,
)
from scripts.generators.static_intercept import StaticInterceptConfigGenerator


OUT_DIR = Path(__file__).resolve().parent
RECORDS_JSON = REPO_ROOT / "scripts/generators/sim_instances/static_intercept_1048_sample_records.json"
SURFACE_RANGE_M = 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot static-intercept terminal speed over the sample surface.")
    parser.add_argument("--controller", default="beihang_minimal")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    controller = str(args.controller)
    run_dir = args.run_dir or OUT_DIR / controller
    trials_csv = run_dir / "trials.csv"
    joined_csv = OUT_DIR / f"{controller}_static_intercept_1048_interception_speeds.csv"
    summary_json = OUT_DIR / f"{controller}_static_intercept_1048_summary.json"
    plot_stem = f"{controller}_static_intercept_1048_terminal_relative_speed_surface"

    trials = _read_trials(trials_csv)
    samples = {int(row["seed"]): row for row in _read_json(RECORDS_JSON)}
    config = StaticInterceptConfigGenerator.default_config()

    rows = []
    for seed, trial in sorted(trials.items()):
        sample = samples[seed]
        coords = _target_relative_coordinates(config, sample, range_m=SURFACE_RANGE_M)
        terminal_relative_speed = float(trial["terminal_relative_speed_mps"])
        caught = trial["caught"] == "True"
        rows.append({
            "seed": seed,
            "caught": caught,
            "terminal_reason": "intercepted" if caught else "not_intercepted",
            "catch_time_s": _optional_float(trial["catch_time_s"]),
            "min_distance_m": float(trial["min_distance_m"]),
            "final_distance_m": float(trial["final_distance_m"]),
            "terminal_pursuer_speed_mps": float(trial["terminal_pursuer_speed_mps"]),
            "terminal_target_speed_mps": float(trial["terminal_target_speed_mps"]),
            "terminal_relative_speed_mps": terminal_relative_speed,
            "camera_elevation_deg": float(sample["camera_elevation_deg"]),
            "camera_u_fraction": float(sample["camera_u_fraction"]),
            "camera_v_fraction": float(sample["camera_v_fraction"]),
            "target_rel_r10_x_m": coords["x"],
            "target_rel_r10_y_m": coords["y"],
            "target_rel_r10_z_m": coords["z"],
            "heading_rel_r10_x_m": coords["heading_x"],
            "heading_rel_r10_y_m": coords["heading_y"],
            "heading_rel_r10_z_m": coords["heading_z"],
        })

    _write_csv(joined_csv, rows)
    summary = _summary(
        rows,
        controller=controller,
        trials_csv=trials_csv,
        joined_csv=joined_csv,
        html_path=OUT_DIR / f"{plot_stem}.html",
        png_path=OUT_DIR / f"{plot_stem}.png",
    )
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_static(rows, OUT_DIR / f"{plot_stem}.png", controller=controller)
    _plot_interactive(rows, OUT_DIR / f"{plot_stem}.html", summary, controller=controller)


def _read_trials(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["seed"]): row for row in csv.DictReader(handle)}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary(
    rows: list[dict[str, Any]],
    *,
    controller: str,
    trials_csv: Path,
    joined_csv: Path,
    html_path: Path,
    png_path: Path,
) -> dict[str, Any]:
    speeds = np.asarray([row["terminal_relative_speed_mps"] for row in rows], dtype=float)
    caught = np.asarray([row["caught"] for row in rows], dtype=bool)
    caught_speeds = speeds[caught]
    return {
        "rows": len(rows),
        "controller": controller,
        "caught": int(np.sum(caught)),
        "catch_fraction": float(np.mean(caught)) if rows else math.nan,
        "terminal_relative_speed_mps": {
            "min": float(np.min(speeds)),
            "p25": float(np.percentile(speeds, 25)),
            "p50": float(np.percentile(speeds, 50)),
            "p75": float(np.percentile(speeds, 75)),
            "max": float(np.max(speeds)),
            "caught_p50": float(np.percentile(caught_speeds, 50)) if caught_speeds.size else math.nan,
        },
        "surface_range_m": SURFACE_RANGE_M,
        "trials": str(trials_csv.relative_to(REPO_ROOT)),
        "joined_csv": str(joined_csv.relative_to(REPO_ROOT)),
        "html": str(html_path.relative_to(REPO_ROOT)),
        "png": str(png_path.relative_to(REPO_ROOT)),
    }


def _target_relative_coordinates(config: dict[str, Any], sample: dict[str, Any], *, range_m: float) -> dict[str, float]:
    values = dict(sample)
    values["camera_azimuth_deg"] = 0.0
    camera_cfg = config["camera"]
    target_dir_c = _camera_ray_from_fov_fraction(camera_cfg, values)
    camera_forward_w = _unit(_spherical_deg(0.0, float(values["camera_elevation_deg"])))
    rotation_wc = _camera_rotation_from_forward(camera_forward_w)
    target_rel = float(range_m) * _unit(rotation_wc @ target_dir_c)
    heading_rel = float(range_m) * camera_forward_w
    return {
        "x": float(target_rel[0]),
        "y": float(target_rel[1]),
        "z": float(target_rel[2]),
        "heading_x": float(heading_rel[0]),
        "heading_y": float(heading_rel[1]),
        "heading_z": float(heading_rel[2]),
    }


def _plot_static(rows: list[dict[str, Any]], path: Path, *, controller: str) -> None:
    speeds = np.asarray([row["terminal_relative_speed_mps"] for row in rows], dtype=float)
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    _draw_static_ribbon(ax, range_m=SURFACE_RANGE_M)
    scatter = ax.scatter(
        [row["target_rel_r10_x_m"] for row in rows],
        [row["target_rel_r10_y_m"] for row in rows],
        [row["target_rel_r10_z_m"] for row in rows],
        c=speeds,
        cmap="viridis",
        s=[30 if row["caught"] else 14 for row in rows],
        alpha=0.86,
        linewidths=0.2,
        edgecolors="#111827",
    )
    ax.scatter([0.0], [0.0], [0.0], marker="x", s=80, color="#111827", label="pursuer")
    ax.set_title(f"{controller} static intercept: terminal relative speed")
    ax.set_xlabel("target x relative to pursuer (m)")
    ax.set_ylabel("target y relative to pursuer (m)")
    ax.set_zlabel("target z relative to pursuer (m)")
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-10.5, 10.5)
    ax.set_zlim(-10.5, 10.5)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-45)
    fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.02, label="terminal relative speed (m/s)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_interactive(rows: list[dict[str, Any]], path: Path, summary: dict[str, Any], *, controller: str) -> None:
    import plotly.graph_objects as go

    speeds = [row["terminal_relative_speed_mps"] for row in rows]
    caught = [row for row in rows if row["caught"]]
    missed = [row for row in rows if not row["caught"]]
    cmin = float(min(speeds))
    cmax = float(max(speeds))

    fig = go.Figure()
    _add_interactive_ribbon(fig, range_m=SURFACE_RANGE_M)
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers",
            name="pursuer",
            marker={"symbol": "x", "size": 7, "color": "#111827"},
            hoverinfo="text",
            text=["pursuer origin"],
        )
    )
    fig.add_trace(_speed_trace(missed, "not intercepted", "circle", cmin, cmax, showscale=True))
    fig.add_trace(_speed_trace(caught, "intercepted", "diamond", cmin, cmax, showscale=False))
    fig.update_layout(
        title=(
            f"{controller} on static_intercept_1048: terminal relative speed "
            f"(catch fraction {summary['catch_fraction']:.3f})"
        ),
        scene={
            "xaxis_title": "target x relative to pursuer (m)",
            "yaxis_title": "target y relative to pursuer (m)",
            "zaxis_title": "target z relative to pursuer (m)",
            "xaxis": {"range": [-10.5, 10.5]},
            "yaxis": {"range": [-10.5, 10.5]},
            "zaxis": {"range": [-10.5, 10.5]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.6, "y": -1.8, "z": 1.1}},
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 48},
    )
    fig.write_html(path, include_plotlyjs="cdn")


def _speed_trace(
    rows: list[dict[str, Any]],
    name: str,
    symbol: str,
    cmin: float,
    cmax: float,
    *,
    showscale: bool,
) -> Any:
    import plotly.graph_objects as go

    return go.Scatter3d(
        x=[row["target_rel_r10_x_m"] for row in rows],
        y=[row["target_rel_r10_y_m"] for row in rows],
        z=[row["target_rel_r10_z_m"] for row in rows],
        mode="markers",
        name=name,
        marker={
            "symbol": symbol,
            "size": 5.5 if symbol == "diamond" else 3.8,
            "color": [row["terminal_relative_speed_mps"] for row in rows],
            "colorscale": "Viridis",
            "cmin": cmin,
            "cmax": cmax,
            "opacity": 0.92 if symbol == "diamond" else 0.62,
            "colorbar": {"title": "terminal relative speed (m/s)"},
            "showscale": showscale,
            "line": {"width": 0.4, "color": "#111827"},
        },
        text=[_hover_text(row) for row in rows],
        hoverinfo="text",
    )


def _hover_text(row: dict[str, Any]) -> str:
    catch_time = "" if row["catch_time_s"] is None else f"<br>catch time: {row['catch_time_s']:.3f} s"
    return (
        f"seed: {row['seed']}"
        f"<br>caught: {row['caught']}"
        f"{catch_time}"
        f"<br>terminal relative speed: {row['terminal_relative_speed_mps']:.3f} m/s"
        f"<br>min distance: {row['min_distance_m']:.3f} m"
        f"<br>final distance: {row['final_distance_m']:.3f} m"
        f"<br>camera elev: {row['camera_elevation_deg']:.2f} deg"
        f"<br>camera u/v: {row['camera_u_fraction']:.3f}, {row['camera_v_fraction']:.3f}"
    )


def _draw_static_ribbon(ax: Any, *, range_m: float) -> None:
    for line in _ribbon_lines(range_m=range_m):
        ax.plot(line["x"], line["y"], line["z"], color=line["color"], alpha=line["alpha"], linewidth=line["width"])


def _add_interactive_ribbon(fig: Any, *, range_m: float) -> None:
    import plotly.graph_objects as go

    for line in _ribbon_lines(range_m=range_m):
        fig.add_trace(
            go.Scatter3d(
                x=line["x"],
                y=line["y"],
                z=line["z"],
                mode="lines",
                name=line["name"],
                line={"color": line["color"], "width": line["width"]},
                opacity=line["alpha"],
                hoverinfo="skip",
                showlegend=line["showlegend"],
            )
        )


def _ribbon_lines(*, range_m: float) -> list[dict[str, Any]]:
    config = StaticInterceptConfigGenerator.default_config()
    heading_elevations = np.linspace(-90.0, 90.0, 91)
    fov_elevations = np.linspace(-75.0, 75.0, 76)
    heading = np.array([_surface_point(config, elev, 0.0, 0.0, range_m) for elev in heading_elevations])
    lines = [
        {
            "name": "heading arc",
            "x": heading[:, 0],
            "y": heading[:, 1],
            "z": heading[:, 2],
            "color": "#111827",
            "alpha": 0.9,
            "width": 3,
            "showlegend": True,
        }
    ]
    for u, name, color in [(-0.9, "left FOV edge", "#64748b"), (0.9, "right FOV edge", "#64748b")]:
        edge = np.array([_surface_point(config, elev, u, 0.0, range_m) for elev in fov_elevations])
        lines.append({
            "name": name,
            "x": edge[:, 0],
            "y": edge[:, 1],
            "z": edge[:, 2],
            "color": color,
            "alpha": 0.55,
            "width": 2,
            "showlegend": True,
        })
    for v, name, color in [(-0.9, "lower FOV edge", "#94a3b8"), (0.9, "upper FOV edge", "#94a3b8")]:
        edge = np.array([_surface_point(config, elev, 0.0, v, range_m) for elev in fov_elevations])
        lines.append({
            "name": name,
            "x": edge[:, 0],
            "y": edge[:, 1],
            "z": edge[:, 2],
            "color": color,
            "alpha": 0.45,
            "width": 2,
            "showlegend": True,
        })
    return lines


def _surface_point(config: dict[str, Any], camera_elevation_deg: float, u: float, v: float, range_m: float) -> np.ndarray:
    sample = {
        "camera_elevation_deg": float(camera_elevation_deg),
        "camera_u_fraction": float(u),
        "camera_v_fraction": float(v),
    }
    coords = _target_relative_coordinates(config, sample, range_m=range_m)
    return np.array([coords["x"], coords["y"], coords["z"]], dtype=float)


if __name__ == "__main__":
    main()
