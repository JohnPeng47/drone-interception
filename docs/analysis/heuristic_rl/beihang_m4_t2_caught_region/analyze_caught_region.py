#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
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
    RobustInterceptConfigGenerator,
    _camera_ray_from_fov_fraction,
    _camera_rotation_from_forward,
    _spherical_deg,
    _unit,
    generate_sample_records,
)

RUN_ROOT = (
    REPO_ROOT
    / ".agents/heuristic-rl-group/beihang-m4-t2-20260601/controller-001-beihang_minimal_sim"
)
OUT_DIR = Path(__file__).resolve().parent

JOINED_CSV = OUT_DIR / "joined_trials.csv"
CAUGHT_SEED_CSV = OUT_DIR / "caught_seed_summary.csv"
SUMMARY_JSON = OUT_DIR / "region_summary.json"
SUMMARY_MD = OUT_DIR / "region_summary.md"

FEATURES = (
    "camera_azimuth_deg",
    "camera_elevation_deg",
    "camera_u_fraction",
    "camera_v_fraction",
    "los_elevation_deg",
    "los_closing_speed_mps",
    "los_lateral_speed_mps",
    "trajectory_angle_deg",
)

TARGET_RELATIVE_FEATURES = (
    "target_rel_x_m",
    "target_rel_y_m",
    "target_rel_z_m",
    "target_rel_heading_x_m",
    "target_rel_heading_y_m",
    "target_rel_heading_z_m",
    "target_rel_r8_x_m",
    "target_rel_r8_y_m",
    "target_rel_r8_z_m",
    "target_rel_r8_heading_x_m",
    "target_rel_r8_heading_y_m",
    "target_rel_r8_heading_z_m",
)

CAMERA_ELEVATION_GROUPS = (
    (-90.0, -60.0, "#1f77b4"),
    (-60.0, -45.0, "#17becf"),
    (-45.0, -30.0, "#2ca02c"),
    (-30.0, -15.0, "#bcbd22"),
    (-15.0, 0.0, "#ff7f0e"),
    (0.0, 15.0, "#d62728"),
    (15.0, 30.0, "#e377c2"),
    (30.0, 45.0, "#9467bd"),
    (45.0, 60.0, "#8c564b"),
    (60.0, 90.0, "#111827"),
)


def main() -> None:
    sample_cache: dict[Path, dict[int, dict[str, Any]]] = {}
    joined = []
    run_summaries = []

    for trial_path in sorted(RUN_ROOT.glob("*/results/trials.csv")):
        run_name = trial_path.parents[1].name
        summary = _read_json(trial_path.parent / "summary.json")
        source = REPO_ROOT / str(summary["source"])
        sample_records = sample_cache.setdefault(source, _sample_records_for_source(source))
        rows = list(csv.DictReader(trial_path.open(newline="", encoding="utf-8")))

        caught_count = 0
        for row in rows:
            seed = int(row["seed"])
            sample = sample_records[seed]
            caught = row.get("caught") == "True"
            caught_count += int(caught)
            joined.append(
                {
                    "run": run_name,
                    "source": str(source.relative_to(REPO_ROOT)),
                    "seed": seed,
                    "caught": caught,
                    "catch_time_s": _optional_float(row.get("catch_time_s")),
                    "min_distance_m": _optional_float(row.get("min_distance_m")),
                    "final_distance_m": _optional_float(row.get("final_distance_m")),
                    "visible_fraction": _optional_float(row.get("visible_fraction")),
                    "range_m": sample["range_m"],
                    "forward_speed_mps": sample["forward_speed_mps"],
                    "stratum": sample["stratum"],
                    **{feature: sample[feature] for feature in FEATURES},
                    **{feature: sample[feature] for feature in TARGET_RELATIVE_FEATURES},
                }
            )

        run_summaries.append(
            {
                "run": run_name,
                "source": str(source.relative_to(REPO_ROOT)),
                "rows": len(rows),
                "caught": caught_count,
                "catch_fraction": caught_count / len(rows) if rows else 0.0,
            }
        )

    _write_csv(JOINED_CSV, joined)
    seed_rows = _caught_seed_summary(joined)
    _write_csv(CAUGHT_SEED_CSV, seed_rows)
    sample_rows_by_source = _sample_rows_by_source(joined, sample_cache)

    final_512 = [row for row in joined if row["run"] == "final-validation-512"]
    best_128 = [row for row in joined if row["run"] == "milestone-011-stronger-aligned-thrust"]
    region_summary = {
        "run_summaries": run_summaries,
        "source_caught_union": _source_caught_union(joined),
        "final_validation_512": _region_stats(final_512),
        "best_128_milestone_011": _region_stats(best_128),
    }
    SUMMARY_JSON.write_text(json.dumps(region_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _plot_projection(final_512, "final_validation_512")
    _plot_projection(best_128, "milestone_011_best_128")
    for source, sample_rows in sample_rows_by_source.items():
        stem = Path(source).stem
        _plot_3d_static(sample_rows, f"{stem}_caught_union_3d")
        _plot_3d_interactive(sample_rows, f"{stem}_caught_union_3d")
        _plot_target_relative_static(sample_rows, f"{stem}_target_relative_r8_3d")
        _plot_target_relative_interactive(sample_rows, f"{stem}_target_relative_r8_3d")
        _plot_target_relative_elevation_groups_static(sample_rows, f"{stem}_target_relative_r8_3d")
        _plot_target_relative_elevation_groups_interactive(sample_rows, f"{stem}_target_relative_r8_3d")
        _plot_target_relative_elevation_groups_static(sample_rows, f"{stem}_target_relative_r8_camera_elevation_groups_3d")
        _plot_target_relative_elevation_groups_interactive(sample_rows, f"{stem}_target_relative_r8_camera_elevation_groups_3d")
    _plot_3d_static(final_512, "final_validation_512_caught_3d")
    _plot_3d_interactive(final_512, "final_validation_512_caught_3d")
    _plot_target_relative_static(final_512, "final_validation_512_target_relative_r8_3d")
    _plot_target_relative_interactive(final_512, "final_validation_512_target_relative_r8_3d")
    _plot_target_relative_elevation_groups_static(final_512, "final_validation_512_target_relative_r8_3d")
    _plot_target_relative_elevation_groups_interactive(final_512, "final_validation_512_target_relative_r8_3d")
    _plot_target_relative_elevation_groups_static(final_512, "final_validation_512_target_relative_r8_camera_elevation_groups_3d")
    _plot_target_relative_elevation_groups_interactive(final_512, "final_validation_512_target_relative_r8_camera_elevation_groups_3d")
    _write_markdown(region_summary)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_records_for_source(source: Path) -> dict[int, dict[str, Any]]:
    metadata = _read_json(source.with_suffix(".json"))
    config = RobustInterceptConfigGenerator.default_config()
    config["sampling"]["n_samples"] = int(metadata["sampling"]["requested_samples"])
    config["sampling"]["strategy"] = metadata["generator"]["strategy"]
    config["sampling"]["seed"] = int(metadata["sampling"]["seed"])
    config["sampling"]["scramble"] = bool(metadata["sampling"]["scramble"])
    config["sampling"]["active_parameters"] = list(metadata["sampling"]["active_parameters"])
    config["parameters"] = metadata["parameters"]
    records = {}
    for row in generate_sample_records(config):
        enriched = dict(row)
        enriched.update(_target_relative_coordinates(config, row, range_m=float(row["range_m"]), prefix="target_rel"))
        enriched.update(_target_relative_coordinates(config, row, range_m=8.0, prefix="target_rel_r8"))
        records[int(row["seed"])] = enriched
    return records


def _target_relative_coordinates(
    config: dict[str, Any],
    sample: dict[str, Any],
    *,
    range_m: float,
    prefix: str,
) -> dict[str, float]:
    values = dict(sample)
    values["camera_azimuth_deg"] = 0.0
    camera_cfg = config["camera"]
    target_dir_c = _camera_ray_from_fov_fraction(camera_cfg, values)
    camera_forward_w = _unit(_spherical_deg(0.0, float(values["camera_elevation_deg"])))
    rotation_wc = _camera_rotation_from_forward(camera_forward_w)
    target_rel = float(range_m) * _unit(rotation_wc @ target_dir_c)
    heading_rel = float(range_m) * camera_forward_w
    return {
        f"{prefix}_x_m": float(target_rel[0]),
        f"{prefix}_y_m": float(target_rel[1]),
        f"{prefix}_z_m": float(target_rel[2]),
        f"{prefix}_heading_x_m": float(heading_rel[0]),
        f"{prefix}_heading_y_m": float(heading_rel[1]),
        f"{prefix}_heading_z_m": float(heading_rel[2]),
    }


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _caught_seed_summary(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        groups[(row["source"], int(row["seed"]))].append(row)

    summary_rows = []
    for (source, seed), rows in sorted(groups.items()):
        caught_rows = [row for row in rows if row["caught"]]
        if not caught_rows:
            continue
        sample = rows[0]
        summary_rows.append(
            {
                "source": source,
                "seed": seed,
                "caught_runs": len(caught_rows),
                "evaluated_runs": len(rows),
                "caught_run_names": ";".join(row["run"] for row in caught_rows),
                "range_m": sample["range_m"],
                "forward_speed_mps": sample["forward_speed_mps"],
                "camera_elevation_deg": sample["camera_elevation_deg"],
                "camera_u_fraction": sample["camera_u_fraction"],
                "camera_v_fraction": sample["camera_v_fraction"],
                "los_elevation_deg": sample["los_elevation_deg"],
                "trajectory_angle_deg": sample["trajectory_angle_deg"],
            }
        )
    return summary_rows


def _source_caught_union(joined: list[dict[str, Any]]) -> dict[str, Any]:
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        source_rows[row["source"]].append(row)

    result = {}
    for source, rows in sorted(source_rows.items()):
        caught_seeds = sorted({int(row["seed"]) for row in rows if row["caught"]})
        result[source] = {
            "unique_caught_count": len(caught_seeds),
            "unique_caught_seeds": caught_seeds,
        }
    return result


def _sample_rows_by_source(
    joined: list[dict[str, Any]],
    sample_cache: dict[Path, dict[int, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    caught_runs_by_source_seed: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in joined:
        if row["caught"]:
            caught_runs_by_source_seed[(row["source"], int(row["seed"]))].append(row["run"])

    result = {}
    for source_path, records_by_seed in sample_cache.items():
        source = str(source_path.relative_to(REPO_ROOT))
        rows = []
        for seed, sample in sorted(records_by_seed.items()):
            caught_runs = caught_runs_by_source_seed[(source, seed)]
            rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "caught": bool(caught_runs),
                    "caught_runs": len(caught_runs),
                    "caught_run_names": ";".join(caught_runs),
                    "range_m": sample["range_m"],
                    "forward_speed_mps": sample["forward_speed_mps"],
                    "stratum": sample["stratum"],
                    **{feature: sample[feature] for feature in FEATURES},
                    **{feature: sample[feature] for feature in TARGET_RELATIVE_FEATURES},
                }
            )
        result[source] = rows
    return result


def _region_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    caught = [row for row in rows if row["caught"]]
    return {
        "rows": len(rows),
        "caught": len(caught),
        "catch_fraction": len(caught) / len(rows) if rows else 0.0,
        "caught_seeds": [int(row["seed"]) for row in caught],
        "grid_rates": _grid_rates(rows),
        "feature_ranges": _feature_ranges(caught),
        "binned_rates": {
            "camera_elevation_deg": _binned_rates(rows, "camera_elevation_deg", [-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90]),
            "los_elevation_deg": _binned_rates(rows, "los_elevation_deg", [-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90]),
            "trajectory_angle_deg": _binned_rates(rows, "trajectory_angle_deg", [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60]),
            "camera_u_fraction": _binned_rates(rows, "camera_u_fraction", [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]),
            "camera_v_fraction": _binned_rates(rows, "camera_v_fraction", [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]),
        },
        "candidate_regions": _candidate_regions(rows),
    }


def _grid_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((row["range_m"], row["forward_speed_mps"]) for row in rows)
    caught = Counter((row["range_m"], row["forward_speed_mps"]) for row in rows if row["caught"])
    return [
        {
            "range_m": range_m,
            "forward_speed_mps": speed,
            "caught": caught[(range_m, speed)],
            "total": counts[(range_m, speed)],
            "catch_fraction": caught[(range_m, speed)] / counts[(range_m, speed)],
        }
        for range_m, speed in sorted(counts)
    ]


def _feature_ranges(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for feature in FEATURES:
        values = [float(row[feature]) for row in rows]
        if not values:
            continue
        result[feature] = {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }
    return result


def _binned_rates(rows: list[dict[str, Any]], feature: str, edges: list[float]) -> list[dict[str, Any]]:
    result = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bin_rows = [row for row in rows if lo <= float(row[feature]) < hi]
        caught = sum(1 for row in bin_rows if row["caught"])
        result.append(
            {
                "lo": lo,
                "hi": hi,
                "caught": caught,
                "total": len(bin_rows),
                "catch_fraction": caught / len(bin_rows) if bin_rows else 0.0,
            }
        )
    return result


def _candidate_regions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _rule_stats(
            rows,
            "range<=8, camera_elevation[-45,15], u[-0.3,0.3]",
            lambda row: row["range_m"] <= 8.0
            and -45.0 <= row["camera_elevation_deg"] <= 15.0
            and -0.3 <= row["camera_u_fraction"] <= 0.3,
        ),
        _rule_stats(
            rows,
            "camera_elevation[-45,15], u[-0.3,0.3]",
            lambda row: -45.0 <= row["camera_elevation_deg"] <= 15.0
            and -0.3 <= row["camera_u_fraction"] <= 0.3,
        ),
        _rule_stats(
            rows,
            "camera_elevation[-45,15]",
            lambda row: -45.0 <= row["camera_elevation_deg"] <= 15.0,
        ),
        _rule_stats(
            rows,
            "LOS_elevation[-45,20], trajectory_angle<=30",
            lambda row: -45.0 <= row["los_elevation_deg"] <= 20.0
            and row["trajectory_angle_deg"] <= 30.0,
        ),
        _rule_stats(
            rows,
            "range=20, camera_elevation[-55,-5], LOS_elevation[-50,-5]",
            lambda row: row["range_m"] == 20.0
            and -55.0 <= row["camera_elevation_deg"] <= -5.0
            and -50.0 <= row["los_elevation_deg"] <= -5.0,
        ),
    ]


def _rule_stats(rows: list[dict[str, Any]], name: str, predicate: Any) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    selected_caught = sum(1 for row in selected if row["caught"])
    total_caught = sum(1 for row in rows if row["caught"])
    return {
        "name": name,
        "selected": len(selected),
        "caught": selected_caught,
        "precision": selected_caught / len(selected) if selected else 0.0,
        "recall": selected_caught / total_caught if total_caught else 0.0,
    }


def _plot_projection(rows: list[dict[str, Any]], stem: str) -> None:
    if not rows:
        return

    caught = [row for row in rows if row["caught"]]
    missed = [row for row in rows if not row["caught"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(stem.replace("_", " "))

    _scatter_panel(axes[0], missed, caught, "camera_elevation_deg", "camera_u_fraction", "camera elevation deg", "camera u fraction")
    _scatter_panel(axes[1], missed, caught, "camera_elevation_deg", "camera_v_fraction", "camera elevation deg", "camera v fraction")
    _scatter_panel(axes[2], missed, caught, "los_elevation_deg", "trajectory_angle_deg", "LOS elevation deg", "trajectory angle deg")
    axes[2].legend(loc="best")

    fig.savefig(OUT_DIR / f"{stem}_projections.png", dpi=160)
    plt.close(fig)


def _scatter_panel(
    ax: Any,
    missed: list[dict[str, Any]],
    caught: list[dict[str, Any]],
    x_name: str,
    y_name: str,
    x_label: str,
    y_label: str,
) -> None:
    ax.scatter([row[x_name] for row in missed], [row[y_name] for row in missed], s=14, alpha=0.25, label="missed")
    ax.scatter([row[x_name] for row in caught], [row[y_name] for row in caught], s=34, alpha=0.9, label="caught")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)


def _plot_3d_static(rows: list[dict[str, Any]], stem: str) -> None:
    if not rows:
        return

    caught = [row for row in rows if row["caught"]]
    missed = [row for row in rows if not row["caught"]]
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        [row["camera_elevation_deg"] for row in missed],
        [row["camera_u_fraction"] for row in missed],
        [row["camera_v_fraction"] for row in missed],
        s=14,
        alpha=0.18,
        color="#6b7280",
        label="not caught",
        depthshade=False,
    )
    ax.scatter(
        [row["camera_elevation_deg"] for row in caught],
        [row["camera_u_fraction"] for row in caught],
        [row["camera_v_fraction"] for row in caught],
        s=[34 + 8 * int(row.get("caught_runs") or 1) for row in caught],
        alpha=0.95,
        color="#f97316",
        edgecolors="#7c2d12",
        linewidths=0.4,
        label="caught",
        depthshade=False,
    )
    ax.set_title(stem.replace("_", " "))
    ax.set_xlabel("camera elevation deg")
    ax.set_ylabel("camera u fraction")
    ax.set_zlabel("camera v fraction")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-0.9, 0.9)
    ax.set_zlim(-0.9, 0.9)
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=180)
    plt.close(fig)


def _plot_3d_interactive(rows: list[dict[str, Any]], stem: str) -> None:
    if not rows:
        return

    import plotly.graph_objects as go

    caught = [row for row in rows if row["caught"]]
    missed = [row for row in rows if not row["caught"]]
    fig = go.Figure()
    fig.add_trace(_scatter3d_trace(missed, "not caught", "rgba(107,114,128,0.28)", 3.5))
    fig.add_trace(_scatter3d_trace(caught, "caught", "#f97316", 6.5))
    fig.update_layout(
        title=stem.replace("_", " "),
        scene={
            "xaxis_title": "camera_elevation_deg",
            "yaxis_title": "camera_u_fraction",
            "zaxis_title": "camera_v_fraction",
            "xaxis": {"range": [-90, 90]},
            "yaxis": {"range": [-0.9, 0.9]},
            "zaxis": {"range": [-0.9, 0.9]},
            "camera": {"eye": {"x": 1.6, "y": -1.8, "z": 1.2}},
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 42},
    )
    fig.write_html(OUT_DIR / f"{stem}.html", include_plotlyjs="cdn")


def _plot_target_relative_static(rows: list[dict[str, Any]], stem: str) -> None:
    if not rows:
        return

    caught = [row for row in rows if row["caught"]]
    missed = [row for row in rows if not row["caught"]]
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    _draw_static_ribbon(ax, range_m=8.0)
    ax.scatter([0.0], [0.0], [0.0], marker="x", s=80, color="#111827", label="pursuer")
    ax.scatter(
        [row["target_rel_r8_x_m"] for row in missed],
        [row["target_rel_r8_y_m"] for row in missed],
        [row["target_rel_r8_z_m"] for row in missed],
        s=14,
        alpha=0.18,
        color="#6b7280",
        label="not caught",
        depthshade=False,
    )
    ax.scatter(
        [row["target_rel_r8_x_m"] for row in caught],
        [row["target_rel_r8_y_m"] for row in caught],
        [row["target_rel_r8_z_m"] for row in caught],
        s=[34 + 8 * int(row.get("caught_runs") or 1) for row in caught],
        alpha=0.95,
        color="#f97316",
        edgecolors="#7c2d12",
        linewidths=0.4,
        label="caught",
        depthshade=False,
    )
    ax.set_title(stem.replace("_", " "))
    ax.set_xlabel("target x relative to pursuer (m)")
    ax.set_ylabel("target y relative to pursuer (m)")
    ax.set_zlabel("target z relative to pursuer (m)")
    ax.set_xlim(-8.5, 8.5)
    ax.set_ylim(-8.5, 8.5)
    ax.set_zlim(-8.5, 8.5)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-45)
    ax.legend(loc="upper left")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=180)
    plt.close(fig)


def _plot_target_relative_interactive(rows: list[dict[str, Any]], stem: str) -> None:
    if not rows:
        return

    import plotly.graph_objects as go

    caught = [row for row in rows if row["caught"]]
    missed = [row for row in rows if not row["caught"]]
    fig = go.Figure()
    _add_interactive_ribbon(fig, range_m=8.0)
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
    fig.add_trace(_target_relative_trace(missed, "not caught", "rgba(107,114,128,0.28)", 3.5))
    fig.add_trace(_target_relative_trace(caught, "caught", "#f97316", 6.5))
    fig.update_layout(
        title=stem.replace("_", " "),
        scene={
            "xaxis_title": "target x relative to pursuer (m)",
            "yaxis_title": "target y relative to pursuer (m)",
            "zaxis_title": "target z relative to pursuer (m)",
            "xaxis": {"range": [-8.5, 8.5]},
            "yaxis": {"range": [-8.5, 8.5]},
            "zaxis": {"range": [-8.5, 8.5]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.6, "y": -1.8, "z": 1.1}},
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 42},
    )
    fig.write_html(OUT_DIR / f"{stem}.html", include_plotlyjs="cdn")


def _plot_target_relative_elevation_groups_static(rows: list[dict[str, Any]], stem: str) -> None:
    if not rows:
        return

    caught = [row for row in rows if row["caught"]]
    missed = [row for row in rows if not row["caught"]]
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    _draw_static_ribbon(ax, range_m=8.0)
    ax.scatter([0.0], [0.0], [0.0], marker="x", s=80, color="#111827", label="pursuer")
    ax.scatter(
        [row["target_rel_r8_x_m"] for row in missed],
        [row["target_rel_r8_y_m"] for row in missed],
        [row["target_rel_r8_z_m"] for row in missed],
        s=12,
        alpha=0.12,
        color="#6b7280",
        label="not caught",
        depthshade=False,
    )

    for label, color, group_rows in _caught_rows_by_camera_elevation_group(caught):
        for row in group_rows:
            ax.plot(
                [0.0, row["target_rel_r8_x_m"]],
                [0.0, row["target_rel_r8_y_m"]],
                [0.0, row["target_rel_r8_z_m"]],
                color=color,
                alpha=0.34,
                linewidth=1.2,
            )
        ax.scatter(
            [row["target_rel_r8_x_m"] for row in group_rows],
            [row["target_rel_r8_y_m"] for row in group_rows],
            [row["target_rel_r8_z_m"] for row in group_rows],
            s=[36 + 8 * int(row.get("caught_runs") or 1) for row in group_rows],
            alpha=0.98,
            color=color,
            edgecolors="#111827",
            linewidths=0.35,
            label=f"{label} ({len(group_rows)})",
            depthshade=False,
        )

    ax.set_title(stem.replace("_", " "))
    ax.set_xlabel("target x relative to pursuer (m)")
    ax.set_ylabel("target y relative to pursuer (m)")
    ax.set_zlabel("target z relative to pursuer (m)")
    ax.set_xlim(-8.5, 8.5)
    ax.set_ylim(-8.5, 8.5)
    ax.set_zlim(-8.5, 8.5)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-45)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=180)
    plt.close(fig)


def _plot_target_relative_elevation_groups_interactive(rows: list[dict[str, Any]], stem: str) -> None:
    if not rows:
        return

    import plotly.graph_objects as go

    caught = [row for row in rows if row["caught"]]
    missed = [row for row in rows if not row["caught"]]
    fig = go.Figure()
    _add_interactive_ribbon(fig, range_m=8.0)
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
    fig.add_trace(_target_relative_trace(missed, "not caught", "rgba(107,114,128,0.20)", 3.0))

    for label, color, group_rows in _caught_rows_by_camera_elevation_group(caught):
        fig.add_trace(_target_relative_ray_trace(group_rows, f"{label} rays", color))
        fig.add_trace(_target_relative_trace(group_rows, f"{label} caught", color, 6.5))

    fig.update_layout(
        title=stem.replace("_", " "),
        scene={
            "xaxis_title": "target x relative to pursuer (m)",
            "yaxis_title": "target y relative to pursuer (m)",
            "zaxis_title": "target z relative to pursuer (m)",
            "xaxis": {"range": [-8.5, 8.5]},
            "yaxis": {"range": [-8.5, 8.5]},
            "zaxis": {"range": [-8.5, 8.5]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.6, "y": -1.8, "z": 1.1}},
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 42},
    )
    fig.write_html(OUT_DIR / f"{stem}.html", include_plotlyjs="cdn")


def _caught_rows_by_camera_elevation_group(
    caught: list[dict[str, Any]],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    grouped: list[tuple[str, str, list[dict[str, Any]]]] = []
    for lo, hi, color in CAMERA_ELEVATION_GROUPS:
        group_rows = [
            row
            for row in caught
            if lo <= float(row["camera_elevation_deg"]) < hi
            or (hi == 90.0 and float(row["camera_elevation_deg"]) <= hi and float(row["camera_elevation_deg"]) >= lo)
        ]
        if group_rows:
            grouped.append((_camera_elevation_group_label(lo, hi), color, group_rows))
    return grouped


def _camera_elevation_group_label(lo: float, hi: float) -> str:
    close = "]" if hi == 90.0 else ")"
    return f"camera elevation [{lo:g}, {hi:g}{close}"


def _target_relative_ray_trace(rows: list[dict[str, Any]], name: str, color: str) -> Any:
    import plotly.graph_objects as go

    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    text: list[str] = []
    for row in rows:
        x.extend([0.0, row["target_rel_r8_x_m"], math.nan])
        y.extend([0.0, row["target_rel_r8_y_m"], math.nan])
        z.extend([0.0, row["target_rel_r8_z_m"], math.nan])
        hover = _target_relative_hover_text(row)
        text.extend([hover, hover, ""])
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=name,
        line={"color": color, "width": 3},
        opacity=0.34,
        text=text,
        hoverinfo="text",
        showlegend=False,
    )


def _target_relative_trace(rows: list[dict[str, Any]], name: str, color: str, size: float) -> Any:
    import plotly.graph_objects as go

    return go.Scatter3d(
        x=[row["target_rel_r8_x_m"] for row in rows],
        y=[row["target_rel_r8_y_m"] for row in rows],
        z=[row["target_rel_r8_z_m"] for row in rows],
        mode="markers",
        name=name,
        marker={
            "size": [size + 0.9 * int(row.get("caught_runs") or 0) for row in rows],
            "color": color,
            "line": {"width": 0.5, "color": "#7c2d12" if name == "caught" else "rgba(107,114,128,0.18)"},
        },
        text=[_target_relative_hover_text(row) for row in rows],
        hoverinfo="text",
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
    config = RobustInterceptConfigGenerator.default_config()
    lines = []
    heading_elevations = np.linspace(-90.0, 90.0, 91)
    fov_elevations = np.linspace(-75.0, 75.0, 76)
    heading = np.array([_heading_point(config, elev, range_m) for elev in heading_elevations])
    lines.append(
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
    )

    for u in (-0.9, 0.9):
        for v in (-0.9, 0.9):
            points = np.array([_target_point(config, elev, u, v, range_m) for elev in fov_elevations])
            lines.append(
                {
                    "name": "FOV corner sweep",
                    "x": points[:, 0],
                    "y": points[:, 1],
                    "z": points[:, 2],
                    "color": "#2563eb",
                    "alpha": 0.28,
                    "width": 2,
                    "showlegend": False,
                }
            )

    for elev in (-75.0, -45.0, -15.0, 15.0, 45.0, 75.0):
        rectangle = _fov_rectangle(config, elev, range_m)
        lines.append(
            {
                "name": "FOV slice",
                "x": rectangle[:, 0],
                "y": rectangle[:, 1],
                "z": rectangle[:, 2],
                "color": "#38bdf8",
                "alpha": 0.32,
                "width": 2,
                "showlegend": False,
            }
        )
    return lines


def _heading_point(config: dict[str, Any], elevation_deg: float, range_m: float) -> np.ndarray:
    sample = {
        "range_m": range_m,
        "camera_elevation_deg": elevation_deg,
        "camera_u_fraction": 0.0,
        "camera_v_fraction": 0.0,
    }
    coords = _target_relative_coordinates(config, sample, range_m=range_m, prefix="point")
    return np.array([
        coords["point_heading_x_m"],
        coords["point_heading_y_m"],
        coords["point_heading_z_m"],
    ])


def _target_point(config: dict[str, Any], elevation_deg: float, u: float, v: float, range_m: float) -> np.ndarray:
    sample = {
        "range_m": range_m,
        "camera_elevation_deg": elevation_deg,
        "camera_u_fraction": u,
        "camera_v_fraction": v,
    }
    coords = _target_relative_coordinates(config, sample, range_m=range_m, prefix="point")
    return np.array([
        coords["point_x_m"],
        coords["point_y_m"],
        coords["point_z_m"],
    ])


def _fov_rectangle(config: dict[str, Any], elevation_deg: float, range_m: float) -> np.ndarray:
    points = []
    for u in np.linspace(-0.9, 0.9, 19):
        points.append(_target_point(config, elevation_deg, float(u), -0.9, range_m))
    for v in np.linspace(-0.9, 0.9, 19):
        points.append(_target_point(config, elevation_deg, 0.9, float(v), range_m))
    for u in np.linspace(0.9, -0.9, 19):
        points.append(_target_point(config, elevation_deg, float(u), 0.9, range_m))
    for v in np.linspace(0.9, -0.9, 19):
        points.append(_target_point(config, elevation_deg, -0.9, float(v), range_m))
    points.append(points[0])
    return np.asarray(points)


def _scatter3d_trace(rows: list[dict[str, Any]], name: str, color: str, size: float) -> Any:
    import plotly.graph_objects as go

    return go.Scatter3d(
        x=[row["camera_elevation_deg"] for row in rows],
        y=[row["camera_u_fraction"] for row in rows],
        z=[row["camera_v_fraction"] for row in rows],
        mode="markers",
        name=name,
        marker={
            "size": [size + 0.9 * int(row.get("caught_runs") or 0) for row in rows],
            "color": color,
            "line": {"width": 0.5, "color": "#7c2d12" if name == "caught" else "rgba(107,114,128,0.18)"},
        },
        text=[_hover_text(row) for row in rows],
        hoverinfo="text",
    )


def _hover_text(row: dict[str, Any]) -> str:
    caught_runs = row.get("caught_runs")
    caught_run_names = row.get("caught_run_names")
    lines = [
        f"seed: {int(row['seed'])}",
        f"caught: {bool(row['caught'])}",
        f"range_m: {float(row['range_m']):g}",
        f"forward_speed_mps: {float(row['forward_speed_mps']):g}",
        f"camera_elevation_deg: {float(row['camera_elevation_deg']):.2f}",
        f"camera_u_fraction: {float(row['camera_u_fraction']):.3f}",
        f"camera_v_fraction: {float(row['camera_v_fraction']):.3f}",
        f"LOS_elevation_deg: {float(row['los_elevation_deg']):.2f}",
        f"trajectory_angle_deg: {float(row['trajectory_angle_deg']):.2f}",
    ]
    if caught_runs is not None:
        lines.append(f"caught_runs: {int(caught_runs)}")
    if caught_run_names:
        lines.append(f"caught_run_names: {caught_run_names}")
    return "<br>".join(lines)


def _target_relative_hover_text(row: dict[str, Any]) -> str:
    lines = [
        f"seed: {int(row['seed'])}",
        f"caught: {bool(row['caught'])}",
        f"actual_range_m: {float(row['range_m']):g}",
        f"forward_speed_mps: {float(row['forward_speed_mps']):g}",
        f"camera_elevation_deg: {float(row['camera_elevation_deg']):.2f}",
        f"camera_u_fraction: {float(row['camera_u_fraction']):.3f}",
        f"camera_v_fraction: {float(row['camera_v_fraction']):.3f}",
        f"target_r8_x_m: {float(row['target_rel_r8_x_m']):.2f}",
        f"target_r8_y_m: {float(row['target_rel_r8_y_m']):.2f}",
        f"target_r8_z_m: {float(row['target_rel_r8_z_m']):.2f}",
    ]
    caught_runs = row.get("caught_runs")
    caught_run_names = row.get("caught_run_names")
    if caught_runs is not None:
        lines.append(f"caught_runs: {int(caught_runs)}")
    if caught_run_names:
        lines.append(f"caught_run_names: {caught_run_names}")
    return "<br>".join(lines)


def _write_markdown(summary: dict[str, Any]) -> None:
    final = summary["final_validation_512"]
    best = summary["best_128_milestone_011"]
    lines = [
        "# Beihang M4 T2 caught-region analysis",
        "",
        "Generated from trial CSVs under `.agents/heuristic-rl-group/beihang-m4-t2-20260601/controller-001-beihang_minimal_sim` joined to records from `scripts/generators/robust_intercept.py`.",
        "",
        "## Main result",
        "",
        "The caught cases form a broad lobe, not a single tight point cluster. `camera_azimuth_deg` is not informative for catch geometry because the current robust-intercept setup is yaw-symmetric about gravity; the useful reduced coordinates are `camera_elevation_deg`, `camera_u_fraction`, and `camera_v_fraction`, plus the range/speed grid.",
        "",
        "For `final-validation-512`, catches are concentrated in near-range and level-to-descending geometries:",
        f"- Total caught: {final['caught']} / {final['rows']} ({100.0 * final['catch_fraction']:.2f}%).",
        "- Range rates: "
        + ", ".join(
            f"R={row['range_m']:g}m: {row['caught']}/{row['total']}"
            for row in _collapse_grid(final["grid_rates"], "range_m")
        )
        + ".",
        "- The strongest elevation band is `camera_elevation_deg` from -15 to 0 deg, with adjacent bands -30 to -15 deg and -45 to -30 deg also elevated.",
        "- The strongest horizontal image-plane band is near center/slightly right: `camera_u_fraction` in [0, 0.3), followed by [-0.3, 0).",
        "- `camera_v_fraction` is not a clean separator; caught cases exist across most vertical FOV values.",
        "- A useful approximate high-yield region is `range <= 8 m`, `camera_elevation_deg` in [-45, 15], and `camera_u_fraction` in [-0.3, 0.3]. It contains 18 of the 32 caught validation samples while selecting 52 of 512 total samples.",
        "",
        "For the best 128-sample milestone (`milestone-011-stronger-aligned-thrust`):",
        f"- Total caught: {best['caught']} / {best['rows']} ({100.0 * best['catch_fraction']:.2f}%).",
        "- It shows the same broad lobe, but the coarser 128 table makes individual caught seeds look more scattered.",
        "",
        "## Artifacts",
        "",
        "- `joined_trials.csv`: every trial row joined with robust-intercept sample coordinates.",
        "- `caught_seed_summary.csv`: one row per source-table seed that was caught at least once.",
        "- `region_summary.json`: machine-readable binned rates and feature ranges.",
        "- `final_validation_512_projections.png` and `milestone_011_best_128_projections.png`: visual projections of the reduced sample space.",
        "- `sobol_samples_512_caught_union_3d.html`: interactive 3D scatter of all 512 sampled points, with every caught seed colored orange.",
        "- `sobol_samples_128_caught_union_3d.html`: equivalent interactive 3D scatter for the 128-sample milestone table.",
        "- `final_validation_512_caught_3d.html`: 3D scatter colored by catches in only the final validation run.",
        "- `sobol_samples_512_target_relative_r8_3d.html`: interactive 3D target-relative view with the pursuer at the origin and target endpoints normalized to 8 m.",
        "- `sobol_samples_128_target_relative_r8_3d.html`: equivalent target-relative view for the 128-sample table.",
        "- `final_validation_512_target_relative_r8_3d.html`: target-relative view colored by catches in only the final validation run.",
        "- `sobol_samples_512_target_relative_r8_camera_elevation_groups_3d.html`: target-relative 3D view with caught rays grouped by camera-elevation band.",
        "- `sobol_samples_128_target_relative_r8_camera_elevation_groups_3d.html`: equivalent grouped view for the 128-sample table.",
        "- `final_validation_512_target_relative_r8_camera_elevation_groups_3d.html`: final-validation grouped view.",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def _collapse_grid(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[float, dict[str, float]] = defaultdict(lambda: {"caught": 0, "total": 0})
    for row in rows:
        group = groups[float(row[key])]
        group["caught"] += int(row["caught"])
        group["total"] += int(row["total"])
    return [
        {"range_m": value, "caught": int(data["caught"]), "total": int(data["total"])}
        for value, data in sorted(groups.items())
    ]


if __name__ == "__main__":
    main()
