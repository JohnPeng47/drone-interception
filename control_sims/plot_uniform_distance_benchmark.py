"""Plot robust uniform-distance benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.generators.robust_intercept_uniform_distance import (
    RobustInterceptUniformDistanceConfigGenerator,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(run_dir / "trials.csv")
    data = _join_generator_geometry(rows)
    written = [
        _plot_initial_sphere(data, out_dir),
        _plot_azimuth_elevation(data, out_dir),
        _plot_camera_uv(data, out_dir),
        _plot_speed_bars(data, out_dir),
        _plot_range_speed(data, out_dir),
        _plot_min_distance_vs_range(data, out_dir),
        _plot_visibility_vs_range(data, out_dir),
        _plot_catch_time(data, out_dir),
    ]
    manifest = {
        "run_dir": str(run_dir),
        "plots": [str(path) for path in written],
    }
    (out_dir / "plots_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _join_generator_geometry(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    generator = RobustInterceptUniformDistanceConfigGenerator()
    values: dict[str, list[Any]] = {
        "seed": [],
        "sim": [],
        "caught": [],
        "range_m": [],
        "closing_speed_mps": [],
        "min_distance_m": [],
        "final_distance_m": [],
        "visible_fraction": [],
        "catch_time_s": [],
        "camera_u_fraction": [],
        "camera_v_fraction": [],
        "camera_azimuth_deg": [],
        "camera_elevation_deg": [],
        "los_azimuth_deg": [],
        "los_elevation_deg": [],
        "rel_x": [],
        "rel_y": [],
        "rel_z": [],
    }
    for row in rows:
        seed = int(row["seed"])
        point = generator._by_seed[seed]
        instance = generator._sample_once(seed=seed)
        target_initial = instance.target_initials[0]
        rel = np.asarray(instance.pursuer_initial.position_w, dtype=float) - np.asarray(target_initial.position_w, dtype=float)
        los = -rel / max(float(np.linalg.norm(rel)), 1e-12)
        los_az, los_el = _az_el_deg(los)

        values["seed"].append(seed)
        values["sim"].append(row["sim"])
        values["caught"].append(_as_bool(row["caught"]))
        values["range_m"].append(float(row["range_m"]))
        values["closing_speed_mps"].append(float(row["closing_speed_mps"]))
        values["min_distance_m"].append(_as_float(row["min_distance_m"]))
        values["final_distance_m"].append(_as_float(row["final_distance_m"]))
        values["visible_fraction"].append(_as_float(row["visible_fraction"]))
        values["catch_time_s"].append(_as_float(row["catch_time_s"]))
        values["camera_u_fraction"].append(float(point.values["camera_u_fraction"]))
        values["camera_v_fraction"].append(float(point.values["camera_v_fraction"]))
        values["camera_azimuth_deg"].append(float(point.values["camera_azimuth_deg"]))
        values["camera_elevation_deg"].append(float(point.values["camera_elevation_deg"]))
        values["los_azimuth_deg"].append(los_az)
        values["los_elevation_deg"].append(los_el)
        values["rel_x"].append(float(rel[0]))
        values["rel_y"].append(float(rel[1]))
        values["rel_z"].append(float(rel[2]))

    return {
        key: np.array(val, dtype=object if key == "sim" else float if key != "caught" else bool)
        for key, val in values.items()
    }


def _plot_initial_sphere(data: dict[str, np.ndarray], out_dir: Path) -> Path:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    caught = data["caught"]
    missed = ~caught

    ax.scatter(
        data["rel_x"][missed],
        data["rel_y"][missed],
        data["rel_z"][missed],
        c="#d62728",
        s=14,
        alpha=0.72,
        label="miss",
        depthshade=False,
    )
    ax.scatter(
        data["rel_x"][caught],
        data["rel_y"][caught],
        data["rel_z"][caught],
        c="#2ca02c",
        s=18,
        alpha=0.85,
        label="hit",
        depthshade=False,
    )
    _wire_sphere(ax, 5.0, "#444444", 0.20)
    _wire_sphere(ax, 20.0, "#777777", 0.10)
    ax.scatter([0.0], [0.0], [0.0], c="black", s=50, marker="*", label="target")
    ax.set_title("Initial positions around target")
    ax.set_xlabel("x relative to target (m)")
    ax.set_ylabel("y relative to target (m)")
    ax.set_zlabel("z relative to target (m)")
    ax.legend(loc="upper right")
    _set_3d_equal(ax, 21.0)
    path = out_dir / "initial_sphere_hit_miss.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_azimuth_elevation(data: dict[str, np.ndarray], out_dir: Path) -> Path:
    path = out_dir / "los_azimuth_elevation_hit_miss.png"
    _plot_binned_outcomes(
        x=data["los_azimuth_deg"],
        y=data["los_elevation_deg"],
        caught=data["caught"],
        x_bins=np.linspace(0.0, 360.0, 37),
        y_bins=np.linspace(-90.0, 90.0, 19),
        x_label="LOS azimuth deg",
        y_label="LOS elevation deg",
        title="LOS azimuth/elevation outcomes",
        path=path,
    )
    return path


def _plot_camera_uv(data: dict[str, np.ndarray], out_dir: Path) -> Path:
    path = out_dir / "camera_uv_hit_miss.png"
    _plot_binned_outcomes(
        x=data["camera_u_fraction"],
        y=data["camera_v_fraction"],
        caught=data["caught"],
        x_bins=np.linspace(-0.9, 0.9, 25),
        y_bins=np.linspace(-0.9, 0.9, 25),
        x_label="camera u fraction",
        y_label="camera v fraction",
        title="Initial image-plane outcomes",
        path=path,
    )
    return path


def _plot_binned_outcomes(
    *,
    x: np.ndarray,
    y: np.ndarray,
    caught: np.ndarray,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    path: Path,
) -> None:
    total = _hist2d(x, y, x_bins, y_bins)
    hits = _hist2d(x[caught], y[caught], x_bins, y_bins)
    misses = _hist2d(x[~caught], y[~caught], x_bins, y_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        hit_rate = np.where(total > 0, hits / total, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    fig.suptitle(title)
    _imshow(axes[0], hits, x_bins, y_bins, "hits", "Greens", x_label, y_label)
    _imshow(axes[1], misses, x_bins, y_bins, "misses", "Reds", x_label, y_label)
    _imshow(axes[2], hit_rate, x_bins, y_bins, "hit rate", "viridis", x_label, y_label, vmin=0.0, vmax=1.0)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_speed_bars(data: dict[str, np.ndarray], out_dir: Path) -> Path:
    speeds = np.array(sorted(set(data["closing_speed_mps"])), dtype=float)
    hit_rate = []
    min_p50 = []
    visible = []
    for speed in speeds:
        mask = data["closing_speed_mps"] == speed
        hit_rate.append(float(np.mean(data["caught"][mask])))
        min_p50.append(float(np.nanpercentile(data["min_distance_m"][mask], 50)))
        visible.append(float(np.nanmean(data["visible_fraction"][mask])))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].bar([str(int(s)) for s in speeds], hit_rate, color="#2ca02c")
    axes[0].set_title("Hit fraction")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("initial closing speed m/s")
    axes[0].set_ylabel("fraction")
    axes[1].bar([str(int(s)) for s in speeds], min_p50, color="#1f77b4")
    axes[1].set_title("Median min distance")
    axes[1].set_xlabel("initial closing speed m/s")
    axes[1].set_ylabel("m")
    axes[2].bar([str(int(s)) for s in speeds], visible, color="#9467bd")
    axes[2].set_title("Mean visible fraction")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_xlabel("initial closing speed m/s")
    path = out_dir / "speed_bucket_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_range_speed(data: dict[str, np.ndarray], out_dir: Path) -> Path:
    path = out_dir / "range_speed_hit_rate.png"
    _plot_binned_outcomes(
        x=data["range_m"],
        y=data["closing_speed_mps"],
        caught=data["caught"],
        x_bins=np.linspace(5.0, 20.0, 16),
        y_bins=np.array([-0.5, 2.5, 7.5, 15.0, 22.5]),
        x_label="initial range m",
        y_label="initial closing speed m/s",
        title="Range and speed outcomes",
        path=path,
    )
    return path


def _plot_min_distance_vs_range(data: dict[str, np.ndarray], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    caught = data["caught"]
    ax.scatter(data["range_m"][~caught], data["min_distance_m"][~caught], c="#d62728", s=12, alpha=0.35, label="miss")
    ax.scatter(data["range_m"][caught], data["min_distance_m"][caught], c="#2ca02c", s=14, alpha=0.7, label="hit")
    ax.axhline(0.5, color="black", linewidth=1.0, linestyle="--", label="capture radius")
    ax.set_title("Minimum distance vs initial range")
    ax.set_xlabel("initial range m")
    ax.set_ylabel("minimum distance m")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = out_dir / "min_distance_vs_range.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_visibility_vs_range(data: dict[str, np.ndarray], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    caught = data["caught"]
    ax.scatter(data["range_m"][~caught], data["visible_fraction"][~caught], c="#d62728", s=12, alpha=0.35, label="miss")
    ax.scatter(data["range_m"][caught], data["visible_fraction"][caught], c="#2ca02c", s=14, alpha=0.7, label="hit")
    ax.set_title("Visibility vs initial range")
    ax.set_xlabel("initial range m")
    ax.set_ylabel("visible fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = out_dir / "visibility_vs_range.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_catch_time(data: dict[str, np.ndarray], out_dir: Path) -> Path:
    caught = data["caught"]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for speed in sorted(set(data["closing_speed_mps"])):
        mask = caught & (data["closing_speed_mps"] == speed)
        values = data["catch_time_s"][mask]
        values = values[np.isfinite(values)]
        if values.size:
            ax.hist(values, bins=np.linspace(0.0, 3.0, 31), alpha=0.45, label=f"{int(speed)} m/s")
    ax.set_title("Catch-time distribution by speed bucket")
    ax.set_xlabel("catch time s")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.legend(title="closing speed")
    path = out_dir / "catch_time_histogram.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _wire_sphere(ax, radius: float, color: str, alpha: float) -> None:
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.5)


def _set_3d_equal(ax, limit: float) -> None:
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))


def _hist2d(x: np.ndarray, y: np.ndarray, x_bins: np.ndarray, y_bins: np.ndarray) -> np.ndarray:
    hist, _, _ = np.histogram2d(x.astype(float), y.astype(float), bins=(x_bins, y_bins))
    return hist.T


def _imshow(
    ax,
    data: np.ndarray,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    title: str,
    cmap: str,
    x_label: str,
    y_label: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(False)
    plt.colorbar(image, ax=ax)


def _az_el_deg(direction: np.ndarray) -> tuple[float, float]:
    unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
    az = float(np.degrees(np.arctan2(unit[1], unit[0])))
    if az < 0.0:
        az += 360.0
    el = float(np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0))))
    return az, el


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: Any) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text)


if __name__ == "__main__":
    raise SystemExit(main())
