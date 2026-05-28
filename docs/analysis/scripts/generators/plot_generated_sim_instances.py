from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backends.csim.generator.instance_store import read_sim_instances


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_SCENARIO_TABLE = Path("scripts/generators/sim_instances/sobol_samples.csimin")
DEFAULT_OUT_DIR = Path("docs/analysis/scripts/generators/generated_sim_instance_plots")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot generated SimInstance initial-condition coverage.")
    parser.add_argument("--scenario-table", type=Path, default=DEFAULT_SCENARIO_TABLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    instances = read_sim_instances(args.scenario_table, count=args.max_samples)
    if not instances:
        raise ValueError(f"{args.scenario_table} did not contain any SimInstances")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _extract_initial_conditions(instances)
    paths = {
        "world_3d": out_dir / "initial_positions_world_3d.png",
        "azimuth_bearing": out_dir / "los_azimuth_elevation.png",
        "uv_offsets": out_dir / "camera_uv_offsets.png",
        "summary": out_dir / "summary.json",
    }

    _plot_world_3d(data, paths["world_3d"])
    _plot_azimuth_bearing(data, paths["azimuth_bearing"])
    _plot_uv_offsets(data, paths["uv_offsets"])
    paths["summary"].write_text(
        json.dumps(_summary(data, args.scenario_table, paths), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))


def _extract_initial_conditions(instances) -> dict[str, Any]:
    pursuer_positions = []
    target_positions = []
    los_vectors = []
    ranges = []
    azimuth_deg = []
    elevation_deg = []
    uv_offsets = []
    visible = []

    for instance in instances:
        pursuer = instance.pursuer_initial
        target = instance.target_initials[0]
        camera = instance.config.cameras[0]
        p_w = np.asarray(pursuer.position_w, dtype=float).reshape(3)
        target_w = np.asarray(target.position_w, dtype=float).reshape(3)
        rel_w = target_w - p_w
        range_m = float(np.linalg.norm(rel_w))
        los_w = rel_w / max(range_m, 1e-12)
        azimuth = math.degrees(math.atan2(float(los_w[1]), float(los_w[0])))
        if azimuth < 0.0:
            azimuth += 360.0
        elevation = math.degrees(math.asin(np.clip(float(los_w[2]), -1.0, 1.0)))

        rotation_wb = _quat_xyzw_to_matrix(np.asarray(pursuer.quat_xyzw, dtype=float).reshape(4))
        camera_position_w = p_w + rotation_wb @ np.asarray(camera.position_b, dtype=float).reshape(3)
        rel_b = rotation_wb.T @ (target_w - camera_position_w)
        rel_c = np.asarray(camera.body_to_camera, dtype=float).reshape(3, 3) @ rel_b
        depth = float(rel_c[0])
        if depth > 1.0e-9:
            uv = np.array([rel_c[1] / depth, rel_c[2] / depth], dtype=float)
            intr = camera.intrinsics
            is_visible = (
                abs(float(uv[0])) <= math.tan(float(intr.hfov_rad) / 2.0)
                and abs(float(uv[1])) <= math.tan(float(intr.vfov_rad) / 2.0)
            )
        else:
            uv = np.array([np.nan, np.nan], dtype=float)
            is_visible = False

        pursuer_positions.append(p_w)
        target_positions.append(target_w)
        los_vectors.append(los_w)
        ranges.append(range_m)
        azimuth_deg.append(azimuth)
        elevation_deg.append(elevation)
        uv_offsets.append(uv)
        visible.append(is_visible)

    return {
        "pursuer_positions": np.asarray(pursuer_positions, dtype=float),
        "target_positions": np.asarray(target_positions, dtype=float),
        "target_center": np.mean(np.asarray(target_positions, dtype=float), axis=0),
        "los_vectors": np.asarray(los_vectors, dtype=float),
        "ranges": np.asarray(ranges, dtype=float),
        "azimuth_deg": np.asarray(azimuth_deg, dtype=float),
        "elevation_deg": np.asarray(elevation_deg, dtype=float),
        "uv_offsets": np.asarray(uv_offsets, dtype=float),
        "visible": np.asarray(visible, dtype=bool),
    }


def _plot_world_3d(data: dict[str, Any], path: Path) -> None:
    positions = data["pursuer_positions"]
    target_center = data["target_center"]
    ranges = data["ranges"]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=ranges,
        s=16,
        cmap="viridis",
        alpha=0.85,
        label="pursuer initial",
    )
    ax.scatter(
        [target_center[0]],
        [target_center[1]],
        [target_center[2]],
        c="#d62728",
        s=80,
        marker="*",
        label="target",
    )

    shell_radii = _shell_radii(ranges)
    for radius in shell_radii:
        _draw_range_shell(ax, target_center, radius)

    _set_equal_3d_axes(ax, np.vstack([positions, target_center.reshape(1, 3)]))
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_zlabel("world z (m)")
    ax.set_title("Generated initial positions in world frame")
    ax.legend(loc="upper right")
    fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08, label="range to target (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_azimuth_bearing(data: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    scatter = ax.scatter(
        data["azimuth_deg"],
        data["elevation_deg"],
        c=data["ranges"],
        s=18,
        cmap="viridis",
        alpha=0.85,
    )
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("LOS azimuth in world frame (deg)")
    ax.set_ylabel("LOS elevation in world frame (deg)")
    ax.set_title("Target bearing from pursuer initial state")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="range to target (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_uv_offsets(data: dict[str, Any], path: Path) -> None:
    uv = data["uv_offsets"]
    visible = data["visible"]
    ranges = data["ranges"]

    fig, ax = plt.subplots(figsize=(7, 7))
    if np.any(~visible):
        ax.scatter(
            uv[~visible, 0],
            uv[~visible, 1],
            c="#a0a0a0",
            s=22,
            marker="x",
            label="outside FOV / behind",
        )
    scatter = ax.scatter(
        uv[visible, 0],
        uv[visible, 1],
        c=ranges[visible],
        s=18,
        cmap="viridis",
        alpha=0.9,
        label="visible",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlabel("camera u = y / x")
    ax.set_ylabel("camera v = z / x")
    ax.set_title("Initial camera-frame target offsets")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    if np.any(visible):
        fig.colorbar(scatter, ax=ax, label="range to target (m)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summary(data: dict[str, Any], scenario_table: Path, paths: dict[str, Path]) -> dict[str, Any]:
    ranges = data["ranges"]
    visible = data["visible"]
    uv = data["uv_offsets"]
    return {
        "scenario_table": str(scenario_table),
        "sample_count": int(ranges.size),
        "target_center_w": [float(value) for value in data["target_center"]],
        "range_m": {
            "min": float(np.min(ranges)),
            "max": float(np.max(ranges)),
            "mean": float(np.mean(ranges)),
            "unique_rounded": [float(value) for value in np.unique(np.round(ranges, 3))[:32]],
        },
        "los_azimuth_deg": {
            "min": float(np.min(data["azimuth_deg"])),
            "max": float(np.max(data["azimuth_deg"])),
        },
        "los_elevation_deg": {
            "min": float(np.min(data["elevation_deg"])),
            "max": float(np.max(data["elevation_deg"])),
        },
        "uv": {
            "visible_fraction": float(np.mean(visible)),
            "u_min": _finite_min(uv[:, 0]),
            "u_max": _finite_max(uv[:, 0]),
            "v_min": _finite_min(uv[:, 1]),
            "v_max": _finite_max(uv[:, 1]),
        },
        "plots": {name: str(path) for name, path in paths.items() if name != "summary"},
    }


def _shell_radii(ranges: np.ndarray) -> list[float]:
    unique = np.unique(np.round(ranges, 3))
    if 1 <= len(unique) <= 6:
        return [float(value) for value in unique]
    return [float(np.median(ranges))]


def _draw_range_shell(ax: Any, center: np.ndarray, radius: float) -> None:
    phi = np.linspace(0.0, math.pi, 24)
    theta = np.linspace(0.0, 2.0 * math.pi, 48)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    x = center[0] + radius * np.sin(phi_grid) * np.cos(theta_grid)
    y = center[1] + radius * np.sin(phi_grid) * np.sin(theta_grid)
    z = center[2] + radius * np.cos(phi_grid)
    ax.plot_wireframe(x, y, z, color="#6f6f6f", linewidth=0.35, alpha=0.18)


def _set_equal_3d_axes(ax: Any, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)), 1.0)
    half = 0.55 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _finite_min(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.min(finite))


def _finite_max(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.max(finite))


if __name__ == "__main__":
    main()
