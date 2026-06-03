from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
PLOT_ROOT = RUN_ROOT / "plots"
POLICIES = ("beihang_minimal", "beihang_paper", "neural_policy")


def main() -> int:
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    caught_by_policy: dict[str, dict[int, list[dict[str, float]]]] = {}

    for policy in POLICIES:
        caught_seeds = _caught_seeds(RUN_ROOT / policy / "trials.csv")
        snapshots = _caught_snapshots(policy, caught_seeds)
        caught_by_policy[policy] = snapshots
        out_dir = PLOT_ROOT / policy
        out_dir.mkdir(parents=True, exist_ok=True)

        for seed, rows in sorted(snapshots.items()):
            _plot_attitude_controls(policy, seed, rows, out_dir)
            _plot_trajectory_3d(policy, seed, rows, out_dir)

        summary[policy] = {
            "caught_seeds": sorted(caught_seeds),
            "caught_count": len(caught_seeds),
            "snapshot_rows": sum(len(rows) for rows in snapshots.values()),
            "plot_dir": str(out_dir.relative_to(RUN_ROOT)),
        }

    _plot_aggregate_thrust_by_policy(caught_by_policy, PLOT_ROOT)
    _plot_aggregate_thrust_overlay(caught_by_policy, PLOT_ROOT)
    summary["aggregate_plots"] = [
        str((PLOT_ROOT / "aggregate_thrust_caught_by_policy.png").relative_to(RUN_ROOT)),
        str((PLOT_ROOT / "aggregate_thrust_caught_mean_overlay.png").relative_to(RUN_ROOT)),
    ]
    summary_path = PLOT_ROOT / "caught_trajectory_plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)
    return 0


def _caught_seeds(trials_path: Path) -> set[int]:
    with trials_path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["seed"])
            for row in csv.DictReader(handle)
            if row.get("caught") == "True" and not row.get("error")
        }


def _caught_snapshots(policy: str, caught_seeds: set[int]) -> dict[int, list[dict[str, float]]]:
    if not caught_seeds:
        return {}
    summary = json.loads((RUN_ROOT / policy / "summary.json").read_text(encoding="utf-8"))
    snapshot_path = Path(summary["snapshot_log"]["path"])
    rows_by_seed: dict[int, list[dict[str, float]]] = defaultdict(list)
    with snapshot_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            if seed not in caught_seeds:
                continue
            rows_by_seed[seed].append(_parse_snapshot_row(row))
    for rows in rows_by_seed.values():
        rows.sort(key=lambda item: item["t_s"])
    return dict(rows_by_seed)


def _parse_snapshot_row(row: dict[str, str]) -> dict[str, float]:
    qx = float(row["pursuer_qx"])
    qy = float(row["pursuer_qy"])
    qz = float(row["pursuer_qz"])
    qw = float(row["pursuer_qw"])
    roll_deg, pitch_deg = _roll_pitch_deg(qx, qy, qz, qw)
    return {
        "t_s": float(row["t_s"]),
        "tick": float(row["tick"]),
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "thrust_n": float(row["command_thrust_n"]),
        "pursuer_x": float(row["pursuer_x_w_m"]),
        "pursuer_y": float(row["pursuer_y_w_m"]),
        "pursuer_z": float(row["pursuer_z_w_m"]),
        "target_x": float(row["target_x_w_m"]),
        "target_y": float(row["target_y_w_m"]),
        "target_z": float(row["target_z_w_m"]),
        "distance_m": float(row["distance_m"]),
    }


def _roll_pitch_deg(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1.0e-12:
        return 0.0, 0.0
    x = qx / norm
    y = qy / norm
    z = qz / norm
    w = qw / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_arg)
    return math.degrees(roll), math.degrees(pitch)


def _plot_attitude_controls(policy: str, seed: int, rows: list[dict[str, float]], out_dir: Path) -> None:
    t = _values(rows, "t_s")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    fig.suptitle(f"{policy} caught trajectory seed {seed}: roll, pitch, thrust")

    axes[0].plot(t, _values(rows, "roll_deg"), color="#0b6e99", linewidth=1.4)
    axes[0].set_ylabel("roll deg")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, _values(rows, "pitch_deg"), color="#b23a48", linewidth=1.4)
    axes[1].set_ylabel("pitch deg")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t, _values(rows, "thrust_n"), color="#3a7d44", linewidth=1.4)
    axes[2].set_ylabel("thrust N")
    axes[2].set_xlabel("time s")
    axes[2].grid(True, alpha=0.25)

    fig.savefig(out_dir / f"seed_{seed:04d}_roll_pitch_thrust.png", dpi=160)
    plt.close(fig)


def _plot_trajectory_3d(policy: str, seed: int, rows: list[dict[str, float]], out_dir: Path) -> None:
    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(f"{policy} caught trajectory seed {seed}: world trajectory")

    px = _values(rows, "pursuer_x")
    py = _values(rows, "pursuer_y")
    pz = _values(rows, "pursuer_z")
    tx = _values(rows, "target_x")
    ty = _values(rows, "target_y")
    tz = _values(rows, "target_z")

    ax.plot(px, py, pz, color="#0b6e99", linewidth=1.8, label="pursuer")
    ax.plot(tx, ty, tz, color="#b23a48", linewidth=1.8, label="target")
    ax.scatter(px[0], py[0], pz[0], color="#0b6e99", marker="o", s=35, label="pursuer start")
    ax.scatter(tx[0], ty[0], tz[0], color="#b23a48", marker="o", s=35, label="target start")
    ax.scatter(px[-1], py[-1], pz[-1], color="#0b6e99", marker="x", s=55, label="pursuer end")
    ax.scatter(tx[-1], ty[-1], tz[-1], color="#b23a48", marker="x", s=55, label="target end")

    ax.set_xlabel("world x m")
    ax.set_ylabel("world y m")
    ax.set_zlabel("world z m")
    _set_equal_3d_axes(ax, np.concatenate([px, tx]), np.concatenate([py, ty]), np.concatenate([pz, tz]))
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out_dir / f"seed_{seed:04d}_trajectory_3d.png", dpi=160)
    plt.close(fig)


def _plot_aggregate_thrust_by_policy(
    caught_by_policy: dict[str, dict[int, list[dict[str, float]]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(len(POLICIES), 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    fig.suptitle("Caught trajectories: total thrust over time by policy")
    for ax, policy in zip(axes, POLICIES):
        _plot_policy_thrust_panel(ax, policy, caught_by_policy[policy])
    axes[-1].set_xlabel("time s")
    fig.savefig(out_dir / "aggregate_thrust_caught_by_policy.png", dpi=170)
    plt.close(fig)


def _plot_aggregate_thrust_overlay(
    caught_by_policy: dict[str, dict[int, list[dict[str, float]]]],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    fig.suptitle("Caught trajectories: mean total thrust by policy")
    colors = {
        "beihang_minimal": "#0b6e99",
        "beihang_paper": "#b23a48",
        "neural_policy": "#3a7d44",
    }
    for policy in POLICIES:
        grid, mean, low, high = _policy_thrust_stats(caught_by_policy[policy])
        if grid.size == 0:
            continue
        ax.plot(grid, mean, linewidth=2.0, color=colors[policy], label=f"{policy} mean")
        ax.fill_between(grid, low, high, color=colors[policy], alpha=0.12, linewidth=0)
    ax.set_xlabel("time s")
    ax.set_ylabel("thrust N")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "aggregate_thrust_caught_mean_overlay.png", dpi=170)
    plt.close(fig)


def _plot_policy_thrust_panel(ax, policy: str, rows_by_seed: dict[int, list[dict[str, float]]]) -> None:
    for seed, rows in sorted(rows_by_seed.items()):
        ax.plot(_values(rows, "t_s"), _values(rows, "thrust_n"), color="#737373", alpha=0.24, linewidth=0.8)
    grid, mean, low, high = _policy_thrust_stats(rows_by_seed)
    if grid.size:
        ax.plot(grid, mean, color="#111111", linewidth=2.0, label="mean")
        ax.fill_between(grid, low, high, color="#111111", alpha=0.12, linewidth=0, label="p10-p90")
    ax.set_title(f"{policy} ({len(rows_by_seed)} caught)")
    ax.set_ylabel("thrust N")
    ax.grid(True, alpha=0.25)
    if rows_by_seed:
        ax.legend(loc="upper right", fontsize=8)


def _policy_thrust_stats(rows_by_seed: dict[int, list[dict[str, float]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not rows_by_seed:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty
    min_dt = min(_median_dt(rows) for rows in rows_by_seed.values() if len(rows) >= 2)
    max_t = max(rows[-1]["t_s"] for rows in rows_by_seed.values())
    grid = np.arange(min_dt, max_t + 0.5 * min_dt, min_dt)
    series = []
    for rows in rows_by_seed.values():
        t = _values(rows, "t_s")
        thrust = _values(rows, "thrust_n")
        interp = np.interp(grid, t, thrust, left=np.nan, right=np.nan)
        series.append(interp)
    stacked = np.vstack(series)
    valid_columns = np.any(np.isfinite(stacked), axis=0)
    grid = grid[valid_columns]
    stacked = stacked[:, valid_columns]
    return (
        grid,
        np.nanmean(stacked, axis=0),
        np.nanpercentile(stacked, 10, axis=0),
        np.nanpercentile(stacked, 90, axis=0),
    )


def _median_dt(rows: list[dict[str, float]]) -> float:
    t = _values(rows, "t_s")
    diffs = np.diff(t)
    diffs = diffs[diffs > 0.0]
    return float(np.median(diffs)) if diffs.size else 0.005


def _values(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=float)


def _set_equal_3d_axes(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    center = np.array([
        0.5 * (float(np.min(x)) + float(np.max(x))),
        0.5 * (float(np.min(y)) + float(np.max(y))),
        0.5 * (float(np.min(z)) + float(np.max(z))),
    ])
    radius = 0.5 * max(
        float(np.max(x) - np.min(x)),
        float(np.max(y) - np.min(y)),
        float(np.max(z) - np.min(z)),
        1.0,
    )
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


if __name__ == "__main__":
    raise SystemExit(main())
