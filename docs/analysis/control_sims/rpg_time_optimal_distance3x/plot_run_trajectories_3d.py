from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all RPG time-optimal run positions in one 3D PNG.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-name", default="rpg_time_optimal_positions_3d.png")
    args = parser.parse_args()

    run_dir = args.run_dir
    snapshot_path = _resolve_snapshot_csv(run_dir)
    snapshots = _read_snapshots(snapshot_path)
    trials = _read_trials(run_dir / "trials.csv")
    seeds = sorted(snapshots)
    if len(seeds) != 6:
        raise ValueError(f"expected 6 seeds, found {len(seeds)}")

    all_points = []
    for rows in snapshots.values():
        all_points.append(rows["pursuer"])
        all_points.append(rows["target"])
    limits = _common_limits(np.vstack(all_points))

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    for index, seed in enumerate(seeds, start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        data = snapshots[seed]
        trial = trials.get(seed, {})
        pursuer = data["pursuer"]
        target = data["target"]
        ax.plot(pursuer[:, 0], pursuer[:, 1], pursuer[:, 2], color="#1f77b4", linewidth=1.8, label="pursuer")
        ax.plot(target[:, 0], target[:, 1], target[:, 2], color="#d62728", linewidth=1.2, linestyle="--", label="target")
        ax.scatter(*pursuer[0], color="#2ca02c", s=34, marker="o", label="start")
        ax.scatter(*pursuer[-1], color="#1f77b4", s=34, marker="x", label="end")
        ax.scatter(*target[0], color="#d62728", s=46, marker="^", label="target")
        status = "caught" if trial.get("caught") == "True" else "miss"
        if trial.get("out_of_bounds") == "True":
            status += ", oob"
        ax.set_title(
            f"seed {seed}: {status}\n"
            f"min {float(trial.get('min_distance_m', 'nan')):.2f} m, "
            f"final {float(trial.get('final_distance_m', 'nan')):.2f} m",
            fontsize=10,
        )
        ax.set_xlabel("x_w m")
        ax.set_ylabel("y_w m")
        ax.set_zlabel("z_w m")
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_zlim(limits[2])
        ax.view_init(elev=24, azim=-54)
        ax.grid(True, alpha=0.25)
        if index == 1:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"{run_dir.name}: world positions", fontsize=14)
    out_path = run_dir / args.out_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(out_path)


def _read_snapshots(path: Path) -> dict[int, dict[str, np.ndarray]]:
    rows_by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_by_seed[int(row["seed"])].append(row)

    snapshots: dict[int, dict[str, np.ndarray]] = {}
    for seed, rows in rows_by_seed.items():
        rows.sort(key=lambda row: int(row["tick"]))
        snapshots[seed] = {
            "pursuer": np.array(
                [
                    [float(row["pursuer_x_w_m"]), float(row["pursuer_y_w_m"]), float(row["pursuer_z_w_m"])]
                    for row in rows
                ],
                dtype=float,
            ),
            "target": np.array(
                [
                    [float(row["target_x_w_m"]), float(row["target_y_w_m"]), float(row["target_z_w_m"])]
                    for row in rows
                ],
                dtype=float,
            ),
        }
    return snapshots


def _resolve_snapshot_csv(run_dir: Path) -> Path:
    snapshots_dir = run_dir / "snapshots"
    default = snapshots_dir / "rpg_time_optimal.csv"
    if default.exists():
        return default
    matches = sorted(snapshots_dir.glob("*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one snapshot CSV in {snapshots_dir}, found {len(matches)}")
    return matches[0]


def _read_trials(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["seed"]): row for row in csv.DictReader(handle)}


def _common_limits(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.55 * float(np.max(maxs - mins)), 1.0)
    return tuple((float(c - radius), float(c + radius)) for c in center)  # type: ignore[return-value]


if __name__ == "__main__":
    main()
