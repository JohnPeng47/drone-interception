from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SNAPSHOT_CSV = ROOT / "snapshots" / "rpg_time_optimal.csv"
TRIALS_CSV = ROOT / "trials.csv"
OUT_PNG = ROOT / "rpg_time_optimal_controller_regression_6_positions_3d.png"


def main() -> None:
    snapshots = _read_snapshots(SNAPSHOT_CSV)
    trials = _read_trials(TRIALS_CSV)
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
        ax.plot(
            pursuer[:, 0],
            pursuer[:, 1],
            pursuer[:, 2],
            color="#1f77b4",
            linewidth=1.8,
            label="pursuer",
        )
        ax.plot(
            target[:, 0],
            target[:, 1],
            target[:, 2],
            color="#d62728",
            linewidth=1.2,
            linestyle="--",
            label="target",
        )
        ax.scatter(*pursuer[0], color="#2ca02c", s=34, marker="o", label="start")
        ax.scatter(*pursuer[-1], color="#1f77b4", s=34, marker="x", label="end")
        ax.scatter(*target[0], color="#d62728", s=46, marker="^", label="target")
        status = "caught" if trial.get("caught") == "True" else "miss"
        if trial.get("out_of_bounds") == "True":
            status += ", oob"
        min_distance = float(trial.get("min_distance_m", "nan"))
        final_distance = float(trial.get("final_distance_m", "nan"))
        ax.set_title(
            f"seed {seed}: {status}\nmin {min_distance:.2f} m, final {final_distance:.2f} m",
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

    fig.suptitle("RPG time-optimal controller: controller_regression_6 world positions", fontsize=14)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)
    print(OUT_PNG)


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
                    [
                        float(row["pursuer_x_w_m"]),
                        float(row["pursuer_y_w_m"]),
                        float(row["pursuer_z_w_m"]),
                    ]
                    for row in rows
                ],
                dtype=float,
            ),
            "target": np.array(
                [
                    [
                        float(row["target_x_w_m"]),
                        float(row["target_y_w_m"]),
                        float(row["target_z_w_m"]),
                    ]
                    for row in rows
                ],
                dtype=float,
            ),
        }
    return snapshots


def _read_trials(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["seed"]): row for row in csv.DictReader(handle)}


def _common_limits(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    radius = max(radius, 1.0)
    return tuple((float(c - radius), float(c + radius)) for c in center)  # type: ignore[return-value]


if __name__ == "__main__":
    main()
