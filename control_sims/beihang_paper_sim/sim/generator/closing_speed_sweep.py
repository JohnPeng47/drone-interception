from __future__ import annotations

import argparse
import datetime as dt
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from backends import SimGenerator, SimInstance

from ...noise_config import NoiseConfig
from .base import circular_error_probable, run_drake_config, write_json
from .red_balloon import RedBalloonConfigGenerator


PIXEL_NOISE_STD_PX = 1.0
DEFAULT_CELLS = (
    (1.0, 8.0, 4.0),
    (10.0, 20.0, 4.0),
    (15.0, 30.0, 4.0),
    (20.0, 40.0, 5.0),
)


class ClosingSpeedSweepGenerator(SimGenerator):
    def __init__(
        self,
        *,
        n_trials: int = 50,
        cells: tuple[tuple[float, float, float], ...] = DEFAULT_CELLS,
        out_root: Path | None = None,
    ):
        self.n_trials = int(n_trials)
        self.cells = tuple((float(cs), float(dist), float(dur)) for cs, dist, dur in cells)
        self.out_root = out_root
        self.config_generator = RedBalloonConfigGenerator()

    def _sample_once(
        self,
        *,
        seed: int,
        closing_speed_mps: float | None = None,
        distance_m: float | None = None,
        duration_s: float | None = None,
        **kwargs: Any,
    ) -> SimInstance:
        if closing_speed_mps is None or distance_m is None or duration_s is None:
            default_closing_speed, default_distance, default_duration = self.cells[0]
            closing_speed_mps = default_closing_speed if closing_speed_mps is None else closing_speed_mps
            distance_m = default_distance if distance_m is None else distance_m
            duration_s = default_duration if duration_s is None else duration_s
        overrides: dict[str, Any] = {
            "sim": {"duration_s": float(duration_s)},
            "perception": {"pixel_noise_std_px": [PIXEL_NOISE_STD_PX, PIXEL_NOISE_STD_PX]},
        }
        if "overrides" in kwargs:
            _deep_update(overrides, dict(kwargs.pop("overrides")))
        return self.config_generator.sample(
            seed=seed,
            distance_m=distance_m,
            closing_speed_mps=closing_speed_mps,
            overrides=overrides,
            **kwargs,
        )

    def run(self) -> dict[str, Any]:
        now = dt.datetime.now()
        group = f"closing_sweep_{now.strftime('%m%d_%H%M%S')}"
        out_root = self.out_root or (
            Path(__file__).resolve().parents[4]
            / ".runs"
            / "scenarios"
            / "beihang_paper_sim_closing_sweep"
        )
        group_dir = out_root / group
        group_dir.mkdir(parents=True, exist_ok=True)

        print(f"{self.n_trials} seeds x {len(self.cells)} closing speeds = {self.n_trials * len(self.cells)} trials")
        print(f"output: {group_dir}\n")

        aggregate = {"cells": []}
        for closing_speed, distance_m, duration_s in self.cells:
            label = f"cs_{int(closing_speed):02d}"
            cell_dir = group_dir / label
            cell_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            cell_t0 = time.perf_counter()
            for seed in range(1, self.n_trials + 1):
                try:
                    instance = self.sample(
                        seed=seed,
                        distance_m=distance_m,
                        closing_speed_mps=closing_speed,
                        duration_s=duration_s,
                    )
                    result = run_drake_config(
                        instance.raw_config,
                        seed=seed,
                        noise_config=NoiseConfig(rng_seed=seed),
                    )
                    row = {
                        **result.row(),
                        "closing_speed_mps": closing_speed,
                        "distance_m": distance_m,
                        "duration_s": duration_s,
                        **_speed_metrics(result.log),
                    }
                    rows.append(row)
                    if seed % 10 == 0:
                        print(
                            f"  [{label}] seed {seed:>2}: miss={row['miss_distance_m']:.3f} "
                            f"catch={row['catch_time_s']}"
                        )
                except Exception:
                    traceback.print_exc()
                    print(f"  [{label}] seed {seed:>2}: FAILED")

            summary = _cell_summary(rows, closing_speed, distance_m, duration_s, time.perf_counter() - cell_t0)
            write_json(cell_dir / "cell_summary.json", {"summary": summary, "rows": rows})
            aggregate["cells"].append(summary)
            print(
                f"  * cs={closing_speed:>5}m/s  CEP_50={summary['CEP_50_miss_m']:.3f}m  "
                f"catch={summary['catch_fraction']*100:.0f}%  ({summary['wall_s']:.0f}s)\n"
            )

        aggregate["config"] = {
            "k_1": "beihang_paper_sim default",
            "init_pitch_deg": "beihang_paper_sim default",
            "pixel_noise_std_px": [PIXEL_NOISE_STD_PX, PIXEL_NOISE_STD_PX],
        }
        write_json(group_dir / "aggregate.json", aggregate)
        return {"aggregate": aggregate, "out_dir": str(group_dir)}


def _speed_metrics(log) -> dict[str, float]:
    if not log:
        return {"peak_closing_rate_mps": float("nan"), "peak_speed_mps": float("nan")}
    v_w = np.array([step.rotorpy_state["v"] for step in log], dtype=float)
    pursuer = np.array([step.scene.pursuer.position_w for step in log], dtype=float)
    target = np.array([step.scene.targets[0].position_w for step in log], dtype=float)
    rel = target - pursuer
    los_unit = rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-9)
    closing_rate = np.sum(v_w * los_unit, axis=1)
    return {
        "peak_closing_rate_mps": float(np.max(closing_rate)),
        "peak_speed_mps": float(np.max(np.linalg.norm(v_w, axis=1))),
    }


def _cell_summary(rows: list[dict[str, Any]], closing_speed: float, distance_m: float, duration_s: float, wall_s: float) -> dict[str, Any]:
    miss = np.array([row["miss_distance_m"] for row in rows], dtype=float)
    min_dist = np.array([row["min_distance_m"] for row in rows], dtype=float)
    catches = np.array([row["catch_time_s"] is not None for row in rows], dtype=bool)
    return {
        "closing_speed_mps": closing_speed,
        "distance_m": distance_m,
        "duration_s": duration_s,
        "n_trials": len(rows),
        "CEP_50_miss_m": circular_error_probable(miss, percentile=50.0),
        "CEP_90_miss_m": circular_error_probable(miss, percentile=90.0),
        "CEP_50_min_dist_m": circular_error_probable(min_dist, percentile=50.0),
        "catch_fraction": float(np.mean(catches)) if rows else 0.0,
        "mean_peak_closing_mps": float(np.mean([row["peak_closing_rate_mps"] for row in rows])) if rows else float("nan"),
        "mean_peak_speed_mps": float(np.mean([row["peak_speed_mps"] for row in rows])) if rows else float("nan"),
        "mean_target_visible_fraction": float(np.mean([row["target_visible_fraction"] for row in rows])) if rows else float("nan"),
        "wall_s": float(wall_s),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()
    ClosingSpeedSweepGenerator(n_trials=args.n_trials, out_root=args.out_root).run()
    return 0


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
