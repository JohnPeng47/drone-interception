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
from .base import circular_error_probable, run_drake_config, write_json, write_log_jsonl_gz
from .red_balloon import RedBalloonConfigGenerator


ZERO_NOISE = {
    "sigma_gyr": 1e-7,
    "sigma_acc": 1e-7,
    "sigma_b_gyr": 1e-9,
    "sigma_b_acc": 1e-9,
    "bias_init_std": 0.0,
    "sigma_img": 1e-6,
}
DEFAULT_ANGLES_DEG = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
CONTROLLER_GAINS = {"k_1": 0.1}


class PitchSweepGenerator(SimGenerator):
    def __init__(
        self,
        *,
        angles_deg: tuple[float, ...] = DEFAULT_ANGLES_DEG,
        n_trials: int = 20,
        telemetry_seeds: tuple[int, ...] = (1,),
        out_root: Path | None = None,
        distance_m: float | None = None,
        duration_s: float | None = None,
        closing_speed_mps: float | None = None,
        omega_max: float | None = None,
    ):
        self.angles_deg = tuple(float(angle) for angle in angles_deg)
        self.n_trials = int(n_trials)
        self.telemetry_seeds = tuple(int(seed) for seed in telemetry_seeds)
        self.out_root = out_root
        self.distance_m = distance_m
        self.duration_s = duration_s
        self.closing_speed_mps = closing_speed_mps
        self.omega_max = omega_max
        self.config_generator = RedBalloonConfigGenerator()

    def _sample_once(self, *, seed: int, pitch_deg: float | None = None, **kwargs: Any) -> SimInstance:
        theta = self.angles_deg[0] if pitch_deg is None else float(pitch_deg)
        gains = dict(CONTROLLER_GAINS)
        if self.omega_max is not None:
            gains["omega_max"] = float(self.omega_max)
        overrides: dict[str, Any] = {
            "perception": {"pixel_noise_std_px": [0.0, 0.0]},
            "vehicle": {"initial_pitch_offset_deg": theta},
            "controller": {"gains": gains},
        }
        if self.duration_s is not None:
            overrides["sim"] = {"duration_s": float(self.duration_s)}
        if "overrides" in kwargs:
            _deep_update(overrides, dict(kwargs.pop("overrides")))
        return self.config_generator.sample(
            seed=seed,
            distance_m=self.distance_m,
            closing_speed_mps=self.closing_speed_mps,
            overrides=overrides,
            **kwargs,
        )

    def run(self) -> dict[str, Any]:
        now = dt.datetime.now()
        group = f"pitch_sweep_{now.strftime('%m%d_%H%M%S')}"
        out_root = self.out_root or (
            Path(__file__).resolve().parents[4]
            / ".runs"
            / "scenarios"
            / "beihang_paper_sim_pitch_sweep"
        )
        group_dir = out_root / group
        group_dir.mkdir(parents=True, exist_ok=True)
        telemetry_set = set(self.telemetry_seeds)

        print(f"{self.n_trials} seeds x {len(self.angles_deg)} angles = {self.n_trials * len(self.angles_deg)} trials")
        print(f"angles: {list(self.angles_deg)}")
        print(f"telemetry seeds: {sorted(telemetry_set)}")
        print(f"output: {group_dir}\n")

        aggregate = {"cells": []}
        for theta in self.angles_deg:
            label = f"angle_{int(round(theta)):02d}deg"
            cell_dir = group_dir / label
            cell_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            cell_t0 = time.perf_counter()
            for seed in range(1, self.n_trials + 1):
                run_dir = cell_dir / f"run_{seed:03d}"
                try:
                    gains = dict(CONTROLLER_GAINS)
                    if self.omega_max is not None:
                        gains["omega_max"] = float(self.omega_max)
                    instance = self.sample(seed=seed, pitch_deg=theta)
                    result = run_drake_config(
                        instance.raw_config,
                        seed=seed,
                        controller_gains=gains,
                        noise_config=NoiseConfig(rng_seed=seed, **ZERO_NOISE),
                    )
                    row = {"pitch_deg": theta, **result.row()}
                    rows.append(row)
                    write_json(run_dir / "scenario_metrics.json", row)
                    if seed in telemetry_set:
                        write_log_jsonl_gz(run_dir / "telemetry.jsonl.gz", result)
                    if seed == 1 or seed % 10 == 0:
                        print(
                            f"  [{label}] seed {seed:>3}: miss={row['miss_distance_m']:.3f} "
                            f"min={row['min_distance_m']:.3f} catch={row['catch_time_s']}"
                        )
                except Exception:
                    traceback.print_exc()
                    print(f"  [{label}] seed {seed:>3}: FAILED")

            summary = _cell_summary(rows, theta, time.perf_counter() - cell_t0)
            write_json(cell_dir / "cell_summary.json", {"summary": summary, "rows": rows})
            aggregate["cells"].append(summary)
            print(
                f"  * theta={theta:>5.1f} deg  CEP_50={summary['CEP_50_miss_m']:.3f}m  "
                f"CEP_90={summary['CEP_90_miss_m']:.3f}m  "
                f"catch={summary['catch_fraction']*100:.0f}%  ({summary['wall_s']:.0f}s)\n"
            )

        aggregate["config"] = {
            "controller_gains": CONTROLLER_GAINS,
            "noise": ZERO_NOISE,
            "pixel_noise_std_px": [0.0, 0.0],
            "n_trials": self.n_trials,
            "angles_deg": list(self.angles_deg),
            "telemetry_seeds": sorted(telemetry_set),
            "distance_m": self.distance_m,
            "duration_s": self.duration_s,
            "closing_speed_mps": self.closing_speed_mps,
        }
        write_json(group_dir / "aggregate.json", aggregate)
        return {"aggregate": aggregate, "out_dir": str(group_dir)}


def _cell_summary(rows: list[dict[str, Any]], theta: float, wall_s: float) -> dict[str, Any]:
    miss = np.array([row["miss_distance_m"] for row in rows], dtype=float)
    min_dist = np.array([row["min_distance_m"] for row in rows], dtype=float)
    catches = np.array([row["catch_time_s"] is not None for row in rows], dtype=bool)
    return {
        "pitch_deg": float(theta),
        "n_trials": len(rows),
        "CEP_50_miss_m": circular_error_probable(miss, percentile=50.0),
        "CEP_90_miss_m": circular_error_probable(miss, percentile=90.0),
        "miss_mean_m": float(np.nanmean(miss)) if rows else float("nan"),
        "miss_std_m": float(np.nanstd(miss)) if rows else float("nan"),
        "CEP_50_min_dist_m": circular_error_probable(min_dist, percentile=50.0),
        "catch_fraction": float(np.mean(catches)) if rows else 0.0,
        "wall_s": float(wall_s),
    }


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angles", type=_parse_float_list, default=DEFAULT_ANGLES_DEG)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--telemetry-seeds", type=_parse_int_list, default=(1,))
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--distance-m", type=float, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--closing-speed-mps", type=float, default=None)
    parser.add_argument("--omega-max", type=float, default=None)
    args = parser.parse_args()
    PitchSweepGenerator(
        angles_deg=args.angles,
        n_trials=args.n_trials,
        telemetry_seeds=args.telemetry_seeds,
        out_root=args.out_root,
        distance_m=args.distance_m,
        duration_s=args.duration_s,
        closing_speed_mps=args.closing_speed_mps,
        omega_max=args.omega_max,
    ).run()
    return 0


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
