from __future__ import annotations

import argparse
import datetime as dt
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from backends import SimGenerator, SimInstance

from ...noise_config import NoiseConfig
from .base import circular_error_probable, run_drake_config, write_json, write_log_jsonl_gz
from .red_balloon import RedBalloonConfigGenerator, RED_BALLOON_X500_CONFIG


class RedBalloonTrialsGenerator(SimGenerator):
    def __init__(
        self,
        *,
        n_trials: int = 50,
        sigma_pixel_px: float = 1.0,
        out_root: Path | None = None,
        duration_s: float | None = None,
    ):
        self.n_trials = int(n_trials)
        self.sigma_pixel_px = float(sigma_pixel_px)
        self.out_root = out_root
        self.duration_s = duration_s
        self.config_generator = RedBalloonConfigGenerator()

    def sample(self, *, seed: int, **kwargs: Any) -> SimInstance:
        overrides: dict[str, Any] = {
            "perception": {"pixel_noise_std_px": [self.sigma_pixel_px, self.sigma_pixel_px]},
        }
        if self.duration_s is not None:
            overrides["sim"] = {"duration_s": float(self.duration_s)}
        if "overrides" in kwargs:
            _deep_update(overrides, dict(kwargs.pop("overrides")))
        return self.config_generator.sample(seed=seed, overrides=overrides, **kwargs)

    def run(self) -> dict[str, Any]:
        now = dt.datetime.now()
        group = f"beihang_paper_sim_{now.strftime('%m%d_%H%M%S')}"
        out_root = self.out_root or (
            Path(__file__).resolve().parents[4]
            / ".runs"
            / "scenarios"
            / "beihang_paper_sim_red_balloon"
        )
        group_dir = out_root / group
        group_dir.mkdir(parents=True, exist_ok=True)

        print(f"trials:  {self.n_trials}, sigma_pixel = {self.sigma_pixel_px} px")
        print(f"outputs: {group_dir}\n")

        rows = []
        for seed in range(1, self.n_trials + 1):
            run_dir = group_dir / f"run_{seed:03d}"
            try:
                instance = self.sample(seed=seed)
                result = run_drake_config(
                    instance.raw_config,
                    seed=seed,
                    controller_gains=instance.raw_config["controller"].get("gains"),
                    noise_config=NoiseConfig(rng_seed=seed),
                )
                row = result.row()
                rows.append(row)
                write_json(run_dir / "scenario_metrics.json", row)
                write_log_jsonl_gz(run_dir / "telemetry.jsonl.gz", result)
                print(
                    f"  seed={seed:>3d}  miss={row['miss_distance_m']:.3f} m  "
                    f"min={row['min_distance_m']:.3f}  "
                    f"catch_t={row['catch_time_s']}  ({row['wall_s']:.2f}s)"
                )
            except Exception:
                traceback.print_exc()
                print(f"  seed={seed:>3d}  FAILED")

        aggregate = _aggregate(rows)
        aggregate.update({
            "catch_radius_m": float(RED_BALLOON_X500_CONFIG["metrics"]["catch_radius_m"]),
            "paper_CEP_50Hz_m": 0.332,
            "sigma_pixel_px": self.sigma_pixel_px,
        })
        write_json(group_dir / "aggregate.json", aggregate)
        print(f"\ntrials succeeded: {len(rows)}/{self.n_trials}")
        print(f"CEP_50 (miss)       = {aggregate['CEP_50_miss_m']:.3f} m   (paper: 0.332 m)")
        print(f"CEP_90 (miss)       = {aggregate['CEP_90_miss_m']:.3f} m")
        print(f"catch fraction      = {aggregate['catch_fraction']:.2f}")
        return {"rows": rows, "aggregate": aggregate, "out_dir": str(group_dir)}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    miss = np.array([row.get("miss_distance_m", float("nan")) for row in rows], dtype=float)
    min_dist = np.array([row.get("min_distance_m", float("nan")) for row in rows], dtype=float)
    caught = np.array([row.get("catch_time_s") is not None for row in rows], dtype=bool)
    return {
        "n_trials": int(len(rows)),
        "catch_fraction": float(np.mean(caught)) if rows else 0.0,
        "CEP_50_miss_m": circular_error_probable(miss, percentile=50.0),
        "CEP_90_miss_m": circular_error_probable(miss, percentile=90.0),
        "CEP_50_min_dist_m": circular_error_probable(min_dist, percentile=50.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--sigma-pixel-px", type=float, default=1.0)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    args = parser.parse_args()
    RedBalloonTrialsGenerator(
        n_trials=args.n_trials,
        sigma_pixel_px=args.sigma_pixel_px,
        out_root=args.out_root,
        duration_s=args.duration_s,
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
