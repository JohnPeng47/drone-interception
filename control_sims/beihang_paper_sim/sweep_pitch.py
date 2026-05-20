"""Initial-pitch CEP sweep for the red-balloon scenario.

Sweeps vehicle.initial_pitch_offset_deg — the forward-pitch (about body-y)
that beihang_paper_sim/diagram.py:_apply_initial_pitch_offset composes onto the
LOS-pointing baseline quat produced by build_red_balloon_config. N seeds per
angle, zero IMU/perception noise so what varies across seeds is just balloon
drift. Mirrors the layout of sweep_closing_speeds.

NOTE: do NOT set vehicle.initial_quat_xyzw here — diagram.py right-multiplies
INITIAL_PITCH_OFFSET_DEG (default 20°) onto whatever quat it finds, so writing
the quat directly stacks the default 20° on top. Use initial_pitch_offset_deg.

Layout:
    .runs/scenarios/beihang_paper_sim_pitch_sweep/<group>/angle_XXdeg/run_NN/
        scenario_metrics.json
        telemetry.jsonl.gz       (only for seeds in --telemetry-seeds)
    .runs/scenarios/beihang_paper_sim_pitch_sweep/<group>/angle_XXdeg/cell_summary.json
    .runs/scenarios/beihang_paper_sim_pitch_sweep/<group>/aggregate.json

Usage:
    python -m control_sims.beihang_paper_sim.sweep_pitch
    python -m control_sims.beihang_paper_sim.sweep_pitch --angles 0,10,20,30 --n-trials 30
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

try:
    from ._paths import ensure_paths
except ImportError:  # Support direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from control_sims.beihang_paper_sim._paths import ensure_paths


ensure_paths()


from pydrake.systems.analysis import Simulator   # noqa: E402

from intercept_sim.analysis import compute_metrics   # noqa: E402
from intercept_sim.experiments.config import ExperimentConfig   # noqa: E402
from intercept_sim.experiments.red_balloon import (   # noqa: E402
    build_red_balloon_config, load_red_balloon_scenario, RedBalloonScenario,
)
from intercept_sim.experiments.telemetry import build_experiment_telemetry   # noqa: E402

from control_sims.beihang_paper_sim.diagram import build_diagram_from_config   # noqa: E402
from control_sims.beihang_paper_sim.noise_config import NoiseConfig   # noqa: E402


YAML = Path(__file__).resolve().parent / "configs" / "red_balloon_x500.yaml"
ZERO_NOISE = dict(
    sigma_gyr=1e-7, sigma_acc=1e-7, sigma_b_gyr=1e-9,
    sigma_b_acc=1e-9, bias_init_std=0.0, sigma_img=1e-6,
)
DEFAULT_ANGLES_DEG = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
CONTROLLER_GAINS = {"k_1": 0.1}


def _save_telemetry_gz(telemetry, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "metadata", **telemetry.to_summary_dict()},
                           sort_keys=True))
        f.write("\n")
        for step in telemetry.steps:
            f.write(json.dumps({"kind": "step", **step.to_dict()}, sort_keys=True))
            f.write("\n")


def run_trial(theta_deg: float, seed: int, out_run_dir: Path,
              save_telemetry: bool,
              distance_m: float | None = None,
              duration_s: float | None = None,
              closing_speed_mps: float | None = None,
              omega_max: float | None = None) -> dict:
    scenario_raw = dict(load_red_balloon_scenario(YAML).raw)
    scenario_overrides = {}
    if distance_m is not None:
        scenario_overrides["distance_m"] = float(distance_m)
    if closing_speed_mps is not None:
        scenario_overrides["closing_speed_mps"] = float(closing_speed_mps)
    if scenario_overrides:
        scenario_raw["scenario"] = {**scenario_raw["scenario"], **scenario_overrides}
    scenario = RedBalloonScenario(raw=scenario_raw, path=YAML)
    cfg = build_red_balloon_config(scenario, seed=seed)
    raw = dict(cfg.raw)
    if duration_s is not None:
        raw["sim"] = {**raw["sim"], "duration_s": float(duration_s)}
    raw["perception"] = {**raw["perception"], "pixel_noise_std_px": [0.0, 0.0]}
    raw["vehicle"] = {**raw["vehicle"],
                      "initial_pitch_offset_deg": float(theta_deg)}
    cfg = ExperimentConfig(raw=raw, path=cfg.path)
    nc = NoiseConfig(rng_seed=seed, **ZERO_NOISE)
    gains = dict(CONTROLLER_GAINS)
    if omega_max is not None:
        gains["omega_max"] = float(omega_max)
    diagram, logger = build_diagram_from_config(
        cfg, controller_gains=gains, noise_config=nc,
    )
    sim = Simulator(diagram)
    sim.Initialize()
    t0 = time.perf_counter()
    sim.AdvanceTo(cfg.duration_s)
    wall = time.perf_counter() - t0

    num_steps = int(math.ceil(cfg.duration_s / cfg.dt))
    log = logger.get_log()[:num_steps]
    metrics = compute_metrics(log, catch_radius_m=cfg.catch_radius_m)

    row = {
        "seed": seed,
        "pitch_deg": float(theta_deg),
        "miss_distance_m": float(metrics.miss_distance_m),
        "min_distance_m": float(metrics.min_distance_m),
        "catch_time_s": metrics.catch_time_s,
        "target_visible_fraction": float(metrics.target_visible_fraction),
        "wall_s": float(wall),
    }

    out_run_dir.mkdir(parents=True, exist_ok=True)
    with (out_run_dir / "scenario_metrics.json").open("w") as f:
        json.dump(row, f, indent=2, sort_keys=True)
        f.write("\n")

    if save_telemetry:
        telemetry = build_experiment_telemetry(
            experiment_id=cfg.name,
            comment=f"sweep_pitch theta={theta_deg:.1f}deg seed={seed}",
            config=cfg.raw,
            metrics=metrics,
            log=log,
        )
        _save_telemetry_gz(telemetry, out_run_dir / "telemetry.jsonl.gz")

    return row


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angles", type=_parse_float_list, default=DEFAULT_ANGLES_DEG,
                        help="Comma-separated pitch angles in degrees.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--telemetry-seeds", type=_parse_int_list, default=[1],
                        help="Seeds to save full telemetry.jsonl.gz for, per angle.")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--distance-m", type=float, default=None,
                        help="Override scenario.distance_m.")
    parser.add_argument("--duration-s", type=float, default=None,
                        help="Override sim.duration_s.")
    parser.add_argument("--closing-speed-mps", type=float, default=None,
                        help="Override scenario.closing_speed_mps.")
    parser.add_argument("--omega-max", type=float, default=None,
                        help="Override controller omega_max (rad/s).")
    args = parser.parse_args()

    now = dt.datetime.now()
    group = f"pitch_sweep_{now.strftime('%m%d_%H%M%S')}"
    out_root = args.out_root or (
        Path(__file__).resolve().parents[2] / ".runs" / "scenarios"
        / "beihang_paper_sim_pitch_sweep"
    )
    group_dir = out_root / group
    group_dir.mkdir(parents=True, exist_ok=True)

    total = args.n_trials * len(args.angles)
    print(f"{args.n_trials} seeds × {len(args.angles)} angles = {total} trials")
    print(f"angles: {args.angles}")
    print(f"telemetry seeds: {args.telemetry_seeds}")
    print(f"output: {group_dir}\n")

    aggregate = {"cells": []}
    telemetry_set = set(args.telemetry_seeds)
    for theta in args.angles:
        label = f"angle_{int(round(theta)):02d}deg"
        cell_dir = group_dir / label
        cell_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        cell_t0 = time.perf_counter()
        for seed in range(1, args.n_trials + 1):
            run_dir = cell_dir / f"run_{seed:03d}"
            try:
                r = run_trial(theta, seed, run_dir,
                              save_telemetry=(seed in telemetry_set),
                              distance_m=args.distance_m,
                              duration_s=args.duration_s,
                              closing_speed_mps=args.closing_speed_mps,
                              omega_max=args.omega_max)
                rows.append(r)
                if seed == 1 or seed % 10 == 0:
                    print(f"  [{label}] seed {seed:>3}: miss={r['miss_distance_m']:.3f} "
                          f"min={r['min_distance_m']:.3f} "
                          f"catch={r['catch_time_s']}")
            except Exception:
                traceback.print_exc()
                print(f"  [{label}] seed {seed:>3}: FAILED")
        cell_wall = time.perf_counter() - cell_t0
        if not rows:
            continue
        miss = np.array([r["miss_distance_m"] for r in rows], float)
        mind = np.array([r["min_distance_m"] for r in rows], float)
        catches = np.array([r["catch_time_s"] is not None for r in rows])
        summary = {
            "pitch_deg":        float(theta),
            "n_trials":         int(len(rows)),
            "CEP_50_miss_m":    float(np.nanpercentile(miss, 50)),
            "CEP_90_miss_m":    float(np.nanpercentile(miss, 90)),
            "miss_mean_m":      float(np.nanmean(miss)),
            "miss_std_m":       float(np.nanstd(miss)),
            "CEP_50_min_dist_m": float(np.nanpercentile(mind, 50)),
            "catch_fraction":   float(np.mean(catches)),
            "wall_s":           float(cell_wall),
        }
        with (cell_dir / "cell_summary.json").open("w") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2, sort_keys=True)
        aggregate["cells"].append(summary)
        print(f"  ★ θ={theta:>5.1f}°  CEP_50={summary['CEP_50_miss_m']:.3f}m  "
              f"CEP_90={summary['CEP_90_miss_m']:.3f}m  "
              f"catch={summary['catch_fraction']*100:.0f}%  ({cell_wall:.0f}s)\n")

    aggregate["config"] = {
        "yaml":              str(YAML),
        "controller_gains":  CONTROLLER_GAINS,
        "noise":             ZERO_NOISE,
        "pixel_noise_std_px": [0.0, 0.0],
        "n_trials":          args.n_trials,
        "angles_deg":        list(args.angles),
        "telemetry_seeds":   sorted(telemetry_set),
        "distance_m":        args.distance_m,
        "duration_s":        args.duration_s,
    }
    with (group_dir / "aggregate.json").open("w") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"\naggregate → {group_dir / 'aggregate.json'}")
    print("\n=== SUMMARY ===")
    for c in aggregate["cells"]:
        print(f"  θ={c['pitch_deg']:>5.1f}°  "
              f"CEP_50={c['CEP_50_miss_m']:.3f}m  "
              f"CEP_90={c['CEP_90_miss_m']:.3f}m  "
              f"catch={c['catch_fraction']*100:>5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
