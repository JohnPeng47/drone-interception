"""Run Beihang paper red-balloon scenario for N trials and report CEP.

Mirrors codex_sim's run_trials.py pattern: build diagram with RunnerStepLogger,
simulate to duration_s, hand the log to intercept_sim's compute_metrics, save
per-trial telemetry.jsonl.gz + scenario_metrics.json + summary.json under
.runs/scenarios/beihang_paper_sim_red_balloon/<group>/run_<seed>/.

Usage:
    python -m control_sims.beihang_paper_sim.run_50_trials
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
    build_red_balloon_config, load_red_balloon_scenario,
)
from intercept_sim.experiments.runner import (   # noqa: E402
    ExperimentResult, save_experiment_result,
)
from intercept_sim.experiments.telemetry import build_experiment_telemetry   # noqa: E402

from control_sims.beihang_paper_sim.diagram import build_diagram_from_config   # noqa: E402
from control_sims.beihang_paper_sim.noise_config import NoiseConfig   # noqa: E402


DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "red_balloon_x500.yaml"
)


# ---------------------------------------------------------------------------
# Single-trial runner — mirrors drake_sims.experiment.run_drake_experiment.
# ---------------------------------------------------------------------------


def _save_telemetry_gz(telemetry, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "metadata", **telemetry.to_summary_dict()},
                           sort_keys=True))
        f.write("\n")
        for step in telemetry.steps:
            f.write(json.dumps({"kind": "step", **step.to_dict()}, sort_keys=True))
            f.write("\n")


def run_one_trial(
    config_path: Path,
    seed: int,
    sigma_pixel_px: float,
    out_run_dir: Path,
    duration_override: float | None = None,
) -> dict:
    # Use the scenario expansion: places vehicle / target at scenario.distance_m
    # apart along scenario.los_w, sets initial velocities, etc.
    scenario = load_red_balloon_scenario(config_path)
    cfg = build_red_balloon_config(scenario, seed=seed)
    raw = dict(cfg.raw)
    if duration_override is not None:
        raw["_override_duration_s"] = duration_override
    raw["perception"] = {
        **raw["perception"],
        "pixel_noise_std_px": [sigma_pixel_px, sigma_pixel_px],
    }
    if "_override_duration_s" in raw:
        raw["sim"] = {**raw["sim"], "duration_s": float(raw.pop("_override_duration_s"))}
    cfg = ExperimentConfig(raw=raw, path=cfg.path)
    nc = NoiseConfig(rng_seed=seed)

    controller_gains = raw.get("controller", {}).get("gains")
    diagram, logger = build_diagram_from_config(
        cfg, controller_gains=controller_gains, noise_config=nc
    )
    sim = Simulator(diagram)
    sim.Initialize()

    t0 = time.perf_counter()
    sim.AdvanceTo(cfg.duration_s)
    wall_s = time.perf_counter() - t0

    num_steps = int(math.ceil(cfg.duration_s / cfg.dt))
    log = logger.get_log()[:num_steps]
    metrics = compute_metrics(log, catch_radius_m=cfg.catch_radius_m)
    telemetry = build_experiment_telemetry(
        experiment_id=cfg.name,
        comment=f"beihang_paper_sim seed={seed}",
        config=cfg.raw,
        metrics=metrics,
        log=log,
    )

    result = ExperimentResult(
        config=cfg, log=log, metrics=metrics,
        comment=f"beihang_paper_sim seed={seed}", telemetry=telemetry,
    )
    out_run_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_result(result, out_run_dir / "summary.json")
    _save_telemetry_gz(telemetry, out_run_dir / "telemetry.jsonl.gz")
    with (out_run_dir / "scenario_metrics.json").open("w") as f:
        json.dump({**metrics.to_dict(), "seed": seed, "wall_s": wall_s},
                  f, indent=2, sort_keys=True)
        f.write("\n")

    return {"seed": seed, "metrics": metrics.to_dict(), "wall_s": wall_s,
            "run_dir": str(out_run_dir)}


# ---------------------------------------------------------------------------
# 50-trial driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--sigma-pixel-px", type=float, default=1.0)
    parser.add_argument("--out-root", type=Path, default=None,
                         help="Defaults to .runs/scenarios/beihang_paper_sim_red_balloon")
    parser.add_argument("--duration-s", type=float, default=None,
                         help="Override scenario duration_s.")
    args = parser.parse_args()

    now = dt.datetime.now()
    group = f"beihang_paper_sim_{now.strftime('%m%d_%H%M%S')}"
    out_root = args.out_root or (
        Path(__file__).resolve().parents[2] / ".runs" / "scenarios" / "beihang_paper_sim_red_balloon"
    )
    group_dir = out_root / group
    group_dir.mkdir(parents=True, exist_ok=True)

    print(f"config:  {args.config}")
    print(f"trials:  {args.n_trials}, σ_pixel = {args.sigma_pixel_px} px")
    print(f"outputs: {group_dir}")
    print()

    rows = []
    for k in range(args.n_trials):
        seed = k + 1
        run_dir = group_dir / f"run_{seed:03d}"
        try:
            r = run_one_trial(args.config, seed, args.sigma_pixel_px, run_dir,
                              duration_override=args.duration_s)
            md = r["metrics"]
            print(f"  seed={seed:>3d}  miss={md.get('miss_distance_m', float('nan')):.3f} m  "
                  f"min={md.get('min_distance_m', float('nan')):.3f}  "
                  f"catch_t={md.get('catch_time_s', None)}  "
                  f"({r['wall_s']:.2f}s)")
            rows.append(r)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print(f"  seed={seed:>3d}  FAILED")

    # Aggregate
    if rows:
        miss = np.array([r["metrics"].get("miss_distance_m", float("nan")) for r in rows], float)
        mind = np.array([r["metrics"].get("min_distance_m", float("nan")) for r in rows], float)
        caught = np.array([r["metrics"].get("catch_time_s") is not None for r in rows])
        cep50 = float(np.nanpercentile(miss, 50))
        cep90 = float(np.nanpercentile(miss, 90))
        cep50_min = float(np.nanpercentile(mind, 50))
        catch_rate = float(np.mean(caught))

        aggregate = {
            "n_trials":         int(len(rows)),
            "catch_radius_m":   float(load_red_balloon_scenario(args.config).raw.get("metrics", {}).get("catch_radius_m", 0.5)),
            "catch_fraction":   catch_rate,
            "CEP_50_miss_m":    cep50,
            "CEP_90_miss_m":    cep90,
            "CEP_50_min_dist_m": cep50_min,
            "paper_CEP_50Hz_m": 0.332,
            "config":           str(args.config),
            "sigma_pixel_px":   args.sigma_pixel_px,
        }
        with (group_dir / "aggregate.json").open("w") as f:
            json.dump(aggregate, f, indent=2, sort_keys=True)
            f.write("\n")

        print()
        print(f"trials succeeded: {len(rows)}/{args.n_trials}")
        print(f"CEP_50 (miss)       = {cep50:.3f} m   (paper: 0.332 m)")
        print(f"CEP_90 (miss)       = {cep90:.3f} m")
        print(f"CEP_50 (min_dist)   = {cep50_min:.3f} m")
        print(f"catch fraction      = {catch_rate:.2f}  "
              f"(radius {aggregate['catch_radius_m']} m)")
        print()
        print(f"aggregate -> {group_dir / 'aggregate.json'}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
