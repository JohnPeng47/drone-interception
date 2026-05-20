"""C2-config CEP sweep across closing speeds, with default noise.

50 seeds × 4 closing speeds (1, 10, 15, 20 m/s) with distance scaled to give
~2 s of ballistic engagement. Random balloon drift direction per seed.

Noise: beihang_paper_sim default NoiseConfig + 1.0 px perception centroid noise
(standard centroid-quality assumption). The pitch (20°) and k_1 (0.1) are
inherited from beihang_paper_sim defaults — see diagram.py and
controller/control_core.py respectively. We deliberately do NOT override
them here.

Outputs aggregate.json + per-trial summary under
.runs/scenarios/beihang_paper_sim_closing_sweep/<group>/cs_XX/run_NN/
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

from control_sims.beihang_paper_sim.diagram import build_diagram_from_config   # noqa: E402
from control_sims.beihang_paper_sim.noise_config import NoiseConfig   # noqa: E402


YAML = Path(__file__).resolve().parent / "configs" / "red_balloon_x500.yaml"
PIXEL_NOISE_STD_PX = 1.0  # standard centroid-quality assumption

CELLS = [
    # (closing_speed_mps, distance_m, duration_s)
    (1.0,   8.0, 4.0),
    (10.0, 20.0, 4.0),
    (15.0, 30.0, 4.0),
    (20.0, 40.0, 5.0),
]


def run_trial(closing_speed, distance_m, duration_s, seed) -> dict:
    scenario_raw = dict(load_red_balloon_scenario(YAML).raw)
    scenario_raw["scenario"] = {
        **scenario_raw["scenario"],
        "closing_speed_mps": float(closing_speed),
        "distance_m": float(distance_m),
    }
    scenario_p = RedBalloonScenario(raw=scenario_raw, path=YAML)
    cfg = build_red_balloon_config(scenario_p, seed=seed)
    raw = dict(cfg.raw)
    raw["sim"] = {**raw["sim"], "duration_s": float(duration_s)}
    raw["perception"] = {
        **raw["perception"],
        "pixel_noise_std_px": [PIXEL_NOISE_STD_PX, PIXEL_NOISE_STD_PX],
    }
    # Pitch (20°) and k_1 (0.1) come from beihang_paper_sim defaults — no override.
    cfg = ExperimentConfig(raw=raw, path=cfg.path)
    nc = NoiseConfig(rng_seed=seed)
    diagram, logger = build_diagram_from_config(cfg, noise_config=nc)
    sim = Simulator(diagram)
    sim.Initialize()
    t0 = time.perf_counter()
    sim.AdvanceTo(cfg.duration_s)
    wall = time.perf_counter() - t0

    log = logger.get_log()
    metrics = compute_metrics(log, catch_radius_m=cfg.catch_radius_m)

    # peak closing rate
    v_w = np.array([s.rotorpy_state["v"] for s in log])
    pursuer = np.array([s.scene.pursuer.position_w for s in log])
    target = np.array([s.scene.targets[0].position_w for s in log])
    rel = target - pursuer
    los_unit = rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-9)
    closing_rate = np.sum(v_w * los_unit, axis=1)

    return {
        "seed": seed,
        "miss_distance_m": float(metrics.miss_distance_m),
        "min_distance_m": float(metrics.min_distance_m),
        "catch_time_s": metrics.catch_time_s,
        "target_visible_fraction": float(metrics.target_visible_fraction),
        "peak_closing_rate_mps": float(closing_rate.max()),
        "peak_speed_mps": float(np.linalg.norm(v_w, axis=1).max()),
        "wall_s": float(wall),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()

    now = dt.datetime.now()
    group = f"closing_sweep_{now.strftime('%m%d_%H%M%S')}"
    out_root = args.out_root or (
        Path(__file__).resolve().parents[2] / ".runs" / "scenarios"
        / "beihang_paper_sim_closing_sweep"
    )
    group_dir = out_root / group
    group_dir.mkdir(parents=True, exist_ok=True)

    print(f"50 seeds × {len(CELLS)} closing speeds = {args.n_trials * len(CELLS)} trials")
    print(f"output: {group_dir}\n")

    aggregate = {"cells": []}
    for cs, dist, dur in CELLS:
        label = f"cs_{int(cs):02d}"
        cell_dir = group_dir / label
        cell_dir.mkdir(parents=True, exist_ok=True)
        misses, mins, catches = [], [], []
        peak_close, peak_spd, visf = [], [], []
        rows = []
        cell_t0 = time.perf_counter()
        for seed in range(1, args.n_trials + 1):
            try:
                r = run_trial(cs, dist, dur, seed)
                rows.append(r)
                misses.append(r["miss_distance_m"])
                mins.append(r["min_distance_m"])
                catches.append(r["catch_time_s"] is not None)
                peak_close.append(r["peak_closing_rate_mps"])
                peak_spd.append(r["peak_speed_mps"])
                visf.append(r["target_visible_fraction"])
                if seed % 10 == 0:
                    print(f"  [{label}] seed {seed:>2}: miss={r['miss_distance_m']:.3f} "
                          f"catch={r['catch_time_s']}")
            except Exception:
                traceback.print_exc()
                print(f"  [{label}] seed {seed:>2}: FAILED")
        cell_wall = time.perf_counter() - cell_t0
        misses = np.array(misses); mins = np.array(mins)
        cep50 = float(np.nanpercentile(misses, 50))
        cep90 = float(np.nanpercentile(misses, 90))
        cep50_min = float(np.nanpercentile(mins, 50))
        catch_frac = float(np.mean(catches))
        summary = {
            "closing_speed_mps": cs,
            "distance_m": dist,
            "duration_s": dur,
            "n_trials": len(misses),
            "CEP_50_miss_m": cep50,
            "CEP_90_miss_m": cep90,
            "CEP_50_min_dist_m": cep50_min,
            "catch_fraction": catch_frac,
            "mean_peak_closing_mps": float(np.mean(peak_close)),
            "mean_peak_speed_mps": float(np.mean(peak_spd)),
            "mean_target_visible_fraction": float(np.mean(visf)),
            "wall_s": cell_wall,
        }
        with (cell_dir / "cell_summary.json").open("w") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2, sort_keys=True)
        aggregate["cells"].append(summary)
        print(f"  ★ cs={cs:>5}m/s  CEP_50={cep50:.3f}m  "
              f"catch={catch_frac*100:.0f}%  ({cell_wall:.0f}s)\n")

    aggregate["config"] = {
        "k_1": "beihang_paper_sim default (control_core.DEFAULT_GAINS['k_1'] = 0.1)",
        "init_pitch_deg": "beihang_paper_sim default (diagram.INITIAL_PITCH_OFFSET_DEG = 20.0)",
        "noise": "beihang_paper_sim default NoiseConfig()",
        "pixel_noise_std_px": [PIXEL_NOISE_STD_PX, PIXEL_NOISE_STD_PX],
    }
    with (group_dir / "aggregate.json").open("w") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"\naggregate → {group_dir / 'aggregate.json'}")
    print("\n=== SUMMARY ===")
    for c in aggregate["cells"]:
        print(f"  cs={c['closing_speed_mps']:>5}m/s  d={c['distance_m']:>5}m  "
              f"CEP_50={c['CEP_50_miss_m']:.3f}m  "
              f"CEP_90={c['CEP_90_miss_m']:.3f}m  "
              f"catch={c['catch_fraction']*100:>5.1f}%")


if __name__ == "__main__":
    main()
