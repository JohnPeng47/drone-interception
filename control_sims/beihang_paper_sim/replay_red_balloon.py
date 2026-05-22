"""Replay fixed red-balloon control-sim scenarios and compare metrics."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from ._paths import ensure_paths
except ImportError:  # Support direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from control_sims.beihang_paper_sim._paths import ensure_paths


ensure_paths()


from pydrake.systems.analysis import Simulator  # noqa: E402

from intercept_sim.analysis import compute_metrics  # noqa: E402

from control_sims.beihang_paper_sim.config import ExperimentConfig  # noqa: E402
from control_sims.beihang_paper_sim.diagram import build_diagram_from_config  # noqa: E402
from control_sims.beihang_paper_sim.noise_config import NoiseConfig  # noqa: E402
from control_sims.beihang_paper_sim.sim_generator import RedBalloonSimGenerator  # noqa: E402


DEFAULT_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "replay_paper_sim_0512_190210.yaml"
)


def _resolve_path(path: str | Path, *, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = base / p
    if candidate.exists():
        return candidate
    return Path.cwd() / p


def _expected_for_seed(expected: dict[Any, Any], seed: int) -> dict[str, Any]:
    for key in (seed, str(seed), f"run_{seed:03d}"):
        if key in expected:
            return dict(expected[key])
    return {}


def _build_trial_config(replay: dict[str, Any], replay_path: Path, seed: int) -> tuple[ExperimentConfig, dict]:
    base_scenario = _resolve_path(
        replay["base_scenario"],
        base=replay_path.parent,
    )
    instance = RedBalloonSimGenerator.from_path(base_scenario).sample(seed=seed)
    raw = copy.deepcopy(instance.raw_config)

    if "sim" in replay:
        raw["sim"] = {**raw["sim"], **dict(replay["sim"])}
    if "vehicle" in replay:
        raw["vehicle"] = {**raw["vehicle"], **dict(replay["vehicle"])}
    if "perception" in replay:
        raw["perception"] = {**raw["perception"], **dict(replay["perception"])}
    if "controller" in replay:
        controller = dict(raw.get("controller", {}))
        replay_controller = dict(replay["controller"])
        gains = {
            **dict(controller.get("gains", {})),
            **dict(replay_controller.pop("gains", {})),
        }
        controller.update(replay_controller)
        if gains:
            controller["gains"] = gains
        raw["controller"] = controller

    return ExperimentConfig(raw=raw, path=instance.path), raw.get("controller", {}).get("gains", {})


def run_trial(replay: dict[str, Any], replay_path: Path, seed: int) -> dict[str, Any]:
    cfg, controller_gains = _build_trial_config(replay, replay_path, seed)
    noise = replay.get("noise", {})
    nc = NoiseConfig(rng_seed=seed, **dict(noise))

    diagram, logger = build_diagram_from_config(
        cfg,
        controller_gains=controller_gains,
        noise_config=nc,
    )
    sim = Simulator(diagram)
    sim.Initialize()

    t0 = time.perf_counter()
    sim.AdvanceTo(cfg.duration_s)
    wall_s = time.perf_counter() - t0

    num_steps = int(math.ceil(cfg.duration_s / cfg.dt))
    log = logger.get_log()[:num_steps]
    metrics = compute_metrics(log, catch_radius_m=cfg.catch_radius_m)
    metrics_dict = metrics.to_dict()
    return {
        "seed": int(seed),
        "backend": str(cfg.raw.get("sim", {}).get("backend", "rotorpy")),
        "miss_distance_m": float(metrics_dict["miss_distance_m"]),
        "min_distance_m": float(metrics_dict["min_distance_m"]),
        "final_distance_m": float(metrics_dict["final_distance_m"]),
        "catch_time_s": metrics_dict.get("catch_time_s"),
        "target_visible_fraction": float(metrics_dict["target_visible_fraction"]),
        "wall_s": float(wall_s),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--miss-atol", type=float, default=1e-5)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        replay = yaml.safe_load(handle)

    rows = []
    expected = dict(replay.get("expected", {}))
    print(f"config: {args.config}")
    print(f"backend: {replay.get('sim', {}).get('backend', 'rotorpy')}")
    print()
    print("seed | expected miss | replay miss | diff | final | status")
    print("-----+---------------+-------------+------+-------+--------")

    ok = True
    for seed in replay["seeds"]:
        seed = int(seed)
        row = run_trial(replay, args.config, seed)
        exp = _expected_for_seed(expected, seed)
        expected_miss = exp.get("miss_distance_m")
        if expected_miss is None:
            diff = None
            status = "no expected"
        else:
            diff = abs(float(row["miss_distance_m"]) - float(expected_miss))
            status = "PASS" if diff <= args.miss_atol else "FAIL"
            ok = ok and status == "PASS"
        row["expected_miss_distance_m"] = None if expected_miss is None else float(expected_miss)
        row["miss_diff_m"] = diff
        row["status"] = status
        rows.append(row)
        expected_s = "n/a" if expected_miss is None else f"{float(expected_miss):.12f}"
        diff_s = "n/a" if diff is None else f"{diff:.3g}"
        print(
            f"{seed:>4} | {expected_s:>13} | "
            f"{row['miss_distance_m']:>11.12f} | {diff_s:>4} | "
            f"{row['final_distance_m']:>5.3f} | {status}"
        )

    result = {
        "config": str(args.config),
        "miss_atol": args.miss_atol,
        "rows": rows,
        "pass": ok,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
