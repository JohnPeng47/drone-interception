from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backends.csim.bindings.types import PursuerInitialState, SimInstance, TargetInitialState
from backends.csim.generator.instance_store import read_sim_instances
from backends.csim.generator.generator import get_config
from control_sims.rpg_time_optimal.adapter import RpgTimeOptimalAdapter, _plan_from_trajectory
from control_sims.rpg_time_optimal.config import RpgTimeOptimalConfig


SCENARIO_TABLE = REPO_ROOT / "scripts/generators/sim_instances/controller_regression_6_distance3x/sobol_samples.csimin"
RUN_DIR = REPO_ROOT / "docs/analysis/control_sims/rpg_time_optimal_controller_regression_6_distance3x_snapshots"
OUT_JSON = Path(__file__).resolve().parent / "cpc_diagnostics.json"


def main() -> None:
    instances = read_sim_instances(SCENARIO_TABLE)
    cfg = RpgTimeOptimalConfig()
    solver_rows = [_solve_diagnostics(instance, cfg) for instance in instances]
    trivial = _solve_diagnostics(_trivial_hover_instance(), cfg)
    execution_rows = _execution_tracking_diagnostics(instances, solver_rows)
    payload = {
        "scenario_table": str(SCENARIO_TABLE.relative_to(REPO_ROOT)),
        "run_dir": str(RUN_DIR.relative_to(REPO_ROOT)),
        "adapter_config": {
            "nodes_per_gate": cfg.nodes_per_gate,
            "velocity_guess_mps": cfg.velocity_guess_mps,
            "ipopt_max_iter": cfg.ipopt_max_iter,
            "position_gain": cfg.position_gain,
            "velocity_gain": cfg.velocity_gain,
            "max_tracking_accel_mps2": cfg.max_tracking_accel_mps2,
        },
        "solver_rows": solver_rows,
        "execution_rows": execution_rows,
        "trivial_hover_5m": trivial,
        "parameter_rows": [_parameter_row(instance) for instance in instances],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_JSON)
    _print_summary(payload)


def _solve_diagnostics(instance: SimInstance, cfg: RpgTimeOptimalConfig) -> dict[str, Any]:
    adapter = RpgTimeOptimalAdapter(cfg)
    modules = adapter._load_modules()
    track = adapter._build_track(instance, modules["Track"])
    quad = adapter._build_quad(instance, modules["Quad"])
    options = {
        "tolerance": max(float(instance.config.intercept_radius_m), 1.0e-3),
        "nodes_per_gate": int(cfg.nodes_per_gate),
        "vel_guess": float(cfg.velocity_guess_mps),
        "solver_options": {
            "ipopt": {"max_iter": int(cfg.ipopt_max_iter), "print_level": int(cfg.ipopt_print_level)},
            "print_time": 0,
        },
    }
    with contextlib.redirect_stdout(io.StringIO()):
        planner = modules["Planner"](quad, track, modules["RungeKutta4"], options)
        planner.setup()
        start = time.perf_counter()
        solution = planner.solve()
        solve_wall_s = time.perf_counter() - start
        trajectory = modules["Trajectory"](solution, NPW=planner.NPW, wp=planner.wp)
    plan = _plan_from_trajectory(instance, trajectory, solve_wall_s)
    target = np.asarray(instance.target_initial.position_w, dtype=float).reshape(3)
    dists = np.linalg.norm(plan.position_w.T - target, axis=1)
    initial_range = float(np.linalg.norm(target - np.asarray(instance.pursuer_initial.position_w, dtype=float)))
    stats = planner.solver.stats()
    return {
        "seed": int(instance.seed),
        "status": str(stats.get("return_status", "")),
        "success": bool(stats.get("success", False)),
        "iter_count": int(stats.get("iter_count", -1)),
        "solve_wall_s": float(solve_wall_s),
        "planned_tf_s": float(plan.total_time_s),
        "initial_range_m": initial_range,
        "planned_min_target_distance_m": float(np.min(dists)),
        "planned_final_target_distance_m": float(dists[-1]),
        "planned_min_target_time_s": float(plan.t_x_s[int(np.argmin(dists))]),
        "planned_terminal_position_w": [float(v) for v in plan.position_w[:, -1]],
        "target_position_w": [float(v) for v in target],
        "planned_max_motor_thrust_n": float(np.max(plan.motor_thrusts_n)),
        "planned_min_motor_thrust_n": float(np.min(plan.motor_thrusts_n)),
        "planned_max_total_thrust_n": float(np.max(np.sum(plan.motor_thrusts_n, axis=0))),
        "planned_max_body_rate_rps": float(np.max(np.linalg.norm(plan.body_rates_b, axis=0))),
    }


def _execution_tracking_diagnostics(
    instances: list[SimInstance],
    solver_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    snapshots = _read_snapshots(RUN_DIR / "snapshots/rpg_time_optimal.csv")
    rows_by_seed = {int(row["seed"]): row for row in solver_rows}
    cfg = RpgTimeOptimalConfig()
    out = []
    for instance in instances:
        seed = int(instance.seed)
        if seed not in snapshots:
            continue
        plan = RpgTimeOptimalAdapter(cfg).solve(instance)
        actual = snapshots[seed]
        errors = []
        for sample in actual:
            t = float(sample["t_s"])
            idx = int(np.searchsorted(plan.t_x_s, np.clip(t, 0.0, plan.total_time_s), side="right") - 1)
            idx = int(np.clip(idx, 0, plan.position_w.shape[1] - 1))
            planned_p = plan.position_w[:, idx]
            actual_p = np.array([sample["pursuer_x_w_m"], sample["pursuer_y_w_m"], sample["pursuer_z_w_m"]], dtype=float)
            errors.append(float(np.linalg.norm(actual_p - planned_p)))
        solver_row = rows_by_seed[seed]
        out.append(
            {
                "seed": seed,
                "samples": len(errors),
                "tracking_error_mean_m": float(np.mean(errors)),
                "tracking_error_max_m": float(np.max(errors)),
                "planned_tf_s": float(solver_row["planned_tf_s"]),
                "sim_end_t_s": float(actual[-1]["t_s"]),
            }
        )
    return out


def _read_snapshots(path: Path) -> dict[int, list[dict[str, float]]]:
    rows: dict[int, list[dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            seed = int(raw["seed"])
            row = {
                "t_s": float(raw["t_s"]),
                "pursuer_x_w_m": float(raw["pursuer_x_w_m"]),
                "pursuer_y_w_m": float(raw["pursuer_y_w_m"]),
                "pursuer_z_w_m": float(raw["pursuer_z_w_m"]),
            }
            rows.setdefault(seed, []).append(row)
    for seed_rows in rows.values():
        seed_rows.sort(key=lambda row: row["t_s"])
    return rows


def _parameter_row(instance: SimInstance) -> dict[str, Any]:
    assert instance.config is not None
    params = instance.config.pursuer
    return {
        "seed": int(instance.seed),
        "sim_mass_kg": float(params.mass_kg),
        "cpc_mass_kg": float(params.mass_kg),
        "sim_collective_max_thrust_n": float(instance.config.max_thrust_n),
        "cpc_per_rotor_max_thrust_n": float(instance.config.max_thrust_n) / 4.0,
        "cpc_collective_max_thrust_n": float(instance.config.max_thrust_n),
        "thrust_to_weight": float(instance.config.max_thrust_n) / float(params.mass_kg * params.gravity_mps2),
        "sim_max_rate_rps": float(instance.config.max_rate_rps),
        "cpc_max_rate_xy_rps": float(instance.config.max_rate_rps),
        "cpc_max_rate_z_rps": float(instance.config.max_rate_rps),
        "sim_motor_tau_s": float(params.motor_tau_s),
        "cpc_motor_tau_s": None,
        "sim_inertia_diag": [float(params.ixx), float(params.iyy), float(params.izz)],
        "sim_arm_len_m": float(params.arm_len_m),
    }


def _trivial_hover_instance() -> SimInstance:
    config = get_config("base")
    return SimInstance(
        seed=999001,
        pursuer_initial=PursuerInitialState(
            position_w=np.array([-5.0, 0.0, 3.0], dtype=float),
            velocity_w=np.zeros(3, dtype=float),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            body_rates_b=np.zeros(3, dtype=float),
            wind_w=np.zeros(3, dtype=float),
        ),
        target_initials=(
            TargetInitialState(
                position_w=np.array([0.0, 0.0, 3.0], dtype=float),
                velocity_w=np.zeros(3, dtype=float),
            ),
        ),
        config=replace(config, intercept_radius_m=0.5),
    )


def _print_summary(payload: dict[str, Any]) -> None:
    print("solver:")
    for row in payload["solver_rows"]:
        print(
            f"seed {row['seed']}: {row['status']}, tf={row['planned_tf_s']:.3f}s, "
            f"planned min={row['planned_min_target_distance_m']:.3f}m at "
            f"{row['planned_min_target_time_s']:.3f}s, final={row['planned_final_target_distance_m']:.3f}m"
        )
    trivial = payload["trivial_hover_5m"]
    print(
        "trivial 5m hover: "
        f"{trivial['status']}, tf={trivial['planned_tf_s']:.3f}s, "
        f"planned min={trivial['planned_min_target_distance_m']:.3f}m"
    )
    print("tracking:")
    for row in payload["execution_rows"]:
        print(
            f"seed {row['seed']}: mean={row['tracking_error_mean_m']:.3f}m, "
            f"max={row['tracking_error_max_m']:.3f}m, "
            f"sim_end={row['sim_end_t_s']:.3f}s, plan_tf={row['planned_tf_s']:.3f}s"
        )


if __name__ == "__main__":
    main()
