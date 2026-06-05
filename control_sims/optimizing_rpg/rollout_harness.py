from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backends.csim.generator.instance_store import read_sim_instances
from control_sims.rpg_time_optimal.portfolio_policy import solve_portfolio_plan

from .baseline_harness import DEFAULT_SCENARIO_TABLE
from .rollout import compare_to_reference_plan, replay_motor_commands_in_simengine, rollout_motor_commands


DEFAULT_BASELINE_SUMMARY = Path(".agents/projects/optimizing-rpg-solver/milestone-1-baseline-harness/artifacts/summary.json")
DEFAULT_OUTPUT_DIR = Path(".agents/projects/optimizing-rpg-solver/milestone-2-numeric-rollout-core/artifacts")


@dataclass(frozen=True)
class RolloutHarnessConfig:
    scenario_table: Path = DEFAULT_SCENARIO_TABLE
    baseline_summary: Path = DEFAULT_BASELINE_SUMMARY
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = 1
    repeats: int = 100
    label: str = "numeric_rollout_core"


def run_rollout_harness(config: RolloutHarnessConfig) -> dict[str, Any]:
    harness_started = time.perf_counter()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_wall_s = _load_baseline_wall_s(Path(config.baseline_summary))
    instance = _load_seed_instance(Path(config.scenario_table), int(config.seed))

    plan_started = time.perf_counter()
    selected = solve_portfolio_plan(instance)
    plan_acquire_wall_s = time.perf_counter() - plan_started
    plan = selected.plan
    if plan.motor_speed_commands_rpm is None:
        raise ValueError("selected plan does not include motor speed command trajectory")
    controls = np.asarray(plan.motor_speed_commands_rpm, dtype=np.float64)
    dynamics_substeps = int(selected.candidate.config.dynamics_substeps)

    trajectory = rollout_motor_commands(
        instance,
        controls,
        float(plan.total_time_s),
        dynamics_substeps=dynamics_substeps,
        control_layout="columns",
    )
    compare_metrics = compare_to_reference_plan(instance, trajectory, plan)
    replay_metrics = replay_motor_commands_in_simengine(instance, controls, float(plan.total_time_s), control_layout="columns")
    scenario_wall_s = time.perf_counter() - harness_started

    repeat_count = max(1, int(config.repeats))
    repeat_started = time.perf_counter()
    for _ in range(repeat_count):
        rollout_motor_commands(
            instance,
            controls,
            float(plan.total_time_s),
            dynamics_substeps=dynamics_substeps,
            control_layout="columns",
        )
    repeated_rollout_wall_s = time.perf_counter() - repeat_started
    mean_rollout_wall_s = repeated_rollout_wall_s / float(repeat_count)

    rollout_delta_vs_baseline_s = float(baseline_wall_s) - float(mean_rollout_wall_s)
    rollout_percent_improvement = (rollout_delta_vs_baseline_s / float(baseline_wall_s)) * 100.0
    scenario_delta_vs_baseline_s = float(baseline_wall_s) - float(scenario_wall_s)
    scenario_percent_improvement = (scenario_delta_vs_baseline_s / float(baseline_wall_s)) * 100.0
    row = {
        "label": str(config.label),
        "seed": int(config.seed),
        "selected_candidate": selected.candidate.name,
        "baseline_wall_s": float(baseline_wall_s),
        "plan_acquire_wall_s": float(plan_acquire_wall_s),
        "single_rollout_wall_s": float(trajectory.rollout_wall_s),
        "repeat_count": int(repeat_count),
        "mean_rollout_wall_s": float(mean_rollout_wall_s),
        "rollout_delta_vs_baseline_s": float(rollout_delta_vs_baseline_s),
        "rollout_percent_improvement_vs_baseline": float(rollout_percent_improvement),
        "scenario_wall_s": float(scenario_wall_s),
        "scenario_delta_vs_baseline_s": float(scenario_delta_vs_baseline_s),
        "scenario_percent_improvement_vs_baseline": float(scenario_percent_improvement),
        "rollout_nodes": int(trajectory.controls.shape[0]),
        "rollout_dt_s": float(trajectory.dt_s),
        "terminal_position_error_m": float(compare_metrics.terminal_position_error_m),
        "position_error_mean_m": float(compare_metrics.position_error_mean_m),
        "position_error_max_m": float(compare_metrics.position_error_max_m),
        "rollout_min_target_distance_m": float(compare_metrics.min_target_distance_m),
        "rollout_final_target_distance_m": float(compare_metrics.final_target_distance_m),
        "rpm_min": float(compare_metrics.rpm_min),
        "rpm_max": float(compare_metrics.rpm_max),
        "body_rate_abs_max_rps": float(compare_metrics.body_rate_abs_max_rps),
        "altitude_min_m": float(compare_metrics.altitude_min_m),
        "altitude_max_m": float(compare_metrics.altitude_max_m),
        "simengine_replay_wall_s": float(replay_metrics.replay_wall_s),
        "simengine_replay_steps": int(replay_metrics.steps),
        "simengine_replay_caught": bool(replay_metrics.caught),
        "simengine_replay_min_distance_m": float(replay_metrics.min_target_distance_m),
        "simengine_replay_final_distance_m": float(replay_metrics.final_target_distance_m),
    }
    summary = {
        "config": {
            **asdict(config),
            "scenario_table": str(config.scenario_table),
            "baseline_summary": str(config.baseline_summary),
            "output_dir": str(config.output_dir),
        },
        "baseline_wall_s": float(baseline_wall_s),
        "elapsed_wall_s": float(scenario_wall_s),
        "delta_vs_baseline_s": float(scenario_delta_vs_baseline_s),
        "percent_improvement_vs_baseline": float(scenario_percent_improvement),
        "scenario_wall_s": float(scenario_wall_s),
        "scenario_delta_vs_baseline_s": float(scenario_delta_vs_baseline_s),
        "scenario_percent_improvement_vs_baseline": float(scenario_percent_improvement),
        "rollout_delta_vs_baseline_s": float(rollout_delta_vs_baseline_s),
        "rollout_percent_improvement_vs_baseline": float(rollout_percent_improvement),
        "plan_acquire_wall_s": float(plan_acquire_wall_s),
        "single_rollout_wall_s": float(trajectory.rollout_wall_s),
        "mean_rollout_wall_s": float(mean_rollout_wall_s),
        "repeat_count": int(repeat_count),
        "passed_acceptance": bool(_passed_acceptance(row)),
        "artifacts": {
            "rollout_rows_csv": "rollout_rows.csv",
            "summary_json": "summary.json",
        },
    }
    _write_csv(output_dir / "rollout_rows.csv", [row])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _passed_acceptance(row: dict[str, Any]) -> bool:
    return bool(
        bool(row["simengine_replay_caught"])
        and np.isfinite(float(row["mean_rollout_wall_s"]))
        and float(row["mean_rollout_wall_s"]) < 0.1
        and np.isfinite(float(row["position_error_max_m"]))
        and float(row["position_error_max_m"]) <= 1.0e-6
        and float(row["terminal_position_error_m"]) <= 1.0e-6
    )


def _load_seed_instance(scenario_table: Path, seed: int):
    instances = read_sim_instances(scenario_table)
    for instance in instances:
        if int(instance.seed) == int(seed):
            return instance
    raise ValueError(f"scenario table {scenario_table} is missing seed {seed}")


def _load_baseline_wall_s(path: Path) -> float:
    summary = json.loads(path.read_text(encoding="utf-8"))
    baseline_wall_s = float(summary["baseline_wall_s"])
    if not math.isfinite(baseline_wall_s) or baseline_wall_s <= 0.0:
        raise ValueError(f"invalid baseline_wall_s in {path}: {baseline_wall_s}")
    return baseline_wall_s


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
