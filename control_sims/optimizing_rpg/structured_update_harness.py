from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backends.csim.generator.instance_store import read_sim_instances
from control_sims.rpg_time_optimal.portfolio_policy import solve_portfolio_plan

from .baseline_harness import DEFAULT_SCENARIO_TABLE
from .rollout_harness import DEFAULT_BASELINE_SUMMARY
from .structured_update import StructuredUpdateConfig, StructuredUpdateResult, run_structured_update


DEFAULT_OUTPUT_DIR = Path(".agents/projects/optimizing-rpg-solver/milestone-4-structured-trajectory-update/artifacts")


@dataclass(frozen=True)
class StructuredUpdateHarnessConfig:
    scenario_table: Path = DEFAULT_SCENARIO_TABLE
    baseline_summary: Path = DEFAULT_BASELINE_SUMMARY
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = 1
    active_window_nodes: int = 8
    label: str = "structured_trajectory_update"


def run_structured_update_harness(config: StructuredUpdateHarnessConfig) -> dict[str, Any]:
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
        raise ValueError("selected plan does not include motor speed commands")

    update_config = StructuredUpdateConfig(active_window_nodes=int(config.active_window_nodes))
    result = run_structured_update(
        instance,
        plan.motor_speed_commands_rpm,
        float(plan.total_time_s),
        dynamics_substeps=int(selected.candidate.config.dynamics_substeps),
        control_layout="columns",
        config=update_config,
    )
    scenario_wall_s = time.perf_counter() - harness_started
    row = _row_from_result(
        config,
        result,
        selected_candidate=selected.candidate.name,
        baseline_wall_s=baseline_wall_s,
        plan_acquire_wall_s=plan_acquire_wall_s,
        scenario_wall_s=scenario_wall_s,
    )
    derivative_row = {
        "label": str(config.label),
        "seed": int(config.seed),
        "active_variables": int(result.active_variables),
        "gradient_norm": float(result.gradient_norm),
        "gradient_abs_max": float(result.gradient_abs_max),
        "direction_derivative_predicted": float(result.direction_derivative_predicted),
        "direction_derivative_actual": float(result.direction_derivative_actual),
        "direction_derivative_abs_error": float(result.direction_derivative_abs_error),
        "direction_derivative_relative_error": float(result.direction_derivative_relative_error),
    }
    _write_csv(output_dir / "structured_update_rows.csv", [row])
    _write_csv(output_dir / "derivative_validation_rows.csv", [derivative_row])

    delta_vs_baseline_s = float(baseline_wall_s) - float(scenario_wall_s)
    percent_improvement = (delta_vs_baseline_s / float(baseline_wall_s)) * 100.0
    summary = {
        "config": {
            **asdict(config),
            "scenario_table": str(config.scenario_table),
            "baseline_summary": str(config.baseline_summary),
            "output_dir": str(config.output_dir),
        },
        "update_config": asdict(update_config),
        "baseline_wall_s": float(baseline_wall_s),
        "elapsed_wall_s": float(scenario_wall_s),
        "delta_vs_baseline_s": float(delta_vs_baseline_s),
        "percent_improvement_vs_baseline": float(percent_improvement),
        "plan_acquire_wall_s": float(plan_acquire_wall_s),
        "structured_update_wall_s": float(result.wall_s),
        "derivative_wall_s": float(result.derivative_wall_s),
        "line_search_wall_s": float(result.line_search_wall_s),
        "replay_wall_s": float(result.replay_wall_s),
        "initial_cost": float(result.initial_cost),
        "accepted_cost": float(result.accepted_cost),
        "accepted_alpha": float(result.accepted_alpha),
        "replay_caught": bool(result.replay_caught),
        "replay_min_distance_m": float(result.replay_min_distance_m),
        "replay_final_distance_m": float(result.replay_final_distance_m),
        "direction_derivative_abs_error": float(result.direction_derivative_abs_error),
        "direction_derivative_relative_error": float(result.direction_derivative_relative_error),
        "passed_acceptance": bool(_passed_acceptance(row)),
        "artifacts": {
            "structured_update_rows_csv": "structured_update_rows.csv",
            "derivative_validation_rows_csv": "derivative_validation_rows.csv",
            "summary_json": "summary.json",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _row_from_result(
    config: StructuredUpdateHarnessConfig,
    result: StructuredUpdateResult,
    *,
    selected_candidate: str,
    baseline_wall_s: float,
    plan_acquire_wall_s: float,
    scenario_wall_s: float,
) -> dict[str, Any]:
    delta_vs_baseline_s = float(baseline_wall_s) - float(scenario_wall_s)
    return {
        "label": str(config.label),
        "seed": int(result.seed),
        "selected_candidate": str(selected_candidate),
        "baseline_wall_s": float(baseline_wall_s),
        "scenario_wall_s": float(scenario_wall_s),
        "delta_vs_baseline_s": float(delta_vs_baseline_s),
        "percent_improvement_vs_baseline": (delta_vs_baseline_s / float(baseline_wall_s)) * 100.0,
        "plan_acquire_wall_s": float(plan_acquire_wall_s),
        "structured_update_wall_s": float(result.wall_s),
        "rollout_wall_s": float(result.rollout_wall_s),
        "derivative_wall_s": float(result.derivative_wall_s),
        "line_search_wall_s": float(result.line_search_wall_s),
        "replay_wall_s": float(result.replay_wall_s),
        "replay_steps": int(result.replay_steps),
        "total_time_s": float(result.total_time_s),
        "initial_cost": float(result.initial_cost),
        "accepted_cost": float(result.accepted_cost),
        "cost_delta": float(result.initial_cost) - float(result.accepted_cost),
        "initial_min_distance_m": float(result.initial_min_distance_m),
        "accepted_min_distance_m": float(result.accepted_min_distance_m),
        "initial_final_distance_m": float(result.initial_final_distance_m),
        "accepted_final_distance_m": float(result.accepted_final_distance_m),
        "accepted_alpha": float(result.accepted_alpha),
        "gradient_norm": float(result.gradient_norm),
        "gradient_abs_max": float(result.gradient_abs_max),
        "active_variables": int(result.active_variables),
        "direction_derivative_abs_error": float(result.direction_derivative_abs_error),
        "direction_derivative_relative_error": float(result.direction_derivative_relative_error),
        "replay_caught": bool(result.replay_caught),
        "replay_min_distance_m": float(result.replay_min_distance_m),
        "replay_final_distance_m": float(result.replay_final_distance_m),
    }


def _passed_acceptance(row: dict[str, Any]) -> bool:
    cost_delta = float(row["cost_delta"])
    return bool(
        bool(row["replay_caught"])
        and math.isfinite(float(row["structured_update_wall_s"]))
        and math.isfinite(float(row["initial_cost"]))
        and math.isfinite(float(row["accepted_cost"]))
        and float(row["accepted_cost"]) <= float(row["initial_cost"]) + 1.0e-12
        and math.isfinite(cost_delta)
        and cost_delta > 1.0e-9
        and math.isfinite(float(row["accepted_alpha"]))
        and float(row["accepted_alpha"]) > 0.0
        and math.isfinite(float(row["gradient_norm"]))
        and float(row["gradient_norm"]) > 0.0
        and math.isfinite(float(row["gradient_abs_max"]))
        and float(row["gradient_abs_max"]) > 0.0
        and math.isfinite(float(row["direction_derivative_relative_error"]))
        and float(row["direction_derivative_relative_error"]) <= 1.0e-5
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
