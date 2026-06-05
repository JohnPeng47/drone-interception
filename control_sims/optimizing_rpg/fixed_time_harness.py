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
from .fixed_time import FixedTimeFeasibilityResult, solve_fixed_time
from .rollout_harness import DEFAULT_BASELINE_SUMMARY


DEFAULT_CATCH_TABLE = Path("scripts/generators/sim_instances/sobol_samples_128.csimin")
DEFAULT_OUTPUT_DIR = Path(".agents/projects/optimizing-rpg-solver/milestone-3-fixed-time-feasibility/artifacts")


@dataclass(frozen=True)
class FixedTimeHarnessConfig:
    scenario_table: Path = DEFAULT_SCENARIO_TABLE
    catch_table: Path = DEFAULT_CATCH_TABLE
    baseline_summary: Path = DEFAULT_BASELINE_SUMMARY
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = 1
    catch_seeds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    label: str = "fixed_time_feasibility"


def run_fixed_time_harness(config: FixedTimeHarnessConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_wall_s = _load_baseline_wall_s(Path(config.baseline_summary))

    scenario_instance = _load_seed_instance(Path(config.scenario_table), int(config.seed))
    benchmark_row = _run_seed(
        config.label,
        scenario_instance,
        baseline_wall_s=baseline_wall_s,
        diagnostic_table=Path(config.scenario_table).stem,
    )

    catch_rows = [
        _run_seed(
            config.label,
            instance,
            baseline_wall_s=baseline_wall_s,
            diagnostic_table=Path(config.catch_table).stem,
        )
        for instance in _load_seed_instances(Path(config.catch_table), config.catch_seeds)
    ]
    _write_csv(output_dir / "fixed_time_rows.csv", [benchmark_row])
    _write_csv(output_dir / "catch_diagnostics.csv", catch_rows)

    scenario_wall_s = float(benchmark_row["scenario_wall_s"])
    delta_vs_baseline_s = float(baseline_wall_s) - scenario_wall_s
    percent_improvement = (delta_vs_baseline_s / float(baseline_wall_s)) * 100.0
    catch_valid = [row for row in catch_rows if row["error"] == ""]
    caught = [bool(row["caught"]) for row in catch_rows]
    catch_acceptance = bool(catch_rows) and all(row["error"] == "" and bool(row["caught"]) for row in catch_rows)
    summary = {
        "config": {
            **asdict(config),
            "scenario_table": str(config.scenario_table),
            "catch_table": str(config.catch_table),
            "baseline_summary": str(config.baseline_summary),
            "output_dir": str(config.output_dir),
            "catch_seeds": list(config.catch_seeds),
        },
        "baseline_wall_s": float(baseline_wall_s),
        "elapsed_wall_s": scenario_wall_s,
        "delta_vs_baseline_s": float(delta_vs_baseline_s),
        "percent_improvement_vs_baseline": float(percent_improvement),
        "fixed_time_wall_s": float(benchmark_row["fixed_time_wall_s"]),
        "plan_acquire_wall_s": float(benchmark_row["plan_acquire_wall_s"]),
        "passed_acceptance": bool(benchmark_row["caught"] and benchmark_row["error"] == "" and catch_acceptance),
        "catch_diagnostics": {
            "table": str(config.catch_table),
            "seeds": list(config.catch_seeds),
            "num_scenarios": int(len(catch_rows)),
            "valid": int(len(catch_valid)),
            "errors": int(len(catch_rows) - len(catch_valid)),
            "catch_fraction": float(np.mean(caught)) if caught else math.nan,
        },
        "artifacts": {
            "fixed_time_rows_csv": "fixed_time_rows.csv",
            "catch_diagnostics_csv": "catch_diagnostics.csv",
            "summary_json": "summary.json",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _run_seed(
    label: str,
    instance,
    *,
    baseline_wall_s: float,
    diagnostic_table: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        plan_started = time.perf_counter()
        selected = solve_portfolio_plan(instance)
        plan_acquire_wall_s = time.perf_counter() - plan_started
        plan = selected.plan
        if plan.motor_speed_commands_rpm is None:
            raise ValueError("selected plan does not include motor speed commands")
        result = solve_fixed_time(
            instance,
            float(plan.total_time_s),
            plan.motor_speed_commands_rpm,
            dynamics_substeps=int(selected.candidate.config.dynamics_substeps),
            control_layout="columns",
        )
        scenario_wall_s = time.perf_counter() - started
        return _row_from_result(
            label,
            result,
            selected_candidate=selected.candidate.name,
            baseline_wall_s=baseline_wall_s,
            plan_acquire_wall_s=plan_acquire_wall_s,
            scenario_wall_s=scenario_wall_s,
            diagnostic_table=diagnostic_table,
            error="",
        )
    except Exception as exc:  # noqa: BLE001
        scenario_wall_s = time.perf_counter() - started
        return {
            "label": str(label),
            "diagnostic_table": str(diagnostic_table),
            "seed": int(instance.seed),
            "selected_candidate": "",
            "baseline_wall_s": float(baseline_wall_s),
            "scenario_wall_s": float(scenario_wall_s),
            "delta_vs_baseline_s": float(baseline_wall_s) - float(scenario_wall_s),
            "percent_improvement_vs_baseline": ((float(baseline_wall_s) - float(scenario_wall_s)) / float(baseline_wall_s)) * 100.0,
            "plan_acquire_wall_s": math.nan,
            "fixed_time_wall_s": math.nan,
            "total_time_s": math.nan,
            "feasible": False,
            "caught": False,
            "failure_reason": "error",
            "replay_min_distance_m": math.inf,
            "replay_final_distance_m": math.inf,
            "replay_wall_s": math.nan,
            "replay_steps": 0,
            "rollout_min_target_distance_m": math.inf,
            "rollout_final_target_distance_m": math.inf,
            "intercept_radius_m": math.nan,
            "error": repr(exc),
        }


def _row_from_result(
    label: str,
    result: FixedTimeFeasibilityResult,
    *,
    selected_candidate: str,
    baseline_wall_s: float,
    plan_acquire_wall_s: float,
    scenario_wall_s: float,
    diagnostic_table: str,
    error: str,
) -> dict[str, Any]:
    delta_vs_baseline_s = float(baseline_wall_s) - float(scenario_wall_s)
    return {
        "label": str(label),
        "diagnostic_table": str(diagnostic_table),
        "seed": int(result.seed),
        "selected_candidate": str(selected_candidate),
        "baseline_wall_s": float(baseline_wall_s),
        "scenario_wall_s": float(scenario_wall_s),
        "delta_vs_baseline_s": float(delta_vs_baseline_s),
        "percent_improvement_vs_baseline": (delta_vs_baseline_s / float(baseline_wall_s)) * 100.0,
        "plan_acquire_wall_s": float(plan_acquire_wall_s),
        "fixed_time_wall_s": float(result.wall_s),
        "total_time_s": float(result.total_time_s),
        "feasible": bool(result.feasible),
        "caught": bool(result.caught),
        "failure_reason": str(result.failure_reason),
        "replay_min_distance_m": float(result.replay_min_distance_m),
        "replay_final_distance_m": float(result.replay_final_distance_m),
        "replay_wall_s": float(result.replay_wall_s),
        "replay_steps": int(result.replay_steps),
        "rollout_min_target_distance_m": float(result.rollout_metrics.min_target_distance_m),
        "rollout_final_target_distance_m": float(result.rollout_metrics.final_target_distance_m),
        "intercept_radius_m": float(result.intercept_radius_m),
        "error": str(error),
    }


def _load_seed_instance(scenario_table: Path, seed: int):
    instances = read_sim_instances(scenario_table)
    for instance in instances:
        if int(instance.seed) == int(seed):
            return instance
    raise ValueError(f"scenario table {scenario_table} is missing seed {seed}")


def _load_seed_instances(scenario_table: Path, seeds: tuple[int, ...]) -> list[Any]:
    by_seed = {int(instance.seed): instance for instance in read_sim_instances(scenario_table)}
    missing = [int(seed) for seed in seeds if int(seed) not in by_seed]
    if missing:
        raise ValueError(f"scenario table {scenario_table} is missing seeds: {missing}")
    return [by_seed[int(seed)] for seed in seeds]


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
