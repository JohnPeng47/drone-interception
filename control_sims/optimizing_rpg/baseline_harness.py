from __future__ import annotations

import concurrent.futures
import csv
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backends.csim.generator.instance_store import read_sim_instances
from backends.csim.bindings.types import SimInstance
from control_sims.rpg_time_optimal.portfolio_policy import solve_portfolio_plan


DEFAULT_SCENARIO_TABLE = Path("scripts/generators/sim_instances/sobol_samples_512.csimin")
DEFAULT_OUTPUT_DIR = Path(".agents/projects/optimizing-rpg-solver/milestone-1-baseline-harness/artifacts")


@dataclass(frozen=True)
class BaselineHarnessConfig:
    scenario_table: Path = DEFAULT_SCENARIO_TABLE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seeds: tuple[int, ...] = (1,)
    workers: int = 1
    label: str = "ipopt_portfolio_baseline"


def run_baseline_harness(config: BaselineHarnessConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    instances = _load_instances(Path(config.scenario_table), config.seeds)
    workers = _resolve_workers(config.workers, len(instances))

    started = time.perf_counter()
    if workers == 1:
        results = [_run_one(config.label, instance) for instance in instances]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_one, config.label, instance) for instance in instances]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
    elapsed_wall_s = time.perf_counter() - started

    rows = [row for row, _ in results]
    candidate_rows = [candidate for _, candidates in results for candidate in candidates]
    rows.sort(key=lambda row: int(row["seed"]))
    candidate_rows.sort(key=lambda row: (int(row["seed"]), int(row["candidate_index"])))

    _write_csv(output_dir / "benchmark_rows.csv", rows)
    _write_csv(output_dir / "candidate_rows.csv", candidate_rows)
    summary = _summary(config, rows, candidate_rows, elapsed_wall_s=elapsed_wall_s, workers=workers)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_instances(scenario_table: Path, seeds: Sequence[int]) -> list[SimInstance]:
    requested = tuple(int(seed) for seed in seeds)
    instances = read_sim_instances(scenario_table)
    by_seed = {int(instance.seed): instance for instance in instances}
    missing = [seed for seed in requested if seed not in by_seed]
    if missing:
        raise ValueError(f"scenario table {scenario_table} is missing requested seeds: {missing}")
    return [by_seed[seed] for seed in requested]


def _run_one(label: str, instance: SimInstance) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assert instance.config is not None
    seed = int(instance.seed)
    started = time.perf_counter()
    try:
        selected = solve_portfolio_plan(instance)
        wall_s = time.perf_counter() - started
        score = selected.score
        row = {
            "label": str(label),
            "seed": seed,
            "error": "",
            "selected_candidate": selected.candidate.name,
            "wall_s": wall_s,
            "caught": bool(score.rollout_caught_radius),
            "plan_total_time_s": float(score.plan_total_time_s),
            "min_distance_m": float(score.rollout_min_distance_m),
            "final_distance_m": float(score.rollout_final_distance_m),
            "capture_steps": int(score.rollout_capture_steps),
            "max_consecutive_capture_steps": int(score.rollout_max_consecutive_capture_steps),
            "tracking_error_mean_m": float(score.rollout_position_tracking_error_mean_m),
            "solver_success": bool(score.solver_success),
            "constraint_violation_max": float(score.constraint_violation_max),
        }
        return row, _candidate_rows(label, seed, selected.traces)
    except Exception as exc:  # noqa: BLE001
        wall_s = time.perf_counter() - started
        return {
            "label": str(label),
            "seed": seed,
            "error": repr(exc),
            "selected_candidate": "",
            "wall_s": wall_s,
            "caught": False,
            "plan_total_time_s": math.nan,
            "min_distance_m": math.inf,
            "final_distance_m": math.inf,
            "capture_steps": 0,
            "max_consecutive_capture_steps": 0,
            "tracking_error_mean_m": math.inf,
            "solver_success": False,
            "constraint_violation_max": math.inf,
        }, []


def _candidate_rows(label: str, seed: int, traces: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, trace in enumerate(traces):
        rows.append(
            {
                "label": str(label),
                "seed": int(seed),
                "candidate_index": int(index),
                "candidate": trace.candidate_name,
                "selected": bool(trace.selected),
                "clean": bool(trace.clean),
                "warm_started": bool(trace.warm_started),
                "skipped": bool(trace.skipped),
                "stop_reason": trace.stop_reason,
                "solve_wall_s": float(trace.solve_wall_s),
                "nlp_build_wall_s": float(trace.nlp_build_wall_s),
                "optimizer_wall_s": float(trace.optimizer_wall_s),
                "optimizer_iterations": int(trace.optimizer_iterations),
                "replay_wall_s": float(trace.replay_wall_s),
                "rollout_caught_radius": bool(trace.rollout_caught_radius),
                "rollout_min_distance_m": float(trace.rollout_min_distance_m),
                "rollout_capture_steps": int(trace.rollout_capture_steps),
                "rollout_max_consecutive_capture_steps": int(trace.rollout_max_consecutive_capture_steps),
                "rollout_position_tracking_error_mean_m": float(trace.rollout_position_tracking_error_mean_m),
                "solver_status": trace.solver_status,
                "solver_success": bool(trace.solver_success),
                "constraint_violation_max": float(trace.constraint_violation_max),
                "error": trace.error,
            }
        )
    return rows


def _summary(
    config: BaselineHarnessConfig,
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    elapsed_wall_s: float,
    workers: int,
) -> dict[str, Any]:
    valid = [row for row in rows if not row["error"]]
    caught = np.array([bool(row["caught"]) for row in rows], dtype=bool)
    wall_s = _finite_array(row["wall_s"] for row in valid)
    seed_one_wall_s = _seed_wall_s(rows, seed=1)
    baseline_wall_s = seed_one_wall_s
    optimizer_s = _finite_array(row["optimizer_wall_s"] for row in candidate_rows if not row["skipped"])
    build_s = _finite_array(row["nlp_build_wall_s"] for row in candidate_rows if not row["skipped"])
    replay_s = _finite_array(row["replay_wall_s"] for row in candidate_rows if not row["skipped"])
    delta_vs_baseline_s = 0.0 if np.isfinite(baseline_wall_s) else math.nan
    percent_improvement_vs_baseline = 0.0 if np.isfinite(baseline_wall_s) else math.nan
    return {
        "config": {
            **asdict(config),
            "scenario_table": str(config.scenario_table),
            "output_dir": str(config.output_dir),
            "seeds": list(config.seeds),
        },
        "workers": int(workers),
        "num_scenarios": int(len(rows)),
        "elapsed_wall_s": float(elapsed_wall_s),
        "valid": int(len(valid)),
        "errors": int(len(rows) - len(valid)),
        "catch_fraction": float(np.mean(caught)) if caught.size else math.nan,
        "wall_s_p50": _percentile(wall_s, 50),
        "wall_s_p90": _percentile(wall_s, 90),
        "single_scenario_reference_wall_s": baseline_wall_s,
        "single_scenario_reference_seed": 1 if np.isfinite(baseline_wall_s) else None,
        "baseline_wall_s": baseline_wall_s,
        "delta_vs_baseline_s": delta_vs_baseline_s,
        "percent_improvement_vs_baseline": percent_improvement_vs_baseline,
        "candidate_nlp_build_wall_s_sum": float(np.sum(build_s)) if build_s.size else 0.0,
        "candidate_optimizer_wall_s_sum": float(np.sum(optimizer_s)) if optimizer_s.size else 0.0,
        "candidate_replay_wall_s_sum": float(np.sum(replay_s)) if replay_s.size else 0.0,
        "passed_acceptance": _passed_acceptance(rows) and bool(np.isfinite(baseline_wall_s)),
        "artifacts": {
            "benchmark_rows_csv": "benchmark_rows.csv",
            "candidate_rows_csv": "candidate_rows.csv",
            "summary_json": "summary.json",
        },
    }


def _passed_acceptance(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    return all(not row["error"] and bool(row["caught"]) for row in rows)


def _seed_wall_s(rows: list[dict[str, Any]], *, seed: int) -> float:
    for row in rows:
        if int(row["seed"]) == int(seed) and not row["error"]:
            return float(row["wall_s"])
    return math.nan


def _resolve_workers(workers: int, task_count: int) -> int:
    if task_count <= 0:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(int(workers), int(task_count), cpu_count))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _finite_array(values) -> np.ndarray:
    array = np.array(list(values), dtype=float)
    return array[np.isfinite(array)]


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else math.nan
