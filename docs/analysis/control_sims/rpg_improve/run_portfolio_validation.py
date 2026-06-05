from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent / "portfolio_validation"
SCENARIO_TABLE = REPO_ROOT / "scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backends.csim.generator.instance_store import read_sim_instances
from control_sims.rpg_time_optimal.portfolio_policy import solve_portfolio_plan


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    instances = read_sim_instances(args.scenario_table)
    if args.seeds:
        requested = {int(seed) for seed in args.seeds.split(",") if seed.strip()}
        instances = [instance for instance in instances if int(instance.seed) in requested]
    workers = _resolve_workers(args.workers, len(instances))
    started = time.perf_counter()
    rows, candidate_rows = _run_instances(args.scenario_table, [int(instance.seed) for instance in instances], workers)
    rows.sort(key=lambda row: int(row["seed"]))
    candidate_rows.sort(key=lambda row: (int(row["seed"]), int(row["candidate_index"])))
    elapsed_wall_s = time.perf_counter() - started
    _write_csv(output_dir / "portfolio_validation.csv", rows)
    _write_csv(output_dir / "portfolio_candidates.csv", candidate_rows)
    caught = [row for row in rows if row["error"] == "" and bool(row["rollout_caught_radius"])]
    summary = {
        "scenario_table": str(args.scenario_table.relative_to(REPO_ROOT) if args.scenario_table.is_relative_to(REPO_ROOT) else args.scenario_table),
        "output_dir": str(output_dir.relative_to(REPO_ROOT) if output_dir.is_relative_to(REPO_ROOT) else output_dir),
        "num_scenarios": len(rows),
        "workers": workers,
        "elapsed_wall_s": elapsed_wall_s,
        "catch_count": len(caught),
        "catch_fraction": len(caught) / max(len(rows), 1),
        "max_min_distance_m": max((float(row["rollout_min_distance_m"]) for row in caught), default=None),
        "candidate_solve_wall_s_sum": sum(float(row["solve_wall_s"]) for row in candidate_rows if row["error"] == ""),
        "candidate_optimizer_wall_s_sum": sum(float(row["optimizer_wall_s"]) for row in candidate_rows if row["error"] == ""),
        "candidate_replay_wall_s_sum": sum(float(row["replay_wall_s"]) for row in candidate_rows if row["error"] == ""),
        "artifacts": {
            "portfolio_validation_csv": "portfolio_validation.csv",
            "portfolio_candidates_csv": "portfolio_candidates.csv",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate production RPG portfolio plan selection.")
    parser.add_argument("--scenario-table", type=Path, default=SCENARIO_TABLE)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def _run_instances(scenario_table: Path, seeds: list[int], workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = [(scenario_table, seed) for seed in seeds]
    if workers == 1:
        results = [_run_one(task) for task in tasks]
        return _split_results(results)
    results = []
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_one, task) for task in tasks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            print(f"completed {index}/{len(tasks)} in {time.perf_counter() - started:.1f}s", flush=True)
    return _split_results(results)


def _split_results(results: list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [row for row, _ in results]
    candidate_rows = [candidate_row for _, candidate_rows in results for candidate_row in candidate_rows]
    return rows, candidate_rows


def _run_one(task: tuple[Path, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenario_table, seed = task
    instances = read_sim_instances(scenario_table)
    instance_by_seed = {int(instance.seed): instance for instance in instances}
    instance = instance_by_seed[int(seed)]
    assert instance.config is not None
    intercept_radius_m = float(instance.config.intercept_radius_m)
    dt_s = float(instance.config.options.backend_dt) * max(1, int(instance.config.options.action_substeps))
    started = time.perf_counter()
    try:
        selected = solve_portfolio_plan(instance)
        score = selected.score
        row = {
            "seed": int(seed),
            "error": "",
            "candidate": selected.candidate.name,
            "intercept_radius_m": intercept_radius_m,
            "rollout_caught_radius": bool(score.rollout_caught_radius),
            "rollout_min_distance_m": float(score.rollout_min_distance_m),
            "replay_margin_m": intercept_radius_m - float(score.rollout_min_distance_m),
            "rollout_capture_steps": int(score.rollout_capture_steps),
            "rollout_max_consecutive_capture_steps": int(score.rollout_max_consecutive_capture_steps),
            "rollout_max_consecutive_capture_duration_s": (
                int(score.rollout_max_consecutive_capture_steps) * dt_s
            ),
            "rollout_position_tracking_error_mean_m": float(score.rollout_position_tracking_error_mean_m),
            "replay_wall_s": float(score.replay_wall_s),
            "plan_total_time_s": float(score.plan_total_time_s),
            "solver_success": bool(score.solver_success),
            "constraint_violation_max": float(score.constraint_violation_max),
            "task_wall_s": time.perf_counter() - started,
        }
        return row, _candidate_rows(int(seed), selected.traces, intercept_radius_m=intercept_radius_m, dt_s=dt_s)
    except Exception as exc:  # noqa: BLE001
        return {
            "seed": int(seed),
            "error": repr(exc),
            "candidate": "",
            "intercept_radius_m": intercept_radius_m,
            "rollout_caught_radius": False,
            "rollout_min_distance_m": float("inf"),
            "replay_margin_m": float("-inf"),
            "rollout_capture_steps": 0,
            "rollout_max_consecutive_capture_steps": 0,
            "rollout_max_consecutive_capture_duration_s": 0.0,
            "rollout_position_tracking_error_mean_m": float("inf"),
            "replay_wall_s": 0.0,
            "plan_total_time_s": float("inf"),
            "solver_success": False,
            "constraint_violation_max": float("inf"),
            "task_wall_s": time.perf_counter() - started,
        }, []


def _candidate_rows(seed: int, traces: Any, *, intercept_radius_m: float, dt_s: float) -> list[dict[str, Any]]:
    rows = []
    for index, trace in enumerate(traces):
        rows.append(
            {
                "seed": int(seed),
                "candidate_index": int(index),
                "candidate": trace.candidate_name,
                "selected": bool(trace.selected),
                "clean": bool(trace.clean),
                "warm_started": bool(trace.warm_started),
                "skipped": bool(trace.skipped),
                "stop_reason": trace.stop_reason,
                "rollout_caught_radius": bool(trace.rollout_caught_radius),
                "rollout_min_distance_m": float(trace.rollout_min_distance_m),
                "replay_margin_m": (
                    float("nan")
                    if trace.skipped
                    else float(intercept_radius_m) - float(trace.rollout_min_distance_m)
                ),
                "rollout_capture_steps": int(trace.rollout_capture_steps),
                "rollout_max_consecutive_capture_steps": int(trace.rollout_max_consecutive_capture_steps),
                "rollout_max_consecutive_capture_duration_s": (
                    int(trace.rollout_max_consecutive_capture_steps) * float(dt_s)
                ),
                "rollout_position_tracking_error_mean_m": float(trace.rollout_position_tracking_error_mean_m),
                "plan_total_time_s": float(trace.plan_total_time_s),
                "solver_status": trace.solver_status,
                "solver_success": bool(trace.solver_success),
                "constraint_violation_max": float(trace.constraint_violation_max),
                "solve_wall_s": float(trace.solve_wall_s),
                "nlp_build_wall_s": float(trace.nlp_build_wall_s),
                "optimizer_wall_s": float(trace.optimizer_wall_s),
                "optimizer_iterations": int(trace.optimizer_iterations),
                "replay_wall_s": float(trace.replay_wall_s),
                "error": trace.error,
            }
        )
    return rows


def _resolve_workers(workers: int | None, task_count: int) -> int:
    if workers is not None:
        return max(1, int(workers))
    cpu_count = os.cpu_count() or 1
    return max(1, min(int(task_count), cpu_count - 1 if cpu_count > 1 else 1))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
