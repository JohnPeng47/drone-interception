from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from backends.csim.generator.instance_store import read_sim_instances
from backends.csim.runner import SimControlPolicy
from control_sims.logging import (
    LoggingConfig,
    snapshot_fieldnames,
    snapshot_logging_metadata,
    snapshot_rows_from_step,
)
from utils.logging import RunsDirLogger


PolicyFactory = Callable[[], SimControlPolicy]


class ControlSimRunsRunner:
    """Own run directory naming and artifact writes for one control sim flavor."""

    def __init__(self, sim_name: str, runs_logger: RunsDirLogger):
        self.sim_name = str(sim_name)
        self.runs_logger = runs_logger

    def create_run_dir(self, *, suffix: str | None = None, out_dir: Path | None = None) -> Path:
        if out_dir is not None:
            run_dir = Path(out_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        return self.runs_logger.create_run_dir(suffix)

    def write_trials(self, run_dir: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> Path:
        return self.runs_logger.write_csv(run_dir, "trials.csv", rows, fieldnames)

    def write_snapshots(
        self,
        run_dir: Path,
        rows: list[Mapping[str, Any]],
        config: LoggingConfig,
    ) -> Path:
        snapshot_path = self.runs_logger.write_csv(
            run_dir,
            Path("snapshots") / f"{self.sim_name}.csv",
            rows,
            snapshot_fieldnames(config),
            extrasaction="ignore",
        )
        self.runs_logger.write_json(
            run_dir,
            Path("snapshots") / "logging_config.json",
            snapshot_logging_metadata(self.sim_name, config),
        )
        return snapshot_path

    def write_summary(self, run_dir: Path, summary: Mapping[str, Any]) -> Path:
        return self.runs_logger.write_json(run_dir, "summary.json", summary)


def run_policy_cli(
    *,
    sim_name: str,
    description: str,
    policy_factory: PolicyFactory,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--scenario-table", type=Path, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--max-envs",
        type=int,
        default=1,
        help="Number of C SimEngine slots to use inside each runner process.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of scenario workers. Defaults to min(samples, CPU count - 1). Use 1 for serial.",
    )
    parser.add_argument("--log-snapshots", action="store_true")
    parser.add_argument(
        "--snapshot-log-rate",
        type=int,
        default=100,
        help="Write one snapshot row every N sim ticks when --log-snapshots is enabled.",
    )
    args = parser.parse_args()

    if int(args.snapshot_log_rate) <= 0:
        raise ValueError("--snapshot-log-rate must be positive")
    if int(args.max_envs) <= 0:
        raise ValueError("--max-envs must be positive")

    runner = ControlSimRunsRunner(sim_name, RunsDirLogger(sim_name))
    run_dir = runner.create_run_dir(suffix=args.run_suffix, out_dir=args.out_dir)

    tasks, source, duration_s, dt_s = _load_tasks(args)
    workers = _resolve_workers(args.workers, len(tasks))
    print(f"running {len(tasks)} scenarios with {workers} worker(s) for {sim_name}", flush=True)

    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    start = time.perf_counter()
    chunk_size = int(args.max_envs)
    task_chunks = _task_chunks(tasks, chunk_size)
    if workers == 1:
        completed_count = 0
        for chunk in task_chunks:
            try:
                result = _run_instances(
                    chunk,
                    sim_name,
                    policy_factory,
                    max_envs=int(args.max_envs),
                    log_snapshots=bool(args.log_snapshots),
                    snapshot_log_rate=int(args.snapshot_log_rate),
                )
                rows.extend(result["rows"])
                snapshots.extend(result["snapshots"])
            except Exception as exc:  # noqa: BLE001
                rows.extend(_error_rows_for_chunk(chunk, sim_name, exc))
            completed_count += len(chunk)
            _print_progress(completed_count, len(tasks), int(args.progress_every), start)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_instances,
                    chunk,
                    sim_name,
                    policy_factory,
                    max_envs=int(args.max_envs),
                    log_snapshots=bool(args.log_snapshots),
                    snapshot_log_rate=int(args.snapshot_log_rate),
                ): chunk
                for chunk in task_chunks
            }
            completed_count = 0
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                chunk = futures[future]
                try:
                    result = future.result()
                    rows.extend(result["rows"])
                    snapshots.extend(result["snapshots"])
                except Exception as exc:  # noqa: BLE001
                    rows.extend(_error_rows_for_chunk(chunk, sim_name, exc))
                completed_count += len(chunk)
                _print_progress(completed_count, len(tasks), int(args.progress_every), start)

    rows.sort(key=lambda row: int(row["seed"]))
    snapshots.sort(key=lambda row: (int(row["seed"]), int(row["tick"])))
    runner.write_trials(run_dir, rows, TRIAL_FIELDNAMES)
    snapshot_path = None
    if args.log_snapshots:
        logging_config = _snapshot_logging_config(run_dir, int(args.snapshot_log_rate))
        snapshot_path = runner.write_snapshots(
            run_dir,
            snapshots,
            logging_config,
        )

    summary = {
        "run_dir": str(run_dir),
        "source": source,
        "sim": sim_name,
        "num_scenarios": int(len(tasks)),
        "offset": int(args.offset),
        "workers": int(workers),
        "max_envs": int(args.max_envs),
        "duration_s": duration_s,
        "dt": dt_s,
        "elapsed_wall_s": time.perf_counter() - start,
        "snapshot_log": {
            "enabled": bool(args.log_snapshots),
            "every_n_ticks": int(args.snapshot_log_rate),
            "path": None if snapshot_path is None else str(snapshot_path),
        },
        "summary": _summarize_subset(rows),
    }
    summary_path = runner.write_summary(run_dir, summary)
    print(summary_path.read_text(encoding="utf-8"), end="")
    return 0


def _load_tasks(args: argparse.Namespace) -> tuple[list[Any], str, float, float]:
    if args.scenario_table is None:
        raise ValueError(
            "--scenario-table is required. Generate a .csimin file under "
            "scripts/generators/sim_instances before running control sims."
        )

    scenario_table = Path(args.scenario_table)
    if not scenario_table.exists():
        raise FileNotFoundError(f"scenario table not found: {scenario_table}")
    instances = read_sim_instances(
        scenario_table,
        count=None if args.samples is None else int(args.samples),
        offset=int(args.offset),
    )
    if not instances:
        return [], str(scenario_table), math.nan, math.nan
    first_config = instances[0].config
    return (
        list(instances),
        str(scenario_table),
        float(first_config.options.duration_s),
        float(first_config.options.backend_dt),
    )


def _resolve_workers(requested: int | None, samples: int) -> int:
    if samples <= 0:
        return 1
    if requested is not None:
        return max(1, min(int(samples), int(requested)))
    cpu_count = os.cpu_count() or 1
    return max(1, min(int(samples), max(1, cpu_count - 1)))


def _task_chunks(tasks: list[Any], chunk_size: int) -> list[list[Any]]:
    size = max(1, int(chunk_size))
    return [tasks[index:index + size] for index in range(0, len(tasks), size)]


def _error_rows_for_chunk(chunk: list[Any], sim_name: str, exc: Exception) -> list[dict[str, Any]]:
    rows = []
    for task in chunk:
        instance, scenario_fields = _materialize_task(task)
        row = _error_row_for_seed(sim_name, int(instance.seed), str(exc))
        row.update(scenario_fields)
        row["wall_s"] = math.nan
        rows.append(row)
    return rows


def _print_progress(completed: int, total: int, progress_every: int, start: float) -> None:
    if progress_every <= 0:
        return
    if completed % progress_every != 0 and completed != total:
        return
    elapsed = time.perf_counter() - start
    print(f"completed {completed}/{total} scenarios in {elapsed:.1f}s", flush=True)


def _run_instances(
    instances: list[Any],
    sim_name: str,
    policy_factory: PolicyFactory,
    *,
    max_envs: int,
    log_snapshots: bool,
    snapshot_log_rate: int,
) -> dict[str, Any]:
    from backends.csim.runner import SimRunner

    start = time.perf_counter()
    runner = SimRunner(max_envs=max_envs)
    result = runner.run(instances, policy_factory())
    elapsed = time.perf_counter() - start

    rows = []
    snapshots: list[dict[str, Any]] = []
    step_filter = _completed_step_filter(result.completed)
    for completed in result.completed:
        row = _row_from_completed(sim_name, completed, result.steps)
        row.update(_scenario_fields_from_instance(completed.instance))
        row["wall_s"] = elapsed / max(len(result.completed), 1)
        row["error"] = None
        rows.append(row)
    if log_snapshots:
        logging_config = _snapshot_logging_config(Path("."), snapshot_log_rate)
        for step in result.steps:
            snapshots.extend(
                row for row in snapshot_rows_from_step(sim_name, logging_config, step)
                if (int(row["slot"]), int(row["workload_index"])) in step_filter
            )
    return {"rows": rows, "snapshots": snapshots}


def _materialize_task(task: Any):
    return task, _scenario_fields_from_instance(task)


def _row_from_completed(sim_name: str, completed, steps: tuple[Any, ...]) -> dict[str, Any]:
    terminal = completed.terminal_snapshot
    effort = _simrunner_control_effort(
        steps,
        dt_s=_dt_from_instance(completed.instance),
        slot=completed.slot,
        workload_index=completed.workload_index,
    )
    visible_fraction = _simrunner_visible_fraction(
        steps,
        slot=completed.slot,
        workload_index=completed.workload_index,
    )
    terminal_speeds = _terminal_speed_fields(terminal)
    return {
        "sim": sim_name,
        "seed": int(completed.seed),
        "caught": completed.terminal_reason == "intercepted",
        "catch_time_s": (
            float(terminal.metrics.intercept_time_s)
            if completed.terminal_reason == "intercepted"
            else None
        ),
        "min_distance_m": float(terminal.metrics.min_distance_m),
        "final_distance_m": float(terminal.metrics.distance_m),
        "visible_fraction": visible_fraction,
        "control_effort": effort,
        **terminal_speeds,
        "steps": int(completed.steps),
        "crashed": False,
        "out_of_bounds": completed.terminal_reason == "oob",
    }


def _terminal_speed_fields(terminal) -> dict[str, float]:
    pursuer_velocity_w = np.asarray(terminal.pursuer.velocity_w, dtype=float).reshape(3)
    target_velocity_w = np.asarray(terminal.target.velocity_w, dtype=float).reshape(3)
    return {
        "terminal_pursuer_speed_mps": float(np.linalg.norm(pursuer_velocity_w)),
        "terminal_target_speed_mps": float(np.linalg.norm(target_velocity_w)),
        "terminal_relative_speed_mps": float(np.linalg.norm(pursuer_velocity_w - target_velocity_w)),
    }


def _dt_from_instance(instance) -> float:
    return float(instance.config.options.backend_dt) * max(1, int(instance.config.options.action_substeps))


def _simrunner_control_effort(
    steps: tuple[Any, ...],
    *,
    dt_s: float,
    slot: int,
    workload_index: int | None = None,
) -> float:
    total = 0.0
    for step in steps:
        if step.commands is None:
            continue
        slot_i = int(slot)
        if not bool(step.state.active[slot_i]):
            continue
        if workload_index is not None and int(step.state.workload_indices[slot_i]) != int(workload_index):
            continue
        if hasattr(step.commands, "motor_speeds_rpm"):
            if slot_i >= len(step.commands.motor_speeds_rpm):
                continue
            motor_speeds_rpm = np.asarray(step.commands.motor_speeds_rpm[slot_i], dtype=float)
            total += 1.0e-4 * float(np.linalg.norm(motor_speeds_rpm)) * float(dt_s)
        else:
            if slot_i >= len(step.commands.thrust_n):
                continue
            thrust_n = float(step.commands.thrust_n[slot_i])
            body_rates_b = np.asarray(step.commands.body_rates_b[slot_i], dtype=float)
            total += (float(np.linalg.norm(body_rates_b)) + 0.02 * abs(thrust_n)) * float(dt_s)
    return float(total)


def _simrunner_visible_fraction(
    steps: tuple[Any, ...],
    *,
    slot: int,
    workload_index: int | None = None,
) -> float:
    active_steps = [
        step for step in steps
        if int(slot) < len(step.state.active) and bool(step.state.active[int(slot)])
        and (workload_index is None or int(step.state.workload_indices[int(slot)]) == int(workload_index))
    ]
    if not active_steps:
        return 0.0
    visible = sum(1 for step in active_steps if step.state.snapshot[int(slot)].camera.detected)
    return visible / len(active_steps)


def _completed_step_filter(completed) -> set[tuple[int, int]]:
    return {
        (int(item.slot), int(item.workload_index))
        for item in completed
    }


def _snapshot_logging_config(run_dir: Path, every_n_ticks: int) -> LoggingConfig:
    return LoggingConfig(output_dir=Path(run_dir) / "snapshots", every_n_ticks=every_n_ticks)


def _scenario_fields_for_task(task: Any) -> dict[str, Any]:
    return _scenario_fields_from_instance(task)


def _scenario_fields_from_instance(instance) -> dict[str, Any]:
    target_initial = instance.target_initials[0]
    rel_pos = np.asarray(target_initial.position_w, dtype=float) - np.asarray(instance.pursuer_initial.position_w, dtype=float)
    rel_vel = np.asarray(instance.pursuer_initial.velocity_w, dtype=float) - np.asarray(target_initial.velocity_w, dtype=float)
    range_m = float(np.linalg.norm(rel_pos))
    los_w = rel_pos / max(range_m, 1e-12)
    return {
        "stratum": "generated",
        "range_m": range_m,
        "closing_speed_mps": float(np.dot(rel_vel, los_w)),
    }


def _error_row_for_seed(sim_name: str, seed: int, error: str) -> dict[str, Any]:
    return {
        "sim": sim_name,
        "seed": int(seed),
        "caught": False,
        "catch_time_s": None,
        "min_distance_m": math.nan,
        "final_distance_m": math.nan,
        "visible_fraction": math.nan,
        "control_effort": math.nan,
        "terminal_pursuer_speed_mps": math.nan,
        "terminal_target_speed_mps": math.nan,
        "terminal_relative_speed_mps": math.nan,
        "steps": 0,
        "crashed": False,
        "out_of_bounds": False,
        "error": error,
    }


TRIAL_FIELDNAMES = [
    "sim",
    "seed",
    "stratum",
    "range_m",
    "closing_speed_mps",
    "caught",
    "catch_time_s",
    "min_distance_m",
    "final_distance_m",
    "visible_fraction",
    "control_effort",
    "terminal_pursuer_speed_mps",
    "terminal_target_speed_mps",
    "terminal_relative_speed_mps",
    "steps",
    "crashed",
    "out_of_bounds",
    "wall_s",
    "error",
]


def _summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    min_distance = _finite_array(row["min_distance_m"] for row in valid)
    final_distance = _finite_array(row["final_distance_m"] for row in valid)
    visible = _finite_array(row["visible_fraction"] for row in valid)
    effort = _finite_array(row["control_effort"] for row in valid)
    caught = np.array([bool(row["caught"]) for row in valid], dtype=bool)
    return {
        "n": int(len(rows)),
        "valid": int(len(valid)),
        "errors": int(len(rows) - len(valid)),
        "catch_fraction": float(np.mean(caught)) if caught.size else math.nan,
        "min_distance_p50_m": _percentile(min_distance, 50),
        "min_distance_p90_m": _percentile(min_distance, 90),
        "final_distance_p50_m": _percentile(final_distance, 50),
        "visible_fraction_mean": _mean(visible),
        "control_effort_mean": _mean(effort),
    }


def _finite_array(values) -> np.ndarray:
    array = np.array(list(values), dtype=float)
    return array[np.isfinite(array)]


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else math.nan


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan
