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
from .fixed_time_harness import DEFAULT_CATCH_TABLE
from .rollout_harness import DEFAULT_BASELINE_SUMMARY
from .time_search import TimeSearchResult, find_fastest_intercept


DEFAULT_OUTPUT_DIR = Path(".agents/projects/optimizing-rpg-solver/milestone-5-parallel-time-search/artifacts")
DEFAULT_TIME_MULTIPLIERS = (0.7, 0.8, 0.9, 1.0, 1.1)


@dataclass(frozen=True)
class TimeSearchHarnessConfig:
    scenario_table: Path = DEFAULT_SCENARIO_TABLE
    catch_table: Path = DEFAULT_CATCH_TABLE
    baseline_summary: Path = DEFAULT_BASELINE_SUMMARY
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = 1
    catch_seeds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    workers: int = 4
    label: str = "parallel_time_search"
    time_multipliers: tuple[float, ...] = DEFAULT_TIME_MULTIPLIERS


def run_time_search_harness(config: TimeSearchHarnessConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_wall_s = _load_baseline_wall_s(Path(config.baseline_summary))

    benchmark_instance = _load_seed_instance(Path(config.scenario_table), int(config.seed))
    benchmark = _run_search_pair(config, benchmark_instance, baseline_wall_s=baseline_wall_s, diagnostic_table=Path(config.scenario_table).stem)
    catch_rows: list[dict[str, Any]] = []
    catch_probe_rows: list[dict[str, Any]] = []
    for instance in _load_seed_instances(Path(config.catch_table), config.catch_seeds):
        row, probes = _run_search_pair(
            config,
            instance,
            baseline_wall_s=baseline_wall_s,
            diagnostic_table=Path(config.catch_table).stem,
        )
        catch_rows.append(row)
        catch_probe_rows.extend(probes)

    benchmark_row, benchmark_probe_rows = benchmark
    _write_csv(output_dir / "time_search_rows.csv", [benchmark_row])
    _write_csv(output_dir / "probe_rows.csv", benchmark_probe_rows)
    _write_csv(output_dir / "catch_search_rows.csv", catch_rows)
    _write_csv(output_dir / "catch_probe_rows.csv", catch_probe_rows)

    scenario_wall_s = float(benchmark_row["scenario_wall_s"])
    delta_vs_baseline_s = float(baseline_wall_s) - scenario_wall_s
    percent_improvement = (delta_vs_baseline_s / float(baseline_wall_s)) * 100.0
    catch_valid = [row for row in catch_rows if row["error"] == ""]
    catch_caught = [bool(row["parallel_caught"]) for row in catch_rows]
    summary = {
        "config": {
            **asdict(config),
            "scenario_table": str(config.scenario_table),
            "catch_table": str(config.catch_table),
            "baseline_summary": str(config.baseline_summary),
            "output_dir": str(config.output_dir),
            "catch_seeds": list(config.catch_seeds),
            "time_multipliers": list(config.time_multipliers),
        },
        "baseline_wall_s": float(baseline_wall_s),
        "elapsed_wall_s": float(scenario_wall_s),
        "delta_vs_baseline_s": float(delta_vs_baseline_s),
        "percent_improvement_vs_baseline": float(percent_improvement),
        "plan_acquire_wall_s": float(benchmark_row["plan_acquire_wall_s"]),
        "serial_wall_s": float(benchmark_row["serial_wall_s"]),
        "parallel_wall_s": float(benchmark_row["parallel_wall_s"]),
        "serial_probes_executed": int(benchmark_row["serial_probes_executed"]),
        "parallel_probes_executed": int(benchmark_row["parallel_probes_executed"]),
        "fastest_caught_time_s": float(benchmark_row["parallel_fastest_caught_time_s"]),
        "passed_acceptance": bool(_passed_acceptance(config, benchmark_row, catch_rows)),
        "catch_search": {
            "table": str(config.catch_table),
            "seeds": list(config.catch_seeds),
            "default_target_seeds": [1, 2, 3, 4, 5, 6, 7, 8],
            "uses_reduced_subset": list(config.catch_seeds) != [1, 2, 3, 4, 5, 6, 7, 8],
            "num_scenarios": int(len(catch_rows)),
            "valid": int(len(catch_valid)),
            "errors": int(len(catch_rows) - len(catch_valid)),
            "catch_fraction": float(sum(catch_caught) / len(catch_caught)) if catch_caught else math.nan,
        },
        "artifacts": {
            "time_search_rows_csv": "time_search_rows.csv",
            "probe_rows_csv": "probe_rows.csv",
            "catch_search_rows_csv": "catch_search_rows.csv",
            "catch_probe_rows_csv": "catch_probe_rows.csv",
            "summary_json": "summary.json",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _run_search_pair(
    config: TimeSearchHarnessConfig,
    instance,
    *,
    baseline_wall_s: float,
    diagnostic_table: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    try:
        plan_started = time.perf_counter()
        selected = solve_portfolio_plan(instance)
        plan_acquire_wall_s = time.perf_counter() - plan_started
        plan = selected.plan
        if plan.motor_speed_commands_rpm is None:
            raise ValueError("selected plan does not include motor speed commands")
        serial = find_fastest_intercept(
            instance,
            plan.motor_speed_commands_rpm,
            float(plan.total_time_s),
            time_multipliers=tuple(config.time_multipliers),
            dynamics_substeps=int(selected.candidate.config.dynamics_substeps),
            control_layout="columns",
            mode="serial",
            workers=1,
        )
        parallel = find_fastest_intercept(
            instance,
            plan.motor_speed_commands_rpm,
            float(plan.total_time_s),
            time_multipliers=tuple(config.time_multipliers),
            dynamics_substeps=int(selected.candidate.config.dynamics_substeps),
            control_layout="columns",
            mode="parallel",
            workers=int(config.workers),
        )
        scenario_wall_s = time.perf_counter() - started
        row = _row_from_results(
            config,
            instance,
            selected_candidate=selected.candidate.name,
            baseline_wall_s=baseline_wall_s,
            plan_acquire_wall_s=plan_acquire_wall_s,
            scenario_wall_s=scenario_wall_s,
            diagnostic_table=diagnostic_table,
            serial=serial,
            parallel=parallel,
            error="",
        )
        return row, _probe_rows(config, instance, diagnostic_table, serial, parallel)
    except Exception as exc:  # noqa: BLE001
        scenario_wall_s = time.perf_counter() - started
        row = {
            "label": str(config.label),
            "diagnostic_table": str(diagnostic_table),
            "seed": int(instance.seed),
            "selected_candidate": "",
            "baseline_wall_s": float(baseline_wall_s),
            "scenario_wall_s": float(scenario_wall_s),
            "delta_vs_baseline_s": float(baseline_wall_s) - float(scenario_wall_s),
            "percent_improvement_vs_baseline": ((float(baseline_wall_s) - float(scenario_wall_s)) / float(baseline_wall_s)) * 100.0,
            "plan_acquire_wall_s": math.nan,
            "serial_wall_s": math.nan,
            "parallel_wall_s": math.nan,
            "serial_probes_executed": 0,
            "parallel_probes_executed": 0,
            "serial_caught": False,
            "parallel_caught": False,
            "serial_fastest_caught_time_s": math.nan,
            "parallel_fastest_caught_time_s": math.nan,
            "serial_probe_errors": 0,
            "parallel_probe_errors": 0,
            "fastest_times_match": False,
            "parallel_workers": int(config.workers),
            "error": repr(exc),
        }
        return row, []


def _row_from_results(
    config: TimeSearchHarnessConfig,
    instance,
    *,
    selected_candidate: str,
    baseline_wall_s: float,
    plan_acquire_wall_s: float,
    scenario_wall_s: float,
    diagnostic_table: str,
    serial: TimeSearchResult,
    parallel: TimeSearchResult,
    error: str,
) -> dict[str, Any]:
    delta_vs_baseline_s = float(baseline_wall_s) - float(scenario_wall_s)
    fastest_match = bool(
        serial.caught
        and parallel.caught
        and abs(float(serial.fastest_caught_time_s) - float(parallel.fastest_caught_time_s)) <= 1.0e-9
    )
    return {
        "label": str(config.label),
        "diagnostic_table": str(diagnostic_table),
        "seed": int(instance.seed),
        "selected_candidate": str(selected_candidate),
        "baseline_wall_s": float(baseline_wall_s),
        "scenario_wall_s": float(scenario_wall_s),
        "delta_vs_baseline_s": float(delta_vs_baseline_s),
        "percent_improvement_vs_baseline": (delta_vs_baseline_s / float(baseline_wall_s)) * 100.0,
        "plan_acquire_wall_s": float(plan_acquire_wall_s),
        "serial_wall_s": float(serial.wall_s),
        "parallel_wall_s": float(parallel.wall_s),
        "serial_probes_executed": int(serial.probes_executed),
        "parallel_probes_executed": int(parallel.probes_executed),
        "serial_caught": bool(serial.caught),
        "parallel_caught": bool(parallel.caught),
        "serial_fastest_caught_time_s": float(serial.fastest_caught_time_s),
        "parallel_fastest_caught_time_s": float(parallel.fastest_caught_time_s),
        "serial_replay_min_distance_m": float(serial.replay_min_distance_m),
        "parallel_replay_min_distance_m": float(parallel.replay_min_distance_m),
        "serial_probe_errors": int(sum(1 for probe in serial.probe_results if probe.error)),
        "parallel_probe_errors": int(sum(1 for probe in parallel.probe_results if probe.error)),
        "serial_early_exit_reason": str(serial.early_exit_reason),
        "parallel_early_exit_reason": str(parallel.early_exit_reason),
        "fastest_times_match": fastest_match,
        "parallel_workers": int(parallel.workers),
        "error": str(error),
    }


def _probe_rows(
    config: TimeSearchHarnessConfig,
    instance,
    diagnostic_table: str,
    serial: TimeSearchResult,
    parallel: TimeSearchResult,
) -> list[dict[str, Any]]:
    rows = []
    for search in (serial, parallel):
        for probe in search.probe_results:
            rows.append(
                {
                    "label": str(config.label),
                    "diagnostic_table": str(diagnostic_table),
                    "seed": int(instance.seed),
                    "mode": str(search.mode),
                    "probe_index": int(probe.index),
                    "total_time_s": float(probe.total_time_s),
                    "wall_s": float(probe.wall_s),
                    "caught": bool(probe.caught),
                    "feasible": bool(probe.feasible),
                    "failure_reason": str(probe.failure_reason),
                    "replay_min_distance_m": float(probe.replay_min_distance_m),
                    "replay_final_distance_m": float(probe.replay_final_distance_m),
                    "replay_wall_s": float(probe.replay_wall_s),
                    "replay_steps": int(probe.replay_steps),
                    "error": str(probe.error),
                }
            )
    return rows


def _passed_acceptance(config: TimeSearchHarnessConfig, benchmark_row: dict[str, Any], catch_rows: list[dict[str, Any]]) -> bool:
    expected_catch_seeds = tuple(int(seed) for seed in config.catch_seeds)
    observed_catch_seeds = tuple(sorted(int(row["seed"]) for row in catch_rows))
    expected_parallel_probes = len({float(value) for value in config.time_multipliers})
    catch_acceptance = (
        observed_catch_seeds == tuple(sorted(expected_catch_seeds))
        and all(_row_passed_scheduler_acceptance(row, expected_parallel_probes=expected_parallel_probes) for row in catch_rows)
    )
    return bool(
        _row_passed_scheduler_acceptance(benchmark_row, expected_parallel_probes=expected_parallel_probes)
        and catch_acceptance
    )


def _row_passed_scheduler_acceptance(row: dict[str, Any], *, expected_parallel_probes: int) -> bool:
    return bool(
        row["error"] == ""
        and bool(row["serial_caught"])
        and bool(row["parallel_caught"])
        and bool(row["fastest_times_match"])
        and int(row["serial_probes_executed"]) >= 1
        and int(row["parallel_probes_executed"]) == int(expected_parallel_probes)
        and int(row["parallel_probes_executed"]) > 1
        and int(row["serial_probe_errors"]) == 0
        and int(row["parallel_probe_errors"]) == 0
    )


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
