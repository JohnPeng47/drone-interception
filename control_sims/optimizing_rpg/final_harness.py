from __future__ import annotations

import concurrent.futures
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backends.csim.bindings.types import SimInstance
from backends.csim.generator.instance_store import read_sim_instances
from control_sims.rpg_time_optimal.portfolio_policy import solve_portfolio_plan

from .baseline_harness import DEFAULT_SCENARIO_TABLE
from .fixed_time_harness import DEFAULT_CATCH_TABLE
from .rollout_harness import DEFAULT_BASELINE_SUMMARY
from .time_search import TimeSearchResult, find_fastest_intercept
from .time_search_harness import DEFAULT_TIME_MULTIPLIERS


DEFAULT_OUTPUT_DIR = Path(".agents/projects/optimizing-rpg-solver/milestone-6-performance-hardening/artifacts")


@dataclass(frozen=True)
class FinalHarnessConfig:
    scenario_table: Path = DEFAULT_SCENARIO_TABLE
    multi_table: Path = DEFAULT_CATCH_TABLE
    baseline_summary: Path = DEFAULT_BASELINE_SUMMARY
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = 1
    multi_seeds: tuple[int, ...] = (1, 2)
    workers: int = 2
    label: str = "performance_hardening"
    time_multipliers: tuple[float, ...] = DEFAULT_TIME_MULTIPLIERS


@dataclass(frozen=True)
class WarmStartBundle:
    instance: SimInstance
    selected_candidate: str
    total_time_s: float
    motor_speed_commands_rpm: Any
    dynamics_substeps: int
    portfolio_wall_s: float
    portfolio_caught: bool
    portfolio_min_distance_m: float
    portfolio_final_distance_m: float


def run_final_harness(config: FinalHarnessConfig) -> dict[str, Any]:
    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_wall_s = _load_baseline_wall_s(Path(config.baseline_summary))

    primary_instance = _load_seed_instance(Path(config.scenario_table), int(config.seed))
    primary_bundle = _acquire_warm_start(primary_instance)
    single_started = time.perf_counter()
    single_serial = _run_search(primary_bundle, config, mode="serial", workers=1)
    single_parallel = _run_search(primary_bundle, config, mode="parallel", workers=int(config.workers))
    single_wall_s = time.perf_counter() - single_started + primary_bundle.portfolio_wall_s
    single_row = _single_row(config, primary_bundle, single_serial, single_parallel, baseline_wall_s, single_wall_s)

    multi_instances = _load_seed_instances(Path(config.multi_table), config.multi_seeds)
    acquire_started = time.perf_counter()
    multi_bundles = [_acquire_warm_start(instance) for instance in multi_instances]
    multi_plan_acquire_wall_s = time.perf_counter() - acquire_started

    multi_serial_started = time.perf_counter()
    multi_serial_results = [_run_search(bundle, config, mode="serial", workers=1) for bundle in multi_bundles]
    multi_serial_wall_s = time.perf_counter() - multi_serial_started

    multi_parallel_started = time.perf_counter()
    scenario_workers = max(1, min(int(config.workers), len(multi_bundles)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=scenario_workers) as executor:
        futures = [executor.submit(_run_search, bundle, config, "parallel", int(config.workers)) for bundle in multi_bundles]
        multi_parallel_results = [future.result() for future in concurrent.futures.as_completed(futures)]
    multi_parallel_wall_s = time.perf_counter() - multi_parallel_started
    multi_rows = _multi_rows(config, multi_bundles, multi_serial_results, multi_parallel_results)

    _write_csv(output_dir / "single_scenario_rows.csv", [single_row])
    _write_csv(output_dir / "multi_scenario_rows.csv", multi_rows)

    delta_vs_baseline_s = float(baseline_wall_s) - float(single_wall_s)
    percent_improvement = (delta_vs_baseline_s / float(baseline_wall_s)) * 100.0
    multi_parallel_caught = [bool(result.caught) for result in multi_parallel_results]
    multi_serial_caught = [bool(result.caught) for result in multi_serial_results]
    multi_parallel_catch_fraction = float(sum(multi_parallel_caught) / len(multi_parallel_caught)) if multi_parallel_caught else math.nan
    multi_serial_catch_fraction = float(sum(multi_serial_caught) / len(multi_serial_caught)) if multi_serial_caught else math.nan
    summary = {
        "config": {
            **asdict(config),
            "scenario_table": str(config.scenario_table),
            "multi_table": str(config.multi_table),
            "baseline_summary": str(config.baseline_summary),
            "output_dir": str(config.output_dir),
            "multi_seeds": list(config.multi_seeds),
            "time_multipliers": list(config.time_multipliers),
        },
        "baseline_wall_s": float(baseline_wall_s),
        "elapsed_wall_s": float(single_wall_s),
        "delta_vs_baseline_s": float(delta_vs_baseline_s),
        "percent_improvement_vs_baseline": float(percent_improvement),
        "single_current_portfolio_wall_s": float(primary_bundle.portfolio_wall_s),
        "single_custom_serial_wall_s": float(single_serial.wall_s),
        "single_custom_parallel_wall_s": float(single_parallel.wall_s),
        "single_custom_fastest_caught_time_s": float(single_parallel.fastest_caught_time_s),
        "single_custom_caught": bool(single_parallel.caught),
        "multi_num_scenarios": int(len(multi_bundles)),
        "multi_current_portfolio_acquire_wall_s": float(multi_plan_acquire_wall_s),
        "multi_custom_serial_wall_s": float(multi_serial_wall_s),
        "multi_custom_parallel_wall_s": float(multi_parallel_wall_s),
        "multi_custom_parallel_scenario_workers": int(scenario_workers),
        "multi_custom_parallel_probe_workers_per_scenario": int(config.workers),
        "multi_custom_serial_catch_fraction": float(multi_serial_catch_fraction),
        "multi_custom_parallel_catch_fraction": float(multi_parallel_catch_fraction),
        "multi_custom_serial_caught": int(sum(multi_serial_caught)),
        "multi_custom_parallel_caught": int(sum(multi_parallel_caught)),
        "multi_custom_parallel_throughput_scenarios_per_s": (
            float(len(multi_bundles)) / float(multi_parallel_wall_s)
            if multi_parallel_wall_s > 0.0 else math.inf
        ),
        "passed_acceptance": bool(_passed_acceptance(single_row, multi_rows)),
        "artifacts": {
            "single_scenario_rows_csv": "single_scenario_rows.csv",
            "multi_scenario_rows_csv": "multi_scenario_rows.csv",
            "summary_json": "summary.json",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _acquire_warm_start(instance: SimInstance) -> WarmStartBundle:
    started = time.perf_counter()
    selected = solve_portfolio_plan(instance)
    wall_s = time.perf_counter() - started
    plan = selected.plan
    if plan.motor_speed_commands_rpm is None:
        raise ValueError("selected plan does not include motor speed commands")
    return WarmStartBundle(
        instance=instance,
        selected_candidate=selected.candidate.name,
        total_time_s=float(plan.total_time_s),
        motor_speed_commands_rpm=plan.motor_speed_commands_rpm,
        dynamics_substeps=int(selected.candidate.config.dynamics_substeps),
        portfolio_wall_s=float(wall_s),
        portfolio_caught=bool(selected.score.rollout_caught_radius),
        portfolio_min_distance_m=float(selected.score.rollout_min_distance_m),
        portfolio_final_distance_m=float(selected.score.rollout_final_distance_m),
    )


def _run_search(bundle: WarmStartBundle, config: FinalHarnessConfig, mode: str, workers: int) -> TimeSearchResult:
    return find_fastest_intercept(
        bundle.instance,
        bundle.motor_speed_commands_rpm,
        bundle.total_time_s,
        time_multipliers=tuple(config.time_multipliers),
        dynamics_substeps=int(bundle.dynamics_substeps),
        control_layout="columns",
        mode=mode,
        workers=int(workers),
    )


def _single_row(
    config: FinalHarnessConfig,
    bundle: WarmStartBundle,
    serial: TimeSearchResult,
    parallel: TimeSearchResult,
    baseline_wall_s: float,
    single_wall_s: float,
) -> dict[str, Any]:
    delta_vs_baseline_s = float(baseline_wall_s) - float(single_wall_s)
    return {
        "label": str(config.label),
        "seed": int(bundle.instance.seed),
        "selected_candidate": str(bundle.selected_candidate),
        "baseline_wall_s": float(baseline_wall_s),
        "scenario_wall_s": float(single_wall_s),
        "delta_vs_baseline_s": float(delta_vs_baseline_s),
        "percent_improvement_vs_baseline": (delta_vs_baseline_s / float(baseline_wall_s)) * 100.0,
        "current_portfolio_wall_s": float(bundle.portfolio_wall_s),
        "current_portfolio_caught": bool(bundle.portfolio_caught),
        "custom_serial_wall_s": float(serial.wall_s),
        "custom_parallel_wall_s": float(parallel.wall_s),
        "custom_serial_caught": bool(serial.caught),
        "custom_parallel_caught": bool(parallel.caught),
        "custom_serial_fastest_caught_time_s": float(serial.fastest_caught_time_s),
        "custom_parallel_fastest_caught_time_s": float(parallel.fastest_caught_time_s),
        "custom_parallel_workers": int(parallel.workers),
        "fastest_times_match": bool(
            serial.caught
            and parallel.caught
            and abs(float(serial.fastest_caught_time_s) - float(parallel.fastest_caught_time_s)) <= 1.0e-9
        ),
        "custom_parallel_probes": int(parallel.probes_executed),
        "error": "",
    }


def _multi_rows(
    config: FinalHarnessConfig,
    bundles: list[WarmStartBundle],
    serial_results: list[TimeSearchResult],
    parallel_results: list[TimeSearchResult],
) -> list[dict[str, Any]]:
    serial_by_seed = {int(result.seed): result for result in serial_results}
    parallel_by_seed = {int(result.seed): result for result in parallel_results}
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        seed = int(bundle.instance.seed)
        serial = serial_by_seed[seed]
        parallel = parallel_by_seed[seed]
        rows.append(
            {
                "label": str(config.label),
                "seed": seed,
                "selected_candidate": str(bundle.selected_candidate),
                "current_portfolio_wall_s": float(bundle.portfolio_wall_s),
                "current_portfolio_caught": bool(bundle.portfolio_caught),
                "current_portfolio_min_distance_m": float(bundle.portfolio_min_distance_m),
                "current_portfolio_final_distance_m": float(bundle.portfolio_final_distance_m),
                "serial_wall_s": float(serial.wall_s),
                "parallel_wall_s": float(parallel.wall_s),
                "serial_caught": bool(serial.caught),
                "parallel_caught": bool(parallel.caught),
                "serial_fastest_caught_time_s": float(serial.fastest_caught_time_s),
                "parallel_fastest_caught_time_s": float(parallel.fastest_caught_time_s),
                "parallel_min_distance_m": float(parallel.replay_min_distance_m),
                "parallel_final_distance_m": float(parallel.replay_final_distance_m),
                "serial_probe_errors": int(sum(1 for probe in serial.probe_results if probe.error)),
                "parallel_probe_errors": int(sum(1 for probe in parallel.probe_results if probe.error)),
                "parallel_probes_executed": int(parallel.probes_executed),
                "parallel_workers": int(parallel.workers),
                "fastest_times_match": bool(
                    serial.caught
                    and parallel.caught
                    and abs(float(serial.fastest_caught_time_s) - float(parallel.fastest_caught_time_s)) <= 1.0e-9
                ),
                "error": "",
            }
        )
    return rows


def _passed_acceptance(single_row: dict[str, Any], multi_rows: list[dict[str, Any]]) -> bool:
    try:
        distinct_seeds = {int(row["seed"]) for row in multi_rows}
    except (KeyError, TypeError, ValueError):
        return False
    if len(multi_rows) <= 1 or len(distinct_seeds) <= 1:
        return False
    if not (
        single_row["error"] == ""
        and bool(single_row["custom_serial_caught"])
        and bool(single_row["custom_parallel_caught"])
        and bool(single_row["fastest_times_match"])
        and int(single_row["custom_parallel_probes"]) > 1
        and int(single_row["custom_parallel_workers"]) > 1
    ):
        return False
    valid_multi = [row for row in multi_rows if row["error"] == ""]
    if len(valid_multi) != len(multi_rows):
        return False
    return all(
        bool(row["serial_caught"])
        and bool(row["parallel_caught"])
        and bool(row["fastest_times_match"])
        and int(row["serial_probe_errors"]) == 0
        and int(row["parallel_probe_errors"]) == 0
        and int(row["parallel_probes_executed"]) > 1
        and int(row["parallel_workers"]) > 1
        for row in valid_multi
    )


def _validate_config(config: FinalHarnessConfig) -> None:
    if int(config.workers) < 2:
        raise ValueError("workers must be at least 2 for final parallelization evidence")
    distinct_multi_seeds = {int(seed) for seed in config.multi_seeds}
    if len(distinct_multi_seeds) != len(config.multi_seeds):
        raise ValueError("multi_seeds must be distinct")
    if len(distinct_multi_seeds) <= 1:
        raise ValueError("multi_seeds must contain more than one distinct seed")


def _load_seed_instance(scenario_table: Path, seed: int) -> SimInstance:
    instances = read_sim_instances(scenario_table)
    for instance in instances:
        if int(instance.seed) == int(seed):
            return instance
    raise ValueError(f"scenario table {scenario_table} is missing seed {seed}")


def _load_seed_instances(scenario_table: Path, seeds: tuple[int, ...]) -> list[SimInstance]:
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
