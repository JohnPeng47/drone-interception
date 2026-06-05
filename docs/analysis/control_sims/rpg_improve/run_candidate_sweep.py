from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from backends.csim.generator.instance_store import read_sim_instances
from control_sims.rpg_time_optimal.config import RpgTimeOptimalConfig
from control_sims.rpg_time_optimal.portfolio_policy import DEFAULT_PORTFOLIO_CANDIDATES as PRODUCTION_PORTFOLIO_CANDIDATES

import run_diagnostics as diag_helpers


SCENARIO_TABLE = REPO_ROOT / "scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin"


@dataclass(frozen=True)
class Candidate:
    name: str
    cpc_tolerance_m: float | None
    plan_time_scale: float
    motor_command_mode: str
    terminal_nodes: int
    dynamics_substeps: int
    planner_rate_limit_scale: float
    command_smoothness_weight: float
    body_rate_smoothness_weight: float
    terminal_capture_window_nodes: int
    ipopt_max_iter: int

    def config(self) -> RpgTimeOptimalConfig:
        return RpgTimeOptimalConfig(
            cpc_tolerance_m=self.cpc_tolerance_m,
            plan_time_scale=self.plan_time_scale,
            motor_command_mode=self.motor_command_mode,
            terminal_nodes=self.terminal_nodes,
            dynamics_substeps=self.dynamics_substeps,
            planner_rate_limit_scale=self.planner_rate_limit_scale,
            command_smoothness_weight=self.command_smoothness_weight,
            body_rate_smoothness_weight=self.body_rate_smoothness_weight,
            terminal_capture_window_nodes=self.terminal_capture_window_nodes,
            ipopt_max_iter=self.ipopt_max_iter,
        )


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = read_sim_instances(args.scenario_table)
    seeds = _selected_seeds(instances, args.seeds)
    candidates = _candidates_from_args(args)
    tasks = [(args.scenario_table, seed, candidate, args.rollout_tail_s, args.post_plan_command_mode) for candidate in candidates for seed in seeds]
    workers = _resolve_workers(args.workers, len(tasks))

    started = time.perf_counter()
    rows = _run_tasks(tasks, workers, output_dir=output_dir, progress_every=int(args.progress_every))
    elapsed_wall_s = time.perf_counter() - started
    rows.sort(key=lambda row: (str(row["candidate"]), int(row["seed"])))

    candidate_summary = _candidate_summary(rows, len(seeds), args.constraint_violation_tolerance)
    best_by_seed = _best_by_seed(rows, args.constraint_violation_tolerance)
    portfolio = {
        "seeds": seeds,
        "catch_count": sum(1 for row in best_by_seed if bool(row["rollout_caught_radius"])),
        "catch_fraction": (
            sum(1 for row in best_by_seed if bool(row["rollout_caught_radius"])) / max(len(seeds), 1)
        ),
        "selected_seed_count": len(best_by_seed),
        "max_best_min_distance_m": _finite_max(row["rollout_min_distance_m"] for row in best_by_seed),
        "mean_best_min_distance_m": _finite_mean(row["rollout_min_distance_m"] for row in best_by_seed),
    }

    _write_csv(output_dir / "candidate_results.csv", rows)
    _write_csv(output_dir / "candidate_summary.csv", candidate_summary)
    _write_csv(output_dir / "best_by_seed.csv", best_by_seed)
    summary = {
        "scenario_table": str(args.scenario_table.relative_to(REPO_ROOT) if args.scenario_table.is_relative_to(REPO_ROOT) else args.scenario_table),
        "output_dir": str(output_dir.relative_to(REPO_ROOT) if output_dir.is_relative_to(REPO_ROOT) else output_dir),
        "num_seeds": len(seeds),
        "num_candidates": len(candidates),
        "num_tasks": len(tasks),
        "workers": workers,
        "elapsed_wall_s": elapsed_wall_s,
        "best_single_candidate": candidate_summary[0] if candidate_summary else None,
        "best_per_seed_portfolio": portfolio,
        "artifacts": {
            "candidate_results_csv": "candidate_results.csv",
            "candidate_results_partial_csv": "candidate_results.partial.csv",
            "candidate_summary_csv": "candidate_summary.csv",
            "best_by_seed_csv": "best_by_seed.csv",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "analysis.md").write_text(_analysis_markdown(summary, candidate_summary, best_by_seed), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel RPG planner candidate sweeps.")
    parser.add_argument("--scenario-table", type=Path, default=SCENARIO_TABLE)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR / "candidate_sweep")
    parser.add_argument("--seeds", default="", help="Comma-separated seed filter. Defaults to all scenarios.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--preset", choices=("iter5", "portfolio3", "custom"), default="iter5")
    parser.add_argument("--terminal-nodes", default="60")
    parser.add_argument("--rate-scales", default="0.5")
    parser.add_argument("--command-smoothness", default="")
    parser.add_argument("--body-rate-smoothness", default="")
    parser.add_argument("--windows", default="")
    parser.add_argument("--plan-time-scales", default="1.0")
    parser.add_argument("--dynamics-substeps", default="1")
    parser.add_argument("--cpc-tolerance-m", type=float, default=0.1)
    parser.add_argument("--ipopt-max-iter", type=int, default=300)
    parser.add_argument("--rollout-tail-s", type=float, default=0.0)
    parser.add_argument("--post-plan-command-mode", choices=("hover", "hold_last"), default="hover")
    parser.add_argument("--constraint-violation-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress and refresh candidate_results.partial.csv every N completed tasks.",
    )
    return parser.parse_args()


def _selected_seeds(instances: list[Any], seeds_arg: str) -> list[int]:
    available = [int(instance.seed) for instance in instances]
    if not seeds_arg.strip():
        return available
    requested = [int(seed) for seed in seeds_arg.split(",") if seed.strip()]
    missing = sorted(set(requested).difference(available))
    if missing:
        raise ValueError(f"Requested seeds not found: {missing}")
    return requested


def _candidates_from_args(args: argparse.Namespace) -> list[Candidate]:
    if args.preset == "iter5":
        specs = [
            (0.5, 0.0, 0.0, 1),
            (0.5, 0.2, 0.2, 8),
            (0.5, 0.0, 0.2, 4),
            (0.5, 0.0, 0.2, 6),
            (0.5, 0.0, 0.2, 8),
            (0.5, 0.05, 0.2, 4),
            (0.5, 0.05, 0.2, 6),
            (0.5, 0.05, 0.2, 8),
            (0.5, 0.2, 0.05, 4),
            (0.5, 0.2, 0.05, 6),
            (0.5, 0.2, 0.05, 8),
        ]
        return [
            _candidate(
                rate=rate,
                command_smoothness=command_smoothness,
                body_smoothness=body_smoothness,
                window=window,
                terminal_nodes=60,
                dynamics_substeps=1,
                plan_time_scale=1.0,
                cpc_tolerance_m=args.cpc_tolerance_m,
                ipopt_max_iter=args.ipopt_max_iter,
            )
            for rate, command_smoothness, body_smoothness, window in specs
        ]

    if args.preset == "portfolio3":
        return [_candidate_from_config(candidate.name, candidate.config) for candidate in PRODUCTION_PORTFOLIO_CANDIDATES]

    candidates = []
    for terminal_nodes in _int_list(args.terminal_nodes):
        for rate in _float_list(args.rate_scales):
            for command_smoothness in _float_list(args.command_smoothness or "0.0"):
                for body_smoothness in _float_list(args.body_rate_smoothness or "0.0"):
                    for window in _int_list(args.windows or "1"):
                        for plan_time_scale in _float_list(args.plan_time_scales):
                            for dynamics_substeps in _int_list(args.dynamics_substeps):
                                candidates.append(
                                    _candidate(
                                        rate=rate,
                                        command_smoothness=command_smoothness,
                                        body_smoothness=body_smoothness,
                                        window=window,
                                        terminal_nodes=terminal_nodes,
                                        dynamics_substeps=dynamics_substeps,
                                        plan_time_scale=plan_time_scale,
                                        cpc_tolerance_m=args.cpc_tolerance_m,
                                        ipopt_max_iter=args.ipopt_max_iter,
                                    )
                                )
    return candidates


def _candidate(
    *,
    rate: float,
    command_smoothness: float,
    body_smoothness: float,
    window: int,
    terminal_nodes: int,
    dynamics_substeps: int,
    plan_time_scale: float,
    cpc_tolerance_m: float | None,
    ipopt_max_iter: int,
) -> Candidate:
    name = (
        f"n{terminal_nodes}_rate{_tag(rate)}_cmd{_tag(command_smoothness)}_"
        f"body{_tag(body_smoothness)}_win{window}_scale{_tag(plan_time_scale)}_sub{dynamics_substeps}"
    )
    return Candidate(
        name=name,
        cpc_tolerance_m=cpc_tolerance_m,
        plan_time_scale=plan_time_scale,
        motor_command_mode="zoh",
        terminal_nodes=int(terminal_nodes),
        dynamics_substeps=int(dynamics_substeps),
        planner_rate_limit_scale=float(rate),
        command_smoothness_weight=float(command_smoothness),
        body_rate_smoothness_weight=float(body_smoothness),
        terminal_capture_window_nodes=int(window),
        ipopt_max_iter=int(ipopt_max_iter),
    )


def _candidate_from_config(name: str, config: RpgTimeOptimalConfig) -> Candidate:
    return Candidate(
        name=str(name),
        cpc_tolerance_m=config.cpc_tolerance_m,
        plan_time_scale=float(config.plan_time_scale),
        motor_command_mode=str(config.motor_command_mode),
        terminal_nodes=int(config.terminal_nodes),
        dynamics_substeps=int(config.dynamics_substeps),
        planner_rate_limit_scale=float(config.planner_rate_limit_scale),
        command_smoothness_weight=float(config.command_smoothness_weight),
        body_rate_smoothness_weight=float(config.body_rate_smoothness_weight),
        terminal_capture_window_nodes=int(config.terminal_capture_window_nodes),
        ipopt_max_iter=int(config.ipopt_max_iter),
    )


def _run_tasks(
    tasks: list[tuple[Path, int, Candidate, float, str]],
    workers: int,
    *,
    output_dir: Path,
    progress_every: int,
) -> list[dict[str, Any]]:
    progress_every = max(1, int(progress_every))
    partial_path = output_dir / "candidate_results.partial.csv"
    started = time.perf_counter()
    if workers == 1:
        rows = []
        for index, task in enumerate(tasks, start=1):
            rows.append(_solve_and_score_task(task))
            if index % progress_every == 0 or index == len(tasks):
                _write_csv(partial_path, rows)
                _print_progress(index, len(tasks), started)
        return rows
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_solve_and_score_task, task) for task in tasks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            rows.append(future.result())
            if index % progress_every == 0 or index == len(tasks):
                _write_csv(partial_path, rows)
                _print_progress(index, len(tasks), started)
    return rows


def _print_progress(done: int, total: int, started: float) -> None:
    elapsed_s = time.perf_counter() - started
    rate = float(done) / max(elapsed_s, 1.0e-9)
    remaining = (float(total - done) / rate) if rate > 0.0 else math.nan
    print(
        f"completed {done}/{total} tasks in {elapsed_s:.1f}s"
        f" ({rate:.2f}/s, eta {remaining:.1f}s)",
        flush=True,
    )


def _solve_and_score_task(task: tuple[Path, int, Candidate, float, str]) -> dict[str, Any]:
    scenario_table, seed, candidate, rollout_tail_s, post_plan_command_mode = task
    instances = read_sim_instances(scenario_table)
    instance_by_seed = {int(instance.seed): instance for instance in instances}
    instance = instance_by_seed[int(seed)]
    config = candidate.config()
    started = time.perf_counter()
    try:
        planner_diag = diag_helpers._solve_planner_diagnostic(instance, config)
        rollout_metrics, _ = diag_helpers._plan_rollout_one(
            planner_diag,
            config,
            rollout_tail_s=float(rollout_tail_s),
            post_plan_command_mode=str(post_plan_command_mode),
        )
        row = {
            "candidate": candidate.name,
            "seed": int(seed),
            "error": "",
            **_candidate_fields(candidate),
            **planner_diag.row,
            **rollout_metrics,
            "task_wall_s": time.perf_counter() - started,
        }
    except Exception as exc:  # noqa: BLE001
        row = {
            "candidate": candidate.name,
            "seed": int(seed),
            "error": repr(exc),
            **_candidate_fields(candidate),
            "task_wall_s": time.perf_counter() - started,
        }
    return row


def _candidate_fields(candidate: Candidate) -> dict[str, Any]:
    fields = asdict(candidate)
    fields.pop("name")
    return {f"candidate_{key}": value for key, value in fields.items()}


def _candidate_summary(rows: list[dict[str, Any]], seed_count: int, constraint_violation_tolerance: float) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate"]), []).append(row)
    summary = []
    for candidate, candidate_rows in by_candidate.items():
        valid_rows = [row for row in candidate_rows if not row.get("error")]
        clean_rows = [row for row in valid_rows if _clean_row(row, constraint_violation_tolerance)]
        caught = [row for row in clean_rows if bool(row.get("rollout_caught_radius", False))]
        summary.append(
            {
                "candidate": candidate,
                "tasks": len(candidate_rows),
                "errors": len(candidate_rows) - len(valid_rows),
                "clean_count": len(clean_rows),
                "catch_count": len(caught),
                "catch_fraction": len(caught) / max(seed_count, 1),
                "rollout_min_distance_mean_m": _finite_mean(row.get("rollout_min_distance_m", math.nan) for row in clean_rows),
                "rollout_min_distance_max_m": _finite_max(row.get("rollout_min_distance_m", math.nan) for row in clean_rows),
                "rollout_tracking_error_mean_m": _finite_mean(row.get("rollout_position_tracking_error_mean_m", math.nan) for row in clean_rows),
                "planner_wall_s_sum": _finite_sum(row.get("planner_wall_s", math.nan) for row in valid_rows),
                "nlp_build_wall_s_sum": _finite_sum(row.get("plan_nlp_build_wall_s", math.nan) for row in valid_rows),
                "optimizer_wall_s_sum": _finite_sum(row.get("plan_optimizer_wall_s", math.nan) for row in valid_rows),
                "task_wall_s_sum": _finite_sum(row.get("task_wall_s", math.nan) for row in valid_rows),
            }
        )
    summary.sort(
        key=lambda row: (
            -int(row["catch_count"]),
            _finite_or_inf(row["rollout_min_distance_max_m"]),
            _finite_or_inf(row["rollout_min_distance_mean_m"]),
        )
    )
    return summary


def _best_by_seed(rows: list[dict[str, Any]], constraint_violation_tolerance: float) -> list[dict[str, Any]]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("error") or not _clean_row(row, constraint_violation_tolerance):
            continue
        by_seed.setdefault(int(row["seed"]), []).append(row)
    best = []
    for seed, seed_rows in sorted(by_seed.items()):
        row = min(seed_rows, key=_production_score_sort_key)
        best.append(row)
    return best


def _production_score_sort_key(row: dict[str, Any]) -> tuple[bool, int, float, float, float]:
    return (
        not _as_bool(row.get("rollout_caught_radius", False)),
        -int(float(row.get("rollout_capture_steps", 0) or 0)),
        _finite_or_inf(row.get("rollout_min_distance_m", math.inf)),
        _finite_or_inf(row.get("rollout_position_tracking_error_mean_m", math.inf)),
        _finite_or_inf(row.get("planned_total_time_s", math.inf)),
    )


def _clean_row(row: dict[str, Any], constraint_violation_tolerance: float) -> bool:
    return (
        not row.get("error")
        and _as_bool(row.get("solver_success", False))
        and _as_bool(row.get("terminal_tolerance_satisfied", False))
        and _as_bool(row.get("planned_feasible", False))
        and float(row.get("constraint_violation_max", math.inf)) <= float(constraint_violation_tolerance)
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _analysis_markdown(
    summary: dict[str, Any],
    candidate_summary: list[dict[str, Any]],
    best_by_seed: list[dict[str, Any]],
) -> str:
    lines = [
        "# RPG Candidate Sweep",
        "",
        f"Tasks: {summary['num_tasks']} across {summary['workers']} workers",
        f"Elapsed wall time: {summary['elapsed_wall_s']:.1f}s",
        "",
        "## Best Single Candidate",
        "",
    ]
    best = summary["best_single_candidate"]
    if best:
        lines.extend(
            [
                f"- Candidate: `{best['candidate']}`",
                f"- Catch count: {best['catch_count']}/{summary['num_seeds']}",
                f"- Worst min distance: {best['rollout_min_distance_max_m']:.3f} m",
                f"- Mean min distance: {best['rollout_min_distance_mean_m']:.3f} m",
                "",
            ]
        )
    portfolio = summary["best_per_seed_portfolio"]
    lines.extend(
        [
            "## Best Per-Seed Portfolio",
            "",
            f"- Catch count: {portfolio['catch_count']}/{summary['num_seeds']}",
            f"- Worst best min distance: {portfolio['max_best_min_distance_m']:.3f} m",
            "",
            "## Per-Seed Choices",
            "",
        ]
    )
    for row in best_by_seed:
        lines.append(
            f"- Seed {int(row['seed'])}: `{row['candidate']}`, min distance {float(row['rollout_min_distance_m']):.3f} m"
        )
    lines.extend(["", "## Candidate Ranking", ""])
    for row in candidate_summary[:20]:
        lines.append(
            f"- `{row['candidate']}`: {int(row['catch_count'])}/{summary['num_seeds']}, "
            f"worst {float(row['rollout_min_distance_max_m']):.3f} m, "
            f"mean {float(row['rollout_min_distance_mean_m']):.3f} m"
        )
    lines.append("")
    return "\n".join(lines)


def _resolve_workers(workers: int | None, task_count: int) -> int:
    if workers is not None:
        return max(1, int(workers))
    cpu_count = os.cpu_count() or 1
    return max(1, min(task_count, cpu_count - 1 if cpu_count > 1 else 1))


def _float_list(value: str) -> list[float]:
    return [float(item) for item in str(value).split(",") if item.strip()]


def _int_list(value: str) -> list[int]:
    return [int(item) for item in str(value).split(",") if item.strip()]


def _tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _finite_values(values: Any) -> list[float]:
    vals = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            vals.append(number)
    return vals


def _finite_mean(values: Any) -> float:
    vals = _finite_values(values)
    return float(np.mean(vals)) if vals else math.nan


def _finite_max(values: Any) -> float:
    vals = _finite_values(values)
    return float(np.max(vals)) if vals else math.nan


def _finite_sum(values: Any) -> float:
    vals = _finite_values(values)
    return float(np.sum(vals)) if vals else math.nan


def _finite_or_inf(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.inf
    return number if np.isfinite(number) else math.inf


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
