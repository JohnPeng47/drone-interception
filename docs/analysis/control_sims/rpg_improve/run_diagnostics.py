from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backends.csim.bindings import BatchPufferSimEngineBackend
from backends.csim.generator.instance_store import read_sim_instances
from backends.csim.runner import MotorSpeedCommandBatch, SimControlPolicy, SimRunResult, SimRunner, SimRunnerState, SimRunnerStep
from control_sims.rpg_time_optimal.config import RpgTimeOptimalConfig
from control_sims.rpg_time_optimal.planner import RpgTimeOptimalPlan, RpgTimeOptimalPlanner
from control_sims.rpg_time_optimal.policy import RpgTimeOptimalControlPolicy
from control_sims.runner import _row_from_completed, _scenario_fields_from_instance, _summarize_subset


OUT_DIR = Path(__file__).resolve().parent
SCENARIO_TABLE = REPO_ROOT / "scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin"


@dataclass(frozen=True)
class PlannerDiagnostic:
    seed: int
    instance: Any
    plan: RpgTimeOptimalPlan
    row: dict[str, Any]


def main() -> int:
    global OUT_DIR
    args = _parse_args()
    OUT_DIR = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    instances = read_sim_instances(SCENARIO_TABLE)
    if args.seeds:
        requested = {int(seed) for seed in args.seeds.split(",") if seed.strip()}
        instances = [instance for instance in instances if int(instance.seed) in requested]
    config = RpgTimeOptimalConfig(
        cpc_tolerance_m=args.cpc_tolerance_m,
        plan_time_scale=args.plan_time_scale,
        motor_command_mode=args.motor_command_mode,
        terminal_nodes=args.terminal_nodes,
        dynamics_substeps=args.dynamics_substeps,
        planner_rate_limit_scale=args.planner_rate_limit_scale,
        command_smoothness_weight=args.command_smoothness_weight,
        body_rate_smoothness_weight=args.body_rate_smoothness_weight,
        terminal_capture_window_nodes=args.terminal_capture_window_nodes,
        ipopt_max_iter=args.ipopt_max_iter,
    )

    start = time.perf_counter()
    planner_workers = _resolve_workers(args.planner_workers, len(instances))
    sim_max_envs = _resolve_max_envs(args.max_envs, len(instances))
    planner_diags = _solve_planner_diagnostics(instances, config, planner_workers)
    _write_csv(OUT_DIR / "planner_metrics.csv", [diag.row for diag in planner_diags])
    _write_csv(OUT_DIR / "planned_trajectories.csv", _planned_trajectory_rows(planner_diags))
    rollout_metrics, rollout_rows = _plan_rollout_diagnostics(
        planner_diags,
        config,
        rollout_tail_s=float(args.rollout_tail_s),
        post_plan_command_mode=str(args.post_plan_command_mode),
        workers=planner_workers,
    )
    _write_csv(OUT_DIR / "plan_rollout_metrics.csv", rollout_metrics)
    _write_csv(OUT_DIR / "plan_rollout_trajectories.csv", rollout_rows)

    requested_policies = {item.strip() for item in args.policies.split(",") if item.strip()}
    policy_runs = []
    if "ctbr" in requested_policies:
        policy_runs.append(
            _run_policy(
                "rpg_time_optimal_ctbr",
                _CachedCtbrPolicy(config, planner_diags),
                instances,
                planner_diags,
                max_envs=sim_max_envs,
            )
        )
    if "motor" in requested_policies:
        policy_runs.append(
            _run_policy(
                "rpg_time_optimal_motor_feedforward",
                _CachedMotorFeedforwardPolicy(config, planner_diags),
                instances,
                planner_diags,
                max_envs=sim_max_envs,
            )
        )
    execution_rows = [row for run in policy_runs for row in run["execution_rows"]]
    classification_rows = [row for run in policy_runs for row in run["classification_rows"]]
    actual_rows = [row for run in policy_runs for row in run["actual_rows"]]

    _write_csv(OUT_DIR / "execution_metrics.csv", execution_rows)
    _write_csv(OUT_DIR / "failure_classification.csv", classification_rows)
    _write_csv(OUT_DIR / "actual_trajectories.csv", actual_rows)

    for run in policy_runs:
        _write_policy_plots(run["policy"], planner_diags, run["actual_by_seed"])

    summary = _summary_payload(
        planner_diags,
        policy_runs,
        rollout_metrics=rollout_metrics,
        config=config,
        rollout_tail_s=float(args.rollout_tail_s),
        post_plan_command_mode=str(args.post_plan_command_mode),
        elapsed_wall_s=time.perf_counter() - start,
        planner_workers=planner_workers,
        sim_max_envs=sim_max_envs,
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT_DIR / "analysis.md").write_text(_analysis_markdown(summary, classification_rows), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RPG time-optimal planner/execution diagnostics.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for generated diagnostics artifacts.",
    )
    parser.add_argument(
        "--cpc-tolerance-m",
        type=float,
        default=None,
        help="Planner terminal tolerance. Defaults to each scenario intercept radius.",
    )
    parser.add_argument(
        "--plan-time-scale",
        type=float,
        default=1.0,
        help="Execution time scale for sampling plans. Values above 1 execute more slowly.",
    )
    parser.add_argument(
        "--motor-command-mode",
        choices=("zoh", "linear"),
        default="zoh",
        help="How to sample planned motor RPM commands during execution diagnostics.",
    )
    parser.add_argument(
        "--terminal-nodes",
        type=int,
        default=30,
        help="Number of planner terminal OCP nodes.",
    )
    parser.add_argument(
        "--dynamics-substeps",
        type=int,
        default=1,
        help="RK4/clamp microsteps inside each planner node interval.",
    )
    parser.add_argument(
        "--planner-rate-limit-scale",
        type=float,
        default=1.0,
        help="Planner-only body-rate component limit multiplier in (0, 1].",
    )
    parser.add_argument(
        "--command-smoothness-weight",
        type=float,
        default=0.0,
        help="OCP objective weight for squared normalized motor command changes.",
    )
    parser.add_argument(
        "--body-rate-smoothness-weight",
        type=float,
        default=0.0,
        help="OCP objective weight for squared normalized body-rate changes.",
    )
    parser.add_argument(
        "--terminal-capture-window-nodes",
        type=int,
        default=1,
        help="Number of final planner nodes constrained inside the scenario capture radius.",
    )
    parser.add_argument(
        "--ipopt-max-iter",
        type=int,
        default=100,
        help="Maximum IPOPT iterations per planner solve.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated seed filter for quick diagnostics.",
    )
    parser.add_argument(
        "--policies",
        default="ctbr,motor",
        help="Comma-separated policy set: ctbr,motor. Use motor for faster hard-seed sweeps.",
    )
    parser.add_argument(
        "--rollout-tail-s",
        type=float,
        default=0.0,
        help="Extra direct-plan rollout time after the planned horizon.",
    )
    parser.add_argument(
        "--post-plan-command-mode",
        choices=("hover", "hold_last"),
        default="hover",
        help="Direct-rollout motor command after the planned horizon.",
    )
    parser.add_argument(
        "--planner-workers",
        type=int,
        default=None,
        help="Parallel worker processes for planner solves and direct rollouts. Defaults to available CPUs.",
    )
    parser.add_argument(
        "--max-envs",
        type=int,
        default=None,
        help="SimRunner batch width for policy execution. Defaults to available CPUs/scenario count.",
    )
    return parser.parse_args()


class _CachedMotorFeedforwardPolicy(SimControlPolicy):
    def __init__(self, config: RpgTimeOptimalConfig, planner_diags: list[PlannerDiagnostic]):
        self.config = config
        self._plans_by_seed = {int(diag.seed): diag.plan for diag in planner_diags}

    def command(self, state: SimRunnerState) -> MotorSpeedCommandBatch:
        motor_speeds_rpm = np.zeros((len(state.instances), 4), dtype=np.float32)
        for slot, instance in enumerate(state.instances):
            if instance is None or not bool(state.active[slot]):
                continue
            plan = self._plans_by_seed[int(instance.seed)]
            motor_speeds_rpm[slot] = _planned_motor_command(instance, plan, float(state.elapsed_s[slot]), self.config)
        return MotorSpeedCommandBatch(motor_speeds_rpm=motor_speeds_rpm)


class _CachedCtbrPolicy(RpgTimeOptimalControlPolicy):
    def __init__(self, config: RpgTimeOptimalConfig, planner_diags: list[PlannerDiagnostic]):
        super().__init__(config)
        self._plans_by_seed = {int(diag.seed): diag.plan for diag in planner_diags}

    def on_slots_started(self, slots: np.ndarray, instances, state: SimRunnerState) -> None:
        for slot in np.asarray(slots, dtype=np.int64).reshape(-1):
            slot_i = int(slot)
            instance = state.instances[slot_i]
            if instance is None:
                continue
            self._slots[slot_i] = self._plans_by_seed[int(instance.seed)]


def _solve_planner_diagnostics(
    instances: list[Any],
    config: RpgTimeOptimalConfig,
    workers: int,
) -> list[PlannerDiagnostic]:
    if workers == 1:
        return [_solve_planner_diagnostic(instance, config) for instance in instances]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_solve_planner_diagnostic, instance, config) for instance in instances]
        return [future.result() for future in futures]


def _solve_planner_diagnostic(instance: Any, config: RpgTimeOptimalConfig) -> PlannerDiagnostic:
    started = time.perf_counter()
    plan = RpgTimeOptimalPlanner(config).solve(instance)
    solve_wall_s = time.perf_counter() - started
    target_positions = _target_positions(instance, plan.t_x_s)
    distances = np.linalg.norm(plan.position_w - target_positions, axis=0)
    min_index = int(np.argmin(distances))
    terminal_distance = float(distances[-1])
    intercept_radius = float(instance.config.intercept_radius_m)
    requested_tolerance = (
        float(config.cpc_tolerance_m)
        if config.cpc_tolerance_m is not None
        else intercept_radius
    )
    max_rate = _max_rate_rps(instance)
    max_thrust = _max_collective_thrust_n(instance)
    max_rpm = float(instance.config.pursuer.max_rpm)
    planned_total_thrust = np.sum(plan.motor_thrusts_n, axis=0)
    max_body_rate_component = float(np.max(np.abs(plan.body_rates_b)))
    max_motor_speed = (
        float(np.max(plan.motor_speed_commands_rpm))
        if plan.motor_speed_commands_rpm is not None
        else math.nan
    )
    smoothness = _plan_smoothness_metrics(plan, distances, intercept_radius)
    row = {
        "seed": int(instance.seed),
        "solver_status": str(plan.solver_status),
        "solver_success": bool(plan.solver_success),
        "constraint_violation_max": float(plan.constraint_violation_max),
        "requested_terminal_tolerance_m": requested_tolerance,
        "terminal_tolerance_satisfied": bool(terminal_distance <= requested_tolerance + 1.0e-6),
        "planned_feasible": bool(float(np.min(distances)) <= intercept_radius),
        "intercept_radius_m": intercept_radius,
        "planned_min_distance_m": float(np.min(distances)),
        "planned_min_distance_time_s": float(plan.t_x_s[min_index]),
        "planned_terminal_distance_m": terminal_distance,
        "planned_total_time_s": float(plan.total_time_s),
        "planner_wall_s": float(solve_wall_s),
        "plan_reported_solve_wall_s": float(plan.solve_wall_s),
        "plan_nlp_build_wall_s": float(plan.nlp_build_wall_s),
        "plan_optimizer_wall_s": float(plan.optimizer_wall_s),
        "planned_max_body_rate_rps": float(np.max(np.linalg.norm(plan.body_rates_b, axis=0))),
        "planned_max_body_rate_component_rps": max_body_rate_component,
        "body_rate_component_violation_rps": max(max_body_rate_component - max_rate, 0.0),
        "planned_max_motor_speed_rpm": max_motor_speed,
        "motor_speed_violation_rpm": max(max_motor_speed - max_rpm, 0.0) if np.isfinite(max_motor_speed) else math.nan,
        "planned_max_total_thrust_n": float(np.max(planned_total_thrust)),
        "total_thrust_violation_n": max(float(np.max(planned_total_thrust)) - max_thrust, 0.0),
        **smoothness,
    }
    row.update(_scenario_fields_from_instance(instance))
    return PlannerDiagnostic(seed=int(instance.seed), instance=instance, plan=plan, row=row)


def _plan_smoothness_metrics(
    plan: RpgTimeOptimalPlan,
    distances: np.ndarray,
    intercept_radius: float,
) -> dict[str, float | int]:
    dt_nodes = np.diff(np.asarray(plan.t_x_s, dtype=float))
    dt_min = float(np.min(dt_nodes)) if len(dt_nodes) else math.nan
    body_rate_step = np.diff(np.asarray(plan.body_rates_b, dtype=float), axis=1)
    body_rate_step_norm = np.linalg.norm(body_rate_step, axis=0) if body_rate_step.size else np.array([], dtype=float)
    thrust_step = np.diff(np.sum(np.asarray(plan.motor_thrusts_n, dtype=float), axis=0))
    if plan.motor_speed_commands_rpm is not None and plan.motor_speed_commands_rpm.shape[1] > 1:
        command_step = np.diff(np.asarray(plan.motor_speed_commands_rpm, dtype=float), axis=1)
        command_step_norm = np.linalg.norm(command_step, axis=0)
    else:
        command_step_norm = np.array([], dtype=float)
    terminal_capture_nodes = 0
    for value in reversed(np.asarray(distances, dtype=float).reshape(-1)):
        if float(value) <= float(intercept_radius):
            terminal_capture_nodes += 1
        else:
            break
    return {
        "node_dt_min_s": dt_min,
        "terminal_capture_nodes": int(terminal_capture_nodes),
        "terminal_capture_duration_s": float(terminal_capture_nodes * dt_min) if np.isfinite(dt_min) else math.nan,
        "body_rate_step_norm_max_rps": _finite_max(body_rate_step_norm),
        "body_rate_step_norm_mean_rps": _finite_mean(body_rate_step_norm),
        "motor_command_step_norm_max_rpm": _finite_max(command_step_norm),
        "motor_command_step_norm_mean_rpm": _finite_mean(command_step_norm),
        "total_thrust_step_max_n": _finite_max(np.abs(thrust_step)),
        "total_thrust_step_mean_n": _finite_mean(np.abs(thrust_step)),
    }


def _run_policy(
    policy_name: str,
    policy: Any,
    instances: list[Any],
    planner_diags: list[PlannerDiagnostic],
    *,
    max_envs: int,
) -> dict[str, Any]:
    runner = SimRunner(max_envs=max_envs)
    started = time.perf_counter()
    result = runner.run(instances, policy)
    elapsed_wall_s = time.perf_counter() - started
    plan_by_seed = {diag.seed: diag for diag in planner_diags}

    execution_rows = []
    for completed in result.completed:
        row = _row_from_completed(policy_name, completed, result.steps)
        row.update(_scenario_fields_from_instance(completed.instance))
        row["policy"] = policy_name
        row["wall_s"] = elapsed_wall_s / max(len(result.completed), 1)
        row["error"] = None
        execution_rows.append(row)

    actual_by_seed = _actual_trajectories_by_seed(policy_name, result, plan_by_seed)
    actual_rows = [row for rows in actual_by_seed.values() for row in rows]
    classification_rows = [
        _classify_execution(row, plan_by_seed[int(row["seed"])], actual_by_seed.get(int(row["seed"]), []))
        for row in execution_rows
    ]

    return {
        "policy": policy_name,
        "result": result,
        "elapsed_wall_s": elapsed_wall_s,
        "execution_rows": execution_rows,
        "classification_rows": classification_rows,
        "actual_rows": actual_rows,
        "actual_by_seed": actual_by_seed,
    }


def _actual_trajectories_by_seed(
    policy_name: str,
    result: SimRunResult,
    plan_by_seed: dict[int, PlannerDiagnostic],
) -> dict[int, list[dict[str, Any]]]:
    completed_by_workload = {int(item.workload_index): item for item in result.completed}
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    for step in result.steps:
        for slot in range(len(step.state.instances)):
            workload_index = int(step.state.workload_indices[slot])
            if workload_index < 0 or workload_index not in completed_by_workload:
                continue
            if not bool(step.state.active[slot]):
                continue
            completed = completed_by_workload[workload_index]
            snapshot = step.state.snapshot[slot]
            seed = int(completed.seed)
            plan_diag = plan_by_seed[seed]
            t_s = float(step.state.elapsed_s[slot])
            planned_position = _interp_columns(plan_diag.plan.t_x_s, plan_diag.plan.position_w, t_s)
            planned_velocity = _interp_columns(plan_diag.plan.t_x_s, plan_diag.plan.velocity_w, t_s)
            actual_position = np.asarray(snapshot.pursuer.position_w, dtype=float).reshape(3)
            actual_velocity = np.asarray(snapshot.pursuer.velocity_w, dtype=float).reshape(3)
            target_position = np.asarray(snapshot.target.position_w, dtype=float).reshape(3)
            target_velocity = np.asarray(snapshot.target.velocity_w, dtype=float).reshape(3)
            command = _command_fields(step, slot)
            row = {
                "policy": policy_name,
                "seed": seed,
                "tick": int(step.state.steps[slot]),
                "t_s": t_s,
                "actual_x_w_m": float(actual_position[0]),
                "actual_y_w_m": float(actual_position[1]),
                "actual_z_w_m": float(actual_position[2]),
                "target_x_w_m": float(target_position[0]),
                "target_y_w_m": float(target_position[1]),
                "target_z_w_m": float(target_position[2]),
                "planned_x_w_m": float(planned_position[0]),
                "planned_y_w_m": float(planned_position[1]),
                "planned_z_w_m": float(planned_position[2]),
                "actual_target_distance_m": float(np.linalg.norm(actual_position - target_position)),
                "planned_target_distance_m": float(np.linalg.norm(planned_position - target_position)),
                "position_tracking_error_m": float(np.linalg.norm(actual_position - planned_position)),
                "velocity_tracking_error_mps": float(np.linalg.norm(actual_velocity - planned_velocity)),
                "relative_speed_mps": float(np.linalg.norm(actual_velocity - target_velocity)),
                "detected": bool(snapshot.camera.detected),
                **command,
            }
            rows_by_seed.setdefault(seed, []).append(row)
    return rows_by_seed


def _classify_execution(
    execution_row: dict[str, Any],
    planner_diag: PlannerDiagnostic,
    actual_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    seed = int(execution_row["seed"])
    policy = str(execution_row["policy"])
    planned_min = float(planner_diag.row["planned_min_distance_m"])
    actual_min = float(execution_row["min_distance_m"])
    intercept_radius = float(planner_diag.row["intercept_radius_m"])
    caught = _as_bool(execution_row["caught"])
    tracking_errors = [float(row["position_tracking_error_m"]) for row in actual_rows]
    max_tracking = max(tracking_errors) if tracking_errors else math.nan
    mean_tracking = float(np.mean(tracking_errors)) if tracking_errors else math.nan
    planned_feasible = planned_min <= intercept_radius
    actual_reached_capture_radius = actual_min <= intercept_radius
    if caught:
        classification = "caught"
    elif not planned_feasible:
        classification = "planner_ideal_misses"
    elif not actual_reached_capture_radius:
        classification = "execution_tracking_or_model_mismatch"
    else:
        classification = "capture_condition_or_timing_mismatch"
    return {
        "policy": policy,
        "seed": seed,
        "classification": classification,
        "caught": caught,
        "intercept_radius_m": intercept_radius,
        "planned_feasible": planned_feasible,
        "planned_min_distance_m": planned_min,
        "actual_min_distance_m": actual_min,
        "actual_reached_capture_radius": actual_reached_capture_radius,
        "position_tracking_error_mean_m": mean_tracking,
        "position_tracking_error_max_m": max_tracking,
        "visible_fraction": float(execution_row["visible_fraction"]),
        "final_distance_m": float(execution_row["final_distance_m"]),
    }


def _planned_trajectory_rows(planner_diags: list[PlannerDiagnostic]) -> list[dict[str, Any]]:
    rows = []
    for diag in planner_diags:
        target_positions = _target_positions(diag.instance, diag.plan.t_x_s)
        for index, t_s in enumerate(diag.plan.t_x_s):
            position = diag.plan.position_w[:, index]
            target = target_positions[:, index]
            rows.append(
                {
                    "seed": int(diag.seed),
                    "node": int(index),
                    "t_s": float(t_s),
                    "planned_x_w_m": float(position[0]),
                    "planned_y_w_m": float(position[1]),
                    "planned_z_w_m": float(position[2]),
                    "target_x_w_m": float(target[0]),
                    "target_y_w_m": float(target[1]),
                    "target_z_w_m": float(target[2]),
                    "planned_target_distance_m": float(np.linalg.norm(position - target)),
                }
            )
    return rows


def _plan_rollout_diagnostics(
    planner_diags: list[PlannerDiagnostic],
    config: RpgTimeOptimalConfig,
    *,
    rollout_tail_s: float,
    post_plan_command_mode: str,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = [(diag, config, float(rollout_tail_s), str(post_plan_command_mode)) for diag in planner_diags]
    if workers == 1:
        results = [_plan_rollout_task(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_plan_rollout_task, task) for task in tasks]
            results = [future.result() for future in futures]
    metrics = []
    rows = []
    for diag_metrics, diag_rows in results:
        metrics.append(diag_metrics)
        rows.extend(diag_rows)
    return metrics, rows


def _plan_rollout_task(
    task: tuple[PlannerDiagnostic, RpgTimeOptimalConfig, float, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    planner_diag, config, rollout_tail_s, post_plan_command_mode = task
    return _plan_rollout_one(
        planner_diag,
        config,
        rollout_tail_s=rollout_tail_s,
        post_plan_command_mode=post_plan_command_mode,
    )


def _plan_rollout_one(
    diag: PlannerDiagnostic,
    config: RpgTimeOptimalConfig,
    *,
    rollout_tail_s: float,
    post_plan_command_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    instance = diag.instance
    plan = diag.plan
    assert instance.config is not None
    backend = BatchPufferSimEngineBackend(1)
    snapshots = backend.reset_many(np.array([0], dtype=np.int64), (instance,))
    dt = float(instance.config.options.backend_dt) * max(1, int(instance.config.options.action_substeps))
    elapsed_s = 0.0
    rows = []
    step_index = 0
    rollout_horizon_s = float(plan.total_time_s) * float(config.plan_time_scale) + max(float(rollout_tail_s), 0.0)
    max_steps = int(math.ceil(rollout_horizon_s / max(dt, 1.0e-9))) + 1
    while step_index <= max_steps:
        snapshot = snapshots[0]
        row = _plan_rollout_row(diag, snapshot, elapsed_s, step_index, config)
        rows.append(row)
        if elapsed_s >= rollout_horizon_s:
            break
        command = _planned_motor_command(
            instance,
            plan,
            elapsed_s,
            config,
            post_plan_command_mode=post_plan_command_mode,
        )
        snapshots = backend.step_motor_speeds_many(command.reshape(1, 4))
        elapsed_s += dt
        step_index += 1

    min_distance = min(float(row["rollout_target_distance_m"]) for row in rows)
    final = rows[-1]
    tracking_errors = [float(row["position_tracking_error_m"]) for row in rows]
    rpm_errors = [float(row["rpm_tracking_error_rpm"]) for row in rows]
    body_rate_errors = [float(row["body_rate_tracking_error_rps"]) for row in rows]
    capture_steps = sum(
        1
        for row in rows
        if float(row["rollout_target_distance_m"]) <= float(instance.config.intercept_radius_m)
    )
    max_consecutive_capture_steps = _max_consecutive_capture_steps(
        rows,
        float(instance.config.intercept_radius_m),
    )
    return (
        {
            "seed": int(diag.seed),
            "rollout_steps": len(rows),
            "rollout_final_time_s": float(rows[-1]["t_s"]),
            "rollout_min_distance_m": min_distance,
            "rollout_final_distance_m": float(final["rollout_target_distance_m"]),
            "rollout_caught_radius": bool(min_distance <= float(instance.config.intercept_radius_m)),
            "rollout_capture_steps": int(capture_steps),
            "rollout_max_consecutive_capture_steps": int(max_consecutive_capture_steps),
            "rollout_position_tracking_error_mean_m": float(np.mean(tracking_errors)),
            "rollout_position_tracking_error_max_m": float(np.max(tracking_errors)),
            "rollout_rpm_tracking_error_mean_rpm": float(np.mean(rpm_errors)),
            "rollout_rpm_tracking_error_max_rpm": float(np.max(rpm_errors)),
            "rollout_body_rate_tracking_error_mean_rps": float(np.mean(body_rate_errors)),
            "rollout_body_rate_tracking_error_max_rps": float(np.max(body_rate_errors)),
        },
        rows,
    )


def _max_consecutive_capture_steps(rows: list[dict[str, Any]], intercept_radius_m: float) -> int:
    current = 0
    best = 0
    for row in rows:
        if float(row["rollout_target_distance_m"]) <= float(intercept_radius_m):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _plan_rollout_row(
    diag: PlannerDiagnostic,
    snapshot: Any,
    elapsed_s: float,
    step_index: int,
    config: RpgTimeOptimalConfig,
) -> dict[str, Any]:
    plan = diag.plan
    plan_time_s = float(elapsed_s) / max(float(config.plan_time_scale), 1.0e-9)
    planned_position = _interp_columns(plan.t_x_s, plan.position_w, plan_time_s)
    planned_velocity = _interp_columns(plan.t_x_s, plan.velocity_w, plan_time_s)
    planned_rpm = _interp_columns(plan.t_x_s, _actual_motor_rpm_from_plan(plan), plan_time_s)
    planned_body_rates = _interp_columns(plan.t_x_s, plan.body_rates_b, plan_time_s)
    actual_position = np.asarray(snapshot.pursuer.position_w, dtype=float).reshape(3)
    actual_velocity = np.asarray(snapshot.pursuer.velocity_w, dtype=float).reshape(3)
    actual_body_rates = np.asarray(snapshot.pursuer.body_rates_b, dtype=float).reshape(3)
    actual_rpm = np.asarray(snapshot.pursuer.rotor_speeds, dtype=float).reshape(4)
    target_position = np.asarray(snapshot.target.position_w, dtype=float).reshape(3)
    return {
        "seed": int(diag.seed),
        "step": int(step_index),
        "t_s": float(elapsed_s),
        "plan_t_s": float(plan_time_s),
        "rollout_x_w_m": float(actual_position[0]),
        "rollout_y_w_m": float(actual_position[1]),
        "rollout_z_w_m": float(actual_position[2]),
        "target_x_w_m": float(target_position[0]),
        "target_y_w_m": float(target_position[1]),
        "target_z_w_m": float(target_position[2]),
        "planned_x_w_m": float(planned_position[0]),
        "planned_y_w_m": float(planned_position[1]),
        "planned_z_w_m": float(planned_position[2]),
        "rollout_target_distance_m": float(np.linalg.norm(actual_position - target_position)),
        "planned_target_distance_m": float(np.linalg.norm(planned_position - target_position)),
        "position_tracking_error_m": float(np.linalg.norm(actual_position - planned_position)),
        "velocity_tracking_error_mps": float(np.linalg.norm(actual_velocity - planned_velocity)),
        "body_rate_tracking_error_rps": float(np.linalg.norm(actual_body_rates - planned_body_rates)),
        "rpm_tracking_error_rpm": float(np.linalg.norm(actual_rpm - planned_rpm)),
    }


def _planned_motor_command(
    instance: Any,
    plan: RpgTimeOptimalPlan,
    elapsed_s: float,
    config: RpgTimeOptimalConfig,
    *,
    post_plan_command_mode: str = "hover",
) -> np.ndarray:
    assert instance.config is not None
    if plan.motor_speed_commands_rpm is None or plan.motor_speed_commands_rpm.size == 0:
        return np.full(4, _hover_rpm(instance.config.pursuer), dtype=np.float32)
    plan_time_s = float(elapsed_s) / max(float(config.plan_time_scale), 1.0e-9)
    if plan_time_s > float(plan.total_time_s):
        if post_plan_command_mode == "hold_last":
            return np.asarray(plan.motor_speed_commands_rpm[:, -1], dtype=np.float32).reshape(4)
        return np.full(4, _hover_rpm(instance.config.pursuer), dtype=np.float32)
    if config.motor_command_mode == "linear":
        t_ref = np.append(np.asarray(plan.t_u_s, dtype=float), float(plan.total_time_s))
        values = np.column_stack((plan.motor_speed_commands_rpm, plan.motor_speed_commands_rpm[:, -1]))
        return np.array(
            [np.interp(max(plan_time_s, 0.0), t_ref, values[row]) for row in range(4)],
            dtype=np.float32,
        )
    index = int(np.searchsorted(plan.t_u_s, max(plan_time_s, 0.0), side="right") - 1)
    index = int(np.clip(index, 0, plan.motor_speed_commands_rpm.shape[1] - 1))
    return np.asarray(plan.motor_speed_commands_rpm[:, index], dtype=np.float32).reshape(4)


def _actual_motor_rpm_from_plan(plan: RpgTimeOptimalPlan) -> np.ndarray:
    return np.asarray(plan.motor_speeds_rpm, dtype=float)


def _write_policy_plots(
    policy_name: str,
    planner_diags: list[PlannerDiagnostic],
    actual_by_seed: dict[int, list[dict[str, Any]]],
) -> None:
    for diag in planner_diags:
        rows = actual_by_seed.get(diag.seed, [])
        if not rows:
            continue
        _plot_trajectory(policy_name, diag, rows)
        _plot_distance(policy_name, diag, rows)


def _plot_trajectory(policy_name: str, diag: PlannerDiagnostic, rows: list[dict[str, Any]]) -> None:
    actual = np.array([[row["actual_x_w_m"], row["actual_y_w_m"], row["actual_z_w_m"]] for row in rows], dtype=float)
    target = np.array([[row["target_x_w_m"], row["target_y_w_m"], row["target_z_w_m"]] for row in rows], dtype=float)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(diag.plan.position_w[0], diag.plan.position_w[1], diag.plan.position_w[2], label="planned", linewidth=2)
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], label="actual", linewidth=1.5)
    ax.plot(target[:, 0], target[:, 1], target[:, 2], label="target", linewidth=1.5)
    ax.scatter(actual[0, 0], actual[0, 1], actual[0, 2], label="start", s=20)
    ax.set_title(f"{policy_name} seed {diag.seed}")
    ax.set_xlabel("x_w m")
    ax.set_ylabel("y_w m")
    ax.set_zlabel("z_w m")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plots" / f"{policy_name}_seed_{diag.seed:04d}_trajectory_3d.png", dpi=150)
    plt.close(fig)


def _plot_distance(policy_name: str, diag: PlannerDiagnostic, rows: list[dict[str, Any]]) -> None:
    t_actual = np.array([row["t_s"] for row in rows], dtype=float)
    actual_dist = np.array([row["actual_target_distance_m"] for row in rows], dtype=float)
    tracking = np.array([row["position_tracking_error_m"] for row in rows], dtype=float)
    target_positions = _target_positions(diag.instance, diag.plan.t_x_s)
    planned_dist = np.linalg.norm(diag.plan.position_w - target_positions, axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(diag.plan.t_x_s, planned_dist, label="planned distance", linewidth=2)
    ax.plot(t_actual, actual_dist, label="actual distance", linewidth=1.5)
    ax.plot(t_actual, tracking, label="tracking error", linewidth=1.2)
    ax.axhline(float(diag.instance.config.intercept_radius_m), color="black", linestyle="--", linewidth=1, label="capture radius")
    ax.set_title(f"{policy_name} seed {diag.seed}")
    ax.set_xlabel("time s")
    ax.set_ylabel("meters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plots" / f"{policy_name}_seed_{diag.seed:04d}_distance.png", dpi=150)
    plt.close(fig)


def _summary_payload(
    planner_diags: list[PlannerDiagnostic],
    policy_runs: list[dict[str, Any]],
    *,
    rollout_metrics: list[dict[str, Any]],
    config: RpgTimeOptimalConfig,
    rollout_tail_s: float,
    post_plan_command_mode: str,
    elapsed_wall_s: float,
    planner_workers: int,
    sim_max_envs: int,
) -> dict[str, Any]:
    planner_rows = [diag.row for diag in planner_diags]
    policies = {}
    for run in policy_runs:
        policies[run["policy"]] = {
            "elapsed_wall_s": float(run["elapsed_wall_s"]),
            "summary": _summarize_subset(run["execution_rows"]),
            "classification_counts": _count_by(run["classification_rows"], "classification"),
            "tracking_error_mean_m": _mean(row["position_tracking_error_mean_m"] for row in run["classification_rows"]),
            "tracking_error_max_m": _max(row["position_tracking_error_max_m"] for row in run["classification_rows"]),
        }
    return {
        "scenario_table": str(SCENARIO_TABLE.relative_to(REPO_ROOT)),
        "output_dir": str(OUT_DIR.relative_to(REPO_ROOT)),
        "num_scenarios": len(planner_diags),
        "elapsed_wall_s": float(elapsed_wall_s),
        "planner_workers": int(planner_workers),
        "sim_max_envs": int(sim_max_envs),
        "planner_config": {
            "cpc_tolerance_m": None if config.cpc_tolerance_m is None else float(config.cpc_tolerance_m),
            "plan_time_scale": float(config.plan_time_scale),
            "motor_command_mode": str(config.motor_command_mode),
            "terminal_nodes": int(config.terminal_nodes),
            "dynamics_substeps": int(config.dynamics_substeps),
            "planner_rate_limit_scale": float(config.planner_rate_limit_scale),
            "command_smoothness_weight": float(config.command_smoothness_weight),
            "body_rate_smoothness_weight": float(config.body_rate_smoothness_weight),
            "terminal_capture_window_nodes": int(config.terminal_capture_window_nodes),
            "ipopt_max_iter": int(config.ipopt_max_iter),
            "rollout_tail_s": float(rollout_tail_s),
            "post_plan_command_mode": str(post_plan_command_mode),
        },
        "planner": {
            "ideal_feasible_fraction": _mean(1.0 if row["planned_feasible"] else 0.0 for row in planner_rows),
            "terminal_tolerance_satisfied_fraction": _mean(
                1.0 if row["terminal_tolerance_satisfied"] else 0.0 for row in planner_rows
            ),
            "planned_min_distance_mean_m": _mean(row["planned_min_distance_m"] for row in planner_rows),
            "planned_min_distance_max_m": _max(row["planned_min_distance_m"] for row in planner_rows),
            "planned_terminal_distance_mean_m": _mean(row["planned_terminal_distance_m"] for row in planner_rows),
            "constraint_violation_max": _max(row["constraint_violation_max"] for row in planner_rows),
            "planner_wall_s_mean": _mean(row["planner_wall_s"] for row in planner_rows),
        },
        "plan_rollout": {
            "rollout_catch_fraction": _mean(1.0 if row["rollout_caught_radius"] else 0.0 for row in rollout_metrics),
            "rollout_min_distance_mean_m": _mean(row["rollout_min_distance_m"] for row in rollout_metrics),
            "rollout_min_distance_max_m": _max(row["rollout_min_distance_m"] for row in rollout_metrics),
            "rollout_position_tracking_error_mean_m": _mean(
                row["rollout_position_tracking_error_mean_m"] for row in rollout_metrics
            ),
            "rollout_position_tracking_error_max_m": _max(
                row["rollout_position_tracking_error_max_m"] for row in rollout_metrics
            ),
            "rollout_body_rate_tracking_error_mean_rps": _mean(
                row["rollout_body_rate_tracking_error_mean_rps"] for row in rollout_metrics
            ),
            "rollout_body_rate_tracking_error_max_rps": _max(
                row["rollout_body_rate_tracking_error_max_rps"] for row in rollout_metrics
            ),
        },
        "policies": policies,
        "artifacts": {
            "planner_metrics_csv": "planner_metrics.csv",
            "plan_rollout_metrics_csv": "plan_rollout_metrics.csv",
            "plan_rollout_trajectories_csv": "plan_rollout_trajectories.csv",
            "execution_metrics_csv": "execution_metrics.csv",
            "failure_classification_csv": "failure_classification.csv",
            "planned_trajectories_csv": "planned_trajectories.csv",
            "actual_trajectories_csv": "actual_trajectories.csv",
            "plots_dir": "plots",
        },
    }


def _analysis_markdown(summary: dict[str, Any], classification_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# RPG Time-Optimal Diagnostics",
        "",
        f"Scenario table: `{summary['scenario_table']}`",
        f"Scenarios: {summary['num_scenarios']}",
        "",
        "## Planner",
        "",
        f"- Ideal feasible fraction: {summary['planner']['ideal_feasible_fraction']:.3f}",
        f"- Terminal tolerance satisfied fraction: {summary['planner']['terminal_tolerance_satisfied_fraction']:.3f}",
        f"- Mean planned min distance: {summary['planner']['planned_min_distance_mean_m']:.3f} m",
        f"- Worst planned min distance: {summary['planner']['planned_min_distance_max_m']:.3f} m",
        f"- Max constraint violation: {summary['planner']['constraint_violation_max']:.3e}",
        "",
        "## Plan Rollout",
        "",
        f"- SimEngine RPM rollout catch fraction: {summary['plan_rollout']['rollout_catch_fraction']:.3f}",
        f"- Mean rollout min distance: {summary['plan_rollout']['rollout_min_distance_mean_m']:.3f} m",
        f"- Worst rollout min distance: {summary['plan_rollout']['rollout_min_distance_max_m']:.3f} m",
        f"- Mean rollout tracking error: {summary['plan_rollout']['rollout_position_tracking_error_mean_m']:.3f} m",
        f"- Mean rollout body-rate tracking error: {summary['plan_rollout']['rollout_body_rate_tracking_error_mean_rps']:.3f} rad/s",
        "",
        "## Execution",
        "",
    ]
    for policy_name, policy_summary in summary["policies"].items():
        sim_summary = policy_summary["summary"]
        lines.extend(
            [
                f"### {policy_name}",
                "",
                f"- Catch fraction: {sim_summary['catch_fraction']:.3f}",
                f"- Errors: {sim_summary['errors']}",
                f"- Median min distance: {sim_summary['min_distance_p50_m']:.3f} m",
                f"- Mean visible fraction: {sim_summary['visible_fraction_mean']:.3f}",
                f"- Mean tracking error: {policy_summary['tracking_error_mean_m']:.3f} m",
                f"- Max tracking error: {policy_summary['tracking_error_max_m']:.3f} m",
                f"- Classifications: `{policy_summary['classification_counts']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            _interpretation(summary, classification_rows),
            "",
        ]
    )
    return "\n".join(lines)


def _interpretation(summary: dict[str, Any], classification_rows: list[dict[str, Any]]) -> str:
    planner_feasible = float(summary["planner"]["ideal_feasible_fraction"])
    if planner_feasible < 1.0:
        return (
            "The planner itself does not produce an ideal trajectory inside the capture radius for every seed. "
            "Fix the terminal OCP before tuning tracking."
        )
    mismatch = [row for row in classification_rows if row["classification"] == "execution_tracking_or_model_mismatch"]
    timing = [row for row in classification_rows if row["classification"] == "capture_condition_or_timing_mismatch"]
    if mismatch:
        return (
            "The planner reaches the target under its own model, but execution usually misses. "
            "The next work should focus on plan tracking, online replanning, or reducing model mismatch."
        )
    if timing:
        return (
            "Execution reaches the capture radius without a recorded catch for some runs. "
            "Inspect capture timing and terminal metrics."
        )
    return "The diagnostics did not isolate a single dominant failure mode."


def _target_positions(instance: Any, t_s: np.ndarray) -> np.ndarray:
    target_position = np.asarray(instance.target_initial.position_w, dtype=float).reshape(3)
    target_velocity = np.asarray(instance.target_initial.velocity_w, dtype=float).reshape(3)
    return target_position[:, None] + target_velocity[:, None] * np.asarray(t_s, dtype=float).reshape(1, -1)


def _max_rate_rps(instance: Any) -> float:
    assert instance.config is not None
    if float(instance.config.max_rate_rps) > 0.0:
        return float(instance.config.max_rate_rps)
    return float(instance.config.pursuer.max_omega_rps)


def _max_collective_thrust_n(instance: Any) -> float:
    assert instance.config is not None
    if float(instance.config.max_thrust_n) > 0.0:
        return float(instance.config.max_thrust_n)
    params = instance.config.pursuer
    return float(4.0 * params.mass_kg * params.gravity_mps2)


def _hover_rpm(params: Any) -> float:
    return float(np.sqrt((float(params.mass_kg) * float(params.gravity_mps2)) / (4.0 * float(params.k_thrust))))


def _interp_columns(t_s: np.ndarray, values: np.ndarray, sample_t_s: float) -> np.ndarray:
    t = np.asarray(t_s, dtype=float).reshape(-1)
    arr = np.asarray(values, dtype=float)
    sample = float(np.clip(sample_t_s, float(t[0]), float(t[-1])))
    return np.array([np.interp(sample, t, arr[row]) for row in range(arr.shape[0])], dtype=float)


def _command_fields(step: SimRunnerStep, slot: int) -> dict[str, Any]:
    if step.commands is None:
        return {
            "command_type": "none",
            "thrust_n": math.nan,
            "body_rate_norm_rps": math.nan,
            "motor_speed_norm_rpm": math.nan,
        }
    if hasattr(step.commands, "motor_speeds_rpm"):
        motor_speeds = np.asarray(step.commands.motor_speeds_rpm[slot], dtype=float).reshape(4)
        return {
            "command_type": "motor_speeds",
            "thrust_n": math.nan,
            "body_rate_norm_rps": math.nan,
            "motor_speed_norm_rpm": float(np.linalg.norm(motor_speeds)),
        }
    body_rates = np.asarray(step.commands.body_rates_b[slot], dtype=float).reshape(3)
    return {
        "command_type": "ctbr",
        "thrust_n": float(step.commands.thrust_n[slot]),
        "body_rate_norm_rps": float(np.linalg.norm(body_rates)),
        "motor_speed_norm_rpm": math.nan,
    }


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def _mean(values) -> float:
    vals = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(vals)) if vals else math.nan


def _max(values) -> float:
    vals = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.max(vals)) if vals else math.nan


def _finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def _finite_max(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else math.nan


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _resolve_workers(workers: int | None, task_count: int) -> int:
    if workers is not None:
        return max(1, int(workers))
    cpu_count = os.cpu_count() or 1
    return max(1, min(int(task_count), cpu_count - 1 if cpu_count > 1 else 1))


def _resolve_max_envs(max_envs: int | None, scenario_count: int) -> int:
    if max_envs is not None:
        return max(1, int(max_envs))
    cpu_count = os.cpu_count() or 1
    return max(1, min(int(scenario_count), cpu_count))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
