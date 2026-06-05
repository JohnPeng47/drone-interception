from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from backends.csim.bindings import BatchPufferSimEngineBackend
from backends.csim.bindings.types import SimInstance
from backends.csim.runner import MotorSpeedCommandBatch, SimControlPolicy, SimRunnerState

from .config import RpgTimeOptimalConfig
from .motor_feedforward_policy import sample_motor_speed_command
from .planner import RpgTimeOptimalPlan, RpgTimeOptimalPlanner


@dataclass(frozen=True)
class RpgTimeOptimalPortfolioCandidate:
    name: str
    config: RpgTimeOptimalConfig


@dataclass(frozen=True)
class RpgPlanReplayScore:
    candidate_name: str
    clean: bool
    rollout_caught_radius: bool
    rollout_min_distance_m: float
    rollout_final_distance_m: float
    rollout_capture_steps: int
    rollout_max_consecutive_capture_steps: int
    rollout_position_tracking_error_mean_m: float
    replay_wall_s: float
    plan_total_time_s: float
    solver_success: bool
    constraint_violation_max: float
    terminal_tolerance_satisfied: bool
    planned_feasible: bool
    error: str = ""


@dataclass(frozen=True)
class RpgSelectedPortfolioPlan:
    candidate: RpgTimeOptimalPortfolioCandidate
    plan: RpgTimeOptimalPlan
    score: RpgPlanReplayScore
    traces: tuple["RpgPortfolioCandidateTrace", ...] = ()


@dataclass(frozen=True)
class RpgPortfolioCandidateTrace:
    candidate_name: str
    selected: bool
    clean: bool
    rollout_caught_radius: bool
    rollout_min_distance_m: float
    rollout_capture_steps: int
    rollout_max_consecutive_capture_steps: int
    rollout_position_tracking_error_mean_m: float
    plan_total_time_s: float
    solver_status: str
    solver_success: bool
    constraint_violation_max: float
    solve_wall_s: float
    nlp_build_wall_s: float
    optimizer_wall_s: float
    optimizer_iterations: int
    replay_wall_s: float
    warm_started: bool
    skipped: bool
    stop_reason: str
    error: str = ""


DEFAULT_PORTFOLIO_CANDIDATES: tuple[RpgTimeOptimalPortfolioCandidate, ...] = (
    RpgTimeOptimalPortfolioCandidate(
        name="rate0p5_body0p2_win8",
        config=RpgTimeOptimalConfig(
            cpc_tolerance_m=0.1,
            terminal_nodes=60,
            planner_rate_limit_scale=0.5,
            body_rate_smoothness_weight=0.2,
            terminal_capture_window_nodes=8,
            ipopt_max_iter=300,
        ),
    ),
    RpgTimeOptimalPortfolioCandidate(
        name="rate0p5_body0p2_win6",
        config=RpgTimeOptimalConfig(
            cpc_tolerance_m=0.1,
            terminal_nodes=60,
            planner_rate_limit_scale=0.5,
            body_rate_smoothness_weight=0.2,
            terminal_capture_window_nodes=6,
            ipopt_max_iter=300,
        ),
    ),
    RpgTimeOptimalPortfolioCandidate(
        name="rate0p5_cmd0p2_body0p05_win8",
        config=RpgTimeOptimalConfig(
            cpc_tolerance_m=0.1,
            terminal_nodes=60,
            planner_rate_limit_scale=0.5,
            command_smoothness_weight=0.2,
            body_rate_smoothness_weight=0.05,
            terminal_capture_window_nodes=8,
            ipopt_max_iter=300,
        ),
    ),
)


class RpgTimeOptimalPortfolioMotorPolicy(SimControlPolicy):
    """Select and execute the most robust RPG motor plan from a fixed candidate set."""

    def __init__(
        self,
        candidates: Sequence[RpgTimeOptimalPortfolioCandidate] = DEFAULT_PORTFOLIO_CANDIDATES,
        *,
        constraint_violation_tolerance: float = 1.0e-4,
        early_accept_min_consecutive_capture_steps: int | None = 15,
        early_accept_max_min_distance_m: float | None = 0.45,
        early_accept_max_tracking_error_mean_m: float | None = 0.5,
    ):
        self.candidates = tuple(candidates)
        if not self.candidates:
            raise ValueError("portfolio policy requires at least one candidate")
        self.constraint_violation_tolerance = float(constraint_violation_tolerance)
        self.early_accept_min_consecutive_capture_steps = (
            None
            if early_accept_min_consecutive_capture_steps is None
            else int(early_accept_min_consecutive_capture_steps)
        )
        self.early_accept_max_min_distance_m = (
            None
            if early_accept_max_min_distance_m is None
            else float(early_accept_max_min_distance_m)
        )
        self.early_accept_max_tracking_error_mean_m = (
            None
            if early_accept_max_tracking_error_mean_m is None
            else float(early_accept_max_tracking_error_mean_m)
        )
        self._slots: dict[int, RpgSelectedPortfolioPlan] = {}

    def reset(self, state: SimRunnerState) -> None:
        self._slots.clear()

    def on_slots_started(self, slots: np.ndarray, instances, state: SimRunnerState) -> None:
        for slot in np.asarray(slots, dtype=np.int64).reshape(-1):
            slot_i = int(slot)
            instance = state.instances[slot_i]
            if instance is None:
                continue
            self._slots[slot_i] = solve_portfolio_plan(
                instance,
                self.candidates,
                constraint_violation_tolerance=self.constraint_violation_tolerance,
                early_accept_min_consecutive_capture_steps=self.early_accept_min_consecutive_capture_steps,
                early_accept_max_min_distance_m=self.early_accept_max_min_distance_m,
                early_accept_max_tracking_error_mean_m=self.early_accept_max_tracking_error_mean_m,
            )

    def command(self, state: SimRunnerState) -> MotorSpeedCommandBatch:
        motor_speeds_rpm = np.zeros((len(state.instances), 4), dtype=np.float32)
        for slot, instance in enumerate(state.instances):
            if instance is None or not bool(state.active[slot]):
                continue
            selected = self._slots.get(slot)
            if selected is None:
                motor_speeds_rpm[slot] = _hover_rpm(instance)
                continue
            motor_speeds_rpm[slot] = sample_motor_speed_command(
                instance,
                selected.plan,
                float(state.elapsed_s[slot]),
                time_scale=float(selected.candidate.config.plan_time_scale),
                command_mode=selected.candidate.config.motor_command_mode,
            )
        return MotorSpeedCommandBatch(motor_speeds_rpm=motor_speeds_rpm)


def solve_portfolio_plan(
    instance: SimInstance,
    candidates: Sequence[RpgTimeOptimalPortfolioCandidate] = DEFAULT_PORTFOLIO_CANDIDATES,
    *,
    constraint_violation_tolerance: float = 1.0e-4,
    early_accept_min_consecutive_capture_steps: int | None = 15,
    early_accept_max_min_distance_m: float | None = 0.45,
    early_accept_max_tracking_error_mean_m: float | None = 0.5,
) -> RpgSelectedPortfolioPlan:
    scored: list[tuple[RpgTimeOptimalPortfolioCandidate, RpgTimeOptimalPlan, RpgPlanReplayScore]] = []
    trace_rows: list[RpgPortfolioCandidateTrace] = []
    warm_start_plan: RpgTimeOptimalPlan | None = None
    for candidate in candidates:
        warm_started = _same_layout(warm_start_plan, candidate.config)
        try:
            plan = RpgTimeOptimalPlanner(candidate.config).solve(
                instance,
                initial_guess=warm_start_plan if warm_started else None,
            )
            score = score_plan_replay(
                instance,
                candidate.name,
                candidate.config,
                plan,
                constraint_violation_tolerance=constraint_violation_tolerance,
            )
        except Exception as exc:  # noqa: BLE001
            score = RpgPlanReplayScore(
                candidate_name=candidate.name,
                clean=False,
                rollout_caught_radius=False,
                rollout_min_distance_m=math.inf,
                rollout_final_distance_m=math.inf,
                rollout_capture_steps=0,
                rollout_max_consecutive_capture_steps=0,
                rollout_position_tracking_error_mean_m=math.inf,
                replay_wall_s=0.0,
                plan_total_time_s=math.inf,
                solver_success=False,
                constraint_violation_max=math.inf,
                terminal_tolerance_satisfied=False,
                planned_feasible=False,
                error=repr(exc),
            )
            trace_rows.append(_trace_from_score(candidate.name, None, score, selected=False, warm_started=warm_started))
            continue
        scored.append((candidate, plan, score))
        trace_rows.append(_trace_from_score(candidate.name, plan, score, selected=False, warm_started=warm_started))
        if _same_layout(plan, candidate.config) and bool(plan.solver_success) and np.isfinite(float(plan.constraint_violation_max)):
            warm_start_plan = plan
        if _early_accept(
            score,
            min_consecutive_capture_steps=early_accept_min_consecutive_capture_steps,
            max_min_distance_m=early_accept_max_min_distance_m,
            max_tracking_error_mean_m=early_accept_max_tracking_error_mean_m,
        ):
            traces = _selected_with_skipped_traces(
                trace_rows,
                candidates,
                selected_name=candidate.name,
                first_skipped_index=len(trace_rows),
                stop_reason="early_accept",
            )
            return RpgSelectedPortfolioPlan(candidate=candidate, plan=plan, score=score, traces=traces)
    if not scored:
        raise RuntimeError("No portfolio candidate produced a plan")
    candidate, plan, score = select_best_scored_plan(scored)
    traces = tuple(
        _replace_selected(trace, selected=trace.candidate_name == candidate.name)
        for trace in trace_rows
    )
    return RpgSelectedPortfolioPlan(candidate=candidate, plan=plan, score=score, traces=traces)


def score_plan_replay(
    instance: SimInstance,
    candidate_name: str,
    config: RpgTimeOptimalConfig,
    plan: RpgTimeOptimalPlan,
    *,
    constraint_violation_tolerance: float = 1.0e-4,
) -> RpgPlanReplayScore:
    assert instance.config is not None
    planned_distances = _planned_target_distances(instance, plan)
    terminal_tolerance = (
        float(config.cpc_tolerance_m)
        if config.cpc_tolerance_m is not None
        else float(instance.config.intercept_radius_m)
    )
    terminal_tolerance_satisfied = bool(float(planned_distances[-1]) <= terminal_tolerance + 1.0e-6)
    planned_feasible = bool(float(np.min(planned_distances)) <= float(instance.config.intercept_radius_m))
    clean = (
        bool(plan.solver_success)
        and bool(np.all(np.isfinite(planned_distances)))
        and bool(terminal_tolerance_satisfied)
        and bool(planned_feasible)
        and bool(np.isfinite(float(plan.constraint_violation_max)))
        and float(plan.constraint_violation_max) <= float(constraint_violation_tolerance)
    )
    if not clean:
        return RpgPlanReplayScore(
            candidate_name=candidate_name,
            clean=False,
            rollout_caught_radius=False,
            rollout_min_distance_m=math.inf,
                rollout_final_distance_m=math.inf,
                rollout_capture_steps=0,
                rollout_max_consecutive_capture_steps=0,
                rollout_position_tracking_error_mean_m=math.inf,
            replay_wall_s=0.0,
            plan_total_time_s=float(plan.total_time_s),
            solver_success=bool(plan.solver_success),
            constraint_violation_max=float(plan.constraint_violation_max),
            terminal_tolerance_satisfied=terminal_tolerance_satisfied,
            planned_feasible=planned_feasible,
        )

    replay_start = time.perf_counter()
    backend = BatchPufferSimEngineBackend(1)
    snapshots = backend.reset_many(np.array([0], dtype=np.int64), (instance,))
    dt = float(instance.config.options.backend_dt) * max(1, int(instance.config.options.action_substeps))
    horizon_s = float(plan.total_time_s) * float(config.plan_time_scale)
    max_steps = int(math.ceil(horizon_s / max(dt, 1.0e-9))) + 1
    elapsed_s = 0.0
    distances: list[float] = []
    tracking_errors: list[float] = []
    capture_steps = 0
    max_consecutive_capture_steps = 0
    consecutive_capture_steps = 0
    for step_index in range(max_steps + 1):
        snapshot = snapshots[0]
        position = np.asarray(snapshot.pursuer.position_w, dtype=float).reshape(3)
        target = np.asarray(snapshot.target.position_w, dtype=float).reshape(3)
        planned_position = _interp_columns(
            plan.t_x_s,
            plan.position_w,
            elapsed_s / max(float(config.plan_time_scale), 1.0e-9),
        )
        distance = float(np.linalg.norm(position - target))
        distances.append(distance)
        tracking_errors.append(float(np.linalg.norm(position - planned_position)))
        if distance <= float(instance.config.intercept_radius_m):
            capture_steps += 1
            consecutive_capture_steps += 1
            max_consecutive_capture_steps = max(max_consecutive_capture_steps, consecutive_capture_steps)
        else:
            consecutive_capture_steps = 0
        if elapsed_s >= horizon_s:
            break
        command = sample_motor_speed_command(
            instance,
            plan,
            elapsed_s,
            time_scale=float(config.plan_time_scale),
            command_mode=config.motor_command_mode,
        )
        snapshots = backend.step_motor_speeds_many(command.reshape(1, 4))
        elapsed_s += dt

    min_distance = float(np.min(distances)) if distances else math.inf
    return RpgPlanReplayScore(
        candidate_name=candidate_name,
        clean=True,
        rollout_caught_radius=bool(min_distance <= float(instance.config.intercept_radius_m)),
        rollout_min_distance_m=min_distance,
        rollout_final_distance_m=float(distances[-1]) if distances else math.inf,
        rollout_capture_steps=int(capture_steps),
        rollout_max_consecutive_capture_steps=int(max_consecutive_capture_steps),
        rollout_position_tracking_error_mean_m=float(np.mean(tracking_errors)) if tracking_errors else math.inf,
        replay_wall_s=time.perf_counter() - replay_start,
        plan_total_time_s=float(plan.total_time_s),
        solver_success=bool(plan.solver_success),
        constraint_violation_max=float(plan.constraint_violation_max),
        terminal_tolerance_satisfied=terminal_tolerance_satisfied,
        planned_feasible=planned_feasible,
    )


def select_best_scored_plan(
    scored: Sequence[tuple[RpgTimeOptimalPortfolioCandidate, RpgTimeOptimalPlan, RpgPlanReplayScore]],
) -> tuple[RpgTimeOptimalPortfolioCandidate, RpgTimeOptimalPlan, RpgPlanReplayScore]:
    clean = [item for item in scored if item[2].clean]
    if not clean:
        raise RuntimeError("No clean portfolio candidate plan is available")
    return min(clean, key=lambda item: _score_sort_key(item[2]))


def _trace_from_score(
    candidate_name: str,
    plan: RpgTimeOptimalPlan | None,
    score: RpgPlanReplayScore,
    *,
    selected: bool,
    warm_started: bool,
) -> RpgPortfolioCandidateTrace:
    return RpgPortfolioCandidateTrace(
        candidate_name=str(candidate_name),
        selected=bool(selected),
        clean=bool(score.clean),
        rollout_caught_radius=bool(score.rollout_caught_radius),
        rollout_min_distance_m=float(score.rollout_min_distance_m),
        rollout_capture_steps=int(score.rollout_capture_steps),
        rollout_max_consecutive_capture_steps=int(score.rollout_max_consecutive_capture_steps),
        rollout_position_tracking_error_mean_m=float(score.rollout_position_tracking_error_mean_m),
        plan_total_time_s=float(score.plan_total_time_s),
        solver_status="" if plan is None else str(plan.solver_status),
        solver_success=bool(score.solver_success),
        constraint_violation_max=float(score.constraint_violation_max),
        solve_wall_s=math.nan if plan is None else float(plan.solve_wall_s),
        nlp_build_wall_s=math.nan if plan is None else float(plan.nlp_build_wall_s),
        optimizer_wall_s=math.nan if plan is None else float(plan.optimizer_wall_s),
        optimizer_iterations=-1 if plan is None else int(plan.optimizer_iterations),
        replay_wall_s=float(score.replay_wall_s),
        warm_started=bool(warm_started),
        skipped=False,
        stop_reason="",
        error=str(score.error),
    )


def _replace_selected(trace: RpgPortfolioCandidateTrace, *, selected: bool) -> RpgPortfolioCandidateTrace:
    return RpgPortfolioCandidateTrace(
        candidate_name=trace.candidate_name,
        selected=bool(selected),
        clean=trace.clean,
        rollout_caught_radius=trace.rollout_caught_radius,
        rollout_min_distance_m=trace.rollout_min_distance_m,
        rollout_capture_steps=trace.rollout_capture_steps,
        rollout_max_consecutive_capture_steps=trace.rollout_max_consecutive_capture_steps,
        rollout_position_tracking_error_mean_m=trace.rollout_position_tracking_error_mean_m,
        plan_total_time_s=trace.plan_total_time_s,
        solver_status=trace.solver_status,
        solver_success=trace.solver_success,
        constraint_violation_max=trace.constraint_violation_max,
        solve_wall_s=trace.solve_wall_s,
        nlp_build_wall_s=trace.nlp_build_wall_s,
        optimizer_wall_s=trace.optimizer_wall_s,
        optimizer_iterations=trace.optimizer_iterations,
        replay_wall_s=trace.replay_wall_s,
        warm_started=trace.warm_started,
        skipped=trace.skipped,
        stop_reason=trace.stop_reason,
        error=trace.error,
    )


def _selected_with_skipped_traces(
    trace_rows: list[RpgPortfolioCandidateTrace],
    candidates: Sequence[RpgTimeOptimalPortfolioCandidate],
    *,
    selected_name: str,
    first_skipped_index: int,
    stop_reason: str,
) -> tuple[RpgPortfolioCandidateTrace, ...]:
    traces = [
        _replace_selected(trace, selected=trace.candidate_name == selected_name)
        for trace in trace_rows
    ]
    for candidate in candidates[first_skipped_index:]:
        traces.append(_skipped_trace(candidate.name, stop_reason=stop_reason))
    return tuple(traces)


def _skipped_trace(candidate_name: str, *, stop_reason: str) -> RpgPortfolioCandidateTrace:
    return RpgPortfolioCandidateTrace(
        candidate_name=str(candidate_name),
        selected=False,
        clean=False,
        rollout_caught_radius=False,
        rollout_min_distance_m=math.inf,
        rollout_capture_steps=0,
        rollout_max_consecutive_capture_steps=0,
        rollout_position_tracking_error_mean_m=math.inf,
        plan_total_time_s=math.inf,
        solver_status="skipped",
        solver_success=False,
        constraint_violation_max=math.inf,
        solve_wall_s=0.0,
        nlp_build_wall_s=0.0,
        optimizer_wall_s=0.0,
        optimizer_iterations=0,
        replay_wall_s=0.0,
        warm_started=False,
        skipped=True,
        stop_reason=str(stop_reason),
        error="",
    )


def _early_accept(
    score: RpgPlanReplayScore,
    min_consecutive_capture_steps: int | None,
    *,
    max_min_distance_m: float | None = None,
    max_tracking_error_mean_m: float | None = None,
) -> bool:
    if min_consecutive_capture_steps is None:
        return False
    accepted = (
        bool(score.clean)
        and bool(score.rollout_caught_radius)
        and int(score.rollout_max_consecutive_capture_steps) >= int(min_consecutive_capture_steps)
    )
    if not accepted:
        return False
    if max_min_distance_m is not None and float(score.rollout_min_distance_m) > float(max_min_distance_m):
        return False
    if (
        max_tracking_error_mean_m is not None
        and float(score.rollout_position_tracking_error_mean_m) > float(max_tracking_error_mean_m)
    ):
        return False
    return True


def _same_layout(plan: RpgTimeOptimalPlan | None, config: RpgTimeOptimalConfig) -> bool:
    if plan is None or plan.decision_vector is None:
        return False
    nodes = int(config.terminal_nodes)
    expected = 1 + 17 + nodes * (4 + 17)
    return (
        int(plan.position_w.shape[1]) == nodes + 1
        and int(np.asarray(plan.decision_vector).reshape(-1).size) == expected
    )


def _score_sort_key(score: RpgPlanReplayScore) -> tuple[bool, int, float, float, float]:
    return (
        not bool(score.rollout_caught_radius),
        -int(score.rollout_capture_steps),
        float(score.rollout_min_distance_m),
        float(score.rollout_position_tracking_error_mean_m),
        float(score.plan_total_time_s),
    )


def _planned_target_distances(instance: SimInstance, plan: RpgTimeOptimalPlan) -> np.ndarray:
    target_position = np.asarray(instance.target_initial.position_w, dtype=float).reshape(3)
    target_velocity = np.asarray(instance.target_initial.velocity_w, dtype=float).reshape(3)
    target_positions = target_position[:, None] + target_velocity[:, None] * np.asarray(plan.t_x_s, dtype=float).reshape(1, -1)
    return np.linalg.norm(np.asarray(plan.position_w, dtype=float) - target_positions, axis=0)


def _interp_columns(t_s: np.ndarray, values: np.ndarray, sample_t_s: float) -> np.ndarray:
    t = np.asarray(t_s, dtype=float).reshape(-1)
    arr = np.asarray(values, dtype=float)
    sample = float(np.clip(sample_t_s, float(t[0]), float(t[-1])))
    return np.array([np.interp(sample, t, arr[row]) for row in range(arr.shape[0])], dtype=float)


def _hover_rpm(instance: SimInstance) -> float:
    assert instance.config is not None
    params = instance.config.pursuer
    return float(np.sqrt((float(params.mass_kg) * float(params.gravity_mps2)) / (4.0 * float(params.k_thrust))))
