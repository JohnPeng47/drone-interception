from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from backends.csim.bindings.types import SimInstance

from .rollout import (
    NumericRolloutMetrics,
    replay_motor_commands_in_simengine,
    rollout_motor_commands,
    target_distances_for_trajectory,
)


@dataclass(frozen=True)
class StructuredUpdateConfig:
    active_window_nodes: int = 8
    finite_difference_rpm: float = 5.0
    max_update_rpm: float = 25.0
    line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)
    terminal_distance_weight: float = 1.0
    min_distance_weight: float = 0.25
    smoothness_weight: float = 1.0e-4
    derivative_validation_seed: int = 17


@dataclass(frozen=True)
class StructuredUpdateResult:
    seed: int
    total_time_s: float
    wall_s: float
    rollout_wall_s: float
    derivative_wall_s: float
    line_search_wall_s: float
    replay_wall_s: float
    replay_steps: int
    initial_cost: float
    accepted_cost: float
    initial_min_distance_m: float
    accepted_min_distance_m: float
    initial_final_distance_m: float
    accepted_final_distance_m: float
    accepted_alpha: float
    gradient_norm: float
    gradient_abs_max: float
    active_variables: int
    direction_derivative_predicted: float
    direction_derivative_actual: float
    direction_derivative_abs_error: float
    direction_derivative_relative_error: float
    replay_caught: bool
    replay_min_distance_m: float
    replay_final_distance_m: float
    accepted_controls: np.ndarray
    accepted_metrics: NumericRolloutMetrics


def run_structured_update(
    instance: SimInstance,
    initial_controls_rpm: np.ndarray,
    total_time_s: float,
    *,
    dynamics_substeps: int = 1,
    control_layout: str = "auto",
    config: StructuredUpdateConfig | None = None,
) -> StructuredUpdateResult:
    if instance.config is None:
        raise ValueError("structured update requires SimInstance.config")
    _validate_supported_target_contract(instance)
    cfg = config or StructuredUpdateConfig()
    started = time.perf_counter()

    initial_rollout = rollout_motor_commands(
        instance,
        initial_controls_rpm,
        total_time_s,
        dynamics_substeps=dynamics_substeps,
        control_layout=control_layout,
    )
    controls = initial_rollout.controls
    active = _active_control_indices(len(controls), int(cfg.active_window_nodes))
    initial_cost, initial_metrics = _trajectory_cost(instance, controls, total_time_s, dynamics_substeps, cfg)

    derivative_started = time.perf_counter()
    gradient = _finite_difference_gradient(
        instance,
        controls,
        total_time_s,
        active,
        dynamics_substeps,
        cfg,
    )
    derivative = _validate_directional_derivative(
        instance,
        controls,
        total_time_s,
        active,
        gradient,
        dynamics_substeps,
        cfg,
    )
    derivative_wall_s = time.perf_counter() - derivative_started

    direction = np.zeros_like(controls, dtype=np.float64)
    if np.any(np.isfinite(gradient)):
        gradient_abs_max = float(np.max(np.abs(gradient[active]))) if active.size else 0.0
    else:
        gradient_abs_max = 0.0
    if gradient_abs_max > 0.0:
        direction[active] = -gradient[active] * (float(cfg.max_update_rpm) / gradient_abs_max)

    line_started = time.perf_counter()
    accepted_controls, accepted_alpha, accepted_cost, accepted_metrics = _line_search(
        instance,
        controls,
        direction,
        total_time_s,
        float(initial_cost),
        dynamics_substeps,
        cfg,
    )
    line_search_wall_s = time.perf_counter() - line_started

    replay = replay_motor_commands_in_simengine(instance, accepted_controls, total_time_s, control_layout="rows")
    wall_s = time.perf_counter() - started
    return StructuredUpdateResult(
        seed=int(instance.seed),
        total_time_s=float(total_time_s),
        wall_s=float(wall_s),
        rollout_wall_s=float(initial_rollout.rollout_wall_s),
        derivative_wall_s=float(derivative_wall_s),
        line_search_wall_s=float(line_search_wall_s),
        replay_wall_s=float(replay.replay_wall_s),
        replay_steps=int(replay.steps),
        initial_cost=float(initial_cost),
        accepted_cost=float(accepted_cost),
        initial_min_distance_m=float(initial_metrics.min_target_distance_m),
        accepted_min_distance_m=float(accepted_metrics.min_target_distance_m),
        initial_final_distance_m=float(initial_metrics.final_target_distance_m),
        accepted_final_distance_m=float(accepted_metrics.final_target_distance_m),
        accepted_alpha=float(accepted_alpha),
        gradient_norm=float(np.linalg.norm(gradient[active])) if active.size else 0.0,
        gradient_abs_max=float(gradient_abs_max),
        active_variables=int(active.size * 4),
        direction_derivative_predicted=float(derivative[0]),
        direction_derivative_actual=float(derivative[1]),
        direction_derivative_abs_error=float(derivative[2]),
        direction_derivative_relative_error=float(derivative[3]),
        replay_caught=bool(replay.caught),
        replay_min_distance_m=float(replay.min_target_distance_m),
        replay_final_distance_m=float(replay.final_target_distance_m),
        accepted_controls=accepted_controls.copy(),
        accepted_metrics=accepted_metrics,
    )


def _active_control_indices(nodes: int, active_window_nodes: int) -> np.ndarray:
    count = min(max(1, int(active_window_nodes)), int(nodes))
    return np.arange(int(nodes) - count, int(nodes), dtype=np.int64)


def _finite_difference_gradient(
    instance: SimInstance,
    controls: np.ndarray,
    total_time_s: float,
    active: np.ndarray,
    dynamics_substeps: int,
    config: StructuredUpdateConfig,
) -> np.ndarray:
    epsilon = float(config.finite_difference_rpm)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("finite_difference_rpm must be finite and positive")
    gradient = np.zeros_like(controls, dtype=np.float64)
    lower, upper = _control_bounds(instance)
    for row in active:
        for motor in range(controls.shape[1]):
            plus = controls.copy()
            minus = controls.copy()
            plus[row, motor] = min(upper, plus[row, motor] + epsilon)
            minus[row, motor] = max(lower, minus[row, motor] - epsilon)
            plus_cost, _ = _trajectory_cost(instance, plus, total_time_s, dynamics_substeps, config)
            minus_cost, _ = _trajectory_cost(instance, minus, total_time_s, dynamics_substeps, config)
            denom = plus[row, motor] - minus[row, motor]
            if denom <= 0.0:
                gradient[row, motor] = 0.0
            else:
                gradient[row, motor] = (plus_cost - minus_cost) / denom
    return gradient


def _validate_directional_derivative(
    instance: SimInstance,
    controls: np.ndarray,
    total_time_s: float,
    active: np.ndarray,
    gradient: np.ndarray,
    dynamics_substeps: int,
    config: StructuredUpdateConfig,
) -> tuple[float, float, float, float]:
    if active.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    rng = np.random.default_rng(int(config.derivative_validation_seed))
    direction = np.zeros_like(controls, dtype=np.float64)
    direction[active] = rng.normal(size=(active.size, controls.shape[1]))
    norm = float(np.linalg.norm(direction[active]))
    if norm <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    epsilon = float(config.finite_difference_rpm)
    direction *= epsilon / norm
    lower, upper = _control_bounds(instance)
    plus = np.clip(controls + direction, lower, upper)
    minus = np.clip(controls - direction, lower, upper)
    actual = (_trajectory_cost(instance, plus, total_time_s, dynamics_substeps, config)[0] - _trajectory_cost(
        instance,
        minus,
        total_time_s,
        dynamics_substeps,
        config,
    )[0]) / 2.0
    predicted = float(np.sum(gradient * direction))
    abs_error = abs(predicted - actual)
    relative_error = abs_error / max(abs(actual), abs(predicted), 1.0e-12)
    return float(predicted), float(actual), float(abs_error), float(relative_error)


def _line_search(
    instance: SimInstance,
    controls: np.ndarray,
    direction: np.ndarray,
    total_time_s: float,
    initial_cost: float,
    dynamics_substeps: int,
    config: StructuredUpdateConfig,
) -> tuple[np.ndarray, float, float, NumericRolloutMetrics]:
    best_controls = controls.copy()
    best_cost, best_metrics = _trajectory_cost(instance, best_controls, total_time_s, dynamics_substeps, config)
    lower, upper = _control_bounds(instance)
    for alpha in config.line_search_alphas:
        alpha_f = float(alpha)
        if not np.isfinite(alpha_f) or alpha_f <= 0.0:
            continue
        candidate = np.clip(controls + alpha_f * direction, lower, upper)
        cost, metrics = _trajectory_cost(instance, candidate, total_time_s, dynamics_substeps, config)
        if cost < best_cost:
            best_controls = candidate
            best_cost = cost
            best_metrics = metrics
    accepted_alpha = 0.0 if np.array_equal(best_controls, controls) else _accepted_alpha(controls, direction, best_controls, config.line_search_alphas)
    if best_cost > initial_cost + 1.0e-12:
        return controls.copy(), 0.0, float(initial_cost), _trajectory_cost(instance, controls, total_time_s, dynamics_substeps, config)[1]
    return best_controls.copy(), float(accepted_alpha), float(best_cost), best_metrics


def _accepted_alpha(
    controls: np.ndarray,
    direction: np.ndarray,
    accepted: np.ndarray,
    alphas: tuple[float, ...],
) -> float:
    lower_mask = np.abs(direction) > 1.0e-12
    if not np.any(lower_mask):
        return 0.0
    for alpha in alphas:
        candidate = controls + float(alpha) * direction
        if np.allclose(candidate[lower_mask], accepted[lower_mask], rtol=1.0e-7, atol=1.0e-7):
            return float(alpha)
    delta = accepted[lower_mask] - controls[lower_mask]
    return float(np.median(delta / direction[lower_mask]))


def _trajectory_cost(
    instance: SimInstance,
    controls: np.ndarray,
    total_time_s: float,
    dynamics_substeps: int,
    config: StructuredUpdateConfig,
) -> tuple[float, NumericRolloutMetrics]:
    trajectory = rollout_motor_commands(
        instance,
        controls,
        total_time_s,
        dynamics_substeps=dynamics_substeps,
        control_layout="rows",
    )
    distances = target_distances_for_trajectory(instance, trajectory)
    final_distance = float(distances[-1])
    min_distance = float(np.min(distances))
    smoothness = 0.0
    if len(trajectory.controls) > 1:
        rpm_scale = max(float(instance.config.pursuer.max_rpm), 1.0)
        smoothness = float(np.mean(np.square(np.diff(trajectory.controls, axis=0) / rpm_scale)))
    cost = (
        float(config.terminal_distance_weight) * final_distance * final_distance
        + float(config.min_distance_weight) * min_distance * min_distance
        + float(config.smoothness_weight) * smoothness
    )
    metrics = NumericRolloutMetrics(
        terminal_position_error_m=math.nan,
        position_error_mean_m=math.nan,
        position_error_max_m=math.nan,
        min_target_distance_m=min_distance,
        final_target_distance_m=final_distance,
        rpm_min=float(np.min(trajectory.motor_rpm)),
        rpm_max=float(np.max(trajectory.motor_rpm)),
        body_rate_abs_max_rps=float(np.max(np.abs(trajectory.body_rates_b))),
        altitude_min_m=float(np.min(trajectory.position_w[:, 2])),
        altitude_max_m=float(np.max(trajectory.position_w[:, 2])),
    )
    return float(cost), metrics


def _validate_supported_target_contract(instance: SimInstance) -> None:
    if instance.config is None:
        raise ValueError("structured update requires SimInstance.config")
    if len(instance.target_initials) != 1 or len(instance.config.targets) != 1:
        raise ValueError("structured update currently supports exactly one target")
    target = instance.config.targets[0]
    behavior = target.behavior
    controller = target.controller
    if behavior.kind != "waypoints":
        raise ValueError(f"structured update does not support target behavior kind {behavior.kind!r}")
    if behavior.waypoints:
        raise ValueError("structured update does not support waypoint target behavior")
    if float(behavior.duration_s) != 0.0 or bool(behavior.loop):
        raise ValueError("structured update does not support nontrivial target behavior timing")
    if controller.kind != "linear":
        raise ValueError(f"structured update does not support target controller kind {controller.kind!r}")
    if (
        float(controller.kp) != 0.0
        or float(controller.kv) != 0.0
        or float(controller.max_accel_mps2) != 0.0
    ):
        raise ValueError("structured update does not support nonzero target controller gains")


def _control_bounds(instance: SimInstance) -> tuple[float, float]:
    assert instance.config is not None
    params = instance.config.pursuer
    rpm_min = float(params.rpm_min) if params.rpm_min is not None else 0.0
    if not np.isfinite(rpm_min):
        rpm_min = 0.0
    return max(0.0, rpm_min), float(params.max_rpm)
