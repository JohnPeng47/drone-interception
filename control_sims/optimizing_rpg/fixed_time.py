from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from backends.csim.bindings.types import SimInstance

from .rollout import (
    NumericRolloutMetrics,
    NumericRolloutTrajectory,
    replay_motor_commands_in_simengine,
    rollout_motor_commands,
    target_distances_for_trajectory,
)


@dataclass(frozen=True)
class FixedTimeFeasibilityResult:
    seed: int
    total_time_s: float
    feasible: bool
    caught: bool
    failure_reason: str
    wall_s: float
    rollout: NumericRolloutTrajectory
    rollout_metrics: NumericRolloutMetrics
    replay_wall_s: float
    replay_steps: int
    replay_min_distance_m: float
    replay_final_distance_m: float
    intercept_radius_m: float


def solve_fixed_time(
    instance: SimInstance,
    total_time_s: float,
    initial_controls_rpm: np.ndarray,
    *,
    dynamics_substeps: int = 1,
    control_layout: str = "auto",
) -> FixedTimeFeasibilityResult:
    """Evaluate whether a fixed-time motor command trajectory catches in replay.

    Milestone 3 intentionally keeps the solve step narrow: it validates and
    rolls out a provided command trajectory at fixed time, then uses SimEngine
    replay as the feasibility oracle. Later milestones replace this evaluation
    with actual command updates and time-search.
    """

    if instance.config is None:
        raise ValueError("fixed-time feasibility requires SimInstance.config")
    started = time.perf_counter()
    rollout = rollout_motor_commands(
        instance,
        initial_controls_rpm,
        total_time_s,
        dynamics_substeps=dynamics_substeps,
        control_layout=control_layout,
    )
    rollout_metrics = _rollout_metrics(instance, rollout)
    replay = replay_motor_commands_in_simengine(
        instance,
        rollout.controls,
        total_time_s,
        control_layout="rows",
    )
    caught = bool(replay.caught)
    feasible = caught
    failure_reason = "" if feasible else "replay_not_caught"
    return FixedTimeFeasibilityResult(
        seed=int(instance.seed),
        total_time_s=float(total_time_s),
        feasible=bool(feasible),
        caught=caught,
        failure_reason=failure_reason,
        wall_s=time.perf_counter() - started,
        rollout=rollout,
        rollout_metrics=rollout_metrics,
        replay_wall_s=float(replay.replay_wall_s),
        replay_steps=int(replay.steps),
        replay_min_distance_m=float(replay.min_target_distance_m),
        replay_final_distance_m=float(replay.final_target_distance_m),
        intercept_radius_m=float(instance.config.intercept_radius_m),
    )


def _rollout_metrics(instance: SimInstance, trajectory: NumericRolloutTrajectory) -> NumericRolloutMetrics:
    distances = target_distances_for_trajectory(instance, trajectory)
    return NumericRolloutMetrics(
        terminal_position_error_m=math.nan,
        position_error_mean_m=math.nan,
        position_error_max_m=math.nan,
        min_target_distance_m=float(np.min(distances)),
        final_target_distance_m=float(distances[-1]),
        rpm_min=float(np.min(trajectory.motor_rpm)),
        rpm_max=float(np.max(trajectory.motor_rpm)),
        body_rate_abs_max_rps=float(np.max(np.abs(trajectory.body_rates_b))),
        altitude_min_m=float(np.min(trajectory.position_w[:, 2])),
        altitude_max_m=float(np.max(trajectory.position_w[:, 2])),
    )
