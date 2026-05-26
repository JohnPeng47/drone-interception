from __future__ import annotations

import numpy as np
from pydrake.common.value import AbstractValue

from .types import (
    CtbrCommand,
    ImageFeature,
    SceneState,
    StrategyObservation,
    TargetState,
    TrialMetrics,
    TrialSample,
    VehicleState,
)


def identity_rotation() -> np.ndarray:
    return np.eye(3, dtype=float)


def hover_command(t: float = 0.0, mass_kg: float = 1.0) -> CtbrCommand:
    return CtbrCommand(t=t, thrust_n=9.81 * mass_kg, body_rates_b=np.zeros(3))


def empty_vehicle_state() -> VehicleState:
    return VehicleState(
        t=0.0,
        position_w=np.zeros(3),
        velocity_w=np.zeros(3),
        rotation_wb=identity_rotation(),
    )


def empty_target_state() -> TargetState:
    return TargetState(
        t=0.0,
        position_w=np.zeros(3),
        velocity_w=np.zeros(3),
        radius_m=0.2,
    )


def empty_scene_state() -> SceneState:
    return SceneState(t=0.0, vehicle=empty_vehicle_state(), target=empty_target_state())


def empty_image_feature() -> ImageFeature:
    return ImageFeature(t=0.0, detected=False, uv_norm=None, depth_m=None, bearing_c=None)


def empty_strategy_observation() -> StrategyObservation:
    return StrategyObservation(
        t=0.0,
        detected=False,
        uv_norm=None,
        uv_dot_norm=np.zeros(2),
        depth_m=None,
        vehicle_velocity_w=np.zeros(3),
        vehicle_rotation_wb=identity_rotation(),
    )


def empty_trial_metrics() -> TrialMetrics:
    return TrialMetrics(
        t=0.0,
        distance_m=float("inf"),
        min_distance_m=float("inf"),
        captured=False,
        capture_time_s=None,
        in_view=False,
        image_error=None,
        control_effort=0.0,
        crashed=False,
        out_of_bounds=False,
    )


def empty_trial_sample() -> TrialSample:
    return TrialSample(
        t=0.0,
        vehicle=empty_vehicle_state(),
        target=empty_target_state(),
        feature=empty_image_feature(),
        observation=empty_strategy_observation(),
        command=hover_command(),
        metrics=empty_trial_metrics(),
    )


def ctbr_value() -> AbstractValue:
    return AbstractValue.Make(hover_command())


def vehicle_state_value() -> AbstractValue:
    return AbstractValue.Make(empty_vehicle_state())


def target_state_value() -> AbstractValue:
    return AbstractValue.Make(empty_target_state())


def scene_state_value() -> AbstractValue:
    return AbstractValue.Make(empty_scene_state())


def image_feature_value() -> AbstractValue:
    return AbstractValue.Make(empty_image_feature())


def strategy_observation_value() -> AbstractValue:
    return AbstractValue.Make(empty_strategy_observation())


def trial_metrics_value() -> AbstractValue:
    return AbstractValue.Make(empty_trial_metrics())

