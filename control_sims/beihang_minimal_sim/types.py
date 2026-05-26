from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CtbrCommand:
    t: float
    thrust_n: float
    body_rates_b: np.ndarray


@dataclass(frozen=True)
class VehicleState:
    t: float
    position_w: np.ndarray
    velocity_w: np.ndarray
    rotation_wb: np.ndarray


@dataclass(frozen=True)
class TargetState:
    t: float
    position_w: np.ndarray
    velocity_w: np.ndarray
    radius_m: float


@dataclass(frozen=True)
class SceneState:
    t: float
    vehicle: VehicleState
    target: TargetState


@dataclass(frozen=True)
class ImageFeature:
    t: float
    detected: bool
    uv_norm: np.ndarray | None
    depth_m: float | None
    bearing_c: np.ndarray | None


@dataclass(frozen=True)
class StrategyObservation:
    t: float
    detected: bool
    uv_norm: np.ndarray | None
    uv_dot_norm: np.ndarray
    depth_m: float | None
    vehicle_velocity_w: np.ndarray
    vehicle_rotation_wb: np.ndarray


@dataclass(frozen=True)
class TrialMetrics:
    t: float
    distance_m: float
    min_distance_m: float
    captured: bool
    capture_time_s: float | None
    in_view: bool
    image_error: float | None
    control_effort: float
    crashed: bool
    out_of_bounds: bool


@dataclass(frozen=True)
class TrialSample:
    t: float
    vehicle: VehicleState
    target: TargetState
    feature: ImageFeature
    observation: StrategyObservation
    command: CtbrCommand
    metrics: TrialMetrics

