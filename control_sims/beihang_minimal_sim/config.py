from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class VehicleConfig:
    mass_kg: float = 1.2
    drag_diag: tuple[float, float, float] = (0.08, 0.08, 0.12)
    max_thrust_n: float = 28.0
    max_body_rate_rad_s: float = 6.0
    initial_position_w: tuple[float, float, float] = (0.0, 0.0, 2.0)
    initial_velocity_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_quat_xyzw: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class TargetConfig:
    radius_m: float = 0.25
    initial_position_w: tuple[float, float, float] = (9.0, 0.8, 2.2)
    base_velocity_w: tuple[float, float, float] = (-0.25, 0.0, 0.0)
    weave_amplitude_m: tuple[float, float] = (1.4, 0.7)
    weave_frequency_hz: tuple[float, float] = (0.18, 0.11)


@dataclass(frozen=True)
class CameraConfig:
    body_to_camera: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    max_uv_norm: float = 1.4
    min_depth_m: float = 0.1


@dataclass(frozen=True)
class BaselineStrategyConfig:
    image_axis_gain: float = 3.0
    thrust_axis_gain: float = 4.0
    desired_speed_mps: float = 7.0
    velocity_gain: float = 1.8
    range_gain: float = 0.2
    thrust_margin_n: float = 1.0


@dataclass(frozen=True)
class TrialConfig:
    dt: float = 0.01
    duration_s: float = 12.0
    capture_radius_m: float = 0.6
    arena_min_w: tuple[float, float, float] = (-4.0, -6.0, 0.0)
    arena_max_w: tuple[float, float, float] = (14.0, 6.0, 6.0)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    strategy: BaselineStrategyConfig = field(default_factory=BaselineStrategyConfig)


def initial_rotation_wb_toward_target(config: TrialConfig) -> np.ndarray:
    if config.vehicle.initial_quat_xyzw is not None:
        return Rotation.from_quat(np.asarray(config.vehicle.initial_quat_xyzw, dtype=float)).as_matrix()

    p0 = np.asarray(config.vehicle.initial_position_w, dtype=float)
    pt = np.asarray(config.target.initial_position_w, dtype=float)
    forward = pt - p0
    forward[2] = 0.0
    norm = float(np.linalg.norm(forward))
    if norm < 1e-9:
        return np.eye(3)
    x_b = forward / norm
    z_b = np.array([0.0, 0.0, 1.0])
    y_b = np.cross(z_b, x_b)
    y_b = y_b / max(float(np.linalg.norm(y_b)), 1e-9)
    z_b = np.cross(x_b, y_b)
    return np.column_stack([x_b, y_b, z_b])
