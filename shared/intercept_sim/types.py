from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationTarget:
    id: str
    kind: str
    position_w: np.ndarray
    velocity_w: np.ndarray
    rotation_wb: np.ndarray
    radius_m: float


@dataclass(frozen=True)
class CameraIntrinsics:
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    hfov_rad: float
    vfov_rad: float


@dataclass(frozen=True)
class CameraRig:
    id: str
    parent_id: str
    position_b: np.ndarray
    body_to_camera: np.ndarray
    intrinsics: CameraIntrinsics
    capture_rate_hz: float


@dataclass(frozen=True)
class SceneSnapshot:
    t: float
    pursuer: SimulationTarget
    targets: tuple[SimulationTarget, ...]
    cameras: tuple[CameraRig, ...]


@dataclass(frozen=True)
class CameraCapture:
    t_capture: float
    camera_id: str
    target_id: str | None
    detected: bool
    uv_px: np.ndarray | None
    uv_norm: np.ndarray | None
    target_pos_c: np.ndarray | None
    range_m: float | None
    apparent_radius_px: float | None


@dataclass(frozen=True)
class ImageFeatureMeasurement:
    t_capture: float
    t_available: float
    camera_id: str
    target_id: str | None
    detected: bool
    uv_px: np.ndarray | None
    uv_norm: np.ndarray | None
    confidence: float


@dataclass(frozen=True)
class ObserverState:
    t: float
    vehicle_state: dict[str, np.ndarray]
    image_feature: ImageFeatureMeasurement | None
    relative_position_w: np.ndarray | None = None
    relative_velocity_w: np.ndarray | None = None
    target_acceleration_w: np.ndarray | None = None
    vehicle_rotation_wb: np.ndarray | None = None


@dataclass(frozen=True)
class CtbrCommand:
    t: float
    thrust_n: float
    body_rates_b: np.ndarray
