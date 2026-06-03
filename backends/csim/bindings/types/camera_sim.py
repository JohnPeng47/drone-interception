from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
class CameraConfig:
    id: str
    parent_id: str
    position_b: np.ndarray
    body_to_camera: np.ndarray
    intrinsics: CameraIntrinsics
    capture_rate_hz: float


@dataclass(frozen=True)
class CameraObservation:
    detected: bool
    uv_norm: np.ndarray
