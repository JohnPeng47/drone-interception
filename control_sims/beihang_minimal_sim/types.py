from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CtbrCommand:
    t: float
    thrust_n: float
    body_rates_b: np.ndarray


@dataclass(frozen=True)
class StrategyObservation:
    t: float
    detected: bool
    uv_norm: np.ndarray | None
    uv_dot_norm: np.ndarray
    depth_m: float | None
    bearing_b: np.ndarray
    vehicle_velocity_w: np.ndarray
    vehicle_rotation_wb: np.ndarray
