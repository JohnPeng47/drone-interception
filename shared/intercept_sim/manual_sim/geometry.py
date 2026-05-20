from __future__ import annotations

import numpy as np


def drone_segments(arm_length_m: float) -> list[tuple[np.ndarray, np.ndarray]]:
    l = float(arm_length_m)
    return [
        (np.array([-l, 0.0, 0.0], dtype=float), np.array([l, 0.0, 0.0], dtype=float)),
        (np.array([0.0, -l, 0.0], dtype=float), np.array([0.0, l, 0.0], dtype=float)),
    ]


def transform_point(rotation_wb: np.ndarray, position_w: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    return np.asarray(position_w, dtype=float) + np.asarray(rotation_wb, dtype=float) @ np.asarray(point_b, dtype=float)

