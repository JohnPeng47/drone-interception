"""Shared controller constants and small SO(3) helpers."""

from __future__ import annotations

import numpy as np


DEFAULT_GAINS = {
    "k_b": 0.8,
    "k_1": 0.1,
    "k_2": 2.0,
    "omega_max": 8.0,
    "f_max": 40.0,
    "drag_diag": (0.10, 0.10, 0.20),
}

G_VEC = np.array([0.0, 0.0, -9.81])


def vex(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])
