"""Thrust planning LeafSystem for the paper controller."""

from __future__ import annotations

import numpy as np
from pydrake.systems.framework import LeafSystem

from .control_math import DEFAULT_GAINS, G_VEC
from .pipeline_types import (
    ThrustPlanState,
    desired_acceleration_value,
    thrust_plan_value,
)


class ThrustPlanningSystem(LeafSystem):
    def __init__(self, mass_kg: float, gains: dict | None = None):
        super().__init__()
        self._m = float(mass_kg)
        g = {**DEFAULT_GAINS, **(gains or {})}
        self._f_max = float(g["f_max"])

        self.DeclareAbstractInputPort(
            "desired_acceleration", desired_acceleration_value()
        )
        self.DeclareAbstractOutputPort("thrust_plan", thrust_plan_value, self._calc)

    def _calc(self, context, output):
        desired = self.GetInputPort("desired_acceleration").Eval(context)
        if not desired.valid:
            output.set_value(ThrustPlanState(valid=False, t=desired.t))
            return

        n_fd_raw = desired.a_d - G_VEC - desired.e_f_drag / self._m
        n_fd = n_fd_raw / max(float(np.linalg.norm(n_fd_raw)), 1e-9)
        R_tilt = _tilt_rotation(desired.n_f, n_fd)
        R_d = R_tilt @ desired.R_wb
        f_raw = float(
            desired.n_f @ (self._m * desired.a_d - self._m * G_VEC - desired.e_f_drag)
        )
        f_d = float(np.clip(f_raw, 0.0, self._f_max))

        output.set_value(
            ThrustPlanState(
                valid=True,
                t=desired.t,
                R_wb=desired.R_wb,
                R_d=R_d,
                b_omega_1=desired.b_omega_1,
                thrust_n=f_d,
            )
        )


def _tilt_rotation(n_f: np.ndarray, n_fd: np.ndarray) -> np.ndarray:
    r = np.cross(n_f, n_fd)
    cos_phi = float(np.clip(n_f @ n_fd, -1.0, 1.0))
    s = float(np.linalg.norm(r))
    if s < 1e-9:
        return np.eye(3)
    r_hat = r / s
    K = np.array([[0.0, -r_hat[2], r_hat[1]],
                  [r_hat[2], 0.0, -r_hat[0]],
                  [-r_hat[1], r_hat[0], 0.0]])
    phi = float(np.arccos(cos_phi))
    return np.eye(3) + np.sin(phi) * K + (1.0 - np.cos(phi)) * (K @ K)
