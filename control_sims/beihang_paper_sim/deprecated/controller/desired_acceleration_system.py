"""Desired acceleration LeafSystem for the paper controller."""

from __future__ import annotations

import numpy as np
from pydrake.systems.framework import LeafSystem

from ...controller.control_math import DEFAULT_GAINS
from .pipeline_types import (
    DesiredAccelerationState,
    desired_acceleration_value,
    los_guidance_value,
)


class DesiredAccelerationSystem(LeafSystem):
    def __init__(self, mass_kg: float, gains: dict | None = None):
        super().__init__()
        self._m = float(mass_kg)
        g = {**DEFAULT_GAINS, **(gains or {})}
        self._k_1 = float(g["k_1"])
        self._k_2 = float(g["k_2"])
        self._D = np.diag(np.asarray(g["drag_diag"], dtype=float))

        self.DeclareAbstractInputPort("los_guidance", los_guidance_value())
        self.DeclareAbstractOutputPort(
            "desired_acceleration", desired_acceleration_value, self._calc,
        )

    def _calc(self, context, output):
        los = self.GetInputPort("los_guidance").Eval(context)
        if not los.valid:
            output.set_value(DesiredAccelerationState(valid=False, t=los.t))
            return

        p_r = los.p_r
        v_r = los.v_r
        R_wb = los.R_wb
        n_t = los.n_t
        n_td = los.n_td
        v_w = los.vehicle_velocity_w
        z_2 = v_r + self._k_1 * p_r
        proj = -np.eye(3) + np.outer(n_t, n_t)
        a_d = (
            -self._k_1 * v_r
            -self._k_2 * z_2
            -p_r
            + los.barrier * (self._m / los.norm_pr) * (proj @ n_td)
        )
        e_f_drag = -R_wb @ self._D @ R_wb.T @ v_w

        output.set_value(
            DesiredAccelerationState(
                valid=True,
                t=los.t,
                R_wb=R_wb,
                n_t=n_t,
                n_f=los.n_f,
                b_omega_1=los.b_omega_1,
                a_d=a_d,
                e_f_drag=e_f_drag,
            )
        )
