"""Line-of-sight guidance LeafSystem for the paper controller."""

from __future__ import annotations

import numpy as np
from pydrake.systems.framework import LeafSystem

from ..drake_compat import observer_state_value
from ..types import CameraRig
from ...controller.control_math import DEFAULT_GAINS
from .pipeline_types import LosGuidanceState, los_guidance_value


class LineOfSightGuidanceSystem(LeafSystem):
    def __init__(self, camera_rig: CameraRig, gains: dict | None = None):
        super().__init__()
        g = {**DEFAULT_GAINS, **(gains or {})}
        self._k_b = float(g["k_b"])

        # paper's R^b_c (camera->body) = body_to_camera.T in this codebase.
        self._R_c2b = np.asarray(camera_rig.body_to_camera, dtype=float).T
        self._optical_axis_c = np.array([1.0, 0.0, 0.0])
        self._n_td_body = self._R_c2b @ self._optical_axis_c
        self._R_f = np.eye(3)

        self.DeclareAbstractInputPort("observer_state", observer_state_value())
        self.DeclareAbstractOutputPort(
            "los_guidance", los_guidance_value, self._calc,
            prerequisites_of_calc={self.time_ticket()},
        )

    def _calc(self, context, output):
        s = self.GetInputPort("observer_state").Eval(context)
        t = float(context.get_time())
        if (
            s.relative_position_w is None
            or s.relative_velocity_w is None
            or s.vehicle_rotation_wb is None
        ):
            output.set_value(LosGuidanceState(valid=False, t=t))
            return

        p_r = np.asarray(s.relative_position_w, dtype=float)
        v_r = np.asarray(s.relative_velocity_w, dtype=float)
        R_wb = np.asarray(s.vehicle_rotation_wb, dtype=float).reshape(3, 3)

        norm_pr = float(np.linalg.norm(p_r))
        if norm_pr < 1e-6:
            output.set_value(LosGuidanceState(valid=False, t=t))
            return

        n_t = -p_r / norm_pr
        n_td = R_wb @ self._n_td_body
        n_f = R_wb @ self._R_f @ np.array([0.0, 0.0, 1.0])

        z_1 = 1.0 - float(n_td @ n_t)
        z_1 = float(np.clip(z_1, -0.99 * self._k_b, 0.99 * self._k_b))
        denom = self._k_b ** 2 - z_1 ** 2
        barrier = z_1 / denom
        b_omega_1 = barrier * (R_wb.T @ np.cross(n_td, n_t))
        v_w = np.asarray(
            s.vehicle_state.get("v", np.zeros(3, dtype=float)), dtype=float
        )

        output.set_value(
            LosGuidanceState(
                valid=True,
                t=t,
                p_r=p_r,
                v_r=v_r,
                R_wb=R_wb,
                n_t=n_t,
                n_td=n_td,
                n_f=n_f,
                b_omega_1=b_omega_1,
                barrier=float(barrier),
                norm_pr=norm_pr,
                vehicle_velocity_w=v_w,
            )
        )
