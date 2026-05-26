"""Minimal CTBR plant with attitude-coupled thrust."""

from __future__ import annotations

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import LeafSystem

from ..config import VehicleConfig
from ..drake_values import ctbr_value, vehicle_state_value
from ..types import VehicleState


class PointMassCtbrPlant(LeafSystem):
    def __init__(self, config: VehicleConfig, initial_rotation_wb: np.ndarray, dt: float):
        super().__init__()
        self._cfg = config
        self._dt = float(dt)
        self._drag = np.diag(np.asarray(config.drag_diag, dtype=float))
        initial = VehicleState(
            t=0.0,
            position_w=np.asarray(config.initial_position_w, dtype=float),
            velocity_w=np.asarray(config.initial_velocity_w, dtype=float),
            rotation_wb=np.asarray(initial_rotation_wb, dtype=float),
        )
        self._state_idx = self.DeclareAbstractState(AbstractValue.Make(initial))
        self.DeclareAbstractInputPort("ctbr_cmd", ctbr_value())
        self.DeclareAbstractOutputPort(
            "vehicle_state",
            vehicle_state_value,
            self._copy_state,
            prerequisites_of_calc={self.abstract_state_ticket(self._state_idx)},
        )
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=self._dt, offset_sec=0.0, update=self._step,
        )

    def _step(self, context, state) -> None:
        current = state.get_mutable_abstract_state(self._state_idx).get_value()
        cmd = self.GetInputPort("ctbr_cmd").Eval(context)
        dt = self._dt
        mass = float(self._cfg.mass_kg)

        omega_b = np.asarray(cmd.body_rates_b, dtype=float)
        omega_norm = float(np.linalg.norm(omega_b))
        if omega_norm > self._cfg.max_body_rate_rad_s:
            omega_b = omega_b * (self._cfg.max_body_rate_rad_s / omega_norm)

        thrust = float(np.clip(cmd.thrust_n, 0.0, self._cfg.max_thrust_n))
        R = current.rotation_wb @ _exp_so3(omega_b * dt)
        R = _orthonormalize(R)
        v = np.asarray(current.velocity_w, dtype=float)
        a = R @ np.array([0.0, 0.0, thrust / mass])
        a += np.array([0.0, 0.0, -9.81])
        a -= self._drag @ v

        v_next = v + a * dt
        p_next = np.asarray(current.position_w, dtype=float) + v * dt + 0.5 * a * dt * dt
        next_state = VehicleState(
            t=float(context.get_time()),
            position_w=p_next,
            velocity_w=v_next,
            rotation_wb=R,
        )
        state.get_mutable_abstract_state(self._state_idx).set_value(next_state)

    def _copy_state(self, context, output) -> None:
        output.set_value(context.get_abstract_state(self._state_idx).get_value())


def _exp_so3(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = rotvec / theta
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    out = u @ vt
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out

