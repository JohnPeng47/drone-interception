from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from backends.csim.bindings.types import SimInstance, SimSnapshot
from backends.csim.runner import CtbrCommandBatch, SimControlPolicy, SimRunnerState

from .controller.control_math import DEFAULT_GAINS, G_VEC, vex


class BeihangPaperSimControlPolicy(SimControlPolicy):
    """Run the Beihang paper LOS controller over typed C SimEngine snapshots."""

    def __init__(self, gains: Mapping[str, float] | None = None):
        self._gains = {**DEFAULT_GAINS, **dict(gains or {})}

    def command(self, state: SimRunnerState) -> CtbrCommandBatch:
        thrust_n = np.zeros(len(state.instances), dtype=np.float32)
        body_rates_b = np.zeros((len(state.instances), 3), dtype=np.float32)
        for slot, instance in enumerate(state.instances):
            if instance is None or not bool(state.active[slot]):
                continue
            command = self._command_one(instance, state.snapshot[slot])
            thrust_n[slot] = np.float32(command[0])
            body_rates_b[slot] = np.asarray(command[1], dtype=np.float32).reshape(3)
        return CtbrCommandBatch(thrust_n=thrust_n, body_rates_b=body_rates_b)

    def _command_one(self, instance: SimInstance, snapshot: SimSnapshot) -> tuple[float, np.ndarray]:
        if instance.config is None or not instance.config.cameras:
            return _hover_command(instance)

        mass_kg = float(instance.config.pursuer.mass_kg)
        camera = instance.config.cameras[0]
        R_wb = _quat_xyzw_to_rot(snapshot.pursuer.quat_xyzw)
        p_r = np.asarray(snapshot.pursuer.position_w, dtype=float) - np.asarray(snapshot.target.position_w, dtype=float)
        v_r = np.asarray(snapshot.pursuer.velocity_w, dtype=float) - np.asarray(snapshot.target.velocity_w, dtype=float)
        norm_pr = float(np.linalg.norm(p_r))
        if norm_pr < 1.0e-6:
            return _hover_command(instance)

        k_b = float(self._gains["k_b"])
        k_1 = float(self._gains["k_1"])
        k_2 = float(self._gains["k_2"])
        f_max = float(self._gains.get("f_max", DEFAULT_GAINS["f_max"]))
        omega_max = float(self._gains.get("omega_max", DEFAULT_GAINS["omega_max"]))
        if instance.config.max_thrust_n > 0.0:
            f_max = min(f_max, float(instance.config.max_thrust_n))
        if instance.config.max_rate_rps > 0.0:
            omega_max = min(omega_max, float(instance.config.max_rate_rps))

        n_t = -p_r / norm_pr
        R_b2c = np.asarray(camera.body_to_camera, dtype=float).reshape(3, 3)
        n_td_body = R_b2c.T @ np.array([1.0, 0.0, 0.0], dtype=float)
        n_td = R_wb @ n_td_body
        n_f = R_wb @ np.array([0.0, 0.0, 1.0], dtype=float)

        z_1 = 1.0 - float(n_td @ n_t)
        z_1 = float(np.clip(z_1, -0.99 * k_b, 0.99 * k_b))
        barrier = z_1 / (k_b**2 - z_1**2)
        b_omega_1 = barrier * (R_wb.T @ np.cross(n_td, n_t))

        z_2 = v_r + k_1 * p_r
        proj = -np.eye(3) + np.outer(n_t, n_t)
        a_d = (
            -k_1 * v_r
            - k_2 * z_2
            - p_r
            + barrier * (mass_kg / norm_pr) * (proj @ n_td)
        )
        drag = np.diag(np.asarray(self._gains["drag_diag"], dtype=float).reshape(3))
        v_w = np.asarray(snapshot.pursuer.velocity_w, dtype=float)
        e_f_drag = -R_wb @ drag @ R_wb.T @ v_w

        n_fd_raw = a_d - G_VEC - e_f_drag / mass_kg
        n_fd = n_fd_raw / max(float(np.linalg.norm(n_fd_raw)), 1.0e-9)
        R_tilt = _tilt_rotation(n_f, n_fd)
        R_d = R_tilt @ R_wb
        f_raw = float(n_f @ (mass_kg * a_d - mass_kg * G_VEC - e_f_drag))
        f_d = float(np.clip(f_raw, 0.0, f_max))

        S = R_d.T @ R_wb - R_wb.T @ R_d
        b_omega_2 = -vex(S)
        b_omega_d = b_omega_1 + b_omega_2
        n_w = float(np.linalg.norm(b_omega_d))
        if n_w > omega_max:
            b_omega_d = b_omega_d * (omega_max / n_w)
        return f_d, b_omega_d


def _hover_command(instance: SimInstance) -> tuple[float, np.ndarray]:
    if instance.config is None:
        return 0.0, np.zeros(3, dtype=float)
    params = instance.config.pursuer
    thrust = float(params.mass_kg * params.gravity_mps2)
    if instance.config.max_thrust_n > 0.0:
        thrust = min(thrust, float(instance.config.max_thrust_n))
    return thrust, np.zeros(3, dtype=float)


def _tilt_rotation(n_f: np.ndarray, n_fd: np.ndarray) -> np.ndarray:
    r = np.cross(n_f, n_fd)
    cos_phi = float(np.clip(n_f @ n_fd, -1.0, 1.0))
    s = float(np.linalg.norm(r))
    if s < 1.0e-9:
        return np.eye(3)
    r_hat = r / s
    K = np.array([
        [0.0, -r_hat[2], r_hat[1]],
        [r_hat[2], 0.0, -r_hat[0]],
        [-r_hat[1], r_hat[0], 0.0],
    ])
    phi = float(np.arccos(cos_phi))
    return np.eye(3) + np.sin(phi) * K + (1.0 - np.cos(phi)) * (K @ K)


def _quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1.0e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])
