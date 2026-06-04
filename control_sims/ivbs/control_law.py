from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from backends.csim.bindings.types import SimInstance, SimSnapshot
from control_sims.beihang_paper_sim.controller.control_math import DEFAULT_GAINS, G_VEC, vex
from control_sims.beihang_paper_sim.policy import _hover_command, _tilt_rotation

from .observer import RelativeStateEstimate


def bearing_error_rad(instance: SimInstance, snapshot: SimSnapshot, estimate: RelativeStateEstimate) -> float:
    if instance.config is None or not instance.config.cameras or not estimate.valid:
        return float("nan")
    r_wb = quat_xyzw_to_rot(snapshot.pursuer.quat_xyzw)
    camera = instance.config.cameras[0]
    r_b2c = np.asarray(camera.body_to_camera, dtype=float).reshape(3, 3)
    n_td_body = r_b2c.T @ np.array([1.0, 0.0, 0.0], dtype=float)
    n_td = r_wb @ n_td_body
    n_t = np.asarray(estimate.bearing_w, dtype=float).reshape(3)
    if not np.all(np.isfinite(n_t)):
        return float("nan")
    n_t = n_t / max(float(np.linalg.norm(n_t)), 1.0e-12)
    return float(np.arccos(np.clip(float(n_td @ n_t), -1.0, 1.0)))


def beihang_command_from_estimate(
    instance: SimInstance,
    snapshot: SimSnapshot,
    estimate: RelativeStateEstimate,
    gains: Mapping[str, float],
) -> tuple[float, np.ndarray]:
    if instance.config is None or not instance.config.cameras or not estimate.valid:
        return _hover_command(instance)

    mass_kg = float(instance.config.pursuer.mass_kg)
    camera = instance.config.cameras[0]
    r_wb = quat_xyzw_to_rot(snapshot.pursuer.quat_xyzw)
    p_r = np.asarray(estimate.p_r_w, dtype=float).reshape(3)
    v_r = np.asarray(estimate.v_r_w, dtype=float).reshape(3)
    norm_pr = float(np.linalg.norm(p_r))
    if norm_pr < 1.0e-6:
        return _hover_command(instance)

    k_b = float(gains["k_b"])
    k_1 = float(gains["k_1"])
    k_2 = float(gains["k_2"])
    f_max = float(gains.get("f_max", DEFAULT_GAINS["f_max"]))
    omega_max = float(gains.get("omega_max", DEFAULT_GAINS["omega_max"]))
    if instance.config.max_thrust_n > 0.0:
        f_max = min(f_max, float(instance.config.max_thrust_n))
    if instance.config.max_rate_rps > 0.0:
        omega_max = min(omega_max, float(instance.config.max_rate_rps))

    n_t = np.asarray(estimate.bearing_w, dtype=float).reshape(3)
    if not np.all(np.isfinite(n_t)) or float(np.linalg.norm(n_t)) < 1.0e-9:
        n_t = -p_r / norm_pr
    else:
        n_t = n_t / max(float(np.linalg.norm(n_t)), 1.0e-12)

    r_b2c = np.asarray(camera.body_to_camera, dtype=float).reshape(3, 3)
    n_td_body = r_b2c.T @ np.array([1.0, 0.0, 0.0], dtype=float)
    n_td = r_wb @ n_td_body
    n_f = r_wb @ np.array([0.0, 0.0, 1.0], dtype=float)

    z_1 = 1.0 - float(n_td @ n_t)
    z_1 = float(np.clip(z_1, -0.99 * k_b, 0.99 * k_b))
    barrier = z_1 / (k_b**2 - z_1**2)
    b_omega_1 = barrier * (r_wb.T @ np.cross(n_td, n_t))

    z_2 = v_r + k_1 * p_r
    proj = -np.eye(3) + np.outer(n_t, n_t)
    a_d = (
        -k_1 * v_r
        - k_2 * z_2
        - p_r
        + barrier * (mass_kg / norm_pr) * (proj @ n_td)
    )
    drag = np.diag(np.asarray(gains["drag_diag"], dtype=float).reshape(3))
    v_w = np.asarray(snapshot.pursuer.velocity_w, dtype=float)
    e_f_drag = -r_wb @ drag @ r_wb.T @ v_w

    n_fd_raw = a_d - G_VEC - e_f_drag / mass_kg
    n_fd = n_fd_raw / max(float(np.linalg.norm(n_fd_raw)), 1.0e-9)
    r_tilt = _tilt_rotation(n_f, n_fd)
    r_d = r_tilt @ r_wb
    f_raw = float(n_f @ (mass_kg * a_d - mass_kg * G_VEC - e_f_drag))
    f_d = float(np.clip(f_raw, 0.0, f_max))

    s = r_d.T @ r_wb - r_wb.T @ r_d
    b_omega_2 = -vex(s)
    b_omega_d = b_omega_1 + b_omega_2
    n_w = float(np.linalg.norm(b_omega_d))
    if n_w > omega_max:
        b_omega_d = b_omega_d * (omega_max / n_w)
    return f_d, b_omega_d


def cautious_bearing_command(
    instance: SimInstance,
    snapshot: SimSnapshot,
    estimate: RelativeStateEstimate,
    gains: Mapping[str, float],
) -> tuple[float, np.ndarray]:
    if instance.config is None or not instance.config.cameras or not estimate.valid:
        return _hover_command(instance)

    mass_kg = float(instance.config.pursuer.mass_kg)
    r_wb = quat_xyzw_to_rot(snapshot.pursuer.quat_xyzw)
    camera = instance.config.cameras[0]
    r_b2c = np.asarray(camera.body_to_camera, dtype=float).reshape(3, 3)
    n_td_body = r_b2c.T @ np.array([1.0, 0.0, 0.0], dtype=float)
    n_td = r_wb @ n_td_body
    n_t = np.asarray(estimate.bearing_w, dtype=float).reshape(3)
    n_t = n_t / max(float(np.linalg.norm(n_t)), 1.0e-12)

    k_b = float(gains["k_b"])
    omega_max = float(gains.get("omega_max", DEFAULT_GAINS["omega_max"]))
    if instance.config.max_rate_rps > 0.0:
        omega_max = min(omega_max, float(instance.config.max_rate_rps))
    z_1 = 1.0 - float(n_td @ n_t)
    z_1 = float(np.clip(z_1, -0.99 * k_b, 0.99 * k_b))
    barrier = z_1 / (k_b**2 - z_1**2)
    body_rates = barrier * (r_wb.T @ np.cross(n_td, n_t))
    n_w = float(np.linalg.norm(body_rates))
    if n_w > omega_max:
        body_rates = body_rates * (omega_max / n_w)

    f_max = float(gains.get("f_max", DEFAULT_GAINS["f_max"]))
    if instance.config.max_thrust_n > 0.0:
        f_max = min(f_max, float(instance.config.max_thrust_n))
    closing_accel = float(gains.get("cautious_closing_accel_mps2", 4.0))
    damping = float(gains.get("cautious_velocity_damping", 0.25))
    pursuer_v_w = np.asarray(snapshot.pursuer.velocity_w, dtype=float).reshape(3)
    a_d = closing_accel * n_t - damping * pursuer_v_w
    drag = np.diag(np.asarray(gains["drag_diag"], dtype=float).reshape(3))
    e_f_drag = -r_wb @ drag @ r_wb.T @ pursuer_v_w
    n_f = r_wb @ np.array([0.0, 0.0, 1.0], dtype=float)
    n_fd_raw = a_d - G_VEC - e_f_drag / mass_kg
    n_fd = n_fd_raw / max(float(np.linalg.norm(n_fd_raw)), 1.0e-9)
    r_tilt = _tilt_rotation(n_f, n_fd)
    r_d = r_tilt @ r_wb
    s = r_d.T @ r_wb - r_wb.T @ r_d
    b_omega_2 = -vex(s)
    body_rates = body_rates + b_omega_2
    n_w = float(np.linalg.norm(body_rates))
    if n_w > omega_max:
        body_rates = body_rates * (omega_max / n_w)
    f_raw = float(n_f @ (mass_kg * a_d - mass_kg * G_VEC - e_f_drag))
    return float(np.clip(f_raw, 0.0, f_max)), body_rates


def quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
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
