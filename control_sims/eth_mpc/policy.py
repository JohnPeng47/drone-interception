from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from backends.csim.bindings.types import SimInstance, SimSnapshot
from backends.csim.runner import CtbrCommandBatch, SimControlPolicy, SimRunnerState

from .config import EthMpcConfig


G_VEC = np.array([0.0, 0.0, -9.81], dtype=float)
Z_B = np.array([0.0, 0.0, 1.0], dtype=float)


@dataclass
class _SlotMemory:
    previous_thrust_n: float = 0.0
    previous_body_rates_b: np.ndarray | None = None
    previous_solution: np.ndarray | None = None
    cached_command: tuple[float, np.ndarray] | None = None
    last_solve_step: int = -1_000_000


@dataclass(frozen=True)
class _PursuitPath:
    origin_w: np.ndarray
    tangent_w: np.ndarray
    normal_w: np.ndarray
    binormal_w: np.ndarray
    length_m: float
    width_m: float
    height_m: float


class EthMpcControlPolicy(SimControlPolicy):
    """MPCC++-style receding-horizon pursuit controller.

    MPCC++ optimizes progress along a centerline while penalizing lag and
    contour errors, and constrains the vehicle to a prismatic tunnel around
    that centerline. This implementation adapts the track centerline to a
    target-interception corridor and solves a small nonlinear OCP at MPCC-like
    rate, holding the command between solves for the faster SimEngine action
    ticks.
    """

    def __init__(self, config: EthMpcConfig | None = None):
        self.config = config or EthMpcConfig()
        self._slots: dict[int, _SlotMemory] = {}

    def reset(self, state: SimRunnerState) -> None:
        self._slots.clear()

    def on_slots_started(self, slots: np.ndarray, instances, state: SimRunnerState) -> None:
        for slot in np.asarray(slots, dtype=np.int64).reshape(-1):
            slot_i = int(slot)
            instance = state.instances[slot_i]
            thrust = 0.0
            if instance is not None and instance.config is not None:
                thrust = float(instance.config.pursuer.mass_kg * instance.config.pursuer.gravity_mps2)
            self._slots[slot_i] = _SlotMemory(
                previous_thrust_n=thrust,
                previous_body_rates_b=np.zeros(3, dtype=float),
                cached_command=None,
                last_solve_step=-1_000_000,
            )

    def command(self, state: SimRunnerState) -> CtbrCommandBatch:
        thrust_n = np.zeros(len(state.instances), dtype=np.float32)
        body_rates_b = np.zeros((len(state.instances), 3), dtype=np.float32)
        for slot, instance in enumerate(state.instances):
            if instance is None or not bool(state.active[slot]):
                continue
            memory = self._slots.setdefault(slot, _SlotMemory())
            command = self._command_one(
                instance,
                state.snapshot[slot],
                memory,
                step=int(state.steps[slot]),
            )
            memory.previous_thrust_n = float(command[0])
            memory.previous_body_rates_b = np.asarray(command[1], dtype=float).reshape(3)
            memory.cached_command = (float(command[0]), np.asarray(command[1], dtype=float).reshape(3).copy())
            thrust_n[slot] = np.float32(command[0])
            body_rates_b[slot] = np.asarray(command[1], dtype=np.float32).reshape(3)
        return CtbrCommandBatch(thrust_n=thrust_n, body_rates_b=body_rates_b)

    def _command_one(
        self,
        instance: SimInstance,
        snapshot: SimSnapshot,
        memory: _SlotMemory,
        *,
        step: int,
    ) -> tuple[float, np.ndarray]:
        if instance.config is None:
            return 0.0, np.zeros(3, dtype=float)

        solve_interval = max(1, int(round(float(self.config.solve_period_s) / _dt_from_instance(instance))))
        if memory.cached_command is not None and step - int(memory.last_solve_step) < solve_interval:
            return memory.cached_command

        path = _build_pursuit_path(snapshot, self.config)
        if path.length_m < 1.0e-6:
            return _hover_command(instance)

        solution = self._solve_ocp(instance, snapshot, path, memory)
        memory.previous_solution = solution.copy()
        memory.last_solve_step = int(step)
        first = solution.reshape(int(self.config.horizon_steps), 5)[0]
        return float(first[0]), np.asarray(first[1:4], dtype=float).reshape(3)

    def _solve_ocp(
        self,
        instance: SimInstance,
        snapshot: SimSnapshot,
        path: _PursuitPath,
        memory: _SlotMemory,
    ) -> np.ndarray:
        cfg = self.config
        n = int(cfg.horizon_steps)
        if memory.previous_solution is None or memory.previous_solution.size != 5 * n:
            x0 = _initial_guess(instance, snapshot, path, cfg)
        else:
            previous = memory.previous_solution.reshape(n, 5)
            shifted = np.vstack([previous[1:], previous[-1:]])
            x0 = shifted.reshape(-1)

        bounds = []
        thrust_max = _max_thrust_n(instance)
        rate_max = _max_rate_rps(instance, snapshot)
        for _ in range(n):
            bounds.append((0.0, thrust_max))
            bounds.extend([(-rate_max, rate_max)] * 3)
            bounds.append((0.0, float(cfg.max_progress_speed_mps)))

        result = minimize(
            _objective,
            x0,
            args=(instance, snapshot, path, cfg),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": int(cfg.optimizer_maxiter),
                "ftol": 1.0e-3,
                "maxls": 8,
            },
        )
        if not result.success and result.x is None:
            return x0
        return np.asarray(result.x, dtype=float).reshape(-1)


def _objective(
    decision: np.ndarray,
    instance: SimInstance,
    snapshot: SimSnapshot,
    path: _PursuitPath,
    cfg: EthMpcConfig,
) -> float:
    n = int(cfg.horizon_steps)
    values = np.asarray(decision, dtype=float).reshape(n, 5)
    thrust = values[:, 0]
    body_rates = values[:, 1:4]
    v_theta = values[:, 4]
    dt_s = float(cfg.horizon_dt_s)

    p0 = np.asarray(snapshot.pursuer.position_w, dtype=float).reshape(3)
    p = p0.copy()
    v = np.asarray(snapshot.pursuer.velocity_w, dtype=float).reshape(3)
    q = _normalize_quat_xyzw(np.asarray(snapshot.pursuer.quat_xyzw, dtype=float).reshape(4))
    omega = np.asarray(snapshot.pursuer.body_rates_b, dtype=float).reshape(3)
    target_p0 = np.asarray(snapshot.target.position_w, dtype=float).reshape(3)
    target_v = np.asarray(snapshot.target.velocity_w, dtype=float).reshape(3)
    mass_kg = float(instance.config.pursuer.mass_kg) if instance.config is not None else 1.0
    drag_diag = np.asarray(cfg.drag_diag, dtype=float).reshape(3)

    theta = float(np.clip((p - path.origin_w) @ path.tangent_w, 0.0, path.length_m))
    previous_thrust = float(snapshot.thrust_n) if snapshot.thrust_n is not None else mass_kg * 9.81
    previous_rates = omega.copy()
    cost = 0.0
    for k in range(n):
        thrust_k = float(thrust[k])
        rates_k = np.asarray(body_rates[k], dtype=float).reshape(3)
        q = _integrate_quat_xyzw(q, rates_k, dt_s)
        R_wb = _quat_xyzw_to_rot(q)
        body_drag = -drag_diag * (R_wb.T @ v)
        drag_w = R_wb @ body_drag
        accel_w = G_VEC + R_wb @ (Z_B * (thrust_k / max(mass_kg, 1.0e-9))) + drag_w / max(mass_kg, 1.0e-9)
        v = _clip_norm(v + accel_w * dt_s, float(cfg.max_pred_speed_mps))
        p = p + v * dt_s
        omega = rates_k
        theta_next = theta + float(v_theta[k]) * dt_s
        terminal_progress_violation = max(0.0, theta_next - path.length_m)
        theta = float(np.clip(theta_next, 0.0, path.length_m))

        path_point = path.origin_w + theta * path.tangent_w
        position_error = p - path_point
        lag_error = float(position_error @ path.tangent_w)
        contour_n = float(position_error @ path.normal_w)
        contour_b = float(position_error @ path.binormal_w)
        contour_error_sq = contour_n * contour_n + contour_b * contour_b
        target_p = target_p0 + target_v * ((k + 1) * dt_s)
        target_error = float(np.linalg.norm(p - target_p))

        n_violation = max(0.0, abs(contour_n) - path.width_m)
        b_violation = max(0.0, abs(contour_b) - path.height_m)
        cost += (
            float(cfg.q_lag) * lag_error * lag_error
            + float(cfg.q_contour) * contour_error_sq
            + float(cfg.q_velocity) * float(v @ v)
            + float(cfg.q_body_rate) * float(omega @ omega)
            + float(cfg.q_thrust_rate) * (thrust_k - previous_thrust) * (thrust_k - previous_thrust)
            + float(cfg.q_accel_rate) * float((rates_k - previous_rates) @ (rates_k - previous_rates))
            + float(cfg.q_progress_rate) * float(v_theta[k] * v_theta[k])
            - float(cfg.progress_reward) * float(v_theta[k])
            + float(cfg.q_tunnel) * (n_violation * n_violation + b_violation * b_violation)
            + float(cfg.q_terminal_set) * terminal_progress_violation * terminal_progress_violation
        )
        if instance.config is not None and instance.config.intercept_radius_m > 0.0:
            intercept_error = max(0.0, target_error - float(instance.config.intercept_radius_m))
            cost += float(cfg.intercept_radius_weight) * intercept_error
        previous_thrust = thrust_k
        previous_rates = rates_k

    terminal_target = target_p0 + target_v * (n * dt_s)
    terminal_error = float(np.linalg.norm(p - terminal_target))
    terminal_contour = p - (path.origin_w + path.length_m * path.tangent_w)
    terminal_lag = float(terminal_contour @ path.tangent_w)
    terminal_cross = terminal_contour - terminal_lag * path.tangent_w
    cost += float(cfg.q_terminal) * terminal_error * terminal_error
    cost += float(cfg.q_terminal_set) * float(terminal_cross @ terminal_cross)
    return float(cost)


def _initial_guess(
    instance: SimInstance,
    snapshot: SimSnapshot,
    path: _PursuitPath,
    cfg: EthMpcConfig,
) -> np.ndarray:
    n = int(cfg.horizon_steps)
    p = np.asarray(snapshot.pursuer.position_w, dtype=float).reshape(3)
    v = np.asarray(snapshot.pursuer.velocity_w, dtype=float).reshape(3)
    target_p = np.asarray(snapshot.target.position_w, dtype=float).reshape(3)
    target_v = np.asarray(snapshot.target.velocity_w, dtype=float).reshape(3)
    desired_speed = min(float(cfg.approach_speed_mps), path.length_m / max(n * float(cfg.horizon_dt_s), 1.0e-6))
    desired_v = target_v + desired_speed * path.tangent_w
    accel0 = 1.4 * (desired_v - v) + 0.25 * (target_p - p)
    accel0 = _clip_norm(accel0, float(cfg.max_accel_mps2))
    progress0 = min(float(cfg.max_progress_speed_mps), max(0.0, desired_speed))
    thrust0, rates0 = _accel_to_ctbr(
        instance,
        snapshot,
        accel0,
        R_wb=_quat_xyzw_to_rot(snapshot.pursuer.quat_xyzw),
        rate_gain=float(cfg.attitude_rate_gain),
        drag_diag=np.asarray(cfg.drag_diag, dtype=float),
    )
    decision = np.zeros((n, 5), dtype=float)
    decision[:, 0] = thrust0
    decision[:, 1:4] = rates0
    decision[:, 4] = progress0
    return decision.reshape(-1)


def _build_pursuit_path(snapshot: SimSnapshot, cfg: EthMpcConfig) -> _PursuitPath:
    p = np.asarray(snapshot.pursuer.position_w, dtype=float).reshape(3)
    target_p = np.asarray(snapshot.target.position_w, dtype=float).reshape(3)
    target_v = np.asarray(snapshot.target.velocity_w, dtype=float).reshape(3)
    terminal_target = target_p + target_v * float(cfg.target_lookahead_s)
    centerline = terminal_target - p
    tangent = _unit(centerline, fallback=np.array([1.0, 0.0, 0.0], dtype=float))
    normal, binormal = _orthonormal_basis(tangent)
    length_m = max(float(np.linalg.norm(centerline)), 1.0e-6)
    width_m = max(float(cfg.min_tunnel_width_m), min(float(cfg.tunnel_radius_m), 0.35 * length_m))
    height_m = width_m
    return _PursuitPath(
        origin_w=p,
        tangent_w=tangent,
        normal_w=normal,
        binormal_w=binormal,
        length_m=length_m,
        width_m=width_m,
        height_m=height_m,
    )


def _accel_to_ctbr(
    instance: SimInstance,
    snapshot: SimSnapshot,
    accel_w: np.ndarray,
    *,
    R_wb: np.ndarray,
    rate_gain: float,
    drag_diag: np.ndarray,
) -> tuple[float, np.ndarray]:
    assert instance.config is not None
    mass_kg = float(instance.config.pursuer.mass_kg)
    n_f = R_wb @ Z_B
    v_w = np.asarray(snapshot.pursuer.velocity_w, dtype=float).reshape(3)
    drag = np.diag(np.asarray(drag_diag, dtype=float).reshape(3))
    e_f_drag = -R_wb @ drag @ R_wb.T @ v_w
    desired_specific_force = np.asarray(accel_w, dtype=float).reshape(3) - G_VEC - e_f_drag / max(mass_kg, 1.0e-9)
    force_norm = float(np.linalg.norm(desired_specific_force))
    if force_norm <= 1.0e-9:
        return _hover_command(instance)

    n_fd = desired_specific_force / force_norm
    R_tilt = _tilt_rotation(n_f, n_fd)
    R_d = R_tilt @ R_wb
    thrust = float(n_f @ (mass_kg * np.asarray(accel_w, dtype=float).reshape(3) - mass_kg * G_VEC - e_f_drag))
    if instance.config.max_thrust_n > 0.0:
        thrust = float(np.clip(thrust, 0.0, float(instance.config.max_thrust_n)))
    else:
        thrust = max(0.0, thrust)

    S = R_d.T @ R_wb - R_wb.T @ R_d
    body_rates_b = -float(rate_gain) * _vex(S)
    max_rate = float(instance.config.max_rate_rps)
    if max_rate <= 0.0:
        max_rate = float(snapshot.max_rate_rps)
    rate_norm = float(np.linalg.norm(body_rates_b))
    if max_rate > 0.0 and rate_norm > max_rate:
        body_rates_b = body_rates_b * (max_rate / rate_norm)
    return thrust, body_rates_b


def _hover_command(instance: SimInstance) -> tuple[float, np.ndarray]:
    if instance.config is None:
        return 0.0, np.zeros(3, dtype=float)
    params = instance.config.pursuer
    thrust = float(params.mass_kg * params.gravity_mps2)
    if instance.config.max_thrust_n > 0.0:
        thrust = min(thrust, float(instance.config.max_thrust_n))
    return thrust, np.zeros(3, dtype=float)


def _dt_from_instance(instance: SimInstance) -> float:
    assert instance.config is not None
    return float(instance.config.options.backend_dt) * max(1, int(instance.config.options.action_substeps))


def _max_thrust_n(instance: SimInstance) -> float:
    assert instance.config is not None
    if instance.config.max_thrust_n > 0.0:
        return float(instance.config.max_thrust_n)
    return float(instance.config.pursuer.mass_kg * instance.config.pursuer.gravity_mps2 * 4.0)


def _max_rate_rps(instance: SimInstance, snapshot: SimSnapshot) -> float:
    assert instance.config is not None
    if instance.config.max_rate_rps > 0.0:
        return float(instance.config.max_rate_rps)
    if snapshot.max_rate_rps > 0.0:
        return float(snapshot.max_rate_rps)
    return 8.0


def _normalize_quat_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / norm


def _integrate_quat_xyzw(q_xyzw: np.ndarray, body_rates_b: np.ndarray, dt_s: float) -> np.ndarray:
    x, y, z, w = _normalize_quat_xyzw(q_xyzw)
    wx, wy, wz = np.asarray(body_rates_b, dtype=float).reshape(3)
    q_dot = 0.5 * np.array([
        w * wx + y * wz - z * wy,
        w * wy + z * wx - x * wz,
        w * wz + x * wy - y * wx,
        -x * wx - y * wy - z * wz,
    ])
    return _normalize_quat_xyzw(np.array([x, y, z, w]) + q_dot * float(dt_s))


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


def _vex(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def _unit(vector: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-9:
        fb = np.asarray(fallback, dtype=float).reshape(3)
        fb_norm = float(np.linalg.norm(fb))
        return fb / max(fb_norm, 1.0e-9)
    return arr / norm


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= float(max_norm) or norm <= 1.0e-12:
        return arr
    return arr * (float(max_norm) / norm)


def _orthonormal_basis(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = _unit(tangent, fallback=np.array([1.0, 0.0, 0.0], dtype=float))
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(t @ helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    b1 = _unit(np.cross(t, helper), fallback=np.array([0.0, 1.0, 0.0], dtype=float))
    b2 = _unit(np.cross(t, b1), fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    return b1, b2
