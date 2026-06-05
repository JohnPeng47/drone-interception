from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from backends.csim.bindings import BatchPufferSimEngineBackend
from backends.csim.bindings.types import PursuerParams, SimInstance


STATE_SIZE = 17
CONTROL_SIZE = 4


@dataclass(frozen=True)
class NumericRolloutTrajectory:
    total_time_s: float
    dt_s: float
    t_s: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    rollout_wall_s: float

    @property
    def position_w(self) -> np.ndarray:
        return self.states[:, 0:3]

    @property
    def velocity_w(self) -> np.ndarray:
        return self.states[:, 3:6]

    @property
    def quat_wxyz(self) -> np.ndarray:
        return self.states[:, 6:10]

    @property
    def body_rates_b(self) -> np.ndarray:
        return self.states[:, 10:13]

    @property
    def motor_rpm(self) -> np.ndarray:
        return self.states[:, 13:17]


@dataclass(frozen=True)
class NumericRolloutMetrics:
    terminal_position_error_m: float
    position_error_mean_m: float
    position_error_max_m: float
    min_target_distance_m: float
    final_target_distance_m: float
    rpm_min: float
    rpm_max: float
    body_rate_abs_max_rps: float
    altitude_min_m: float
    altitude_max_m: float


@dataclass(frozen=True)
class SimEngineReplayMetrics:
    replay_wall_s: float
    min_target_distance_m: float
    final_target_distance_m: float
    steps: int
    caught: bool


def rollout_motor_commands(
    instance: SimInstance,
    controls_rpm: np.ndarray,
    total_time_s: float,
    *,
    dynamics_substeps: int = 1,
    control_layout: str = "auto",
) -> NumericRolloutTrajectory:
    if instance.config is None:
        raise ValueError("numeric rollout requires SimInstance.config")
    controls = _controls_as_rows(controls_rpm, layout=control_layout)
    nodes = int(controls.shape[0])
    if nodes <= 0:
        raise ValueError("controls must contain at least one node")
    total_time = float(total_time_s)
    if not np.isfinite(total_time) or total_time <= 0.0:
        raise ValueError("total_time_s must be finite and positive")
    dt = total_time / float(nodes)
    params = instance.config.pursuer
    controls = np.clip(controls, _min_rpm(params), float(params.max_rpm))
    state = _initial_state_wxyz_rpm(instance)
    states = np.empty((nodes + 1, STATE_SIZE), dtype=np.float64)
    states[0] = state
    started = time.perf_counter()
    for index in range(nodes):
        state = _simengine_like_motor_lag_step_np(
            state,
            controls[index],
            dt,
            params,
            substeps=dynamics_substeps,
        )
        states[index + 1] = state
    rollout_wall_s = time.perf_counter() - started
    return NumericRolloutTrajectory(
        total_time_s=total_time,
        dt_s=dt,
        t_s=np.linspace(0.0, total_time, nodes + 1, dtype=np.float64),
        states=states,
        controls=controls.copy(),
        rollout_wall_s=rollout_wall_s,
    )


def compare_to_reference_plan(instance: SimInstance, trajectory: NumericRolloutTrajectory, plan: Any) -> NumericRolloutMetrics:
    reference_position = np.asarray(plan.position_w, dtype=np.float64).T
    if reference_position.shape != trajectory.position_w.shape:
        raise ValueError(
            "reference plan position shape does not match rollout: "
            f"{reference_position.shape} != {trajectory.position_w.shape}"
        )
    position_error = np.linalg.norm(trajectory.position_w - reference_position, axis=1)
    target_distances = target_distances_for_trajectory(instance, trajectory)
    return NumericRolloutMetrics(
        terminal_position_error_m=float(position_error[-1]),
        position_error_mean_m=float(np.mean(position_error)),
        position_error_max_m=float(np.max(position_error)),
        min_target_distance_m=float(np.min(target_distances)),
        final_target_distance_m=float(target_distances[-1]),
        rpm_min=float(np.min(trajectory.motor_rpm)),
        rpm_max=float(np.max(trajectory.motor_rpm)),
        body_rate_abs_max_rps=float(np.max(np.abs(trajectory.body_rates_b))),
        altitude_min_m=float(np.min(trajectory.position_w[:, 2])),
        altitude_max_m=float(np.max(trajectory.position_w[:, 2])),
    )


def replay_motor_commands_in_simengine(
    instance: SimInstance,
    controls_rpm: np.ndarray,
    total_time_s: float,
    *,
    control_layout: str = "auto",
) -> SimEngineReplayMetrics:
    if instance.config is None:
        raise ValueError("SimEngine replay requires SimInstance.config")
    controls = _controls_as_rows(controls_rpm, layout=control_layout)
    if int(controls.shape[0]) <= 0:
        raise ValueError("controls must contain at least one node")
    backend_dt = float(instance.config.options.backend_dt) * max(1, int(instance.config.options.action_substeps))
    if not np.isfinite(backend_dt) or backend_dt <= 0.0:
        raise ValueError("effective backend dt must be finite and positive")
    backend = BatchPufferSimEngineBackend(1)
    snapshots = backend.reset_many(np.array([0], dtype=np.int64), (instance,))
    total_time = float(total_time_s)
    if not np.isfinite(total_time) or total_time <= 0.0:
        raise ValueError("total_time_s must be finite and positive")
    control_interval_s = total_time / float(len(controls))
    time_tol = max(1.0e-12 * max(1.0, total_time, backend_dt, control_interval_s), np.finfo(np.float64).eps)
    elapsed_s = 0.0
    command_index = 0
    steps = 0
    distances: list[float] = []
    started = time.perf_counter()
    while elapsed_s < total_time - time_tol:
        snapshot = snapshots[0]
        position = np.asarray(snapshot.pursuer.position_w, dtype=np.float64).reshape(3)
        target = np.asarray(snapshot.target.position_w, dtype=np.float64).reshape(3)
        distances.append(float(np.linalg.norm(position - target)))
        while (
            command_index < len(controls) - 1
            and elapsed_s >= ((command_index + 1) * control_interval_s) - time_tol
        ):
            command_index += 1
        next_control_boundary_s = min((command_index + 1) * control_interval_s, total_time)
        boundary_remaining_s = next_control_boundary_s - elapsed_s
        if boundary_remaining_s <= time_tol:
            if command_index < len(controls) - 1:
                elapsed_s = max(elapsed_s, next_control_boundary_s)
                command_index += 1
                continue
            break
        command = controls[command_index]
        step_dt = min(backend_dt, total_time - elapsed_s, boundary_remaining_s)
        if not np.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError(f"invalid replay step dt: {step_dt}")
        snapshots = backend.step_motor_speeds_many_dt(command.reshape(1, CONTROL_SIZE), step_dt)
        elapsed_s += step_dt
        steps += 1
    snapshot = snapshots[0]
    position = np.asarray(snapshot.pursuer.position_w, dtype=np.float64).reshape(3)
    target = np.asarray(snapshot.target.position_w, dtype=np.float64).reshape(3)
    distances.append(float(np.linalg.norm(position - target)))
    replay_wall_s = time.perf_counter() - started
    min_distance = float(np.min(distances)) if distances else math.inf
    return SimEngineReplayMetrics(
        replay_wall_s=replay_wall_s,
        min_target_distance_m=min_distance,
        final_target_distance_m=float(distances[-1]) if distances else math.inf,
        steps=int(steps),
        caught=bool(min_distance <= float(instance.config.intercept_radius_m)),
    )


def target_distances_for_trajectory(instance: SimInstance, trajectory: NumericRolloutTrajectory) -> np.ndarray:
    target_position = np.asarray(instance.target_initial.position_w, dtype=np.float64).reshape(3)
    target_velocity = np.asarray(instance.target_initial.velocity_w, dtype=np.float64).reshape(3)
    target_positions = target_position.reshape(1, 3) + trajectory.t_s.reshape(-1, 1) * target_velocity.reshape(1, 3)
    return np.linalg.norm(trajectory.position_w - target_positions, axis=1)


def _controls_as_rows(controls_rpm: np.ndarray, *, layout: str = "auto") -> np.ndarray:
    controls = np.asarray(controls_rpm, dtype=np.float64)
    if controls.ndim != 2:
        raise ValueError("controls must be a 2D array")
    if not np.all(np.isfinite(controls)):
        raise ValueError("controls must be finite")
    if layout not in {"auto", "rows", "columns"}:
        raise ValueError("control_layout must be 'auto', 'rows', or 'columns'")
    if layout == "rows":
        if controls.shape[1] != CONTROL_SIZE:
            raise ValueError(f"row-oriented controls must have shape (N, {CONTROL_SIZE}); got {controls.shape}")
        return controls.copy()
    if layout == "columns":
        if controls.shape[0] != CONTROL_SIZE:
            raise ValueError(f"column-oriented controls must have shape ({CONTROL_SIZE}, N); got {controls.shape}")
        return controls.T.copy()
    if controls.shape == (CONTROL_SIZE, CONTROL_SIZE):
        raise ValueError("ambiguous 4x4 controls require control_layout='rows' or 'columns'")
    if controls.shape[1] == CONTROL_SIZE:
        return controls.copy()
    if controls.shape[0] == CONTROL_SIZE:
        return controls.T.copy()
    raise ValueError(f"controls must have one axis of length {CONTROL_SIZE}; got {controls.shape}")


def _simengine_like_motor_lag_step_np(
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    params: PursuerParams,
    *,
    substeps: int,
) -> np.ndarray:
    state = np.asarray(x, dtype=np.float64).reshape(STATE_SIZE)
    command = np.asarray(u, dtype=np.float64).reshape(CONTROL_SIZE)
    count = max(1, int(substeps))
    sub_dt = float(dt) / float(count)
    for _ in range(count):
        state = _rk4_motor_lag_step_np(state, command, sub_dt, params)
        state = _apply_simengine_state_clamps_np(state, params)
    return state


def _rk4_motor_lag_step_np(x: np.ndarray, u: np.ndarray, dt: float, params: PursuerParams) -> np.ndarray:
    k1 = _motor_lag_dynamics_np(x, u, params)
    k2 = _motor_lag_dynamics_np(_normalize_state_quat_np(x + 0.5 * dt * k1), u, params)
    k3 = _motor_lag_dynamics_np(_normalize_state_quat_np(x + 0.5 * dt * k2), u, params)
    k4 = _motor_lag_dynamics_np(_normalize_state_quat_np(x + dt * k3), u, params)
    return _normalize_state_quat_np(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def _motor_lag_dynamics_np(x: np.ndarray, u: np.ndarray, params: PursuerParams) -> np.ndarray:
    velocity = x[3:6]
    quat = _normalize_quat_wxyz_np(x[6:10])
    omega = x[10:13]
    rpm = x[13:17]
    thrusts = float(params.k_thrust) * np.square(rpm)
    thrust_total = float(np.sum(thrusts))
    force_w = _quat_rotate_wxyz_np(quat, np.array([0.0, 0.0, thrust_total], dtype=np.float64))
    accel_w = force_w / float(params.mass_kg)
    if float(params.b_drag) != 0.0:
        accel_w += (-float(params.b_drag) * velocity) / float(params.mass_kg)
    accel_w += np.array([0.0, 0.0, -float(params.gravity_mps2)], dtype=np.float64)

    qdot = 0.5 * _quat_mul_wxyz_np(quat, np.array([0.0, *omega], dtype=np.float64))
    moment = _sim_moment_from_thrusts_np(thrusts, params)
    if float(params.k_ang_damp) != 0.0:
        moment += -float(params.k_ang_damp) * omega
    wx, wy, wz = omega
    ixx, iyy, izz = float(params.ixx), float(params.iyy), float(params.izz)
    inertial = np.array([(iyy - izz) * wy * wz, (izz - ixx) * wz * wx, (ixx - iyy) * wx * wy], dtype=np.float64)
    omega_dot = np.array(
        [
            (moment[0] + inertial[0]) / ixx,
            (moment[1] + inertial[1]) / iyy,
            (moment[2] + inertial[2]) / izz,
        ],
        dtype=np.float64,
    )
    rpm_dot = (u - rpm) / max(float(params.motor_tau_s), 1.0e-6)
    return np.concatenate([velocity, accel_w, qdot, omega_dot, rpm_dot])


def _sim_moment_from_thrusts_np(thrusts: np.ndarray, params: PursuerParams) -> np.ndarray:
    rotor_positions = params.rotor_positions_b
    rotor_directions = params.rotor_directions
    if rotor_positions is None or rotor_directions is None:
        arm_factor = float(params.arm_len_m) / np.sqrt(2.0)
        return np.array(
            [
                arm_factor * ((thrusts[2] + thrusts[3]) - (thrusts[0] + thrusts[1])),
                arm_factor * ((thrusts[1] + thrusts[2]) - (thrusts[0] + thrusts[3])),
                float(params.k_yaw) * (-thrusts[0] + thrusts[1] - thrusts[2] + thrusts[3]),
            ],
            dtype=np.float64,
        )
    rotor_positions_arr = np.asarray(rotor_positions, dtype=np.float64).reshape(CONTROL_SIZE, 3)
    rotor_directions_arr = np.asarray(rotor_directions, dtype=np.float64).reshape(CONTROL_SIZE)
    return np.array(
        [
            float(np.dot(rotor_positions_arr[:, 1], thrusts)),
            float(np.dot(-rotor_positions_arr[:, 0], thrusts)),
            float(np.dot(float(params.k_yaw) * rotor_directions_arr, thrusts)),
        ],
        dtype=np.float64,
    )


def _apply_simengine_state_clamps_np(x: np.ndarray, params: PursuerParams) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).reshape(STATE_SIZE).copy()
    out[3:6] = np.clip(out[3:6], -float(params.max_vel_mps), float(params.max_vel_mps))
    out[6:10] = _normalize_quat_wxyz_np(out[6:10])
    out[10:13] = np.clip(out[10:13], -float(params.max_omega_rps), float(params.max_omega_rps))
    out[13:17] = np.clip(out[13:17], 0.0, float(params.max_rpm))
    return out


def _normalize_state_quat_np(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).reshape(STATE_SIZE).copy()
    out[6:10] = _normalize_quat_wxyz_np(out[6:10])
    return out


def _normalize_quat_wxyz_np(q: np.ndarray) -> np.ndarray:
    quat = np.asarray(q, dtype=np.float64).reshape(4)
    return quat / np.sqrt(float(np.dot(quat, quat)) + 1.0e-12)


def _quat_rotate_wxyz_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    rotated = _quat_mul_wxyz_np(
        _quat_mul_wxyz_np(q, np.array([0.0, *np.asarray(v, dtype=np.float64).reshape(3)], dtype=np.float64)),
        np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64),
    )
    return rotated[1:4]


def _quat_mul_wxyz_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64).reshape(4)
    bb = np.asarray(b, dtype=np.float64).reshape(4)
    return np.array(
        [
            aa[0] * bb[0] - aa[1] * bb[1] - aa[2] * bb[2] - aa[3] * bb[3],
            aa[0] * bb[1] + aa[1] * bb[0] + aa[2] * bb[3] - aa[3] * bb[2],
            aa[0] * bb[2] - aa[1] * bb[3] + aa[2] * bb[0] + aa[3] * bb[1],
            aa[0] * bb[3] + aa[1] * bb[2] - aa[2] * bb[1] + aa[3] * bb[0],
        ],
        dtype=np.float64,
    )


def _initial_state_wxyz_rpm(instance: SimInstance) -> np.ndarray:
    if instance.config is None:
        raise ValueError("numeric rollout requires SimInstance.config")
    params = instance.config.pursuer
    q_xyzw = np.asarray(instance.pursuer_initial.quat_xyzw, dtype=np.float64).reshape(4)
    base = [
        *np.asarray(instance.pursuer_initial.position_w, dtype=np.float64).reshape(3),
        *np.asarray(instance.pursuer_initial.velocity_w, dtype=np.float64).reshape(3),
        float(q_xyzw[3]),
        float(q_xyzw[0]),
        float(q_xyzw[1]),
        float(q_xyzw[2]),
        *np.asarray(instance.pursuer_initial.body_rates_b, dtype=np.float64).reshape(3),
    ]
    if instance.pursuer_initial.rotor_speeds is None:
        rpm = np.full(CONTROL_SIZE, _hover_rpm(params), dtype=np.float64)
    else:
        rpm = np.asarray(instance.pursuer_initial.rotor_speeds, dtype=np.float64).reshape(CONTROL_SIZE)
    rpm = np.clip(rpm, _min_rpm(params), float(params.max_rpm))
    return np.array([*base, *rpm], dtype=np.float64)


def _hover_rpm(params: PursuerParams) -> float:
    return float(np.sqrt((float(params.mass_kg) * float(params.gravity_mps2)) / (4.0 * float(params.k_thrust))))


def _min_rpm(params: PursuerParams) -> float:
    if params.rpm_min is not None:
        return float(np.clip(params.rpm_min, 0.0, params.max_rpm))
    min_rpm = 2.0 * _hover_rpm(params) - float(params.max_rpm)
    return float(np.clip(min_rpm, 0.0, float(params.max_rpm)))
