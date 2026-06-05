from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from backends.csim.bindings import BatchPufferSimEngineBackend
from backends.csim.bindings.types import PursuerParams, SimInstance


@dataclass(frozen=True)
class SwitchingTemplateConfig:
    min_time_s: float = 0.6
    max_time_s: float = 1.5
    time_step_s: float = 0.15
    thrust_fractions: tuple[float, ...] = (0.55, 0.7, 0.9, 1.0)
    rate_fractions: tuple[float, ...] = (0.45, 0.7, 0.95)
    first_switch_fractions: tuple[float, ...] = (0.08, 0.18, 0.32)
    second_switch_fractions: tuple[float, ...] = (0.55, 0.8, 1.0)
    counter_rate_fractions: tuple[float, ...] = (0.0, 0.5, 1.0)
    vertical_bias_gains: tuple[float, ...] = (-0.4, 0.0, 0.4, 0.8)
    direction_signs: tuple[float, ...] = (1.0, -1.0)
    velocity_gain: float = 0.25
    replay_top_k: int = 2
    replay_sample_dt_s: float | None = None
    screen_replay_margin_m: float = 0.5


@dataclass(frozen=True)
class SwitchingTemplateCandidate:
    steps: int
    total_time_s: float
    thrust_fraction: float
    rate_fraction: float
    first_switch_fraction: float
    second_switch_fraction: float
    counter_rate_fraction: float
    vertical_bias_gain: float
    direction_sign: float
    axis_b: tuple[float, float, float]


@dataclass(frozen=True)
class SwitchingTemplateResult:
    seed: int
    wall_s: float
    caught: bool
    fastest_caught_time_s: float
    catch_time_s: float
    min_distance_m: float
    final_distance_m: float
    templates_evaluated: int
    time_groups_evaluated: int
    best_candidate: SwitchingTemplateCandidate | None
    best_uncaught_candidate: SwitchingTemplateCandidate | None
    distance_source: str
    error: str = ""


def find_switching_template_intercept(
    instance: SimInstance,
    config: SwitchingTemplateConfig | None = None,
) -> SwitchingTemplateResult:
    if instance.config is None:
        raise ValueError("switching-template search requires SimInstance.config")
    cfg = config or SwitchingTemplateConfig()
    _validate_config(cfg)
    _validate_instance_contract(instance)
    started = time.perf_counter()
    templates_evaluated = 0
    groups_evaluated = 0
    best_uncaught_candidate: SwitchingTemplateCandidate | None = None
    best_uncaught_min_distance = float("inf")
    best_uncaught_final_distance = float("inf")

    replay_sample_dt = _backend_dt(instance) if cfg.replay_sample_dt_s is None else float(cfg.replay_sample_dt_s)
    for steps, total_time_s in _time_grid(instance, cfg):
        candidates = _templates_for_steps(instance, cfg, steps, total_time_s)
        groups_evaluated += 1
        templates_evaluated += len(candidates)
        group = _evaluate_template_group(
            instance,
            candidates,
            replay_top_k=int(cfg.replay_top_k),
            replay_sample_dt_s=float(replay_sample_dt),
            screen_replay_margin_m=float(cfg.screen_replay_margin_m),
        )
        if group.best_uncaught_min_distance < best_uncaught_min_distance:
            best_uncaught_min_distance = group.best_uncaught_min_distance
            best_uncaught_final_distance = group.best_uncaught_final_distance
            best_uncaught_candidate = group.best_uncaught_candidate
        if group.caught_candidate is not None:
            catch_time_s = float(group.catch_step) * _backend_dt(instance)
            return SwitchingTemplateResult(
                seed=int(instance.seed),
                wall_s=time.perf_counter() - started,
                caught=True,
                fastest_caught_time_s=float(group.caught_candidate.total_time_s),
                catch_time_s=float(catch_time_s),
                min_distance_m=float(group.caught_min_distance),
                final_distance_m=float(group.caught_final_distance),
                templates_evaluated=int(templates_evaluated),
                time_groups_evaluated=int(groups_evaluated),
                best_candidate=group.caught_candidate,
                best_uncaught_candidate=best_uncaught_candidate,
                distance_source="simengine_replay",
                error="",
            )

    return SwitchingTemplateResult(
        seed=int(instance.seed),
        wall_s=time.perf_counter() - started,
        caught=False,
        fastest_caught_time_s=float("nan"),
        catch_time_s=float("nan"),
        min_distance_m=float(best_uncaught_min_distance),
        final_distance_m=float(best_uncaught_final_distance),
        templates_evaluated=int(templates_evaluated),
        time_groups_evaluated=int(groups_evaluated),
        best_candidate=None,
        best_uncaught_candidate=best_uncaught_candidate,
        distance_source="screen_estimate",
        error="",
    )


@dataclass(frozen=True)
class _GroupResult:
    caught_candidate: SwitchingTemplateCandidate | None
    catch_step: int
    caught_min_distance: float
    caught_final_distance: float
    best_uncaught_candidate: SwitchingTemplateCandidate | None
    best_uncaught_min_distance: float
    best_uncaught_final_distance: float


def _evaluate_template_group(
    instance: SimInstance,
    candidates: tuple[SwitchingTemplateCandidate, ...],
    *,
    replay_top_k: int,
    replay_sample_dt_s: float,
    screen_replay_margin_m: float,
) -> _GroupResult:
    if not candidates:
        raise ValueError("candidates must not be empty")
    screen = _screen_template_group(instance, candidates)
    top_count = min(int(replay_top_k), len(candidates))
    top_indices = np.argsort(screen.min_distances)[:top_count]
    caught_candidate: SwitchingTemplateCandidate | None = None
    caught_min_distance = float("inf")
    caught_final_distance = float("inf")
    catch_step = -1
    best_uncaught_index = int(top_indices[0])
    best_uncaught_min_distance = float(screen.min_distances[best_uncaught_index])
    best_uncaught_final_distance = float(screen.final_distances[best_uncaught_index])
    dt = _backend_dt(instance)
    replay_threshold = float(instance.config.intercept_radius_m) + max(0.0, float(screen_replay_margin_m))
    if best_uncaught_min_distance > replay_threshold:
        return _GroupResult(
            caught_candidate=None,
            catch_step=-1,
            caught_min_distance=float("inf"),
            caught_final_distance=float("inf"),
            best_uncaught_candidate=candidates[best_uncaught_index],
            best_uncaught_min_distance=float(best_uncaught_min_distance),
            best_uncaught_final_distance=float(best_uncaught_final_distance),
        )
    for index in top_indices:
        candidate = candidates[int(index)]
        controls = _motor_commands_for_candidate(instance, candidate)
        replay = _fast_replay_backend_tick_commands(instance, controls, sample_dt_s=float(replay_sample_dt_s))
        if replay.min_distance_m < best_uncaught_min_distance:
            best_uncaught_index = int(index)
            best_uncaught_min_distance = float(replay.min_distance_m)
            best_uncaught_final_distance = float(replay.final_distance_m)
        if replay.caught:
            caught_candidate = candidate
            caught_min_distance = float(replay.min_distance_m)
            caught_final_distance = float(replay.final_distance_m)
            catch_step = int(round(float(replay.catch_time_s) / dt))
            break

    return _GroupResult(
        caught_candidate=caught_candidate,
        catch_step=int(catch_step),
        caught_min_distance=float(caught_min_distance),
        caught_final_distance=float(caught_final_distance),
        best_uncaught_candidate=candidates[best_uncaught_index],
        best_uncaught_min_distance=float(best_uncaught_min_distance),
        best_uncaught_final_distance=float(best_uncaught_final_distance),
    )


def _effective_max_thrust_n(instance: SimInstance) -> float:
    configured = float(instance.config.max_thrust_n)
    if configured > 0.0:
        return configured
    params = instance.config.pursuer
    return 2.0 * float(params.mass_kg) * float(params.gravity_mps2)


def _effective_max_rate_rps(instance: SimInstance) -> float:
    configured = float(instance.config.max_rate_rps)
    if configured > 0.0:
        return configured
    return float(instance.config.pursuer.max_omega_rps)


@dataclass(frozen=True)
class _ScreenResult:
    min_distances: np.ndarray
    final_distances: np.ndarray


def _screen_template_group(
    instance: SimInstance,
    candidates: tuple[SwitchingTemplateCandidate, ...],
) -> _ScreenResult:
    count = len(candidates)
    steps = int(candidates[0].steps)
    dt = _backend_dt(instance)
    params = instance.config.pursuer
    state = np.repeat(_initial_state_wxyz_rpm(instance).reshape(1, 17), count, axis=0)
    target_position = np.asarray(instance.target_initial.position_w, dtype=np.float64).reshape(1, 3)
    target_velocity = np.asarray(instance.target_initial.velocity_w, dtype=np.float64).reshape(1, 3)
    min_distances = np.full(count, np.inf, dtype=np.float64)
    final_distances = np.full(count, np.inf, dtype=np.float64)
    first_commands, cruise_commands, counter_commands = _candidate_arc_commands(instance, candidates)
    first_switch = np.array([candidate.first_switch_fraction for candidate in candidates], dtype=np.float64)
    second_switch = np.array([candidate.second_switch_fraction for candidate in candidates], dtype=np.float64)
    for step in range(steps + 1):
        t = float(step) * dt
        target = target_position + t * target_velocity
        distances = np.linalg.norm(state[:, 0:3] - target, axis=1)
        min_distances = np.minimum(min_distances, distances)
        final_distances = distances
        if step == steps:
            break
        fraction = float(step) / float(max(steps, 1))
        commands = cruise_commands.copy()
        first_arc = fraction < first_switch
        third_arc = fraction >= second_switch
        commands[first_arc] = first_commands[first_arc]
        commands[third_arc] = counter_commands[third_arc]
        state = _euler_motor_lag_step_batch(state, commands, dt, params)
    return _ScreenResult(min_distances=min_distances, final_distances=final_distances)


def _motor_commands_for_candidate(instance: SimInstance, candidate: SwitchingTemplateCandidate) -> np.ndarray:
    first, cruise, counter = _candidate_arc_commands(instance, (candidate,))
    controls = np.empty((int(candidate.steps), 4), dtype=np.float64)
    for step in range(int(candidate.steps)):
        fraction = float(step) / float(max(int(candidate.steps), 1))
        if fraction < float(candidate.first_switch_fraction):
            controls[step] = first[0]
        elif fraction >= float(candidate.second_switch_fraction):
            controls[step] = counter[0]
        else:
            controls[step] = cruise[0]
    return controls


def _candidate_arc_commands(
    instance: SimInstance,
    candidates: tuple[SwitchingTemplateCandidate, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(candidates)
    axes = np.array([candidate.axis_b for candidate in candidates], dtype=np.float64).reshape(count, 3)
    thrust = _effective_max_thrust_n(instance) * np.array(
        [candidate.thrust_fraction for candidate in candidates],
        dtype=np.float64,
    )
    rate = _effective_max_rate_rps(instance) * np.array(
        [candidate.rate_fraction for candidate in candidates],
        dtype=np.float64,
    )
    counter = np.array([candidate.counter_rate_fraction for candidate in candidates], dtype=np.float64)
    first_rates = axes * rate.reshape(-1, 1)
    cruise_rates = np.zeros((count, 3), dtype=np.float64)
    counter_rates = -axes * (rate * counter).reshape(-1, 1)
    return (
        _ctbr_to_motor_speeds_batch(instance.config.pursuer, thrust, first_rates),
        _ctbr_to_motor_speeds_batch(instance.config.pursuer, thrust, cruise_rates),
        _ctbr_to_motor_speeds_batch(instance.config.pursuer, thrust, counter_rates),
    )


@dataclass(frozen=True)
class _FastReplayResult:
    min_distance_m: float
    final_distance_m: float
    caught: bool
    catch_time_s: float


def _fast_replay_backend_tick_commands(
    instance: SimInstance,
    controls: np.ndarray,
    *,
    sample_dt_s: float,
) -> _FastReplayResult:
    commands = np.asarray(controls, dtype=np.float64).reshape(-1, 4)
    if len(commands) <= 0:
        raise ValueError("controls must contain at least one command")
    backend_dt = _backend_dt(instance)
    sample_steps = max(1, int(round(float(sample_dt_s) / backend_dt)))
    backend = BatchPufferSimEngineBackend(1)
    snapshots = backend.reset_many(np.array([0], dtype=np.int64), (instance,))
    min_distance = float("inf")
    final_distance = float("inf")
    caught = False
    catch_time_s = float("nan")
    elapsed_s = 0.0
    index = 0
    while index < len(commands):
        pursuer_position = np.asarray(snapshots.arrays.pursuer[0, 0:3], dtype=np.float64)
        target_position = np.asarray(snapshots.arrays.target[0, 0:3], dtype=np.float64)
        final_distance = float(np.linalg.norm(pursuer_position - target_position))
        min_distance = min(min_distance, final_distance)
        if not caught and final_distance <= float(instance.config.intercept_radius_m):
            caught = True
            catch_time_s = float(elapsed_s)
        end = min(index + sample_steps, len(commands))
        while end > index + 1 and not np.allclose(commands[end - 1], commands[index], rtol=0.0, atol=1.0e-9):
            end -= 1
        step_count = max(1, end - index)
        snapshots = backend.step_motor_speeds_many_dt(commands[index].reshape(1, 4), backend_dt * float(step_count))
        elapsed_s += backend_dt * float(step_count)
        index += step_count
    pursuer_position = np.asarray(snapshots.arrays.pursuer[0, 0:3], dtype=np.float64)
    target_position = np.asarray(snapshots.arrays.target[0, 0:3], dtype=np.float64)
    final_distance = float(np.linalg.norm(pursuer_position - target_position))
    min_distance = min(min_distance, final_distance)
    if not caught and final_distance <= float(instance.config.intercept_radius_m):
        caught = True
        catch_time_s = float(elapsed_s)
    return _FastReplayResult(
        min_distance_m=float(min_distance),
        final_distance_m=float(final_distance),
        caught=bool(caught),
        catch_time_s=float(catch_time_s),
    )


def _templates_for_steps(
    instance: SimInstance,
    config: SwitchingTemplateConfig,
    steps: int,
    total_time_s: float,
) -> tuple[SwitchingTemplateCandidate, ...]:
    candidates: list[SwitchingTemplateCandidate] = []
    for vertical_bias in config.vertical_bias_gains:
        for sign in config.direction_signs:
            axis = _initial_rotation_axis_b(
                instance,
                vertical_bias_gain=float(vertical_bias),
                velocity_gain=float(config.velocity_gain),
                direction_sign=float(sign),
            )
            axis_tuple = (float(axis[0]), float(axis[1]), float(axis[2]))
            for thrust_fraction in config.thrust_fractions:
                for rate_fraction in config.rate_fractions:
                    for first_switch in config.first_switch_fractions:
                        for second_switch in config.second_switch_fractions:
                            if float(second_switch) <= float(first_switch):
                                continue
                            for counter_rate in config.counter_rate_fractions:
                                candidates.append(
                                    SwitchingTemplateCandidate(
                                        steps=int(steps),
                                        total_time_s=float(total_time_s),
                                        thrust_fraction=float(thrust_fraction),
                                        rate_fraction=float(rate_fraction),
                                        first_switch_fraction=float(first_switch),
                                        second_switch_fraction=float(second_switch),
                                        counter_rate_fraction=float(counter_rate),
                                        vertical_bias_gain=float(vertical_bias),
                                        direction_sign=float(sign),
                                        axis_b=axis_tuple,
                                    )
                                )
    return tuple(candidates)


def _time_grid(instance: SimInstance, config: SwitchingTemplateConfig) -> tuple[tuple[int, float], ...]:
    dt = _backend_dt(instance)
    min_steps = int(np.ceil(round(float(config.min_time_s) / dt, 6)))
    max_steps = int(np.floor(round(float(config.max_time_s) / dt, 6)))
    step_stride = max(1, int(round(float(config.time_step_s) / dt)))
    if max_steps < min_steps:
        raise ValueError("max_time_s must be greater than or equal to min_time_s")
    grid: list[tuple[int, float]] = []
    for steps in range(min_steps, max_steps + 1, step_stride):
        grid.append((int(steps), float(steps) * dt))
    if not grid or grid[-1][0] != max_steps:
        grid.append((int(max_steps), float(max_steps) * dt))
    return tuple(grid)


def _initial_rotation_axis_b(
    instance: SimInstance,
    *,
    vertical_bias_gain: float,
    velocity_gain: float,
    direction_sign: float,
) -> np.ndarray:
    pursuer_position = np.asarray(instance.pursuer_initial.position_w, dtype=np.float64).reshape(3)
    pursuer_velocity = np.asarray(instance.pursuer_initial.velocity_w, dtype=np.float64).reshape(3)
    target_position = np.asarray(instance.target_initial.position_w, dtype=np.float64).reshape(3)
    desired = (target_position - pursuer_position) - float(velocity_gain) * pursuer_velocity
    desired[2] += float(vertical_bias_gain) * float(instance.config.pursuer.gravity_mps2)
    desired_norm = float(np.linalg.norm(desired))
    if desired_norm <= 1.0e-12:
        desired = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        desired = desired / desired_norm

    rotation = _quat_xyzw_to_rotation_matrix(np.asarray(instance.pursuer_initial.quat_xyzw, dtype=np.float64))
    body_z_w = rotation[:, 2]
    axis_w = np.cross(body_z_w, desired)
    axis_norm = float(np.linalg.norm(axis_w))
    if axis_norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    axis_b = rotation.T @ (axis_w / axis_norm)
    axis_b[2] = 0.0
    lateral_norm = float(np.linalg.norm(axis_b[0:2]))
    if lateral_norm > 1.0e-12:
        axis_b[0:2] /= lateral_norm
    return float(direction_sign) * axis_b


def _quat_xyzw_to_rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _ctbr_to_motor_speeds_batch(
    params: PursuerParams,
    thrust_n: np.ndarray,
    body_rates_b: np.ndarray,
) -> np.ndarray:
    thrust = np.asarray(thrust_n, dtype=np.float64).reshape(-1)
    rates = np.asarray(body_rates_b, dtype=np.float64).reshape(len(thrust), 3)
    inertia = np.array([float(params.ixx), float(params.iyy), float(params.izz)], dtype=np.float64)
    moments = rates * (float(params.k_w) * inertia).reshape(1, 3)
    desired = np.column_stack((np.maximum(thrust, 0.0), moments))
    allocation = _sim_allocation_matrix(params)
    rotor_thrusts = np.linalg.solve(allocation, desired.T).T
    speed_sq = rotor_thrusts / max(float(params.k_thrust), 1.0e-12)
    speeds = np.sign(speed_sq) * np.sqrt(np.abs(speed_sq))
    return np.clip(speeds, _min_rpm(params), float(params.max_rpm))


def _sim_allocation_matrix(params: PursuerParams) -> np.ndarray:
    rotor_positions = params.rotor_positions_b
    rotor_directions = params.rotor_directions
    if rotor_positions is None or rotor_directions is None:
        arm_factor = float(params.arm_len_m) / np.sqrt(2.0)
        return np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [-arm_factor, -arm_factor, arm_factor, arm_factor],
                [-arm_factor, arm_factor, arm_factor, -arm_factor],
                [-float(params.k_yaw), float(params.k_yaw), -float(params.k_yaw), float(params.k_yaw)],
            ],
            dtype=np.float64,
        )
    rotor_positions_arr = np.asarray(rotor_positions, dtype=np.float64).reshape(4, 3)
    rotor_directions_arr = np.asarray(rotor_directions, dtype=np.float64).reshape(4)
    return np.vstack(
        (
            np.ones(4, dtype=np.float64),
            rotor_positions_arr[:, 1],
            -rotor_positions_arr[:, 0],
            float(params.k_yaw) * rotor_directions_arr,
        )
    )


def _euler_motor_lag_step_batch(
    state: np.ndarray,
    command_rpm: np.ndarray,
    dt: float,
    params: PursuerParams,
) -> np.ndarray:
    out = np.asarray(state, dtype=np.float64).reshape(-1, 17).copy()
    command = np.asarray(command_rpm, dtype=np.float64).reshape(len(out), 4)
    quat = _normalize_quat_batch(out[:, 6:10])
    omega = out[:, 10:13]
    rpm = out[:, 13:17]
    thrusts = float(params.k_thrust) * np.square(np.maximum(rpm, 0.0))
    thrust_total = np.sum(thrusts, axis=1)
    force_b = np.column_stack((np.zeros(len(out)), np.zeros(len(out)), thrust_total))
    force_w = _quat_rotate_batch(quat, force_b)
    accel = force_w / float(params.mass_kg)
    accel[:, 2] -= float(params.gravity_mps2)
    qdot = 0.5 * _quat_mul_batch(quat, np.column_stack((np.zeros(len(out)), omega)))
    moment = _sim_moment_from_thrusts_batch(thrusts, params)
    wx = omega[:, 0]
    wy = omega[:, 1]
    wz = omega[:, 2]
    inertial = np.column_stack(
        (
            (float(params.iyy) - float(params.izz)) * wy * wz,
            (float(params.izz) - float(params.ixx)) * wz * wx,
            (float(params.ixx) - float(params.iyy)) * wx * wy,
        )
    )
    omega_dot = np.column_stack(
        (
            (moment[:, 0] + inertial[:, 0]) / float(params.ixx),
            (moment[:, 1] + inertial[:, 1]) / float(params.iyy),
            (moment[:, 2] + inertial[:, 2]) / float(params.izz),
        )
    )
    rpm_dot = (command - rpm) / max(float(params.motor_tau_s), 1.0e-6)
    out[:, 0:3] += out[:, 3:6] * float(dt)
    out[:, 3:6] += accel * float(dt)
    out[:, 6:10] = _normalize_quat_batch(out[:, 6:10] + qdot * float(dt))
    out[:, 10:13] += omega_dot * float(dt)
    out[:, 13:17] += rpm_dot * float(dt)
    out[:, 3:6] = np.clip(out[:, 3:6], -float(params.max_vel_mps), float(params.max_vel_mps))
    out[:, 10:13] = np.clip(out[:, 10:13], -float(params.max_omega_rps), float(params.max_omega_rps))
    out[:, 13:17] = np.clip(out[:, 13:17], 0.0, float(params.max_rpm))
    return out


def _sim_moment_from_thrusts_batch(thrusts: np.ndarray, params: PursuerParams) -> np.ndarray:
    rotor_positions = params.rotor_positions_b
    rotor_directions = params.rotor_directions
    thrust_arr = np.asarray(thrusts, dtype=np.float64).reshape(-1, 4)
    if rotor_positions is None or rotor_directions is None:
        arm_factor = float(params.arm_len_m) / np.sqrt(2.0)
        return np.column_stack(
            (
                arm_factor * ((thrust_arr[:, 2] + thrust_arr[:, 3]) - (thrust_arr[:, 0] + thrust_arr[:, 1])),
                arm_factor * ((thrust_arr[:, 1] + thrust_arr[:, 2]) - (thrust_arr[:, 0] + thrust_arr[:, 3])),
                float(params.k_yaw) * (-thrust_arr[:, 0] + thrust_arr[:, 1] - thrust_arr[:, 2] + thrust_arr[:, 3]),
            )
        )
    rotor_positions_arr = np.asarray(rotor_positions, dtype=np.float64).reshape(4, 3)
    rotor_directions_arr = np.asarray(rotor_directions, dtype=np.float64).reshape(4)
    return np.column_stack(
        (
            thrust_arr @ rotor_positions_arr[:, 1],
            thrust_arr @ (-rotor_positions_arr[:, 0]),
            thrust_arr @ (float(params.k_yaw) * rotor_directions_arr),
        )
    )


def _quat_rotate_batch(quat_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    vq = np.column_stack((np.zeros(len(q)), np.asarray(vectors, dtype=np.float64).reshape(len(q), 3)))
    q_conj = q.copy()
    q_conj[:, 1:4] *= -1.0
    return _quat_mul_batch(_quat_mul_batch(q, vq), q_conj)[:, 1:4]


def _quat_mul_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    bb = np.asarray(b, dtype=np.float64).reshape(len(aa), 4)
    return np.column_stack(
        (
            aa[:, 0] * bb[:, 0] - aa[:, 1] * bb[:, 1] - aa[:, 2] * bb[:, 2] - aa[:, 3] * bb[:, 3],
            aa[:, 0] * bb[:, 1] + aa[:, 1] * bb[:, 0] + aa[:, 2] * bb[:, 3] - aa[:, 3] * bb[:, 2],
            aa[:, 0] * bb[:, 2] - aa[:, 1] * bb[:, 3] + aa[:, 2] * bb[:, 0] + aa[:, 3] * bb[:, 1],
            aa[:, 0] * bb[:, 3] + aa[:, 1] * bb[:, 2] - aa[:, 2] * bb[:, 1] + aa[:, 3] * bb[:, 0],
        )
    )


def _normalize_quat_batch(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    return q / np.sqrt(np.sum(np.square(q), axis=1, keepdims=True) + 1.0e-12)


def _initial_state_wxyz_rpm(instance: SimInstance) -> np.ndarray:
    params = instance.config.pursuer
    q_xyzw = np.asarray(instance.pursuer_initial.quat_xyzw, dtype=np.float64).reshape(4)
    if instance.pursuer_initial.rotor_speeds is None:
        rpm = np.full(4, _hover_rpm(params), dtype=np.float64)
    else:
        rpm = np.asarray(instance.pursuer_initial.rotor_speeds, dtype=np.float64).reshape(4)
    rpm = np.clip(rpm, _min_rpm(params), float(params.max_rpm))
    return np.array(
        [
            *np.asarray(instance.pursuer_initial.position_w, dtype=np.float64).reshape(3),
            *np.asarray(instance.pursuer_initial.velocity_w, dtype=np.float64).reshape(3),
            float(q_xyzw[3]),
            float(q_xyzw[0]),
            float(q_xyzw[1]),
            float(q_xyzw[2]),
            *np.asarray(instance.pursuer_initial.body_rates_b, dtype=np.float64).reshape(3),
            *rpm,
        ],
        dtype=np.float64,
    )


def _hover_rpm(params: PursuerParams) -> float:
    return float(np.sqrt((float(params.mass_kg) * float(params.gravity_mps2)) / (4.0 * float(params.k_thrust))))


def _min_rpm(params: PursuerParams) -> float:
    if params.rpm_min is not None:
        return float(np.clip(params.rpm_min, 0.0, params.max_rpm))
    return float(np.clip(2.0 * _hover_rpm(params) - float(params.max_rpm), 0.0, float(params.max_rpm)))


def _backend_dt(instance: SimInstance) -> float:
    if instance.config is None:
        raise ValueError("SimInstance.config is required")
    dt = float(instance.config.options.backend_dt) * max(1, int(instance.config.options.action_substeps))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("effective backend dt must be finite and positive")
    return dt


def _validate_config(config: SwitchingTemplateConfig) -> None:
    scalar_fields = {
        "min_time_s": config.min_time_s,
        "max_time_s": config.max_time_s,
        "time_step_s": config.time_step_s,
        "velocity_gain": config.velocity_gain,
        "replay_top_k": config.replay_top_k,
        "replay_sample_dt_s": _backend_dt_value_for_validation(config.replay_sample_dt_s),
        "screen_replay_margin_m": config.screen_replay_margin_m,
    }
    for name, value in scalar_fields.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(config.min_time_s) <= 0.0 or float(config.max_time_s) <= 0.0 or float(config.time_step_s) <= 0.0:
        raise ValueError("time bounds and time step must be positive")
    if int(config.replay_top_k) <= 0:
        raise ValueError("replay_top_k must be positive")
    if config.replay_sample_dt_s is not None and float(config.replay_sample_dt_s) <= 0.0:
        raise ValueError("replay_sample_dt_s must be positive")
    if float(config.screen_replay_margin_m) < 0.0:
        raise ValueError("screen_replay_margin_m must be non-negative")
    if float(config.max_time_s) < float(config.min_time_s):
        raise ValueError("max_time_s must be greater than or equal to min_time_s")
    for name, values in {
        "thrust_fractions": config.thrust_fractions,
        "rate_fractions": config.rate_fractions,
        "first_switch_fractions": config.first_switch_fractions,
        "second_switch_fractions": config.second_switch_fractions,
        "counter_rate_fractions": config.counter_rate_fractions,
        "vertical_bias_gains": config.vertical_bias_gains,
        "direction_signs": config.direction_signs,
    }.items():
        if not values:
            raise ValueError(f"{name} must not be empty")
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError(f"{name} must contain finite values")


def _backend_dt_value_for_validation(value: float | None) -> float:
    return 1.0 if value is None else float(value)


def _validate_instance_contract(instance: SimInstance) -> None:
    if instance.config is None:
        raise ValueError("SimInstance.config is required")
    if len(instance.config.targets) != 1 or len(instance.target_initials) != 1:
        raise ValueError("switching-template search supports exactly one target")
    target = instance.config.targets[0]
    behavior = target.behavior
    if behavior.kind not in {"waypoints", "linear"}:
        raise ValueError(f"unsupported target behavior kind: {behavior.kind}")
    if len(behavior.waypoints) != 0 or float(behavior.duration_s) != 0.0 or bool(behavior.loop):
        raise ValueError("switching-template search requires constant-velocity target behavior")
    controller = target.controller
    if (
        controller.kind != "linear"
        or float(controller.kp) != 0.0
        or float(controller.kv) != 0.0
        or float(controller.max_accel_mps2) != 0.0
    ):
        raise ValueError("switching-template search requires passive linear target controller")
