from __future__ import annotations

import numpy as np

from backends.csim.bindings.types import PursuerParams, SimInstance
from backends.csim.runner import MotorSpeedCommandBatch, SimControlPolicy, SimRunnerState

from .planner import RpgTimeOptimalPlanner, RpgTimeOptimalPlan
from .config import RpgTimeOptimalConfig


class RpgTimeOptimalMotorFeedforwardPolicy(SimControlPolicy):
    """Execute CPC rotor-thrust plans as SimEngine motor-speed commands."""

    def __init__(self, config: RpgTimeOptimalConfig | None = None):
        self.config = config or RpgTimeOptimalConfig()
        self.planner = RpgTimeOptimalPlanner(self.config)
        self._slots: dict[int, RpgTimeOptimalPlan] = {}

    def reset(self, state: SimRunnerState) -> None:
        self._slots.clear()

    def on_slots_started(self, slots: np.ndarray, instances, state: SimRunnerState) -> None:
        for slot in np.asarray(slots, dtype=np.int64).reshape(-1):
            slot_i = int(slot)
            instance = state.instances[slot_i]
            if instance is not None:
                self._slots[slot_i] = self.planner.solve(instance)

    def command(self, state: SimRunnerState) -> MotorSpeedCommandBatch:
        motor_speeds_rpm = np.zeros((len(state.instances), 4), dtype=np.float32)
        for slot, instance in enumerate(state.instances):
            if instance is None or not bool(state.active[slot]):
                continue
            plan = self._slots.get(slot)
            if plan is None:
                motor_speeds_rpm[slot] = _hover_rpm(instance.config.pursuer)
                continue
            motor_speeds_rpm[slot] = sample_motor_speed_command(
                instance,
                plan,
                float(state.elapsed_s[slot]),
                time_scale=float(self.config.plan_time_scale),
                command_mode=self.config.motor_command_mode,
            )
        return MotorSpeedCommandBatch(motor_speeds_rpm=motor_speeds_rpm)


def sample_motor_speed_command(
    instance: SimInstance,
    plan: RpgTimeOptimalPlan,
    elapsed_s: float,
    *,
    time_scale: float,
    command_mode: str,
) -> np.ndarray:
    assert instance.config is not None
    params = instance.config.pursuer
    plan_time_s = float(elapsed_s) / max(float(time_scale), 1.0e-9)
    if plan_time_s > float(plan.total_time_s):
        return np.full(4, _hover_rpm(params), dtype=np.float32)
    if plan.motor_speed_commands_rpm is not None:
        return _sample_motor_commands(plan, plan_time_s, command_mode).astype(np.float32)
    index = int(np.searchsorted(plan.t_u_s, max(plan_time_s, 0.0), side="right") - 1)
    index = int(np.clip(index, 0, plan.motor_thrusts_n.shape[1] - 1))
    cpc_rotor_thrusts_n = np.asarray(plan.motor_thrusts_n[:, index], dtype=float).reshape(4)
    wrench = _cpc_wrench_from_rotor_thrusts(cpc_rotor_thrusts_n, params)
    sim_rotor_thrusts_n = _sim_rotor_thrusts_from_wrench(wrench, params)
    motor_speeds = _rotor_thrusts_to_rpm(sim_rotor_thrusts_n, params)
    return motor_speeds.astype(np.float32)


def _sample_motor_commands(plan: RpgTimeOptimalPlan, plan_time_s: float, command_mode: str) -> np.ndarray:
    assert plan.motor_speed_commands_rpm is not None
    sample_t = max(float(plan_time_s), 0.0)
    if command_mode == "linear":
        t_u = np.asarray(plan.t_u_s, dtype=float)
        commands = np.asarray(plan.motor_speed_commands_rpm, dtype=float)
        if commands.shape[1] == 1:
            return commands[:, 0].copy()
        t_ref = np.append(t_u, float(plan.total_time_s))
        values = np.column_stack((commands, commands[:, -1]))
        return np.array([np.interp(sample_t, t_ref, values[row]) for row in range(4)], dtype=float)
    index = int(np.searchsorted(plan.t_u_s, sample_t, side="right") - 1)
    index = int(np.clip(index, 0, plan.motor_speed_commands_rpm.shape[1] - 1))
    return np.asarray(plan.motor_speed_commands_rpm[:, index], dtype=float).reshape(4)


def _cpc_wrench_from_rotor_thrusts(rotor_thrusts_n: np.ndarray, params: PursuerParams) -> np.ndarray:
    thrusts = np.asarray(rotor_thrusts_n, dtype=float).reshape(4)
    arm_len = float(params.arm_len_m)
    yaw_coeff = float(params.k_yaw)
    return np.array(
        [
            float(np.sum(thrusts)),
            arm_len * (thrusts[0] - thrusts[1] - thrusts[2] + thrusts[3]),
            arm_len * (-thrusts[0] - thrusts[1] + thrusts[2] + thrusts[3]),
            yaw_coeff * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3]),
        ],
        dtype=float,
    )


def _sim_rotor_thrusts_from_wrench(wrench: np.ndarray, params: PursuerParams) -> np.ndarray:
    allocation = _sim_allocation_matrix(params)
    rotor_thrusts = np.linalg.solve(allocation, np.asarray(wrench, dtype=float).reshape(4))
    return np.clip(rotor_thrusts, 0.0, None)


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
            dtype=float,
        )
    rotor_positions = np.asarray(rotor_positions, dtype=float).reshape(4, 3)
    rotor_directions = np.asarray(rotor_directions, dtype=float).reshape(4)
    yaw = float(params.k_yaw) * rotor_directions
    return np.vstack(
        (
            np.ones(4, dtype=float),
            rotor_positions[:, 1],
            -rotor_positions[:, 0],
            yaw,
        )
    )


def _rotor_thrusts_to_rpm(rotor_thrusts_n: np.ndarray, params: PursuerParams) -> np.ndarray:
    thrusts = np.clip(np.asarray(rotor_thrusts_n, dtype=float).reshape(4), 0.0, None)
    speeds = np.sqrt(thrusts / max(float(params.k_thrust), 1.0e-12))
    return np.clip(speeds, _min_rpm(params), float(params.max_rpm))


def _hover_rpm(params: PursuerParams) -> float:
    return float(np.sqrt((float(params.mass_kg) * float(params.gravity_mps2)) / (4.0 * float(params.k_thrust))))


def _min_rpm(params: PursuerParams) -> float:
    if params.rpm_min is not None:
        return float(np.clip(params.rpm_min, 0.0, params.max_rpm))
    return float(np.clip(2.0 * _hover_rpm(params) - float(params.max_rpm), 0.0, float(params.max_rpm)))
