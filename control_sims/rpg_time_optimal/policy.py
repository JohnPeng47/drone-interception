from __future__ import annotations

import numpy as np

from backends.csim.bindings.types import SimInstance, SimSnapshot
from backends.csim.runner import CtbrCommandBatch, SimControlPolicy, SimRunnerState
from control_sims.eth_mpc.policy import _accel_to_ctbr, _hover_command

from .planner import RpgTimeOptimalPlanner, RpgTimeOptimalPlan
from .config import RpgTimeOptimalConfig


class RpgTimeOptimalControlPolicy(SimControlPolicy):
    """Track RPG time-optimal plans as CTBR commands in the C SimRunner."""

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
            if instance is None:
                continue
            self._slots[slot_i] = self.planner.solve(instance)

    def command(self, state: SimRunnerState) -> CtbrCommandBatch:
        thrust_n = np.zeros(len(state.instances), dtype=np.float32)
        body_rates_b = np.zeros((len(state.instances), 3), dtype=np.float32)
        for slot, instance in enumerate(state.instances):
            if instance is None or not bool(state.active[slot]):
                continue
            slot_plan = self._slots.get(slot)
            if slot_plan is None:
                command = _hover_command(instance)
            else:
                command = self._command_one(instance, state.snapshot[slot], slot_plan, float(state.elapsed_s[slot]))
            thrust_n[slot] = np.float32(command[0])
            body_rates_b[slot] = np.asarray(command[1], dtype=np.float32).reshape(3)
        return CtbrCommandBatch(thrust_n=thrust_n, body_rates_b=body_rates_b)

    def _command_one(
        self,
        instance: SimInstance,
        snapshot: SimSnapshot,
        plan: RpgTimeOptimalPlan,
        elapsed_s: float,
    ) -> tuple[float, np.ndarray]:
        desired = _sample_plan(plan, elapsed_s / float(self.config.plan_time_scale))
        pos_error = desired["position_w"] - np.asarray(snapshot.pursuer.position_w, dtype=float).reshape(3)
        vel_error = desired["velocity_w"] - np.asarray(snapshot.pursuer.velocity_w, dtype=float).reshape(3)
        tracking_accel = (
            desired["acceleration_w"]
            + float(self.config.position_gain) * pos_error
            + float(self.config.velocity_gain) * vel_error
        )
        tracking_accel = _clip_norm(tracking_accel, float(self.config.max_tracking_accel_mps2))
        thrust, body_rates = _accel_to_ctbr(
            instance,
            snapshot,
            tracking_accel,
            R_wb=_quat_xyzw_to_rot(np.asarray(snapshot.pursuer.quat_xyzw, dtype=float).reshape(4)),
            rate_gain=1.0,
            drag_diag=np.zeros(3, dtype=float),
        )
        body_rates = body_rates + desired["body_rates_b"]
        body_rates = _clip_norm(body_rates, _max_rate_rps(instance, snapshot))
        return thrust, body_rates


def _sample_plan(plan: RpgTimeOptimalPlan, elapsed_s: float) -> dict[str, np.ndarray]:
    t = float(np.clip(elapsed_s, 0.0, max(float(plan.total_time_s), 0.0)))
    state_index = int(np.searchsorted(plan.t_x_s, t, side="right") - 1)
    state_index = int(np.clip(state_index, 0, plan.position_w.shape[1] - 1))
    command_index = int(np.searchsorted(plan.t_u_s, t, side="right") - 1)
    command_index = int(np.clip(command_index, 0, plan.motor_thrusts_n.shape[1] - 1))
    return {
        "position_w": plan.position_w[:, state_index].astype(float, copy=True),
        "velocity_w": plan.velocity_w[:, state_index].astype(float, copy=True),
        "acceleration_w": plan.acceleration_w[:, state_index].astype(float, copy=True),
        "body_rates_b": plan.body_rates_b[:, state_index].astype(float, copy=True),
        "motor_thrusts_n": plan.motor_thrusts_n[:, command_index].astype(float, copy=True),
    }


def _quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1.0e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _max_rate_rps(instance: SimInstance, snapshot: SimSnapshot) -> float:
    if instance.config is not None and float(instance.config.max_rate_rps) > 0.0:
        return float(instance.config.max_rate_rps)
    if float(snapshot.max_rate_rps) > 0.0:
        return float(snapshot.max_rate_rps)
    return 8.0


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= float(max_norm) or norm <= 1.0e-12:
        return arr
    return arr * (float(max_norm) / norm)
