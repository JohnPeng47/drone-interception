from __future__ import annotations

import numpy as np

from backends.csim.bindings.types import SimInstance, SimSnapshot
from backends.csim.runner import CtbrCommandBatch, SimControlPolicy, SimRunnerState

from .config import BaselineStrategyConfig, VehicleConfig
from .controller.beihang_baseline_strategy import BeihangBaselineStrategy
from .types import StrategyObservation


class BeihangMinimalSimControlPolicy(SimControlPolicy):
    """Run the minimal Beihang image-centering controller over SimRunner state."""

    def __init__(self):
        self._slot_strategies: dict[int, BeihangBaselineStrategy] = {}
        self._slot_vehicle_configs: dict[int, VehicleConfig] = {}
        self._last_uv: dict[int, np.ndarray | None] = {}

    def reset(self, state: SimRunnerState) -> None:
        self._slot_strategies.clear()
        self._slot_vehicle_configs.clear()
        self._last_uv.clear()

    def on_slots_started(
        self,
        slots: np.ndarray,
        instances,
        state: SimRunnerState,
    ) -> None:
        for slot, instance in zip(np.asarray(slots, dtype=np.int64).reshape(-1), instances):
            slot_i = int(slot)
            vehicle = _vehicle_config_from_instance(instance)
            self._slot_vehicle_configs[slot_i] = vehicle
            self._slot_strategies[slot_i] = BeihangBaselineStrategy(
                vehicle=vehicle,
                config=BaselineStrategyConfig(),
            )
            self._last_uv[slot_i] = None

    def command(self, state: SimRunnerState) -> CtbrCommandBatch:
        thrust_n = np.zeros(len(state.instances), dtype=np.float32)
        body_rates_b = np.zeros((len(state.instances), 3), dtype=np.float32)
        for slot, instance in enumerate(state.instances):
            if instance is None or not bool(state.active[slot]):
                self._last_uv[slot] = None
                continue
            strategy = self._slot_strategies.get(slot)
            if strategy is None:
                vehicle = _vehicle_config_from_instance(instance)
                self._slot_vehicle_configs[slot] = vehicle
                strategy = BeihangBaselineStrategy(vehicle=vehicle, config=BaselineStrategyConfig())
                self._slot_strategies[slot] = strategy
            snapshot = state.snapshot[slot]
            observation = self._observation(slot, instance, snapshot, float(state.elapsed_s[slot]))
            command = strategy.command(observation, float(state.elapsed_s[slot]))
            thrust_n[slot] = np.float32(command.thrust_n)
            body_rates_b[slot] = np.asarray(command.body_rates_b, dtype=np.float32).reshape(3)
        return CtbrCommandBatch(thrust_n=thrust_n, body_rates_b=body_rates_b)

    def _observation(
        self,
        slot: int,
        instance: SimInstance,
        snapshot: SimSnapshot,
        t_s: float,
    ) -> StrategyObservation:
        R_wb = _quat_xyzw_to_rot(snapshot.pursuer.quat_xyzw)
        camera = instance.config.cameras[0]
        rel_w = np.asarray(snapshot.target.position_w, dtype=float) - np.asarray(snapshot.pursuer.position_w, dtype=float)
        rel_b = R_wb.T @ rel_w
        R_b2c = np.asarray(camera.body_to_camera, dtype=float).reshape(3, 3)
        rel_c = R_b2c @ (rel_b - np.asarray(camera.position_b, dtype=float).reshape(3))
        depth_m = float(rel_c[0])

        uv_norm = snapshot.camera.uv_norm.copy() if snapshot.camera.detected else None
        bearing_b = np.array([1.0, 0.0, 0.0], dtype=float)
        if uv_norm is not None:
            bearing_c = np.array([1.0, float(uv_norm[0]), float(uv_norm[1])], dtype=float)
            bearing_c /= max(float(np.linalg.norm(bearing_c)), 1e-12)
            bearing_b = R_b2c.T @ bearing_c
            bearing_b /= max(float(np.linalg.norm(bearing_b)), 1e-12)
        uv_dot = np.zeros(2, dtype=float)
        previous_uv = self._last_uv.get(slot)
        if uv_norm is not None and previous_uv is not None:
            dt_s = _dt_from_instance(instance)
            uv_dot = (uv_norm - previous_uv) / max(dt_s, 1e-9)
        self._last_uv[slot] = None if uv_norm is None else uv_norm.copy()

        return StrategyObservation(
            t=float(t_s),
            detected=bool(snapshot.camera.detected),
            uv_norm=uv_norm,
            uv_dot_norm=uv_dot,
            depth_m=depth_m,
            bearing_b=bearing_b,
            vehicle_velocity_w=np.asarray(snapshot.pursuer.velocity_w, dtype=float).copy(),
            vehicle_rotation_wb=R_wb,
        )


def _vehicle_config_from_instance(instance: SimInstance) -> VehicleConfig:
    if instance.config is None:
        raise ValueError("BeihangMinimalSimControlPolicy requires SimInstance.config")
    return VehicleConfig(
        mass_kg=float(instance.config.pursuer.mass_kg),
        max_thrust_n=float(instance.config.max_thrust_n),
        max_body_rate_rad_s=float(instance.config.max_rate_rps),
        initial_position_w=tuple(float(x) for x in instance.pursuer_initial.position_w),
        initial_velocity_w=tuple(float(x) for x in instance.pursuer_initial.velocity_w),
        initial_quat_xyzw=tuple(float(x) for x in instance.pursuer_initial.quat_xyzw),
    )


def _dt_from_instance(instance: SimInstance) -> float:
    assert instance.config is not None
    return float(instance.config.options.backend_dt) * max(1, int(instance.config.options.action_substeps))


def _quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])
