"""Drake adapter for the C SimEngine world."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import LeafSystem
from scipy.spatial.transform import Rotation

from backends import PufferSimEngineBackend, SimOptions
from backends.csim.bindings import (
    initial_state_from_rotorpy,
    vehicle_params_from_quad_params,
)

from ..drake_compat import ctbr_value, make_scene_snapshot, scene_value, vehicle_state_value
from ..targets import KinematicTarget
from ..types import CameraRig, SceneSnapshot, SimulationTarget


class PufferSimEngineSystem(LeafSystem):
    """Combined pursuer/target world backed by the shared C SimEngine."""

    def __init__(
        self,
        quad_params: dict[str, Any],
        initial_state: dict[str, np.ndarray],
        dt: float,
        target: KinematicTarget,
        camera_rig: CameraRig,
        intercept_radius_m: float = 0.0,
        options: SimOptions | None = None,
    ):
        super().__init__()
        self._dt = float(dt)
        self._target = target
        self._cameras = (camera_rig,)
        self._params = vehicle_params_from_quad_params(quad_params)
        self._backend = PufferSimEngineBackend(self._params, options=options)
        self._intercept_radius_m = float(intercept_radius_m)
        self._initial_snapshot = self._backend.reset(
            initial_state_from_rotorpy(initial_state),
            targets=(target,),
            intercept_radius_m=self._intercept_radius_m,
        )

        self._state_index = self.DeclareAbstractState(
            AbstractValue.Make(_copy_snapshot(self._initial_snapshot))
        )
        self.DeclareAbstractInputPort("ctbr_cmd", ctbr_value())
        self.DeclareAbstractOutputPort(
            "vehicle_state_dict",
            vehicle_state_value,
            self._copy_vehicle_out,
            prerequisites_of_calc={self.abstract_state_ticket(self._state_index)},
        )
        self.DeclareAbstractOutputPort(
            "scene",
            scene_value,
            self._copy_scene_out,
            prerequisites_of_calc={
                self.abstract_state_ticket(self._state_index),
                self.time_ticket(),
            },
        )
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=self._dt, offset_sec=0.0, update=self._step,
        )

    def _step(self, context, state) -> None:
        current = state.get_mutable_abstract_state(self._state_index).get_value()
        cmd = self.GetInputPort("ctbr_cmd").Eval(context)
        new_snapshot = self._backend.step_ctbr(current, cmd, self._dt)
        state.get_mutable_abstract_state(self._state_index).set_value(
            _copy_snapshot(new_snapshot)
        )

    def _copy_vehicle_out(self, context, output) -> None:
        snapshot = context.get_abstract_state(self._state_index).get_value()
        output.set_value(_copy_state(snapshot["vehicle_state"]))

    def _copy_scene_out(self, context, output) -> None:
        snapshot = context.get_abstract_state(self._state_index).get_value()
        output.set_value(self._scene_from_snapshot(context.get_time(), snapshot))

    def _scene_from_snapshot(self, t: float, snapshot: dict[str, Any]) -> SceneSnapshot:
        vehicle_state = snapshot["vehicle_state"]
        pursuer = SimulationTarget(
            id="interceptor",
            kind="multirotor",
            position_w=np.asarray(vehicle_state["x"], dtype=float).copy(),
            velocity_w=np.asarray(vehicle_state["v"], dtype=float).copy(),
            rotation_wb=Rotation.from_quat(vehicle_state["q"]).as_matrix(),
            radius_m=0.15,
        )
        targets = tuple(
            SimulationTarget(
                id=str(target_state.get("id", self._target.target_id)),
                kind=str(target_state.get("kind", self._target.kind)),
                position_w=np.asarray(target_state["position_w"], dtype=float).copy(),
                velocity_w=np.asarray(target_state["velocity_w"], dtype=float).copy(),
                rotation_wb=np.eye(3, dtype=float),
                radius_m=float(target_state.get("radius_m", self._target.radius_m)),
            )
            for target_state in snapshot["target_states"]
        )
        return make_scene_snapshot(t, pursuer, targets, self._cameras)


def _copy_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "vehicle_state": _copy_state(snapshot["vehicle_state"]),
        "target_states": tuple(_copy_target_state(target) for target in snapshot["target_states"]),
        "intercept_radius_m": float(snapshot.get("intercept_radius_m", 0.0)),
        "metrics": _copy_metrics(snapshot["metrics"]),
    }


def _copy_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=float).copy() for key, value in state.items()}


def _copy_target_state(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "c_id": int(target["c_id"]),
        "id": str(target["id"]),
        "kind": str(target["kind"]),
        "radius_m": float(target["radius_m"]),
        "position_w": np.asarray(target["position_w"], dtype=float).copy(),
        "velocity_w": np.asarray(target["velocity_w"], dtype=float).copy(),
    }


def _copy_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "distance_m": float(metrics["distance_m"]),
        "min_distance_m": float(metrics["min_distance_m"]),
        "intercepted": bool(metrics["intercepted"]),
        "intercept_time_s": (
            None if metrics.get("intercept_time_s") is None
            else float(metrics["intercept_time_s"])
        ),
        "target_index": int(metrics["target_index"]),
    }
