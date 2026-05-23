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

from ..drake_compat import capture_value, ctbr_value, make_scene_snapshot, scene_value, vehicle_state_value
from ..targets import KinematicTarget
from ..types import CameraCapture, CameraRig, SceneSnapshot, SimulationTarget


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
            cameras=(camera_rig,),
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
        self.DeclareAbstractOutputPort(
            "capture",
            capture_value,
            self._copy_capture_out,
            prerequisites_of_calc={self.abstract_state_ticket(self._state_index)},
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

    def _copy_capture_out(self, context, output) -> None:
        snapshot = context.get_abstract_state(self._state_index).get_value()
        output.set_value(_capture_from_snapshot(snapshot))

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
        "camera_states": tuple(_copy_camera_state(camera) for camera in snapshot.get("camera_states", ())),
        "camera_outputs": tuple(_copy_camera_output(out) for out in snapshot.get("camera_outputs", ())),
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


def _copy_camera_output(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "camera_id": str(output["camera_id"]),
        "target_id": output.get("target_id"),
        "target_index": int(output["target_index"]),
        "captured": bool(output["captured"]),
        "detected": bool(output["detected"]),
        "t_capture": float(output["t_capture"]),
        "target_pos_c": np.asarray(output["target_pos_c"], dtype=float).copy(),
        "range_m": float(output["range_m"]),
        "uv_norm": (
            None if output.get("uv_norm") is None
            else np.asarray(output["uv_norm"], dtype=float).copy()
        ),
        "uv_px": (
            None if output.get("uv_px") is None
            else np.asarray(output["uv_px"], dtype=float).copy()
        ),
        "apparent_radius_px": (
            None if output.get("apparent_radius_px") is None
            else float(output["apparent_radius_px"])
        ),
        "has_frame": bool(output["has_frame"]),
        "frame_width_px": int(output["frame_width_px"]),
        "frame_height_px": int(output["frame_height_px"]),
        "frame_channels": int(output["frame_channels"]),
        "frame_rgb": None,
    }


def _copy_camera_state(camera: dict[str, Any]) -> dict[str, Any]:
    return {
        "c_id": int(camera["c_id"]),
        "position_b": np.asarray(camera["position_b"], dtype=float).copy(),
        "body_to_camera": np.asarray(camera["body_to_camera"], dtype=float).copy(),
        "width_px": int(camera["width_px"]),
        "height_px": int(camera["height_px"]),
        "fx_px": float(camera["fx_px"]),
        "fy_px": float(camera["fy_px"]),
        "cx_px": float(camera["cx_px"]),
        "cy_px": float(camera["cy_px"]),
        "hfov_rad": float(camera["hfov_rad"]),
        "vfov_rad": float(camera["vfov_rad"]),
        "capture_rate_hz": float(camera["capture_rate_hz"]),
        "next_capture_t": float(camera["next_capture_t"]),
    }


def _capture_from_snapshot(snapshot: dict[str, Any]) -> CameraCapture | None:
    outputs = snapshot.get("camera_outputs", ())
    if not outputs:
        return None
    out = outputs[0]
    return CameraCapture(
        t_capture=float(out["t_capture"]),
        camera_id=str(out["camera_id"]),
        target_id=out.get("target_id"),
        detected=bool(out["detected"]),
        uv_px=None if out.get("uv_px") is None else np.asarray(out["uv_px"], dtype=float).copy(),
        uv_norm=None if out.get("uv_norm") is None else np.asarray(out["uv_norm"], dtype=float).copy(),
        target_pos_c=np.asarray(out["target_pos_c"], dtype=float).copy(),
        range_m=float(out["range_m"]),
        apparent_radius_px=(
            None if out.get("apparent_radius_px") is None
            else float(out["apparent_radius_px"])
        ),
    )
