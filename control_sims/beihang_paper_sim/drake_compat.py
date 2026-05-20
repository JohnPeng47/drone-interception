"""Small Drake compatibility surface used by the Beihang paper simulation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import LeafSystem

from intercept_sim.rotorpy_adapter import (
    ctbr_to_rotorpy,
    hover_ctbr,
    rotorpy_state_to_target,
)
from intercept_sim.runner import RunnerStep
from intercept_sim.scene import make_scene_snapshot
from intercept_sim.sensors import FeaturePerceptionModel, GeometryCamera
from intercept_sim.targets import KinematicTarget
from intercept_sim.types import (
    CameraCapture,
    CameraIntrinsics,
    CameraRig,
    CtbrCommand,
    ImageFeatureMeasurement,
    ObserverState,
    SceneSnapshot,
    SimulationTarget,
)


_SENTINEL_TARGET = SimulationTarget(
    id="",
    kind="",
    position_w=np.zeros(3, dtype=float),
    velocity_w=np.zeros(3, dtype=float),
    rotation_wb=np.eye(3, dtype=float),
    radius_m=0.0,
)

_SENTINEL_INTRINSICS = CameraIntrinsics(
    width_px=0, height_px=0, fx_px=0.0, fy_px=0.0,
    cx_px=0.0, cy_px=0.0, hfov_rad=0.0, vfov_rad=0.0,
)

_SENTINEL_RIG = CameraRig(
    id="",
    parent_id="",
    position_b=np.zeros(3, dtype=float),
    body_to_camera=np.eye(3, dtype=float),
    intrinsics=_SENTINEL_INTRINSICS,
    capture_rate_hz=0.0,
)

_SENTINEL_SCENE = SceneSnapshot(
    t=0.0, pursuer=_SENTINEL_TARGET, targets=(), cameras=(),
)

_SENTINEL_CAPTURE = CameraCapture(
    t_capture=0.0, camera_id="", target_id=None, detected=False,
    uv_px=None, uv_norm=None, target_pos_c=None,
    range_m=None, apparent_radius_px=None,
)

_SENTINEL_OBSERVER_STATE = ObserverState(
    t=0.0,
    vehicle_state={},
    image_feature=None,
    relative_position_w=None,
    relative_velocity_w=None,
    target_acceleration_w=None,
    vehicle_rotation_wb=None,
)

_SENTINEL_CTBR = CtbrCommand(t=0.0, thrust_n=0.0, body_rates_b=np.zeros(3, dtype=float))

_SENTINEL_VEHICLE_STATE: dict[str, np.ndarray] = {
    "x": np.zeros(3, dtype=float),
    "v": np.zeros(3, dtype=float),
    "q": np.array([0.0, 0.0, 0.0, 1.0]),
    "w": np.zeros(3, dtype=float),
    "wind": np.zeros(3, dtype=float),
    "rotor_speeds": np.zeros(0, dtype=float),
}


def vehicle_state_value() -> AbstractValue:
    return AbstractValue.Make(dict(_SENTINEL_VEHICLE_STATE))


def scene_value() -> AbstractValue:
    return AbstractValue.Make(_SENTINEL_SCENE)


def target_value() -> AbstractValue:
    return AbstractValue.Make(_SENTINEL_TARGET)


def capture_value() -> AbstractValue:
    return AbstractValue.Make(_SENTINEL_CAPTURE)


def measurements_value() -> AbstractValue:
    return AbstractValue.Make(tuple())


def observer_state_value() -> AbstractValue:
    return AbstractValue.Make(_SENTINEL_OBSERVER_STATE)


def ctbr_value() -> AbstractValue:
    return AbstractValue.Make(_SENTINEL_CTBR)


class RunnerStepLogger(LeafSystem):
    def __init__(self, dt: float):
        super().__init__()
        self._dt = float(dt)
        self._log: list[RunnerStep] = []

        self.DeclareAbstractInputPort("vehicle_state_dict", vehicle_state_value())
        self.DeclareAbstractInputPort("scene", scene_value())
        self.DeclareAbstractInputPort("capture", capture_value())
        self.DeclareAbstractInputPort("measurements", measurements_value())
        self.DeclareAbstractInputPort("observer_state", observer_state_value())
        self.DeclareAbstractInputPort("ctbr_cmd", ctbr_value())
        self.DeclarePeriodicPublishEvent(
            period_sec=self._dt, offset_sec=0.0, publish=self._record,
        )

    def get_log(self) -> list[RunnerStep]:
        return list(self._log)

    def reset(self) -> None:
        self._log.clear()

    def _record(self, context):
        t = context.get_time()
        state = self.GetInputPort("vehicle_state_dict").Eval(context)
        scene = self.GetInputPort("scene").Eval(context)
        capture = self.GetInputPort("capture").Eval(context)
        measurements = self.GetInputPort("measurements").Eval(context)
        observer_state = self.GetInputPort("observer_state").Eval(context)
        cmd = self.GetInputPort("ctbr_cmd").Eval(context)

        if cmd is None:
            cmd = CtbrCommand(t=t, thrust_n=0.0, body_rates_b=np.zeros(3, dtype=float))

        rotorpy_state = {k: np.asarray(v, dtype=float).copy() for k, v in state.items()}
        self._log.append(
            RunnerStep(
                t=float(t),
                rotorpy_state=rotorpy_state,
                scene=scene,
                capture=capture,
                measurements=tuple(measurements),
                observer_state=observer_state,
                command=cmd,
            )
        )


class PixhawkInterface(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareAbstractInputPort("ctbr_cmd", ctbr_value())
        self.DeclareAbstractOutputPort("rate_cmd", ctbr_value, self._passthrough)

    def _passthrough(self, context, output):
        output.set_value(self.GetInputPort("ctbr_cmd").Eval(context))


class FeaturePerceptionSystem(LeafSystem):
    def __init__(self, perception: FeaturePerceptionModel, dt: float):
        super().__init__()
        self._perception = perception
        self._dt = float(dt)
        self.DeclareAbstractInputPort("capture", capture_value())
        self.DeclareAbstractOutputPort(
            "measurements", measurements_value, self._calc,
            prerequisites_of_calc={self.time_ticket()},
        )

    def _calc(self, context, output):
        capture = self.GetInputPort("capture").Eval(context)
        if capture is not None:
            self._perception.submit_capture(capture)
        output.set_value(tuple(self._perception.pop_available(context.get_time())))


class GeometryCameraSystem(LeafSystem):
    def __init__(self, camera: GeometryCamera, dt: float):
        super().__init__()
        self._camera = camera
        self._dt = float(dt)
        self.DeclareAbstractInputPort("scene", scene_value())
        self.DeclareAbstractOutputPort(
            "capture", capture_value, self._calc,
            prerequisites_of_calc={self.time_ticket()},
        )

    def _calc(self, context, output):
        output.set_value(self._camera.maybe_capture(self.GetInputPort("scene").Eval(context)))


class KinematicTargetSystem(LeafSystem):
    def __init__(self, target: KinematicTarget):
        super().__init__()
        self._target = target
        self.DeclareAbstractOutputPort("target_state", target_value, self._calc)

    def _calc(self, context, output):
        output.set_value(self._target.state_at(context.get_time()))


class SceneAssembler(LeafSystem):
    def __init__(self, camera_rig: CameraRig):
        super().__init__()
        self._cameras = (camera_rig,)
        self.DeclareAbstractInputPort("vehicle_state_dict", vehicle_state_value())
        self.DeclareAbstractInputPort("target_state", target_value())
        self.DeclareAbstractOutputPort(
            "scene", scene_value, self._calc,
            prerequisites_of_calc={self.time_ticket()},
        )

    def _calc(self, context, output):
        vehicle_state = self.GetInputPort("vehicle_state_dict").Eval(context)
        target = self.GetInputPort("target_state").Eval(context)
        pursuer = rotorpy_state_to_target(vehicle_state)
        scene = make_scene_snapshot(
            context.get_time(), pursuer, [target], list(self._cameras),
        )
        output.set_value(scene)


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_quad_params(vehicle_config: dict[str, Any]) -> dict[str, Any]:
    model = str(vehicle_config.get("model", "hummingbird")).lower()
    overrides = vehicle_config.get("params_override", {}) or {}

    if model in {"x500", "x500_v2", "holybro_x500", "holybro_x500_v2"}:
        params = deepcopy(_X500_QUAD_PARAMS)
    else:
        from intercept_sim.experiments.runner import _vehicle_params_from_config
        params = _vehicle_params_from_config(vehicle_config)

    if overrides:
        params = _deep_merge(params, overrides)
    return params


_X500_QUAD_PARAMS = {
    "mass": 2.064,
    "Ixx": 0.0217,
    "Iyy": 0.0217,
    "Izz": 0.0400,
    "Ixy": 0.0,
    "Iyz": 0.0,
    "Ixz": 0.0,
    "num_rotors": 4,
    "rotor_radius": 0.127,
    "rotor_pos": {
        "r1": np.array([0.174, 0.174, 0.0]),
        "r2": np.array([0.174, -0.174, 0.0]),
        "r3": np.array([-0.174, -0.174, 0.0]),
        "r4": np.array([-0.174, 0.174, 0.0]),
    },
    "rotor_directions": np.array([1, -1, 1, -1]),
    "rI": np.zeros(3),
    "c_Dx": 0.5e-2,
    "c_Dy": 0.5e-2,
    "c_Dz": 1.0e-2,
    "k_eta": 8.54858e-6,
    "k_m": 1.368e-7,
    "k_d": 8.06428e-5,
    "k_z": 2.32e-4,
    "k_h": 3.39e-3,
    "k_flap": 1.0e-6,
    "tau_m": 0.019,
    "rotor_speed_min": 0,
    "rotor_speed_max": 1000,
    "motor_noise_std": 0.0,
    "k_w": 1.0,
    "k_v": 10.0,
    "kp_att": 544.0,
    "kd_att": 46.64,
}
