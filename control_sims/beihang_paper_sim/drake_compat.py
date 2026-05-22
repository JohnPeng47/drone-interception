"""Small Drake compatibility surface used by the Beihang paper simulation."""

from __future__ import annotations

from copy import deepcopy
import importlib
from typing import Any

import numpy as np
from pydrake.common.value import AbstractValue
from scipy.spatial.transform import Rotation

from .types import (
    CameraCapture,
    CameraIntrinsics,
    CameraRig,
    CtbrCommand,
    ObserverState,
    SceneSnapshot,
    SimulationTarget,
)


def rotorpy_state_to_target(
    state: dict[str, np.ndarray],
    *,
    target_id: str = "interceptor",
    kind: str = "multirotor",
    radius_m: float = 0.15,
) -> SimulationTarget:
    return SimulationTarget(
        id=target_id,
        kind=kind,
        position_w=np.asarray(state["x"], dtype=float).copy(),
        velocity_w=np.asarray(state["v"], dtype=float).copy(),
        rotation_wb=Rotation.from_quat(state["q"]).as_matrix(),
        radius_m=radius_m,
    )


def ctbr_to_rotorpy(command: CtbrCommand) -> dict[str, np.ndarray | float]:
    return {
        "cmd_thrust": float(command.thrust_n),
        "cmd_w": np.asarray(command.body_rates_b, dtype=float).copy(),
    }


def hover_ctbr(t: float, mass_kg: float, gravity_mps2: float = 9.81) -> CtbrCommand:
    return CtbrCommand(
        t=float(t),
        thrust_n=float(mass_kg * gravity_mps2),
        body_rates_b=np.zeros(3, dtype=float),
    )


def make_scene_snapshot(
    t: float,
    pursuer: SimulationTarget,
    targets: list[SimulationTarget] | tuple[SimulationTarget, ...],
    cameras: list[CameraRig] | tuple[CameraRig, ...],
) -> SceneSnapshot:
    return SceneSnapshot(
        t=float(t),
        pursuer=pursuer,
        targets=tuple(targets),
        cameras=tuple(cameras),
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
        params = _vehicle_params_from_config(vehicle_config)

    if overrides:
        params = _deep_merge(params, overrides)
    return params


def _vehicle_params_from_config(vehicle_config: dict[str, Any]) -> dict[str, Any]:
    model = str(vehicle_config.get("model", "hummingbird")).lower()
    module_names = {
        "hummingbird": "rotorpy.vehicles.hummingbird_params",
        "crazyflie": "rotorpy.vehicles.crazyflie_params",
        "crazyfliebrushless": "rotorpy.vehicles.crazyfliebrushless_params",
        "px4_sihsim_quadx": "rotorpy.vehicles.px4_sihsim_quadx_params",
    }
    if model not in module_names:
        raise ValueError(f"Unsupported vehicle model: {model}")
    module = importlib.import_module(module_names[model])
    return dict(module.quad_params)


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
