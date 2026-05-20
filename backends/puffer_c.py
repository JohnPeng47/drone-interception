"""ctypes binding for the shared Puffer C drone simulation core."""

from __future__ import annotations

import ctypes as C
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PUFFER_DT = 0.002
PUFFER_ACTION_SUBSTEPS = 5
PUFFER_ACTION_DT = PUFFER_DT * PUFFER_ACTION_SUBSTEPS
DEFAULT_MAX_VEL_MPS = 100.0
DEFAULT_MAX_OMEGA_RPS = 100.0


@dataclass(frozen=True)
class VehicleParams:
    mass_kg: float
    ixx: float
    iyy: float
    izz: float
    arm_len_m: float
    k_thrust: float
    k_yaw: float
    k_ang_damp: float = 0.0
    b_drag: float = 0.0
    gravity_mps2: float = 9.81
    max_rpm: float = 21702.0
    max_vel_mps: float = DEFAULT_MAX_VEL_MPS
    max_omega_rps: float = DEFAULT_MAX_OMEGA_RPS
    motor_tau_s: float = 0.15
    rpm_min: float | None = None
    rotor_positions_b: np.ndarray | None = None
    rotor_directions: np.ndarray | None = None
    k_w: float = 1.0


@dataclass(frozen=True)
class InitialState:
    position_w: np.ndarray
    velocity_w: np.ndarray
    quat_xyzw: np.ndarray
    body_rates_b: np.ndarray
    rotor_speeds: np.ndarray | None = None
    wind_w: np.ndarray | None = None


@dataclass(frozen=True)
class SimOptions:
    backend_dt: float = PUFFER_DT
    action_substeps: int = PUFFER_ACTION_SUBSTEPS
    command_mode: str = "ctbr"
    ctbr_rate_gain: float = 0.08
    randomize_params: bool = False


class _CQuat(C.Structure):
    _fields_ = [("w", C.c_float), ("x", C.c_float), ("y", C.c_float), ("z", C.c_float)]


class _CVec3(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float)]


class _CState(C.Structure):
    _fields_ = [
        ("pos", _CVec3),
        ("vel", _CVec3),
        ("quat", _CQuat),
        ("omega", _CVec3),
        ("rpms", C.c_float * 4),
    ]


class _CParams(C.Structure):
    _fields_ = [
        ("mass", C.c_float),
        ("ixx", C.c_float),
        ("iyy", C.c_float),
        ("izz", C.c_float),
        ("arm_len", C.c_float),
        ("k_thrust", C.c_float),
        ("k_ang_damp", C.c_float),
        ("k_drag", C.c_float),
        ("b_drag", C.c_float),
        ("gravity", C.c_float),
        ("max_rpm", C.c_float),
        ("max_vel", C.c_float),
        ("max_omega", C.c_float),
        ("k_mot", C.c_float),
        ("rotor_pos_x", C.c_float * 4),
        ("rotor_pos_y", C.c_float * 4),
        ("rotor_dir", C.c_float * 4),
    ]


class _CDroneSim(C.Structure):
    _fields_ = [("state", _CState), ("params", _CParams)]


_LIB: C.CDLL | None = None


def vehicle_params_from_quad_params(quad_params: dict[str, Any]) -> VehicleParams:
    rotor_pos = quad_params.get("rotor_pos", {})
    arm_len = _infer_x_arm_len(rotor_pos)
    rotor_positions = _rotor_positions_array(rotor_pos)
    rotor_directions = np.asarray(quad_params.get("rotor_directions", np.ones(4)), dtype=float).reshape(4)
    return VehicleParams(
        mass_kg=float(quad_params["mass"]),
        ixx=float(quad_params["Ixx"]),
        iyy=float(quad_params["Iyy"]),
        izz=float(quad_params["Izz"]),
        arm_len_m=arm_len,
        k_thrust=float(quad_params["k_eta"]),
        # Puffer's k_drag is torque per thrust. RotorPy k_m is torque per
        # speed^2, so divide by k_eta to get the equivalent ratio.
        k_yaw=(
            float(quad_params.get("k_m", 0.0))
            / max(float(quad_params["k_eta"]), 1e-12)
        ),
        k_ang_damp=float(quad_params.get("k_ang_damp", 0.0)),
        b_drag=0.0,
        gravity_mps2=9.81,
        max_rpm=float(quad_params.get("rotor_speed_max", 21702.0)),
        max_vel_mps=float(quad_params.get("max_vel", DEFAULT_MAX_VEL_MPS)),
        max_omega_rps=float(quad_params.get("max_omega", DEFAULT_MAX_OMEGA_RPS)),
        motor_tau_s=float(quad_params.get("tau_m", 0.15)),
        rpm_min=float(quad_params.get("rotor_speed_min", 0.0)),
        rotor_positions_b=rotor_positions,
        rotor_directions=rotor_directions,
        k_w=float(quad_params.get("k_w", 1.0)),
    )


def initial_state_from_rotorpy(state: dict[str, np.ndarray]) -> InitialState:
    return InitialState(
        position_w=np.asarray(state["x"], dtype=float).copy(),
        velocity_w=np.asarray(state.get("v", np.zeros(3)), dtype=float).copy(),
        quat_xyzw=np.asarray(state.get("q", np.array([0.0, 0.0, 0.0, 1.0])), dtype=float).copy(),
        body_rates_b=np.asarray(state.get("w", np.zeros(3)), dtype=float).copy(),
        rotor_speeds=np.asarray(state.get("rotor_speeds"), dtype=float).copy()
        if state.get("rotor_speeds") is not None else None,
        wind_w=np.asarray(state.get("wind", np.zeros(3)), dtype=float).copy(),
    )


class PufferDroneBackend:
    def __init__(self, params: VehicleParams, options: SimOptions | None = None):
        self.params = params
        self.options = options or SimOptions()
        self.mass_kg = params.mass_kg
        self.dt = self.options.backend_dt * self.options.action_substeps
        self._lib = _load_lib()
        self._sim = _CDroneSim()

    def reset(self, initial_state: InitialState | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if isinstance(initial_state, dict):
            initial_state = initial_state_from_rotorpy(initial_state)
        rotor_speeds = initial_state.rotor_speeds
        if rotor_speeds is None:
            rotor_speeds = np.full(4, self._hover_rpm(), dtype=float)
        c_state = _state_to_c({
            "x": np.asarray(initial_state.position_w, dtype=float),
            "v": np.asarray(initial_state.velocity_w, dtype=float),
            "q": _normalize(np.asarray(initial_state.quat_xyzw, dtype=float)),
            "w": np.asarray(initial_state.body_rates_b, dtype=float),
            "rotor_speeds": np.asarray(rotor_speeds, dtype=float),
        })
        self._lib.drone_sim_init(C.byref(self._sim), _params_to_c(self.params), c_state)
        out = _state_from_c(self._sim.state)
        out["wind"] = (
            np.zeros(3, dtype=float)
            if initial_state.wind_w is None
            else np.asarray(initial_state.wind_w, dtype=float).copy()
        )
        return out

    def step_motor(
        self,
        state: dict[str, np.ndarray],
        motor_action: np.ndarray,
        dt: float | None = None,
    ) -> dict[str, np.ndarray]:
        dt = self.dt if dt is None else float(dt)
        wind = np.asarray(state.get("wind", np.zeros(3)), dtype=float).copy()
        self._lib.drone_sim_reset(C.byref(self._sim), _state_to_c(state))
        action_arr = (C.c_float * 4)(*np.clip(np.asarray(motor_action, dtype=float), -1.0, 1.0).reshape(4))
        self._lib.drone_sim_step_motor_dt(
            C.byref(self._sim),
            action_arr,
            C.c_float(dt),
            C.c_int(max(1, int(self.options.action_substeps))),
        )
        out = _state_from_c(self._sim.state)
        out["wind"] = wind
        return out

    def step_motor_speeds(
        self,
        state: dict[str, np.ndarray],
        cmd_motor_speeds: np.ndarray,
        dt: float | None = None,
    ) -> dict[str, np.ndarray]:
        dt = self.dt if dt is None else float(dt)
        wind = np.asarray(state.get("wind", np.zeros(3)), dtype=float).copy()
        self._lib.drone_sim_reset(C.byref(self._sim), _state_to_c(state))
        speeds = np.clip(np.asarray(cmd_motor_speeds, dtype=float).reshape(4), self._min_rpm(), self.params.max_rpm)
        speed_arr = (C.c_float * 4)(*speeds.astype(np.float32))
        self._lib.drone_sim_step_motor_speeds_dt(
            C.byref(self._sim),
            speed_arr,
            C.c_float(dt),
            C.c_int(max(1, int(self.options.action_substeps))),
        )
        out = _state_from_c(self._sim.state)
        out["wind"] = wind
        return out

    def step_ctbr(
        self,
        state: dict[str, np.ndarray],
        command: Any,
        dt: float | None = None,
    ) -> dict[str, np.ndarray]:
        if hasattr(command, "thrust_n"):
            thrust_n = float(command.thrust_n)
            body_rates_b = np.asarray(command.body_rates_b, dtype=float)
        else:
            thrust_n = float(command["thrust_n"])
            body_rates_b = np.asarray(command["body_rates_b"], dtype=float)
        cmd_speeds = self.ctbr_to_motor_speeds(state, thrust_n, body_rates_b)
        return self.step_motor_speeds(state, cmd_speeds, dt)

    def ctbr_to_motor_speeds(
        self,
        state: dict[str, np.ndarray],
        thrust_n: float,
        body_rates_b: np.ndarray,
    ) -> np.ndarray:
        omega = np.asarray(state.get("w", np.zeros(3)), dtype=float).reshape(3)
        body_rates_b = np.asarray(body_rates_b, dtype=float).reshape(3)
        wdot_cmd = self.params.k_w * (body_rates_b - omega)
        cmd_moment = np.array(
            [self.params.ixx, self.params.iyy, self.params.izz], dtype=float
        ) * wdot_cmd
        desired = np.array([max(thrust_n, 0.0), *cmd_moment], dtype=float)
        rotor_thrusts = self._tm_to_f() @ desired
        rotor_speed_sq = rotor_thrusts / max(self.params.k_thrust, 1e-12)
        cmd_speeds = np.sign(rotor_speed_sq) * np.sqrt(np.abs(rotor_speed_sq))
        return np.clip(cmd_speeds, self._min_rpm(), self.params.max_rpm)

    def ctbr_to_motor_action(
        self,
        state: dict[str, np.ndarray],
        thrust_n: float,
        body_rates_b: np.ndarray,
    ) -> np.ndarray:
        omega = np.asarray(state.get("w", np.zeros(3)), dtype=float)
        rate_error = np.asarray(body_rates_b, dtype=float).reshape(3) - omega
        tau = self.options.ctbr_rate_gain * np.array(
            [self.params.ixx, self.params.iyy, self.params.izz], dtype=float
        ) * rate_error

        arm_factor = self.params.arm_len_m / np.sqrt(2.0)
        allocation = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-arm_factor, -arm_factor, arm_factor, arm_factor],
            [-arm_factor, arm_factor, arm_factor, -arm_factor],
            [-self.params.k_yaw, self.params.k_yaw, -self.params.k_yaw, self.params.k_yaw],
        ], dtype=float)
        desired = np.array([max(thrust_n, 0.0), tau[0], tau[1], tau[2]], dtype=float)
        rotor_thrusts = np.clip(np.linalg.pinv(allocation) @ desired, 0.0, None)
        target_rpms = np.sqrt(rotor_thrusts / max(self.params.k_thrust, 1e-12))
        min_rpm = self._min_rpm()
        denom = max(self.params.max_rpm - min_rpm, 1e-9)
        return np.clip(2.0 * ((target_rpms - min_rpm) / denom) - 1.0, -1.0, 1.0)

    def _tm_to_f(self) -> np.ndarray:
        rotor_positions = self.params.rotor_positions_b
        rotor_directions = self.params.rotor_directions
        if rotor_positions is None or rotor_directions is None:
            arm_factor = self.params.arm_len_m / np.sqrt(2.0)
            allocation = np.array([
                [1.0, 1.0, 1.0, 1.0],
                [-arm_factor, -arm_factor, arm_factor, arm_factor],
                [-arm_factor, arm_factor, arm_factor, -arm_factor],
                [-self.params.k_yaw, self.params.k_yaw, -self.params.k_yaw, self.params.k_yaw],
            ], dtype=float)
        else:
            rotor_positions = np.asarray(rotor_positions, dtype=float).reshape(4, 3)
            rotor_directions = np.asarray(rotor_directions, dtype=float).reshape(4)
            yaw = self.params.k_yaw * rotor_directions
            allocation = np.vstack((
                np.ones(4),
                rotor_positions[:, 1],
                -rotor_positions[:, 0],
                yaw,
            ))
        return np.linalg.inv(allocation)

    def _hover_rpm(self) -> float:
        return float(np.sqrt((self.params.mass_kg * self.params.gravity_mps2) / (4.0 * self.params.k_thrust)))

    def _min_rpm(self) -> float:
        if self.params.rpm_min is not None:
            return float(np.clip(self.params.rpm_min, 0.0, self.params.max_rpm))
        # Match intercept_env/dronelib.h:rpm_min_for_centered_hover for the
        # legacy normalized action path when no vehicle lower bound is known.
        min_rpm = 2.0 * self._hover_rpm() - self.params.max_rpm
        return float(np.clip(min_rpm, 0.0, self.params.max_rpm))


def _load_lib() -> C.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    src = Path(__file__).resolve().parents[1] / "intercept_env" / "sim_core.c"
    header = Path(__file__).resolve().parents[1] / "intercept_env" / "sim_core.h"
    dronelib = Path(__file__).resolve().parents[1] / "intercept_env" / "dronelib.h"
    build_dir = Path(__file__).resolve().parent / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    lib_path = build_dir / "libpuffer_sim_core.so"
    newest_source_mtime = max(src.stat().st_mtime, header.stat().st_mtime, dronelib.stat().st_mtime)
    if (not lib_path.exists()) or lib_path.stat().st_mtime < newest_source_mtime:
        subprocess.run(
            ["cc", "-std=gnu99", "-O3", "-fPIC", "-shared", str(src), "-lm", "-o", str(lib_path)],
            check=True,
        )

    lib = C.CDLL(str(lib_path))
    lib.drone_sim_init.argtypes = [C.POINTER(_CDroneSim), _CParams, _CState]
    lib.drone_sim_init.restype = None
    lib.drone_sim_reset.argtypes = [C.POINTER(_CDroneSim), _CState]
    lib.drone_sim_reset.restype = None
    lib.drone_sim_step_motor.argtypes = [C.POINTER(_CDroneSim), C.POINTER(C.c_float)]
    lib.drone_sim_step_motor.restype = None
    lib.drone_sim_step_motor_dt.argtypes = [C.POINTER(_CDroneSim), C.POINTER(C.c_float), C.c_float, C.c_int]
    lib.drone_sim_step_motor_dt.restype = None
    lib.drone_sim_step_motor_speeds_dt.argtypes = [C.POINTER(_CDroneSim), C.POINTER(C.c_float), C.c_float, C.c_int]
    lib.drone_sim_step_motor_speeds_dt.restype = None
    lib.drone_sim_get_state.argtypes = [C.POINTER(_CDroneSim)]
    lib.drone_sim_get_state.restype = _CState
    _LIB = lib
    return lib


def _params_to_c(p: VehicleParams) -> _CParams:
    rotor_positions = (
        np.zeros((4, 3), dtype=float)
        if p.rotor_positions_b is None
        else np.asarray(p.rotor_positions_b, dtype=float).reshape(4, 3)
    )
    rotor_directions = (
        np.zeros(4, dtype=float)
        if p.rotor_directions is None
        else np.asarray(p.rotor_directions, dtype=float).reshape(4)
    )
    return _CParams(
        C.c_float(p.mass_kg),
        C.c_float(p.ixx),
        C.c_float(p.iyy),
        C.c_float(p.izz),
        C.c_float(p.arm_len_m),
        C.c_float(p.k_thrust),
        C.c_float(p.k_ang_damp),
        C.c_float(p.k_yaw),
        C.c_float(p.b_drag),
        C.c_float(p.gravity_mps2),
        C.c_float(p.max_rpm),
        C.c_float(p.max_vel_mps),
        C.c_float(p.max_omega_rps),
        C.c_float(p.motor_tau_s),
        (C.c_float * 4)(*rotor_positions[:, 0].astype(np.float32)),
        (C.c_float * 4)(*rotor_positions[:, 1].astype(np.float32)),
        (C.c_float * 4)(*rotor_directions.astype(np.float32)),
    )


def _state_to_c(state: dict[str, np.ndarray]) -> _CState:
    x = np.asarray(state["x"], dtype=float).reshape(3)
    v = np.asarray(state["v"], dtype=float).reshape(3)
    q = _normalize(np.asarray(state["q"], dtype=float).reshape(4))
    w = np.asarray(state["w"], dtype=float).reshape(3)
    rpms = np.asarray(state["rotor_speeds"], dtype=float).reshape(4)
    return _CState(
        _CVec3(C.c_float(x[0]), C.c_float(x[1]), C.c_float(x[2])),
        _CVec3(C.c_float(v[0]), C.c_float(v[1]), C.c_float(v[2])),
        _CQuat(C.c_float(q[3]), C.c_float(q[0]), C.c_float(q[1]), C.c_float(q[2])),
        _CVec3(C.c_float(w[0]), C.c_float(w[1]), C.c_float(w[2])),
        (C.c_float * 4)(*rpms.astype(np.float32)),
    )


def _state_from_c(state: _CState) -> dict[str, np.ndarray]:
    return {
        "x": np.array([state.pos.x, state.pos.y, state.pos.z], dtype=float),
        "v": np.array([state.vel.x, state.vel.y, state.vel.z], dtype=float),
        "q": _normalize(np.array([state.quat.x, state.quat.y, state.quat.z, state.quat.w], dtype=float)),
        "w": np.array([state.omega.x, state.omega.y, state.omega.z], dtype=float),
        "rotor_speeds": np.array(list(state.rpms), dtype=float),
    }


def _infer_x_arm_len(rotor_pos: dict[str, np.ndarray]) -> float:
    if not rotor_pos:
        return 0.0396
    first = np.asarray(next(iter(rotor_pos.values())), dtype=float)
    return float(np.linalg.norm(first[:2]))


def _rotor_positions_array(rotor_pos: dict[str, np.ndarray]) -> np.ndarray | None:
    if not rotor_pos:
        return None
    return np.vstack([np.asarray(rotor_pos[key], dtype=float).reshape(3) for key in rotor_pos])


def _normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q if n <= 1e-12 else q / n
