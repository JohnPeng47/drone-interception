"""ctypes binding for the shared Puffer C simulation core."""

from __future__ import annotations

import ctypes as C
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .types import (
    DEFAULT_MAX_OMEGA_RPS,
    DEFAULT_MAX_VEL_MPS,
    PursuerInitialState,
    PursuerParams,
    RenderConfig,
    SimConfig,
    SimInstance,
    SimOptions,
    TargetConfig,
)


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


class _CPursuerParams(C.Structure):
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


class _CPursuerSim(C.Structure):
    _fields_ = [("state", _CState), ("params", _CPursuerParams)]


SIM_MAX_TARGETS = 16
SIM_MAX_WAYPOINTS = 64
SIM_MAX_CAMERAS = 8
SIM_MAX_CAMERA_OUTPUTS = 8
TARGET_CONTROLLER_LINEAR = 0
TARGET_BEHAVIOR_WAYPOINTS = 0


class _CTargetState(C.Structure):
    _fields_ = [("pos", _CVec3), ("vel", _CVec3)]


class _CTargetReference(C.Structure):
    _fields_ = [("pos", _CVec3), ("vel", _CVec3)]


class _CTargetCommand(C.Structure):
    _fields_ = [("accel", _CVec3)]


class _CTargetControllerConfig(C.Structure):
    _fields_ = [
        ("kind", C.c_int),
        ("kp", C.c_float),
        ("kv", C.c_float),
        ("max_accel", C.c_float),
    ]


class _CTargetBehaviorConfig(C.Structure):
    _fields_ = [
        ("kind", C.c_int),
        ("num_waypoints", C.c_int),
        ("waypoints", _CVec3 * SIM_MAX_WAYPOINTS),
        ("duration", C.c_float),
        ("loop", C.c_int),
    ]


class _CTargetSim(C.Structure):
    _fields_ = [
        ("id", C.c_int),
        ("radius", C.c_float),
        ("state", _CTargetState),
        ("behavior", _CTargetBehaviorConfig),
        ("controller", _CTargetControllerConfig),
    ]


class _CInterceptMetrics(C.Structure):
    _fields_ = [
        ("distance_m", C.c_float),
        ("min_distance_m", C.c_float),
        ("intercepted", C.c_int),
        ("intercept_time_s", C.c_float),
        ("target_index", C.c_int),
    ]


class _CCameraIntrinsics(C.Structure):
    _fields_ = [
        ("width_px", C.c_int),
        ("height_px", C.c_int),
        ("fx_px", C.c_float),
        ("fy_px", C.c_float),
        ("cx_px", C.c_float),
        ("cy_px", C.c_float),
        ("hfov_rad", C.c_float),
        ("vfov_rad", C.c_float),
    ]


class _CMat3(C.Structure):
    _fields_ = [("m", (C.c_float * 3) * 3)]


class _CCameraSim(C.Structure):
    _fields_ = [
        ("id", C.c_int),
        ("parent_actor", C.c_int),
        ("position_b", _CVec3),
        ("body_to_camera", _CMat3),
        ("intrinsics", _CCameraIntrinsics),
        ("capture_rate_hz", C.c_float),
        ("next_capture_t", C.c_float),
    ]


class _CCameraObservation(C.Structure):
    _fields_ = [
        ("camera_id", C.c_int),
        ("target_index", C.c_int),
        ("captured", C.c_int),
        ("detected", C.c_int),
        ("t_capture", C.c_float),
        ("target_pos_c", _CVec3),
        ("range_m", C.c_float),
        ("uv_norm", C.c_float * 2),
        ("uv_px", C.c_float * 2),
        ("apparent_radius_px", C.c_float),
    ]


class _CCameraOutput(C.Structure):
    _fields_ = [
        ("observation", _CCameraObservation),
        ("has_frame", C.c_int),
        ("frame_width_px", C.c_int),
        ("frame_height_px", C.c_int),
        ("frame_channels", C.c_int),
        ("frame_rgb", C.c_void_p),
    ]


class _CSimEngine(C.Structure):
    _fields_ = [
        ("pursuer", _CPursuerSim),
        ("targets", _CTargetSim * SIM_MAX_TARGETS),
        ("cameras", _CCameraSim * SIM_MAX_CAMERAS),
        ("num_targets", C.c_int),
        ("num_cameras", C.c_int),
        ("t", C.c_float),
        ("intercept_radius_m", C.c_float),
        ("metrics", _CInterceptMetrics),
    ]


_LIB: C.CDLL | None = None


def vehicle_params_from_quad_params(quad_params: dict[str, Any]) -> PursuerParams:
    rotor_pos = quad_params.get("rotor_pos", {})
    arm_len = _infer_x_arm_len(rotor_pos)
    rotor_positions = _rotor_positions_array(rotor_pos)
    rotor_directions = np.asarray(quad_params.get("rotor_directions", np.ones(4)), dtype=float).reshape(4)
    return PursuerParams(
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


def initial_state_from_rotorpy(state: dict[str, np.ndarray]) -> PursuerInitialState:
    return PursuerInitialState(
        position_w=np.asarray(state["x"], dtype=float).copy(),
        velocity_w=np.asarray(state.get("v", np.zeros(3)), dtype=float).copy(),
        quat_xyzw=np.asarray(state.get("q", np.array([0.0, 0.0, 0.0, 1.0])), dtype=float).copy(),
        body_rates_b=np.asarray(state.get("w", np.zeros(3)), dtype=float).copy(),
        rotor_speeds=np.asarray(state.get("rotor_speeds"), dtype=float).copy()
        if state.get("rotor_speeds") is not None else None,
        wind_w=np.asarray(state.get("wind", np.zeros(3)), dtype=float).copy(),
    )


class PufferDroneBackend:
    def __init__(self, params: PursuerParams, options: SimOptions | None = None):
        self.params = params
        self.options = options or SimOptions()
        self.mass_kg = params.mass_kg
        self.dt = self.options.backend_dt * self.options.action_substeps
        self._lib = _load_lib()
        self._sim = _CPursuerSim()

    def reset(self, initial_state: PursuerInitialState | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
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
        self._lib.pursuer_sim_init(C.byref(self._sim), _params_to_c(self.params), c_state)
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
        self._lib.pursuer_sim_reset(C.byref(self._sim), _state_to_c(state))
        action_arr = (C.c_float * 4)(*np.clip(np.asarray(motor_action, dtype=float), -1.0, 1.0).reshape(4))
        self._lib.pursuer_sim_step_motor_dt(
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
        self._lib.pursuer_sim_reset(C.byref(self._sim), _state_to_c(state))
        speeds = np.clip(np.asarray(cmd_motor_speeds, dtype=float).reshape(4), self._min_rpm(), self.params.max_rpm)
        speed_arr = (C.c_float * 4)(*speeds.astype(np.float32))
        self._lib.pursuer_sim_step_motor_speeds_dt(
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
        # Match backends/csim/pursuer_sim.c:rpm_min_for_centered_hover for the
        # legacy normalized action path when no vehicle lower bound is known.
        min_rpm = 2.0 * self._hover_rpm() - self.params.max_rpm
        return float(np.clip(min_rpm, 0.0, self.params.max_rpm))


class PufferSimEngineBackend(PufferDroneBackend):
    """Python adapter for the C SimEngine pursuer plus target world."""

    def __init__(self, params: PursuerParams | SimConfig, options: SimOptions | None = None):
        self.config = params if isinstance(params, SimConfig) else None
        self.params = params.pursuer if isinstance(params, SimConfig) else params
        self.options = options or (params.options if isinstance(params, SimConfig) else SimOptions())
        self.render_config = params.render if isinstance(params, SimConfig) else RenderConfig()
        self.mass_kg = self.params.mass_kg
        self.dt = self.options.backend_dt * self.options.action_substeps
        self._lib = _load_lib()
        self._engine = _CSimEngine()
        self._target_specs: tuple[dict[str, Any], ...] = ()
        self._camera_specs: tuple[dict[str, Any], ...] = ()
        self._renderer: Any | None = None
        self._renderer_key: tuple[Any, ...] | None = None
        self._render_sequence_id = 0

    def reset(
        self,
        initial_state: PursuerInitialState | SimInstance | dict[str, np.ndarray],
        targets: list[Any] | tuple[Any, ...] = (),
        cameras: list[Any] | tuple[Any, ...] = (),
        intercept_radius_m: float | None = None,
    ) -> dict[str, Any]:
        if isinstance(initial_state, SimInstance):
            instance = initial_state
            initial_state = instance.pursuer_initial
            targets = instance.targets
            cameras = instance.cameras
            if instance.config is not None:
                self.render_config = instance.config.render
            if intercept_radius_m is None:
                intercept_radius_m = (
                    instance.config.intercept_radius_m
                    if instance.config is not None
                    else (
                        self.config.intercept_radius_m
                        if self.config is not None
                        else 0.0
                    )
                )
        if intercept_radius_m is None:
            intercept_radius_m = self.config.intercept_radius_m if self.config is not None else 0.0
        if isinstance(initial_state, dict):
            initial_state = initial_state_from_rotorpy(initial_state)
        rotor_speeds = initial_state.rotor_speeds
        if rotor_speeds is None:
            rotor_speeds = np.full(4, self._hover_rpm(), dtype=float)

        wind = (
            np.zeros(3, dtype=float)
            if initial_state.wind_w is None
            else np.asarray(initial_state.wind_w, dtype=float).copy()
        )
        c_state = _state_to_c({
            "x": np.asarray(initial_state.position_w, dtype=float),
            "v": np.asarray(initial_state.velocity_w, dtype=float),
            "q": _normalize(np.asarray(initial_state.quat_xyzw, dtype=float)),
            "w": np.asarray(initial_state.body_rates_b, dtype=float),
            "rotor_speeds": np.asarray(rotor_speeds, dtype=float),
        })
        self._lib.sim_engine_init(C.byref(self._engine), _params_to_c(self.params), c_state)
        self._lib.sim_engine_set_intercept_radius(
            C.byref(self._engine),
            C.c_float(float(intercept_radius_m)),
        )
        self._target_specs = tuple(_target_spec_from_python(target, i) for i, target in enumerate(targets))
        self._set_engine_targets(self._target_specs)
        self._camera_specs = tuple(_camera_spec_from_python(camera, i) for i, camera in enumerate(cameras))
        self._set_engine_cameras(self._camera_specs)
        self._configure_renderer(self.render_config)
        camera_outputs = self._collect_camera_outputs()
        return self._snapshot_from_engine(wind, camera_outputs=camera_outputs)

    def step_ctbr(
        self,
        snapshot: dict[str, Any],
        command: Any,
        dt: float | None = None,
    ) -> dict[str, Any]:
        if hasattr(command, "thrust_n"):
            thrust_n = float(command.thrust_n)
            body_rates_b = np.asarray(command.body_rates_b, dtype=float)
        else:
            thrust_n = float(command["thrust_n"])
            body_rates_b = np.asarray(command["body_rates_b"], dtype=float)

        dt = self.dt if dt is None else float(dt)
        vehicle_state = _copy_numeric_state(snapshot["vehicle_state"])
        wind = np.asarray(vehicle_state.get("wind", np.zeros(3)), dtype=float).copy()
        self._lib.sim_engine_init(
            C.byref(self._engine),
            _params_to_c(self.params),
            _state_to_c(vehicle_state),
        )
        self._lib.sim_engine_set_intercept_radius(
            C.byref(self._engine),
            C.c_float(float(snapshot.get("intercept_radius_m", 0.0))),
        )
        metrics = snapshot.get("metrics")
        if metrics is not None:
            self._engine.metrics = _metrics_to_c(metrics)
        target_specs = _target_specs_from_snapshot(
            snapshot.get("target_states", ()),
            self._target_specs,
        )
        self._set_engine_targets(target_specs)
        camera_specs = _camera_specs_from_snapshot(
            snapshot.get("camera_states", ()),
            self._camera_specs,
        )
        self._set_engine_cameras(camera_specs)

        cmd_speeds = self.ctbr_to_motor_speeds(vehicle_state, thrust_n, body_rates_b)
        speed_arr = (C.c_float * 4)(*cmd_speeds.astype(np.float32))
        self._lib.sim_engine_step_motor_speeds_dt(
            C.byref(self._engine),
            speed_arr,
            C.c_float(dt),
            C.c_int(max(1, int(self.options.action_substeps))),
        )
        camera_outputs = self._collect_camera_outputs()
        return self._snapshot_from_engine(wind, camera_outputs=camera_outputs)

    def _set_engine_targets(self, specs: tuple[dict[str, Any], ...]) -> None:
        c_targets = [_target_to_c(spec) for spec in specs]
        if not c_targets:
            self._lib.sim_engine_clear_targets(C.byref(self._engine))
            return
        arr_type = _CTargetSim * len(c_targets)
        count = self._lib.sim_engine_set_targets(
            C.byref(self._engine),
            arr_type(*c_targets),
            C.c_int(len(c_targets)),
        )
        if count != len(c_targets):
            raise ValueError(f"SimEngine accepted {count} targets out of {len(c_targets)}")

    def _set_engine_cameras(self, specs: tuple[dict[str, Any], ...]) -> None:
        c_cameras = [_camera_to_c(spec) for spec in specs]
        if not c_cameras:
            self._lib.sim_engine_clear_cameras(C.byref(self._engine))
            return
        arr_type = _CCameraSim * len(c_cameras)
        count = self._lib.sim_engine_set_cameras(
            C.byref(self._engine),
            arr_type(*c_cameras),
            C.c_int(len(c_cameras)),
        )
        if count != len(c_cameras):
            raise ValueError(f"SimEngine accepted {count} cameras out of {len(c_cameras)}")

    def _collect_camera_outputs(self) -> tuple[dict[str, Any], ...]:
        outputs = (_CCameraOutput * SIM_MAX_CAMERA_OUTPUTS)()
        count = self._lib.sim_engine_collect_camera_outputs(
            C.byref(self._engine),
            outputs,
            C.c_int(SIM_MAX_CAMERA_OUTPUTS),
        )
        camera_outputs = tuple(
            _camera_output_from_c(outputs[i], self._target_specs, self._camera_specs)
            for i in range(int(count))
        )
        return self._render_camera_outputs(camera_outputs)

    def _configure_renderer(self, config: RenderConfig) -> None:
        key = _render_config_key(config)
        if not config.enabled:
            self._close_renderer()
            self._renderer_key = key
            return
        if self._renderer is not None and self._renderer_key == key:
            return
        self._close_renderer()
        from rendering.python import NativeRenderEngine

        self._renderer = NativeRenderEngine(config)
        self._renderer_key = key

    def _close_renderer(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _render_camera_outputs(
        self,
        camera_outputs: tuple[dict[str, Any], ...],
    ) -> tuple[dict[str, Any], ...]:
        # Stage A renderer hook: keep C sim geometry authoritative while the
        # production C-level renderer integration is still being built.
        if self._renderer is None or not self.render_config.enabled:
            return camera_outputs
        rendered = []
        for output in camera_outputs:
            if not self._render_output_selected(output):
                rendered.append(output)
                continue
            self._render_sequence_id += 1
            result = self._renderer.render_frame(
                drone=_render_drone_state_from_engine(self._engine),
                camera=_camera_render_state(self._engine, output),
                targets=_render_targets_from_engine(self._engine, self._target_specs),
                sequence_id=self._render_sequence_id,
            )
            if self.render_config.fail_on_error and result.status not in (0, 1):
                from rendering.python import RenderError

                raise RenderError(result.status, f"render failed: {result.status_name}")
            rendered_output = dict(output)
            rendered_output.update({
                "render_status": result.status,
                "render_status_name": result.status_name,
                "has_frame": result.has_frame,
                "frame_width_px": result.width_px or output["frame_width_px"],
                "frame_height_px": result.height_px or output["frame_height_px"],
                "frame_channels": result.channels or output["frame_channels"],
                "frame_stride_bytes": result.stride_bytes,
                "frame_rgb": result.pixels,
            })
            rendered.append(rendered_output)
        return tuple(rendered)

    def _render_output_selected(self, output: dict[str, Any]) -> bool:
        camera_id = self.render_config.camera_id
        if camera_id is None:
            return True
        return str(camera_id) in {
            str(output.get("camera_id")),
            str(output.get("c_camera_id")),
        }

    def _snapshot_from_engine(
        self,
        wind: np.ndarray,
        camera_outputs: tuple[dict[str, Any], ...] = (),
    ) -> dict[str, Any]:
        vehicle = _state_from_c(self._lib.sim_engine_get_pursuer_state(C.byref(self._engine)))
        vehicle["wind"] = np.asarray(wind, dtype=float).copy()
        targets = []
        count = int(self._lib.sim_engine_get_num_targets(C.byref(self._engine)))
        for i in range(count):
            state = self._lib.sim_engine_get_target_state(C.byref(self._engine), C.c_int(i))
            spec = self._target_specs[i] if i < len(self._target_specs) else {}
            targets.append(_target_snapshot_from_c(state, spec, self._engine.targets[i]))
        return {
            "vehicle_state": vehicle,
            "target_states": tuple(targets),
            "intercept_radius_m": float(self._engine.intercept_radius_m),
            "metrics": _metrics_from_c(
                self._lib.sim_engine_get_metrics(C.byref(self._engine))
            ),
            "camera_states": _camera_states_from_engine(self._engine, self._camera_specs),
            "camera_outputs": tuple(camera_outputs),
        }


def _load_lib() -> C.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    csim_dir = Path(__file__).resolve().parents[1]
    sources = [
        csim_dir / "pursuer_sim.c",
        csim_dir / "target_sim.c",
        csim_dir / "sim_engine.c",
        csim_dir / "camera_sim.c",
    ]
    headers = [
        csim_dir / "sim_core.h",
        csim_dir / "target_sim.h",
        csim_dir / "sim_engine.h",
        csim_dir / "camera_sim.h",
        csim_dir / "sim_types.h",
        csim_dir / "sim_math.h",
    ]
    build_dir = Path(__file__).resolve().parent / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    lib_path = build_dir / "libpuffer_sim_core.so"
    newest_source_mtime = max(
        *(path.stat().st_mtime for path in sources),
        *(path.stat().st_mtime for path in headers),
    )
    if (not lib_path.exists()) or lib_path.stat().st_mtime < newest_source_mtime:
        subprocess.run(
            [
                "cc", "-std=gnu99", "-O3", "-fPIC", "-shared",
                *(str(src) for src in sources),
                "-lm", "-o", str(lib_path),
            ],
            check=True,
        )

    lib = C.CDLL(str(lib_path))
    lib.pursuer_sim_init.argtypes = [C.POINTER(_CPursuerSim), _CPursuerParams, _CState]
    lib.pursuer_sim_init.restype = None
    lib.pursuer_sim_reset.argtypes = [C.POINTER(_CPursuerSim), _CState]
    lib.pursuer_sim_reset.restype = None
    lib.pursuer_sim_step_motor.argtypes = [C.POINTER(_CPursuerSim), C.POINTER(C.c_float)]
    lib.pursuer_sim_step_motor.restype = None
    lib.pursuer_sim_step_motor_dt.argtypes = [C.POINTER(_CPursuerSim), C.POINTER(C.c_float), C.c_float, C.c_int]
    lib.pursuer_sim_step_motor_dt.restype = None
    lib.pursuer_sim_step_motor_speeds_dt.argtypes = [C.POINTER(_CPursuerSim), C.POINTER(C.c_float), C.c_float, C.c_int]
    lib.pursuer_sim_step_motor_speeds_dt.restype = None
    lib.pursuer_sim_get_state.argtypes = [C.POINTER(_CPursuerSim)]
    lib.pursuer_sim_get_state.restype = _CState
    lib.sim_engine_init.argtypes = [C.POINTER(_CSimEngine), _CPursuerParams, _CState]
    lib.sim_engine_init.restype = None
    lib.sim_engine_reset.argtypes = [C.POINTER(_CSimEngine), _CState]
    lib.sim_engine_reset.restype = None
    lib.sim_engine_set_intercept_radius.argtypes = [C.POINTER(_CSimEngine), C.c_float]
    lib.sim_engine_set_intercept_radius.restype = None
    lib.sim_engine_clear_targets.argtypes = [C.POINTER(_CSimEngine)]
    lib.sim_engine_clear_targets.restype = None
    lib.sim_engine_clear_cameras.argtypes = [C.POINTER(_CSimEngine)]
    lib.sim_engine_clear_cameras.restype = None
    lib.sim_engine_set_targets.argtypes = [C.POINTER(_CSimEngine), C.POINTER(_CTargetSim), C.c_int]
    lib.sim_engine_set_targets.restype = C.c_int
    lib.sim_engine_add_target.argtypes = [C.POINTER(_CSimEngine), _CTargetSim]
    lib.sim_engine_add_target.restype = C.c_int
    lib.sim_engine_set_cameras.argtypes = [C.POINTER(_CSimEngine), C.POINTER(_CCameraSim), C.c_int]
    lib.sim_engine_set_cameras.restype = C.c_int
    lib.sim_engine_add_camera.argtypes = [C.POINTER(_CSimEngine), _CCameraSim]
    lib.sim_engine_add_camera.restype = C.c_int
    lib.sim_engine_collect_camera_outputs.argtypes = [C.POINTER(_CSimEngine), C.POINTER(_CCameraOutput), C.c_int]
    lib.sim_engine_collect_camera_outputs.restype = C.c_int
    lib.sim_engine_step_motor_dt.argtypes = [C.POINTER(_CSimEngine), C.POINTER(C.c_float), C.c_float, C.c_int]
    lib.sim_engine_step_motor_dt.restype = None
    lib.sim_engine_step_motor_speeds_dt.argtypes = [C.POINTER(_CSimEngine), C.POINTER(C.c_float), C.c_float, C.c_int]
    lib.sim_engine_step_motor_speeds_dt.restype = None
    lib.sim_engine_get_pursuer_state.argtypes = [C.POINTER(_CSimEngine)]
    lib.sim_engine_get_pursuer_state.restype = _CState
    lib.sim_engine_get_num_targets.argtypes = [C.POINTER(_CSimEngine)]
    lib.sim_engine_get_num_targets.restype = C.c_int
    lib.sim_engine_get_target_state.argtypes = [C.POINTER(_CSimEngine), C.c_int]
    lib.sim_engine_get_target_state.restype = _CTargetState
    lib.sim_engine_get_metrics.argtypes = [C.POINTER(_CSimEngine)]
    lib.sim_engine_get_metrics.restype = _CInterceptMetrics
    _LIB = lib
    return lib


def _params_to_c(p: PursuerParams) -> _CPursuerParams:
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
    return _CPursuerParams(
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


def _target_spec_from_python(target: Any, index: int = 0) -> dict[str, Any]:
    behavior_kind = TARGET_BEHAVIOR_WAYPOINTS
    waypoints = None
    duration = 0.0
    loop = 0
    controller_kind = TARGET_CONTROLLER_LINEAR
    kp = 0.0
    kv = 0.0
    max_accel = 0.0
    if isinstance(target, dict):
        pos = target.get("position_w", target.get("initial_position_w", target.get("pos")))
        vel = target.get("velocity_w", target.get("vel"))
        target_id = target.get("id", target.get("target_id", str(index)))
        kind = target.get("kind", "target")
        radius = target.get("radius_m", target.get("radius", 0.0))
    elif isinstance(target, TargetConfig):
        pos = target.initial.position_w
        vel = target.initial.velocity_w
        target_id = target.id
        kind = target.kind
        radius = target.radius_m
        waypoints = target.behavior.waypoints
        duration = target.behavior.duration_s
        loop = int(target.behavior.loop)
        kp = target.controller.kp
        kv = target.controller.kv
        max_accel = target.controller.max_accel_mps2
    else:
        initial = getattr(target, "initial", None)
        pos = (
            getattr(initial, "position_w", None)
            if initial is not None
            else getattr(target, "initial_position_w", getattr(target, "position_w", None))
        )
        vel = (
            getattr(initial, "velocity_w", None)
            if initial is not None
            else getattr(target, "velocity_w", None)
        )
        target_id = getattr(target, "target_id", getattr(target, "id", str(index)))
        kind = getattr(target, "kind", "target")
        radius = getattr(target, "radius_m", getattr(target, "radius", 0.0))
        behavior = getattr(target, "behavior", None)
        controller = getattr(target, "controller", None)
        if behavior is not None:
            waypoints = getattr(behavior, "waypoints", None)
            duration = getattr(behavior, "duration_s", getattr(behavior, "duration", duration))
            loop = int(bool(getattr(behavior, "loop", loop)))
        if controller is not None:
            kp = getattr(controller, "kp", kp)
            kv = getattr(controller, "kv", kv)
            max_accel = getattr(controller, "max_accel_mps2", getattr(controller, "max_accel", max_accel))
    if pos is None or vel is None:
        raise ValueError(f"Target {index} must provide position and velocity")
    waypoints = () if waypoints is None else tuple(waypoints)
    if not waypoints:
        waypoints = (np.asarray(pos, dtype=float).reshape(3).copy(),)
    return {
        "c_id": int(index),
        "id": str(target_id),
        "kind": str(kind),
        "radius_m": float(radius),
        "position_w": np.asarray(pos, dtype=float).reshape(3).copy(),
        "velocity_w": np.asarray(vel, dtype=float).reshape(3).copy(),
        "behavior_kind": behavior_kind,
        "waypoints": tuple(np.asarray(wp, dtype=float).reshape(3).copy() for wp in waypoints),
        "duration": float(duration),
        "loop": int(loop),
        "controller_kind": controller_kind,
        "kp": float(kp),
        "kv": float(kv),
        "max_accel": float(max_accel),
    }


def _target_specs_from_snapshot(
    snapshots: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    base_specs: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for i, snap in enumerate(snapshots):
        base = dict(base_specs[i]) if i < len(base_specs) else {
            "c_id": i,
            "id": str(i),
            "kind": "target",
            "radius_m": 0.0,
            "behavior_kind": TARGET_BEHAVIOR_WAYPOINTS,
            "waypoints": (np.asarray(snap["position_w"], dtype=float).reshape(3).copy(),),
            "duration": 0.0,
            "loop": 0,
            "controller_kind": TARGET_CONTROLLER_LINEAR,
            "kp": 0.0,
            "kv": 0.0,
            "max_accel": 0.0,
        }
        base["position_w"] = np.asarray(snap["position_w"], dtype=float).reshape(3).copy()
        base["velocity_w"] = np.asarray(snap["velocity_w"], dtype=float).reshape(3).copy()
        specs.append(base)
    return tuple(specs)


def _target_to_c(spec: dict[str, Any]) -> _CTargetSim:
    waypoints = (_CVec3 * SIM_MAX_WAYPOINTS)()
    waypoint_values = tuple(spec.get("waypoints", (spec["position_w"],)))
    num_waypoints = min(len(waypoint_values), SIM_MAX_WAYPOINTS)
    for i in range(num_waypoints):
        waypoints[i] = _vec3(waypoint_values[i])
    state = _CTargetState(
        _vec3(spec["position_w"]),
        _vec3(spec["velocity_w"]),
    )
    behavior = _CTargetBehaviorConfig(
        C.c_int(int(spec.get("behavior_kind", TARGET_BEHAVIOR_WAYPOINTS))),
        C.c_int(num_waypoints),
        waypoints,
        C.c_float(float(spec.get("duration", 0.0))),
        C.c_int(int(spec.get("loop", 0))),
    )
    controller = _CTargetControllerConfig(
        C.c_int(int(spec.get("controller_kind", TARGET_CONTROLLER_LINEAR))),
        C.c_float(float(spec.get("kp", 0.0))),
        C.c_float(float(spec.get("kv", 0.0))),
        C.c_float(float(spec.get("max_accel", 0.0))),
    )
    return _CTargetSim(
        C.c_int(int(spec["c_id"])),
        C.c_float(float(spec["radius_m"])),
        state,
        behavior,
        controller,
    )


def _target_snapshot_from_c(
    state: _CTargetState,
    spec: dict[str, Any],
    target: _CTargetSim | None = None,
) -> dict[str, Any]:
    return {
        "c_id": int(spec.get("c_id", 0)),
        "id": str(spec.get("id", spec.get("c_id", 0))),
        "kind": str(spec.get("kind", "target")),
        "radius_m": float(spec.get("radius_m", 0.0)),
        "position_w": np.array([state.pos.x, state.pos.y, state.pos.z], dtype=float),
        "velocity_w": np.array([state.vel.x, state.vel.y, state.vel.z], dtype=float),
    }


def _camera_spec_from_python(camera: Any, index: int = 0) -> dict[str, Any]:
    intr = getattr(camera, "intrinsics", None)
    if intr is None:
        intr = camera["intrinsics"]
        width_px = intr["width_px"]
        height_px = intr["height_px"]
        fx_px = intr["fx_px"]
        fy_px = intr["fy_px"]
        cx_px = intr.get("cx_px", width_px / 2.0)
        cy_px = intr.get("cy_px", height_px / 2.0)
        hfov_rad = intr["hfov_rad"]
        vfov_rad = intr["vfov_rad"]
        position_b = camera.get("position_b", [0.0, 0.0, 0.0])
        body_to_camera = camera.get("body_to_camera", np.eye(3))
        capture_rate_hz = camera["capture_rate_hz"]
        camera_id = camera.get("id", str(index))
    else:
        width_px = intr.width_px
        height_px = intr.height_px
        fx_px = intr.fx_px
        fy_px = intr.fy_px
        cx_px = intr.cx_px
        cy_px = intr.cy_px
        hfov_rad = intr.hfov_rad
        vfov_rad = intr.vfov_rad
        position_b = camera.position_b
        body_to_camera = camera.body_to_camera
        capture_rate_hz = camera.capture_rate_hz
        camera_id = camera.id

    return {
        "c_id": int(index),
        "id": str(camera_id),
        "position_b": np.asarray(position_b, dtype=float).reshape(3).copy(),
        "body_to_camera": np.asarray(body_to_camera, dtype=float).reshape(3, 3).copy(),
        "width_px": int(width_px),
        "height_px": int(height_px),
        "fx_px": float(fx_px),
        "fy_px": float(fy_px),
        "cx_px": float(cx_px),
        "cy_px": float(cy_px),
        "hfov_rad": float(hfov_rad),
        "vfov_rad": float(vfov_rad),
        "capture_rate_hz": float(capture_rate_hz),
        "next_capture_t": 0.0,
    }


def _camera_specs_from_snapshot(
    snapshots: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    base_specs: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    specs = []
    for i, snap in enumerate(snapshots):
        base = dict(base_specs[i]) if i < len(base_specs) else {"c_id": i}
        base.update({
            "position_b": np.asarray(snap["position_b"], dtype=float).reshape(3).copy(),
            "body_to_camera": np.asarray(snap["body_to_camera"], dtype=float).reshape(3, 3).copy(),
            "width_px": int(snap["width_px"]),
            "height_px": int(snap["height_px"]),
            "fx_px": float(snap["fx_px"]),
            "fy_px": float(snap["fy_px"]),
            "cx_px": float(snap["cx_px"]),
            "cy_px": float(snap["cy_px"]),
            "hfov_rad": float(snap["hfov_rad"]),
            "vfov_rad": float(snap["vfov_rad"]),
            "capture_rate_hz": float(snap["capture_rate_hz"]),
            "next_capture_t": float(snap["next_capture_t"]),
        })
        if "id" in snap:
            base["id"] = str(snap["id"])
        specs.append(base)
    return tuple(specs)


def _camera_to_c(spec: dict[str, Any]) -> _CCameraSim:
    return _CCameraSim(
        C.c_int(int(spec["c_id"])),
        C.c_int(0),
        _vec3(spec["position_b"]),
        _mat3(spec["body_to_camera"]),
        _CCameraIntrinsics(
            C.c_int(int(spec["width_px"])),
            C.c_int(int(spec["height_px"])),
            C.c_float(float(spec["fx_px"])),
            C.c_float(float(spec["fy_px"])),
            C.c_float(float(spec["cx_px"])),
            C.c_float(float(spec["cy_px"])),
            C.c_float(float(spec["hfov_rad"])),
            C.c_float(float(spec["vfov_rad"])),
        ),
        C.c_float(float(spec["capture_rate_hz"])),
        C.c_float(float(spec.get("next_capture_t", 0.0))),
    )


def _camera_states_from_engine(
    engine: _CSimEngine,
    camera_specs: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    states = []
    for i in range(int(engine.num_cameras)):
        camera = engine.cameras[i]
        spec = camera_specs[i] if i < len(camera_specs) else {}
        body_to_camera = np.array(
            [[camera.body_to_camera.m[r][c] for c in range(3)] for r in range(3)],
            dtype=float,
        )
        states.append({
            "c_id": int(camera.id),
            "position_b": np.array([camera.position_b.x, camera.position_b.y, camera.position_b.z], dtype=float),
            "body_to_camera": body_to_camera,
            "width_px": int(camera.intrinsics.width_px),
            "height_px": int(camera.intrinsics.height_px),
            "fx_px": float(camera.intrinsics.fx_px),
            "fy_px": float(camera.intrinsics.fy_px),
            "cx_px": float(camera.intrinsics.cx_px),
            "cy_px": float(camera.intrinsics.cy_px),
            "hfov_rad": float(camera.intrinsics.hfov_rad),
            "vfov_rad": float(camera.intrinsics.vfov_rad),
            "capture_rate_hz": float(camera.capture_rate_hz),
            "next_capture_t": float(camera.next_capture_t),
            **{k: v for k, v in spec.items() if k not in {"next_capture_t"}},
        })
    return tuple(states)


def _camera_output_from_c(
    output: _CCameraOutput,
    target_specs: tuple[dict[str, Any], ...],
    camera_specs: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    obs = output.observation
    target_index = int(obs.target_index)
    target_id = None
    if 0 <= target_index < len(target_specs):
        target_id = str(target_specs[target_index].get("id", target_index))
    c_camera_id = int(obs.camera_id)
    camera_id = str(c_camera_id)
    for spec in camera_specs:
        if int(spec.get("c_id", -1)) == c_camera_id:
            camera_id = str(spec.get("id", camera_id))
            break
    return {
        "camera_id": camera_id,
        "c_camera_id": c_camera_id,
        "target_id": target_id,
        "target_index": target_index,
        "captured": bool(obs.captured),
        "detected": bool(obs.detected),
        "t_capture": float(obs.t_capture),
        "target_pos_c": np.array([obs.target_pos_c.x, obs.target_pos_c.y, obs.target_pos_c.z], dtype=float),
        "range_m": float(obs.range_m),
        "uv_norm": np.array(list(obs.uv_norm), dtype=float) if obs.detected else None,
        "uv_px": np.array(list(obs.uv_px), dtype=float) if obs.detected else None,
        "apparent_radius_px": float(obs.apparent_radius_px) if obs.detected else None,
        "has_frame": bool(output.has_frame),
        "frame_width_px": int(output.frame_width_px),
        "frame_height_px": int(output.frame_height_px),
        "frame_channels": int(output.frame_channels),
        "frame_stride_bytes": 0,
        "frame_rgb": None,
        "render_status": None,
        "render_status_name": None,
    }


def _render_config_key(config: RenderConfig) -> tuple[Any, ...]:
    return (
        bool(config.enabled),
        config.camera_id,
        config.backend,
        config.platform,
        config.scene_id,
        int(config.timeout_ms),
        bool(config.fail_on_error),
    )


def _render_drone_state_from_engine(engine: _CSimEngine) -> dict[str, Any]:
    state = _state_from_c(engine.pursuer.state)
    state["t"] = float(engine.t)
    return state


def _camera_render_state(engine: _CSimEngine, output: dict[str, Any]) -> dict[str, Any]:
    camera_index = int(output["c_camera_id"])
    camera = engine.cameras[camera_index]
    return {
        "c_id": int(camera.id),
        "position_b": np.array([camera.position_b.x, camera.position_b.y, camera.position_b.z], dtype=float),
        "body_to_camera": np.array(
            [[camera.body_to_camera.m[r][c] for c in range(3)] for r in range(3)],
            dtype=float,
        ),
        "width_px": int(camera.intrinsics.width_px),
        "height_px": int(camera.intrinsics.height_px),
        "fx_px": float(camera.intrinsics.fx_px),
        "fy_px": float(camera.intrinsics.fy_px),
        "cx_px": float(camera.intrinsics.cx_px),
        "cy_px": float(camera.intrinsics.cy_px),
        "hfov_rad": float(camera.intrinsics.hfov_rad),
        "vfov_rad": float(camera.intrinsics.vfov_rad),
    }


def _render_targets_from_engine(
    engine: _CSimEngine,
    target_specs: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    targets = []
    for i in range(int(engine.num_targets)):
        state = engine.targets[i].state
        spec = target_specs[i] if i < len(target_specs) else {}
        targets.append({
            "c_id": int(engine.targets[i].id),
            "position_w": np.array([state.pos.x, state.pos.y, state.pos.z], dtype=float),
            "velocity_w": np.array([state.vel.x, state.vel.y, state.vel.z], dtype=float),
            "radius_m": float(spec.get("radius_m", engine.targets[i].radius)),
        })
    return tuple(targets)


def _metrics_from_c(metrics: _CInterceptMetrics) -> dict[str, float | int | bool | None]:
    intercepted = bool(metrics.intercepted)
    return {
        "distance_m": float(metrics.distance_m),
        "min_distance_m": float(metrics.min_distance_m),
        "intercepted": intercepted,
        "intercept_time_s": float(metrics.intercept_time_s) if intercepted else None,
        "target_index": int(metrics.target_index),
    }


def _metrics_to_c(metrics: dict[str, Any]) -> _CInterceptMetrics:
    intercepted = bool(metrics.get("intercepted", False))
    return _CInterceptMetrics(
        C.c_float(float(metrics.get("distance_m", 0.0))),
        C.c_float(float(metrics.get("min_distance_m", float("inf")))),
        C.c_int(1 if intercepted else 0),
        C.c_float(float(metrics.get("intercept_time_s", -1.0) if intercepted else -1.0)),
        C.c_int(int(metrics.get("target_index", -1))),
    )


def _vec3(value: np.ndarray | list[float] | tuple[float, ...]) -> _CVec3:
    arr = np.asarray(value, dtype=float).reshape(3)
    return _CVec3(C.c_float(arr[0]), C.c_float(arr[1]), C.c_float(arr[2]))


def _mat3(value: np.ndarray | list[list[float]]) -> _CMat3:
    arr = np.asarray(value, dtype=float).reshape(3, 3)
    rows = ((C.c_float * 3) * 3)()
    for r in range(3):
        for c in range(3):
            rows[r][c] = C.c_float(arr[r, c])
    return _CMat3(rows)


def _copy_numeric_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=float).copy() for key, value in state.items()}


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
