from __future__ import annotations

import numpy as np

from backends import (
    CameraConfig,
    CameraIntrinsics,
    PursuerInitialState,
    PursuerParams,
    PufferDroneBackend,
    PufferSimEngineBackend,
    SimConfig,
    TargetConfig,
    TargetState,
)


def test_backend_hover_smoke():
    params = PursuerParams(
        mass_kg=0.027,
        ixx=3.85e-6,
        iyy=3.85e-6,
        izz=5.9675e-6,
        arm_len_m=0.0396,
        k_thrust=3.16e-10,
        k_yaw=0.005964552,
        max_rpm=21702.0,
    )
    backend = PufferDroneBackend(params)
    state = backend.reset(
        PursuerInitialState(
            position_w=np.zeros(3),
            velocity_w=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            body_rates_b=np.zeros(3),
        )
    )

    action = backend.ctbr_to_motor_action(
        state,
        thrust_n=params.mass_kg * params.gravity_mps2,
        body_rates_b=np.zeros(3),
    )
    next_state = backend.step_motor(state, action, 0.01)

    np.testing.assert_allclose(action, np.zeros(4), atol=1e-6)
    assert abs(float(next_state["x"][2])) < 1e-6
    np.testing.assert_allclose(np.linalg.norm(next_state["q"]), 1.0, atol=1e-9)


def test_motor_speed_and_state_clamps_are_enforced():
    params = PursuerParams(
        mass_kg=0.027,
        ixx=3.85e-6,
        iyy=3.85e-6,
        izz=5.9675e-6,
        arm_len_m=0.0396,
        k_thrust=3.16e-10,
        k_yaw=0.005964552,
        max_rpm=1000.0,
        max_vel_mps=2.0,
        max_omega_rps=3.0,
        rpm_min=0.0,
    )
    backend = PufferDroneBackend(params)
    state = backend.reset(
        PursuerInitialState(
            position_w=np.zeros(3),
            velocity_w=np.array([10.0, -10.0, 10.0]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            body_rates_b=np.array([10.0, -10.0, 10.0]),
            rotor_speeds=np.full(4, 5000.0),
        )
    )

    next_state = backend.step_motor_speeds(state, np.full(4, 5000.0), 0.01)

    assert np.max(next_state["rotor_speeds"]) <= params.max_rpm
    assert np.max(np.abs(next_state["v"])) <= params.max_vel_mps
    assert np.max(np.abs(next_state["w"])) <= params.max_omega_rps


def test_sim_engine_tracks_intercept_metrics():
    params = PursuerParams(
        mass_kg=0.027,
        ixx=3.85e-6,
        iyy=3.85e-6,
        izz=5.9675e-6,
        arm_len_m=0.0396,
        k_thrust=3.16e-10,
        k_yaw=0.005964552,
        max_rpm=21702.0,
    )
    backend = PufferSimEngineBackend(params)
    snapshot = backend.reset(
        PursuerInitialState(
            position_w=np.zeros(3),
            velocity_w=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            body_rates_b=np.zeros(3),
        ),
        targets=(
            _target(position_w=np.array([0.25, 0.0, 0.0])),
        ),
        intercept_radius_m=0.5,
    )

    assert snapshot["metrics"]["intercepted"] is True
    assert snapshot["metrics"]["intercept_time_s"] == 0.0
    np.testing.assert_allclose(snapshot["metrics"]["distance_m"], 0.25, atol=1e-6)
    np.testing.assert_allclose(snapshot["metrics"]["min_distance_m"], 0.25, atol=1e-6)
    assert snapshot["metrics"]["target_index"] == 0


def test_sim_engine_emits_camera_outputs():
    params = PursuerParams(
        mass_kg=0.027,
        ixx=3.85e-6,
        iyy=3.85e-6,
        izz=5.9675e-6,
        arm_len_m=0.0396,
        k_thrust=3.16e-10,
        k_yaw=0.005964552,
        max_rpm=21702.0,
    )
    backend = PufferSimEngineBackend(params)
    snapshot = backend.reset(
        PursuerInitialState(
            position_w=np.zeros(3),
            velocity_w=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            body_rates_b=np.zeros(3),
        ),
        targets=(
            _target(position_w=np.array([2.0, 0.0, 0.0])),
        ),
        cameras=(
            _camera(width_px=640, height_px=480, fx_px=320.0, fy_px=320.0, cx_px=320.0, cy_px=240.0),
        ),
    )

    assert len(snapshot["camera_outputs"]) == 1
    output = snapshot["camera_outputs"][0]
    assert output["detected"] is True
    assert output["target_id"] == "target"
    np.testing.assert_allclose(output["uv_norm"], np.zeros(2), atol=1e-6)
    np.testing.assert_allclose(output["uv_px"], np.array([320.0, 240.0]), atol=1e-5)


def test_sim_engine_attaches_liftoff_frame_when_enabled():
    params = PursuerParams(
        mass_kg=0.027,
        ixx=3.85e-6,
        iyy=3.85e-6,
        izz=5.9675e-6,
        arm_len_m=0.0396,
        k_thrust=3.16e-10,
        k_yaw=0.005964552,
        max_rpm=21702.0,
    )
    render_engine = _FakeRenderEngine()
    backend = PufferSimEngineBackend(
        SimConfig(
            pursuer=params,
            render_frames=True,
            render_camera_id="front",
        ),
        render_engine=render_engine,
    )

    snapshot = backend.reset(
        PursuerInitialState(
            position_w=np.zeros(3),
            velocity_w=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            body_rates_b=np.zeros(3),
        ),
        targets=(
            _target(position_w=np.array([2.0, 0.0, 0.0])),
        ),
        cameras=(
            _camera(width_px=4, height_px=3, fx_px=2.0, fy_px=2.0, cx_px=2.0, cy_px=1.5),
        ),
    )

    assert len(render_engine.requests) == 1
    assert render_engine.requests[0].camera_id == "front"
    output = snapshot["camera_outputs"][0]
    assert output["camera_id"] == "front"
    assert output["has_frame"] is True
    assert output["frame_rgb"].shape == (3, 4, 3)
    assert output["frame_rgb"].dtype == np.uint8


class _FakeRenderFrame:
    width_px = 4
    height_px = 3
    channels = 3
    rgb = np.full((3, 4, 3), 127, dtype=np.uint8)


class _FakeRenderEngine:
    def __init__(self):
        self.requests = []

    def render_frame(self, request):
        self.requests.append(request)
        return _FakeRenderFrame()


def _target(position_w: np.ndarray, velocity_w: np.ndarray | None = None) -> TargetConfig:
    return TargetConfig(
        id="target",
        kind="target",
        radius_m=0.2,
        initial=TargetState(
            position_w=np.asarray(position_w, dtype=float),
            velocity_w=np.zeros(3) if velocity_w is None else np.asarray(velocity_w, dtype=float),
        ),
    )


def _camera(
    *,
    width_px: int,
    height_px: int,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
) -> CameraConfig:
    return CameraConfig(
        id="front",
        parent_id="interceptor",
        position_b=np.zeros(3),
        body_to_camera=np.eye(3),
        intrinsics=CameraIntrinsics(
            width_px=width_px,
            height_px=height_px,
            fx_px=fx_px,
            fy_px=fy_px,
            cx_px=cx_px,
            cy_px=cy_px,
            hfov_rad=np.deg2rad(90.0),
            vfov_rad=np.deg2rad(60.0),
        ),
        capture_rate_hz=30.0,
    )
