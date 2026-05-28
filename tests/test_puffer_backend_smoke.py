from __future__ import annotations

import numpy as np

import pytest

from backends import (
    RenderConfig,
    SimConfig,
    PursuerInitialState,
    PursuerParams,
    BatchPufferSimEngineBackend,
    PufferDroneBackend,
    PufferSimEngineBackend,
    SimInstance,
    SimOptions,
    TargetConfig,
    TargetInitialState,
)
from control_sims.sim_runner import BatchSimEngineRunner, BatchSimEngineRunnerConfig
from backends.csim.rendering.python import LIFTOFF_RENDER_BACKEND_UNAVAILABLE, RenderError
from backends.csim.rendering.python import LIFTOFF_RENDER_OK


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
            {
                "position_w": np.array([0.25, 0.0, 0.0]),
                "velocity_w": np.zeros(3),
                "radius_m": 0.2,
            },
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
            {
                "id": "target",
                "position_w": np.array([2.0, 0.0, 0.0]),
                "velocity_w": np.zeros(3),
                "radius_m": 0.2,
            },
        ),
        cameras=(
            {
                "position_b": np.zeros(3),
                "body_to_camera": np.eye(3),
                "capture_rate_hz": 30.0,
                "intrinsics": {
                    "width_px": 640,
                    "height_px": 480,
                    "fx_px": 320.0,
                    "fy_px": 320.0,
                    "cx_px": 320.0,
                    "cy_px": 240.0,
                    "hfov_rad": np.deg2rad(90.0),
                    "vfov_rad": np.deg2rad(60.0),
                },
            },
        ),
    )

    assert len(snapshot["camera_outputs"]) == 1
    output = snapshot["camera_outputs"][0]
    assert output["detected"] is True
    assert output["target_id"] == "target"
    assert output["camera_id"] == "0"
    assert output["render_status"] is None
    np.testing.assert_allclose(output["uv_norm"], np.zeros(2), atol=1e-6)
    np.testing.assert_allclose(output["uv_px"], np.array([320.0, 240.0]), atol=1e-5)


def test_sim_engine_preserves_time_and_camera_schedule_across_snapshots():
    params = _params()
    backend = PufferSimEngineBackend(params)
    snapshot = backend.reset(
        PursuerInitialState(
            position_w=np.zeros(3),
            velocity_w=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            body_rates_b=np.zeros(3),
        ),
        targets=(
            {
                "id": "target",
                "position_w": np.array([2.0, 0.0, 0.0]),
                "velocity_w": np.zeros(3),
                "radius_m": 0.2,
            },
        ),
        cameras=(_camera("front"),),
    )

    assert snapshot["t"] == 0.0
    assert len(snapshot["camera_outputs"]) == 1
    np.testing.assert_allclose(snapshot["camera_outputs"][0]["t_capture"], 0.0, atol=1e-7)
    np.testing.assert_allclose(snapshot["camera_states"][0]["next_capture_t"], 1.0 / 30.0, atol=1e-6)

    command = {
        "thrust_n": params.mass_kg * params.gravity_mps2,
        "body_rates_b": np.zeros(3),
    }
    outputs_by_step = []
    for step in range(1, 11):
        snapshot = backend.step_ctbr(snapshot, command, 0.005)
        outputs_by_step.append((step, snapshot["t"], tuple(snapshot["camera_outputs"])))

    assert [step for step, _t, outputs in outputs_by_step if outputs] == [7]
    np.testing.assert_allclose(outputs_by_step[6][1], 0.035, atol=1e-7)
    np.testing.assert_allclose(outputs_by_step[6][2][0]["t_capture"], 0.035, atol=1e-7)
    np.testing.assert_allclose(snapshot["t"], 0.05, atol=1e-7)
    np.testing.assert_allclose(snapshot["camera_states"][0]["next_capture_t"], 2.0 / 30.0, atol=1e-6)


def test_batch_sim_engine_matches_scalar_step():
    params = _params()
    target = TargetConfig(
        id="target",
        kind="target",
        radius_m=0.2,
    )
    config = SimConfig(pursuer=params, targets=(target,), intercept_radius_m=0.1)
    initial = PursuerInitialState(
        position_w=np.zeros(3),
        velocity_w=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        body_rates_b=np.zeros(3),
    )
    target_initial = TargetInitialState(
        position_w=np.array([2.0, 0.0, 0.0]),
        velocity_w=np.zeros(3),
    )
    instance = SimInstance(seed=1, pursuer_initial=initial, target_initials=(target_initial,), config=config)
    scalar = PufferSimEngineBackend(config)
    scalar_snapshot = scalar.reset(instance)
    command = {
        "thrust_n": params.mass_kg * params.gravity_mps2,
        "body_rates_b": np.zeros(3),
    }
    scalar_snapshot = scalar.step_ctbr(scalar_snapshot, command)

    batch = BatchPufferSimEngineBackend(1)
    batch.reset_many(np.array([0]), (instance,))
    batch_snapshot = batch.step_ctbr_many(np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    np.testing.assert_allclose(batch_snapshot["pursuer"][0, 0:3], scalar_snapshot["vehicle_state"]["x"], atol=1e-5)
    np.testing.assert_allclose(batch_snapshot["pursuer"][0, 3:6], scalar_snapshot["vehicle_state"]["v"], atol=1e-5)
    np.testing.assert_allclose(batch_snapshot["metrics"][0, 0], scalar_snapshot["metrics"]["distance_m"], atol=1e-5)


def test_batch_sim_engine_accepts_physical_ctbr_commands():
    params = _params()
    target = TargetConfig(id="target", kind="target", radius_m=0.2)
    config = SimConfig(pursuer=params, targets=(target,), intercept_radius_m=0.1)
    initial = PursuerInitialState(
        position_w=np.zeros(3),
        velocity_w=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        body_rates_b=np.zeros(3),
    )
    target_initial = TargetInitialState(
        position_w=np.array([2.0, 0.0, 0.0]),
        velocity_w=np.zeros(3),
    )
    instance = SimInstance(seed=1, pursuer_initial=initial, target_initials=(target_initial,), config=config)

    scalar = PufferSimEngineBackend(config)
    scalar_snapshot = scalar.reset(instance)
    command = {
        "thrust_n": params.mass_kg * params.gravity_mps2,
        "body_rates_b": np.zeros(3),
    }
    scalar_snapshot = scalar.step_ctbr(scalar_snapshot, command)

    batch = BatchPufferSimEngineBackend(1)
    batch.reset_many(np.array([0]), (instance,))
    batch_snapshot = batch.step_ctbr_commands_many(
        np.array([params.mass_kg * params.gravity_mps2], dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
    )

    np.testing.assert_allclose(batch_snapshot["pursuer"][0, 0:3], scalar_snapshot["vehicle_state"]["x"], atol=1e-5)
    np.testing.assert_allclose(batch_snapshot["pursuer"][0, 3:6], scalar_snapshot["vehicle_state"]["v"], atol=1e-5)
    np.testing.assert_allclose(batch_snapshot["metrics"][0, 0], scalar_snapshot["metrics"]["distance_m"], atol=1e-5)


def test_batch_sim_engine_runner_refills_completed_slots():
    params = _params()
    target = TargetConfig(id="target", kind="target", radius_m=0.2)
    config = SimConfig(
        pursuer=params,
        options=SimOptions(duration_s=1.0),
        targets=(target,),
        intercept_radius_m=0.1,
    )
    initial = PursuerInitialState(
        position_w=np.zeros(3),
        velocity_w=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        body_rates_b=np.zeros(3),
    )
    target_initial = TargetInitialState(
        position_w=np.array([2.0, 0.0, 0.0]),
        velocity_w=np.zeros(3),
    )
    instances = tuple(
        SimInstance(seed=seed, pursuer_initial=initial, target_initials=(target_initial,), config=config)
        for seed in range(3)
    )

    runner = BatchSimEngineRunner(BatchSimEngineRunnerConfig(max_envs=2, max_episode_steps=1))
    state = runner.reset(instances)
    assert state.active.tolist() == [True, True]
    assert state.workload_indices.tolist() == [0, 1]

    step = runner.step({
        "thrust_n": np.full(2, params.mass_kg * params.gravity_mps2, dtype=np.float32),
        "body_rates_b": np.zeros((2, 3), dtype=np.float32),
    })
    assert [item.workload_index for item in step.completed] == [0, 1]
    assert [item.terminal_reason for item in step.completed] == ["timeout", "timeout"]
    assert step.state.active.tolist() == [True, False]
    assert step.state.workload_indices.tolist() == [2, -1]

    step = runner.step({
        "thrust_n": np.full(2, params.mass_kg * params.gravity_mps2, dtype=np.float32),
        "body_rates_b": np.zeros((2, 3), dtype=np.float32),
    })
    assert [item.workload_index for item in step.completed] == [2]
    assert step.state.active.tolist() == [False, False]


def test_sim_engine_render_config_requests_only_selected_camera():
    params = _params()
    backend = PufferSimEngineBackend(
        SimConfig(
            pursuer=params,
            rendering=True,
            render=RenderConfig(camera_id="front", backend="unity"),
        )
    )

    snapshot = backend.reset(
        PursuerInitialState(
            position_w=np.zeros(3),
            velocity_w=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            body_rates_b=np.zeros(3),
        ),
        targets=(
            {
                "id": "target",
                "position_w": np.array([2.0, 0.0, 0.0]),
                "velocity_w": np.zeros(3),
                "radius_m": 0.2,
            },
        ),
        cameras=(_camera("front"), _camera("down")),
    )

    outputs = {output["camera_id"]: output for output in snapshot["camera_outputs"]}
    assert set(outputs) == {"front", "down"}
    assert outputs["front"]["render_status"] == LIFTOFF_RENDER_BACKEND_UNAVAILABLE
    assert outputs["front"]["render_status_name"] == "backend_unavailable"
    assert outputs["front"]["has_frame"] is False
    assert outputs["front"]["frame_rgb"] is None
    assert outputs["down"]["render_status"] is None
    assert outputs["down"]["frame_rgb"] is None


def test_sim_engine_render_fail_on_error_raises():
    backend = PufferSimEngineBackend(
        SimConfig(
            pursuer=_params(),
            rendering=True,
            render=RenderConfig(camera_id="front", backend="unity", fail_on_error=True),
        )
    )

    with pytest.raises(RenderError) as exc:
        backend.reset(
            PursuerInitialState(
                position_w=np.zeros(3),
                velocity_w=np.zeros(3),
                quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                body_rates_b=np.zeros(3),
            ),
            targets=(
                {
                    "id": "target",
                    "position_w": np.array([2.0, 0.0, 0.0]),
                    "velocity_w": np.zeros(3),
                    "radius_m": 0.2,
                },
            ),
            cameras=(_camera("front"),),
        )

    assert exc.value.status == LIFTOFF_RENDER_BACKEND_UNAVAILABLE


def test_sim_engine_software_render_outputs_frame_bytes():
    backend = PufferSimEngineBackend(
        SimConfig(
            pursuer=_params(),
            rendering=True,
            render=RenderConfig(camera_id="front", backend="software"),
        )
    )

    snapshot = backend.reset(
        PursuerInitialState(
            position_w=np.zeros(3),
            velocity_w=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            body_rates_b=np.zeros(3),
        ),
        targets=(
            {
                "id": "target",
                "position_w": np.array([2.0, 0.0, 0.0]),
                "velocity_w": np.zeros(3),
                "radius_m": 0.2,
            },
        ),
        cameras=(_camera("front"),),
    )

    output = snapshot["camera_outputs"][0]
    assert output["render_status"] == LIFTOFF_RENDER_OK
    assert output["render_status_name"] == "ok"
    assert output["has_frame"] is True
    assert output["frame_width_px"] == 640
    assert output["frame_height_px"] == 480
    assert output["frame_channels"] == 3
    assert output["frame_stride_bytes"] == 640 * 3
    assert output["frame_rgb"] is not None
    assert len(output["frame_rgb"]) == 640 * 480 * 3


def _params() -> PursuerParams:
    return PursuerParams(
        mass_kg=0.027,
        ixx=3.85e-6,
        iyy=3.85e-6,
        izz=5.9675e-6,
        arm_len_m=0.0396,
        k_thrust=3.16e-10,
        k_yaw=0.005964552,
        max_rpm=21702.0,
    )


def _camera(camera_id: str) -> dict:
    return {
        "id": camera_id,
        "position_b": np.zeros(3),
        "body_to_camera": np.eye(3),
        "capture_rate_hz": 30.0,
        "intrinsics": {
            "width_px": 640,
            "height_px": 480,
            "fx_px": 320.0,
            "fy_px": 320.0,
            "cx_px": 320.0,
            "cy_px": 240.0,
            "hfov_rad": np.deg2rad(90.0),
            "vfov_rad": np.deg2rad(60.0),
        },
    }
