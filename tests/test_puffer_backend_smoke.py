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
    SimSnapshot,
    SimSnapshots,
    TargetConfig,
    TargetInitialState,
)
from backends.csim.runner import SimRunner
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

    assert isinstance(batch_snapshot, SimSnapshots)
    assert isinstance(batch_snapshot[0], SimSnapshot)
    np.testing.assert_allclose(batch_snapshot[0].pursuer.position_w, scalar_snapshot["vehicle_state"]["x"], atol=1e-5)
    np.testing.assert_allclose(batch_snapshot[0].pursuer.velocity_w, scalar_snapshot["vehicle_state"]["v"], atol=1e-5)
    np.testing.assert_allclose(batch_snapshot[0].metrics.distance_m, scalar_snapshot["metrics"]["distance_m"], atol=1e-5)


def test_scalar_sim_engine_reset_instance_overrides_constructor_config():
    params = _params()
    target = TargetConfig(id="target", kind="target", radius_m=0.2)
    config = SimConfig(
        pursuer=params,
        options=SimOptions(backend_dt=0.007, action_substeps=3),
        targets=(target,),
        intercept_radius_m=0.1,
    )
    wrong_params = PursuerParams(
        mass_kg=1.0,
        ixx=3.85e-6,
        iyy=3.85e-6,
        izz=5.9675e-6,
        arm_len_m=0.0396,
        k_thrust=3.16e-10,
        k_yaw=0.005964552,
        max_rpm=21702.0,
    )
    wrong_config = SimConfig(
        pursuer=wrong_params,
        options=SimOptions(backend_dt=0.02, action_substeps=5),
        intercept_radius_m=9.0,
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
    instance = SimInstance(seed=1, pursuer_initial=initial, target_initials=(target_initial,), config=config)

    scalar = PufferSimEngineBackend(wrong_config)
    scalar_snapshot = scalar.reset(instance)
    batch = BatchPufferSimEngineBackend(1)
    batch_snapshot = batch.reset_many(np.array([0]), (instance,))

    assert scalar.params == params
    assert scalar.options == config.options
    assert scalar.dt == config.options.backend_dt * config.options.action_substeps
    np.testing.assert_allclose(
        scalar_snapshot["vehicle_state"]["rotor_speeds"],
        batch_snapshot[0].pursuer.rotor_speeds,
        atol=1e-5,
    )


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

    assert isinstance(batch_snapshot, SimSnapshots)
    assert isinstance(batch_snapshot[0], SimSnapshot)
    np.testing.assert_allclose(batch_snapshot[0].thrust_n, params.mass_kg * params.gravity_mps2, atol=1e-6)
    np.testing.assert_allclose(batch_snapshot[0].body_rates_b, np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(batch_snapshot[0].pursuer.position_w, scalar_snapshot["vehicle_state"]["x"], atol=1e-5)
    np.testing.assert_allclose(batch_snapshot[0].pursuer.velocity_w, scalar_snapshot["vehicle_state"]["v"], atol=1e-5)
    np.testing.assert_allclose(batch_snapshot[0].metrics.distance_m, scalar_snapshot["metrics"]["distance_m"], atol=1e-5)


@pytest.mark.parametrize(
    ("rotor_positions_b", "rotor_directions"),
    [
        (None, np.array([-1.0, 1.0, -1.0, 1.0])),
        (
            np.array([
                [0.03, 0.03, 0.0],
                [-0.03, 0.03, 0.0],
                [-0.03, -0.03, 0.0],
                [0.03, -0.03, 0.0],
            ]),
            np.array([-1.0, 1.0, -1.0, 1.0]),
        ),
    ],
)
def test_batch_ctbr_matches_reference_motor_speed_conversion(rotor_positions_b, rotor_directions):
    base = _params()
    params = PursuerParams(
        mass_kg=base.mass_kg,
        ixx=base.ixx,
        iyy=base.iyy,
        izz=base.izz,
        arm_len_m=base.arm_len_m,
        k_thrust=base.k_thrust,
        k_yaw=base.k_yaw,
        k_ang_damp=base.k_ang_damp,
        b_drag=base.b_drag,
        gravity_mps2=base.gravity_mps2,
        max_rpm=base.max_rpm,
        max_vel_mps=base.max_vel_mps,
        max_omega_rps=base.max_omega_rps,
        motor_tau_s=base.motor_tau_s,
        k_w=base.k_w,
        rpm_min=base.rpm_min,
        rotor_positions_b=rotor_positions_b,
        rotor_directions=rotor_directions,
    )
    target = TargetConfig(id="target", kind="target", radius_m=0.2)
    config = SimConfig(
        pursuer=params,
        options=SimOptions(action_substeps=2),
        targets=(target,),
        intercept_radius_m=0.1,
        max_thrust_n=params.mass_kg * params.gravity_mps2 * 2.0,
        max_rate_rps=params.max_omega_rps,
    )
    initial = PursuerInitialState(
        position_w=np.zeros(3),
        velocity_w=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        body_rates_b=np.array([0.1, -0.2, 0.05]),
    )
    target_initial = TargetInitialState(
        position_w=np.array([2.0, 0.0, 0.0]),
        velocity_w=np.zeros(3),
    )
    instance = SimInstance(seed=1, pursuer_initial=initial, target_initials=(target_initial,), config=config)
    thrust = np.array([params.mass_kg * params.gravity_mps2 * 1.1], dtype=np.float32)
    rates = np.array([[0.4, -0.3, 0.2]], dtype=np.float32)

    ctbr_backend = BatchPufferSimEngineBackend(1)
    ctbr_backend.reset_many(np.array([0]), (instance,))
    motor_backend = BatchPufferSimEngineBackend(1)
    motor_backend.reset_many(np.array([0]), (instance,))
    reference_rpms = _reference_ctbr_to_motor_speeds(params, initial.body_rates_b, thrust[0], rates[0])

    ctbr_snapshot = ctbr_backend.step_ctbr_commands_many(thrust, rates)
    motor_snapshot = motor_backend.step_motor_speeds_many(reference_rpms.reshape(1, 4))

    np.testing.assert_allclose(ctbr_snapshot.arrays.pursuer, motor_snapshot.arrays.pursuer, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ctbr_snapshot.arrays.metrics, motor_snapshot.arrays.metrics, rtol=1e-5, atol=1e-5)


def test_batch_sim_engine_runner_refills_completed_slots():
    params = _params()
    target = TargetConfig(id="target", kind="target", radius_m=0.2)
    config = SimConfig(
        pursuer=params,
        options=SimOptions(duration_s=0.02),
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

    runner = SimRunner(max_envs=2)
    state = runner.reset(instances)
    assert state.active.tolist() == [True, True]
    assert state.workload_indices.tolist() == [0, 1]

    step = runner.step({
        "thrust_n": np.full(2, params.mass_kg * params.gravity_mps2, dtype=np.float32),
        "body_rates_b": np.zeros((2, 3), dtype=np.float32),
    })
    assert step.completed == ()
    assert step.state.active.tolist() == [True, True]

    step = runner.step({
        "thrust_n": np.full(2, params.mass_kg * params.gravity_mps2, dtype=np.float32),
        "body_rates_b": np.zeros((2, 3), dtype=np.float32),
    })
    assert [item.workload_index for item in step.completed] == [0, 1]
    assert [item.terminal_reason for item in step.completed] == ["timeout", "timeout"]
    assert step.state.active.tolist() == [True, True]
    assert step.state.workload_indices.tolist() == [0, 1]
    assert runner.state().active.tolist() == [True, False]
    assert runner.state().workload_indices.tolist() == [2, -1]

    step = runner.step({
        "thrust_n": np.full(2, params.mass_kg * params.gravity_mps2, dtype=np.float32),
        "body_rates_b": np.zeros((2, 3), dtype=np.float32),
    })
    assert step.completed == ()

    step = runner.step({
        "thrust_n": np.full(2, params.mass_kg * params.gravity_mps2, dtype=np.float32),
        "body_rates_b": np.zeros((2, 3), dtype=np.float32),
    })
    assert [item.workload_index for item in step.completed] == [2]
    assert step.state.active.tolist() == [True, False]
    assert runner.state().active.tolist() == [False, False]


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


def _reference_ctbr_to_motor_speeds(
    params: PursuerParams,
    omega: np.ndarray,
    thrust_n: float,
    body_rates_b: np.ndarray,
) -> np.ndarray:
    arm_factor = params.arm_len_m / np.sqrt(2.0)
    rotor_positions = params.rotor_positions_b
    if rotor_positions is not None and np.any(np.abs(np.asarray(rotor_positions)[:, :2]) > 1e-9):
        rotor_positions = np.asarray(rotor_positions, dtype=float).reshape(4, 3)
        rotor_directions = np.asarray(params.rotor_directions, dtype=float).reshape(4)
        allocation = np.vstack((
            np.ones(4),
            rotor_positions[:, 1],
            -rotor_positions[:, 0],
            params.k_yaw * rotor_directions,
        ))
    else:
        allocation = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-arm_factor, -arm_factor, arm_factor, arm_factor],
            [-arm_factor, arm_factor, arm_factor, -arm_factor],
            [-params.k_yaw, params.k_yaw, -params.k_yaw, params.k_yaw],
        ])
    wdot_cmd = params.k_w * (
        np.asarray(body_rates_b, dtype=float).reshape(3)
        - np.asarray(omega, dtype=float).reshape(3)
    )
    moment = np.array([params.ixx, params.iyy, params.izz], dtype=float) * wdot_cmd
    desired = np.array([max(float(thrust_n), 0.0), *moment], dtype=float)
    rotor_thrusts = np.linalg.solve(allocation, desired)
    speed_sq = rotor_thrusts / max(params.k_thrust, 1e-12)
    rpms = np.sign(speed_sq) * np.sqrt(np.abs(speed_sq))
    if params.rpm_min is None:
        hover_rpm = np.sqrt((params.mass_kg * params.gravity_mps2) / (4.0 * params.k_thrust))
        min_rpm = float(np.clip(2.0 * hover_rpm - params.max_rpm, 0.0, params.max_rpm))
    else:
        min_rpm = float(np.clip(params.rpm_min, 0.0, params.max_rpm))
    return np.clip(rpms, min_rpm, params.max_rpm).astype(np.float32)
