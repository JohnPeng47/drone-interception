from __future__ import annotations

import numpy as np

from backends import InitialState, PufferDroneBackend, VehicleParams


def test_backend_hover_smoke():
    params = VehicleParams(
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
        InitialState(
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
    params = VehicleParams(
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
        InitialState(
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
