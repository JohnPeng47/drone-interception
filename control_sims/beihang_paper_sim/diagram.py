"""build_diagram_from_config — flat top-level Drake diagram for paper_sim.

Rates (paper Fig. 3, Fig. 6, §V-B):
    inner_dt = 1/200 s — multicopter physics + IMU + logger
    outer_dt = 1/50  s — DKF + (effective) collinear/attitude controller
    camera   = 1/30  s — image capture (governed by YAML capture_rate_hz)

The controller is compute-on-demand and its only state-bearing input is the
DKF's `observer_state`, so it naturally inherits the 50 Hz outer-loop rate.
A separate 200 Hz inner attitude loop would require a gyro-integrated
high-rate attitude estimator and is deferred.

Swaps in three paper-faithful blocks vs codex_sim:
    sensing/imu_system.py       paper Eqs. (7)–(10), Gaussian noise + Wiener biases
    estimation/dkf_observer.py  paper Algorithm 2, Eqs. (30)–(36), real EKF
    controller/control_core.py  paper Eqs. (12)–(28), 2-step Lyapunov

INITIAL PITCH OFFSET — important for catch rate.
At t=0 the controller's desired thrust direction n_fd points up-and-forward
(it needs both gravity-cancellation and horizontal acceleration toward the
target). Starting level (body-z = world-z) means the drone has zero
horizontal thrust until the attitude loop rotates it to the desired pitch
(~37° for the red_balloon scenario), which takes ~80 ms at ω_max=8 rad/s.
During that transient the drone falls behind the velocity-tracking schedule
v_rd = -k_1·p_r, z_2 grows, and the controller never recovers — miss
plateaus at ~1.3 m.

Pre-pitching the drone by ~20° at t=0 cuts the attitude-loop transient
roughly in half AND provides g·tan(20°) ≈ 3.6 m/s² of immediate horizontal
acceleration. Empirically this is the single biggest knob for catch rate at
slow initial closing speeds. Paper §V-B-4 acknowledges the operational
analogue: "the interceptor multicopter initially ascends to 3 m **before
starting its mission, adjusting its motion direction** early in the
interception" — i.e. the real drone enters the IBVS regime pre-pitched, not
from level hover.
"""

from __future__ import annotations

import math

import numpy as np
from pydrake.systems.framework import Diagram, DiagramBuilder

from intercept_sim.experiments.config import ExperimentConfig
from intercept_sim.experiments.runner import (
    _camera_from_config,
    _initial_rotorpy_state,
    _perception_from_config,
    _target_from_config,
)
from intercept_sim.sensors import GeometryCamera

from .actuator.actuator_diagram import add_actuator
from .controller.controller_diagram import add_controller
from .drake_compat import RunnerStepLogger, resolve_quad_params
from .estimation.estimation_diagram import add_estimation
from .noise_config import NoiseConfig
from .sensing.sensing_diagram import add_sensing
from .world.world_diagram import add_world


INNER_RATE_HZ = 200.0
# DKF runs at IMU rate per paper Algorithm 2 (it recurses through every
# stored IMU sample inside the delay window). Running DKF slower than the
# IMU drops samples on the floor and breaks the per-IMU-tick replay path
# that the paper depends on.
OUTER_RATE_HZ = 200.0

# Default forward pitch (nose-down rotation about body-y) composed onto the
# los-pointing quaternion that build_red_balloon_config produces. Skip the
# attitude transient by starting on the steady-state-ish pitch the
# controller would otherwise demand at t=0. Override per-scenario by setting
# `vehicle.initial_pitch_offset_deg` in the raw config.
INITIAL_PITCH_OFFSET_DEG = 20.0


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, xyzw convention (matches rotorpy / our other code)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def _apply_initial_pitch_offset(raw: dict) -> None:
    """Right-multiply the current initial_quat_xyzw by a pitch-about-body-y.

    The scenario builder (build_red_balloon_config) sets initial_quat_xyzw to
    a los-pointing quat — body-x along the LOS to the target. We compose a
    rotation about the *body-y* axis so the drone tilts its nose down while
    keeping its forward direction in the LOS plane. This is a right-multiply
    in Hamilton xyzw convention.
    """
    vehicle = raw.setdefault("vehicle", {})
    pitch_deg = float(vehicle.get("initial_pitch_offset_deg", INITIAL_PITCH_OFFSET_DEG))
    if abs(pitch_deg) < 1e-9:
        return
    q_los = np.asarray(vehicle.get("initial_quat_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float)
    theta = math.radians(pitch_deg)
    q_pitch = np.array([0.0, math.sin(theta / 2.0), 0.0, math.cos(theta / 2.0)])
    q_new = _quat_mul(q_los, q_pitch)
    q_new = q_new / max(float(np.linalg.norm(q_new)), 1e-12)
    vehicle["initial_quat_xyzw"] = q_new.tolist()


def build_diagram_from_config(
    config: ExperimentConfig,
    controller_gains: dict | None = None,
    noise_config: NoiseConfig | None = None,
) -> tuple[Diagram, RunnerStepLogger]:
    from rotorpy.vehicles.multirotor import Multirotor

    raw = config.raw
    # Pre-pitch the drone before _initial_rotorpy_state reads the quat —
    # see INITIAL_PITCH_OFFSET_DEG docstring above for why this matters.
    _apply_initial_pitch_offset(raw)
    quad_params = resolve_quad_params(raw["vehicle"])
    initial_state = _initial_rotorpy_state(raw["vehicle"], quad_params)
    vehicle = Multirotor(
        quad_params,
        initial_state=initial_state,
        control_abstraction="cmd_ctbr",
        aero=bool(raw["vehicle"].get("aero", False)),
        integrator_kwargs=raw["vehicle"].get(
            "integrator_kwargs", {"method": "RK45", "rtol": 1e-6, "atol": 1e-9}
        ),
    )
    target = _target_from_config(raw["target"])
    camera_rig = _camera_from_config(raw["camera"])
    perception = _perception_from_config(raw["perception"])
    perception.pixel_to_norm = np.array([
        1.0 / float(raw["camera"]["fx_px"]),
        1.0 / float(raw["camera"]["fy_px"]),
    ])
    geometry_camera = GeometryCamera(camera_rig)
    mass_kg = float(quad_params["mass"])

    inner_dt = 1.0 / INNER_RATE_HZ
    outer_dt = 1.0 / OUTER_RATE_HZ
    nc = noise_config or NoiseConfig()
    builder = DiagramBuilder()

    backend = str(raw.get("sim", {}).get("backend", "rotorpy"))
    world = add_world(
        builder,
        vehicle=vehicle, initial_state=initial_state, dt=inner_dt,
        target=target, camera_rig=camera_rig,
        backend=backend, quad_params=quad_params,
    )
    sensing = add_sensing(
        builder, camera=geometry_camera, perception=perception, dt=inner_dt,
        noise_config=nc,
    )
    estimation = add_estimation(
        builder, camera_rig=camera_rig, dt=outer_dt, noise_config=nc,
    )
    controller_group = add_controller(
        builder, mass_kg=mass_kg, dt=outer_dt,
        camera_rig=camera_rig, gains=controller_gains,
    )
    actuator = add_actuator(builder)
    logger = builder.AddSystem(RunnerStepLogger(dt=inner_dt))

    # ----- world → sensing
    builder.Connect(world["scene"].GetOutputPort("scene"),
                    sensing["camera"].GetInputPort("scene"))
    builder.Connect(world["plant"].GetOutputPort("vehicle_state_dict"),
                    sensing["imu"].GetInputPort("vehicle_state_dict"))

    # ----- world/sensing → estimation
    builder.Connect(world["scene"].GetOutputPort("scene"),
                    estimation["core"].GetInputPort("scene"))
    builder.Connect(sensing["perception"].GetOutputPort("measurements"),
                    estimation["core"].GetInputPort("measurements"))
    builder.Connect(world["plant"].GetOutputPort("vehicle_state_dict"),
                    estimation["core"].GetInputPort("vehicle_state_dict"))
    builder.Connect(sensing["imu"].GetOutputPort("gyro_b"),
                    estimation["core"].GetInputPort("gyro"))
    builder.Connect(sensing["imu"].GetOutputPort("accel_b"),
                    estimation["core"].GetInputPort("accel"))

    # ----- estimation → controller → actuator → world
    builder.Connect(estimation["core"].GetOutputPort("observer_state"),
                    controller_group["core"].GetInputPort("observer_state"))
    builder.Connect(controller_group["core"].GetOutputPort("ctbr_cmd"),
                    actuator["pixhawk"].GetInputPort("ctbr_cmd"))
    builder.Connect(actuator["pixhawk"].GetOutputPort("rate_cmd"),
                    world["plant"].GetInputPort("ctbr_cmd"))

    # ----- logger
    builder.Connect(world["plant"].GetOutputPort("vehicle_state_dict"),
                    logger.GetInputPort("vehicle_state_dict"))
    builder.Connect(world["scene"].GetOutputPort("scene"),
                    logger.GetInputPort("scene"))
    builder.Connect(sensing["camera"].GetOutputPort("capture"),
                    logger.GetInputPort("capture"))
    builder.Connect(sensing["perception"].GetOutputPort("measurements"),
                    logger.GetInputPort("measurements"))
    builder.Connect(estimation["core"].GetOutputPort("observer_state"),
                    logger.GetInputPort("observer_state"))
    builder.Connect(controller_group["core"].GetOutputPort("ctbr_cmd"),
                    logger.GetInputPort("ctbr_cmd"))

    diagram = builder.Build()
    return diagram, logger
