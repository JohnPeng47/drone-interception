from __future__ import annotations

import numpy as np

from backends import (
    CameraConfig,
    CameraIntrinsics,
    NoiseConfig,
    PregeneratedSimGenerator,
    PursuerInitialState,
    PursuerParams,
    SimConfig,
    SimInstance,
    SimOptions,
    TargetBehaviorConfig,
    TargetConfig,
    TargetControllerConfig,
    TargetState,
    read_sim_instances,
    write_sim_instances,
)


def test_sim_instance_binary_round_trip(tmp_path):
    path = tmp_path / "instances.bin"
    instance = _instance(seed=7)

    write_sim_instances(path, [instance])
    loaded = read_sim_instances(path)

    assert len(loaded) == 1
    restored = loaded[0]
    assert restored.seed == 7
    np.testing.assert_allclose(restored.pursuer_initial.position_w, instance.pursuer_initial.position_w)
    np.testing.assert_allclose(restored.pursuer_initial.quat_xyzw, instance.pursuer_initial.quat_xyzw)
    np.testing.assert_allclose(restored.targets[0].initial.position_w, instance.targets[0].initial.position_w)
    np.testing.assert_allclose(restored.targets[0].behavior.waypoints[0], instance.targets[0].behavior.waypoints[0])
    np.testing.assert_allclose(restored.cameras[0].body_to_camera, np.eye(3))
    assert restored.config is not None
    assert restored.config.options.action_substeps == 5
    assert restored.config.options.duration_s == 2.5
    assert restored.config.options.validation_dt == np.float32(0.04)
    assert restored.config.max_thrust_n == np.float32(1.2)
    assert restored.config.max_rate_rps == np.float32(3.4)
    assert restored.config.noise.pixel_noise_std_px == (1.0, 2.0)
    assert restored.config.noise.dropout_probability == 0.25
    assert restored.config.noise.rng_seed == 99
    assert restored.config.render_frames is True
    assert restored.config.render_camera_id == "front"


def test_pregenerated_generator_reads_slices_from_disk(tmp_path):
    path = tmp_path / "instances.bin"
    write_sim_instances(path, [_instance(seed=1), _instance(seed=2), _instance(seed=3)])

    assert [instance.seed for instance in PregeneratedSimGenerator.sample_many_from_disk(path, count=2, offset=1)] == [2, 3]

    generator = PregeneratedSimGenerator.from_disk(path)
    assert generator.sample(seed=2).seed == 2
    assert [instance.seed for instance in generator.sample_many(count=2, seed_start=1)] == [1, 2]


def _instance(seed: int) -> SimInstance:
    target = TargetConfig(
        id="red_balloon",
        kind="target",
        radius_m=0.2,
        initial=TargetState(
            position_w=np.array([1.0, 2.0, 3.0]),
            velocity_w=np.array([0.1, 0.2, 0.3]),
        ),
        behavior=TargetBehaviorConfig(
            waypoints=(np.array([1.0, 2.0, 3.0]),),
            duration_s=1.5,
            loop=True,
        ),
        controller=TargetControllerConfig(kp=1.0, kv=2.0, max_accel_mps2=3.0),
    )
    camera = CameraConfig(
        id="front",
        parent_id="interceptor",
        position_b=np.array([0.0, 0.0, 0.1]),
        body_to_camera=np.eye(3),
        intrinsics=CameraIntrinsics(
            width_px=640,
            height_px=480,
            fx_px=320.0,
            fy_px=321.0,
            cx_px=320.0,
            cy_px=240.0,
            hfov_rad=1.0,
            vfov_rad=0.8,
        ),
        capture_rate_hz=30.0,
    )
    config = SimConfig(
        pursuer=PursuerParams(
            mass_kg=0.027,
            ixx=3.85e-6,
            iyy=3.85e-6,
            izz=5.9675e-6,
            arm_len_m=0.0396,
            k_thrust=3.16e-10,
            k_yaw=0.0059,
            rotor_positions_b=np.zeros((4, 3)),
            rotor_directions=np.array([1.0, -1.0, 1.0, -1.0]),
        ),
        options=SimOptions(action_substeps=5, duration_s=2.5, validation_dt=0.04),
        intercept_radius_m=0.5,
        max_thrust_n=1.2,
        max_rate_rps=3.4,
        noise=NoiseConfig(
            processing_delay_s=0.08,
            pixel_noise_std_px=(1.0, 2.0),
            dropout_probability=0.25,
            sigma_img=1.0e-3,
            sigma_gyr=0.01,
            sigma_acc=0.05,
            sigma_b_gyr=1.0e-4,
            sigma_b_acc=1.0e-3,
            bias_init_std=0.005,
            rng_seed=99,
        ),
        render_frames=True,
        render_camera_id="front",
    )
    return SimInstance(
        seed=seed,
        pursuer_initial=PursuerInitialState(
            position_w=np.array([0.0, 0.0, 0.0]),
            velocity_w=np.array([0.0, 0.0, 0.0]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            body_rates_b=np.array([0.0, 0.0, 0.0]),
            rotor_speeds=np.full(4, 100.0),
            wind_w=np.zeros(3),
        ),
        targets=(target,),
        cameras=(camera,),
        config=config,
    )
