from __future__ import annotations

import json

import numpy as np

from backends import (
    CameraConfig,
    CameraIntrinsics,
    NoiseConfig,
    PursuerInitialState,
    PursuerParams,
    RenderConfig,
    SimConfig,
    SimGenerator,
    SimInstance,
    SimOptions,
    TargetBehaviorConfig,
    TargetConfig,
    TargetControllerConfig,
    TargetInitialState,
    read_sim_instances,
    write_sim_instances,
)
from backends.csim.generator.instance_store import read_sim_instances_by_index
from backends.csim.generator.metadata import write_sample_metadata


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
    np.testing.assert_allclose(restored.target_initials[0].position_w, instance.target_initials[0].position_w)
    assert restored.config is not None
    assert instance.config is not None
    np.testing.assert_allclose(restored.config.targets[0].behavior.waypoints[0], instance.config.targets[0].behavior.waypoints[0])
    np.testing.assert_allclose(restored.config.cameras[0].body_to_camera, np.eye(3))
    assert restored.config.options.action_substeps == 5
    assert restored.config.options.duration_s == 2.5
    assert restored.config.options.validation_dt == np.float32(0.04)
    assert restored.config.max_thrust_n == np.float32(1.2)
    assert restored.config.max_rate_rps == np.float32(3.4)
    assert restored.config.pursuer.max_omega_rps == np.float32(3.4)
    assert restored.config.bounds_w == (30.0, 31.0, 32.0)
    assert restored.config.noise.pixel_noise_std_px == (1.0, 2.0)
    assert restored.config.noise.dropout_probability == 0.25
    assert restored.config.noise.rng_seed == 99
    assert restored.config.rendering is True
    assert restored.config.render.camera_id == "front"
    assert restored.config.render.backend == "none"
    assert restored.config.render.platform == "linux"
    assert restored.config.render.scene_id == "liftoff_test"
    assert restored.config.render.timeout_ms == 7
    assert restored.config.render.fail_on_error is True


def test_sim_generator_reads_slices_from_disk(tmp_path):
    path = tmp_path / "instances.bin"
    write_sim_instances(path, [_instance(seed=1), _instance(seed=2), _instance(seed=3)])

    assert [instance.seed for instance in SimGenerator.sample_many_from_disk(path, count=2, offset=1)] == [2, 3]

    generator = SimGenerator.from_disk(path)
    assert generator.sample(seed=2).seed == 2
    assert [instance.seed for instance in generator.sample_many(count=2, seed_start=1)] == [1, 2]


def test_read_sim_instances_supports_bounded_reads(tmp_path):
    path = tmp_path / "instances.bin"
    write_sim_instances(path, [_instance(seed=1), _instance(seed=2), _instance(seed=3), _instance(seed=4)])

    assert [instance.seed for instance in read_sim_instances(path, count=2)] == [1, 2]
    assert [instance.seed for instance in read_sim_instances(path, count=2, offset=2)] == [3, 4]
    assert read_sim_instances(path, count=2, offset=10) == []


def test_read_sim_instances_by_index_returns_selected_records(tmp_path):
    path = tmp_path / "instances.bin"
    write_sim_instances(path, [_instance(seed=1), _instance(seed=2), _instance(seed=3), _instance(seed=4)])

    selected, total_count = read_sim_instances_by_index(path, (3, 1, 3))

    assert total_count == 4
    assert sorted(selected) == [1, 3]
    assert selected[1].seed == 2
    assert selected[3].seed == 4


def test_sample_metadata_sidecar_records_table_summary(tmp_path):
    path = tmp_path / "sobol_samples.csimin"
    write_sim_instances(path, [_instance(seed=1), _instance(seed=2)])

    metadata_path = write_sample_metadata(
        path,
        generator="TestGenerator",
        strategy="sobol",
        config={
            "sampling": {
                "n_samples": 3,
                "seed": 7,
                "scramble": True,
                "active_parameters": ["range_m"],
            },
            "sim": {"backend": "puffer_c", "duration_s": 3.0, "dt": 0.005},
            "parameters": {"range_m": {"min": 5.0, "max": 20.0, "distribution": "uniform"}},
        },
        total_samples=3,
        written_samples=2,
        invalid_samples=1,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_path == tmp_path / "sobol_samples.json"
    assert metadata["samples"]["path"] == str(path)
    assert metadata["samples"]["count"] == 2
    assert metadata["samples"]["file_size_bytes"] == path.stat().st_size
    assert metadata["samples"]["format_magic"] == "CSIMINST"
    assert metadata["generator"] == {"name": "TestGenerator", "strategy": "sobol"}
    assert metadata["sampling"]["requested_samples"] == 3
    assert metadata["sampling"]["total_samples"] == 3
    assert metadata["sampling"]["written_samples"] == 2
    assert metadata["sampling"]["invalid_samples"] == 1
    assert metadata["sim"]["dt"] == 0.005


def _instance(seed: int) -> SimInstance:
    target = TargetConfig(
        id="red_balloon",
        kind="target",
        radius_m=0.2,
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
            max_omega_rps=3.4,
            rotor_positions_b=np.zeros((4, 3)),
            rotor_directions=np.array([1.0, -1.0, 1.0, -1.0]),
        ),
        options=SimOptions(action_substeps=5, duration_s=2.5, validation_dt=0.04),
        targets=(target,),
        cameras=(camera,),
        intercept_radius_m=0.5,
        max_thrust_n=1.2,
        max_rate_rps=3.4,
        bounds_w=(30.0, 31.0, 32.0),
        noise=NoiseConfig(
            camera_image_delay_s=0.08,
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
        rendering=True,
        render=RenderConfig(
            camera_id="front",
            backend="none",
            platform="linux",
            scene_id="liftoff_test",
            timeout_ms=7,
            fail_on_error=True,
        ),
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
        target_initials=(
            TargetInitialState(
                position_w=np.array([1.0, 2.0, 3.0]),
                velocity_w=np.array([0.1, 0.2, 0.3]),
            ),
        ),
        config=config,
    )
