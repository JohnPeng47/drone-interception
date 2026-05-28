from __future__ import annotations

import numpy as np

from backends.csim.bindings.types import SimConfig
from backends.csim.generator.generator import SimInstanceGenerator, get_config
from scripts.generators.robust_intercept import RobustInterceptConfigGenerator, evaluate_samples


def test_base_config_resolver_returns_typed_sim_config():
    config = get_config("base")

    assert isinstance(config, SimConfig)
    assert config.targets[0].radius_m == 0.2
    assert config.options.duration_s == 3.0
    assert len(config.cameras) == 1
    assert SimInstanceGenerator.get_config("base").targets[0].id == "target"


def test_camera_bearing_offset_drives_current_path_lateral_miss():
    config = RobustInterceptConfigGenerator.default_config()
    config["grid"] = None
    config["sampling"]["n_samples"] = 1
    config["sampling"]["scramble"] = False
    config["parameters"]["range_m"] = {"min": 8.0, "max": 8.0, "distribution": "uniform"}
    config["parameters"]["camera_azimuth_deg"] = {"min": 0.0, "max": 0.0, "distribution": "uniform"}
    config["parameters"]["camera_elevation_deg"] = {"min": 0.0, "max": 0.0, "distribution": "uniform_sin"}
    config["parameters"]["camera_u_fraction"] = {"min": 0.9, "max": 0.9, "distribution": "uniform"}
    config["parameters"]["camera_v_fraction"] = {"min": 0.0, "max": 0.0, "distribution": "uniform"}
    config["parameters"]["forward_speed_mps"] = {"min": 8.0, "max": 8.0, "distribution": "uniform"}

    instance = RobustInterceptConfigGenerator(config).sample(seed=1)

    target_position = instance.target_initials[0].position_w
    pursuer_position = instance.pursuer_initial.position_w
    los_w = (target_position - pursuer_position) / np.linalg.norm(target_position - pursuer_position)
    relative_velocity_w = instance.pursuer_initial.velocity_w - instance.target_initials[0].velocity_w

    np.testing.assert_allclose(relative_velocity_w, np.array([8.0, 0.0, 0.0]), atol=1e-6)
    assert float(np.linalg.norm(np.cross(los_w, relative_velocity_w))) > 1.0


def test_straight_current_path_capture_is_labeled():
    config = RobustInterceptConfigGenerator.default_config()
    config["grid"] = None
    config["sampling"]["n_samples"] = 1
    config["sampling"]["scramble"] = False
    config["parameters"]["range_m"] = {"min": 5.0, "max": 5.0, "distribution": "uniform"}
    config["parameters"]["camera_azimuth_deg"] = {"min": 0.0, "max": 0.0, "distribution": "uniform"}
    config["parameters"]["camera_elevation_deg"] = {"min": 0.0, "max": 0.0, "distribution": "uniform_sin"}
    config["parameters"]["camera_u_fraction"] = {"min": 0.0, "max": 0.0, "distribution": "uniform"}
    config["parameters"]["camera_v_fraction"] = {"min": 0.0, "max": 0.0, "distribution": "uniform"}
    config["parameters"]["forward_speed_mps"] = {"min": 8.0, "max": 8.0, "distribution": "uniform"}

    [evaluation] = evaluate_samples(config)

    assert evaluation.labels["straight_path_capture"] is True
    assert "straight-line current path captures target" in evaluation.label_details["straight_path_capture"]
