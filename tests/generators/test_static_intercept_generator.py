from __future__ import annotations

import numpy as np

from backends import PufferSimEngineBackend
from scripts.generators.static_intercept import StaticInterceptConfigGenerator, evaluate_samples


def _initial_distance(instance):
    delta = instance.target_initials[0].position_w - instance.pursuer_initial.position_w
    return float(np.linalg.norm(delta))


def _single_sample_config():
    config = StaticInterceptConfigGenerator.default_config()
    config["grid"] = None
    config["sampling"]["n_samples"] = 1
    config["sampling"]["scramble"] = False
    config["parameters"]["range_m"] = {"min": 8.0, "max": 8.0, "distribution": "uniform"}
    config["parameters"]["camera_azimuth_deg"] = {"min": 0.0, "max": 0.0, "distribution": "uniform"}
    config["parameters"]["camera_elevation_deg"] = {"min": 0.0, "max": 0.0, "distribution": "uniform_sin"}
    config["parameters"]["camera_u_fraction"] = {"min": 0.0, "max": 0.0, "distribution": "uniform"}
    config["parameters"]["camera_v_fraction"] = {"min": 0.0, "max": 0.0, "distribution": "uniform"}
    config["parameters"]["forward_speed_mps"] = {"min": 8.0, "max": 8.0, "distribution": "uniform"}
    config["parameters"]["target_speed_mps"] = {"min": 5.0, "max": 5.0, "distribution": "uniform"}
    config["parameters"]["target_azimuth_rad"] = {"min": 1.57079632679, "max": 1.57079632679, "distribution": "uniform"}
    return config


def test_static_intercept_defaults_to_1048_fixed_10m_samples():
    generator = StaticInterceptConfigGenerator()

    assert len(generator._sample_points) == 1048
    np.testing.assert_allclose(_initial_distance(generator.sample(seed=1)), 10.0, atol=1e-6)
    np.testing.assert_allclose(_initial_distance(generator.sample(seed=1048)), 10.0, atol=1e-6)


def test_static_intercept_overrides_sampled_target_velocity_to_zero():
    instance = StaticInterceptConfigGenerator(_single_sample_config()).sample(seed=1)

    np.testing.assert_allclose(instance.target_initials[0].velocity_w, np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(instance.pursuer_initial.velocity_w, np.array([8.0, 0.0, 0.0]), atol=1e-6)


def test_static_intercept_evaluation_records_static_scenario():
    [evaluation] = evaluate_samples(_single_sample_config())

    assert evaluation.record["scenario"] == "static_intercept"
    np.testing.assert_allclose(evaluation.instance.target_initials[0].velocity_w, np.zeros(3), atol=1e-9)


def test_static_intercept_target_stays_fixed_through_sim_engine():
    instance = StaticInterceptConfigGenerator(_single_sample_config()).sample(seed=1)
    backend = PufferSimEngineBackend(instance.config)
    snapshot = backend.reset(instance)
    target_position = snapshot["target_states"][0]["position_w"].copy()

    command = {
        "thrust_n": instance.config.pursuer.mass_kg * instance.config.pursuer.gravity_mps2,
        "body_rates_b": np.zeros(3),
    }
    for _ in range(5):
        snapshot = backend.step_ctbr(snapshot, command, 0.005)

    np.testing.assert_allclose(snapshot["target_states"][0]["position_w"], target_position, atol=1e-7)
    np.testing.assert_allclose(snapshot["target_states"][0]["velocity_w"], np.zeros(3), atol=1e-7)
