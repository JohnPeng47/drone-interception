from __future__ import annotations

from pathlib import Path

import numpy as np

from ai.rl.simengine_batch.benchmark_modes import (
    _load_native_backend_class,
    run_puffer_native_benchmark,
    run_simengine_batch_benchmark,
)
from ai.rl.simengine_batch.generator import BatchSimGenerator
from ai.rl.simengine_batch.runner import BatchRunnerConfig, BatchSimRunner
from ai.rl.simengine_env.scenario_table import ScenarioTable


SCENARIOS = Path("scripts/generators/sim_instances/sobol_samples_512.csimin")


def test_intercept_benchmark_modes_smoke() -> None:
    simengine = run_simengine_batch_benchmark(SCENARIOS, num_envs=2, steps=1)
    native = run_puffer_native_benchmark(SCENARIOS, num_envs=2, steps=1)

    assert simengine.scenario_count == 512
    assert native.scenario_count == 512
    assert simengine.obs_shape == (2, 26)
    assert native.obs_shape == (2, 26)
    assert simengine.env_steps == 2
    assert native.env_steps == 2
    assert simengine.sim_sps > 0.0
    assert native.sim_sps > 0.0


def test_puffer_native_matches_simengine_batch_for_same_actions() -> None:
    table = ScenarioTable(SCENARIOS, max_scenarios=4)
    generator = BatchSimGenerator(table, num_envs=2, strategy="sequential_epoch")
    runner = BatchSimRunner(generator)
    native = _load_native_backend_class()(SCENARIOS, num_envs=2)
    try:
        py_obs, _ = runner.reset()
        c_obs = native.reset()
        np.testing.assert_allclose(c_obs, py_obs, rtol=1e-5, atol=1e-5)

        actions = np.array(
            [
                [0.1, 0.2, -0.15, 0.05],
                [0.0, -0.1, 0.25, -0.2],
            ],
            dtype=np.float32,
        )
        for _ in range(3):
            py_obs, py_rewards, py_dones, _ = runner.step(actions)
            c_obs, c_rewards, c_dones = native.step(actions)
            np.testing.assert_allclose(c_obs, py_obs, rtol=2e-4, atol=2e-4)
            np.testing.assert_allclose(c_rewards, py_rewards, rtol=2e-4, atol=2e-4)
            np.testing.assert_array_equal(c_dones, py_dones)
    finally:
        native.close()


def test_puffer_native_matches_simengine_batch_terminal_refills() -> None:
    table = ScenarioTable(SCENARIOS)
    generator = BatchSimGenerator(table, num_envs=2, strategy="sequential_epoch")
    runner = BatchSimRunner(generator, config=BatchRunnerConfig(max_episode_steps=1))
    native = _load_native_backend_class()(SCENARIOS, num_envs=2, max_episode_steps=1)
    try:
        py_obs, _ = runner.reset()
        c_obs = native.reset()
        np.testing.assert_allclose(c_obs, py_obs, rtol=1e-5, atol=1e-5)

        actions = np.array(
            [
                [0.0, 0.05, -0.05, 0.0],
                [0.1, -0.05, 0.05, 0.0],
            ],
            dtype=np.float32,
        )
        for _ in range(3):
            py_obs, py_rewards, py_dones, _ = runner.step(actions)
            c_obs, c_rewards, c_dones = native.step(actions)
            np.testing.assert_allclose(c_rewards, py_rewards, rtol=2e-4, atol=2e-4)
            np.testing.assert_array_equal(c_dones, py_dones)
            np.testing.assert_allclose(c_obs, py_obs, rtol=2e-4, atol=2e-4)
    finally:
        native.close()
