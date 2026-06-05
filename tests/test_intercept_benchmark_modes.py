from __future__ import annotations

from pathlib import Path

import numpy as np

from ai.rl.puffer_intercept.benchmark_modes import run_puffer_native_benchmark
from ai.rl.puffer_intercept.native_backend import NativeInterceptBackend


SCENARIOS = Path("scripts/generators/sim_instances/sobol_samples_512.csimin")


def test_intercept_benchmark_modes_smoke() -> None:
    native = run_puffer_native_benchmark(SCENARIOS, num_envs=2, steps=1)

    assert native.scenario_count == 512
    assert native.obs_shape == (2, 25)
    assert native.env_steps == 2
    assert native.sim_sps > 0.0


def test_puffer_native_steps_with_fixed_actions() -> None:
    native = NativeInterceptBackend(SCENARIOS, num_envs=2)
    try:
        c_obs = native.reset()
        assert c_obs.shape == (2, 25)

        actions = np.array(
            [
                [0.1, 0.2, -0.15, 0.05],
                [0.0, -0.1, 0.25, -0.2],
            ],
            dtype=np.float32,
        )
        for _ in range(3):
            c_obs, c_rewards, c_dones = native.step(actions)
            assert c_obs.shape == (2, 25)
            assert c_rewards.shape == (2,)
            assert c_dones.shape == (2,)
            assert np.all(np.isfinite(c_obs))
            assert np.all(np.isfinite(c_rewards))
    finally:
        native.close()


def test_puffer_native_terminal_refills() -> None:
    native = NativeInterceptBackend(SCENARIOS, num_envs=2, max_episode_steps=1)
    try:
        c_obs = native.reset()
        assert c_obs.shape == (2, 25)

        actions = np.array(
            [
                [0.0, 0.05, -0.05, 0.0],
                [0.1, -0.05, 0.05, 0.0],
            ],
            dtype=np.float32,
        )
        for _ in range(3):
            c_obs, c_rewards, c_dones = native.step(actions)
            assert c_obs.shape == (2, 25)
            assert c_rewards.shape == (2,)
            assert c_dones.tolist() == [True, True]
    finally:
        native.close()
