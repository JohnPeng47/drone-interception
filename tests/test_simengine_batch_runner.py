from __future__ import annotations

import numpy as np

from ai.rl.simengine_batch.generator import BatchSimGenerator
from ai.rl.simengine_batch.runner import BatchRunnerConfig, BatchSimRunner
from ai.rl.simengine_env.scenario_table import ScenarioTable
from backends import (
    PursuerInitialState,
    PursuerParams,
    SimConfig,
    SimInstance,
    SimOptions,
    TargetConfig,
    TargetInitialState,
    write_sim_instances,
)
from backends.csim.bindings.types import SimSnapshots


def test_batch_sim_runner_consumes_typed_snapshots(tmp_path):
    path = tmp_path / "samples.csimin"
    write_sim_instances(path, _instances(3))
    table = ScenarioTable(path)
    generator = BatchSimGenerator(table, num_envs=2, seed=1, strategy="sequential_epoch")
    runner = BatchSimRunner(generator, config=BatchRunnerConfig(max_episode_steps=2))

    obs, infos = runner.reset()

    assert isinstance(runner.snapshot, SimSnapshots)
    assert obs.shape == (2, runner.observation_size)
    assert len(infos) == 2

    actions = np.zeros((2, runner.action_size), dtype=np.float32)
    next_obs, rewards, dones, infos = runner.step(actions)

    assert isinstance(runner.snapshot, SimSnapshots)
    assert next_obs.shape == (2, runner.observation_size)
    assert rewards.shape == (2,)
    assert dones.shape == (2,)
    assert len(infos) == 2
    assert np.all(np.isfinite(next_obs))
    assert np.all(np.isfinite(rewards))


def test_batch_sim_runner_uses_generated_scenario_bounds(tmp_path):
    path = tmp_path / "oob.csimin"
    instance = _instance(
        seed=200,
        position_w=np.array([2.0, 0.0, 0.0]),
        target_position_w=np.array([3.0, 0.0, 0.0]),
        bounds_w=(1.0, 1.0, 1.0),
        duration_s=1.0,
    )
    write_sim_instances(path, (instance,))
    table = ScenarioTable(path)
    generator = BatchSimGenerator(table, num_envs=1, seed=1, strategy="sequential_epoch")
    runner = BatchSimRunner(generator)
    runner.reset()

    _, _, dones, infos = runner.step(np.zeros((1, runner.action_size), dtype=np.float32))

    assert dones.tolist() == [True]
    assert infos[0]["terminal_reason"] == "oob"


def _instances(count: int) -> tuple[SimInstance, ...]:
    return tuple(_instance(seed=100 + index) for index in range(count))


def _instance(
    *,
    seed: int,
    position_w: np.ndarray | None = None,
    target_position_w: np.ndarray | None = None,
    bounds_w: tuple[float, float, float] | None = (30.0, 30.0, 20.0),
    duration_s: float = 0.02,
) -> SimInstance:
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
    config = SimConfig(
        pursuer=params,
        options=SimOptions(duration_s=duration_s),
        targets=(TargetConfig(id="target", kind="target", radius_m=0.2),),
        intercept_radius_m=0.1,
        max_thrust_n=0.5,
        max_rate_rps=8.0,
        bounds_w=bounds_w,
    )
    initial = PursuerInitialState(
        position_w=np.zeros(3) if position_w is None else np.asarray(position_w, dtype=float).copy(),
        velocity_w=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        body_rates_b=np.zeros(3),
    )
    target_initial = TargetInitialState(
        position_w=(
            np.array([2.0, 0.0, 0.0])
            if target_position_w is None
            else np.asarray(target_position_w, dtype=float).copy()
        ),
        velocity_w=np.zeros(3),
    )
    return SimInstance(
        seed=seed,
        pursuer_initial=initial,
        target_initials=(target_initial,),
        config=config,
    )
