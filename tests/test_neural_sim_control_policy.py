from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch

from ai.rl.simengine_batch.checkpointing import CHECKPOINT_TYPE, SCHEMA_VERSION
from ai.rl.simengine_batch.observations import OBS_SIZE
from ai.rl.simengine_batch.policy import NeuralNetworkSimControlPolicy
from ai.rl.simengine_batch.puffer_ppo import PufferMLPPolicy
from backends.csim.generator.instance_store import read_sim_instances
from backends.csim.runner import SimRunner, SimRunnerState


SCENARIO_TABLE = "scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin"


def test_neural_policy_loads_checkpoint_and_runs_simrunner(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "latest.pt")
    instances = read_sim_instances(SCENARIO_TABLE, count=2)
    policy = NeuralNetworkSimControlPolicy(checkpoint, device="cpu", deterministic=True)

    runner = SimRunner(max_envs=2)
    state = runner.reset(instances)
    commands = policy.command(state)

    assert commands.thrust_n.shape == (2,)
    assert commands.body_rates_b.shape == (2, 3)
    assert np.all(np.isfinite(commands.thrust_n))
    assert np.all(np.isfinite(commands.body_rates_b))
    for slot, instance in enumerate(instances):
        assert 0.0 <= commands.thrust_n[slot] <= instance.config.max_thrust_n
        assert np.all(np.abs(commands.body_rates_b[slot]) <= instance.config.max_rate_rps)

    inactive_state = SimRunnerState(
        snapshot=state.snapshot,
        active=np.array([True, False]),
        workload_indices=state.workload_indices.copy(),
        instances=(state.instances[0], None),
        elapsed_s=state.elapsed_s.copy(),
        steps=state.steps.copy(),
    )
    inactive_commands = policy.command(inactive_state)
    assert inactive_commands.thrust_n[1] == 0.0
    np.testing.assert_allclose(inactive_commands.body_rates_b[1], np.zeros(3))

    result = runner.run(instances, policy)
    assert len(result.completed) == 2
    assert all(completed.steps > 0 for completed in result.completed)
    assert policy.metadata()["global_step"] == 123


def _write_checkpoint(path):
    model = PufferMLPPolicy(OBS_SIZE, 4, hidden_size=16, num_layers=2)
    torch.save(
        {
            "checkpoint_type": CHECKPOINT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "model": model.state_dict(),
            "optimizer": {},
            "ppo_epoch": 4,
            "generator": {},
            "global_step": 123,
            "args": vars(Namespace(hidden_size=16, num_layers=2)),
            "rng": {},
        },
        path,
    )
    return path
