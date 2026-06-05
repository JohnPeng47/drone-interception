from __future__ import annotations

import numpy as np
import torch

from ai.rl.puffer_intercept.puffer_ppo import PufferMLPPolicy, PufferPPO, PufferPPOConfig, sample_logits


def test_puffer_ppo_returns_bounded_diagnostics() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    horizon = 4
    num_envs = 2
    obs_size = 3
    action_size = 2
    model = PufferMLPPolicy(obs_size, action_size, hidden_size=8, num_layers=2)
    ppo = PufferPPO(
        model,
        PufferPPOConfig(
            total_timesteps=horizon * num_envs,
            horizon=horizon,
            num_envs=num_envs,
            minibatch_size=horizon,
            replay_ratio=1.0,
        ),
    )
    obs = np.random.normal(size=(horizon, num_envs, obs_size)).astype(np.float32)
    obs_t = torch.as_tensor(obs.reshape(horizon * num_envs, obs_size))
    with torch.no_grad():
        dist, values, _state = model.forward_eval(obs_t)
        actions, logprobs, _entropy = sample_logits(dist)
    rollout = {
        "obs": obs,
        "actions": actions.numpy().reshape(horizon, num_envs, action_size),
        "logprobs": logprobs.numpy().reshape(horizon, num_envs),
        "rewards": np.random.normal(size=(horizon, num_envs)).astype(np.float32),
        "dones": np.zeros((horizon, num_envs), dtype=np.float32),
        "values": values.numpy().reshape(horizon, num_envs),
    }

    losses, diagnostics = ppo.update(
        rollout,
        torch.device("cpu"),
        collect_diagnostics=True,
        diagnostic_sample_size=5,
    )

    assert "grad_norm" in losses
    assert set(diagnostics) == {
        "action_entropy",
        "advantage",
        "policy_ratio",
        "td_error",
        "value_return",
    }
    for key, values in diagnostics.items():
        assert 0 < len(values) <= 5
        assert np.all(np.isfinite(values))
        if key == "value_return":
            assert values.shape[1] == 2


def test_puffer_ppo_skips_diagnostics_when_disabled() -> None:
    torch.manual_seed(0)
    horizon = 4
    num_envs = 1
    obs_size = 3
    action_size = 2
    model = PufferMLPPolicy(obs_size, action_size, hidden_size=8, num_layers=1)
    ppo = PufferPPO(
        model,
        PufferPPOConfig(
            total_timesteps=horizon * num_envs,
            horizon=horizon,
            num_envs=num_envs,
            minibatch_size=horizon,
            replay_ratio=1.0,
        ),
    )
    obs = np.zeros((horizon, num_envs, obs_size), dtype=np.float32)
    with torch.no_grad():
        dist, values, _state = model.forward_eval(torch.as_tensor(obs.reshape(horizon * num_envs, obs_size)))
        actions, logprobs, _entropy = sample_logits(dist)
    rollout = {
        "obs": obs,
        "actions": actions.numpy().reshape(horizon, num_envs, action_size),
        "logprobs": logprobs.numpy().reshape(horizon, num_envs),
        "rewards": np.zeros((horizon, num_envs), dtype=np.float32),
        "dones": np.zeros((horizon, num_envs), dtype=np.float32),
        "values": values.numpy().reshape(horizon, num_envs),
    }

    _losses, diagnostics = ppo.update(rollout, torch.device("cpu"))

    assert diagnostics == {}
