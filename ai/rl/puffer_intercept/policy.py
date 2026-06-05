from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backends.csim.runner import CtbrCommandBatch, SimControlPolicy, SimRunnerState

from .checkpointing import CHECKPOINT_TYPE, SCHEMA_VERSION
from .observations import OBS_SIZE, observation_from_batch_snapshot
from .puffer_ppo import PufferMLPPolicy


@dataclass(frozen=True)
class NeuralPolicyCheckpointInfo:
    path: str
    global_step: int | None
    ppo_epoch: int | None
    hidden_size: int
    num_layers: int


class NeuralNetworkSimControlPolicy(SimControlPolicy):
    """Run deterministic PPO policy inference over typed SimRunner snapshots."""

    action_size = 4
    observation_size = OBS_SIZE

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        deterministic: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self._model, self.checkpoint_info = _load_policy_checkpoint(self.checkpoint_path, self.device)
        self._model.eval()

    def command(self, state: SimRunnerState) -> CtbrCommandBatch:
        obs = observation_from_batch_snapshot(state.snapshot)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            dist, _value, _state = self._model.forward_eval(obs_t)
            if self.deterministic:
                actions_t = dist.loc
            else:
                actions_t = dist.sample()
        actions = actions_t.detach().cpu().numpy().astype(np.float32, copy=False)
        thrust_n, body_rates_b = _actions_to_ctbr(actions, state)
        return CtbrCommandBatch(thrust_n=thrust_n, body_rates_b=body_rates_b)

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_info.path,
            "global_step": self.checkpoint_info.global_step,
            "ppo_epoch": self.checkpoint_info.ppo_epoch,
            "hidden_size": self.checkpoint_info.hidden_size,
            "num_layers": self.checkpoint_info.num_layers,
            "deterministic": self.deterministic,
            "device": str(self.device),
        }


def _load_policy_checkpoint(path: Path, device: torch.device) -> tuple[PufferMLPPolicy, NeuralPolicyCheckpointInfo]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError(f"{path} is not a {CHECKPOINT_TYPE} checkpoint")
    if int(checkpoint.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema in {path}: {checkpoint.get('schema_version')!r}")

    args = checkpoint.get("args", {}) or {}
    hidden_size = int(args.get("hidden_size", 128))
    num_layers = int(args.get("num_layers", 4))
    model = PufferMLPPolicy(
        OBS_SIZE,
        NeuralNetworkSimControlPolicy.action_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    info = NeuralPolicyCheckpointInfo(
        path=str(path),
        global_step=None if checkpoint.get("global_step") is None else int(checkpoint["global_step"]),
        ppo_epoch=None if checkpoint.get("ppo_epoch") is None else int(checkpoint["ppo_epoch"]),
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    return model, info


def _actions_to_ctbr(actions: np.ndarray, state: SimRunnerState) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32).reshape(len(state.instances), NeuralNetworkSimControlPolicy.action_size)
    thrust_n = np.zeros(len(state.instances), dtype=np.float32)
    body_rates_b = np.zeros((len(state.instances), 3), dtype=np.float32)
    for slot, instance in enumerate(state.instances):
        if instance is None or not bool(state.active[slot]):
            continue
        if instance.config is None:
            raise ValueError("NeuralNetworkSimControlPolicy requires SimInstance.config")
        max_thrust = float(instance.config.max_thrust_n)
        if max_thrust <= 0.0:
            params = instance.config.pursuer
            max_thrust = float(params.mass_kg * params.gravity_mps2 * 2.0)
        max_rate = float(instance.config.max_rate_rps)
        if max_rate <= 0.0:
            max_rate = float(instance.config.pursuer.max_omega_rps)
        thrust_n[slot] = np.float32(np.clip((actions[slot, 0] + 1.0) * 0.5, 0.0, 1.0) * max_thrust)
        body_rates_b[slot] = np.clip(actions[slot, 1:4], -1.0, 1.0) * np.float32(max_rate)
    return thrust_n, body_rates_b
