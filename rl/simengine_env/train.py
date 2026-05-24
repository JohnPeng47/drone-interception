from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .metrics import MetricAccumulator
from .scenario_table import ScenarioTable
from .vector_env import ParallelSimEngineVectorEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.logstd = nn.Parameter(torch.full((action_size,), -0.5))
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        return self.actor_mean(h), self.logstd.expand_as(self.actor_mean(h)), self.critic(h).squeeze(-1)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logstd, value = self(obs)
        dist = Normal(mean, logstd.exp())
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        logprob = dist.log_prob(raw_action).sum(-1) - torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        return action, logprob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action = action.clamp(-0.999, 0.999)
        raw_action = torch.atanh(action)
        mean, logstd, value = self(obs)
        dist = Normal(mean, logstd.exp())
        logprob = dist.log_prob(raw_action).sum(-1) - torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprob, entropy, value


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    table_info = ScenarioTable(args.scenario_table, manifest_path=args.manifest, max_scenarios=args.max_scenarios)
    table_info_count = table_info.count
    del table_info
    wandb_run = _init_wandb(args, scenario_count=table_info_count)

    env = ParallelSimEngineVectorEnv(
        scenario_table=args.scenario_table,
        manifest=args.manifest,
        num_workers=args.num_workers,
        envs_per_worker=args.envs_per_worker,
        seed=args.seed,
        max_scenarios=args.max_scenarios,
        max_episode_steps=args.max_episode_steps,
    )
    model = ActorCritic(env.observation_size, env.action_size, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    metrics = MetricAccumulator()
    obs_np, infos = env.reset()
    metrics.observe_infos(infos)
    global_step = 0
    start = time.time()
    try:
        while global_step < args.total_timesteps:
            rollout = _collect_rollout(model, env, obs_np, args.horizon, device, metrics)
            obs_np = rollout.pop("next_obs")
            global_step += args.horizon * env.num_envs
            _ppo_update(model, optimizer, rollout, args, device)
            if global_step == 0 or global_step % args.log_interval_steps < args.horizon * env.num_envs:
                elapsed = max(time.time() - start, 1e-6)
                summary = metrics.summary(scenario_count=table_info_count)
                summary.update({
                    "global_step": float(global_step),
                    "sps": float(global_step / elapsed),
                    "num_envs": float(env.num_envs),
                    "scenario_count": float(table_info_count),
                })
                print(json.dumps(summary, sort_keys=True), flush=True)
                if wandb_run is not None:
                    wandb_run.log(summary, step=global_step)
            if args.checkpoint_dir and global_step % args.checkpoint_interval_steps < args.horizon * env.num_envs:
                _save_checkpoint(model, Path(args.checkpoint_dir), global_step, args)
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()


def _collect_rollout(
    model: ActorCritic,
    env: ParallelSimEngineVectorEnv,
    obs_np: np.ndarray,
    horizon: int,
    device: torch.device,
    metrics: MetricAccumulator,
) -> dict[str, Any]:
    obs_buf = []
    action_buf = []
    logprob_buf = []
    reward_buf = []
    done_buf = []
    value_buf = []
    for _ in range(int(horizon)):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, logprob, value = model.act(obs)
        next_obs, rewards, dones, infos = env.step(action.cpu().numpy())
        metrics.observe_infos(infos)
        obs_buf.append(obs_np.copy())
        action_buf.append(action.cpu().numpy())
        logprob_buf.append(logprob.cpu().numpy())
        reward_buf.append(rewards.copy())
        done_buf.append(dones.astype(np.float32))
        value_buf.append(value.cpu().numpy())
        obs_np = next_obs

    with torch.no_grad():
        next_value = model(torch.as_tensor(obs_np, dtype=torch.float32, device=device))[2].cpu().numpy()
    return {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "actions": np.asarray(action_buf, dtype=np.float32),
        "logprobs": np.asarray(logprob_buf, dtype=np.float32),
        "rewards": np.asarray(reward_buf, dtype=np.float32),
        "dones": np.asarray(done_buf, dtype=np.float32),
        "values": np.asarray(value_buf, dtype=np.float32),
        "next_value": next_value.astype(np.float32),
        "next_obs": obs_np,
    }


def _ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer, rollout: dict[str, Any], args: argparse.Namespace, device: torch.device) -> None:
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = np.zeros(rewards.shape[1], dtype=np.float32)
    next_values = rollout["next_value"]
    for t in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
        lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam
        next_values = values[t]
    returns = advantages + values

    flat = {
        "obs": torch.as_tensor(rollout["obs"].reshape(-1, rollout["obs"].shape[-1]), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(rollout["actions"].reshape(-1, rollout["actions"].shape[-1]), dtype=torch.float32, device=device),
        "logprobs": torch.as_tensor(rollout["logprobs"].reshape(-1), dtype=torch.float32, device=device),
        "advantages": torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device),
        "returns": torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device),
        "values": torch.as_tensor(values.reshape(-1), dtype=torch.float32, device=device),
    }
    flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / (flat["advantages"].std() + 1e-8)
    batch_size = flat["obs"].shape[0]
    minibatch_size = min(int(args.minibatch_size), batch_size)
    for _ in range(int(args.update_epochs)):
        permutation = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            idx = permutation[start: start + minibatch_size]
            new_logprob, entropy, new_value = model.evaluate(flat["obs"][idx], flat["actions"][idx])
            logratio = new_logprob - flat["logprobs"][idx]
            ratio = logratio.exp()
            pg_loss_1 = -flat["advantages"][idx] * ratio
            pg_loss_2 = -flat["advantages"][idx] * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
            pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()
            value_loss = 0.5 * (new_value - flat["returns"][idx]).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()


def _save_checkpoint(model: nn.Module, checkpoint_dir: Path, global_step: int, args: argparse.Namespace) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "global_step": global_step, "args": vars(args)},
        checkpoint_dir / f"{global_step:012d}.pt",
    )


def _init_wandb(args: argparse.Namespace, *, scenario_count: int):
    if not args.wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        config={**vars(args), "scenario_count": scenario_count},
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PPO policy on generated SimEngine scenarios.")
    parser.add_argument("--scenario-table", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--envs-per-worker", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval-steps", type=int, default=8192)
    parser.add_argument("--checkpoint-dir", default="checkpoints/simengine")
    parser.add_argument("--checkpoint-interval-steps", type=int, default=1000000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="drone-interception")
    parser.add_argument("--wandb-group", default="simengine")
    parser.add_argument("--wandb-name", default=None)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    print(json.dumps({"event": "start", "args": vars(args), "cuda": torch.cuda.is_available()}, sort_keys=True), flush=True)
    train(args)


if __name__ == "__main__":
    main()
