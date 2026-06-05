from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpointing import (
    download_s3_uri,
    load_training_checkpoint,
    save_training_checkpoint,
    upload_checkpoint_to_s3,
)
from .puffer_ppo import PufferMLPPolicy, PufferPPO, PufferPPOConfig, sample_logits
from .native_backend import ACTION_SIZE, OBS_SIZE, NativeInterceptBackend, reward_source_sha256, resolve_reward_source


@dataclass
class NativeTrainingMetrics:
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    min_distances: np.ndarray
    recent_episodes: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=5000))
    terminal_count: int = 0

    @classmethod
    def create(cls, num_envs: int, obs: np.ndarray) -> "NativeTrainingMetrics":
        return cls(
            episode_returns=np.zeros(num_envs, dtype=np.float32),
            episode_lengths=np.zeros(num_envs, dtype=np.int32),
            min_distances=_distance_from_obs(obs),
        )

    def observe_step(self, obs_before_step: np.ndarray, rewards: np.ndarray, dones: np.ndarray) -> None:
        distances = _distance_from_obs(obs_before_step)
        self.min_distances = np.minimum(self.min_distances, distances)
        self.episode_returns += rewards.astype(np.float32, copy=False)
        self.episode_lengths += 1
        done_indices = np.flatnonzero(dones)
        self.terminal_count += int(len(done_indices))
        for index in done_indices:
            i = int(index)
            self.recent_episodes.append({
                "return": float(self.episode_returns[i]),
                "length": float(self.episode_lengths[i]),
                "min_distance_m": float(self.min_distances[i]),
            })
            self.episode_returns[i] = 0.0
            self.episode_lengths[i] = 0
            self.min_distances[i] = distances[i]

    def summary(self) -> dict[str, float]:
        episodes = list(self.recent_episodes)
        out = {
            "episodes": float(len(episodes)),
            "terminal_count": float(self.terminal_count),
        }
        if not episodes:
            return out
        returns = np.asarray([ep["return"] for ep in episodes], dtype=np.float32)
        lengths = np.asarray([ep["length"] for ep in episodes], dtype=np.float32)
        min_distances = np.asarray([ep["min_distance_m"] for ep in episodes], dtype=np.float32)
        out.update({
            "episode_return": float(np.mean(returns)),
            "episode_length": float(np.mean(lengths)),
            "min_distance_m": float(np.mean(min_distances)),
        })
        return out


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    reward_source = resolve_reward_source(getattr(args, "reward_source", None))
    reward_sha256 = reward_source_sha256(reward_source)
    args.reward_source = str(reward_source)
    args.reward_source_sha256 = reward_sha256
    env = NativeInterceptBackend(
        args.scenario_table,
        num_envs=args.num_envs,
        max_episode_steps=args.max_episode_steps,
        reward_source=reward_source,
    )
    wandb_run = None
    try:
        wandb_run = _init_wandb(args, scenario_count=env.scenario_count)
        model = PufferMLPPolicy(
            OBS_SIZE,
            ACTION_SIZE,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        ).to(device)
        ppo = PufferPPO(model, _ppo_config(args, args.num_envs))
        global_step = 0
        resume_path = _materialize_resume_checkpoint(args)
        if resume_path is not None:
            checkpoint = load_training_checkpoint(resume_path, model=model, optimizer=ppo.optimizer, device=device)
            ppo.epoch = int(checkpoint["ppo_epoch"])
            global_step = int(checkpoint["global_step"])
            print(json.dumps({"event": "resumed", "checkpoint": str(resume_path), "global_step": global_step}, sort_keys=True), flush=True)

        obs_np = env.reset()
        metrics = NativeTrainingMetrics.create(args.num_envs, obs_np)
        start = time.time()
        start_step = global_step

        while global_step < args.total_timesteps:
            steps_per_update = args.horizon * args.num_envs
            next_global_step = global_step + steps_per_update
            should_log = next_global_step == 0 or next_global_step % args.log_interval_steps < steps_per_update
            rollout, timing = collect_rollout(model, env, obs_np, args.horizon, device, metrics)
            obs_np = rollout.pop("next_obs")
            global_step = next_global_step
            update_start = time.perf_counter()
            losses, diagnostics = ppo.update(
                rollout,
                device,
                collect_diagnostics=should_log and args.diagnostic_sample_size > 0,
                diagnostic_sample_size=args.diagnostic_sample_size,
            )
            timing["update_s"] = time.perf_counter() - update_start

            if should_log:
                elapsed = max(time.time() - start, 1e-6)
                process_steps = max(global_step - start_step, 0)
                summary = metrics.summary()
                summary.update({
                    "global_step": float(global_step),
                    "sps": float(process_steps / elapsed),
                    "num_envs": float(args.num_envs),
                    "scenario_count": float(env.scenario_count),
                    "rollout_collect_s": float(timing["collect_s"]),
                    "rollout_policy_s": float(timing["policy_s"]),
                    "rollout_sim_s": float(timing["sim_s"]),
                    "ppo_update_s": float(timing["update_s"]),
                    "sim_sps": float(steps_per_update / max(timing["sim_s"], 1e-9)),
                    **{f"loss/{key}": float(value) for key, value in losses.items()},
                })
                summary.update(_diagnostic_scalar_summary(diagnostics))
                print(json.dumps(summary, sort_keys=True), flush=True)
                if wandb_run is not None:
                    wandb_payload = dict(summary)
                    if args.wandb_diagnostic_visuals:
                        wandb_payload.update(_wandb_diagnostic_visuals(diagnostics))
                    wandb_run.log(wandb_payload, step=global_step)

            if args.checkpoint_dir and global_step % args.checkpoint_interval_steps < args.horizon * args.num_envs:
                saved = save_training_checkpoint(
                    model=model,
                    optimizer=ppo.optimizer,
                    ppo_epoch=ppo.epoch,
                    generator_state={
                        "backend": "puffer_intercept",
                        "scenario_table": str(args.scenario_table),
                        "scenario_count": int(env.scenario_count),
                        "reward_source": str(reward_source),
                        "reward_source_sha256": reward_sha256,
                    },
                    checkpoint_dir=Path(args.checkpoint_dir),
                    global_step=global_step,
                    args=args,
                    save_latest=args.save_latest_checkpoint,
                )
                print(json.dumps({"event": "checkpoint_saved", "path": str(saved.checkpoint_path), "global_step": global_step}, sort_keys=True), flush=True)
                if args.s3_checkpoint_prefix:
                    uploaded = upload_checkpoint_to_s3(saved, args.s3_checkpoint_prefix)
                    print(json.dumps({"event": "checkpoint_uploaded", "global_step": global_step, "uris": uploaded}, sort_keys=True), flush=True)
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()


def collect_rollout(
    model: PufferMLPPolicy,
    env: NativeInterceptBackend,
    obs_np: np.ndarray,
    horizon: int,
    device: torch.device,
    metrics: NativeTrainingMetrics | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    obs_buf = []
    action_buf = []
    logprob_buf = []
    reward_buf = []
    done_buf = []
    value_buf = []
    policy_s = 0.0
    sim_s = 0.0
    reward_np = np.zeros(env.num_envs, dtype=np.float32)
    done_np = np.zeros(env.num_envs, dtype=np.float32)
    state = model.initial_state(env.num_envs, device)
    collect_start = time.perf_counter()
    for _ in range(int(horizon)):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        policy_start = time.perf_counter()
        with torch.no_grad():
            logits, value, state = model.forward_eval(obs, state)
            action, logprob, _entropy = sample_logits(logits)
        policy_s += time.perf_counter() - policy_start
        obs_buf.append(obs_np.copy())
        action_buf.append(action.cpu().numpy())
        logprob_buf.append(logprob.cpu().numpy())
        reward_buf.append(reward_np.copy())
        done_buf.append(done_np.copy())
        value_buf.append(value.cpu().numpy())

        sim_start = time.perf_counter()
        next_obs, rewards, dones = env.step(action.cpu().numpy())
        sim_s += time.perf_counter() - sim_start
        rewards = rewards.astype(np.float32, copy=True)
        dones = dones.astype(bool, copy=True)
        if metrics is not None:
            metrics.observe_step(obs_np, rewards, dones)
        obs_np = next_obs.copy()
        reward_np = rewards
        done_np = dones.astype(np.float32, copy=False)
    rollout = {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "actions": np.asarray(action_buf, dtype=np.float32),
        "logprobs": np.asarray(logprob_buf, dtype=np.float32),
        "rewards": np.asarray(reward_buf, dtype=np.float32),
        "dones": np.asarray(done_buf, dtype=np.float32),
        "values": np.asarray(value_buf, dtype=np.float32),
        "next_obs": obs_np,
    }
    return rollout, {
        "collect_s": time.perf_counter() - collect_start,
        "policy_s": policy_s,
        "sim_s": sim_s,
    }


def _ppo_config(args: argparse.Namespace, num_envs: int) -> PufferPPOConfig:
    return PufferPPOConfig(
        total_timesteps=args.total_timesteps,
        horizon=args.horizon,
        num_envs=num_envs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        anneal_lr=args.anneal_lr,
        min_lr_ratio=args.min_lr_ratio,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        replay_ratio=args.replay_ratio,
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        vf_clip_coef=args.vf_clip_coef,
        max_grad_norm=args.max_grad_norm,
        ent_coef=args.ent_coef,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        vtrace_rho_clip=args.vtrace_rho_clip,
        vtrace_c_clip=args.vtrace_c_clip,
        prio_alpha=args.prio_alpha,
        prio_beta0=args.prio_beta0,
        optimizer=args.optimizer,
    )


def _materialize_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_s3_uri:
        destination = Path(args.resume_from) if args.resume_from else Path(args.checkpoint_dir) / "resume.pt"
        path = download_s3_uri(args.resume_s3_uri, destination)
        print(json.dumps({"event": "checkpoint_downloaded", "uri": args.resume_s3_uri, "path": str(path)}, sort_keys=True), flush=True)
        return path
    if args.resume_from:
        return Path(args.resume_from)
    return None


def _distance_from_obs(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    return np.linalg.norm(obs[:, 19:22] - obs[:, 0:3], axis=1).astype(np.float32, copy=False)


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


def _diagnostic_scalar_summary(diagnostics: dict[str, np.ndarray]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key, values in diagnostics.items():
        if key == "value_return":
            pairs = np.asarray(values, dtype=np.float32).reshape(-1, 2)
            if pairs.size == 0:
                continue
            predictions = pairs[:, 0]
            returns = pairs[:, 1]
            summary.update(_array_stats("diagnostics/value_prediction", predictions))
            summary.update(_array_stats("diagnostics/return_target", returns))
            summary["diagnostics/value_return_mae"] = float(np.mean(np.abs(predictions - returns)))
            if len(pairs) > 1 and np.std(predictions) > 0.0 and np.std(returns) > 0.0:
                summary["diagnostics/value_return_corr"] = float(np.corrcoef(predictions, returns)[0, 1])
            else:
                summary["diagnostics/value_return_corr"] = float("nan")
            continue
        summary.update(_array_stats(f"diagnostics/{key}", values))
    return summary


def _array_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    quantiles = np.quantile(arr, [0.01, 0.5, 0.99])
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_p01": float(quantiles[0]),
        f"{prefix}_p50": float(quantiles[1]),
        f"{prefix}_p99": float(quantiles[2]),
        f"{prefix}_max": float(np.max(arr)),
    }


def _wandb_diagnostic_visuals(diagnostics: dict[str, np.ndarray]) -> dict[str, object]:
    if not diagnostics:
        return {}
    import wandb

    visuals: dict[str, object] = {}
    for key in ("advantage", "td_error", "policy_ratio", "action_entropy"):
        values = np.asarray(diagnostics.get(key, ()), dtype=np.float32).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            visuals[f"diagnostics/{key}_hist"] = wandb.Histogram(values)
    pairs = np.asarray(diagnostics.get("value_return", ()), dtype=np.float32).reshape(-1, 2)
    if pairs.size:
        finite = np.all(np.isfinite(pairs), axis=1)
        pairs = pairs[finite]
        if len(pairs):
            table = wandb.Table(
                data=pairs.tolist(),
                columns=["value_prediction", "return_target"],
            )
            visuals["diagnostics/value_vs_return"] = wandb.plot.scatter(
                table,
                "value_prediction",
                "return_target",
                title="Value prediction vs return target",
            )
    return visuals
