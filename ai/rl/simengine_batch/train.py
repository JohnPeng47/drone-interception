from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ai.rl.simengine_env.metrics import MetricAccumulator
from ai.rl.simengine_env.scenario_table import ScenarioTable
from ai.rl.simengine_env.train import _init_wandb

from .checkpointing import download_s3_uri, load_training_checkpoint, save_training_checkpoint, upload_checkpoint_to_s3
from .generator import BatchSimGenerator
from .puffer_ppo import PufferMLPPolicy, PufferPPO, PufferPPOConfig, sample_logits
from .runner import BatchRunnerConfig, BatchSimRunner


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    table = ScenarioTable(args.scenario_table, manifest_path=args.manifest, max_scenarios=args.max_scenarios)
    table_count = table.count
    wandb_run = _init_wandb(args, scenario_count=table_count)
    generator = BatchSimGenerator(
        table,
        num_envs=args.num_envs,
        seed=args.seed,
        strategy=args.scenario_strategy,
    )
    runner = BatchSimRunner(
        generator,
        config=BatchRunnerConfig(max_episode_steps=args.max_episode_steps),
    )
    model = PufferMLPPolicy(
        runner.observation_size,
        runner.action_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    ppo = PufferPPO(model, _ppo_config(args, runner.num_envs))
    global_step = 0
    resume_path = _materialize_resume_checkpoint(args)
    if resume_path is not None:
        checkpoint = load_training_checkpoint(resume_path, model=model, optimizer=ppo.optimizer, device=device)
        ppo.epoch = int(checkpoint["ppo_epoch"])
        generator.load_state_dict(checkpoint["generator"])
        global_step = int(checkpoint["global_step"])
        print(json.dumps({"event": "resumed", "checkpoint": str(resume_path), "global_step": global_step}, sort_keys=True), flush=True)
    metrics = MetricAccumulator()
    obs_np, infos = runner.reset()
    metrics.observe_infos(infos)
    start = time.time()
    start_step = global_step

    while global_step < args.total_timesteps:
        rollout, timing = _collect_rollout(model, runner, obs_np, args.horizon, device, metrics)
        obs_np = rollout.pop("next_obs")
        global_step += args.horizon * runner.num_envs
        update_start = time.perf_counter()
        losses = ppo.update(rollout, device)
        timing["update_s"] = time.perf_counter() - update_start

        if global_step == 0 or global_step % args.log_interval_steps < args.horizon * runner.num_envs:
            elapsed = max(time.time() - start, 1e-6)
            process_steps = max(global_step - start_step, 0)
            summary = metrics.summary(scenario_count=table_count)
            summary.update({
                "global_step": float(global_step),
                "sps": float(process_steps / elapsed),
                "num_envs": float(runner.num_envs),
                "scenario_count": float(table_count),
                "rollout_collect_s": float(timing["collect_s"]),
                "rollout_policy_s": float(timing["policy_s"]),
                "rollout_sim_s": float(timing["sim_s"]),
                "ppo_update_s": float(timing["update_s"]),
                "sim_sps": float((args.horizon * runner.num_envs) / max(timing["sim_s"], 1e-9)),
                **{f"loss/{key}": float(value) for key, value in losses.items()},
            })
            print(json.dumps(summary, sort_keys=True), flush=True)
            if wandb_run is not None:
                wandb_run.log(summary, step=global_step)
        if args.checkpoint_dir and global_step % args.checkpoint_interval_steps < args.horizon * runner.num_envs:
            saved = save_training_checkpoint(
                model=model,
                optimizer=ppo.optimizer,
                ppo_epoch=ppo.epoch,
                generator_state=generator.state_dict(),
                checkpoint_dir=Path(args.checkpoint_dir),
                global_step=global_step,
                args=args,
                save_latest=args.save_latest_checkpoint,
            )
            print(json.dumps({"event": "checkpoint_saved", "path": str(saved.checkpoint_path), "global_step": global_step}, sort_keys=True), flush=True)
            if args.s3_checkpoint_prefix:
                uploaded = upload_checkpoint_to_s3(saved, args.s3_checkpoint_prefix)
                print(json.dumps({"event": "checkpoint_uploaded", "global_step": global_step, "uris": uploaded}, sort_keys=True), flush=True)

    if wandb_run is not None:
        wandb_run.finish()


def _materialize_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_s3_uri:
        destination = Path(args.resume_from) if args.resume_from else Path(args.checkpoint_dir) / "resume.pt"
        path = download_s3_uri(args.resume_s3_uri, destination)
        print(json.dumps({"event": "checkpoint_downloaded", "uri": args.resume_s3_uri, "path": str(path)}, sort_keys=True), flush=True)
        return path
    if args.resume_from:
        return Path(args.resume_from)
    return None


def _collect_rollout(
    model: PufferMLPPolicy,
    runner: BatchSimRunner,
    obs_np: np.ndarray,
    horizon: int,
    device: torch.device,
    metrics: MetricAccumulator,
) -> tuple[dict[str, Any], dict[str, float]]:
    obs_buf = []
    action_buf = []
    logprob_buf = []
    reward_buf = []
    done_buf = []
    value_buf = []
    policy_s = 0.0
    sim_s = 0.0
    reward_np = np.zeros(runner.num_envs, dtype=np.float32)
    done_np = np.zeros(runner.num_envs, dtype=np.float32)
    state = model.initial_state(runner.num_envs, device)
    collect_start = time.perf_counter()
    for _ in range(int(horizon)):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        policy_start = time.perf_counter()
        with torch.no_grad():
            logits, value, state = model.forward_eval(obs, state)
            action, logprob, _ = sample_logits(logits)
        policy_s += time.perf_counter() - policy_start
        obs_buf.append(obs_np.copy())
        action_buf.append(action.cpu().numpy())
        logprob_buf.append(logprob.cpu().numpy())
        reward_buf.append(reward_np.copy())
        done_buf.append(done_np.copy())
        value_buf.append(value.cpu().numpy())

        sim_start = time.perf_counter()
        next_obs, rewards, dones, infos = runner.step(action.cpu().numpy())
        sim_s += time.perf_counter() - sim_start
        metrics.observe_infos(infos)
        obs_np = next_obs
        reward_np = rewards.astype(np.float32, copy=False)
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


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO with batched C SimEngine slots.")
    parser.add_argument("--scenario-table", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--scenario-strategy", choices=["random", "grid_balanced", "sequential_epoch"], default="grid_balanced")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=2.0)
    parser.add_argument("--vf-clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.5)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--vtrace-rho-clip", type=float, default=1.0)
    parser.add_argument("--vtrace-c-clip", type=float, default=1.0)
    parser.add_argument("--prio-alpha", type=float, default=0.8)
    parser.add_argument("--prio-beta0", type=float, default=0.2)
    parser.add_argument("--optimizer", choices=["adam", "muon"], default="adam")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval-steps", type=int, default=8192)
    parser.add_argument("--checkpoint-dir", default="checkpoints/simengine_batch")
    parser.add_argument("--checkpoint-interval-steps", type=int, default=1000000)
    parser.add_argument("--save-latest-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-s3-uri", default=None)
    parser.add_argument("--s3-checkpoint-prefix", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="drone-interception")
    parser.add_argument("--wandb-group", default="simengine-batch")
    parser.add_argument("--wandb-name", default=None)
    return parser


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


def main() -> None:
    args = make_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    print(json.dumps({"event": "start", "args": vars(args), "cuda": torch.cuda.is_available()}, sort_keys=True), flush=True)
    train(args)


if __name__ == "__main__":
    main()
