from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ai.rl.puffer_intercept.train import train


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO with the puffer_intercept native vecenv.")
    parser.add_argument("--scenario-table", required=True)
    parser.add_argument("--reward-source", default="ai/rl/puffer_intercept/rewards/default.c")
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
    parser.add_argument("--checkpoint-dir", default="checkpoints/puffer_intercept")
    parser.add_argument("--checkpoint-interval-steps", type=int, default=1000000)
    parser.add_argument("--save-latest-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-s3-uri", default=None)
    parser.add_argument("--s3-checkpoint-prefix", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="drone-interception")
    parser.add_argument("--wandb-group", default="puffer_intercept")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--diagnostic-sample-size", type=int, default=4096)
    parser.add_argument("--wandb-diagnostic-visuals", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    print(json.dumps({"event": "start", "args": vars(args), "cuda": torch.cuda.is_available()}, sort_keys=True), flush=True)
    train(args)


if __name__ == "__main__":
    main()
