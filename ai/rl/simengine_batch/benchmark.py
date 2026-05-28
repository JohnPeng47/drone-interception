from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ai.rl.simengine_env.scenario_table import ScenarioTable

from .generator import BatchSimGenerator
from .runner import BatchRunnerConfig, BatchSimRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure batched SimEngine step throughput.")
    parser.add_argument("--scenario-table", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    args = parser.parse_args()

    table = ScenarioTable(args.scenario_table, manifest_path=args.manifest, max_scenarios=args.max_scenarios)
    for num_envs in args.num_envs:
        generator = BatchSimGenerator(table, num_envs=num_envs, seed=args.seed, strategy="grid_balanced")
        runner = BatchSimRunner(generator, config=BatchRunnerConfig(max_episode_steps=args.max_episode_steps))
        obs, _ = runner.reset()
        rng = np.random.default_rng(args.seed)
        actions = rng.uniform(-1.0, 1.0, size=(num_envs, runner.action_size)).astype(np.float32)
        start = time.perf_counter()
        for _ in range(args.steps):
            obs, rewards, dones, infos = runner.step(actions)
        elapsed = max(time.perf_counter() - start, 1e-9)
        print(json.dumps({
            "num_envs": num_envs,
            "steps": args.steps,
            "env_steps": num_envs * args.steps,
            "elapsed_s": elapsed,
            "sim_sps": (num_envs * args.steps) / elapsed,
            "obs_shape": list(obs.shape),
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
