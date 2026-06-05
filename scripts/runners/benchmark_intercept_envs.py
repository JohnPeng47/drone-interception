from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.rl.puffer_intercept.benchmark_modes import run_puffer_native_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark interception rollout env paths.")
    parser.add_argument(
        "--scenario-file",
        default="scripts/generators/sim_instances/sobol_samples_512.csimin",
        help="Generated .csimin scenario file.",
    )
    parser.add_argument(
        "--mode",
        choices=("puffer_native",),
        default="puffer_native",
    )
    parser.add_argument("--num-envs", type=int, nargs="+", default=[os.cpu_count() or 1])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy-latency-us", type=float, default=0.0)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    args = parser.parse_args()

    for num_envs in args.num_envs:
        result = run_puffer_native_benchmark(
            args.scenario_file,
            num_envs=num_envs,
            steps=args.steps,
            seed=args.seed,
            policy_latency_us=args.policy_latency_us,
            max_episode_steps=args.max_episode_steps,
        )
        print(json.dumps(result.to_dict(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
