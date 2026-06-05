from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ai.rl.puffer_intercept.snapshot_eval import SnapshotEvalConfig, run_snapshot_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Run post-hoc snapshot eval for a SimEngine PPO checkpoint.")
    parser.add_argument("--scenario-table", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("snapshots/puffer_intercept"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=36)
    parser.add_argument("--samples-per-cell", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--snapshot-stride", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    args = parser.parse_args()

    summary_path = run_snapshot_eval(
        SnapshotEvalConfig(
            scenario_table=args.scenario_table,
            checkpoint=args.checkpoint,
            out_dir=args.out_dir,
            manifest=args.manifest,
            max_scenarios=args.max_scenarios,
            max_episodes=args.max_episodes,
            samples_per_cell=args.samples_per_cell,
            num_envs=args.num_envs,
            seed=args.seed,
            device=args.device,
            stochastic=bool(args.stochastic),
            snapshot_stride=args.snapshot_stride,
            max_episode_steps=args.max_episode_steps,
        )
    )
    print(summary_path.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
