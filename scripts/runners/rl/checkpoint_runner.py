from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ai.rl.puffer_intercept.snapshot_eval import (
    SnapshotEvalConfig,
    run_snapshot_eval,
    select_stratified_indices,
)
from ai.rl.puffer_intercept.scenario_table import ScenarioTable


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed-scenario snapshot eval over PPO checkpoints.")
    parser.add_argument("--scenario-table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("snapshots/checkpoint_runner"))
    parser.add_argument("--scenario-indices-from", type=Path, default=None)
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

    checkpoints = _checkpoint_paths(args.checkpoint, args.checkpoint_dir)
    if not checkpoints:
        raise ValueError("provide --checkpoint at least once or --checkpoint-dir")

    scenario_indices = _scenario_indices(args)
    summaries = []
    for checkpoint in checkpoints:
        summary_path = run_snapshot_eval(
            SnapshotEvalConfig(
                scenario_table=args.scenario_table,
                checkpoint=checkpoint,
                out_dir=args.out_dir,
                manifest=args.manifest,
                max_scenarios=args.max_scenarios,
                max_episodes=len(scenario_indices),
                samples_per_cell=args.samples_per_cell,
                num_envs=args.num_envs,
                seed=args.seed,
                device=args.device,
                stochastic=bool(args.stochastic),
                snapshot_stride=args.snapshot_stride,
                max_episode_steps=args.max_episode_steps,
                scenario_indices=tuple(scenario_indices),
            )
        )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append({
            "checkpoint": str(checkpoint),
            "summary_path": str(summary_path),
            "checkpoint_info": payload["checkpoint_info"],
            "summary": payload["summary"],
            "timing": payload["summary"].get("timing", {}),
        })

    print(json.dumps({
        "scenario_table": str(args.scenario_table),
        "manifest": None if args.manifest is None else str(args.manifest),
        "scenario_indices": scenario_indices,
        "checkpoints": summaries,
    }, indent=2, sort_keys=True))
    return 0


def _checkpoint_paths(paths: list[Path], checkpoint_dir: Path | None) -> list[Path]:
    out = [path for path in paths]
    if checkpoint_dir is not None:
        out.extend(
            path for path in sorted(checkpoint_dir.glob("*.pt"))
            if path.name not in {"latest.pt", "resume.pt"}
        )
    return out


def _scenario_indices(args: argparse.Namespace) -> list[int]:
    if args.scenario_indices_from is not None:
        payload = json.loads(args.scenario_indices_from.read_text(encoding="utf-8"))
        return [int(index) for index in payload["selected_scenario_indices"]]
    table = ScenarioTable(args.scenario_table, manifest_path=args.manifest, max_scenarios=args.max_scenarios)
    return select_stratified_indices(
        table,
        max_episodes=int(args.max_episodes),
        samples_per_cell=args.samples_per_cell,
        seed=int(args.seed),
    )


if __name__ == "__main__":
    raise SystemExit(main())
