from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ai.rl.puffer_intercept.scenario_table import ScenarioTable
from ai.rl.puffer_intercept.snapshot_eval import SnapshotEvalConfig, run_snapshot_eval


DEFAULT_RUN_ROOT = Path("ai/rl/runs/2026-06-04")
DEFAULT_SCENARIO_TABLE = Path("scripts/generators/sim_instances/rl/stationary_target_512/sobol_samples.csimin")
DEFAULT_SNAPSHOT_DIR_NAME = "stationary_target_512"
SKIPPED_CHECKPOINT_NAMES = {"latest.pt", "resume.pt"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run post-training snapshot eval for all checkpoints in RL run directories.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_RUN_ROOT],
        help="Run directories or a parent directory containing run directories.",
    )
    parser.add_argument("--scenario-table", type=Path, default=DEFAULT_SCENARIO_TABLE)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--snapshot-dir-name", default=DEFAULT_SNAPSHOT_DIR_NAME)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--snapshot-stride", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit evaluated scenarios; defaults to all scenarios.")
    args = parser.parse_args()

    run_dirs = _run_dirs(args.paths)
    if not run_dirs:
        raise ValueError(f"no run directories with checkpoints found under: {', '.join(str(path) for path in args.paths)}")

    table = ScenarioTable(args.scenario_table, manifest_path=args.manifest)
    scenario_count = table.count if args.max_episodes is None else min(int(args.max_episodes), table.count)
    scenario_indices = tuple(range(scenario_count))

    started = time.perf_counter()
    run_summaries = []
    for run_dir in run_dirs:
        checkpoints = _checkpoint_paths(run_dir)
        if not checkpoints:
            continue
        out_dir = run_dir / "snapshots" / str(args.snapshot_dir_name)
        checkpoint_summaries = []
        for checkpoint in checkpoints:
            print(f"[post_run] evaluating {checkpoint} on {scenario_count} scenarios", flush=True)
            summary_path = run_snapshot_eval(
                SnapshotEvalConfig(
                    scenario_table=args.scenario_table,
                    checkpoint=checkpoint,
                    out_dir=out_dir,
                    manifest=args.manifest,
                    max_episodes=scenario_count,
                    num_envs=int(args.num_envs),
                    seed=int(args.seed),
                    device=str(args.device),
                    stochastic=bool(args.stochastic),
                    snapshot_stride=int(args.snapshot_stride),
                    max_episode_steps=args.max_episode_steps,
                    scenario_indices=scenario_indices,
                )
            )
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoint_summaries.append(
                {
                    "checkpoint": str(checkpoint),
                    "summary_path": str(summary_path),
                    "checkpoint_info": payload["checkpoint_info"],
                    "summary": payload["summary"],
                }
            )
        run_summary = {
            "run_dir": str(run_dir),
            "out_dir": str(out_dir),
            "checkpoint_count": len(checkpoint_summaries),
            "checkpoints": checkpoint_summaries,
        }
        run_summaries.append(run_summary)
        _write_run_summary(out_dir, args, scenario_indices, run_summary)

    result = {
        "scenario_table": str(args.scenario_table),
        "manifest": None if args.manifest is None else str(args.manifest),
        "scenario_count": scenario_count,
        "snapshot_stride": int(args.snapshot_stride),
        "num_envs": int(args.num_envs),
        "run_count": len(run_summaries),
        "runs": run_summaries,
        "elapsed_wall_s": time.perf_counter() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_dirs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path = Path(path)
        candidates = [path] if _checkpoint_paths(path) else sorted(child for child in path.iterdir() if child.is_dir())
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or not _checkpoint_paths(candidate):
                continue
            seen.add(resolved)
            out.append(candidate)
    return out


def _checkpoint_paths(run_dir: Path) -> list[Path]:
    search_dirs = [
        run_dir / "checkpoints" / "puffer_intercept",
        run_dir / "checkpoints",
        run_dir,
    ]
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        checkpoints = _checkpoint_files(directory)
        if checkpoints:
            return checkpoints
    return []


def _checkpoint_files(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.glob("*.pt"))
        if path.name not in SKIPPED_CHECKPOINT_NAMES and not path.name.startswith(".")
    ]


def _write_run_summary(
    out_dir: Path,
    args: argparse.Namespace,
    scenario_indices: tuple[int, ...],
    run_summary: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario_table": str(args.scenario_table),
        "manifest": None if args.manifest is None else str(args.manifest),
        "selected_scenario_indices": list(scenario_indices),
        "snapshot_stride": int(args.snapshot_stride),
        "num_envs": int(args.num_envs),
        **run_summary,
    }
    (out_dir / "post_run_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
