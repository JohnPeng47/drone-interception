from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ai.rl.simengine_batch.policy import NeuralNetworkSimControlPolicy
from backends.csim.generator.instance_store import read_sim_instances
from backends.csim.runner import SimRunner
from control_sims.logging import snapshot_rows_from_step
from control_sims.runner import (
    ControlSimRunsRunner,
    TRIAL_FIELDNAMES,
    _completed_step_filter,
    _row_from_completed,
    _scenario_fields_from_instance,
    _snapshot_logging_config,
)
from utils.logging import RunsDirLogger


def main() -> int:
    parser = argparse.ArgumentParser(description="Run neural PPO policy inference over generated SimEngine scenarios.")
    parser.add_argument("--scenario-table", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-envs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-actions", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--log-snapshots", action="store_true")
    parser.add_argument("--snapshot-log-rate", type=int, default=100)
    args = parser.parse_args()

    if int(args.max_envs) <= 0:
        raise ValueError("--max-envs must be positive")
    if int(args.snapshot_log_rate) <= 0:
        raise ValueError("--snapshot-log-rate must be positive")

    instances = read_sim_instances(args.scenario_table, count=args.samples, offset=int(args.offset))
    run_artifacts = ControlSimRunsRunner("neural_policy", RunsDirLogger("neural_policy"))
    run_dir = run_artifacts.create_run_dir(suffix=args.run_suffix, out_dir=args.out_dir)

    start = time.perf_counter()
    policy = NeuralNetworkSimControlPolicy(
        args.checkpoint,
        device=args.device,
        deterministic=not bool(args.sample_actions),
    )
    runner = SimRunner(max_envs=min(int(args.max_envs), max(len(instances), 1)))
    result = runner.run(instances, policy) if instances else None

    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    if result is not None:
        for completed in result.completed:
            row = _row_from_completed("neural_policy", completed, result.steps)
            row.update(_scenario_fields_from_instance(completed.instance))
            row["wall_s"] = (time.perf_counter() - start) / max(len(result.completed), 1)
            row["error"] = None
            rows.append(row)
        if args.log_snapshots:
            logging_config = _snapshot_logging_config(run_dir, int(args.snapshot_log_rate))
            step_filter = _completed_step_filter(result.completed)
            for step in result.steps:
                snapshots.extend(
                    row for row in snapshot_rows_from_step("neural_policy", logging_config, step)
                    if (int(row["slot"]), int(row["workload_index"])) in step_filter
                )

    rows.sort(key=lambda row: int(row["seed"]))
    snapshots.sort(key=lambda row: (int(row["seed"]), int(row["tick"])))
    trials_path = run_artifacts.write_trials(run_dir, rows, TRIAL_FIELDNAMES)
    snapshot_path = None
    if args.log_snapshots:
        snapshot_path = run_artifacts.write_snapshots(
            run_dir,
            snapshots,
            _snapshot_logging_config(run_dir, int(args.snapshot_log_rate)),
        )

    summary = {
        "run_dir": str(run_dir),
        "source": str(args.scenario_table),
        "sim": "neural_policy",
        "num_scenarios": len(instances),
        "offset": int(args.offset),
        "max_envs": int(args.max_envs),
        "elapsed_wall_s": time.perf_counter() - start,
        "policy": policy.metadata(),
        "snapshot_log": {
            "enabled": bool(args.log_snapshots),
            "every_n_ticks": int(args.snapshot_log_rate),
            "path": None if snapshot_path is None else str(snapshot_path),
        },
        "summary": _summarize(rows),
        "trials_path": str(trials_path),
    }
    summary_path = run_artifacts.write_summary(run_dir, summary)
    print(summary_path.read_text(encoding="utf-8"), end="")
    return 0


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    caught = np.asarray([bool(row["caught"]) for row in valid], dtype=bool)
    min_distance = _finite(row["min_distance_m"] for row in valid)
    final_distance = _finite(row["final_distance_m"] for row in valid)
    return {
        "n": len(rows),
        "valid": len(valid),
        "errors": len(rows) - len(valid),
        "catch_fraction": float(np.mean(caught)) if caught.size else math.nan,
        "min_distance_p50_m": _percentile(min_distance, 50),
        "final_distance_p50_m": _percentile(final_distance, 50),
    }


def _finite(values) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else math.nan


if __name__ == "__main__":
    raise SystemExit(main())
