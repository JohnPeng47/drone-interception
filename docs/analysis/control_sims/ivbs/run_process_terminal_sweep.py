from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from backends.csim.generator.instance_store import read_sim_instances
from backends.csim.runner import SimRunner
from control_sims.ivbs.policy import IVBSControlPolicy
from control_sims.runner import _row_from_completed


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep IVBS observer process models and terminal visual behavior.")
    parser.add_argument("--scenario-table", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-envs", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    instances = read_sim_instances(args.scenario_table, count=args.samples, offset=int(args.offset))
    if not instances:
        raise ValueError("no scenarios selected")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for candidate in _candidates():
        candidate_dir = args.out_dir / candidate["name"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        rows = _run_candidate(
            instances,
            gains=candidate["gains"],
            observer_config=candidate["observer_config"],
            max_envs=int(args.max_envs),
        )
        summary = _summarize(rows)
        summary.update(
            {
                "name": candidate["name"],
                "elapsed_wall_s": time.perf_counter() - start,
                "gains": candidate["gains"],
                "observer_config": candidate["observer_config"],
            }
        )
        _write_csv(candidate_dir / "trials.csv", rows)
        (candidate_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    summaries.sort(
        key=lambda row: (
            -float(row["catch_fraction"]),
            float(row["min_distance_p50_m"]),
            -float(row["visible_fraction_mean"]),
        )
    )
    _write_csv(args.out_dir / "summary.csv", summaries)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "source": str(args.scenario_table),
                "samples": len(instances),
                "offset": int(args.offset),
                "max_envs": int(args.max_envs),
                "best": summaries[0] if summaries else None,
                "candidates": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "current_default",
            "gains": {"terminal_visual_range_m": 0.0},
            "observer_config": {"target_process_model": "constant_velocity"},
        },
        {
            "name": "constant_velocity_terminal",
            "gains": {"terminal_visual_range_m": 2.5},
            "observer_config": {"target_process_model": "constant_velocity"},
        },
        {
            "name": "stationary_terminal",
            "gains": {"terminal_visual_range_m": 2.5},
            "observer_config": {"target_process_model": "stationary"},
        },
        {
            "name": "damped_terminal",
            "gains": {"terminal_visual_range_m": 2.5},
            "observer_config": {"target_process_model": "damped_velocity"},
        },
        {
            "name": "damped_no_terminal",
            "gains": {"terminal_visual_range_m": 0.0},
            "observer_config": {"target_process_model": "damped_velocity"},
        },
        {
            "name": "damped_conservative_terminal",
            "gains": {
                "terminal_visual_range_m": 2.0,
                "terminal_closing_speed_max_mps": 2.0,
                "terminal_closing_accel_mps2": 0.75,
            },
            "observer_config": {"target_process_model": "damped_velocity"},
        },
    ]


def _run_candidate(
    instances,
    *,
    gains: dict[str, float],
    observer_config: dict[str, float],
    max_envs: int,
):
    result = SimRunner(max_envs=max_envs).run(
        instances,
        IVBSControlPolicy(gains=gains, observer_config=observer_config),
    )
    return [_row_from_completed("ivbs", completed, result.steps) for completed in result.completed]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    caught = sum(1 for row in rows if bool(row["caught"]))
    return {
        "n": len(rows),
        "caught": caught,
        "catch_fraction": caught / max(len(rows), 1),
        "min_distance_p50_m": _percentile([row["min_distance_m"] for row in rows], 50),
        "visible_fraction_mean": _mean(row["visible_fraction"] for row in rows),
        "control_effort_mean": _mean(row["control_effort"] for row in rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _percentile(values, percentile: float) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.percentile(finite, percentile)) if finite else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
