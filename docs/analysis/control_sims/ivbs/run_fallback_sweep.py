from __future__ import annotations

import argparse
import concurrent.futures
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
    parser = argparse.ArgumentParser(description="Sweep IVBS FOV-retention/fallback candidates.")
    parser.add_argument("--scenario-table", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-envs", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
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
            workers=int(args.workers),
        )
        elapsed = time.perf_counter() - start
        summary = _summarize(rows)
        summary.update(
            {
                "name": candidate["name"],
                "elapsed_wall_s": elapsed,
                "gains": candidate["gains"],
                "observer_config": candidate["observer_config"],
            }
        )
        _write_csv(candidate_dir / "trials.csv", rows)
        (candidate_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    summaries.sort(
        key=lambda row: (
            -float(row["catch_fraction"]),
            -float(row["visible_fraction_mean"]),
            float(row["min_distance_p50_m"]),
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
                "workers": int(args.workers),
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
            "name": "pre_m2_baseline",
            "gains": {
                "k_b": 0.8,
                "cautious_closing_accel_mps2": 4.0,
                "cautious_velocity_damping": 0.25,
            },
            "observer_config": {
                "metric_range_std_threshold_m": 1.5,
            },
        },
        {
            "name": "current_default",
            "gains": {
                "k_b": 0.65,
                "cautious_closing_accel_mps2": 3.0,
                "cautious_velocity_damping": 0.25,
            },
            "observer_config": {
                "min_detections_for_metric": 2,
                "metric_range_std_threshold_m": 2.0,
            },
        },
        {
            "name": "slow_close_strict_metric",
            "gains": {
                "cautious_closing_accel_mps2": 2.0,
                "cautious_velocity_damping": 0.35,
            },
            "observer_config": {
                "min_detections_for_metric": 4,
                "metric_range_std_threshold_m": 1.0,
            },
        },
        {
            "name": "slow_close_loose_metric",
            "gains": {
                "cautious_closing_accel_mps2": 2.0,
                "cautious_velocity_damping": 0.25,
            },
            "observer_config": {
                "min_detections_for_metric": 2,
                "metric_range_std_threshold_m": 2.0,
            },
        },
        {
            "name": "medium_close_strict_metric",
            "gains": {
                "cautious_closing_accel_mps2": 3.0,
                "cautious_velocity_damping": 0.35,
            },
            "observer_config": {
                "min_detections_for_metric": 4,
                "metric_range_std_threshold_m": 1.0,
            },
        },
        {
            "name": "fast_center_low_close",
            "gains": {
                "k_b": 0.65,
                "cautious_closing_accel_mps2": 1.5,
                "cautious_velocity_damping": 0.45,
            },
            "observer_config": {
                "min_detections_for_metric": 5,
                "metric_range_std_threshold_m": 1.0,
            },
        },
        {
            "name": "center_loose_metric",
            "gains": {
                "k_b": 0.65,
                "cautious_closing_accel_mps2": 2.0,
                "cautious_velocity_damping": 0.25,
            },
            "observer_config": {
                "min_detections_for_metric": 2,
                "metric_range_std_threshold_m": 2.0,
            },
        },
        {
            "name": "center_medium_close_loose_metric",
            "gains": {
                "k_b": 0.65,
                "cautious_closing_accel_mps2": 3.0,
                "cautious_velocity_damping": 0.25,
            },
            "observer_config": {
                "min_detections_for_metric": 2,
                "metric_range_std_threshold_m": 2.0,
            },
        },
    ]


def _run_candidate(
    instances,
    *,
    gains: dict[str, float],
    observer_config: dict[str, float],
    max_envs: int,
    workers: int,
):
    chunks = [instances[index:index + max_envs] for index in range(0, len(instances), max_envs)]
    if workers <= 1:
        results = [
            _run_chunk(chunk, gains=gains, observer_config=observer_config, max_envs=max_envs)
            for chunk in chunks
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = [
                executor.submit(
                    _run_chunk,
                    chunk,
                    gains=gains,
                    observer_config=observer_config,
                    max_envs=max_envs,
                )
                for chunk in chunks
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
    rows = [row for result_rows in results for row in result_rows]
    rows.sort(key=lambda row: int(row["seed"]))
    return rows


def _run_chunk(instances, *, gains: dict[str, float], observer_config: dict[str, float], max_envs: int):
    policy = IVBSControlPolicy(gains=gains, observer_config=observer_config)
    result = SimRunner(max_envs=max_envs).run(instances, policy)
    return [
        _row_from_completed("ivbs", completed, result.steps)
        for completed in result.completed
    ]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    caught = np.array([bool(row["caught"]) for row in rows], dtype=bool)
    min_distance = np.asarray([float(row["min_distance_m"]) for row in rows], dtype=float)
    final_distance = np.asarray([float(row["final_distance_m"]) for row in rows], dtype=float)
    visible = np.asarray([float(row["visible_fraction"]) for row in rows], dtype=float)
    effort = np.asarray([float(row["control_effort"]) for row in rows], dtype=float)
    return {
        "n": int(len(rows)),
        "caught": int(np.sum(caught)),
        "catch_fraction": float(np.mean(caught)) if caught.size else float("nan"),
        "min_distance_p50_m": _percentile(min_distance, 50),
        "min_distance_p90_m": _percentile(min_distance, 90),
        "final_distance_p50_m": _percentile(final_distance, 50),
        "visible_fraction_mean": _mean(visible),
        "control_effort_mean": _mean(effort),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: np.ndarray, percentile: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def _mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
