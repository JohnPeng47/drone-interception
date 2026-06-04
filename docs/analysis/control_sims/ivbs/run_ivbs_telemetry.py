from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from backends.csim.generator.instance_store import read_sim_instances
from backends.csim.runner import SimRunner
from control_sims.ivbs.policy import IVBSControlPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description="Run IVBS with command/observer telemetry.")
    parser.add_argument("--scenario-table", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-envs", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    instances = read_sim_instances(args.scenario_table, count=args.samples, offset=int(args.offset))
    if not instances:
        raise ValueError("no scenarios selected")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = IVBSControlPolicy(record_telemetry=True)
    result = SimRunner(max_envs=int(args.max_envs)).run(instances, policy)

    trial_rows = [_trial_row(completed) for completed in result.completed]
    telemetry_rows = list(policy.telemetry_rows)
    _write_csv(out_dir / "trials.csv", trial_rows)
    _write_csv(out_dir / "telemetry.csv", telemetry_rows)
    summary = _summary(trial_rows, telemetry_rows)
    summary.update(
        {
            "source": str(args.scenario_table),
            "samples": len(instances),
            "offset": int(args.offset),
            "max_envs": int(args.max_envs),
            "artifacts": {
                "trials": str(out_dir / "trials.csv"),
                "telemetry": str(out_dir / "telemetry.csv"),
            },
        }
    )
    encoded = json.dumps(summary, allow_nan=False, indent=2, sort_keys=True)
    (out_dir / "summary.json").write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


def _trial_row(completed) -> dict[str, Any]:
    terminal = completed.terminal_snapshot
    return {
        "seed": int(completed.seed),
        "workload_index": int(completed.workload_index),
        "caught": completed.terminal_reason == "intercepted",
        "terminal_reason": completed.terminal_reason,
        "steps": int(completed.steps),
        "elapsed_s": float(completed.elapsed_s),
        "min_distance_m": float(terminal.metrics.min_distance_m),
        "final_distance_m": float(terminal.metrics.distance_m),
    }


def _summary(trial_rows: list[dict[str, Any]], telemetry_rows: list[dict[str, Any]]) -> dict[str, Any]:
    caught_by_workload = {
        (int(row["seed"]), int(row["workload_index"])): bool(row["caught"])
        for row in trial_rows
    }
    modes = Counter(str(row["mode"]) for row in telemetry_rows)
    per_workload = defaultdict(list)
    for row in telemetry_rows:
        per_workload[(int(row["seed"]), int(row["workload_index"]))].append(row)

    seed_rows = []
    for (seed, workload_index), rows in per_workload.items():
        detected = sum(1 for row in rows if _bool(row["detected"]))
        metric = sum(1 for row in rows if str(row["mode"]) == "metric")
        fallback = sum(1 for row in rows if str(row["mode"]) == "bearing_fallback")
        hover = sum(1 for row in rows if str(row["mode"]) == "hover")
        seed_rows.append(
            {
                "seed": seed,
                "workload_index": workload_index,
                "caught": caught_by_workload.get((seed, workload_index), False),
                "visible_fraction": detected / max(len(rows), 1),
                "metric_fraction": metric / max(len(rows), 1),
                "fallback_fraction": fallback / max(len(rows), 1),
                "hover_fraction": hover / max(len(rows), 1),
                "mean_range_std_m": _mean(float(row["range_std_m"]) for row in rows),
                "mean_bearing_error_rad": _mean(float(row["bearing_error_rad"]) for row in rows),
            }
        )

    buckets = {}
    for lo, hi in ((0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 1.0)):
        if hi >= 1.0:
            rows = [row for row in seed_rows if lo <= float(row["visible_fraction"]) <= hi]
        else:
            rows = [row for row in seed_rows if lo <= float(row["visible_fraction"]) < hi]
        label = f"{lo:.2f}-{hi:.2f}"
        buckets[label] = {
            "n": len(rows),
            "caught": sum(1 for row in rows if row["caught"]),
            "metric_fraction_mean": _mean(float(row["metric_fraction"]) for row in rows),
            "fallback_fraction_mean": _mean(float(row["fallback_fraction"]) for row in rows),
            "hover_fraction_mean": _mean(float(row["hover_fraction"]) for row in rows),
            "range_std_mean": _mean(float(row["mean_range_std_m"]) for row in rows),
            "bearing_error_mean_rad": _mean(float(row["mean_bearing_error_rad"]) for row in rows),
        }

    return {
        "n": len(trial_rows),
        "caught": sum(1 for row in trial_rows if row["caught"]),
        "catch_fraction": sum(1 for row in trial_rows if row["caught"]) / max(len(trial_rows), 1),
        "telemetry_rows": len(telemetry_rows),
        "mode_counts": dict(sorted(modes.items())),
        "visibility_buckets": buckets,
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


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


if __name__ == "__main__":
    raise SystemExit(main())
