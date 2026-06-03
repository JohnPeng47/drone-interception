"""Run a small deterministic sweep for the minimal interception task."""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from ..config import TargetConfig, TrialConfig
from .replay import run_trial


def config_for_seed(seed: int, duration_s: float, dt: float) -> TrialConfig:
    rng = np.random.default_rng(seed)
    target = TargetConfig(
        initial_position_w=(
            8.0 + float(rng.uniform(-1.0, 1.0)),
            float(rng.uniform(-1.4, 1.4)),
            2.1 + float(rng.uniform(-0.4, 0.4)),
        ),
        base_velocity_w=(-0.2 + float(rng.uniform(-0.15, 0.05)), 0.0, 0.0),
        weave_amplitude_m=(
            0.8 + float(rng.uniform(0.0, 1.0)),
            0.3 + float(rng.uniform(0.0, 0.6)),
        ),
        weave_frequency_hz=(
            0.12 + float(rng.uniform(0.0, 0.15)),
            0.08 + float(rng.uniform(0.0, 0.12)),
        ),
    )
    return replace(TrialConfig(duration_s=duration_s, dt=dt), target=target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--duration", type=float, default=TrialConfig.duration_s)
    parser.add_argument("--dt", type=float, default=TrialConfig.dt)
    args = parser.parse_args()

    rows = []
    for seed in range(1, args.trials + 1):
        metrics, _ = run_trial(config_for_seed(seed, args.duration, args.dt))
        rows.append((seed, metrics))

    print("seed | captured | capture_time | min_dist | final_dist | effort | crashed | oob")
    print("-----+----------+--------------+----------+------------+--------+---------+----")
    for seed, metrics in rows:
        capture_time = "-" if metrics.capture_time_s is None else f"{metrics.capture_time_s:.2f}"
        print(
            f"{seed:4d} | {str(metrics.captured):8s} | {capture_time:12s} | "
            f"{metrics.min_distance_m:8.3f} | {metrics.distance_m:10.3f} | "
            f"{metrics.control_effort:6.2f} | {str(metrics.crashed):7s} | "
            f"{str(metrics.out_of_bounds):3s}"
        )

    captured = sum(1 for _, metrics in rows if metrics.captured)
    mean_min = sum(metrics.min_distance_m for _, metrics in rows) / max(len(rows), 1)
    print()
    print(f"capture_rate: {captured}/{len(rows)}")
    print(f"mean_min_distance_m: {mean_min:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
