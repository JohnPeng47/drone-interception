"""Run one minimal Beihang-inspired interception trial."""

from __future__ import annotations

import argparse

from pydrake.systems.analysis import Simulator

from ..config import TrialConfig
from .diagram import build_minimal_diagram


def run_trial(config: TrialConfig):
    diagram, logger = build_minimal_diagram(config)
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(0.0)
    simulator.Initialize()
    t = 0.0
    while t < config.duration_s:
        t = min(t + config.dt, config.duration_s)
        simulator.AdvanceTo(t)
        metrics = logger.final_metrics()
        if metrics is not None and (
            metrics.captured or metrics.crashed or metrics.out_of_bounds
        ):
            break
    return logger.final_metrics(), logger.samples()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=TrialConfig.duration_s)
    parser.add_argument("--dt", type=float, default=TrialConfig.dt)
    args = parser.parse_args()

    config = TrialConfig(duration_s=args.duration, dt=args.dt)
    metrics, samples = run_trial(config)
    if metrics is None:
        print("no samples recorded")
        return 1

    print("minimal Beihang-inspired interception replay")
    print(f"samples: {len(samples)}")
    print(f"captured: {metrics.captured}")
    print(f"capture_time_s: {metrics.capture_time_s}")
    print(f"final_distance_m: {metrics.distance_m:.3f}")
    print(f"min_distance_m: {metrics.min_distance_m:.3f}")
    print(f"final_in_view: {metrics.in_view}")
    print(f"final_image_error: {metrics.image_error}")
    print(f"control_effort: {metrics.control_effort:.3f}")
    print(f"crashed: {metrics.crashed}")
    print(f"out_of_bounds: {metrics.out_of_bounds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
