from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_sims.rpg_time_optimal.config import RpgTimeOptimalConfig
from control_sims.rpg_time_optimal.motor_feedforward_policy import RpgTimeOptimalMotorFeedforwardPolicy
from control_sims.runner import run_policy_cli


@dataclass(frozen=True)
class MotorFeedforwardPolicyFactory:
    cpc_tolerance: float | None
    time_scale: float

    def __call__(self) -> RpgTimeOptimalMotorFeedforwardPolicy:
        return RpgTimeOptimalMotorFeedforwardPolicy(
            RpgTimeOptimalConfig(
                cpc_tolerance_m=self.cpc_tolerance,
                plan_time_scale=float(self.time_scale),
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CPC motor feedforward through SimEngine.")
    parser.add_argument("--cpc-tolerance", type=float, default=None)
    parser.add_argument("--time-scale", type=float, default=1.0)
    args, passthrough = parser.parse_known_args()

    suffix = "default" if args.cpc_tolerance is None else f"{float(args.cpc_tolerance):g}"
    sys.argv = [sys.argv[0], *passthrough]
    return run_policy_cli(
        sim_name=f"rpg_time_optimal_motor_ff_cpc_tol_{suffix}_time_scale_{float(args.time_scale):g}",
        description="Run RPG/CPC rotor-thrust plans as SimEngine motor-speed commands.",
        policy_factory=MotorFeedforwardPolicyFactory(args.cpc_tolerance, float(args.time_scale)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
