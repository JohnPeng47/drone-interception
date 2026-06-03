from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_sims.rpg_time_optimal.config import RpgTimeOptimalConfig
from control_sims.rpg_time_optimal.policy import RpgTimeOptimalControlPolicy
from control_sims.runner import run_policy_cli


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RPG controller with a planner-only CPC tolerance override.")
    parser.add_argument("--cpc-tolerance", type=float, required=True)
    parser.add_argument("--time-scale", type=float, default=1.0)
    args, passthrough = parser.parse_known_args()

    def policy_factory() -> RpgTimeOptimalControlPolicy:
        return RpgTimeOptimalControlPolicy(
            RpgTimeOptimalConfig(
                cpc_tolerance_m=float(args.cpc_tolerance),
                plan_time_scale=float(args.time_scale),
            )
        )

    sys.argv = [sys.argv[0], *passthrough]
    return run_policy_cli(
        sim_name=f"rpg_time_optimal_cpc_tol_{float(args.cpc_tolerance):g}_time_scale_{float(args.time_scale):g}",
        description="Run RPG time-optimal control with planner-only CPC tolerance override.",
        policy_factory=policy_factory,
    )


if __name__ == "__main__":
    raise SystemExit(main())
