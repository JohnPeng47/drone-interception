from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from control_sims.rpg_time_optimal.policy import RpgTimeOptimalControlPolicy
from control_sims.runner import run_policy_cli


def main() -> int:
    return run_policy_cli(
        sim_name="rpg_time_optimal",
        description="Run RPG time-optimal trajectory plans on generated pursuit scenarios.",
        policy_factory=RpgTimeOptimalControlPolicy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
