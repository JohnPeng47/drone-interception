from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from control_sims.beihang_minimal_sim.policy import BeihangMinimalSimControlPolicy

from control_sims.runner import run_policy_cli


def main() -> int:
    return run_policy_cli(
        sim_name="beihang_minimal",
        description="Run beihang_minimal_sim scenarios.",
        policy_factory=BeihangMinimalSimControlPolicy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
