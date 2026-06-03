from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from control_sims.beihang_paper_sim.policy_dkf import BeihangPaperDkfControlPolicy

from control_sims.runner import run_policy_cli


def main() -> int:
    return run_policy_cli(
        sim_name="beihang_paper_dkf",
        description="Run beihang_paper_sim scenarios with the DKF policy.",
        policy_factory=BeihangPaperDkfControlPolicy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
