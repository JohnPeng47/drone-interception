from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from control_sims.eth_mpc.policy import EthMpcControlPolicy
from control_sims.runner import run_policy_cli


def main() -> int:
    return run_policy_cli(
        sim_name="eth_mpc",
        description="Run ETH MPCC++-style MPC pursuit scenarios.",
        policy_factory=EthMpcControlPolicy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
