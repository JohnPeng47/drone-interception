from __future__ import annotations

import math

from control_sims.eth_mpc.policy import EthMpcControlPolicy
from control_sims.runner import _run_instances
from _robust_intercept_cases import read_six_robust_intercept_samples


def test_eth_mpc_controller_runs_generated_scenarios_without_errors():
    instances = read_six_robust_intercept_samples()[:2]

    rows = _run_instances(
        instances,
        "eth_mpc",
        EthMpcControlPolicy,
        max_envs=1,
        log_snapshots=False,
        snapshot_log_rate=100,
    )["rows"]

    assert len(rows) == len(instances)
    for row in rows:
        assert row["error"] is None
        assert int(row["steps"]) > 0
        assert math.isfinite(float(row["min_distance_m"]))
        assert math.isfinite(float(row["final_distance_m"]))
        assert math.isfinite(float(row["control_effort"]))
