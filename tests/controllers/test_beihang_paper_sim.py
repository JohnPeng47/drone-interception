from __future__ import annotations

import pytest

from control_sims.beihang_paper_sim.policy import BeihangPaperSimControlPolicy
from control_sims.runner import _run_instances
from _robust_intercept_cases import read_golden_trial_rows, read_six_robust_intercept_samples


def test_beihang_paper_controller_matches_golden_runner_output():
    instances = read_six_robust_intercept_samples()
    golden = read_golden_trial_rows("beihang_paper")

    rows = [
        _run_instances(
            [instance],
            "beihang_paper",
            BeihangPaperSimControlPolicy,
            max_envs=1,
            log_snapshots=False,
            snapshot_log_rate=100,
        )["rows"][0]
        for instance in instances
    ]

    assert [row["seed"] for row in rows] == list(golden)
    for row in rows:
        expected = golden[int(row["seed"])]
        assert row["error"] is None
        assert row["caught"] is (expected["caught"] == "True")
        assert row["catch_time_s"] == _optional_float_approx(expected["catch_time_s"])
        assert row["steps"] == int(expected["steps"])
        assert row["crashed"] is (expected["crashed"] == "True")
        assert row["out_of_bounds"] is (expected["out_of_bounds"] == "True")
        assert row["min_distance_m"] == pytest.approx(float(expected["min_distance_m"]), abs=1.0e-9)
        assert row["final_distance_m"] == pytest.approx(float(expected["final_distance_m"]), abs=1.0e-9)
        assert row["visible_fraction"] == pytest.approx(float(expected["visible_fraction"]), abs=1.0e-12)
        assert row["control_effort"] == pytest.approx(float(expected["control_effort"]), abs=1.0e-9)


def _optional_float_approx(value: str):
    if value == "":
        return None
    return pytest.approx(float(value), abs=1.0e-9)
