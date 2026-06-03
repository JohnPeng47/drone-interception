from __future__ import annotations

import math

from control_sims.rpg_time_optimal.adapter import RpgTimeOptimalAdapter
from control_sims.rpg_time_optimal.config import RpgTimeOptimalConfig
from control_sims.rpg_time_optimal.motor_feedforward_policy import RpgTimeOptimalMotorFeedforwardPolicy
from control_sims.rpg_time_optimal.policy import RpgTimeOptimalControlPolicy
from control_sims.runner import _run_instances
from _robust_intercept_cases import read_six_robust_intercept_samples


def test_rpg_time_optimal_adapter_solves_generated_scenario():
    instance = read_six_robust_intercept_samples()[0]
    plan = RpgTimeOptimalAdapter(RpgTimeOptimalConfig(ipopt_max_iter=80)).solve(instance)

    assert plan.seed == instance.seed
    assert math.isfinite(plan.solve_wall_s)
    assert plan.total_time_s > 0.0
    assert plan.position_w.shape[0] == 3
    assert plan.motor_thrusts_n.shape[0] == 4


def test_rpg_time_optimal_policy_runs_generated_scenario_without_errors():
    instances = read_six_robust_intercept_samples()[:1]

    rows = _run_instances(
        instances,
        "rpg_time_optimal",
        RpgTimeOptimalControlPolicy,
        max_envs=1,
        log_snapshots=False,
        snapshot_log_rate=100,
    )["rows"]

    assert len(rows) == 1
    assert rows[0]["error"] is None
    assert int(rows[0]["steps"]) > 0
    assert math.isfinite(float(rows[0]["min_distance_m"]))


def test_rpg_time_optimal_motor_feedforward_policy_runs_generated_scenario_without_errors():
    instances = read_six_robust_intercept_samples()[:1]

    rows = _run_instances(
        instances,
        "rpg_time_optimal_motor_ff",
        RpgTimeOptimalMotorFeedforwardPolicy,
        max_envs=1,
        log_snapshots=False,
        snapshot_log_rate=100,
    )["rows"]

    assert len(rows) == 1
    assert rows[0]["error"] is None
    assert int(rows[0]["steps"]) > 0
    assert math.isfinite(float(rows[0]["min_distance_m"]))
