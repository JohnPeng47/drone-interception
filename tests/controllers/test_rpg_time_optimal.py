from __future__ import annotations

import math

from control_sims.rpg_time_optimal.planner import RpgTimeOptimalPlanner
from control_sims.rpg_time_optimal.config import RpgTimeOptimalConfig
from control_sims.rpg_time_optimal.motor_feedforward_policy import RpgTimeOptimalMotorFeedforwardPolicy
from control_sims.rpg_time_optimal.portfolio_policy import (
    RpgPlanReplayScore,
    RpgTimeOptimalPortfolioCandidate,
    _early_accept,
    select_best_scored_plan,
)
from control_sims.rpg_time_optimal.policy import RpgTimeOptimalControlPolicy
from control_sims.runner import _run_instances
from _robust_intercept_cases import read_six_robust_intercept_samples


def test_rpg_time_optimal_planner_solves_generated_scenario():
    instance = read_six_robust_intercept_samples()[0]
    plan = RpgTimeOptimalPlanner(RpgTimeOptimalConfig(ipopt_max_iter=80)).solve(instance)

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


def test_rpg_portfolio_selector_rejects_dirty_lower_distance_plan():
    clean_candidate = RpgTimeOptimalPortfolioCandidate("clean", RpgTimeOptimalConfig())
    dirty_candidate = RpgTimeOptimalPortfolioCandidate("dirty", RpgTimeOptimalConfig())
    clean_score = _portfolio_score(
        "clean",
        clean=True,
        caught=True,
        min_distance_m=0.42,
        capture_steps=2,
    )
    dirty_score = _portfolio_score(
        "dirty",
        clean=False,
        caught=True,
        min_distance_m=0.01,
        capture_steps=10,
    )

    selected_candidate, _, selected_score = select_best_scored_plan(
        (
            (dirty_candidate, object(), dirty_score),
            (clean_candidate, object(), clean_score),
        )
    )

    assert selected_candidate.name == "clean"
    assert selected_score is clean_score


def test_rpg_portfolio_selector_prefers_capture_then_dwell_then_margin():
    caught_short = RpgTimeOptimalPortfolioCandidate("caught_short", RpgTimeOptimalConfig())
    caught_long = RpgTimeOptimalPortfolioCandidate("caught_long", RpgTimeOptimalConfig())
    missed_close = RpgTimeOptimalPortfolioCandidate("missed_close", RpgTimeOptimalConfig())

    selected_candidate, _, _ = select_best_scored_plan(
        (
            (
                missed_close,
                object(),
                _portfolio_score("missed_close", clean=True, caught=False, min_distance_m=0.04, capture_steps=0),
            ),
            (
                caught_short,
                object(),
                _portfolio_score("caught_short", clean=True, caught=True, min_distance_m=0.20, capture_steps=1),
            ),
            (
                caught_long,
                object(),
                _portfolio_score("caught_long", clean=True, caught=True, min_distance_m=0.30, capture_steps=3),
            ),
        )
    )

    assert selected_candidate.name == "caught_long"


def test_rpg_portfolio_early_accept_requires_clean_consecutive_dwell():
    assert _early_accept(
        _portfolio_score("strong", clean=True, caught=True, min_distance_m=0.3, capture_steps=20),
        15,
        max_min_distance_m=0.45,
        max_tracking_error_mean_m=0.5,
    )
    assert not _early_accept(
        _portfolio_score("dirty", clean=False, caught=True, min_distance_m=0.01, capture_steps=30),
        15,
        max_min_distance_m=0.45,
        max_tracking_error_mean_m=0.5,
    )
    assert not _early_accept(
        _portfolio_score("weak", clean=True, caught=True, min_distance_m=0.49, capture_steps=3),
        15,
        max_min_distance_m=0.45,
        max_tracking_error_mean_m=0.5,
    )
    assert not _early_accept(
        _portfolio_score("missed", clean=True, caught=False, min_distance_m=0.7, capture_steps=0),
        15,
        max_min_distance_m=0.45,
        max_tracking_error_mean_m=0.5,
    )
    assert not _early_accept(
        _portfolio_score("poor_margin", clean=True, caught=True, min_distance_m=0.46, capture_steps=20),
        15,
        max_min_distance_m=0.45,
        max_tracking_error_mean_m=0.5,
    )
    assert not _early_accept(
        _portfolio_score(
            "poor_tracking",
            clean=True,
            caught=True,
            min_distance_m=0.3,
            capture_steps=20,
            tracking_error_m=0.6,
        ),
        15,
        max_min_distance_m=0.45,
        max_tracking_error_mean_m=0.5,
    )


def _portfolio_score(
    candidate_name: str,
    *,
    clean: bool,
    caught: bool,
    min_distance_m: float,
    capture_steps: int,
    tracking_error_m: float = 0.1,
) -> RpgPlanReplayScore:
    return RpgPlanReplayScore(
        candidate_name=candidate_name,
        clean=clean,
        rollout_caught_radius=caught,
        rollout_min_distance_m=min_distance_m,
        rollout_final_distance_m=min_distance_m,
        rollout_capture_steps=capture_steps,
        rollout_max_consecutive_capture_steps=capture_steps,
        rollout_position_tracking_error_mean_m=tracking_error_m,
        replay_wall_s=0.0,
        plan_total_time_s=1.0,
        solver_success=clean,
        constraint_violation_max=0.0 if clean else 1.0,
        terminal_tolerance_satisfied=clean,
        planned_feasible=clean,
    )
