from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from backends import (
    PursuerInitialState,
    PursuerParams,
    SimConfig,
    SimInstance,
    SimOptions,
    TargetBehaviorConfig,
    TargetConfig,
    TargetInitialState,
)
from control_sims.optimizing_rpg.fixed_time import solve_fixed_time
from control_sims.optimizing_rpg.final_harness import _passed_acceptance as _final_harness_passed_acceptance
from control_sims.optimizing_rpg.final_harness import FinalHarnessConfig, run_final_harness
from control_sims.optimizing_rpg.rollout import STATE_SIZE, replay_motor_commands_in_simengine, rollout_motor_commands
from control_sims.optimizing_rpg.rollout_harness import _passed_acceptance
from control_sims.optimizing_rpg.structured_update_harness import _passed_acceptance as _structured_update_passed_acceptance
from control_sims.optimizing_rpg.structured_update import StructuredUpdateConfig, run_structured_update
from control_sims.optimizing_rpg.switching_template import (
    SwitchingTemplateConfig,
    _backend_dt,
    _ctbr_to_motor_speeds_batch,
    _effective_max_rate_rps,
    _effective_max_thrust_n,
    _fast_replay_backend_tick_commands,
    _templates_for_steps,
    _time_grid,
    find_switching_template_intercept,
)
from control_sims.optimizing_rpg.switching_template_harness import (
    SwitchingTemplateHarnessConfig,
    _summary as _switching_template_summary,
)
from control_sims.optimizing_rpg.time_search import find_fastest_intercept
from control_sims.optimizing_rpg.time_search_harness import (
    TimeSearchHarnessConfig,
    _passed_acceptance as _time_search_passed_acceptance,
)


def test_rollout_accepts_transposed_controls_and_clips_command_rpm():
    instance = _instance()
    assert instance.config is not None
    controls = np.array(
        [
            [-100.0, 150.0],
            [0.0, 200.0],
            [1250.0, 300.0],
            [5000.0, 400.0],
        ],
        dtype=float,
    )

    trajectory = rollout_motor_commands(instance, controls, total_time_s=0.04)

    assert trajectory.controls.shape == (2, 4)
    np.testing.assert_allclose(trajectory.controls[0], [100.0, 100.0, 1000.0, 1000.0])
    np.testing.assert_allclose(trajectory.controls[1], [150.0, 200.0, 300.0, 400.0])
    assert trajectory.states.shape == (3, STATE_SIZE)
    assert np.all(np.isfinite(trajectory.states))
    np.testing.assert_allclose(np.linalg.norm(trajectory.quat_wxyz, axis=1), 1.0, atol=1.0e-9)


def test_rollout_rejects_invalid_control_shape_and_time():
    instance = _instance()

    with pytest.raises(ValueError, match="one axis of length 4"):
        rollout_motor_commands(instance, np.zeros((3, 3)), total_time_s=0.04)

    with pytest.raises(ValueError, match="finite and positive"):
        rollout_motor_commands(instance, np.zeros((2, 4)), total_time_s=0.0)

    with pytest.raises(ValueError, match="finite and positive"):
        rollout_motor_commands(instance, np.zeros((2, 4)), total_time_s=float("nan"))

    with pytest.raises(ValueError, match="at least one node"):
        rollout_motor_commands(instance, np.zeros((0, 4)), total_time_s=0.04)

    controls = np.zeros((2, 4), dtype=float)
    controls[0, 0] = float("nan")
    with pytest.raises(ValueError, match="controls must be finite"):
        rollout_motor_commands(instance, controls, total_time_s=0.04)


def test_simengine_replay_rejects_empty_controls():
    instance = _instance()

    with pytest.raises(ValueError, match="at least one node"):
        replay_motor_commands_in_simengine(instance, np.zeros((0, 4)), total_time_s=0.04)


@pytest.mark.parametrize("backend_dt", [0.0, -0.005])
def test_simengine_replay_rejects_non_positive_effective_backend_dt(backend_dt):
    instance = _instance(backend_dt=backend_dt)

    with pytest.raises(ValueError, match="effective backend dt"):
        replay_motor_commands_in_simengine(instance, np.full((1, 4), 100.0), total_time_s=0.001)


def test_simengine_replay_splits_steps_at_command_boundaries():
    instance = _instance(backend_dt=0.02)
    controls = np.array(
        [
            [100.0, 100.0, 100.0, 100.0],
            [1000.0, 100.0, 100.0, 100.0],
            [100.0, 1000.0, 100.0, 100.0],
            [100.0, 100.0, 1000.0, 100.0],
        ],
        dtype=float,
    )

    replay = replay_motor_commands_in_simengine(instance, controls, total_time_s=0.04, control_layout="rows")

    assert replay.steps == 4


def test_simengine_replay_ignores_roundoff_zero_length_command_boundary_steps():
    instance = _instance(backend_dt=0.005)
    controls = np.full((60, 4), 100.0, dtype=float)

    replay = replay_motor_commands_in_simengine(
        instance,
        controls,
        total_time_s=0.9593003864952115,
        control_layout="rows",
    )

    assert replay.steps > 0
    assert np.isfinite(replay.min_target_distance_m)


def test_rollout_requires_layout_for_ambiguous_four_by_four_controls():
    instance = _instance()
    controls = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],
            [500.0, 600.0, 700.0, 800.0],
            [900.0, 1000.0, 1100.0, 1200.0],
            [1300.0, 1400.0, 1500.0, 1600.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="ambiguous 4x4"):
        rollout_motor_commands(instance, controls, total_time_s=0.04)

    trajectory = rollout_motor_commands(instance, controls, total_time_s=0.04, control_layout="columns")

    assert trajectory.controls.shape == (4, 4)
    np.testing.assert_allclose(trajectory.controls[0], [100.0, 500.0, 900.0, 1000.0])
    np.testing.assert_allclose(trajectory.controls[-1], [400.0, 800.0, 1000.0, 1000.0])


def test_rollout_harness_acceptance_requires_position_accuracy():
    good = _acceptance_row(position_error=1.0e-7, caught=True)
    too_much_position_error = _acceptance_row(position_error=1.0e-4, caught=True)
    not_caught = _acceptance_row(position_error=1.0e-7, caught=False)

    assert _passed_acceptance(good) is True
    assert _passed_acceptance(too_much_position_error) is False
    assert _passed_acceptance(not_caught) is False


def test_fixed_time_feasibility_reports_replay_failure():
    instance = _instance()
    controls = np.full((2, 4), 100.0, dtype=float)

    result = solve_fixed_time(instance, 0.04, controls, control_layout="rows")

    assert result.seed == 1
    assert result.feasible is False
    assert result.caught is False
    assert result.failure_reason == "replay_not_caught"
    assert result.replay_steps > 0
    assert result.replay_min_distance_m > result.intercept_radius_m


def test_fixed_time_feasibility_does_not_accept_post_horizon_catch():
    instance = _instance(target_position=(0.1015, 0.0, 0.0), target_velocity=(-1.0, 0.0, 0.0), intercept_radius_m=0.1)
    controls = np.full((1, 4), 100.0, dtype=float)

    result = solve_fixed_time(instance, 0.001, controls, control_layout="rows")

    assert result.feasible is False
    assert result.caught is False
    assert result.failure_reason == "replay_not_caught"
    assert result.replay_steps == 1
    assert result.replay_min_distance_m > result.intercept_radius_m


def test_fixed_time_feasibility_accepts_short_horizon_catch():
    instance = _instance(target_position=(0.0, 0.0, -1.2e-5), intercept_radius_m=1.0e-5)
    controls = np.full((1, 4), 100.0, dtype=float)

    result = solve_fixed_time(instance, 0.001, controls, control_layout="rows")

    assert result.feasible is True
    assert result.caught is True
    assert result.failure_reason == ""
    assert result.replay_steps == 1
    assert result.replay_min_distance_m <= result.intercept_radius_m


def test_structured_update_validates_derivative_and_keeps_cost_non_worse():
    instance = _instance(target_position=(0.0, 0.0, -0.02), intercept_radius_m=0.05)
    controls = np.full((4, 4), 600.0, dtype=float)

    result = run_structured_update(
        instance,
        controls,
        0.04,
        control_layout="rows",
        config=StructuredUpdateConfig(
            active_window_nodes=2,
            finite_difference_rpm=2.0,
            max_update_rpm=10.0,
            line_search_alphas=(1.0, 0.5, 0.25),
        ),
    )

    assert result.accepted_cost <= result.initial_cost + 1.0e-12
    assert result.active_variables == 8
    assert result.direction_derivative_relative_error <= 1.0e-6
    assert result.replay_caught is True


def test_structured_update_rejects_invalid_finite_difference_step():
    instance = _instance()
    controls = np.full((2, 4), 500.0, dtype=float)

    with pytest.raises(ValueError, match="finite_difference_rpm"):
        run_structured_update(
            instance,
            controls,
            0.04,
            control_layout="rows",
            config=StructuredUpdateConfig(finite_difference_rpm=0.0),
        )


def test_structured_update_harness_rejects_noop_update_row():
    row = {
        "replay_caught": True,
        "structured_update_wall_s": 0.1,
        "initial_cost": 1.0,
        "accepted_cost": 1.0,
        "cost_delta": 0.0,
        "accepted_alpha": 0.0,
        "gradient_norm": 0.0,
        "gradient_abs_max": 0.0,
        "direction_derivative_relative_error": 0.0,
    }

    assert _structured_update_passed_acceptance(row) is False


def test_structured_update_rejects_nontrivial_target_behavior():
    instance = _instance(target_behavior=TargetBehaviorConfig(waypoints=(np.array([1.0, 0.0, 0.0]),), duration_s=1.0))
    controls = np.full((2, 4), 500.0, dtype=float)

    with pytest.raises(ValueError, match="waypoint target behavior"):
        run_structured_update(instance, controls, 0.04, control_layout="rows")


def test_time_search_serial_and_parallel_agree_on_fastest_probe():
    instance = _instance(target_position=(0.0, 0.0, -0.02), intercept_radius_m=0.05)
    controls = np.full((2, 4), 100.0, dtype=float)

    serial = find_fastest_intercept(
        instance,
        controls,
        0.04,
        time_multipliers=(0.5, 1.0),
        control_layout="rows",
        mode="serial",
    )
    parallel = find_fastest_intercept(
        instance,
        controls,
        0.04,
        time_multipliers=(0.5, 1.0),
        control_layout="rows",
        mode="parallel",
        workers=2,
    )

    assert serial.caught is True
    assert parallel.caught is True
    assert serial.fastest_caught_time_s == pytest.approx(parallel.fastest_caught_time_s)
    assert serial.probes_executed <= parallel.probes_executed
    assert parallel.probes_executed == 2


def test_time_search_rejects_invalid_time_multiplier():
    instance = _instance()
    controls = np.full((2, 4), 100.0, dtype=float)

    with pytest.raises(ValueError, match="time_multipliers"):
        find_fastest_intercept(instance, controls, 0.04, time_multipliers=(0.0,), control_layout="rows")


def test_switching_template_generates_low_dimensional_candidates():
    instance = _instance()
    config = SwitchingTemplateConfig(
        min_time_s=0.01,
        max_time_s=0.01,
        thrust_fractions=(1.0,),
        rate_fractions=(1.0,),
        first_switch_fractions=(0.25,),
        second_switch_fractions=(0.75,),
        counter_rate_fractions=(0.0,),
        vertical_bias_gains=(0.0,),
        direction_signs=(1.0,),
    )
    steps = int(round(float(config.min_time_s) / _backend_dt(instance)))

    candidates = _templates_for_steps(instance, config, steps, float(config.min_time_s))

    assert len(candidates) == 1
    assert candidates[0].steps == steps
    assert len(candidates[0].axis_b) == 3


def test_switching_template_harness_import_does_not_load_portfolio_stack():
    script = """
import json
import sys
import control_sims.optimizing_rpg.switching_template_harness  # noqa: F401
print(json.dumps({
    'casadi': 'casadi' in sys.modules,
    'portfolio_policy': 'control_sims.rpg_time_optimal.portfolio_policy' in sys.modules,
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = json.loads(completed.stdout)

    assert loaded == {"casadi": False, "portfolio_policy": False}


def test_switching_template_time_grid_keeps_float32_dt_endpoint():
    instance = _instance()
    dt = _backend_dt(instance)
    config = SwitchingTemplateConfig(min_time_s=dt, max_time_s=dt, time_step_s=dt)

    grid = _time_grid(instance, config)

    assert grid == ((1, dt),)


def test_switching_template_fast_replay_matches_exact_small_replay():
    instance = _instance(target_position=(0.0, 0.0, 0.0), intercept_radius_m=0.5)
    controls = np.full((2, 4), 200.0, dtype=float)

    fast = _fast_replay_backend_tick_commands(instance, controls, sample_dt_s=_backend_dt(instance))
    exact = replay_motor_commands_in_simengine(
        instance,
        controls,
        total_time_s=_backend_dt(instance) * len(controls),
        control_layout="rows",
    )

    assert fast.caught == exact.caught
    assert fast.min_distance_m == pytest.approx(exact.min_target_distance_m)
    assert fast.final_distance_m == pytest.approx(exact.final_target_distance_m)


def test_switching_template_config_validation_rejects_empty_or_nonfinite_options():
    instance = _instance()

    with pytest.raises(ValueError, match="thrust_fractions"):
        find_switching_template_intercept(instance, SwitchingTemplateConfig(thrust_fractions=()))

    with pytest.raises(ValueError, match="thrust_fractions"):
        find_switching_template_intercept(instance, SwitchingTemplateConfig(thrust_fractions=(float("nan"),)))


def test_switching_template_uses_backend_fallback_limits_for_zero_config_limits():
    instance = _instance()
    assert instance.config.max_thrust_n == 0.0
    assert instance.config.max_rate_rps == 0.0

    thrust = _effective_max_thrust_n(instance)
    rate = _effective_max_rate_rps(instance)
    commands = _ctbr_to_motor_speeds_batch(
        instance.config.pursuer,
        np.array([thrust], dtype=float),
        np.array([[rate, 0.0, 0.0]], dtype=float),
    )

    assert thrust == pytest.approx(2.0 * instance.config.pursuer.mass_kg * instance.config.pursuer.gravity_mps2)
    assert rate == pytest.approx(instance.config.pursuer.max_omega_rps)
    assert np.all(np.isfinite(commands))
    assert np.min(commands) >= instance.config.pursuer.rpm_min
    assert np.max(commands) <= instance.config.pursuer.max_rpm


def test_switching_template_rejects_non_constant_target_behavior():
    instance = _instance(target_behavior=TargetBehaviorConfig(waypoints=(np.array([1.0, 0.0, 0.0]),), duration_s=1.0))

    with pytest.raises(ValueError, match="constant-velocity"):
        find_switching_template_intercept(instance, SwitchingTemplateConfig())


def test_switching_template_summary_separates_attempted_and_catch_equivalent_speedups():
    rows = [
        {
            "caught": True,
            "error": "",
            "wall_s": 2.0,
            "portfolio_wall_s": 10.0,
            "portfolio_caught": "True",
        },
        {
            "caught": False,
            "error": "",
            "wall_s": 1.0,
            "portfolio_wall_s": 20.0,
            "portfolio_caught": "True",
        },
    ]

    summary = _switching_template_summary(SwitchingTemplateHarnessConfig(), rows)

    assert summary["catch_fraction"] == pytest.approx(0.5)
    assert summary["portfolio_vs_switching_attempted_total_speedup"] == pytest.approx(10.0)
    assert summary["catch_equivalent_num_scenarios"] == 1
    assert summary["catch_equivalent_portfolio_vs_switching_speedup"] == pytest.approx(5.0)


def test_switching_template_attempted_speedup_requires_full_portfolio_coverage():
    rows = [
        {"caught": True, "error": "", "wall_s": 2.0, "portfolio_wall_s": 10.0, "portfolio_caught": "True"},
        {"caught": True, "error": "", "wall_s": 2.0, "portfolio_wall_s": float("nan"), "portfolio_caught": ""},
    ]

    summary = _switching_template_summary(SwitchingTemplateHarnessConfig(), rows)

    assert summary["portfolio_covers_all_rows"] is False
    assert np.isnan(summary["portfolio_vs_switching_attempted_total_speedup"])


def test_switching_template_finds_near_initial_catch_without_ipopt():
    instance = _instance(target_position=(0.0, 0.0, 0.0), intercept_radius_m=0.5)

    result = find_switching_template_intercept(
        instance,
        SwitchingTemplateConfig(
            min_time_s=0.05,
            max_time_s=0.05,
            time_step_s=0.05,
            thrust_fractions=(0.7,),
            rate_fractions=(0.7,),
            first_switch_fractions=(0.25,),
            second_switch_fractions=(0.75,),
            counter_rate_fractions=(0.0,),
            vertical_bias_gains=(0.0,),
            direction_signs=(1.0,),
            replay_sample_dt_s=0.05,
        ),
    )

    assert result.caught is True
    assert result.templates_evaluated == 1
    assert result.best_candidate is not None


def test_time_search_harness_acceptance_rejects_missing_catch_seed():
    config = TimeSearchHarnessConfig(catch_seeds=(1, 2), time_multipliers=(0.5, 1.0))
    benchmark = _time_search_acceptance_row(seed=1)
    catch_rows = [_time_search_acceptance_row(seed=1)]

    assert _time_search_passed_acceptance(config, benchmark, catch_rows) is False


def test_time_search_harness_acceptance_rejects_probe_errors():
    config = TimeSearchHarnessConfig(catch_seeds=(1,), time_multipliers=(0.5, 1.0))
    benchmark = _time_search_acceptance_row(seed=1)
    catch = _time_search_acceptance_row(seed=1)
    catch["parallel_probe_errors"] = 1

    assert _time_search_passed_acceptance(config, benchmark, [catch]) is False


def test_final_harness_acceptance_requires_more_than_one_multi_scenario():
    single = {
        "error": "",
        "custom_serial_caught": True,
        "custom_parallel_caught": True,
        "fastest_times_match": True,
        "custom_parallel_probes": 2,
        "custom_parallel_workers": 2,
    }
    multi = [{"error": "", "parallel_caught": True}]

    assert _final_harness_passed_acceptance(single, multi) is False


def test_final_harness_acceptance_rejects_duplicate_multi_seed_rows():
    single = _final_single_acceptance_row()
    multi = [_final_multi_acceptance_row(seed=1), _final_multi_acceptance_row(seed=1)]

    assert _final_harness_passed_acceptance(single, multi) is False


def test_final_harness_acceptance_rejects_missed_multi_scenario():
    single = _final_single_acceptance_row()
    multi = [_final_multi_acceptance_row(seed=1), _final_multi_acceptance_row(seed=2, parallel_caught=False)]

    assert _final_harness_passed_acceptance(single, multi) is False


def test_final_harness_acceptance_rejects_single_worker_parallel_evidence():
    single = _final_single_acceptance_row()
    single["custom_parallel_workers"] = 1
    multi = [_final_multi_acceptance_row(seed=1), _final_multi_acceptance_row(seed=2)]

    assert _final_harness_passed_acceptance(single, multi) is False


def test_final_harness_rejects_workers_less_than_two(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        run_final_harness(FinalHarnessConfig(output_dir=tmp_path, multi_seeds=(1, 2), workers=1))


def _acceptance_row(*, position_error: float, caught: bool) -> dict[str, float | bool]:
    return {
        "simengine_replay_caught": caught,
        "mean_rollout_wall_s": 0.001,
        "position_error_max_m": position_error,
        "terminal_position_error_m": position_error,
    }


def _time_search_acceptance_row(seed: int) -> dict[str, object]:
    return {
        "seed": int(seed),
        "error": "",
        "serial_caught": True,
        "parallel_caught": True,
        "fastest_times_match": True,
        "serial_probes_executed": 1,
        "parallel_probes_executed": 2,
        "serial_probe_errors": 0,
        "parallel_probe_errors": 0,
    }


def _final_single_acceptance_row() -> dict[str, object]:
    return {
        "error": "",
        "custom_serial_caught": True,
        "custom_parallel_caught": True,
        "fastest_times_match": True,
        "custom_parallel_probes": 2,
        "custom_parallel_workers": 2,
    }


def _final_multi_acceptance_row(seed: int, *, parallel_caught: bool = True) -> dict[str, object]:
    return {
        "seed": int(seed),
        "error": "",
        "serial_caught": True,
        "parallel_caught": parallel_caught,
        "fastest_times_match": True,
        "serial_probe_errors": 0,
        "parallel_probe_errors": 0,
        "parallel_probes_executed": 2,
        "parallel_workers": 2,
    }


def _instance(
    *,
    target_position: tuple[float, float, float] = (1.0, 0.0, 0.0),
    target_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    intercept_radius_m: float = 0.1,
    backend_dt: float = 0.005,
    target_behavior: TargetBehaviorConfig | None = None,
) -> SimInstance:
    params = PursuerParams(
        mass_kg=0.027,
        ixx=3.85e-6,
        iyy=3.85e-6,
        izz=5.9675e-6,
        arm_len_m=0.0396,
        k_thrust=3.16e-10,
        k_yaw=0.005964552,
        max_rpm=1000.0,
        max_vel_mps=20.0,
        max_omega_rps=20.0,
        motor_tau_s=0.05,
        rpm_min=100.0,
    )
    config = SimConfig(
        pursuer=params,
        options=SimOptions(backend_dt=float(backend_dt), duration_s=0.05),
        targets=(TargetConfig(id="target", kind="target", radius_m=0.2, behavior=target_behavior or TargetBehaviorConfig()),),
        intercept_radius_m=float(intercept_radius_m),
    )
    return SimInstance(
        seed=1,
        pursuer_initial=PursuerInitialState(
            position_w=np.zeros(3),
            velocity_w=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            body_rates_b=np.zeros(3),
            rotor_speeds=np.full(4, 200.0),
        ),
        target_initials=(
            TargetInitialState(
                position_w=np.array(target_position, dtype=float),
                velocity_w=np.array(target_velocity, dtype=float),
            ),
        ),
        config=config,
    )
