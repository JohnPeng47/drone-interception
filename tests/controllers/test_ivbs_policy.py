from __future__ import annotations

from dataclasses import replace

import numpy as np

from backends.csim.bindings.types import SimSnapshots
from backends.csim.bindings.types.target_sim import TargetInitialState
from backends.csim.runner import SimRunner
from control_sims.ivbs.observer import VisualObserverConfig, VisualRelativeStateObserver
from control_sims.ivbs.policy import IVBSControlPolicy
from control_sims.runner import _run_instances
from _robust_intercept_cases import read_six_robust_intercept_samples


def test_ivbs_command_does_not_depend_on_snapshot_target_truth():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = runner.reset([instance])

    altered_instance, altered_state = _alter_forbidden_truth(instance, state)

    command_a = _first_command_for_state(instance, state)
    command_b = _first_command_for_state(altered_instance, altered_state)

    np.testing.assert_allclose(command_a.thrust_n, command_b.thrust_n, atol=0.0)
    np.testing.assert_allclose(command_a.body_rates_b, command_b.body_rates_b, atol=0.0)


def test_ivbs_command_does_not_depend_on_forbidden_truth_after_step():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = runner.reset([instance])
    altered_instance, altered_state = _alter_forbidden_truth(instance, state)
    policy_a = IVBSControlPolicy(record_telemetry=True)
    policy_b = IVBSControlPolicy(record_telemetry=True)
    policy_a.reset(state)
    policy_b.reset(altered_state)
    policy_a.on_slots_started(np.array([0], dtype=np.int64), (instance,), state)
    policy_b.on_slots_started(np.array([0], dtype=np.int64), (altered_instance,), altered_state)

    command_a = policy_a.command(state)
    command_b = policy_b.command(altered_state)
    np.testing.assert_allclose(command_a.thrust_n, command_b.thrust_n, atol=0.0)
    np.testing.assert_allclose(command_a.body_rates_b, command_b.body_rates_b, atol=0.0)
    assert policy_a.telemetry_rows[-1] == policy_b.telemetry_rows[-1]

    next_state = runner.step(command_a).state
    altered_next_instance, altered_next_state = _alter_forbidden_truth(instance, next_state)
    next_command_a = policy_a.command(next_state)
    next_command_b = policy_b.command(altered_next_state)

    np.testing.assert_allclose(next_command_a.thrust_n, next_command_b.thrust_n, atol=0.0)
    np.testing.assert_allclose(next_command_a.body_rates_b, next_command_b.body_rates_b, atol=0.0)
    assert policy_a.telemetry_rows[-1] == policy_b.telemetry_rows[-1]


def test_ivbs_controller_runs_generated_scenarios_without_errors():
    instances = read_six_robust_intercept_samples()[:2]
    result = _run_instances(
        list(instances),
        "ivbs",
        IVBSControlPolicy,
        max_envs=1,
        log_snapshots=False,
        snapshot_log_rate=100,
    )
    assert len(result["rows"]) == len(instances)
    assert all(row["error"] is None for row in result["rows"])


def test_ivbs_prior_only_estimate_is_not_metric_confident():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = runner.reset([instance])
    observer = VisualRelativeStateObserver()
    observer.start_slots(np.array([0], dtype=np.int64), (instance,), (state.snapshot[0],))

    estimate = observer.estimate(0, instance, state.snapshot[0], t_s=0.0)

    assert estimate.valid
    assert not estimate.metric_confident


def test_ivbs_failed_projection_does_not_refresh_detection_liveness():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = runner.reset([instance])
    observer = VisualRelativeStateObserver(VisualObserverConfig(stale_timeout_s=0.1))
    observer.start_slots(np.array([0], dtype=np.int64), (instance,), (state.snapshot[0],))
    observer.estimate(0, instance, state.snapshot[0], t_s=0.0)
    broken_state = _state_with_camera(state, detected=True, uv_norm=(0.0, 0.0))
    slot_observer = observer._slots[0]
    assert slot_observer.x is not None
    slot_observer.x[0:3] = np.asarray(state.snapshot[0].pursuer.position_w, dtype=float)

    estimate = observer.estimate(0, instance, broken_state.snapshot[0], t_s=1.0)

    assert not estimate.valid
    measured_bearing = slot_observer._bearing_from_snapshot(broken_state.snapshot[0])
    np.testing.assert_allclose(estimate.bearing_w, measured_bearing)


def test_ivbs_bearing_updates_from_prediction_when_detection_is_lost():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = runner.reset([instance])
    observer = VisualRelativeStateObserver()
    observer.start_slots(np.array([0], dtype=np.int64), (instance,), (state.snapshot[0],))
    initial = observer.estimate(0, instance, state.snapshot[0], t_s=0.0)
    moved_state = _state_with_pursuer_position(
        _state_with_camera(state, detected=False, uv_norm=(0.0, 0.0)),
        np.asarray(state.snapshot[0].pursuer.position_w, dtype=float) + np.array([0.0, 1.0, 0.0]),
    )

    predicted = observer.estimate(0, instance, moved_state.snapshot[0], t_s=0.1)

    assert predicted.valid
    assert not np.allclose(predicted.bearing_w, initial.bearing_w)


def test_ivbs_policy_records_command_telemetry():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = runner.reset([instance])
    policy = IVBSControlPolicy(record_telemetry=True)
    policy.reset(state)
    policy.on_slots_started(np.array([0], dtype=np.int64), (instance,), state)

    policy.command(state)

    assert len(policy.telemetry_rows) == 1
    row = policy.telemetry_rows[0]
    assert row["seed"] == int(instance.seed)
    assert row["mode"] in {"metric", "bearing_fallback", "hover"}
    assert "estimated_range_m" in row
    assert "range_std_m" in row
    assert "bearing_error_rad" in row


def _first_command_for_state(instance, state):
    policy = IVBSControlPolicy()
    policy.reset(state)
    policy.on_slots_started(np.array([0], dtype=np.int64), (instance,), state)
    return policy.command(state)


def _alter_forbidden_truth(instance, state):
    arrays = state.snapshot.arrays
    target = arrays.target.copy()
    target[0, 0:3] += np.array([20.0, -10.0, 5.0], dtype=np.float32)
    target[0, 3:6] += np.array([-3.0, 4.0, 2.0], dtype=np.float32)
    metrics = arrays.metrics.copy()
    metrics[0, :] = np.array([123.0, 122.0, 1.0, 0.25, 0.0], dtype=np.float32)
    altered_snapshots = SimSnapshots.from_arrays(
        arrays.pursuer,
        target,
        metrics,
        arrays.camera,
        arrays.max_rate_rps,
        arrays.max_rpm,
        body_rates_b=arrays.body_rates_b,
        thrust_n=arrays.thrust_n,
    )
    altered_initial = TargetInitialState(
        position_w=np.asarray(instance.target_initial.position_w, dtype=float) + np.array([30.0, -20.0, 10.0]),
        velocity_w=np.asarray(instance.target_initial.velocity_w, dtype=float) + np.array([-4.0, 3.0, 2.0]),
    )
    altered_instance = replace(instance, target_initials=(altered_initial,))
    altered_instances = tuple(altered_instance if item is instance else item for item in state.instances)
    return altered_instance, replace(state, snapshot=altered_snapshots, instances=altered_instances)


def _state_with_camera(state, *, detected: bool, uv_norm: tuple[float, float]):
    arrays = state.snapshot.arrays
    camera = arrays.camera.copy()
    camera[0, 0] = 1.0 if detected else 0.0
    camera[0, 1:3] = np.asarray(uv_norm, dtype=np.float32)
    snapshots = SimSnapshots.from_arrays(
        arrays.pursuer,
        arrays.target,
        arrays.metrics,
        camera,
        arrays.max_rate_rps,
        arrays.max_rpm,
        body_rates_b=arrays.body_rates_b,
        thrust_n=arrays.thrust_n,
    )
    return replace(state, snapshot=snapshots)


def _state_with_pursuer_position(state, position_w):
    arrays = state.snapshot.arrays
    pursuer = arrays.pursuer.copy()
    pursuer[0, 0:3] = np.asarray(position_w, dtype=np.float32)
    snapshots = SimSnapshots.from_arrays(
        pursuer,
        arrays.target,
        arrays.metrics,
        arrays.camera,
        arrays.max_rate_rps,
        arrays.max_rpm,
        body_rates_b=arrays.body_rates_b,
        thrust_n=arrays.thrust_n,
    )
    return replace(state, snapshot=snapshots)
