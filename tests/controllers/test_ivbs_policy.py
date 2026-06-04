from __future__ import annotations

from dataclasses import replace

import numpy as np

from backends.csim.bindings.types import SimSnapshots
from backends.csim.bindings.types.target_sim import TargetInitialState
from backends.csim.runner import SimRunner
from control_sims.ivbs.cv_detection import TraditionalCvMeasurement, detect_dark_blob
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


def test_ivbs_metric_command_does_not_depend_on_forbidden_truth():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = runner.reset([instance])
    altered_instance, altered_state = _alter_forbidden_truth(instance, state)
    observer_config = {
        "min_detections_for_metric": 0,
        "metric_range_std_threshold_m": 99.0,
        "metric_position_std_threshold_m": 99.0,
        "metric_velocity_std_threshold_mps": 99.0,
    }
    policy_a = IVBSControlPolicy(observer_config=observer_config, record_telemetry=True)
    policy_b = IVBSControlPolicy(observer_config=observer_config, record_telemetry=True)
    policy_a.reset(state)
    policy_b.reset(altered_state)
    policy_a.on_slots_started(np.array([0], dtype=np.int64), (instance,), state)
    policy_b.on_slots_started(np.array([0], dtype=np.int64), (altered_instance,), altered_state)

    command_a = policy_a.command(state)
    command_b = policy_b.command(altered_state)

    assert policy_a.telemetry_rows[-1]["mode"] == "metric"
    assert policy_b.telemetry_rows[-1]["mode"] == "metric"
    np.testing.assert_allclose(command_a.thrust_n, command_b.thrust_n, atol=0.0)
    np.testing.assert_allclose(command_a.body_rates_b, command_b.body_rates_b, atol=0.0)
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


def test_traditional_cv_detects_dark_blob_and_apparent_size():
    frame = np.full((80, 100, 3), 220, dtype=np.uint8)
    yy, xx = np.ogrid[:80, :100]
    mask = (yy - 30) ** 2 + (xx - 60) ** 2 <= 8 ** 2
    frame[mask] = np.array([15, 15, 15], dtype=np.uint8)

    measurement = detect_dark_blob(frame, fx_px=50.0, fy_px=50.0, cx_px=50.0, cy_px=40.0)

    assert measurement.detected
    np.testing.assert_allclose(measurement.uv_norm, np.array([0.2, -0.2]), atol=0.03)
    assert 6.0 <= measurement.apparent_radius_px <= 9.5
    assert measurement.confidence > 0.0


def test_traditional_cv_rejects_invalid_intrinsics():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    measurement = detect_dark_blob(frame, fx_px=0.0, fy_px=50.0, cx_px=4.0, cy_px=4.0)

    assert not measurement.detected
    np.testing.assert_allclose(measurement.uv_norm, np.zeros(2))

    measurement = detect_dark_blob(frame, fx_px=50.0, fy_px=50.0, cx_px=float("nan"), cy_px=4.0)

    assert not measurement.detected
    np.testing.assert_allclose(measurement.uv_norm, np.zeros(2))

    measurement = detect_dark_blob(frame, fx_px=50.0, fy_px=-1.0, cx_px=4.0, cy_px=4.0)

    assert not measurement.detected
    np.testing.assert_allclose(measurement.uv_norm, np.zeros(2))

    measurement = detect_dark_blob(frame, fx_px=50.0, fy_px=50.0, cx_px=4.0, cy_px=float("inf"))

    assert not measurement.detected
    np.testing.assert_allclose(measurement.uv_norm, np.zeros(2))


def test_ivbs_apparent_size_measurement_can_change_command_without_truth():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = runner.reset([instance])
    altered_instance, altered_state = _alter_forbidden_truth(instance, state)
    uv_norm = np.asarray(state.snapshot[0].camera.uv_norm, dtype=float).reshape(2)
    near_provider = _fixed_measurement_provider(uv_norm=uv_norm, apparent_radius_px=30.0)
    far_provider = _fixed_measurement_provider(uv_norm=uv_norm, apparent_radius_px=5.0)
    policy_near_a = IVBSControlPolicy(
        image_measurement_provider=near_provider,
        observer_config=_metric_test_observer_config(),
    )
    policy_near_b = IVBSControlPolicy(
        image_measurement_provider=near_provider,
        observer_config=_metric_test_observer_config(),
    )
    policy_far = IVBSControlPolicy(
        image_measurement_provider=far_provider,
        observer_config=_metric_test_observer_config(),
    )
    for policy, policy_state, policy_instance in (
        (policy_near_a, state, instance),
        (policy_near_b, altered_state, altered_instance),
        (policy_far, state, instance),
    ):
        policy.reset(policy_state)
        policy.on_slots_started(np.array([0], dtype=np.int64), (policy_instance,), policy_state)

    command_near_a = policy_near_a.command(state)
    command_near_b = policy_near_b.command(altered_state)
    command_far = policy_far.command(state)

    np.testing.assert_allclose(command_near_a.thrust_n, command_near_b.thrust_n, atol=0.0)
    np.testing.assert_allclose(command_near_a.body_rates_b, command_near_b.body_rates_b, atol=0.0)
    assert not np.allclose(command_near_a.thrust_n, command_far.thrust_n)


def test_ivbs_pixel_measurement_provider_runs_through_policy():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = runner.reset([instance])
    frame = np.full((80, 100, 3), 220, dtype=np.uint8)
    yy, xx = np.ogrid[:80, :100]
    frame[(yy - 40) ** 2 + (xx - 50) ** 2 <= 8 ** 2] = np.array([10, 10, 10], dtype=np.uint8)

    def provider(slot):
        assert slot == 0
        return detect_dark_blob(frame, fx_px=50.0, fy_px=50.0, cx_px=50.0, cy_px=40.0)

    policy = IVBSControlPolicy(
        image_measurement_provider=provider,
        observer_config=_metric_test_observer_config(),
    )
    policy.reset(state)
    policy.on_slots_started(np.array([0], dtype=np.int64), (instance,), state)

    command = policy.command(state)

    assert np.all(np.isfinite(command.thrust_n))
    assert np.all(np.isfinite(command.body_rates_b))


def test_ivbs_cv_miss_does_not_fall_back_to_snapshot_camera_bearing():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = runner.reset([instance])
    observer = VisualRelativeStateObserver()
    initial_measurement = TraditionalCvMeasurement(
        detected=True,
        uv_norm=np.array([0.0, 0.0], dtype=float),
        apparent_radius_px=10.0,
        confidence=1.0,
    )
    observer.start_slots(
        np.array([0], dtype=np.int64),
        (instance,),
        (state.snapshot[0],),
        image_measurements={0: initial_measurement},
    )
    misleading_state = _state_with_camera(state, detected=True, uv_norm=(0.8, -0.8))
    missed_measurement = TraditionalCvMeasurement(
        detected=False,
        uv_norm=np.zeros(2, dtype=float),
        apparent_radius_px=0.0,
        confidence=0.0,
    )

    estimate = observer.estimate(
        0,
        instance,
        misleading_state.snapshot[0],
        t_s=0.1,
        image_measurement=missed_measurement,
    )

    snapshot_camera_bearing = observer._slots[0]._bearing_from_snapshot(misleading_state.snapshot[0])
    predicted_bearing = observer._slots[0]._bearing_to_target(
        misleading_state.snapshot[0],
        observer._slots[0].x[0:3],
    )
    assert snapshot_camera_bearing is not None
    assert predicted_bearing is not None
    assert not np.allclose(estimate.bearing_w, snapshot_camera_bearing)
    np.testing.assert_allclose(estimate.bearing_w, predicted_bearing)


def test_ivbs_nonfinite_cv_uv_does_not_fall_back_to_snapshot_camera_bearing():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = runner.reset([instance])
    observer = VisualRelativeStateObserver()
    observer.start_slots(
        np.array([0], dtype=np.int64),
        (instance,),
        (state.snapshot[0],),
        image_measurements={
            0: TraditionalCvMeasurement(
                detected=True,
                uv_norm=np.array([0.0, 0.0], dtype=float),
                apparent_radius_px=10.0,
                confidence=1.0,
            )
        },
    )
    misleading_state = _state_with_camera(state, detected=True, uv_norm=(0.8, -0.8))
    invalid_measurement = TraditionalCvMeasurement(
        detected=True,
        uv_norm=np.array([np.nan, 0.0], dtype=float),
        apparent_radius_px=10.0,
        confidence=1.0,
    )

    estimate = observer.estimate(
        0,
        instance,
        misleading_state.snapshot[0],
        t_s=0.1,
        image_measurement=invalid_measurement,
    )

    snapshot_camera_bearing = observer._slots[0]._bearing_from_snapshot(misleading_state.snapshot[0])
    assert snapshot_camera_bearing is not None
    assert not np.allclose(estimate.bearing_w, snapshot_camera_bearing)


def test_ivbs_startup_none_measurement_does_not_fall_back_to_snapshot_camera_bearing():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = _state_with_camera(runner.reset([instance]), detected=True, uv_norm=(0.7, -0.6))
    observer = VisualRelativeStateObserver()

    observer.start_slots(
        np.array([0], dtype=np.int64),
        (instance,),
        (state.snapshot[0],),
        image_measurements={0: None},
    )

    assert observer._slots[0].x is None


def test_ivbs_provider_none_does_not_fall_back_to_snapshot_camera_on_startup():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = _state_with_camera(runner.reset([instance]), detected=True, uv_norm=(0.7, -0.6))
    policy = IVBSControlPolicy(image_measurement_provider=lambda slot: None, record_telemetry=True)
    policy.reset(state)

    policy.on_slots_started(np.array([0], dtype=np.int64), (instance,), state)
    command = policy.command(state)

    assert policy._observer._slots[0].x is None
    assert policy.telemetry_rows[-1]["detected"] is False
    assert policy.telemetry_rows[-1]["mode"] == "hover"
    assert np.all(np.isfinite(command.thrust_n))
    assert np.all(np.isfinite(command.body_rates_b))


def test_ivbs_provider_none_does_not_fall_back_to_snapshot_camera_on_command_tick():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = runner.reset([instance])
    misleading_state = _state_with_camera(state, detected=True, uv_norm=(0.8, -0.8))
    missing_camera_state = _state_with_camera(state, detected=False, uv_norm=(0.0, 0.0))
    uv_norm = np.asarray(state.snapshot[0].camera.uv_norm, dtype=float).reshape(2)

    policy_misleading = IVBSControlPolicy(
        image_measurement_provider=_sequence_measurement_provider(
            TraditionalCvMeasurement(True, uv_norm, 10.0, 1.0),
            None,
        ),
        record_telemetry=True,
    )
    policy_missing = IVBSControlPolicy(
        image_measurement_provider=_sequence_measurement_provider(
            TraditionalCvMeasurement(True, uv_norm, 10.0, 1.0),
            None,
        ),
        record_telemetry=True,
    )
    for policy in (policy_misleading, policy_missing):
        policy.reset(state)
        policy.on_slots_started(np.array([0], dtype=np.int64), (instance,), state)

    command_misleading = policy_misleading.command(misleading_state)
    command_missing = policy_missing.command(missing_camera_state)

    np.testing.assert_allclose(command_misleading.thrust_n, command_missing.thrust_n, atol=0.0)
    np.testing.assert_allclose(command_misleading.body_rates_b, command_missing.body_rates_b, atol=0.0)
    assert policy_misleading.telemetry_rows[-1]["detected"] is False
    assert policy_missing.telemetry_rows[-1]["detected"] is False


def test_ivbs_provider_detection_drives_telemetry_when_camera_disagrees():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = _state_with_camera(runner.reset([instance]), detected=False, uv_norm=(0.0, 0.0))
    measurement = TraditionalCvMeasurement(
        detected=True,
        uv_norm=np.zeros(2, dtype=float),
        apparent_radius_px=10.0,
        confidence=1.0,
    )
    policy = IVBSControlPolicy(image_measurement_provider=lambda slot: measurement, record_telemetry=True)
    policy.reset(state)
    policy.on_slots_started(np.array([0], dtype=np.int64), (instance,), state)

    policy.command(state)

    assert policy.telemetry_rows[-1]["detected"] is True


def test_ivbs_apparent_size_measurement_reduces_range_covariance():
    instance = read_six_robust_intercept_samples()[0]
    runner = SimRunner(max_envs=1)
    state = runner.reset([instance])
    observer_without_size = VisualRelativeStateObserver()
    observer_with_size = VisualRelativeStateObserver()
    observer_without_size.start_slots(np.array([0], dtype=np.int64), (instance,), (state.snapshot[0],))
    observer_with_size.start_slots(np.array([0], dtype=np.int64), (instance,), (state.snapshot[0],))
    uv_norm = np.asarray(state.snapshot[0].camera.uv_norm, dtype=float).reshape(2)
    bearing_only = TraditionalCvMeasurement(
        detected=True,
        uv_norm=uv_norm,
        apparent_radius_px=10.0,
        confidence=0.0,
    )
    with_size = TraditionalCvMeasurement(
        detected=True,
        uv_norm=uv_norm,
        apparent_radius_px=10.0,
        confidence=1.0,
    )

    estimate_without_size = observer_without_size.estimate(
        0,
        instance,
        state.snapshot[0],
        t_s=0.0,
        image_measurement=bearing_only,
    )
    estimate_with_size = observer_with_size.estimate(
        0,
        instance,
        state.snapshot[0],
        t_s=0.0,
        image_measurement=with_size,
    )

    assert estimate_with_size.range_std_m < estimate_without_size.range_std_m


def _first_command_for_state(instance, state):
    policy = IVBSControlPolicy()
    policy.reset(state)
    policy.on_slots_started(np.array([0], dtype=np.int64), (instance,), state)
    return policy.command(state)


def _metric_test_observer_config():
    return {
        "min_detections_for_metric": 0,
        "metric_range_std_threshold_m": 99.0,
        "metric_position_std_threshold_m": 99.0,
        "metric_velocity_std_threshold_mps": 99.0,
    }


def _fixed_measurement_provider(*, uv_norm: np.ndarray, apparent_radius_px: float):
    def provider(slot):
        return TraditionalCvMeasurement(
            detected=True,
            uv_norm=np.asarray(uv_norm, dtype=float).reshape(2),
            apparent_radius_px=float(apparent_radius_px),
            confidence=1.0,
        )

    return provider


def _sequence_measurement_provider(*measurements):
    items = list(measurements)

    def provider(slot):
        assert slot == 0
        if items:
            return items.pop(0)
        return None

    return provider


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
