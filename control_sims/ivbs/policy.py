from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from backends.csim.bindings.types import SimInstance
from backends.csim.runner import CtbrCommandBatch, SimControlPolicy, SimRunnerState
from control_sims.beihang_paper_sim.controller.control_math import DEFAULT_GAINS

from .control_law import bearing_error_rad, beihang_command_from_estimate, cautious_bearing_command
from .cv_detection import TraditionalCvMeasurement, missed_measurement
from .observer import VisualObserverConfig, VisualRelativeStateObserver


ImageMeasurementProvider = Callable[[int], TraditionalCvMeasurement | None]


class IVBSControlPolicy(SimControlPolicy):
    """Beihang-style IVBS controller driven by visual relative-state estimates."""

    def __init__(
        self,
        gains: Mapping[str, float] | None = None,
        observer_config: VisualObserverConfig | Mapping[str, float] | None = None,
        record_telemetry: bool = False,
        image_measurement_provider: ImageMeasurementProvider | None = None,
    ):
        self._gains = {
            **DEFAULT_GAINS,
            "k_b": 0.65,
            "cautious_closing_accel_mps2": 3.0,
            "cautious_velocity_damping": 0.25,
            **dict(gains or {}),
        }
        self._observer = VisualRelativeStateObserver(observer_config)
        self._record_telemetry = bool(record_telemetry)
        self._image_measurement_provider = image_measurement_provider
        self.telemetry_rows: list[dict[str, float | int | str | bool]] = []

    def reset(self, state: SimRunnerState) -> None:
        self._observer.reset()
        self.telemetry_rows.clear()

    def on_slots_started(
        self,
        slots: np.ndarray,
        instances: Sequence[SimInstance],
        state: SimRunnerState,
    ) -> None:
        slot_array = np.asarray(slots, dtype=np.int64).reshape(-1)
        snapshots = tuple(state.snapshot[int(slot)] for slot in slot_array)
        image_measurements = {int(slot): self._image_measurement(int(slot)) for slot in slot_array}
        self._observer.start_slots(slots, instances, snapshots, image_measurements=image_measurements)

    def command(self, state: SimRunnerState) -> CtbrCommandBatch:
        thrust_n = np.zeros(len(state.instances), dtype=np.float32)
        body_rates_b = np.zeros((len(state.instances), 3), dtype=np.float32)
        for slot, instance in enumerate(state.instances):
            if instance is None or not bool(state.active[slot]):
                self._observer.stop_slot(slot)
                continue
            snapshot = state.snapshot[slot]
            image_measurement = self._image_measurement(slot)
            estimate = self._observer.estimate(
                slot,
                instance,
                snapshot,
                t_s=float(state.elapsed_s[slot]),
                image_measurement=image_measurement,
            )
            if estimate.metric_confident:
                mode = "metric"
                command = beihang_command_from_estimate(instance, snapshot, estimate, self._gains)
            else:
                mode = "bearing_fallback" if estimate.valid else "hover"
                command = cautious_bearing_command(instance, snapshot, estimate, self._gains)
            thrust_n[slot] = np.float32(command[0])
            body_rates_b[slot] = np.asarray(command[1], dtype=np.float32).reshape(3)
            if self._record_telemetry:
                self._append_telemetry_row(
                    state,
                    slot,
                    instance,
                    mode,
                    estimate,
                    float(command[0]),
                    np.asarray(command[1], dtype=float).reshape(3),
                    image_measurement,
                )
        return CtbrCommandBatch(thrust_n=thrust_n, body_rates_b=body_rates_b)

    def _append_telemetry_row(
        self,
        state: SimRunnerState,
        slot: int,
        instance: SimInstance,
        mode: str,
        estimate,
        thrust_n: float,
        body_rates_b: np.ndarray,
        image_measurement: TraditionalCvMeasurement | None,
    ) -> None:
        snapshot = state.snapshot[slot]
        detected = bool(snapshot.camera.detected) if image_measurement is None else bool(image_measurement.detected)
        self.telemetry_rows.append(
            {
                "seed": int(instance.seed),
                "slot": int(slot),
                "workload_index": int(state.workload_indices[slot]),
                "tick": int(state.steps[slot]),
                "t_s": float(state.elapsed_s[slot]),
                "mode": str(mode),
                "detected": detected,
                "valid": bool(estimate.valid),
                "metric_confident": bool(estimate.metric_confident),
                "stale_s": float(estimate.stale_s),
                "detection_count": int(estimate.detection_count),
                "estimated_range_m": float(estimate.estimated_range_m),
                "range_std_m": float(estimate.range_std_m),
                "position_std_m": float(estimate.position_std_m),
                "velocity_std_mps": float(estimate.velocity_std_m),
                "bearing_error_rad": bearing_error_rad(instance, snapshot, estimate),
                "thrust_n": float(thrust_n),
                "body_rate_norm": float(np.linalg.norm(body_rates_b)),
            }
        )

    def _image_measurement(
        self,
        slot: int,
    ) -> TraditionalCvMeasurement | None:
        if self._image_measurement_provider is None:
            return None
        measurement = self._image_measurement_provider(int(slot))
        return missed_measurement() if measurement is None else measurement
