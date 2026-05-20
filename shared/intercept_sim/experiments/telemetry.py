from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from intercept_sim.analysis import ExperimentMetrics
from intercept_sim.runner import RunnerStep
from intercept_sim.types import CameraCapture, ImageFeatureMeasurement, SimulationTarget


@dataclass(frozen=True)
class TargetTelemetry:
    id: str
    kind: str
    position_w: list[float]
    velocity_w: list[float]
    radius_m: float


@dataclass(frozen=True)
class CaptureTelemetry:
    t_capture: float
    camera_id: str
    target_id: str | None
    detected: bool
    uv_px: list[float] | None
    uv_norm: list[float] | None
    target_pos_c: list[float] | None
    range_m: float | None
    apparent_radius_px: float | None


@dataclass(frozen=True)
class ImageFeatureTelemetry:
    t_capture: float
    t_available: float
    camera_id: str
    target_id: str | None
    detected: bool
    uv_px: list[float] | None
    uv_norm: list[float] | None
    confidence: float


@dataclass(frozen=True)
class TelemetryStep:
    t: float
    pursuer_position_w: list[float]
    pursuer_velocity_w: list[float]
    pursuer_rotation_wb: list[list[float]]
    target_states: list[TargetTelemetry]
    capture: CaptureTelemetry | None
    measurements: list[ImageFeatureTelemetry]
    observer_feature: ImageFeatureTelemetry | None
    observer_relative_position_w: list[float] | None
    observer_relative_velocity_w: list[float] | None
    command_thrust_n: float
    command_body_rates_b: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentTelemetry:
    experiment_id: str
    comment: str
    config: dict[str, Any]
    summary_metrics: ExperimentMetrics
    steps: list[TelemetryStep]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment_id,
            "comment": self.comment,
            "duration_s": float(self.config["sim"]["duration_s"]),
            "dt": float(self.config["sim"]["dt"]),
            "steps": len(self.steps),
            "metrics": self.summary_metrics.to_dict(),
        }


def build_experiment_telemetry(
    *,
    experiment_id: str,
    comment: str,
    config: dict[str, Any],
    metrics: ExperimentMetrics,
    log: list[RunnerStep],
) -> ExperimentTelemetry:
    return ExperimentTelemetry(
        experiment_id=experiment_id,
        comment=str(comment),
        config=config,
        summary_metrics=metrics,
        steps=[telemetry_step_from_runner_step(step) for step in log],
    )


def telemetry_step_from_runner_step(step: RunnerStep) -> TelemetryStep:
    observer_state = step.observer_state
    return TelemetryStep(
        t=float(step.t),
        pursuer_position_w=_array_list(step.scene.pursuer.position_w),
        pursuer_velocity_w=_array_list(step.scene.pursuer.velocity_w),
        pursuer_rotation_wb=np.asarray(step.scene.pursuer.rotation_wb, dtype=float).tolist(),
        target_states=[target_telemetry(target) for target in step.scene.targets],
        capture=capture_telemetry(step.capture),
        measurements=[feature_telemetry(measurement) for measurement in step.measurements],
        observer_feature=feature_telemetry(observer_state.image_feature),
        observer_relative_position_w=_array_list_or_none(observer_state.relative_position_w),
        observer_relative_velocity_w=_array_list_or_none(observer_state.relative_velocity_w),
        command_thrust_n=float(step.command.thrust_n),
        command_body_rates_b=_array_list(step.command.body_rates_b),
    )


def target_telemetry(target: SimulationTarget) -> TargetTelemetry:
    return TargetTelemetry(
        id=target.id,
        kind=target.kind,
        position_w=_array_list(target.position_w),
        velocity_w=_array_list(target.velocity_w),
        radius_m=float(target.radius_m),
    )


def capture_telemetry(capture: CameraCapture | None) -> CaptureTelemetry | None:
    if capture is None:
        return None
    return CaptureTelemetry(
        t_capture=float(capture.t_capture),
        camera_id=capture.camera_id,
        target_id=capture.target_id,
        detected=bool(capture.detected),
        uv_px=_array_list_or_none(capture.uv_px),
        uv_norm=_array_list_or_none(capture.uv_norm),
        target_pos_c=_array_list_or_none(capture.target_pos_c),
        range_m=None if capture.range_m is None else float(capture.range_m),
        apparent_radius_px=None if capture.apparent_radius_px is None else float(capture.apparent_radius_px),
    )


def feature_telemetry(feature: ImageFeatureMeasurement | None) -> ImageFeatureTelemetry | None:
    if feature is None:
        return None
    return ImageFeatureTelemetry(
        t_capture=float(feature.t_capture),
        t_available=float(feature.t_available),
        camera_id=feature.camera_id,
        target_id=feature.target_id,
        detected=bool(feature.detected),
        uv_px=_array_list_or_none(feature.uv_px),
        uv_norm=_array_list_or_none(feature.uv_norm),
        confidence=float(feature.confidence),
    )


def _array_list(value: np.ndarray) -> list[float]:
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def _array_list_or_none(value: np.ndarray | None) -> list[float] | None:
    if value is None:
        return None
    return _array_list(value)
