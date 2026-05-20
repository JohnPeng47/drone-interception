from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from intercept_sim.runner import RunnerStep


@dataclass(frozen=True)
class ExperimentMetrics:
    min_distance_m: float
    final_distance_m: float
    catch_time_s: float | None
    target_visible_fraction: float
    image_feature_availability_fraction: float
    average_image_error_norm: float | None
    miss_distance_m: float

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def compute_metrics(log: list[RunnerStep], *, catch_radius_m: float) -> ExperimentMetrics:
    if not log:
        return ExperimentMetrics(
            min_distance_m=float("nan"),
            final_distance_m=float("nan"),
            catch_time_s=None,
            target_visible_fraction=0.0,
            image_feature_availability_fraction=0.0,
            average_image_error_norm=None,
            miss_distance_m=float("nan"),
        )

    distances = np.array([_distance_to_primary_target(step) for step in log], dtype=float)
    catch_indices = np.flatnonzero(distances <= catch_radius_m)
    catch_time = float(log[int(catch_indices[0])].t) if catch_indices.size else None

    captures = [step.capture for step in log if step.capture is not None]
    visible_count = sum(1 for capture in captures if capture.detected)
    target_visible_fraction = visible_count / len(captures) if captures else 0.0

    features = [step.observer_state.image_feature for step in log]
    available_features = [
        feature for feature in features if feature is not None and feature.detected and feature.uv_norm is not None
    ]
    image_feature_availability_fraction = len(available_features) / len(log)
    if available_features:
        average_image_error_norm = float(
            np.mean([np.linalg.norm(np.asarray(feature.uv_norm, dtype=float)) for feature in available_features])
        )
    else:
        average_image_error_norm = None

    return ExperimentMetrics(
        min_distance_m=float(np.min(distances)),
        final_distance_m=float(distances[-1]),
        catch_time_s=catch_time,
        target_visible_fraction=float(target_visible_fraction),
        image_feature_availability_fraction=float(image_feature_availability_fraction),
        average_image_error_norm=average_image_error_norm,
        miss_distance_m=float(np.min(distances)),
    )


def circular_error_probable(miss_distances_m: list[float] | np.ndarray, *, percentile: float = 50.0) -> float:
    distances = np.asarray(miss_distances_m, dtype=float)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, percentile))


def _distance_to_primary_target(step: RunnerStep) -> float:
    if not step.scene.targets:
        return float("nan")
    pursuer_pos = np.asarray(step.scene.pursuer.position_w, dtype=float)
    target_pos = np.asarray(step.scene.targets[0].position_w, dtype=float)
    return float(np.linalg.norm(target_pos - pursuer_pos))
