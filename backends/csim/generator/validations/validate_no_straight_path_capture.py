from __future__ import annotations

import math

import numpy as np

from backends.csim.bindings.types import SimInstance


def validate_no_straight_path_capture(instance: SimInstance) -> None:
    """Reject scenarios already captured by current straight-line motion."""
    if instance.config is None:
        raise ValueError("straight-path capture validation requires SimInstance.config")
    if not instance.target_initials:
        raise ValueError("straight-path capture validation requires at least one target")

    radius_m = float(instance.config.intercept_radius_m)
    horizon_s = float(instance.config.options.duration_s)
    if radius_m <= 0.0:
        raise ValueError(f"invalid intercept radius: {radius_m}")
    if horizon_s <= 0.0:
        raise ValueError(f"invalid validation horizon: {horizon_s}")

    pursuer_position = np.asarray(instance.pursuer_initial.position_w, dtype=float)
    pursuer_velocity = np.asarray(instance.pursuer_initial.velocity_w, dtype=float)
    for target_index, target in enumerate(instance.target_initials):
        relative_position = pursuer_position - np.asarray(target.position_w, dtype=float)
        relative_velocity = pursuer_velocity - np.asarray(target.velocity_w, dtype=float)
        capture_time = _straight_path_capture_time(
            relative_position=relative_position,
            relative_velocity=relative_velocity,
            horizon_s=horizon_s,
            radius_m=radius_m,
        )
        if capture_time is not None:
            raise ValueError(
                "straight-line current path captures target: "
                f"target_index={target_index}, capture_time_s={capture_time:.6g}, "
                f"horizon_s={horizon_s:.6g}, intercept_radius_m={radius_m:.6g}"
            )


def _straight_path_capture_time(
    *,
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    horizon_s: float,
    radius_m: float,
) -> float | None:
    radius_sq = float(radius_m) ** 2
    if float(np.dot(relative_position, relative_position)) <= radius_sq:
        return 0.0

    a = float(np.dot(relative_velocity, relative_velocity))
    if a <= 1.0e-12:
        return None

    b = 2.0 * float(np.dot(relative_position, relative_velocity))
    c = float(np.dot(relative_position, relative_position)) - radius_sq
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None

    sqrt_discriminant = math.sqrt(max(discriminant, 0.0))
    roots = sorted(((-b - sqrt_discriminant) / (2.0 * a), (-b + sqrt_discriminant) / (2.0 * a)))
    for root in roots:
        if 0.0 <= root <= horizon_s:
            return float(root)
    return None
