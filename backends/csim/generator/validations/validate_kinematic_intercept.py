from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from backends.csim.bindings.types import SimInstance


DEFAULT_GRAVITY_W = np.array([0.0, 0.0, -9.81], dtype=float)
BODY_THRUST_AXIS_B = np.array([0.0, 0.0, 1.0], dtype=float)


def validate_kinematic_intercept(instance: SimInstance) -> None:
    """Reject samples that are obviously unreachable under CTBR-like limits.

    This is a cheap optimistic feasibility check, not a proof. It assumes the
    pursuer slews its thrust axis at `max_rate_rps`, then applies one constant
    world acceleration for the remaining time.
    """
    if instance.config is None:
        raise ValueError("kinematic intercept validation requires SimInstance.config")
    if not instance.targets:
        raise ValueError("kinematic intercept validation requires at least one target")

    initial = instance.pursuer_initial
    target = instance.targets[0]
    config = instance.config
    options = config.options

    p0 = np.asarray(initial.position_w, dtype=float)
    v0 = np.asarray(initial.velocity_w, dtype=float)
    rotation_wb = Rotation.from_quat(np.asarray(initial.quat_xyzw, dtype=float)).as_matrix()
    thrust_axis_now_w = _unit(rotation_wb @ BODY_THRUST_AXIS_B)

    mass_kg = float(config.pursuer.mass_kg)
    max_thrust_n = float(config.max_thrust_n)
    max_rate_rps = float(config.max_rate_rps)
    intercept_radius_m = float(config.intercept_radius_m)
    horizon_s = float(options.duration_s)
    dt = float(options.validation_dt if options.validation_dt is not None else max(float(options.backend_dt), 0.02))

    if mass_kg <= 0.0:
        raise ValueError(f"invalid pursuer mass: {mass_kg}")
    if max_thrust_n <= 0.0:
        raise ValueError(f"invalid max thrust: {max_thrust_n}")
    if max_rate_rps <= 0.0:
        raise ValueError(f"invalid max body rate: {max_rate_rps}")
    if horizon_s <= 0.0:
        raise ValueError(f"invalid validation horizon: {horizon_s}")

    target_p0 = np.asarray(target.initial.position_w, dtype=float)
    target_v = np.asarray(target.initial.velocity_w, dtype=float)
    best_required_thrust = float("inf")
    best_t = None
    for t in np.arange(dt, horizon_s + 0.5 * dt, dt):
        target_t = target_p0 + target_v * float(t)
        displacement = target_t - p0 - v0 * float(t)
        if float(np.linalg.norm(displacement)) <= intercept_radius_m:
            return

        required = _required_thrust_with_slew(
            displacement=displacement,
            t=float(t),
            thrust_axis_now_w=thrust_axis_now_w,
            max_rate_rps=max_rate_rps,
            mass_kg=mass_kg,
        )
        if required is None:
            continue
        required_thrust_n, _t_rotate_s = required
        if required_thrust_n < best_required_thrust:
            best_required_thrust = required_thrust_n
            best_t = float(t)
        if required_thrust_n <= max_thrust_n:
            return

    best = "none" if best_t is None else f"{best_required_thrust:.3f} N at t={best_t:.3f}s"
    raise ValueError(
        "no kinematically feasible intercept found: "
        f"horizon_s={horizon_s}, max_thrust_n={max_thrust_n}, "
        f"max_rate_rps={max_rate_rps}, best_required_thrust={best}"
    )


def _required_thrust_with_slew(
    *,
    displacement: np.ndarray,
    t: float,
    thrust_axis_now_w: np.ndarray,
    max_rate_rps: float,
    mass_kg: float,
) -> tuple[float, float] | None:
    t_rotate = 0.0
    thrust_accel_req = None
    for _ in range(3):
        t_accel = t - t_rotate
        if t_accel <= 1e-6:
            return None
        a_req_w = 2.0 * displacement / (t_accel * t_accel)
        thrust_accel_req = a_req_w - DEFAULT_GRAVITY_W
        thrust_norm = float(np.linalg.norm(thrust_accel_req))
        if thrust_norm <= 1e-9:
            t_rotate = 0.0
            break
        thrust_axis_req_w = thrust_accel_req / thrust_norm
        theta = _angle_between(thrust_axis_now_w, thrust_axis_req_w)
        t_rotate = theta / max_rate_rps

    if thrust_accel_req is None:
        return None
    return mass_kg * float(np.linalg.norm(thrust_accel_req)), t_rotate


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(float(np.dot(_unit(a), _unit(b))), -1.0, 1.0)))


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize zero vector")
    return np.asarray(value, dtype=float) / norm
