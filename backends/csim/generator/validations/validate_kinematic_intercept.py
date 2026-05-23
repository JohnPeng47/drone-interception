from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .validate_target_in_fov import apply_initial_pitch_offset


DEFAULT_X500_MASS_KG = 2.064
DEFAULT_GRAVITY_W = np.array([0.0, 0.0, -9.81], dtype=float)
BODY_THRUST_AXIS_B = np.array([0.0, 0.0, 1.0], dtype=float)


def validate_kinematic_intercept(raw_config: dict[str, Any]) -> None:
    """Reject samples that are obviously unreachable under CTBR-like limits.

    This is a cheap optimistic feasibility check, not a proof. It assumes the
    pursuer slews its thrust axis at `max_rate_rps`, then applies one constant
    world acceleration for the remaining time.
    """

    vehicle = raw_config["vehicle"]
    target_cfg = raw_config["target"].get("initial_state", raw_config["target"])
    controller = raw_config["controller"]
    sim = raw_config["sim"]
    metrics = raw_config["metrics"]

    p0 = _array(vehicle["initial_position_w"], length=3)
    v0 = _array(vehicle.get("initial_velocity_w", [0.0, 0.0, 0.0]), length=3)
    q_xyzw = _array(vehicle.get("initial_quat_xyzw", [0.0, 0.0, 0.0, 1.0]), length=4)
    q_xyzw = apply_initial_pitch_offset(q_xyzw, vehicle)
    rotation_wb = Rotation.from_quat(q_xyzw).as_matrix()
    thrust_axis_now_w = _unit(rotation_wb @ BODY_THRUST_AXIS_B)

    target_p0 = _array(
        target_cfg["position_w"] if "position_w" in target_cfg else raw_config["target"]["initial_position_w"],
        length=3,
    )
    target_v = _array(
        target_cfg["velocity_w"] if "velocity_w" in target_cfg else raw_config["target"]["velocity_w"],
        length=3,
    )

    mass_kg = _mass_kg(vehicle)
    max_thrust_n = _max_thrust_n(controller)
    max_rate_rps = _max_rate_rps(controller)
    intercept_radius_m = float(metrics.get("catch_radius_m", 0.0))
    horizon_s = float(sim["duration_s"])
    dt = float(sim.get("validation_dt", max(float(sim.get("dt", 0.005)), 0.02)))

    if mass_kg <= 0.0:
        raise ValueError(f"invalid pursuer mass: {mass_kg}")
    if max_thrust_n <= 0.0:
        raise ValueError(f"invalid max thrust: {max_thrust_n}")
    if max_rate_rps <= 0.0:
        raise ValueError(f"invalid max body rate: {max_rate_rps}")

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


def _mass_kg(vehicle: dict[str, Any]) -> float:
    params = vehicle.get("params_override", {}) or {}
    for source in (vehicle, params):
        for key in ("mass_kg", "mass"):
            if key in source:
                return float(source[key])
    model = str(vehicle.get("model", "")).lower()
    if model in {"x500", "x500_v2", "holybro_x500", "holybro_x500_v2"}:
        return DEFAULT_X500_MASS_KG
    raise ValueError("vehicle mass is required for kinematic intercept validation")


def _max_thrust_n(controller: dict[str, Any]) -> float:
    gains = controller.get("gains", {}) or {}
    if "f_max" in gains:
        return float(gains["f_max"])
    return float(controller["max_thrust_n"])


def _max_rate_rps(controller: dict[str, Any]) -> float:
    gains = controller.get("gains", {}) or {}
    if "omega_max" in gains:
        return float(gains["omega_max"])
    return float(controller["max_rate_rps"])


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(float(np.dot(_unit(a), _unit(b))), -1.0, 1.0)))


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize zero vector")
    return np.asarray(value, dtype=float) / norm


def _array(value: Any, *, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"Expected array of shape ({length},), got {arr.shape}")
    return arr.copy()
