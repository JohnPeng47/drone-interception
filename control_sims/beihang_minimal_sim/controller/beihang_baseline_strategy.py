"""Simple two-objective image-centering and closing-speed heuristic."""

from __future__ import annotations

import numpy as np

from ..config import BaselineStrategyConfig, VehicleConfig
from ..types import CtbrCommand, StrategyObservation


class BeihangBaselineStrategy:
    """Baseline strategy preserving the paper's two control objectives.

    Objective 1: drive the image feature toward the optical axis.
    Objective 2: use more thrust when the feature is centered, creating a
    simple image-conditioned closing behavior under CTBR dynamics.
    """

    def __init__(self, vehicle: VehicleConfig, config: BaselineStrategyConfig):
        self._vehicle = vehicle
        self._cfg = config

    def command(self, observation: StrategyObservation, t: float) -> CtbrCommand:
        hover = self._vehicle.mass_kg * 9.81
        if not observation.detected or observation.uv_norm is None:
            return CtbrCommand(
                t=t,
                thrust_n=hover,
                body_rates_b=np.array([0.0, 0.0, 0.5]),
            )

        uv = np.asarray(observation.uv_norm, dtype=float)
        R_wb = np.asarray(observation.vehicle_rotation_wb, dtype=float)
        v_w = np.asarray(observation.vehicle_velocity_w, dtype=float)

        bearing_b = np.array([1.0, uv[0], uv[1]], dtype=float)
        bearing_b /= max(float(np.linalg.norm(bearing_b)), 1e-12)
        bearing_w = R_wb @ bearing_b
        body_x_w = R_wb[:, 0]
        body_z_w = R_wb[:, 2]

        desired_v = self._cfg.desired_speed_mps * bearing_w
        range_term = self._cfg.range_gain * float(observation.depth_m or 0.0) * bearing_w
        desired_accel_w = self._cfg.velocity_gain * (desired_v - v_w) + range_term
        thrust_vector_w = desired_accel_w - np.array([0.0, 0.0, -9.81])
        thrust_axis_w = thrust_vector_w / max(float(np.linalg.norm(thrust_vector_w)), 1e-12)

        optical_error_b = R_wb.T @ np.cross(body_x_w, bearing_w)
        thrust_error_b = R_wb.T @ np.cross(body_z_w, thrust_axis_w)
        omega = (
            self._cfg.image_axis_gain * optical_error_b
            + self._cfg.thrust_axis_gain * thrust_error_b
        )
        omega_norm = float(np.linalg.norm(omega))
        if omega_norm > self._vehicle.max_body_rate_rad_s:
            omega = omega * (self._vehicle.max_body_rate_rad_s / omega_norm)

        thrust = self._vehicle.mass_kg * float(thrust_vector_w @ body_z_w)
        thrust += self._cfg.thrust_margin_n
        thrust = float(np.clip(thrust, 0.0, self._vehicle.max_thrust_n))
        return CtbrCommand(t=t, thrust_n=thrust, body_rates_b=omega)
