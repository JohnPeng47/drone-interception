from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from intercept_sim.types import CtbrCommand, ObserverState


@dataclass
class ImageFeatureIbvsController:
    mass_kg: float
    gravity_mps2: float = 9.81
    k_yaw: float = 2.0
    k_pitch: float = 2.0
    max_rate_rps: float = 2.0

    def update(self, t: float, observer_state: ObserverState) -> CtbrCommand:
        feature = observer_state.image_feature
        if feature is None or not feature.detected or feature.uv_norm is None:
            return CtbrCommand(
                t=float(t),
                thrust_n=self.mass_kg * self.gravity_mps2,
                body_rates_b=np.zeros(3, dtype=float),
            )

        ex = float(feature.uv_norm[0])
        ey = float(feature.uv_norm[1])
        return CtbrCommand(
            t=float(t),
            thrust_n=self.mass_kg * self.gravity_mps2,
            body_rates_b=np.array(
                [
                    0.0,
                    np.clip(-self.k_pitch * ey, -self.max_rate_rps, self.max_rate_rps),
                    np.clip(-self.k_yaw * ex, -self.max_rate_rps, self.max_rate_rps),
                ],
                dtype=float,
            ),
        )


@dataclass
class GeometricImageFeatureController:
    mass_kg: float
    gravity_mps2: float = 9.81
    k_align: float = 2.0
    max_rate_rps: float = 2.0
    camera_to_body: np.ndarray | None = None
    desired_los_b: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.camera_to_body = np.eye(3, dtype=float) if self.camera_to_body is None else np.asarray(self.camera_to_body, dtype=float)
        self.desired_los_b = _unit(
            np.array([1.0, 0.0, 0.0], dtype=float)
            if self.desired_los_b is None
            else np.asarray(self.desired_los_b, dtype=float)
        )

    def update(self, t: float, observer_state: ObserverState) -> CtbrCommand:
        feature = observer_state.image_feature
        if feature is None or not feature.detected or feature.uv_norm is None:
            return CtbrCommand(
                t=float(t),
                thrust_n=self.mass_kg * self.gravity_mps2,
                body_rates_b=np.zeros(3, dtype=float),
            )

        target_los_c = _unit(np.array([1.0, float(feature.uv_norm[0]), float(feature.uv_norm[1])], dtype=float))
        target_los_b = _unit(np.asarray(self.camera_to_body, dtype=float) @ target_los_c)
        desired_los_b = np.asarray(self.desired_los_b, dtype=float)
        los_error_b = np.cross(target_los_b, desired_los_b)
        body_rates_b = self.k_align * np.array([los_error_b[0], -los_error_b[1], los_error_b[2]], dtype=float)
        body_rates_b = np.clip(body_rates_b, -self.max_rate_rps, self.max_rate_rps)
        return CtbrCommand(
            t=float(t),
            thrust_n=self.mass_kg * self.gravity_mps2,
            body_rates_b=body_rates_b,
        )


@dataclass
class BeihangBacksteppingController:
    mass_kg: float
    gravity_mps2: float = 9.801
    barrier_k: float = 0.25
    c1: float = 1.0
    alpha1_gain: float = 10.0
    c2: float = 0.8
    c3: float = 0.3
    max_rate_rps: float = 8.0
    max_thrust_n: float | None = None
    camera_to_body: np.ndarray | None = None
    camera_optical_axis_c: np.ndarray | None = None
    last_vehicle_velocity_w: np.ndarray | None = None
    last_t: float | None = None

    def __post_init__(self) -> None:
        self.camera_to_body = np.eye(3, dtype=float) if self.camera_to_body is None else np.asarray(self.camera_to_body, dtype=float)
        self.camera_optical_axis_c = _unit(
            np.array([1.0, 0.0, 0.0], dtype=float)
            if self.camera_optical_axis_c is None
            else np.asarray(self.camera_optical_axis_c, dtype=float)
        )

    def update(self, t: float, observer_state: ObserverState) -> CtbrCommand:
        feature = observer_state.image_feature
        if (
            feature is None
            or not feature.detected
            or feature.uv_norm is None
            or observer_state.relative_position_w is None
            or observer_state.relative_velocity_w is None
        ):
            return self._hover(t)

        p_r = np.asarray(observer_state.relative_position_w, dtype=float).reshape(3, 1)
        v_r = np.asarray(observer_state.relative_velocity_w, dtype=float).reshape(3, 1)
        range_m = float(np.linalg.norm(p_r))
        if range_m <= 1e-9:
            return self._hover(t)

        rotation_wb = self._rotation_wb(observer_state)
        n_t_w = self._target_los_w(feature, rotation_wb)
        n_c_w = self._camera_axis_w(rotation_wb)
        n_td_w = n_c_w

        z1 = float((1.0 - n_td_w.T @ n_t_w).item())
        if z1 >= self.barrier_k:
            z1 = self.barrier_k - 1e-3
        k_term = z1 / (self.barrier_k * self.barrier_k - z1 * z1)

        z2 = p_r
        alpha1 = self.alpha1_gain * n_t_w
        z3 = v_r - alpha1

        eye3 = np.eye(3, dtype=float)
        projection_term = (-eye3 + n_t_w @ n_t_w.T) @ n_td_w
        gravity_w = np.array([[0.0], [0.0], [-self.gravity_mps2]], dtype=float)
        alpha2 = (
            -self.c2 * self.mass_kg * z3
            - self.mass_kg * gravity_w
            - self.c1 * self.mass_kg * v_r
            + self.mass_kg * n_t_w
            + k_term * self.mass_kg / range_m * projection_term
        )

        n_f = rotation_wb @ np.array([[0.0], [0.0], [1.0]], dtype=float)
        thrust_n = float(np.linalg.norm(alpha2))
        z4 = (thrust_n * n_f - alpha2) / self.mass_kg

        vehicle_acc_w = self._vehicle_acceleration_w(float(t), observer_state.vehicle_state)
        target_acc_w = (
            np.zeros((3, 1), dtype=float)
            if observer_state.target_acceleration_w is None
            else np.asarray(observer_state.target_acceleration_w, dtype=float).reshape(3, 1)
        )
        a_r = vehicle_acc_w - target_acc_w
        z2_dot = v_r
        z3_dot = thrust_n / self.mass_kg * n_f + gravity_w + self.c1 * v_r
        alpha2_dot = -self.c2 * self.mass_kg * z3_dot - self.c1 * self.mass_kg * a_r - self.mass_kg * z2_dot

        n_f_x = _skew(n_f)
        a_row = k_term * np.cross(n_td_w.T, n_t_w.T) + thrust_n / self.mass_kg * z4.T @ n_f_x
        b_scalar = z4.T @ (z3 - alpha2_dot / self.mass_kg + self.c3 * z4)
        omega_w = np.linalg.pinv(a_row) @ b_scalar
        omega_b = (rotation_wb.T @ omega_w).reshape(3)
        omega_b = np.clip(omega_b, -self.max_rate_rps, self.max_rate_rps)

        if self.max_thrust_n is not None:
            thrust_n = float(np.clip(thrust_n, 0.0, self.max_thrust_n))

        return CtbrCommand(t=float(t), thrust_n=thrust_n, body_rates_b=omega_b)

    def reset(self) -> None:
        self.last_vehicle_velocity_w = None
        self.last_t = None

    def _hover(self, t: float) -> CtbrCommand:
        return CtbrCommand(
            t=float(t),
            thrust_n=self.mass_kg * self.gravity_mps2,
            body_rates_b=np.zeros(3, dtype=float),
        )

    def _target_los_w(self, feature: object, rotation_wb: np.ndarray) -> np.ndarray:
        uv_norm = np.asarray(feature.uv_norm, dtype=float)
        n_t_c = _unit(np.array([1.0, uv_norm[0], uv_norm[1]], dtype=float)).reshape(3, 1)
        return rotation_wb @ np.asarray(self.camera_to_body, dtype=float) @ n_t_c

    def _camera_axis_w(self, rotation_wb: np.ndarray) -> np.ndarray:
        n_c_c = np.asarray(self.camera_optical_axis_c, dtype=float).reshape(3, 1)
        return rotation_wb @ np.asarray(self.camera_to_body, dtype=float) @ n_c_c

    def _rotation_wb(self, observer_state: ObserverState) -> np.ndarray:
        if observer_state.vehicle_rotation_wb is not None:
            return np.asarray(observer_state.vehicle_rotation_wb, dtype=float).reshape(3, 3)
        return Rotation.from_quat(observer_state.vehicle_state["q"]).as_matrix()

    def _vehicle_acceleration_w(self, t: float, vehicle_state: dict[str, np.ndarray]) -> np.ndarray:
        velocity_w = np.asarray(vehicle_state.get("v", np.zeros(3, dtype=float)), dtype=float).reshape(3, 1)
        if self.last_vehicle_velocity_w is None or self.last_t is None:
            acc_w = np.zeros((3, 1), dtype=float)
        else:
            dt = max(float(t) - self.last_t, 1e-9)
            acc_w = (velocity_w - self.last_vehicle_velocity_w) / dt
        self.last_vehicle_velocity_w = velocity_w.copy()
        self.last_t = float(t)
        return acc_w


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def _skew(vector: np.ndarray) -> np.ndarray:
    flat = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -flat[2], flat[1]],
            [flat[2], 0.0, -flat[0]],
            [-flat[1], flat[0], 0.0],
        ],
        dtype=float,
    )
