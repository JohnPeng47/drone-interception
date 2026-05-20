from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from intercept_sim.types import ImageFeatureMeasurement, ObserverState, SceneSnapshot


@dataclass(frozen=True)
class BeihangEkfNoise:
    acc_random_walk: float = 4.33e-3
    acc_bias_walk: float = 2.66e-5
    gyro_noise_density: float = 1.87e-3
    gyro_bias_walk: float = 2.66e-5
    img_noise_density: float = 1.87e-3
    gps_pos_noise_density: float = 0.5
    gps_vel_noise_density: float = 0.01


@dataclass(frozen=True)
class BeihangEkfSnapshot:
    t: float
    x: np.ndarray
    p: np.ndarray
    phi: np.ndarray
    q: np.ndarray
    g: np.ndarray


@dataclass
class BeihangImageImuEkf:
    noise: BeihangEkfNoise = field(default_factory=BeihangEkfNoise)
    max_abs_image_feature: float = 2.0
    max_covariance: float = 1e3
    dim: int = 18
    x: np.ndarray = field(default_factory=lambda: np.zeros(18, dtype=float), init=False)
    p: np.ndarray = field(default_factory=lambda: np.eye(18, dtype=float), init=False)
    h: np.ndarray = field(default_factory=lambda: np.zeros((2, 18), dtype=float), init=False)
    q_process: np.ndarray = field(default_factory=lambda: np.zeros((6, 6), dtype=float), init=False)
    r_meas: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=float), init=False)
    initialized: bool = False

    def __post_init__(self) -> None:
        self.h[:, 10:12] = np.eye(2, dtype=float)
        self.q_process[:3, :3] = np.eye(3, dtype=float) * self.noise.gyro_bias_walk**2
        self.q_process[3:, 3:] = np.eye(3, dtype=float) * self.noise.acc_random_walk**2
        self.r_meas = np.eye(2, dtype=float) * self.noise.img_noise_density**2

    @property
    def image_feature(self) -> np.ndarray:
        return self.x[10:12].copy()

    def initialize(self, *, quat_xyzw: np.ndarray, pos_w: np.ndarray, vel_w: np.ndarray, image_uv_norm: np.ndarray) -> None:
        self.x = np.zeros(self.dim, dtype=float)
        self.x[0:4] = _xyzw_to_wxyz(_unit_quat_xyzw(quat_xyzw))
        self.x[4:7] = np.asarray(pos_w, dtype=float).reshape(3)
        self.x[7:10] = np.asarray(vel_w, dtype=float).reshape(3)
        self.x[10:12] = np.asarray(image_uv_norm, dtype=float).reshape(2)

        diag = np.ones(self.dim, dtype=float)
        diag[0:4] = self.noise.gyro_noise_density**2
        diag[4:7] = self.noise.gps_pos_noise_density**2
        diag[7:10] = self.noise.gps_vel_noise_density**2
        diag[10:12] = self.noise.img_noise_density**2
        diag[12:15] = self.noise.gyro_bias_walk**2
        diag[15:18] = self.noise.acc_random_walk**2
        self.p = np.diag(diag)
        self.initialized = True

    def predict(
        self,
        *,
        t: float,
        dt: float,
        quat_xyzw: np.ndarray,
        pos_w: np.ndarray,
        vel_w: np.ndarray,
        gyro_b: np.ndarray | None = None,
        acc_w: np.ndarray | None = None,
    ) -> BeihangEkfSnapshot:
        if not self.initialized:
            raise RuntimeError("EKF must be initialized before predict")

        phi = self._transition(dt=dt, quat_xyzw=quat_xyzw, vel_w=vel_w, gyro_b=gyro_b)
        g = np.zeros((self.dim, 6), dtype=float)
        g[0:4, 0:3] = _quat_gyro_jacobian_wxyz(self.x[0:4])
        g[7:10, 3:6] = np.eye(3, dtype=float)
        g[10:12, 0:3] = self._image_gyro_jacobian(gyro_b)

        self.x = phi @ self.x
        self.x[0:4] = _xyzw_to_wxyz(_unit_quat_xyzw(quat_xyzw))
        self.x[4:7] = np.asarray(pos_w, dtype=float).reshape(3)
        self.x[7:10] = np.asarray(vel_w, dtype=float).reshape(3)
        self.x[10:12] = np.clip(self.x[10:12], -self.max_abs_image_feature, self.max_abs_image_feature)
        self.p = phi @ self.p @ phi.T + g @ self.q_process @ g.T
        self.p = self._bounded_covariance(self.p)

        return BeihangEkfSnapshot(
            t=float(t),
            x=self.x.copy(),
            p=self.p.copy(),
            phi=phi.copy(),
            q=self.q_process.copy(),
            g=g.copy(),
        )

    def update_image(self, uv_norm: np.ndarray) -> None:
        if not self.initialized:
            raise RuntimeError("EKF must be initialized before update")
        z = np.asarray(uv_norm, dtype=float).reshape(2)
        innovation = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + self.r_meas
        k = self.p @ self.h.T @ np.linalg.pinv(s)
        self.x = self.x + k @ innovation
        self.x[10:12] = np.clip(self.x[10:12], -self.max_abs_image_feature, self.max_abs_image_feature)
        self.p = (np.eye(self.dim, dtype=float) - k @ self.h) @ self.p
        self.p = self._bounded_covariance(self.p)

    def replay_image_update(self, measurement: ImageFeatureMeasurement, history: list[BeihangEkfSnapshot]) -> None:
        if not self.initialized or not measurement.detected or measurement.uv_norm is None:
            return
        if not history:
            self.update_image(measurement.uv_norm)
            return

        insert_idx = max(
            (idx for idx, snapshot in enumerate(history) if snapshot.t <= measurement.t_capture + 1e-12),
            default=0,
        )
        anchor = history[insert_idx]
        self.x = anchor.x.copy()
        self.p = anchor.p.copy()
        self.update_image(measurement.uv_norm)
        for snapshot in history[insert_idx + 1 :]:
            self.x = snapshot.phi @ self.x
            self.x[10:12] = np.clip(self.x[10:12], -self.max_abs_image_feature, self.max_abs_image_feature)
            self.p = snapshot.phi @ self.p @ snapshot.phi.T + snapshot.g @ snapshot.q @ snapshot.g.T
            self.p = self._bounded_covariance(self.p)

    def _transition(
        self,
        *,
        dt: float,
        quat_xyzw: np.ndarray,
        vel_w: np.ndarray,
        gyro_b: np.ndarray | None,
    ) -> np.ndarray:
        phi = np.eye(self.dim, dtype=float)
        phi[4:7, 7:10] = np.eye(3, dtype=float) * float(dt)

        uv = self.x[10:12]
        range_proxy = max(float(np.linalg.norm(self.x[4:7])), 1.0)
        vel_c = Rotation.from_quat(_unit_quat_xyzw(quat_xyzw)).as_matrix().T @ np.asarray(vel_w, dtype=float).reshape(3)
        phi[10:12, 7:10] = np.array(
            [
                [-1.0 / range_proxy, 0.0, uv[0] / range_proxy],
                [0.0, -1.0 / range_proxy, uv[1] / range_proxy],
            ],
            dtype=float,
        ) * float(dt)

        w_c = np.zeros(3, dtype=float) if gyro_b is None else np.asarray(gyro_b, dtype=float).reshape(3) * float(dt)
        image_flow = np.array(
            [
                [uv[1] * w_c[0] - 2.0 * uv[0] * w_c[1], uv[0] * w_c[0] + w_c[2]],
                [-uv[1] * w_c[1] - w_c[2], 2.0 * uv[1] * w_c[0] - uv[0] * w_c[1]],
            ],
            dtype=float,
        )
        image_flow += np.eye(2, dtype=float) * vel_c[0] / range_proxy * float(dt)
        phi[10:12, 10:12] = np.eye(2, dtype=float) + image_flow
        phi[10:12, 12:15] = self._image_gyro_jacobian(gyro_b) * float(dt)
        return phi

    def _image_gyro_jacobian(self, gyro_b: np.ndarray | None) -> np.ndarray:
        uv = self.x[10:12]
        return np.array(
            [
                [uv[0] * uv[1], -(1.0 + uv[0] * uv[0]), uv[1]],
                [1.0 + uv[1] * uv[1], -uv[0] * uv[1], -uv[0]],
            ],
            dtype=float,
        )

    def _bounded_covariance(self, covariance: np.ndarray) -> np.ndarray:
        covariance = np.nan_to_num(
            _symmetrize(covariance),
            nan=self.max_covariance,
            posinf=self.max_covariance,
            neginf=-self.max_covariance,
        )
        return np.clip(covariance, -self.max_covariance, self.max_covariance)


@dataclass
class BeihangImageEkfObserver:
    history_size: int = 50
    ekf: BeihangImageImuEkf = field(default_factory=BeihangImageImuEkf)
    latest_feature: ImageFeatureMeasurement | None = None
    _pending_measurements: list[ImageFeatureMeasurement] = field(default_factory=list, init=False, repr=False)
    _history: list[BeihangEkfSnapshot] = field(default_factory=list, init=False, repr=False)
    _last_t: float | None = None
    _last_vel_w: np.ndarray | None = None
    relative_position_w: np.ndarray | None = None
    relative_velocity_w: np.ndarray | None = None
    target_acceleration_w: np.ndarray | None = None
    vehicle_rotation_wb: np.ndarray | None = None

    def update_scene(self, scene: SceneSnapshot) -> None:
        self.vehicle_rotation_wb = scene.pursuer.rotation_wb.copy()
        if not scene.targets:
            self.relative_position_w = None
            self.relative_velocity_w = None
            self.target_acceleration_w = None
            return
        target = scene.targets[0]
        self.relative_position_w = scene.pursuer.position_w - target.position_w
        self.relative_velocity_w = scene.pursuer.velocity_w - target.velocity_w
        self.target_acceleration_w = np.zeros(3, dtype=float)

    def predict(self, t: float, vehicle_state: dict[str, np.ndarray]) -> ObserverState:
        self._initialize_if_needed(vehicle_state)
        if self.ekf.initialized:
            dt = 0.0 if self._last_t is None else max(float(t) - self._last_t, 0.0)
            pos_w = self._filter_position(vehicle_state)
            vel_w = self._filter_velocity(vehicle_state)
            acc_w = None if self._last_vel_w is None or dt <= 1e-12 else (vel_w - self._last_vel_w) / dt
            snapshot = self.ekf.predict(
                t=float(t),
                dt=dt,
                quat_xyzw=np.asarray(vehicle_state.get("q", [0.0, 0.0, 0.0, 1.0]), dtype=float),
                pos_w=pos_w,
                vel_w=vel_w,
                gyro_b=np.asarray(vehicle_state.get("w", np.zeros(3, dtype=float)), dtype=float),
                acc_w=acc_w,
            )
            self._history.append(snapshot)
            self._history = self._history[-self.history_size :]
            for measurement in self._pending_measurements:
                self.ekf.replay_image_update(measurement, self._history)
            self._pending_measurements.clear()
            self.latest_feature = self._feature_from_state(float(t))
            self._last_vel_w = vel_w.copy()
        self._last_t = float(t)
        return ObserverState(
            t=float(t),
            vehicle_state=vehicle_state,
            image_feature=self.latest_feature,
            relative_position_w=None if self.relative_position_w is None else self.relative_position_w.copy(),
            relative_velocity_w=None if self.relative_velocity_w is None else self.relative_velocity_w.copy(),
            target_acceleration_w=None if self.target_acceleration_w is None else self.target_acceleration_w.copy(),
            vehicle_rotation_wb=None if self.vehicle_rotation_wb is None else self.vehicle_rotation_wb.copy(),
        )

    def update_image_feature(self, measurement: ImageFeatureMeasurement) -> None:
        if not measurement.detected or measurement.uv_norm is None:
            self.latest_feature = measurement
            return
        self.latest_feature = measurement
        self._pending_measurements.append(measurement)

    def reset(self) -> None:
        self.ekf = BeihangImageImuEkf()
        self.latest_feature = None
        self._pending_measurements.clear()
        self._history.clear()
        self._last_t = None
        self._last_vel_w = None
        self.relative_position_w = None
        self.relative_velocity_w = None
        self.target_acceleration_w = None
        self.vehicle_rotation_wb = None

    def _initialize_if_needed(self, vehicle_state: dict[str, np.ndarray]) -> None:
        if self.ekf.initialized or self.latest_feature is None or self.latest_feature.uv_norm is None:
            return
        self.ekf.initialize(
            quat_xyzw=np.asarray(vehicle_state.get("q", [0.0, 0.0, 0.0, 1.0]), dtype=float),
            pos_w=self._filter_position(vehicle_state),
            vel_w=self._filter_velocity(vehicle_state),
            image_uv_norm=np.asarray(self.latest_feature.uv_norm, dtype=float),
        )

    def _filter_position(self, vehicle_state: dict[str, np.ndarray]) -> np.ndarray:
        if self.relative_position_w is not None:
            return self.relative_position_w.copy()
        return np.asarray(vehicle_state.get("x", np.zeros(3, dtype=float)), dtype=float)

    def _filter_velocity(self, vehicle_state: dict[str, np.ndarray]) -> np.ndarray:
        if self.relative_velocity_w is not None:
            return self.relative_velocity_w.copy()
        return np.asarray(vehicle_state.get("v", np.zeros(3, dtype=float)), dtype=float)

    def _feature_from_state(self, t: float) -> ImageFeatureMeasurement:
        source = self.latest_feature
        return ImageFeatureMeasurement(
            t_capture=source.t_capture if source is not None else float(t),
            t_available=float(t),
            camera_id=source.camera_id if source is not None else "front",
            target_id=source.target_id if source is not None else None,
            detected=True,
            uv_px=None,
            uv_norm=self.ekf.image_feature,
            confidence=source.confidence if source is not None else 1.0,
        )


def _unit_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def _quat_gyro_jacobian_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = np.asarray(quat_wxyz, dtype=float).reshape(4)
    return 0.5 * np.array(
        [
            [q1, q2, q3],
            [-q0, q3, -q2],
            [-q3, -q0, q1],
            [q2, -q1, -q0],
        ],
        dtype=float,
    )


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
