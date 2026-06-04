from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from backends.csim.bindings.types import SimInstance, SimSnapshot

from .cv_detection import TraditionalCvMeasurement, missed_measurement


@dataclass(frozen=True)
class VisualObserverConfig:
    range_prior_m: float = 5.0
    range_prior_std_m: float = 2.0
    target_velocity_prior_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_velocity_prior_std_mps: float = 1.5
    stale_timeout_s: float = 3.0
    metric_position_std_threshold_m: float = 3.0
    metric_velocity_std_threshold_mps: float = 3.0
    metric_range_std_threshold_m: float = 2.0
    process_position_noise_mps: float = 0.15
    process_velocity_noise_mps2: float = 0.4
    target_process_model: str = "constant_velocity"
    target_velocity_damping_per_s: float = 0.5
    image_noise_norm: float = 1.0e-3
    size_noise_fraction: float = 0.25
    min_size_confidence: float = 0.05
    min_detections_for_metric: int = 2


@dataclass(frozen=True)
class RelativeStateEstimate:
    p_r_w: np.ndarray
    v_r_w: np.ndarray
    covariance: np.ndarray
    valid: bool
    stale_s: float
    bearing_w: np.ndarray
    metric_confident: bool
    detection_count: int
    estimated_range_m: float
    range_std_m: float
    position_std_m: float
    velocity_std_m: float


class VisualRelativeStateObserver:
    """Per-slot visual observer using only pursuer state and image bearings."""

    def __init__(self, config: VisualObserverConfig | Mapping[str, float] | None = None):
        if isinstance(config, VisualObserverConfig):
            self.config = config
        else:
            self.config = VisualObserverConfig(**dict(config or {}))
        self._slots: dict[int, _SlotObserver] = {}

    def reset(self) -> None:
        self._slots.clear()

    def start_slots(
        self,
        slots: np.ndarray,
        instances: Sequence[SimInstance],
        snapshots: Sequence[SimSnapshot],
        image_measurements: Mapping[int, TraditionalCvMeasurement | None] | None = None,
    ) -> None:
        for raw_slot, instance, snapshot in zip(np.asarray(slots, dtype=np.int64).reshape(-1), instances, snapshots):
            slot = int(raw_slot)
            measurement = None
            if image_measurements is not None and slot in image_measurements:
                measurement = image_measurements[slot] or missed_measurement()
            self._slots[slot] = _SlotObserver(instance, snapshot, self.config, image_measurement=measurement)

    def stop_slot(self, slot: int) -> None:
        self._slots.pop(int(slot), None)

    def estimate(
        self,
        slot: int,
        instance: SimInstance,
        snapshot: SimSnapshot,
        t_s: float,
        image_measurement: TraditionalCvMeasurement | None = None,
    ) -> RelativeStateEstimate:
        observer = self._slots.get(int(slot))
        if observer is None:
            observer = _SlotObserver(instance, snapshot, self.config, image_measurement=image_measurement)
            self._slots[int(slot)] = observer
        return observer.update(snapshot, t_s, image_measurement=image_measurement)


class _SlotObserver:
    def __init__(
        self,
        instance: SimInstance,
        snapshot: SimSnapshot,
        config: VisualObserverConfig,
        *,
        image_measurement: TraditionalCvMeasurement | None = None,
    ):
        if instance.config is None or not instance.config.cameras:
            raise ValueError("IVBS observer requires SimInstance.config with a camera")
        self.instance = instance
        self.camera = instance.config.cameras[0]
        self.config = config
        self.x: np.ndarray | None = None
        self.p: np.ndarray | None = None
        self.last_t = 0.0
        self.last_detection_t: float | None = None
        self.detection_count = 0
        self.last_bearing_w = np.array([1.0, 0.0, 0.0], dtype=float)
        initial_uv = self._uv_from_measurement_or_snapshot(snapshot, image_measurement)
        initial_bearing = None if initial_uv is None else self._bearing_from_uv(snapshot, initial_uv)
        if initial_bearing is not None:
            self._initialize(snapshot, initial_bearing)

    def update(
        self,
        snapshot: SimSnapshot,
        t_s: float,
        *,
        image_measurement: TraditionalCvMeasurement | None = None,
    ) -> RelativeStateEstimate:
        dt = max(0.0, float(t_s) - float(self.last_t))
        if self.x is not None and self.p is not None and dt > 0.0:
            self._predict(dt)
        uv_norm = self._uv_from_measurement_or_snapshot(snapshot, image_measurement)
        bearing_w = None if uv_norm is None else self._bearing_from_uv(snapshot, uv_norm)
        if bearing_w is not None and uv_norm is not None:
            if self.x is None or self.p is None:
                self._initialize(snapshot, bearing_w)
            corrected = self._correct_uv(snapshot, uv_norm)
            if corrected and image_measurement is not None and image_measurement.detected:
                self._correct_size(snapshot, image_measurement)
            if corrected:
                self.last_detection_t = float(t_s)
                self.detection_count += 1
            self.last_bearing_w = bearing_w
        self.last_t = float(t_s)
        return self._estimate_from_state(snapshot, t_s, measurement_bearing_w=bearing_w, image_measurement=image_measurement)

    def _initialize(self, snapshot: SimSnapshot, bearing_w: np.ndarray) -> None:
        r_wb = _quat_xyzw_to_rot(snapshot.pursuer.quat_xyzw)
        p_pursuer_w = np.asarray(snapshot.pursuer.position_w, dtype=float).reshape(3)
        p_camera_w = p_pursuer_w + r_wb @ np.asarray(self.camera.position_b, dtype=float).reshape(3)
        target_p_w = p_camera_w + float(self.config.range_prior_m) * bearing_w
        target_v_w = np.asarray(self.config.target_velocity_prior_w, dtype=float).reshape(3)
        self.x = np.concatenate([target_p_w, target_v_w]).astype(float)
        pos_var = float(self.config.range_prior_std_m) ** 2
        vel_var = float(self.config.target_velocity_prior_std_mps) ** 2
        self.p = np.diag([pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]).astype(float)

    def _predict(self, dt: float) -> None:
        assert self.x is not None and self.p is not None
        f = np.eye(6, dtype=float)
        model = str(self.config.target_process_model)
        if model == "constant_velocity":
            f[0:3, 3:6] = np.eye(3) * float(dt)
        elif model == "damped_velocity":
            damping = max(float(self.config.target_velocity_damping_per_s), 0.0)
            decay = float(np.exp(-damping * float(dt))) if damping > 1.0e-9 else 1.0
            alpha = (1.0 - decay) / damping if damping > 1.0e-9 else float(dt)
            f[0:3, 3:6] = np.eye(3) * alpha
            f[3:6, 3:6] = np.eye(3) * decay
        elif model == "stationary":
            f[3:6, 3:6] = np.zeros((3, 3), dtype=float)
        else:
            raise ValueError(f"unsupported IVBS target_process_model: {model}")
        q = np.zeros((6, 6), dtype=float)
        q_pos = float(self.config.process_position_noise_mps) ** 2 * float(dt)
        q_vel = float(self.config.process_velocity_noise_mps2) ** 2 * float(dt)
        q[0:3, 0:3] = np.eye(3) * q_pos
        q[3:6, 3:6] = np.eye(3) * q_vel
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q
        self.p = 0.5 * (self.p + self.p.T)

    def _estimate_from_state(
        self,
        snapshot: SimSnapshot,
        t_s: float,
        *,
        measurement_bearing_w: np.ndarray | None = None,
        image_measurement: TraditionalCvMeasurement | None = None,
    ) -> RelativeStateEstimate:
        if self.x is None or self.p is None:
            return RelativeStateEstimate(
                p_r_w=np.zeros(3, dtype=float),
                v_r_w=np.zeros(3, dtype=float),
                covariance=np.eye(6, dtype=float) * 1.0e6,
                valid=False,
                stale_s=float("inf"),
                bearing_w=self.last_bearing_w.copy(),
                metric_confident=False,
                detection_count=0,
                estimated_range_m=float("nan"),
                range_std_m=float("inf"),
                position_std_m=float("inf"),
                velocity_std_m=float("inf"),
            )
        pursuer_p_w = np.asarray(snapshot.pursuer.position_w, dtype=float).reshape(3)
        pursuer_v_w = np.asarray(snapshot.pursuer.velocity_w, dtype=float).reshape(3)
        target_p_w = np.asarray(self.x[0:3], dtype=float)
        target_v_w = np.asarray(self.x[3:6], dtype=float)
        stale_s = float("inf") if self.last_detection_t is None else max(0.0, float(t_s) - self.last_detection_t)
        valid = bool(stale_s <= float(self.config.stale_timeout_s))
        pos_std = float(np.sqrt(max(np.trace(self.p[0:3, 0:3]) / 3.0, 0.0)))
        vel_std = float(np.sqrt(max(np.trace(self.p[3:6, 3:6]) / 3.0, 0.0)))
        measured_bearing = measurement_bearing_w
        if measured_bearing is None and image_measurement is None:
            measured_bearing = self._bearing_from_snapshot(snapshot)
        predicted_bearing = self._bearing_to_target(snapshot, target_p_w)
        if measured_bearing is not None:
            bearing_w = measured_bearing
        elif predicted_bearing is not None:
            bearing_w = predicted_bearing
        else:
            bearing_w = self.last_bearing_w.copy()
        range_std = _directional_std(self.p[0:3, 0:3], bearing_w)
        metric_confident = bool(
            valid
            and self.detection_count >= int(self.config.min_detections_for_metric)
            and pos_std <= float(self.config.metric_position_std_threshold_m)
            and vel_std <= float(self.config.metric_velocity_std_threshold_mps)
            and range_std <= float(self.config.metric_range_std_threshold_m)
        )
        return RelativeStateEstimate(
            p_r_w=pursuer_p_w - target_p_w,
            v_r_w=pursuer_v_w - target_v_w,
            covariance=self.p.copy(),
            valid=valid,
            stale_s=stale_s,
            bearing_w=bearing_w,
            metric_confident=metric_confident,
            detection_count=int(self.detection_count),
            estimated_range_m=float(np.linalg.norm(pursuer_p_w - target_p_w)),
            range_std_m=range_std,
            position_std_m=pos_std,
            velocity_std_m=vel_std,
        )

    def _bearing_from_snapshot(self, snapshot: SimSnapshot) -> np.ndarray | None:
        if not snapshot.camera.detected:
            return None
        uv = np.asarray(snapshot.camera.uv_norm, dtype=float).reshape(2)
        if not np.all(np.isfinite(uv)):
            return None
        return self._bearing_from_uv(snapshot, uv)

    def _uv_from_measurement_or_snapshot(
        self,
        snapshot: SimSnapshot,
        image_measurement: TraditionalCvMeasurement | None,
    ) -> np.ndarray | None:
        if image_measurement is not None:
            if not image_measurement.detected:
                return None
            uv = np.asarray(image_measurement.uv_norm, dtype=float).reshape(2)
            if np.all(np.isfinite(uv)):
                return uv
            return None
        if not snapshot.camera.detected:
            return None
        uv = np.asarray(snapshot.camera.uv_norm, dtype=float).reshape(2)
        return uv if np.all(np.isfinite(uv)) else None

    def _bearing_from_uv(self, snapshot: SimSnapshot, uv: np.ndarray) -> np.ndarray:
        bearing_c = np.array([1.0, float(uv[0]), float(uv[1])], dtype=float)
        bearing_c /= max(float(np.linalg.norm(bearing_c)), 1.0e-12)
        r_b2c = np.asarray(self.camera.body_to_camera, dtype=float).reshape(3, 3)
        bearing_b = r_b2c.T @ bearing_c
        bearing_b /= max(float(np.linalg.norm(bearing_b)), 1.0e-12)
        r_wb = _quat_xyzw_to_rot(snapshot.pursuer.quat_xyzw)
        bearing_w = r_wb @ bearing_b
        return bearing_w / max(float(np.linalg.norm(bearing_w)), 1.0e-12)

    def _correct_uv(self, snapshot: SimSnapshot, uv_norm: np.ndarray) -> bool:
        assert self.x is not None and self.p is not None
        z = np.asarray(uv_norm, dtype=float).reshape(2)
        z_hat = self._project_uv(snapshot, self.x[0:3])
        if z_hat is None:
            return False
        h = self._measurement_jacobian(snapshot, self.x[0:3])
        if h is None:
            return False
        r = np.eye(2, dtype=float) * max(float(self.config.image_noise_norm), 1.0e-9) ** 2
        innovation = z - z_hat
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.pinv(s)
        self.x = self.x + k @ innovation
        i_kh = np.eye(6, dtype=float) - k @ h
        self.p = i_kh @ self.p @ i_kh.T + k @ r @ k.T
        self.p = 0.5 * (self.p + self.p.T)
        return True

    def _correct_size(self, snapshot: SimSnapshot, measurement: TraditionalCvMeasurement) -> bool:
        assert self.x is not None and self.p is not None
        confidence = float(measurement.confidence)
        if not np.isfinite(confidence) or confidence < float(self.config.min_size_confidence):
            return False
        apparent_radius = float(measurement.apparent_radius_px)
        if not np.isfinite(apparent_radius) or apparent_radius <= 1.0e-6:
            return False
        target_radius_m = self._target_radius_m()
        if target_radius_m <= 0.0:
            return False
        focal_px = self._range_focal_px()
        measured_range = target_radius_m * focal_px / apparent_radius
        if not np.isfinite(measured_range) or measured_range <= 0.0:
            return False
        predicted_range = self._range_to_target(snapshot, self.x[0:3])
        if predicted_range is None:
            return False
        h = self._range_jacobian(snapshot, self.x[0:3])
        confidence_scale = 1.0 / max(confidence, float(self.config.min_size_confidence))
        noise = max(float(self.config.size_noise_fraction) * measured_range * confidence_scale, 1.0e-3)
        r = np.array([[noise * noise]], dtype=float)
        innovation = np.array([measured_range - predicted_range], dtype=float)
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.pinv(s)
        self.x = self.x + k @ innovation
        i_kh = np.eye(6, dtype=float) - k @ h
        self.p = i_kh @ self.p @ i_kh.T + k @ r @ k.T
        self.p = 0.5 * (self.p + self.p.T)
        return True

    def _measurement_jacobian(self, snapshot: SimSnapshot, target_p_w: np.ndarray) -> np.ndarray | None:
        h = np.zeros((2, 6), dtype=float)
        base = self._project_uv(snapshot, target_p_w)
        if base is None:
            return None
        eps = 1.0e-5
        for index in range(3):
            perturbed = np.asarray(target_p_w, dtype=float).reshape(3).copy()
            perturbed[index] += eps
            uv = self._project_uv(snapshot, perturbed)
            if uv is None:
                return None
            h[:, index] = (uv - base) / eps
        return h

    def _project_uv(self, snapshot: SimSnapshot, target_p_w: np.ndarray) -> np.ndarray | None:
        r_wb = _quat_xyzw_to_rot(snapshot.pursuer.quat_xyzw)
        pursuer_p_w = np.asarray(snapshot.pursuer.position_w, dtype=float).reshape(3)
        camera_p_w = pursuer_p_w + r_wb @ np.asarray(self.camera.position_b, dtype=float).reshape(3)
        delta_w = np.asarray(target_p_w, dtype=float).reshape(3) - camera_p_w
        r_b2c = np.asarray(self.camera.body_to_camera, dtype=float).reshape(3, 3)
        target_c = r_b2c @ (r_wb.T @ delta_w)
        depth = float(target_c[0])
        if depth <= 1.0e-9:
            return None
        return np.array([target_c[1] / depth, target_c[2] / depth], dtype=float)

    def _bearing_to_target(self, snapshot: SimSnapshot, target_p_w: np.ndarray) -> np.ndarray | None:
        r_wb = _quat_xyzw_to_rot(snapshot.pursuer.quat_xyzw)
        pursuer_p_w = np.asarray(snapshot.pursuer.position_w, dtype=float).reshape(3)
        camera_p_w = pursuer_p_w + r_wb @ np.asarray(self.camera.position_b, dtype=float).reshape(3)
        delta_w = np.asarray(target_p_w, dtype=float).reshape(3) - camera_p_w
        norm = float(np.linalg.norm(delta_w))
        if norm <= 1.0e-9:
            return None
        return delta_w / norm

    def _range_to_target(self, snapshot: SimSnapshot, target_p_w: np.ndarray) -> float | None:
        r_wb = _quat_xyzw_to_rot(snapshot.pursuer.quat_xyzw)
        pursuer_p_w = np.asarray(snapshot.pursuer.position_w, dtype=float).reshape(3)
        camera_p_w = pursuer_p_w + r_wb @ np.asarray(self.camera.position_b, dtype=float).reshape(3)
        distance = float(np.linalg.norm(np.asarray(target_p_w, dtype=float).reshape(3) - camera_p_w))
        return distance if distance > 1.0e-9 else None

    def _range_jacobian(self, snapshot: SimSnapshot, target_p_w: np.ndarray) -> np.ndarray:
        h = np.zeros((1, 6), dtype=float)
        r_wb = _quat_xyzw_to_rot(snapshot.pursuer.quat_xyzw)
        pursuer_p_w = np.asarray(snapshot.pursuer.position_w, dtype=float).reshape(3)
        camera_p_w = pursuer_p_w + r_wb @ np.asarray(self.camera.position_b, dtype=float).reshape(3)
        delta = np.asarray(target_p_w, dtype=float).reshape(3) - camera_p_w
        h[0, 0:3] = delta / max(float(np.linalg.norm(delta)), 1.0e-12)
        return h

    def _target_radius_m(self) -> float:
        if self.instance.config is None or not self.instance.config.targets:
            return 0.0
        return float(self.instance.config.targets[0].radius_m)

    def _range_focal_px(self) -> float:
        intrinsics = self.camera.intrinsics
        return 0.5 * (float(intrinsics.fx_px) + float(intrinsics.fy_px))


def _quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1.0e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _directional_std(covariance: np.ndarray, direction: np.ndarray) -> float:
    unit = np.asarray(direction, dtype=float).reshape(3)
    unit /= max(float(np.linalg.norm(unit)), 1.0e-12)
    variance = float(unit @ np.asarray(covariance, dtype=float).reshape(3, 3) @ unit)
    return float(np.sqrt(max(variance, 0.0)))
