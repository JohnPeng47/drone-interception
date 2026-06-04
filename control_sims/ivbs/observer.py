from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from backends.csim.bindings.types import SimInstance, SimSnapshot


@dataclass(frozen=True)
class VisualObserverConfig:
    range_prior_m: float = 5.0
    range_prior_std_m: float = 2.0
    target_velocity_prior_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_velocity_prior_std_mps: float = 1.5
    stale_timeout_s: float = 3.0
    metric_position_std_threshold_m: float = 3.0
    metric_velocity_std_threshold_mps: float = 3.0
    metric_range_std_threshold_m: float = 1.5
    process_position_noise_mps: float = 0.15
    process_velocity_noise_mps2: float = 0.4
    image_noise_norm: float = 1.0e-3
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
    ) -> None:
        for raw_slot, instance, snapshot in zip(np.asarray(slots, dtype=np.int64).reshape(-1), instances, snapshots):
            self._slots[int(raw_slot)] = _SlotObserver(instance, snapshot, self.config)

    def stop_slot(self, slot: int) -> None:
        self._slots.pop(int(slot), None)

    def estimate(self, slot: int, instance: SimInstance, snapshot: SimSnapshot, t_s: float) -> RelativeStateEstimate:
        observer = self._slots.get(int(slot))
        if observer is None:
            observer = _SlotObserver(instance, snapshot, self.config)
            self._slots[int(slot)] = observer
        return observer.update(snapshot, t_s)


class _SlotObserver:
    def __init__(self, instance: SimInstance, snapshot: SimSnapshot, config: VisualObserverConfig):
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
        initial_bearing = self._bearing_from_snapshot(snapshot)
        if initial_bearing is not None:
            self._initialize(snapshot, initial_bearing)

    def update(self, snapshot: SimSnapshot, t_s: float) -> RelativeStateEstimate:
        dt = max(0.0, float(t_s) - float(self.last_t))
        if self.x is not None and self.p is not None and dt > 0.0:
            self._predict(dt)
        bearing_w = self._bearing_from_snapshot(snapshot)
        if bearing_w is not None:
            if self.x is None or self.p is None:
                self._initialize(snapshot, bearing_w)
            if self._correct(snapshot):
                self.last_detection_t = float(t_s)
                self.detection_count += 1
            self.last_bearing_w = bearing_w
        self.last_t = float(t_s)
        return self._estimate_from_state(snapshot, t_s)

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
        f[0:3, 3:6] = np.eye(3) * float(dt)
        q = np.zeros((6, 6), dtype=float)
        q_pos = float(self.config.process_position_noise_mps) ** 2 * float(dt)
        q_vel = float(self.config.process_velocity_noise_mps2) ** 2 * float(dt)
        q[0:3, 0:3] = np.eye(3) * q_pos
        q[3:6, 3:6] = np.eye(3) * q_vel
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q
        self.p = 0.5 * (self.p + self.p.T)

    def _estimate_from_state(self, snapshot: SimSnapshot, t_s: float) -> RelativeStateEstimate:
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
        bearing_c = np.array([1.0, float(uv[0]), float(uv[1])], dtype=float)
        bearing_c /= max(float(np.linalg.norm(bearing_c)), 1.0e-12)
        r_b2c = np.asarray(self.camera.body_to_camera, dtype=float).reshape(3, 3)
        bearing_b = r_b2c.T @ bearing_c
        bearing_b /= max(float(np.linalg.norm(bearing_b)), 1.0e-12)
        r_wb = _quat_xyzw_to_rot(snapshot.pursuer.quat_xyzw)
        bearing_w = r_wb @ bearing_b
        return bearing_w / max(float(np.linalg.norm(bearing_w)), 1.0e-12)

    def _correct(self, snapshot: SimSnapshot) -> bool:
        assert self.x is not None and self.p is not None
        z = np.asarray(snapshot.camera.uv_norm, dtype=float).reshape(2)
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
