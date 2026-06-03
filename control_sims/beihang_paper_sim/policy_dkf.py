from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from backends.csim.bindings.types import CameraConfig, SimInstance, SimSnapshot
from backends.csim.runner import CtbrCommandBatch, SimControlPolicy, SimRunnerState

from .controller.control_math import DEFAULT_GAINS, G_VEC, vex
from .noise_config import NoiseConfig as PaperNoiseConfig
from .policy import _hover_command, _tilt_rotation


@dataclass(frozen=True)
class _ImageMeasurement:
    t_capture: float
    t_available: float
    uv_norm: np.ndarray


@dataclass(frozen=True)
class _HistoryEntry:
    t: float
    x: np.ndarray
    p: np.ndarray
    gyro_b: np.ndarray
    accel_b: np.ndarray
    dt: float


class BeihangPaperDkfControlPolicy(SimControlPolicy):
    """Paper-style delayed Kalman filter adapter for the C SimRunner.

    The C runner owns the plant. This policy owns the paper's observer/control
    side: delayed image measurements, IMU prediction, historical correction and
    replay, then SO(3) interception control from the filtered state.
    """

    def __init__(self, gains: Mapping[str, float] | None = None):
        self._gains = {**DEFAULT_GAINS, **dict(gains or {})}
        self._slots: dict[int, _SlotDkfController] = {}

    def reset(self, state: SimRunnerState) -> None:
        self._slots.clear()

    def on_slots_started(
        self,
        slots: np.ndarray,
        instances: Sequence[SimInstance],
        state: SimRunnerState,
    ) -> None:
        for raw_slot, instance in zip(np.asarray(slots, dtype=np.int64).reshape(-1), instances):
            slot = int(raw_slot)
            self._slots[slot] = _SlotDkfController(instance, state.snapshot[slot], self._gains)

    def command(self, state: SimRunnerState) -> CtbrCommandBatch:
        thrust_n = np.zeros(len(state.instances), dtype=np.float32)
        body_rates_b = np.zeros((len(state.instances), 3), dtype=np.float32)
        for slot, instance in enumerate(state.instances):
            if instance is None or not bool(state.active[slot]):
                self._slots.pop(slot, None)
                continue
            controller = self._slots.get(slot)
            if controller is None:
                controller = _SlotDkfController(instance, state.snapshot[slot], self._gains)
                self._slots[slot] = controller
            command = controller.command(
                state.snapshot[slot],
                t_s=float(state.elapsed_s[slot]),
            )
            thrust_n[slot] = np.float32(command[0])
            body_rates_b[slot] = np.asarray(command[1], dtype=np.float32).reshape(3)
        return CtbrCommandBatch(thrust_n=thrust_n, body_rates_b=body_rates_b)


class _SlotDkfController:
    Q_IDX = slice(0, 4)
    PR_IDX = slice(4, 7)
    VR_IDX = slice(7, 10)
    IP_IDX = slice(10, 12)
    BG_IDX = slice(12, 15)
    BA_IDX = slice(15, 18)

    def __init__(
        self,
        instance: SimInstance,
        snapshot: SimSnapshot,
        gains: Mapping[str, float],
    ):
        if instance.config is None or not instance.config.cameras:
            raise ValueError("BeihangPaperDkfControlPolicy requires SimInstance.config with a camera")
        self.instance = instance
        self.config = instance.config
        self.camera = instance.config.cameras[0]
        self.gains = dict(gains)
        self.dt_s = float(instance.config.options.backend_dt) * max(1, int(instance.config.options.action_substeps))
        self.noise = _paper_noise_from_config(instance)
        self.image_delay_s = max(0.0, float(instance.config.noise.camera_image_delay_s))
        self.history: deque[_HistoryEntry] = deque(maxlen=max(16, int(np.ceil((self.image_delay_s + 1.0) / self.dt_s)) + 8))
        self.pending_measurements: deque[_ImageMeasurement] = deque()
        self.last_t = 0.0
        self.last_velocity_w = np.asarray(snapshot.pursuer.velocity_w, dtype=float).copy()
        self.x = self._initial_state(snapshot)
        self.p = self._initial_covariance()
        self.history.append(
            _HistoryEntry(
                t=0.0,
                x=self.x.copy(),
                p=self.p.copy(),
                gyro_b=np.asarray(snapshot.pursuer.body_rates_b, dtype=float).copy(),
                accel_b=np.zeros(3, dtype=float),
                dt=0.0,
            )
        )

    def command(self, snapshot: SimSnapshot, *, t_s: float) -> tuple[float, np.ndarray]:
        self._observe_camera(snapshot, t_s)
        self._predict_to(snapshot, t_s)
        self._apply_available_measurements(t_s)
        return self._control_command(snapshot)

    def _initial_state(self, snapshot: SimSnapshot) -> np.ndarray:
        x = np.zeros(18, dtype=float)
        x[self.Q_IDX] = _normalize_quat(np.asarray(snapshot.pursuer.quat_xyzw, dtype=float))
        x[self.PR_IDX] = (
            np.asarray(snapshot.pursuer.position_w, dtype=float)
            - np.asarray(snapshot.target.position_w, dtype=float)
        )
        x[self.VR_IDX] = (
            np.asarray(snapshot.pursuer.velocity_w, dtype=float)
            - np.asarray(snapshot.target.velocity_w, dtype=float)
        )
        if snapshot.camera.detected:
            x[self.IP_IDX] = np.asarray(snapshot.camera.uv_norm, dtype=float)
        else:
            projected = self._project_ip(_quat_xyzw_to_rot(x[self.Q_IDX]), x[self.PR_IDX])
            x[self.IP_IDX] = np.zeros(2, dtype=float) if projected is None else projected
        return x

    def _initial_covariance(self) -> np.ndarray:
        n = self.noise
        p = np.zeros((18, 18), dtype=float)
        for block, value in (
            (self.Q_IDX, n.P0_q),
            (self.PR_IDX, n.P0_pr),
            (self.VR_IDX, n.P0_vr),
            (self.IP_IDX, n.P0_ip),
            (self.BG_IDX, n.P0_b_gyr),
            (self.BA_IDX, n.P0_b_acc),
        ):
            for index in range(block.start, block.stop):
                p[index, index] = float(value)
        return p

    def _observe_camera(self, snapshot: SimSnapshot, t_s: float) -> None:
        if not snapshot.camera.detected:
            return
        uv_norm = np.asarray(snapshot.camera.uv_norm, dtype=float).reshape(2)
        if not np.all(np.isfinite(uv_norm)):
            return
        self.pending_measurements.append(
            _ImageMeasurement(
                t_capture=float(t_s),
                t_available=float(t_s) + self.image_delay_s,
                uv_norm=uv_norm.copy(),
            )
        )

    def _predict_to(self, snapshot: SimSnapshot, t_s: float) -> None:
        dt = float(t_s) - float(self.last_t)
        if dt <= 1.0e-12:
            return
        if dt > 5.0 * self.dt_s:
            dt = self.dt_s
        current_velocity_w = np.asarray(snapshot.pursuer.velocity_w, dtype=float)
        a_world = (current_velocity_w - self.last_velocity_w) / max(dt, 1.0e-9)
        q = _normalize_quat(np.asarray(snapshot.pursuer.quat_xyzw, dtype=float))
        r_wb = _quat_xyzw_to_rot(q)
        gyro_b = np.asarray(snapshot.pursuer.body_rates_b, dtype=float)
        accel_b = r_wb.T @ (a_world - G_VEC)

        self.x, self.p = self._predict(self.x, self.p, gyro_b, accel_b, dt)
        self.history.append(
            _HistoryEntry(
                t=float(t_s),
                x=self.x.copy(),
                p=self.p.copy(),
                gyro_b=gyro_b.copy(),
                accel_b=accel_b.copy(),
                dt=dt,
            )
        )
        self.last_t = float(t_s)
        self.last_velocity_w = current_velocity_w.copy()

    def _apply_available_measurements(self, t_s: float) -> None:
        while self.pending_measurements and self.pending_measurements[0].t_available <= t_s + 1.0e-12:
            measurement = self.pending_measurements.popleft()
            index = self._find_history_index(measurement.t_capture)
            if index is None:
                continue
            entries = list(self.history)
            entry = entries[index]
            x_h, p_h = self._correct(entry.x, entry.p, measurement.uv_norm)
            entries[index] = _HistoryEntry(
                t=entry.t,
                x=x_h.copy(),
                p=p_h.copy(),
                gyro_b=entry.gyro_b,
                accel_b=entry.accel_b,
                dt=entry.dt,
            )
            for replay_index in range(index + 1, len(entries)):
                replay = entries[replay_index]
                x_h, p_h = self._predict(x_h, p_h, replay.gyro_b, replay.accel_b, replay.dt)
                entries[replay_index] = _HistoryEntry(
                    t=replay.t,
                    x=x_h.copy(),
                    p=p_h.copy(),
                    gyro_b=replay.gyro_b,
                    accel_b=replay.accel_b,
                    dt=replay.dt,
                )
            self.history = deque(entries, maxlen=self.history.maxlen)
            self.x = x_h
            self.p = p_h

    def _find_history_index(self, t_capture: float) -> int | None:
        if not self.history:
            return None
        best_index = None
        best_dt = float("inf")
        for index, entry in enumerate(self.history):
            delta = abs(float(entry.t) - float(t_capture))
            if delta < best_dt:
                best_dt = delta
                best_index = index
        return best_index

    def _predict(
        self,
        x: np.ndarray,
        p: np.ndarray,
        gyro_b: np.ndarray,
        accel_b: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_new = self._integrate(x, gyro_b, accel_b, dt)
        f = self._transition_jacobian(x, x_new, dt)
        g = self._process_noise_jacobian(dt)
        q6 = self._bias_noise_covariance()
        q_extra = self._extra_process_noise(dt)
        p_new = f @ p @ f.T + g @ q6 @ g.T + q_extra
        p_new = 0.5 * (p_new + p_new.T)
        return x_new, p_new

    def _integrate(
        self,
        x: np.ndarray,
        gyro_b: np.ndarray,
        accel_b: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        q = _normalize_quat(x[self.Q_IDX])
        p_r = np.asarray(x[self.PR_IDX], dtype=float)
        v_r = np.asarray(x[self.VR_IDX], dtype=float)
        ip_bar = np.asarray(x[self.IP_IDX], dtype=float)
        b_gyr = np.asarray(x[self.BG_IDX], dtype=float)
        b_acc = np.asarray(x[self.BA_IDX], dtype=float)

        omega_b = np.asarray(gyro_b, dtype=float) - b_gyr
        accel_body = np.asarray(accel_b, dtype=float) - b_acc
        dq = _quat_from_omega(omega_b, dt)
        q_new = _normalize_quat(_quat_mul(q, dq))
        r_wb_new = _quat_xyzw_to_rot(q_new)

        accel_world = r_wb_new @ accel_body + G_VEC
        v_r_new = v_r + accel_world * dt
        p_r_new = p_r + 0.5 * (v_r + v_r_new) * dt
        projected_ip = self._project_ip(r_wb_new, p_r_new)
        ip_bar_new = ip_bar if projected_ip is None else projected_ip

        x_new = np.zeros(18, dtype=float)
        x_new[self.Q_IDX] = q_new
        x_new[self.PR_IDX] = p_r_new
        x_new[self.VR_IDX] = v_r_new
        x_new[self.IP_IDX] = ip_bar_new
        x_new[self.BG_IDX] = b_gyr
        x_new[self.BA_IDX] = b_acc
        return x_new

    def _transition_jacobian(
        self,
        x: np.ndarray,
        x_new: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        f = np.eye(18, dtype=float)
        r_wb = _quat_xyzw_to_rot(x_new[self.Q_IDX])
        f[self.PR_IDX, self.VR_IDX] = np.eye(3) * float(dt)
        f[self.VR_IDX, self.BA_IDX] = -r_wb * float(dt)
        f[self.PR_IDX, self.BA_IDX] = -0.5 * r_wb * float(dt) ** 2

        j_ip_pr = self._projection_jacobian_pr(r_wb, x_new[self.PR_IDX])
        if j_ip_pr is not None:
            f[self.IP_IDX, :] = 0.0
            f[self.IP_IDX, self.PR_IDX] = j_ip_pr
            f[self.IP_IDX, self.VR_IDX] = j_ip_pr * float(dt)
            f[self.IP_IDX, self.BA_IDX] = j_ip_pr @ (-0.5 * r_wb * float(dt) ** 2)
        return f

    def _process_noise_jacobian(self, dt: float) -> np.ndarray:
        g = np.zeros((18, 6), dtype=float)
        sqrt_dt = float(np.sqrt(max(dt, 0.0)))
        g[self.BG_IDX, 0:3] = np.eye(3) * sqrt_dt
        g[self.BA_IDX, 3:6] = np.eye(3) * sqrt_dt
        return g

    def _bias_noise_covariance(self) -> np.ndarray:
        n = self.noise
        return np.diag([
            n.sigma_b_gyr**2,
            n.sigma_b_gyr**2,
            n.sigma_b_gyr**2,
            n.sigma_b_acc**2,
            n.sigma_b_acc**2,
            n.sigma_b_acc**2,
        ])

    def _extra_process_noise(self, dt: float) -> np.ndarray:
        n = self.noise
        q = np.zeros((18, 18), dtype=float)
        for block, value in (
            (self.Q_IDX, n.Q_q),
            (self.PR_IDX, n.Q_pr),
            (self.VR_IDX, n.Q_vr),
            (self.IP_IDX, n.Q_ip),
        ):
            for index in range(block.start, block.stop):
                q[index, index] = float(value) * float(dt)
        return q

    def _correct(
        self,
        x: np.ndarray,
        p: np.ndarray,
        uv_norm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h = np.zeros((2, 18), dtype=float)
        h[0, self.IP_IDX.start] = 1.0
        h[1, self.IP_IDX.start + 1] = 1.0
        innovation = np.asarray(uv_norm, dtype=float).reshape(2) - x[self.IP_IDX]
        r = np.eye(2, dtype=float) * max(float(self.noise.sigma_img), 1.0e-6) ** 2
        s = h @ p @ h.T + r
        k = p @ h.T @ np.linalg.pinv(s)
        x_new = x + k @ innovation
        x_new[self.Q_IDX] = _normalize_quat(x_new[self.Q_IDX])
        p_new = (np.eye(18) - k @ h) @ p
        p_new = 0.5 * (p_new + p_new.T)
        return x_new, p_new

    def _control_command(self, snapshot: SimSnapshot) -> tuple[float, np.ndarray]:
        if self.instance.config is None:
            return _hover_command(self.instance)

        mass_kg = float(self.instance.config.pursuer.mass_kg)
        r_wb = _quat_xyzw_to_rot(self.x[self.Q_IDX])
        p_r = np.asarray(self.x[self.PR_IDX], dtype=float)
        v_r = np.asarray(self.x[self.VR_IDX], dtype=float)
        norm_pr = float(np.linalg.norm(p_r))
        if norm_pr < 1.0e-6:
            return _hover_command(self.instance)

        k_b = float(self.gains["k_b"])
        k_1 = float(self.gains["k_1"])
        k_2 = float(self.gains["k_2"])
        f_max = float(self.gains.get("f_max", DEFAULT_GAINS["f_max"]))
        omega_max = float(self.gains.get("omega_max", DEFAULT_GAINS["omega_max"]))
        if self.instance.config.max_thrust_n > 0.0:
            f_max = min(f_max, float(self.instance.config.max_thrust_n))
        if self.instance.config.max_rate_rps > 0.0:
            omega_max = min(omega_max, float(self.instance.config.max_rate_rps))

        n_t = self._los_from_image(r_wb)
        n_td_body = self._camera_to_body(np.array([1.0, 0.0, 0.0], dtype=float))
        n_td = r_wb @ n_td_body
        n_f = r_wb @ np.array([0.0, 0.0, 1.0], dtype=float)

        z_1 = 1.0 - float(n_td @ n_t)
        z_1 = float(np.clip(z_1, -0.99 * k_b, 0.99 * k_b))
        barrier = z_1 / (k_b**2 - z_1**2)
        b_omega_1 = barrier * (r_wb.T @ np.cross(n_td, n_t))

        z_2 = v_r + k_1 * p_r
        proj = -np.eye(3) + np.outer(n_t, n_t)
        a_d = (
            -k_1 * v_r
            - k_2 * z_2
            - p_r
            + barrier * (mass_kg / norm_pr) * (proj @ n_td)
        )
        drag = np.diag(np.asarray(self.gains["drag_diag"], dtype=float).reshape(3))
        v_w = np.asarray(snapshot.pursuer.velocity_w, dtype=float)
        e_f_drag = -r_wb @ drag @ r_wb.T @ v_w

        n_fd_raw = a_d - G_VEC - e_f_drag / mass_kg
        n_fd = n_fd_raw / max(float(np.linalg.norm(n_fd_raw)), 1.0e-9)
        r_tilt = _tilt_rotation(n_f, n_fd)
        r_d = r_tilt @ r_wb
        f_raw = float(n_f @ (mass_kg * a_d - mass_kg * G_VEC - e_f_drag))
        f_d = float(np.clip(f_raw, 0.0, f_max))

        s = r_d.T @ r_wb - r_wb.T @ r_d
        b_omega_2 = -vex(s)
        b_omega_d = b_omega_1 + b_omega_2
        n_w = float(np.linalg.norm(b_omega_d))
        if n_w > omega_max:
            b_omega_d = b_omega_d * (omega_max / n_w)
        return f_d, b_omega_d

    def _project_ip(self, r_wb: np.ndarray, p_r: np.ndarray) -> np.ndarray | None:
        target_delta_w = -np.asarray(p_r, dtype=float)
        camera_offset_w = r_wb @ np.asarray(self.camera.position_b, dtype=float).reshape(3)
        target_delta_camera_origin_w = target_delta_w - camera_offset_w
        target_c = np.asarray(self.camera.body_to_camera, dtype=float).reshape(3, 3) @ (
            r_wb.T @ target_delta_camera_origin_w
        )
        depth = float(target_c[0])
        if depth <= 1.0e-9:
            return None
        return np.array([target_c[1] / depth, target_c[2] / depth], dtype=float)

    def _projection_jacobian_pr(self, r_wb: np.ndarray, p_r: np.ndarray) -> np.ndarray | None:
        r_b2c = np.asarray(self.camera.body_to_camera, dtype=float).reshape(3, 3)
        target_delta_w = -np.asarray(p_r, dtype=float)
        camera_offset_w = r_wb @ np.asarray(self.camera.position_b, dtype=float).reshape(3)
        target_delta_camera_origin_w = target_delta_w - camera_offset_w
        world_to_camera = r_b2c @ r_wb.T
        target_c = world_to_camera @ target_delta_camera_origin_w
        depth = float(target_c[0])
        if depth <= 1.0e-9:
            return None
        inv_depth = 1.0 / depth
        uv_wrt_c = np.array([
            [-target_c[1] * inv_depth * inv_depth, inv_depth, 0.0],
            [-target_c[2] * inv_depth * inv_depth, 0.0, inv_depth],
        ], dtype=float)
        target_c_wrt_pr = -world_to_camera
        return uv_wrt_c @ target_c_wrt_pr

    def _los_from_image(self, r_wb: np.ndarray) -> np.ndarray:
        uv = np.asarray(self.x[self.IP_IDX], dtype=float)
        if not np.all(np.isfinite(uv)):
            p_r = np.asarray(self.x[self.PR_IDX], dtype=float)
            return -p_r / max(float(np.linalg.norm(p_r)), 1.0e-12)
        bearing_c = np.array([1.0, float(uv[0]), float(uv[1])], dtype=float)
        bearing_c /= max(float(np.linalg.norm(bearing_c)), 1.0e-12)
        return r_wb @ self._camera_to_body(bearing_c)

    def _camera_to_body(self, vector_c: np.ndarray) -> np.ndarray:
        vector_b = np.asarray(self.camera.body_to_camera, dtype=float).reshape(3, 3).T @ np.asarray(vector_c, dtype=float)
        return vector_b / max(float(np.linalg.norm(vector_b)), 1.0e-12)


def _paper_noise_from_config(instance: SimInstance) -> PaperNoiseConfig:
    defaults = PaperNoiseConfig()
    if instance.config is None:
        return defaults
    sim_noise = instance.config.noise
    pixel_sigma = max(float(sim_noise.pixel_noise_std_px[0]), float(sim_noise.pixel_noise_std_px[1]), 0.0)
    camera: CameraConfig | None = instance.config.cameras[0] if instance.config.cameras else None
    sigma_img = float(sim_noise.sigma_img)
    if sigma_img <= 0.0 and pixel_sigma > 0.0 and camera is not None:
        focal = max(float(camera.intrinsics.fx_px), float(camera.intrinsics.fy_px), 1.0)
        sigma_img = pixel_sigma / focal
    return PaperNoiseConfig(
        sigma_gyr=max(float(sim_noise.sigma_gyr), defaults.sigma_gyr),
        sigma_acc=max(float(sim_noise.sigma_acc), defaults.sigma_acc),
        sigma_b_gyr=max(float(sim_noise.sigma_b_gyr), defaults.sigma_b_gyr),
        sigma_b_acc=max(float(sim_noise.sigma_b_acc), defaults.sigma_b_acc),
        bias_init_std=max(float(sim_noise.bias_init_std), defaults.bias_init_std),
        sigma_img=max(sigma_img, defaults.sigma_img),
        rng_seed=int(sim_noise.rng_seed),
    )


def _quat_from_omega(omega_b: np.ndarray, dt: float) -> np.ndarray:
    omega = np.asarray(omega_b, dtype=float).reshape(3)
    angle = float(np.linalg.norm(omega) * dt)
    if angle < 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    axis = omega / max(float(np.linalg.norm(omega)), 1.0e-12)
    half = 0.5 * angle
    return np.array([
        axis[0] * np.sin(half),
        axis[1] * np.sin(half),
        axis[2] * np.sin(half),
        np.cos(half),
    ], dtype=float)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.asarray(q1, dtype=float).reshape(4)
    x2, y2, z2, w2 = np.asarray(q2, dtype=float).reshape(4)
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=float)


def _normalize_quat(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / norm


def _quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quat(q_xyzw)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)
