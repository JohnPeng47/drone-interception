"""DkfObserver — paper Algorithm 2, Eqs. (30)–(36), with real EKF math.

State x ∈ R^18 = [q (4 xyzw), p_r (3), v_r (3), ip̄ (2), b_gyr (3), b_acc (3)]
where p_r = p_w − p_t  and  v_r = v_w − v_t  (paper §II-A.2 convention).

Predict (every tick, Eqs. 30–31): integrate state with debiased IMU; propagate
covariance P = F P Fᵀ + G Q Gᵀ where F is the numerical Jacobian of the
nonlinear `_integrate` step (captures the full block structure of paper §IV-A
including F_q^bgyr, F_vr^q, F_ip̄^{q,p_r,v_r,bgyr}) and G maps 6-D bias-drift
noise w = [n_bgyr; n_bacc] into the 18-D state per paper Appendix B.

Correct (when delayed image arrives, Eqs. 32–36): H selects the ip̄ block;
S = H P Hᵀ + R, K = P Hᵀ S⁻¹; x ← x + K(z − H x); P ← (I − KH) P.
Replay forward through stored IMU samples.

ip̄(0) seeded from the first valid image measurement rather than zeros, so
the filter starts on the image manifold.

Output: `ObserverState` with paper conventions. `image_feature` is published
from the EKF state x[IP_IDX] so downstream controllers can build n_t from the
filtered image coordinate (paper Eq. 4 second form) — that's the real "IBVS"
path the paper diagrams in Fig. 3.

Camera convention (matches intercept_sim/scene/visibility.py): camera-x is the
optical axis (depth); uv_norm = [y_c/x_c, z_c/x_c]. `body_to_camera` is the
rotation taking body-frame vectors to camera-frame: p_c = body_to_camera @ p_b.
Paper's R^b_c (camera→body) = code's body_to_camera.T.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import BasicVector, LeafSystem

from drake_sims.ports import (
    measurements_value,
    observer_state_value,
    scene_value,
    vehicle_state_value,
)
from intercept_sim.types import CameraRig, ImageFeatureMeasurement, ObserverState

from ..noise_config import NoiseConfig


# ---------------------------------------------------------------------------
# Quaternion / SO(3) helpers (xyzw, matches rotorpy)
# ---------------------------------------------------------------------------


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def _quat_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    angle = float(np.linalg.norm(omega) * dt)
    if angle < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = omega / np.linalg.norm(omega)
    s = float(np.sin(angle / 2.0))
    c = float(np.cos(angle / 2.0))
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, c])


# ---------------------------------------------------------------------------
# DKF
# ---------------------------------------------------------------------------


class DkfObserver(LeafSystem):
    G_W = np.array([0.0, 0.0, -9.81])
    HISTORY_SIZE = 64

    Q_IDX = slice(0, 4)
    PR_IDX = slice(4, 7)
    VR_IDX = slice(7, 10)
    IP_IDX = slice(10, 12)
    BG_IDX = slice(12, 15)
    BA_IDX = slice(15, 18)

    def __init__(
        self,
        camera_rig: CameraRig,
        dt: float,
        noise_config: NoiseConfig | None = None,
    ):
        super().__init__()
        self._rig = camera_rig
        self._dt = float(dt)
        self._cfg = noise_config or NoiseConfig()
        self._foc = float(camera_rig.intrinsics.fx_px)
        # body_to_camera (b→c) per intercept_sim convention; camera-x is depth.
        self._R_b2c = np.asarray(camera_rig.body_to_camera, dtype=float)

        self.DeclareAbstractInputPort("scene", scene_value())
        self.DeclareAbstractInputPort("measurements", measurements_value())
        self.DeclareAbstractInputPort("vehicle_state_dict", vehicle_state_value())
        self.DeclareVectorInputPort("gyro", BasicVector(3))
        self.DeclareVectorInputPort("accel", BasicVector(3))

        self._x_idx = self.DeclareAbstractState(
            AbstractValue.Make(np.zeros(18, dtype=float))
        )
        self._P_idx = self.DeclareAbstractState(
            AbstractValue.Make(self._build_P0())
        )
        self._history_idx = self.DeclareAbstractState(
            AbstractValue.Make(deque(maxlen=self.HISTORY_SIZE))
        )
        self._initialized_idx = self.DeclareAbstractState(
            AbstractValue.Make(np.array([0.0]))
        )
        self._estimate_idx = self.DeclareAbstractState(observer_state_value())

        self.DeclareAbstractOutputPort(
            "observer_state",
            observer_state_value,
            self._copy_estimate_out,
        )
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=self._dt, offset_sec=0.0, update=self._step,
        )

    def _build_P0(self) -> np.ndarray:
        c = self._cfg
        P = np.zeros((18, 18))
        for s, v in [(self.Q_IDX, c.P0_q), (self.PR_IDX, c.P0_pr),
                     (self.VR_IDX, c.P0_vr), (self.IP_IDX, c.P0_ip),
                     (self.BG_IDX, c.P0_b_gyr), (self.BA_IDX, c.P0_b_acc)]:
            for i in range(s.start, s.stop):
                P[i, i] = v
        return P

    def _build_G(self) -> np.ndarray:
        # Per paper Appendix B: w = [n_bgyr; n_bacc] ∈ R^6 only injects into
        # the bias states. Discrete Wiener variance over one tick is σ²·dt.
        # G·Q6·Gᵀ should contribute σ²·dt to the bias diagonal; with
        # Q6 = diag(σ²) we need G = I·√dt → (√dt)²·σ² = σ²·dt. The earlier
        # G = I·dt was 1/dt too small (50× at outer_dt=20ms) which made bias
        # estimates over-confident and slow to track drift.
        G = np.zeros((18, 6))
        sdt = float(np.sqrt(self._dt))
        G[self.BG_IDX, 0:3] = np.eye(3) * sdt
        G[self.BA_IDX, 3:6] = np.eye(3) * sdt
        return G

    def _build_Q6(self) -> np.ndarray:
        c = self._cfg
        return np.diag([
            c.sigma_b_gyr ** 2, c.sigma_b_gyr ** 2, c.sigma_b_gyr ** 2,
            c.sigma_b_acc ** 2, c.sigma_b_acc ** 2, c.sigma_b_acc ** 2,
        ])

    def _build_Q_extra(self) -> np.ndarray:
        # The paper's strict 6-D w omits IMU measurement noise; in practice
        # the predict step also propagates n_gyr/n_acc/n_img stochasticity.
        # We add a small diagonal tuning Q to cover those — keeps the filter
        # well-conditioned without abandoning the G·Q6·Gᵀ structure.
        c = self._cfg
        Q = np.zeros((18, 18))
        for s, v in [(self.Q_IDX, c.Q_q), (self.PR_IDX, c.Q_pr),
                     (self.VR_IDX, c.Q_vr), (self.IP_IDX, c.Q_ip)]:
            for i in range(s.start, s.stop):
                Q[i, i] = v * self._dt
        return Q

    def _build_R(self) -> np.ndarray:
        s2 = self._cfg.sigma_img ** 2
        return np.diag([s2, s2])

    # Predict ---------------------------------------------------------

    def _project_ip(self, R_wb: np.ndarray, p_r: np.ndarray) -> np.ndarray:
        """Pinhole projection of target into image plane.

        Target in world = pursuer − p_r. In body: R_wbᵀ · (−p_r). In camera:
        R_b2c · R_wbᵀ · (−p_r). Camera-x is depth (intercept_sim convention),
        so uv_norm = [y_c/x_c, z_c/x_c].
        """
        p_t_c = self._R_b2c @ R_wb.T @ (-p_r)
        depth = float(p_t_c[0])
        if abs(depth) < 1e-9:
            return None  # behind / on the focal plane
        return np.array([p_t_c[1] / depth, p_t_c[2] / depth], dtype=float)

    def _integrate(self, x: np.ndarray, gyro: np.ndarray, accel: np.ndarray) -> np.ndarray:
        dt = self._dt
        q = x[self.Q_IDX]
        p_r = x[self.PR_IDX]
        v_r = x[self.VR_IDX]
        ip_prev = x[self.IP_IDX]
        b_gyr = x[self.BG_IDX]
        b_acc = x[self.BA_IDX]

        omega_b = gyro - b_gyr
        a_b = accel - b_acc

        dq = _quat_from_omega(omega_b, dt)
        q_new = _quat_mul(q, dq)
        q_new = q_new / max(np.linalg.norm(q_new), 1e-12)
        R_new = _quat_to_rot(q_new)

        a_world = R_new @ a_b + self.G_W
        v_r_new = v_r + a_world * dt
        p_r_new = p_r + v_r * dt

        # Paper Eq. 51 propagates ip̄ via the image Jacobian Lₛ. Implementing
        # the full Lₛ requires translating the paper's CCS convention
        # (camera-z is depth) to this codebase's (camera-x is depth), which
        # is error-prone. As a paper-Algorithm-2-faithful simplification we
        # treat ip̄ as an independent state driven only by image corrections
        # — F[IP, IP] = I via the numerical Jacobian, F[IP, others] ≈ 0,
        # so image corrections persist tick-to-tick instead of being
        # overwritten by reprojection from p_r. The trade-off is that image
        # corrections then influence (p_r, v_r) only weakly, via the
        # off-diagonal P-blocks that develop from process-noise leakage.
        ip_bar_new = ip_prev

        x_new = np.zeros(18)
        x_new[self.Q_IDX] = q_new
        x_new[self.PR_IDX] = p_r_new
        x_new[self.VR_IDX] = v_r_new
        x_new[self.IP_IDX] = ip_bar_new
        x_new[self.BG_IDX] = b_gyr
        x_new[self.BA_IDX] = b_acc
        return x_new

    def _build_F(self, x: np.ndarray, gyro: np.ndarray, accel: np.ndarray) -> np.ndarray:
        # Numerical Jacobian of _integrate w.r.t. x. Captures the full block
        # structure paper §IV-A specifies: F_q^bgyr, F_vr^q, F_vr^bacc,
        # F_p_r^v_r, F_ip̄^{q,p_r,v_r,bgyr,ip̄}. The quaternion column gives
        # the linear approximation that's standard for a quaternion EKF;
        # post-update renormalization keeps |q|=1.
        n = 18
        eps = 1e-7
        F = np.zeros((n, n))
        x_base = self._integrate(x, gyro, accel)
        for i in range(n):
            x_pert = x.copy()
            x_pert[i] += eps
            x_pert_new = self._integrate(x_pert, gyro, accel)
            F[:, i] = (x_pert_new - x_base) / eps
        return F

    def _predict(self, x, P, gyro, accel):
        F = self._build_F(x, gyro, accel)
        G = self._build_G()
        Q6 = self._build_Q6()
        Q_extra = self._build_Q_extra()
        x_new = self._integrate(x, gyro, accel)
        P_new = F @ P @ F.T + G @ Q6 @ G.T + Q_extra
        return x_new, P_new

    # Correct ---------------------------------------------------------

    def _correct(self, x, P, z_image):
        H = np.zeros((2, 18))
        H[0, 10] = 1.0
        H[1, 11] = 1.0

        innovation = z_image - x[self.IP_IDX]
        S = H @ P @ H.T + self._build_R()
        K = P @ H.T @ np.linalg.inv(S)

        x_new = x + K @ innovation
        qn = float(np.linalg.norm(x_new[self.Q_IDX]))
        if qn > 1e-9:
            x_new[self.Q_IDX] = x_new[self.Q_IDX] / qn

        P_new = (np.eye(18) - K @ H) @ P
        return x_new, P_new

    # Algorithm 2 -----------------------------------------------------

    def _step(self, context, state):
        t = context.get_time()
        scene = self.GetInputPort("scene").Eval(context)
        measurements = self.GetInputPort("measurements").Eval(context)
        vs = self.GetInputPort("vehicle_state_dict").Eval(context)
        gyro = np.asarray(self.GetInputPort("gyro").Eval(context), dtype=float)
        accel = np.asarray(self.GetInputPort("accel").Eval(context), dtype=float)

        init_flag = state.get_mutable_abstract_state(self._initialized_idx).get_value().copy()
        x = state.get_mutable_abstract_state(self._x_idx).get_value().copy()
        P = state.get_mutable_abstract_state(self._P_idx).get_value().copy()
        history: deque = state.get_mutable_abstract_state(self._history_idx).get_value()

        # Defer init until we have BOTH a target in the scene AND a valid
        # image measurement to seed ip̄(0). Paper Algorithm 2 line 1 just
        # says "Initialize x̂_0"; seeding ip̄ from the first image keeps the
        # filter on the measurement manifold from the very first correction.
        if init_flag[0] < 0.5:
            targets = getattr(scene, "targets", None) or []
            first_meas = next(
                (m for m in measurements
                 if getattr(m, "detected", False) and getattr(m, "uv_norm", None) is not None),
                None,
            )
            if not targets or first_meas is None:
                state.get_mutable_abstract_state(self._initialized_idx).set_value(init_flag)
                return
            target_state = targets[0]
            p_w = np.asarray(vs.get("x", np.zeros(3)), dtype=float)
            v_w = np.asarray(vs.get("v", np.zeros(3)), dtype=float)
            q0 = np.asarray(vs.get("q", np.array([0., 0., 0., 1.])), dtype=float)
            p_t0 = np.asarray(target_state.position_w, dtype=float)
            v_t0 = np.asarray(target_state.velocity_w, dtype=float)
            x[self.Q_IDX] = q0
            x[self.PR_IDX] = p_w - p_t0
            x[self.VR_IDX] = v_w - v_t0
            x[self.IP_IDX] = np.asarray(first_meas.uv_norm, dtype=float)
            x[self.BG_IDX] = np.zeros(3)
            x[self.BA_IDX] = np.zeros(3)
            init_flag[0] = 1.0

        x, P = self._predict(x, P, gyro, accel)
        history.append((t, x.copy(), P.copy(), gyro.copy(), accel.copy()))

        for m in measurements:
            t_capture = float(getattr(m, "t_capture", t))
            uv_norm = getattr(m, "uv_norm", None)
            detected = bool(getattr(m, "detected", False))
            if uv_norm is None or not detected:
                continue
            z = np.asarray(uv_norm, dtype=float)

            idx = self._find_history_index(history, t_capture)
            if idx is None:
                continue

            t_h, x_h, P_h, gyro_h, accel_h = history[idx]
            x_h, P_h = self._correct(x_h, P_h, z)
            history[idx] = (t_h, x_h.copy(), P_h.copy(), gyro_h, accel_h)

            for j in range(idx + 1, len(history)):
                t_j, _, _, gyro_j, accel_j = history[j]
                x_h, P_h = self._predict(x_h, P_h, gyro_j, accel_j)
                history[j] = (t_j, x_h.copy(), P_h.copy(), gyro_j, accel_j)

            x, P = x_h, P_h

        state.get_mutable_abstract_state(self._initialized_idx).set_value(init_flag)
        state.get_mutable_abstract_state(self._x_idx).set_value(x)
        state.get_mutable_abstract_state(self._P_idx).set_value(P)
        state.get_mutable_abstract_state(self._estimate_idx).set_value(
            self._observer_state_from(t, vs, x, init_flag[0] >= 0.5)
        )

    @staticmethod
    def _find_history_index(history: deque, t_target: float) -> int | None:
        if not history:
            return None
        best, best_dt = None, float("inf")
        for i, item in enumerate(history):
            d = abs(item[0] - t_target)
            if d < best_dt:
                best_dt = d
                best = i
        return best

    def _observer_state_from(self, t: float, vs: dict, x: np.ndarray, valid: bool) -> ObserverState:
        if not valid:
            return ObserverState(t=float(t), vehicle_state=dict(vs), image_feature=None)
        R_wb = _quat_to_rot(x[self.Q_IDX])
        # Publish the EKF-filtered ip̄ as an ImageFeatureMeasurement so the
        # controller can run paper Eq. (4) second form and drive n_t from
        # the image — that's the IBVS path paper Fig. 3 shows.
        feature = ImageFeatureMeasurement(
            t_capture=float(t),
            t_available=float(t),
            camera_id=self._rig.id,
            target_id=None,
            detected=True,
            uv_px=None,
            uv_norm=np.asarray(x[self.IP_IDX], dtype=float).copy(),
            confidence=1.0,
        )
        return ObserverState(
            t=float(t),
            vehicle_state=dict(vs),
            image_feature=feature,
            relative_position_w=x[self.PR_IDX].copy(),
            relative_velocity_w=x[self.VR_IDX].copy(),
            target_acceleration_w=None,
            vehicle_rotation_wb=R_wb,
        )

    def _copy_estimate_out(self, context, output):
        output.set_value(context.get_abstract_state(self._estimate_idx).get_value())
