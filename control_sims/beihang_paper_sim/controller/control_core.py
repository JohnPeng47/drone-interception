"""ControlCore — paper Eqs. (12)–(28), 2-step Lyapunov backstepping.

In:  ObserverState — uses image_feature.uv_norm,
     relative_position_w, relative_velocity_w, vehicle_rotation_wb,
     vehicle_state['v'] (interceptor absolute velocity for drag).
Out: CtbrCommand(t, thrust_n, body_rates_b).

Conventions (paper §II-A.2):
    p_r = p_w − p_t,  v_r = v_w − v_t
Gravity g = [0, 0, −9.81] (ENU).

Camera/optical-axis (intercept_sim/scene/visibility.py convention):
    camera-x is the optical axis (depth); uv_norm = [y_c/x_c, z_c/x_c].
    body_to_camera (b→c) maps body vectors to camera frame.
    Paper's R^b_c (camera→body) = body_to_camera.T.

n_t (LOS, world) — paper Eq. (4) second form, image-based:
    n_t_c = unit([1, uv_norm[0], uv_norm[1]])
    n_t   = R_wb · R_b2c.T · n_t_c

n_td (designed LOS, world) — optical axis lifted to world:
    n_td_c = [1, 0, 0]   (camera-x is forward)
    n_td   = R_wb · R_b2c.T · n_td_c
"""

from __future__ import annotations

import numpy as np
from pydrake.systems.framework import LeafSystem

from ..drake_compat import ctbr_value, hover_ctbr, observer_state_value
from ..types import CameraRig, CtbrCommand


DEFAULT_GAINS = {
    "k_b":      0.8,
    # k_1 is the velocity-tracking gain in paper Eq. 19: v_rd = -k_1 · p_r.
    "k_1":      0.1,
    "k_2":      2.0,
    "omega_max": 8.0,
    "f_max":    40.0,
    "drag_diag": (0.10, 0.10, 0.20),
}

G_VEC = np.array([0.0, 0.0, -9.81])


def _vex(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else v / n


class ControlCore(LeafSystem):
    def __init__(
        self,
        mass_kg: float,
        dt: float,
        camera_rig: CameraRig,
        gains: dict | None = None,
    ):
        super().__init__()
        self._m = float(mass_kg)
        self._dt = float(dt)
        g = {**DEFAULT_GAINS, **(gains or {})}
        self._k_b = float(g["k_b"])
        self._k_1 = float(g["k_1"])
        self._k_2 = float(g["k_2"])
        self._omega_max = float(g["omega_max"])
        self._f_max = float(g["f_max"])
        self._D = np.diag(np.asarray(g["drag_diag"], dtype=float))

        # paper's R^b_c (camera→body) = body_to_camera.T in this codebase.
        self._R_c2b = np.asarray(camera_rig.body_to_camera, dtype=float).T
        # n_td: optical axis lifted to body (paper §II-A.4, Eq. 4). Camera-x
        # is depth in intercept_sim/scene/visibility.py, so the optical axis
        # in CCS is [1,0,0]. Lifted to body via R^b_c.
        self._optical_axis_c = np.array([1.0, 0.0, 0.0])
        self._n_td_body = self._R_c2b @ self._optical_axis_c
        self._R_f = np.eye(3)  # thrust along body z (paper Eq. 2 with Rf = I)

        self.DeclareAbstractInputPort("observer_state", observer_state_value())
        self.DeclareAbstractOutputPort(
            "ctbr_cmd", ctbr_value, self._calc,
            prerequisites_of_calc={self.time_ticket()},
        )

    def _calc(self, context, output):
        s = self.GetInputPort("observer_state").Eval(context)
        t = float(context.get_time())

        if (s.relative_position_w is None or s.relative_velocity_w is None
                or s.vehicle_rotation_wb is None):
            output.set_value(hover_ctbr(t, self._m, gravity_mps2=abs(G_VEC[2])))
            return

        p_r = np.asarray(s.relative_position_w, dtype=float)
        v_r = np.asarray(s.relative_velocity_w, dtype=float)
        R_wb = np.asarray(s.vehicle_rotation_wb, dtype=float).reshape(3, 3)

        norm_pr = float(np.linalg.norm(p_r))
        if norm_pr < 1e-6:
            output.set_value(hover_ctbr(t, self._m, gravity_mps2=abs(G_VEC[2])))
            return

        # paper Eq. (4): n_t = -p_r / ‖p_r‖ (first form). The image-based
        # second form is mathematically equivalent when the EKF estimates of
        # p_r and ip̄ are self-consistent (which the DKF enforces by including
        # ip̄ in the state and correcting against image measurements). The
        # first form is numerically more robust at FOV edges, so we use it
        # here — the image error still drives the controller indirectly via
        # the DKF cross-corrections to p̂_r, which is the paper's Fig. 3 path.
        n_t = -p_r / norm_pr

        # n_td (optical axis) lifted to world via the current attitude.
        n_td = R_wb @ self._n_td_body
        n_f = R_wb @ self._R_f @ np.array([0.0, 0.0, 1.0])

        # Saturate z_1 inside the barrier basin so denom stays positive and
        # sign-preserved. Paper Theorem 1 requires |z_1| < k_b; clamping to
        # 0.99·k_b prevents the denom from flipping sign on transient FOV
        # excursions (which would invert b_omega_1 direction).
        z_1 = 1.0 - float(n_td @ n_t)
        z_1 = float(np.clip(z_1, -0.99 * self._k_b, 0.99 * self._k_b))
        denom = self._k_b ** 2 - z_1 ** 2
        barrier = z_1 / denom

        # Eq. (13)
        b_omega_1 = barrier * (R_wb.T @ np.cross(n_td, n_t))

        # Eq. (3) drag uses interceptor absolute velocity v_w in EFCS.
        v_w = np.asarray(
            s.vehicle_state.get("v", np.zeros(3, dtype=float)), dtype=float
        )
        e_f_drag = -R_wb @ self._D @ R_wb.T @ v_w

        # Eq. (19)
        z_2 = v_r + self._k_1 * p_r
        proj = -np.eye(3) + np.outer(n_t, n_t)
        a_d = (
            -self._k_1 * v_r
            -self._k_2 * z_2
            - p_r
            + barrier * (self._m / norm_pr) * (proj @ n_td)
        )

        # Eq. (21)
        n_fd_raw = a_d - G_VEC - e_f_drag / self._m
        n_fd = n_fd_raw / max(float(np.linalg.norm(n_fd_raw)), 1e-9)

        # Eq. (22)
        R_tilt = self._tilt_rotation(n_f, n_fd)
        R_d = R_tilt @ R_wb

        # Eq. (23)
        f_raw = float(n_f @ (self._m * a_d - self._m * G_VEC - e_f_drag))
        f_d = float(np.clip(f_raw, 0.0, self._f_max))

        # Eq. (26)
        S = R_d.T @ R_wb - R_wb.T @ R_d
        b_omega_2 = -_vex(S)

        # Eq. (28)
        b_omega_d = b_omega_1 + b_omega_2
        n_w = float(np.linalg.norm(b_omega_d))
        if n_w > self._omega_max:
            b_omega_d = b_omega_d * (self._omega_max / n_w)

        output.set_value(CtbrCommand(t=t, thrust_n=f_d, body_rates_b=b_omega_d))

    def _tilt_rotation(self, n_f: np.ndarray, n_fd: np.ndarray) -> np.ndarray:
        r = np.cross(n_f, n_fd)
        cos_phi = float(np.clip(n_f @ n_fd, -1.0, 1.0))
        s = float(np.linalg.norm(r))
        if s < 1e-9:
            return np.eye(3)
        r_hat = r / s
        K = np.array([[      0.0, -r_hat[2],  r_hat[1]],
                      [ r_hat[2],       0.0, -r_hat[0]],
                      [-r_hat[1],  r_hat[0],       0.0]])
        phi = float(np.arccos(cos_phi))
        return np.eye(3) + np.sin(phi) * K + (1.0 - np.cos(phi)) * (K @ K)
