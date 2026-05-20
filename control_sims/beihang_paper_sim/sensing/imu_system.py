"""ImuSystem — paper Eqs. (7)–(10), with Gaussian noise + Wiener-walked biases.

    Eq. (7):  bω = ω_gyr − b_gyr − n_gyr     (gyro model)
    Eq. (8):  ḃ_gyr = n_b_gyr                (gyro bias Wiener)
    Eq. (9):  ea = g·e3 + R_e^b (acc − b_acc − n_acc)   (accel model)
    Eq. (10): ḃ_acc = n_b_acc                (accel bias Wiener)

So the published outputs are:
    ω_gyr = bω_truth + b_gyr + n_gyr
    acc   = R_e^bᵀ (a_world_truth − g·e3) + b_acc + n_acc

a_world is approximated by numerical differentiation of v_w across one tick.
Gravity vector follows ENU convention: g·e3 = [0, 0, −9.81].
"""

from __future__ import annotations

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import BasicVector, LeafSystem

from ..noise_config import NoiseConfig
from ..drake_compat import vehicle_state_value


def _quat_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


class ImuSystem(LeafSystem):
    G = np.array([0.0, 0.0, -9.81])  # ENU, gravity points -z (paper Eq. 9)

    def __init__(self, dt: float, noise_config: NoiseConfig | None = None):
        super().__init__()
        self._dt = float(dt)
        self._cfg = noise_config or NoiseConfig()
        self._rng = np.random.default_rng(self._cfg.rng_seed)

        self.DeclareAbstractInputPort("vehicle_state_dict", vehicle_state_value())

        # Discrete state: previous v_w, [b_gyr, b_acc] biases, last meas.
        self._v_prev_idx = self.DeclareAbstractState(
            AbstractValue.Make(np.zeros(3, dtype=float))
        )
        b_gyr_0 = self._rng.normal(0.0, self._cfg.bias_init_std, 3)
        b_acc_0 = self._rng.normal(0.0, self._cfg.bias_init_std, 3)
        self._biases_idx = self.DeclareAbstractState(
            AbstractValue.Make(np.concatenate([b_gyr_0, b_acc_0]))
        )
        self._last_meas_idx = self.DeclareAbstractState(
            AbstractValue.Make(np.zeros(6, dtype=float))
        )
        self._initialized_idx = self.DeclareAbstractState(
            AbstractValue.Make(np.array([0.0]))  # 0 = not yet, 1 = initialized
        )

        self.DeclareVectorOutputPort("gyro_b", BasicVector(3), self._gyro_out)
        self.DeclareVectorOutputPort("accel_b", BasicVector(3), self._accel_out)
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=self._dt, offset_sec=0.0, update=self._step,
        )

    def _step(self, context, state):
        vs = self.GetInputPort("vehicle_state_dict").Eval(context)
        v_now = np.asarray(vs.get("v", np.zeros(3)), dtype=float).copy()
        omega_truth = np.asarray(vs.get("w", np.zeros(3)), dtype=float)
        q = np.asarray(vs.get("q", np.array([0., 0., 0., 1.])), dtype=float)
        R_wb = _quat_to_rot(q)

        init_flag = state.get_mutable_abstract_state(self._initialized_idx).get_value().copy()
        v_prev = state.get_mutable_abstract_state(self._v_prev_idx).get_value().copy()
        if init_flag[0] < 0.5:
            # First tick: no valid v_prev → assume at rest, specific force = -g.
            a_world = np.zeros(3)
            init_flag[0] = 1.0
        else:
            a_world = (v_now - v_prev) / max(self._dt, 1e-9)
        acc_truth_body = R_wb.T @ (a_world - self.G)  # paper Eq. 9 inverted

        # Walk biases via Wiener increments (per √dt) — Eqs. (8), (10).
        biases = state.get_mutable_abstract_state(self._biases_idx).get_value().copy()
        sqrt_dt = float(np.sqrt(self._dt))
        biases[:3] += self._rng.normal(0.0, self._cfg.sigma_b_gyr * sqrt_dt, 3)
        biases[3:] += self._rng.normal(0.0, self._cfg.sigma_b_acc * sqrt_dt, 3)

        # White measurement noise — Eqs. (7), (9).
        n_gyr = self._rng.normal(0.0, self._cfg.sigma_gyr, 3)
        n_acc = self._rng.normal(0.0, self._cfg.sigma_acc, 3)

        gyro_meas = omega_truth + biases[:3] + n_gyr
        accel_meas = acc_truth_body + biases[3:] + n_acc

        state.get_mutable_abstract_state(self._initialized_idx).set_value(init_flag)
        state.get_mutable_abstract_state(self._v_prev_idx).set_value(v_now)
        state.get_mutable_abstract_state(self._biases_idx).set_value(biases)
        state.get_mutable_abstract_state(self._last_meas_idx).set_value(
            np.concatenate([gyro_meas, accel_meas])
        )

    def _gyro_out(self, context, output):
        meas = context.get_abstract_state(self._last_meas_idx).get_value()
        output.SetFromVector(meas[:3])

    def _accel_out(self, context, output):
        meas = context.get_abstract_state(self._last_meas_idx).get_value()
        output.SetFromVector(meas[3:])
