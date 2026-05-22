from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PUFFER_DT = 0.002
PUFFER_ACTION_SUBSTEPS = 5
PUFFER_ACTION_DT = PUFFER_DT * PUFFER_ACTION_SUBSTEPS
DEFAULT_MAX_VEL_MPS = 100.0
DEFAULT_MAX_OMEGA_RPS = 100.0


@dataclass(frozen=True)
class PursuerParams:
    mass_kg: float
    ixx: float
    iyy: float
    izz: float
    arm_len_m: float
    k_thrust: float
    k_yaw: float
    k_ang_damp: float = 0.0
    b_drag: float = 0.0
    gravity_mps2: float = 9.81
    max_rpm: float = 21702.0
    max_vel_mps: float = DEFAULT_MAX_VEL_MPS
    max_omega_rps: float = DEFAULT_MAX_OMEGA_RPS
    motor_tau_s: float = 0.15
    rpm_min: float | None = None
    rotor_positions_b: np.ndarray | None = None
    rotor_directions: np.ndarray | None = None
    k_w: float = 1.0


@dataclass(frozen=True)
class PursuerInitialState:
    position_w: np.ndarray
    velocity_w: np.ndarray
    quat_xyzw: np.ndarray
    body_rates_b: np.ndarray
    rotor_speeds: np.ndarray | None = None
    wind_w: np.ndarray | None = None


VehicleParams = PursuerParams
InitialState = PursuerInitialState
