"""Shared value types for the paper-pipeline controller systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydrake.common.value import AbstractValue


@dataclass(frozen=True)
class LosGuidanceState:
    valid: bool = False
    t: float = 0.0
    p_r: np.ndarray | None = None
    v_r: np.ndarray | None = None
    R_wb: np.ndarray | None = None
    n_t: np.ndarray | None = None
    n_td: np.ndarray | None = None
    n_f: np.ndarray | None = None
    b_omega_1: np.ndarray | None = None
    barrier: float = 0.0
    norm_pr: float = 0.0
    vehicle_velocity_w: np.ndarray | None = None


@dataclass(frozen=True)
class DesiredAccelerationState:
    valid: bool = False
    t: float = 0.0
    R_wb: np.ndarray | None = None
    n_t: np.ndarray | None = None
    n_f: np.ndarray | None = None
    b_omega_1: np.ndarray | None = None
    a_d: np.ndarray | None = None
    e_f_drag: np.ndarray | None = None


@dataclass(frozen=True)
class ThrustPlanState:
    valid: bool = False
    t: float = 0.0
    R_wb: np.ndarray | None = None
    R_d: np.ndarray | None = None
    b_omega_1: np.ndarray | None = None
    thrust_n: float = 0.0


def los_guidance_value() -> AbstractValue:
    return AbstractValue.Make(LosGuidanceState())


def desired_acceleration_value() -> AbstractValue:
    return AbstractValue.Make(DesiredAccelerationState())


def thrust_plan_value() -> AbstractValue:
    return AbstractValue.Make(ThrustPlanState())
