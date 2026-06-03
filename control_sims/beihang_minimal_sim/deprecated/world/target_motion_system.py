"""Deterministic evasive target motion for the minimal task."""

from __future__ import annotations

import math

import numpy as np
from pydrake.systems.framework import LeafSystem

from ...config import TargetConfig
from ..drake_values import target_state_value
from ...types import TargetState


class TargetMotionSystem(LeafSystem):
    def __init__(self, config: TargetConfig):
        super().__init__()
        self._cfg = config
        self.DeclareAbstractOutputPort(
            "target_state",
            target_state_value,
            self._calc,
            prerequisites_of_calc={self.time_ticket()},
        )

    def _calc(self, context, output) -> None:
        t = float(context.get_time())
        p0 = np.asarray(self._cfg.initial_position_w, dtype=float)
        base_v = np.asarray(self._cfg.base_velocity_w, dtype=float)
        ay, az = self._cfg.weave_amplitude_m
        fy, fz = self._cfg.weave_frequency_hz
        wy = 2.0 * math.pi * fy
        wz = 2.0 * math.pi * fz

        offset = np.array([
            0.0,
            ay * math.sin(wy * t),
            az * math.sin(wz * t + 0.7),
        ])
        weave_v = np.array([
            0.0,
            ay * wy * math.cos(wy * t),
            az * wz * math.cos(wz * t + 0.7),
        ])
        output.set_value(
            TargetState(
                t=t,
                position_w=p0 + base_v * t + offset,
                velocity_w=base_v + weave_v,
                radius_m=float(self._cfg.radius_m),
            )
        )
