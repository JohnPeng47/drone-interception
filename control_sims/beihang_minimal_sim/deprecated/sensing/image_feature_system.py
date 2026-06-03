"""Build a compact strategy observation from image and ownship state."""

from __future__ import annotations

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import LeafSystem

from ..drake_values import (
    image_feature_value,
    strategy_observation_value,
    vehicle_state_value,
)
from ...types import StrategyObservation


class ImageFeatureSystem(LeafSystem):
    def __init__(self, dt: float):
        super().__init__()
        self._dt = float(dt)
        self._last_uv_idx = self.DeclareAbstractState(
            AbstractValue.Make(np.array([np.nan, np.nan], dtype=float))
        )
        self._obs_idx = self.DeclareAbstractState(strategy_observation_value())
        self.DeclareAbstractInputPort("image_feature", image_feature_value())
        self.DeclareAbstractInputPort("vehicle_state", vehicle_state_value())
        self.DeclareAbstractOutputPort(
            "observation",
            strategy_observation_value,
            self._copy_observation,
            prerequisites_of_calc={self.abstract_state_ticket(self._obs_idx)},
        )
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=self._dt, offset_sec=0.0, update=self._step,
        )

    def _step(self, context, state) -> None:
        feature = self.GetInputPort("image_feature").Eval(context)
        vehicle = self.GetInputPort("vehicle_state").Eval(context)
        last_uv = state.get_mutable_abstract_state(self._last_uv_idx).get_value().copy()

        uv_dot = np.zeros(2)
        if feature.detected and feature.uv_norm is not None:
            uv = np.asarray(feature.uv_norm, dtype=float)
            if np.all(np.isfinite(last_uv)):
                uv_dot = (uv - last_uv) / self._dt
            next_last = uv
        else:
            next_last = np.array([np.nan, np.nan], dtype=float)

        obs = StrategyObservation(
            t=float(context.get_time()),
            detected=bool(feature.detected),
            uv_norm=None if feature.uv_norm is None else np.asarray(feature.uv_norm, dtype=float),
            uv_dot_norm=uv_dot,
            depth_m=feature.depth_m,
            vehicle_velocity_w=np.asarray(vehicle.velocity_w, dtype=float),
            vehicle_rotation_wb=np.asarray(vehicle.rotation_wb, dtype=float),
        )
        state.get_mutable_abstract_state(self._last_uv_idx).set_value(next_last)
        state.get_mutable_abstract_state(self._obs_idx).set_value(obs)

    def _copy_observation(self, context, output) -> None:
        output.set_value(context.get_abstract_state(self._obs_idx).get_value())
