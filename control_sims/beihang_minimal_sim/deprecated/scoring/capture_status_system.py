"""Trial metrics for the minimal interception task."""

from __future__ import annotations

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import LeafSystem

from ...config import TrialConfig
from ..drake_values import (
    ctbr_value,
    image_feature_value,
    scene_state_value,
    trial_metrics_value,
)
from ...types import TrialMetrics


class CaptureStatusSystem(LeafSystem):
    def __init__(self, config: TrialConfig):
        super().__init__()
        self._cfg = config
        self._dt = float(config.dt)
        self._min_distance_idx = self.DeclareAbstractState(
            AbstractValue.Make(np.array([float("inf")], dtype=float))
        )
        self._capture_time_idx = self.DeclareAbstractState(
            AbstractValue.Make(np.array([np.nan], dtype=float))
        )
        self._effort_idx = self.DeclareAbstractState(
            AbstractValue.Make(np.array([0.0], dtype=float))
        )
        self._metrics_idx = self.DeclareAbstractState(trial_metrics_value())
        self.DeclareAbstractInputPort("scene", scene_state_value())
        self.DeclareAbstractInputPort("image_feature", image_feature_value())
        self.DeclareAbstractInputPort("ctbr_cmd", ctbr_value())
        self.DeclareAbstractOutputPort(
            "metrics",
            trial_metrics_value,
            self._copy_metrics,
            prerequisites_of_calc={self.abstract_state_ticket(self._metrics_idx)},
        )
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=self._dt, offset_sec=0.0, update=self._step,
        )

    def _step(self, context, state) -> None:
        scene = self.GetInputPort("scene").Eval(context)
        feature = self.GetInputPort("image_feature").Eval(context)
        cmd = self.GetInputPort("ctbr_cmd").Eval(context)
        t = float(context.get_time())

        distance = float(np.linalg.norm(scene.target.position_w - scene.vehicle.position_w))
        min_distance_arr = state.get_mutable_abstract_state(self._min_distance_idx).get_value().copy()
        min_distance_arr[0] = min(float(min_distance_arr[0]), distance)

        capture_time_arr = state.get_mutable_abstract_state(self._capture_time_idx).get_value().copy()
        captured_now = distance <= self._cfg.capture_radius_m
        if captured_now and not np.isfinite(capture_time_arr[0]):
            capture_time_arr[0] = t

        effort_arr = state.get_mutable_abstract_state(self._effort_idx).get_value().copy()
        effort_arr[0] += float(np.linalg.norm(cmd.body_rates_b) + 0.02 * abs(cmd.thrust_n)) * self._dt

        arena_min = np.asarray(self._cfg.arena_min_w, dtype=float)
        arena_max = np.asarray(self._cfg.arena_max_w, dtype=float)
        p = np.asarray(scene.vehicle.position_w, dtype=float)
        out_of_bounds = bool(np.any(p < arena_min) or np.any(p > arena_max))
        crashed = bool(p[2] <= arena_min[2] + 1e-6)
        image_error = (
            None if not feature.detected or feature.uv_norm is None
            else float(np.linalg.norm(feature.uv_norm))
        )
        metrics = TrialMetrics(
            t=t,
            distance_m=distance,
            min_distance_m=float(min_distance_arr[0]),
            captured=bool(np.isfinite(capture_time_arr[0])),
            capture_time_s=(
                None if not np.isfinite(capture_time_arr[0])
                else float(capture_time_arr[0])
            ),
            in_view=bool(feature.detected),
            image_error=image_error,
            control_effort=float(effort_arr[0]),
            crashed=crashed,
            out_of_bounds=out_of_bounds,
        )
        state.get_mutable_abstract_state(self._min_distance_idx).set_value(min_distance_arr)
        state.get_mutable_abstract_state(self._capture_time_idx).set_value(capture_time_arr)
        state.get_mutable_abstract_state(self._effort_idx).set_value(effort_arr)
        state.get_mutable_abstract_state(self._metrics_idx).set_value(metrics)

    def _copy_metrics(self, context, output) -> None:
        output.set_value(context.get_abstract_state(self._metrics_idx).get_value())
