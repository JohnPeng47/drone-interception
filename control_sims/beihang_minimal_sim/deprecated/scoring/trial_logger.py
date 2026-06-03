"""In-memory trial logger for the minimal task."""

from __future__ import annotations

from pydrake.systems.framework import LeafSystem

from ..drake_values import (
    ctbr_value,
    image_feature_value,
    strategy_observation_value,
    target_state_value,
    trial_metrics_value,
    vehicle_state_value,
)
from ...types import TrialSample


class TrialLogger(LeafSystem):
    def __init__(self, dt: float):
        super().__init__()
        self._samples: list[TrialSample] = []
        self.DeclareAbstractInputPort("vehicle_state", vehicle_state_value())
        self.DeclareAbstractInputPort("target_state", target_state_value())
        self.DeclareAbstractInputPort("image_feature", image_feature_value())
        self.DeclareAbstractInputPort("observation", strategy_observation_value())
        self.DeclareAbstractInputPort("ctbr_cmd", ctbr_value())
        self.DeclareAbstractInputPort("metrics", trial_metrics_value())
        self.DeclarePeriodicPublishEvent(period_sec=float(dt), offset_sec=0.0, publish=self._record)

    def samples(self) -> list[TrialSample]:
        return list(self._samples)

    def final_metrics(self):
        return None if not self._samples else self._samples[-1].metrics

    def _record(self, context) -> None:
        self._samples.append(
            TrialSample(
                t=float(context.get_time()),
                vehicle=self.GetInputPort("vehicle_state").Eval(context),
                target=self.GetInputPort("target_state").Eval(context),
                feature=self.GetInputPort("image_feature").Eval(context),
                observation=self.GetInputPort("observation").Eval(context),
                command=self.GetInputPort("ctbr_cmd").Eval(context),
                metrics=self.GetInputPort("metrics").Eval(context),
            )
        )
