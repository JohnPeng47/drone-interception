"""Feature detection and delayed measurement LeafSystems."""

from __future__ import annotations

from pydrake.systems.framework import LeafSystem

from ..drake_compat import capture_value, measurements_value
from .perception import FeaturePerceptionModel


class FeatureDetectionSystem(LeafSystem):
    def __init__(self, perception: FeaturePerceptionModel, dt: float):
        super().__init__()
        self._perception = perception
        self._dt = float(dt)
        self.DeclareAbstractInputPort("capture", capture_value())
        self.DeclareAbstractOutputPort(
            "measurements", measurements_value, self._calc,
            prerequisites_of_calc={self.time_ticket()},
        )

    def _calc(self, context, output):
        capture = self.GetInputPort("capture").Eval(context)
        if capture is not None:
            self._perception.submit_capture(capture)
        output.set_value(tuple(self._perception.pop_available(context.get_time())))
