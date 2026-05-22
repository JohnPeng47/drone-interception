"""Camera capture LeafSystem."""

from __future__ import annotations

from pydrake.systems.framework import LeafSystem

from ..drake_compat import capture_value, scene_value
from intercept_sim.sensors import GeometryCamera


class CameraCaptureSystem(LeafSystem):
    def __init__(self, camera: GeometryCamera, dt: float):
        super().__init__()
        self._camera = camera
        self._dt = float(dt)
        self.DeclareAbstractInputPort("scene", scene_value())
        self.DeclareAbstractOutputPort(
            "capture", capture_value, self._calc,
            prerequisites_of_calc={self.time_ticket()},
        )

    def _calc(self, context, output):
        output.set_value(self._camera.maybe_capture(self.GetInputPort("scene").Eval(context)))
