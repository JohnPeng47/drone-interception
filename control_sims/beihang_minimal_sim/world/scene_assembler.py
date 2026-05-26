"""Assemble vehicle and target state into a scene value."""

from __future__ import annotations

from pydrake.systems.framework import LeafSystem

from ..drake_values import scene_state_value, target_state_value, vehicle_state_value
from ..types import SceneState


class SceneAssembler(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareAbstractInputPort("vehicle_state", vehicle_state_value())
        self.DeclareAbstractInputPort("target_state", target_state_value())
        self.DeclareAbstractOutputPort(
            "scene",
            scene_state_value,
            self._calc,
            prerequisites_of_calc={self.time_ticket()},
        )

    def _calc(self, context, output) -> None:
        vehicle = self.GetInputPort("vehicle_state").Eval(context)
        target = self.GetInputPort("target_state").Eval(context)
        output.set_value(SceneState(t=float(context.get_time()), vehicle=vehicle, target=target))

