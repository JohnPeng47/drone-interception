"""Scene assembly LeafSystem."""

from __future__ import annotations

from pydrake.systems.framework import LeafSystem

from ..drake_compat import (
    make_scene_snapshot,
    rotorpy_state_to_target,
    scene_value,
    target_value,
    vehicle_state_value,
)
from ..types import CameraRig


class SceneAssembler(LeafSystem):
    def __init__(self, camera_rig: CameraRig):
        super().__init__()
        self._cameras = (camera_rig,)
        self.DeclareAbstractInputPort("vehicle_state_dict", vehicle_state_value())
        self.DeclareAbstractInputPort("target_state", target_value())
        self.DeclareAbstractOutputPort(
            "scene", scene_value, self._calc,
            prerequisites_of_calc={self.time_ticket()},
        )

    def _calc(self, context, output):
        vehicle_state = self.GetInputPort("vehicle_state_dict").Eval(context)
        target = self.GetInputPort("target_state").Eval(context)
        pursuer = rotorpy_state_to_target(vehicle_state)
        scene = make_scene_snapshot(
            context.get_time(), pursuer, [target], list(self._cameras),
        )
        output.set_value(scene)
