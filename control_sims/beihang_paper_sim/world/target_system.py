"""Kinematic target LeafSystem."""

from __future__ import annotations

from pydrake.systems.framework import LeafSystem

from ..drake_compat import target_value
from ..targets import KinematicTarget


class KinematicTargetSystem(LeafSystem):
    def __init__(self, target: KinematicTarget):
        super().__init__()
        self._target = target
        self.DeclareAbstractOutputPort("target_state", target_value, self._calc)

    def _calc(self, context, output):
        output.set_value(self._target.state_at(context.get_time()))
