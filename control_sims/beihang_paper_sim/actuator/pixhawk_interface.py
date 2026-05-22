"""Pixhawk passthrough adapter for CTBR commands."""

from __future__ import annotations

from pydrake.systems.framework import LeafSystem

from ..drake_compat import ctbr_value


class PixhawkInterface(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareAbstractInputPort("ctbr_cmd", ctbr_value())
        self.DeclareAbstractOutputPort("rate_cmd", ctbr_value, self._passthrough)

    def _passthrough(self, context, output):
        output.set_value(self.GetInputPort("ctbr_cmd").Eval(context))
