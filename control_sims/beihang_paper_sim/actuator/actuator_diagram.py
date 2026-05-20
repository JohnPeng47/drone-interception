"""add_actuator: PixhawkInterface passthrough — reuses codex_sim's class."""

from __future__ import annotations

from pydrake.systems.framework import DiagramBuilder

from ..drake_compat import PixhawkInterface


def add_actuator(builder: DiagramBuilder) -> dict:
    pixhawk = builder.AddSystem(PixhawkInterface())
    return {"pixhawk": pixhawk}
