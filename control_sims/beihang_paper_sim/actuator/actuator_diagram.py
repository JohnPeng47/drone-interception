"""add_actuator: PixhawkInterface passthrough — reuses codex_sim's class."""

from __future__ import annotations

from pydrake.systems.framework import DiagramBuilder

from codex_sim.actuator.pixhawk_interface import PixhawkInterface


def add_actuator(builder: DiagramBuilder) -> dict:
    pixhawk = builder.AddSystem(PixhawkInterface())
    return {"pixhawk": pixhawk}
