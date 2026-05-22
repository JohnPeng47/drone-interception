"""add_actuator: PixhawkInterface passthrough."""

from __future__ import annotations

from pydrake.systems.framework import DiagramBuilder

from .pixhawk_interface import PixhawkInterface


def add_actuator(builder: DiagramBuilder) -> dict:
    pixhawk = builder.AddSystem(PixhawkInterface())
    return {"pixhawk": pixhawk}
