"""add_controller: ControlCore (paper Eqs. 12-28).

camera_rig is required to derive n_td from R_b^c (paper §II-A.4) — the
designed LOS vector is the optical axis lifted into body frame.
"""

from __future__ import annotations

from pydrake.systems.framework import DiagramBuilder

from .control_core import ControlCore
from ..types import CameraRig


def add_controller(
    builder: DiagramBuilder,
    *,
    mass_kg: float,
    dt: float,
    camera_rig: CameraRig,
    gains: dict | None = None,
) -> dict:
    core = builder.AddSystem(
        ControlCore(mass_kg=mass_kg, dt=dt, camera_rig=camera_rig, gains=gains)
    )
    return {"core": core}
