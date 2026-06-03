"""add_controller: paper-pipeline controller systems (Eqs. 12-28).

camera_rig is required to derive n_td from R_b^c (paper §II-A.4) — the
designed LOS vector is the optical axis lifted into body frame.
"""

from __future__ import annotations

from pydrake.systems.framework import DiagramBuilder

from ..types import CameraRig
from .body_rate_command_system import BodyRateCommandSystem
from .desired_acceleration_system import DesiredAccelerationSystem
from .line_of_sight_guidance_system import LineOfSightGuidanceSystem
from .thrust_planning_system import ThrustPlanningSystem


def add_controller(
    builder: DiagramBuilder,
    *,
    mass_kg: float,
    dt: float,
    camera_rig: CameraRig,
    gains: dict | None = None,
) -> dict:
    los = builder.AddSystem(LineOfSightGuidanceSystem(camera_rig=camera_rig, gains=gains))
    desired = builder.AddSystem(DesiredAccelerationSystem(mass_kg=mass_kg, gains=gains))
    thrust = builder.AddSystem(ThrustPlanningSystem(mass_kg=mass_kg, gains=gains))
    rates = builder.AddSystem(BodyRateCommandSystem(mass_kg=mass_kg, gains=gains))

    builder.Connect(los.GetOutputPort("los_guidance"),
                    desired.GetInputPort("los_guidance"))
    builder.Connect(desired.GetOutputPort("desired_acceleration"),
                    thrust.GetInputPort("desired_acceleration"))
    builder.Connect(thrust.GetOutputPort("thrust_plan"),
                    rates.GetInputPort("thrust_plan"))

    return {
        "los": los,
        "desired_acceleration": desired,
        "thrust": thrust,
        "rates": rates,
        "observer_input": los,
        "command": rates,
    }
