"""add_world: Drake world wiring with selectable plant backend."""

from __future__ import annotations

from pydrake.systems.framework import DiagramBuilder

from backends import RotorPyMultirotorPlant
from intercept_sim.targets import KinematicTarget
from intercept_sim.types import CameraRig

from ..drake_compat import KinematicTargetSystem, SceneAssembler
from .puffer_multirotor_plant import PufferMultirotorPlant


def add_world(
    builder: DiagramBuilder,
    *,
    vehicle,
    initial_state: dict,
    dt: float,
    target: KinematicTarget,
    camera_rig: CameraRig,
    backend: str = "rotorpy",
    quad_params: dict | None = None,
) -> dict:
    backend = str(backend)
    if backend == "rotorpy":
        plant = builder.AddSystem(RotorPyMultirotorPlant(vehicle, initial_state, dt))
    elif backend == "puffer_c":
        if quad_params is None:
            raise ValueError("quad_params is required when backend='puffer_c'")
        plant = builder.AddSystem(PufferMultirotorPlant(quad_params, initial_state, dt))
    else:
        raise ValueError(f"Unsupported beihang_paper_sim world backend: {backend}")
    target_sys = builder.AddSystem(KinematicTargetSystem(target))
    scene = builder.AddSystem(SceneAssembler(camera_rig))

    builder.Connect(plant.GetOutputPort("vehicle_state_dict"),
                    scene.GetInputPort("vehicle_state_dict"))
    builder.Connect(target_sys.GetOutputPort("target_state"),
                    scene.GetInputPort("target_state"))

    return {"plant": plant, "target": target_sys, "scene": scene}
