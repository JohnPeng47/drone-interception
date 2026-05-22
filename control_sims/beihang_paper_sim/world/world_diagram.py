"""add_world: Drake world wiring with selectable plant backend."""

from __future__ import annotations

from pydrake.systems.framework import DiagramBuilder

from backends import RotorPyMultirotorPlant

from .scene_assembler import SceneAssembler
from .target_system import KinematicTargetSystem
from ..targets import KinematicTarget
from ..types import CameraRig
from .puffer_sim_engine_system import PufferSimEngineSystem


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
    intercept_radius_m: float = 0.0,
) -> dict:
    backend = str(backend)
    if backend == "rotorpy":
        plant = builder.AddSystem(RotorPyMultirotorPlant(vehicle, initial_state, dt))
        target_sys = builder.AddSystem(KinematicTargetSystem(target))
        scene = builder.AddSystem(SceneAssembler(camera_rig))

        builder.Connect(plant.GetOutputPort("vehicle_state_dict"),
                        scene.GetInputPort("vehicle_state_dict"))
        builder.Connect(target_sys.GetOutputPort("target_state"),
                        scene.GetInputPort("target_state"))

        return {"plant": plant, "target": target_sys, "scene": scene}
    elif backend == "puffer_c":
        if quad_params is None:
            raise ValueError("quad_params is required when backend='puffer_c'")
        world = builder.AddSystem(
            PufferSimEngineSystem(
                quad_params=quad_params,
                initial_state=initial_state,
                dt=dt,
                target=target,
                camera_rig=camera_rig,
                intercept_radius_m=intercept_radius_m,
            )
        )
        return {"plant": world, "target": world, "scene": world}
    else:
        raise ValueError(f"Unsupported beihang_paper_sim world backend: {backend}")
