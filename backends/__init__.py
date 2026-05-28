from .csim.bindings import BatchPufferSimEngineBackend, PufferDroneBackend, PufferSimEngineBackend
from .csim.bindings.types import (
    CameraConfig,
    CameraIntrinsics,
    NoiseConfig,
    PursuerInitialState,
    PursuerParams,
    RenderConfig,
    SimConfig,
    SimInstance,
    SimOptions,
    TargetBehaviorConfig,
    TargetConfig,
    TargetControllerConfig,
    TargetInitialState,
    TargetState,
)
from .csim.generator import SimGenerator, SimInstanceGenerator, read_sim_instances, write_sim_instances


def __getattr__(name: str):
    if name in {"RotorPyDroneBackend", "RotorPyMultirotorPlant"}:
        from .rotorpy import RotorPyDroneBackend, RotorPyMultirotorPlant

        return {
            "RotorPyDroneBackend": RotorPyDroneBackend,
            "RotorPyMultirotorPlant": RotorPyMultirotorPlant,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CameraConfig",
    "CameraIntrinsics",
    "NoiseConfig",
    "BatchPufferSimEngineBackend",
    "PufferDroneBackend",
    "PufferSimEngineBackend",
    "PursuerInitialState",
    "PursuerParams",
    "RenderConfig",
    "RotorPyDroneBackend",
    "RotorPyMultirotorPlant",
    "SimConfig",
    "SimGenerator",
    "SimInstanceGenerator",
    "SimInstance",
    "SimOptions",
    "TargetBehaviorConfig",
    "TargetConfig",
    "TargetControllerConfig",
    "TargetInitialState",
    "TargetState",
    "read_sim_instances",
    "write_sim_instances",
]
