from .csim.bindings import BatchPufferSimEngineBackend, PufferDroneBackend, PufferSimEngineBackend
from .csim.bindings.types import (
    CameraConfig,
    CameraIntrinsics,
    CameraObservation,
    InterceptMetrics,
    NoiseConfig,
    PursuerInitialState,
    PursuerParams,
    PursuerState,
    RenderConfig,
    SimConfig,
    SimInstance,
    SimOptions,
    SimSnapshot,
    SimSnapshotArrays,
    SimSnapshots,
    TargetBehaviorConfig,
    TargetConfig,
    TargetControllerConfig,
    TargetInitialState,
    TargetState,
)
from .csim.generator import SimGenerator, SimInstanceGenerator, read_sim_instances, write_sim_instances
from .csim.runner import (
    CompletedSim,
    CtbrCommandBatch,
    MotorSpeedCommandBatch,
    SimControlPolicy,
    SimRunResult,
    SimRunner,
    SimRunnerState,
    SimRunnerStep,
)


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
    "CameraObservation",
    "NoiseConfig",
    "BatchPufferSimEngineBackend",
    "InterceptMetrics",
    "PufferDroneBackend",
    "PufferSimEngineBackend",
    "PursuerInitialState",
    "PursuerParams",
    "PursuerState",
    "RenderConfig",
    "RotorPyDroneBackend",
    "RotorPyMultirotorPlant",
    "SimConfig",
    "SimControlPolicy",
    "SimGenerator",
    "SimInstanceGenerator",
    "SimInstance",
    "SimOptions",
    "SimRunner",
    "SimRunResult",
    "SimSnapshot",
    "SimSnapshotArrays",
    "SimSnapshots",
    "SimRunnerState",
    "SimRunnerStep",
    "TargetBehaviorConfig",
    "TargetConfig",
    "TargetControllerConfig",
    "TargetInitialState",
    "TargetState",
    "CompletedSim",
    "CtbrCommandBatch",
    "MotorSpeedCommandBatch",
    "read_sim_instances",
    "write_sim_instances",
]
