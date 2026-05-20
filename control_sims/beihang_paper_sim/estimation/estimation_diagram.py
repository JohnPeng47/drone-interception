"""add_estimation: paper-faithful DKF only (TruthRelativeOracle dropped)."""

from __future__ import annotations

from pydrake.systems.framework import DiagramBuilder

from intercept_sim.types import CameraRig

from ..noise_config import NoiseConfig
from .dkf_observer import DkfObserver


def add_estimation(
    builder: DiagramBuilder,
    *,
    camera_rig: CameraRig,
    dt: float,
    noise_config: NoiseConfig | None = None,
) -> dict:
    core = builder.AddSystem(
        DkfObserver(camera_rig=camera_rig, dt=dt, noise_config=noise_config)
    )
    return {"core": core}
