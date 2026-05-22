from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .camera_sim import CameraConfig
from .sim_types import PUFFER_ACTION_SUBSTEPS, PUFFER_DT, PursuerInitialState, PursuerParams
from .target_sim import TargetConfig, TargetInitialState


@dataclass(frozen=True)
class SimOptions:
    backend_dt: float = PUFFER_DT
    action_substeps: int = PUFFER_ACTION_SUBSTEPS
    command_mode: str = "ctbr"
    ctbr_rate_gain: float = 0.08
    randomize_params: bool = False


@dataclass(frozen=True)
class SimConfig:
    pursuer: PursuerParams
    options: SimOptions = field(default_factory=SimOptions)
    intercept_radius_m: float = 0.0


@dataclass(frozen=True)
class SimInstance:
    seed: int
    pursuer_initial: PursuerInitialState
    targets: tuple[TargetConfig, ...]
    cameras: tuple[CameraConfig, ...] = ()
    config: SimConfig | None = None
    raw_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_initial(self) -> TargetInitialState:
        if not self.targets:
            raise AttributeError("SimInstance has no targets")
        target = self.targets[0]
        return TargetInitialState(
            position_w=target.initial.position_w,
            velocity_w=target.initial.velocity_w,
            radius_m=target.radius_m,
        )
