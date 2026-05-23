from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TargetState:
    position_w: np.ndarray
    velocity_w: np.ndarray


@dataclass(frozen=True)
class TargetInitialState:
    position_w: np.ndarray
    velocity_w: np.ndarray
    radius_m: float


@dataclass(frozen=True)
class TargetBehaviorConfig:
    kind: str = "waypoints"
    waypoints: tuple[np.ndarray, ...] = ()
    duration_s: float = 0.0
    loop: bool = False


@dataclass(frozen=True)
class TargetControllerConfig:
    kind: str = "linear"
    kp: float = 0.0
    kv: float = 0.0
    max_accel_mps2: float = 0.0


@dataclass(frozen=True)
class TargetConfig:
    id: str
    kind: str
    radius_m: float
    initial: TargetState
    behavior: TargetBehaviorConfig = field(default_factory=TargetBehaviorConfig)
    controller: TargetControllerConfig = field(default_factory=TargetControllerConfig)

    @classmethod
    def from_initial(
        cls,
        initial: TargetInitialState,
        *,
        id: str = "target",
        kind: str = "target",
    ) -> "TargetConfig":
        return cls(
            id=id,
            kind=kind,
            radius_m=float(initial.radius_m),
            initial=TargetState(
                position_w=initial.position_w,
                velocity_w=initial.velocity_w,
            ),
        )
