from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .input import InitialState


@dataclass(frozen=True)
class TargetInitialState:
    position_w: np.ndarray
    velocity_w: np.ndarray
    radius_m: float


@dataclass(frozen=True)
class SimInstance:
    seed: int
    pursuer_initial: InitialState
    target_initial: TargetInitialState
    raw_config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class SimGenerator(ABC):
    """Shared scenario-generation contract for control sim and RL.

    `sample()` is the required boundary: it resolves a distribution/scenario
    seed into concrete initial conditions and backend-consumable config.
    `run()` is optional execution glue for deterministic scripts or RL jobs.
    """

    @abstractmethod
    def sample(self, *, seed: int, **kwargs: Any) -> SimInstance:
        raise NotImplementedError

    def run(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not implement run()")
