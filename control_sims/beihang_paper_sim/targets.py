from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import SimulationTarget


@dataclass
class KinematicTarget:
    target_id: str
    kind: str
    initial_position_w: np.ndarray
    velocity_w: np.ndarray
    radius_m: float

    def state_at(self, t: float) -> SimulationTarget:
        return SimulationTarget(
            id=self.target_id,
            kind=self.kind,
            position_w=np.asarray(self.initial_position_w, dtype=float)
            + float(t) * np.asarray(self.velocity_w, dtype=float),
            velocity_w=np.asarray(self.velocity_w, dtype=float).copy(),
            rotation_wb=np.eye(3, dtype=float),
            radius_m=float(self.radius_m),
        )
