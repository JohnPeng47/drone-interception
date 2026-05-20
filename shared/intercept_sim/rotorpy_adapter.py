from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from intercept_sim.types import CtbrCommand, SimulationTarget


def rotorpy_state_to_target(
    state: dict[str, np.ndarray],
    *,
    target_id: str = "interceptor",
    kind: str = "multirotor",
    radius_m: float = 0.15,
) -> SimulationTarget:
    return SimulationTarget(
        id=target_id,
        kind=kind,
        position_w=np.asarray(state["x"], dtype=float).copy(),
        velocity_w=np.asarray(state["v"], dtype=float).copy(),
        rotation_wb=Rotation.from_quat(state["q"]).as_matrix(),
        radius_m=radius_m,
    )


def ctbr_to_rotorpy(command: CtbrCommand) -> dict[str, np.ndarray | float]:
    return {
        "cmd_thrust": float(command.thrust_n),
        "cmd_w": np.asarray(command.body_rates_b, dtype=float).copy(),
    }


def hover_ctbr(t: float, mass_kg: float, gravity_mps2: float = 9.81) -> CtbrCommand:
    return CtbrCommand(
        t=float(t),
        thrust_n=float(mass_kg * gravity_mps2),
        body_rates_b=np.zeros(3, dtype=float),
    )

