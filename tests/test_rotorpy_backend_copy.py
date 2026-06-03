from __future__ import annotations

import numpy as np

from backends.rotorpy import RotorPyDroneBackend
from control_sims.beihang_paper_sim.deprecated.types import CtbrCommand


class _FakeVehicle:
    mass = 1.0

    def step(self, state, control, dt):
        assert "cmd_thrust" in control
        assert "cmd_w" in control
        out = {k: np.asarray(v, dtype=float).copy() for k, v in state.items()}
        out["x"] = out["x"] + dt * out["v"]
        return out


def test_rotorpy_backend_step_ctbr_shape():
    state = {
        "x": np.zeros(3),
        "v": np.array([1.0, 0.0, 0.0]),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
        "wind": np.zeros(3),
        "rotor_speeds": np.zeros(4),
    }
    backend = RotorPyDroneBackend(_FakeVehicle(), state, 0.1)
    out = backend.step_ctbr(state, CtbrCommand(0.0, 1.0, np.zeros(3)), 0.1)

    np.testing.assert_allclose(out["x"], np.array([0.1, 0.0, 0.0]))
    np.testing.assert_allclose(state["x"], np.zeros(3))
