"""RotorPy backend and Drake plant wrapper.

This is a local copy of the original Drake/RotorPy plant path used by
drake_sims, factored behind the same high-level backend shape as puffer_c:
state dict in, CtbrCommand in, state dict out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import LeafSystem

from drake_sims.adapters import ctbr_to_rotorpy
from drake_sims.ports import ctbr_value


@dataclass
class RotorPyDroneBackend:
    vehicle: Any
    initial_state: dict[str, np.ndarray]
    dt: float

    def __post_init__(self) -> None:
        self.initial_state = _copy_state(self.initial_state)
        self.dt = float(self.dt)

    @property
    def mass_kg(self) -> float:
        return float(getattr(self.vehicle, "mass"))

    def reset(self, initial_state: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        return _copy_state(self.initial_state if initial_state is None else initial_state)

    def step_ctbr(
        self,
        state: dict[str, np.ndarray],
        command: Any,
        dt: float | None = None,
    ) -> dict[str, np.ndarray]:
        step_dt = self.dt if dt is None else float(dt)
        new_state = self.vehicle.step(state, ctbr_to_rotorpy(command), step_dt)
        return _copy_state(new_state)


class RotorPyMultirotorPlant(LeafSystem):
    """Discrete Drake LeafSystem wrapping RotorPy Multirotor.

    This is intentionally equivalent to the original
    codex_sim.world.MultirotorPlant, but lives under gavin_puffer/backends so
    backend selection is explicit and local to the migrated tree.
    """

    def __init__(self, vehicle, initial_state: dict[str, np.ndarray], dt: float):
        super().__init__()
        self._backend = RotorPyDroneBackend(vehicle=vehicle, initial_state=initial_state, dt=dt)
        self._initial_state = self._backend.reset()
        self._dt = float(dt)

        self._state_index = self.DeclareAbstractState(
            AbstractValue.Make(_copy_state(self._initial_state))
        )
        self.DeclareAbstractInputPort("ctbr_cmd", ctbr_value())
        self.DeclareAbstractOutputPort(
            "vehicle_state_dict",
            lambda: AbstractValue.Make(_copy_state(self._initial_state)),
            self._copy_state_out,
            prerequisites_of_calc={self.abstract_state_ticket(self._state_index)},
        )
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=self._dt, offset_sec=0.0, update=self._step,
        )

    def _step(self, context, state):
        current = state.get_mutable_abstract_state(self._state_index).get_value()
        cmd = self.GetInputPort("ctbr_cmd").Eval(context)
        new_state = self._backend.step_ctbr(current, cmd, self._dt)
        state.get_mutable_abstract_state(self._state_index).set_value(new_state)

    def _copy_state_out(self, context, output):
        s = context.get_abstract_state(self._state_index).get_value()
        output.set_value(_copy_state(s))


def _copy_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=float).copy() for k, v in state.items()}

