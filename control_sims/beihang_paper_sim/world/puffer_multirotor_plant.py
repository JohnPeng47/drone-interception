"""Drake LeafSystem wrapping the shared Puffer drone backend."""

from __future__ import annotations

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import LeafSystem

from backends import PufferDroneBackend, SimOptions
from backends.csim.bindings import (
    initial_state_from_rotorpy,
    vehicle_params_from_quad_params,
)

from ..drake_compat import ctbr_value


class PufferMultirotorPlant(LeafSystem):
    """Discrete Drake plant backed by the Puffer drone dynamics.

    The public ports intentionally match codex_sim.world.MultirotorPlant:

      input:  ctbr_cmd
      output: vehicle_state_dict

    That lets beihang_paper_sim keep the same Drake graph while swapping only the
    plant/backend implementation.
    """

    def __init__(
        self,
        quad_params: dict,
        initial_state: dict[str, np.ndarray],
        dt: float,
        options: SimOptions | None = None,
    ):
        super().__init__()
        self._dt = float(dt)
        self._params = vehicle_params_from_quad_params(quad_params)
        self._backend = PufferDroneBackend(self._params, options=options)
        self._initial_state = self._backend.reset(initial_state_from_rotorpy(initial_state))

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
        state.get_mutable_abstract_state(self._state_index).set_value(_copy_state(new_state))

    def _copy_state_out(self, context, output):
        s = context.get_abstract_state(self._state_index).get_value()
        output.set_value(_copy_state(s))


def _copy_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=float).copy() for k, v in state.items()}
