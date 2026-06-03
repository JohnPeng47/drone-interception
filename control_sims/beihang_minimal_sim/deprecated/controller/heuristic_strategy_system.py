"""Drake adapter for heuristic strategy objects."""

from __future__ import annotations

from pydrake.systems.framework import LeafSystem

from ..drake_values import ctbr_value, strategy_observation_value
from .strategy_api import Strategy


class HeuristicStrategySystem(LeafSystem):
    def __init__(self, strategy: Strategy):
        super().__init__()
        self._strategy = strategy
        self.DeclareAbstractInputPort("observation", strategy_observation_value())
        self.DeclareAbstractOutputPort(
            "ctbr_cmd",
            ctbr_value,
            self._calc,
            prerequisites_of_calc={self.time_ticket()},
        )

    def _calc(self, context, output) -> None:
        obs = self.GetInputPort("observation").Eval(context)
        output.set_value(self._strategy.command(obs, float(context.get_time())))

