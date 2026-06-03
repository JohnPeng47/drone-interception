"""Small functional API for LLM-mutated heuristic strategies."""

from __future__ import annotations

from typing import Protocol

from ...types import CtbrCommand, StrategyObservation


class Strategy(Protocol):
    def command(self, observation: StrategyObservation, t: float) -> CtbrCommand:
        ...
