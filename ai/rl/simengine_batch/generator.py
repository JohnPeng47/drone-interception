from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from backends.csim.bindings.types import SimInstance
from ai.rl.simengine_env.scenario_table import ScenarioLabel, ScenarioTable


SamplingStrategy = Literal["random", "grid_balanced", "sequential_epoch"]


@dataclass(frozen=True)
class BatchReset:
    indices: np.ndarray
    scenario_indices: np.ndarray
    instances: tuple[SimInstance, ...]
    labels: tuple[ScenarioLabel, ...]
    duration_s: np.ndarray


class BatchSimGenerator:
    """Keeps a fixed-width batch of SimEngine slots filled with scenarios."""

    def __init__(
        self,
        table: ScenarioTable,
        *,
        num_envs: int,
        seed: int = 1,
        strategy: SamplingStrategy = "grid_balanced",
    ):
        self.table = table
        self.num_envs = int(num_envs)
        self.strategy = strategy
        self.rng = np.random.default_rng(int(seed))
        self._cursor = 0
        self._cell_cursor = 0
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if strategy not in {"random", "grid_balanced", "sequential_epoch"}:
            raise ValueError(f"unknown sampling strategy {strategy!r}")

    def initial_batch(self) -> BatchReset:
        return self.refill(np.arange(self.num_envs, dtype=np.int64))

    def refill(self, indices: np.ndarray) -> BatchReset:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        scenario_indices = self._sample_indices(len(indices))
        instances = tuple(self.table.get(int(idx)) for idx in scenario_indices)
        labels = tuple(self.table.label(int(idx)) for idx in scenario_indices)
        duration_s = np.asarray([
            0.0 if instance.config is None else float(instance.config.options.duration_s)
            for instance in instances
        ], dtype=np.float32)
        return BatchReset(
            indices=indices,
            scenario_indices=scenario_indices.astype(np.int64, copy=False),
            instances=instances,
            labels=labels,
            duration_s=duration_s,
        )

    def _sample_indices(self, count: int) -> np.ndarray:
        count = int(count)
        if self.strategy == "random":
            return self.rng.integers(0, self.table.count, size=count, dtype=np.int64)
        if self.strategy == "sequential_epoch":
            values = (np.arange(count, dtype=np.int64) + self._cursor) % self.table.count
            self._cursor = int((self._cursor + count) % self.table.count)
            return values
        return self._sample_grid_balanced(count)

    def _sample_grid_balanced(self, count: int) -> np.ndarray:
        if not self.table.cells or not self.table.samples_per_cell:
            return self.rng.integers(0, self.table.count, size=count, dtype=np.int64)
        samples_per_cell = int(self.table.samples_per_cell)
        cells = min(len(self.table.cells), max(1, (self.table.count + samples_per_cell - 1) // samples_per_cell))
        cell_ids = (np.arange(count, dtype=np.int64) + self._cell_cursor) % cells
        self._cell_cursor = int((self._cell_cursor + count) % cells)
        out = np.empty(count, dtype=np.int64)
        for i, cell_id in enumerate(cell_ids):
            start = int(cell_id) * samples_per_cell
            stop = min(start + samples_per_cell, self.table.count)
            out[i] = start + int(self.rng.integers(0, max(stop - start, 1)))
        return out

    def state_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "num_envs": self.num_envs,
            "rng_state": self.rng.bit_generator.state,
            "cursor": self._cursor,
            "cell_cursor": self._cell_cursor,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("strategy") != self.strategy:
            raise ValueError(f"checkpoint strategy {state.get('strategy')!r} does not match {self.strategy!r}")
        if int(state.get("num_envs", -1)) != self.num_envs:
            raise ValueError(f"checkpoint num_envs {state.get('num_envs')!r} does not match {self.num_envs}")
        self.rng.bit_generator.state = state["rng_state"]
        self._cursor = int(state["cursor"])
        self._cell_cursor = int(state["cell_cursor"])
