from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from backends.csim.bindings import BatchPufferSimEngineBackend, SimInstance


@dataclass(frozen=True)
class BatchSimEngineRunnerConfig:
    max_envs: int
    max_episode_steps: int | None = None
    bounds_w: tuple[float, float, float] = (30.0, 30.0, 20.0)


@dataclass(frozen=True)
class CtbrCommandBatch:
    thrust_n: np.ndarray
    body_rates_b: np.ndarray


@dataclass(frozen=True)
class CompletedSim:
    slot: int
    workload_index: int
    instance: SimInstance
    seed: int
    elapsed_s: float
    steps: int
    terminal_reason: str
    terminal_snapshot: dict[str, Any]
    visible_fraction: float
    control_effort: float


@dataclass(frozen=True)
class BatchSimEngineRunnerState:
    snapshot: dict[str, np.ndarray]
    active: np.ndarray
    workload_indices: np.ndarray
    instances: tuple[SimInstance | None, ...]
    elapsed_s: np.ndarray
    steps: np.ndarray


@dataclass(frozen=True)
class BatchSimEngineStep:
    state: BatchSimEngineRunnerState
    completed: tuple[CompletedSim, ...]


CommandProvider = Callable[[BatchSimEngineRunnerState], CtbrCommandBatch | Mapping[str, Any]]
StepCallback = Callable[[BatchSimEngineStep], None]


class BatchSimEngineRunner:
    """Run a SimInstance workload through fixed-width C SimEngine slots.

    The runner is batch-native: callers provide physical CTBR commands for every
    slot, completed slots are refilled immediately, and control-sim-specific
    policies can live outside this class.
    """

    def __init__(
        self,
        config: BatchSimEngineRunnerConfig,
        *,
        step_callbacks: Sequence[StepCallback] = (),
    ):
        self.config = config
        self.step_callbacks = tuple(step_callbacks)
        self.max_envs = int(config.max_envs)
        if self.max_envs <= 0:
            raise ValueError("max_envs must be positive")

        self.num_envs = 0
        self.backend: BatchPufferSimEngineBackend | None = None
        self.snapshot: dict[str, np.ndarray] | None = None
        self._workload: tuple[SimInstance, ...] = ()
        self._next_workload_index = 0
        self._active = np.zeros(0, dtype=bool)
        self._workload_indices = np.full(0, -1, dtype=np.int64)
        self._instances: list[SimInstance | None] = []
        self._duration_s = np.zeros(0, dtype=np.float32)
        self._elapsed_s = np.zeros(0, dtype=np.float32)
        self._steps = np.zeros(0, dtype=np.int32)
        self._visible_count = np.zeros(0, dtype=np.int32)
        self._sample_count = np.zeros(0, dtype=np.int32)
        self._control_effort = np.zeros(0, dtype=np.float32)
        self._dt_s = 0.0

    @property
    def active(self) -> np.ndarray:
        return self._active.copy()

    @property
    def has_active(self) -> bool:
        return bool(np.any(self._active))

    def reset(self, instances: Sequence[SimInstance]) -> BatchSimEngineRunnerState:
        self._workload = tuple(instances)
        if not self._workload:
            raise ValueError("BatchSimEngineRunner requires at least one SimInstance")

        self.num_envs = min(self.max_envs, len(self._workload))
        self.backend = BatchPufferSimEngineBackend(self.num_envs)
        self.snapshot = None
        self._next_workload_index = 0
        self._active = np.zeros(self.num_envs, dtype=bool)
        self._workload_indices = np.full(self.num_envs, -1, dtype=np.int64)
        self._instances = [None] * self.num_envs
        self._duration_s = np.zeros(self.num_envs, dtype=np.float32)
        self._elapsed_s = np.zeros(self.num_envs, dtype=np.float32)
        self._steps = np.zeros(self.num_envs, dtype=np.int32)
        self._visible_count = np.zeros(self.num_envs, dtype=np.int32)
        self._sample_count = np.zeros(self.num_envs, dtype=np.int32)
        self._control_effort = np.zeros(self.num_envs, dtype=np.float32)
        self._dt_s = self._dt_from_instance(self._workload[0])
        self._fill_slots(np.arange(self.num_envs, dtype=np.int64))
        return self.state()

    def state(self) -> BatchSimEngineRunnerState:
        if self.snapshot is None:
            raise RuntimeError("BatchSimEngineRunner has not been reset")
        return BatchSimEngineRunnerState(
            snapshot=self.snapshot,
            active=self._active.copy(),
            workload_indices=self._workload_indices.copy(),
            instances=tuple(self._instances),
            elapsed_s=self._elapsed_s.copy(),
            steps=self._steps.copy(),
        )

    def step(self, commands: CtbrCommandBatch | Mapping[str, Any]) -> BatchSimEngineStep:
        if self.backend is None or self.snapshot is None:
            raise RuntimeError("BatchSimEngineRunner has not been reset")
        if not self.has_active:
            return BatchSimEngineStep(state=self.state(), completed=())

        thrust_n, body_rates_b = self._commands_to_arrays(commands)
        thrust_n = np.where(self._active, thrust_n, 0.0).astype(np.float32, copy=False)
        body_rates_b = np.where(self._active[:, None], body_rates_b, 0.0).astype(np.float32, copy=False)

        terminal_snapshot = self.backend.step_ctbr_commands_many(thrust_n, body_rates_b)
        self.snapshot = terminal_snapshot
        self._elapsed_s[self._active] += self._dt_s
        self._steps[self._active] += 1
        self._sample_outputs(terminal_snapshot, thrust_n, body_rates_b)

        done, reasons = self._done(terminal_snapshot)
        completed = tuple(
            self._completed_from_slot(slot, reasons[slot], terminal_snapshot)
            for slot in np.flatnonzero(done)
        )

        if completed:
            done_slots = np.flatnonzero(done)
            self._active[done_slots] = False
            self._workload_indices[done_slots] = -1
            for slot in done_slots:
                self._instances[int(slot)] = None
            self._fill_slots(done_slots)

        step = BatchSimEngineStep(state=self.state(), completed=completed)
        for callback in self.step_callbacks:
            callback(step)
        return step

    def run(
        self,
        instances: Sequence[SimInstance],
        command_provider: CommandProvider,
    ) -> tuple[CompletedSim, ...]:
        self.reset(instances)
        completed: list[CompletedSim] = []
        while self.has_active:
            step = self.step(command_provider(self.state()))
            completed.extend(step.completed)
        return tuple(completed)

    def _fill_slots(self, candidate_slots: np.ndarray) -> None:
        assert self.backend is not None
        slots: list[int] = []
        instances: list[SimInstance] = []
        for raw_slot in np.asarray(candidate_slots, dtype=np.int64).reshape(-1):
            if self._next_workload_index >= len(self._workload):
                break
            slot = int(raw_slot)
            instance = self._workload[self._next_workload_index]
            self._validate_instance(instance)
            slots.append(slot)
            instances.append(instance)
            self._active[slot] = True
            self._workload_indices[slot] = self._next_workload_index
            self._instances[slot] = instance
            self._duration_s[slot] = float(instance.config.options.duration_s)
            self._elapsed_s[slot] = 0.0
            self._steps[slot] = 0
            self._visible_count[slot] = 0
            self._sample_count[slot] = 0
            self._control_effort[slot] = 0.0
            self._next_workload_index += 1

        if slots:
            self.snapshot = self.backend.reset_many(np.asarray(slots, dtype=np.int64), tuple(instances))

    def _sample_outputs(
        self,
        snapshot: dict[str, np.ndarray],
        thrust_n: np.ndarray,
        body_rates_b: np.ndarray,
    ) -> None:
        active = self._active.copy()
        self._sample_count[active] += 1
        if "camera" in snapshot:
            detected = np.asarray(snapshot["camera"][:, 0] > 0.5, dtype=bool)
            self._visible_count[active & detected] += 1
        effort = np.linalg.norm(body_rates_b, axis=1) + 0.02 * np.abs(thrust_n)
        self._control_effort[active] += effort[active].astype(np.float32) * float(self._dt_s)

    def _done(self, snapshot: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
        metrics = snapshot["metrics"]
        intercepted = metrics[:, 2] > 0.5
        failed, fail_reasons = self._failed(snapshot)
        timeout = self._timeout()
        done = self._active & (intercepted | failed | timeout)
        reasons = [
            "intercepted" if intercepted[i] else fail_reasons[i] if failed[i] else "timeout" if timeout[i] else ""
            for i in range(self.num_envs)
        ]
        return done, reasons

    def _failed(self, snapshot: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
        pos = snapshot["pursuer"][:, 0:3]
        bounds = np.asarray(self.config.bounds_w, dtype=np.float32)
        oob = np.any(np.abs(pos) > bounds[None, :], axis=1)
        nonfinite = ~np.all(np.isfinite(pos), axis=1)
        failed = self._active & (oob | nonfinite)
        reasons = ["nonfinite" if nonfinite[i] else "oob" if oob[i] else "" for i in range(self.num_envs)]
        return failed, reasons

    def _timeout(self) -> np.ndarray:
        timeout = np.zeros(self.num_envs, dtype=bool)
        if self.config.max_episode_steps is not None:
            timeout |= self._steps >= int(self.config.max_episode_steps)
        timeout |= (self._duration_s > 0.0) & (self._elapsed_s >= self._duration_s)
        return self._active & timeout

    def _completed_from_slot(
        self,
        slot: int,
        terminal_reason: str,
        snapshot: dict[str, np.ndarray],
    ) -> CompletedSim:
        instance = self._instances[slot]
        if instance is None:
            raise RuntimeError(f"slot {slot} has no active SimInstance")
        return CompletedSim(
            slot=int(slot),
            workload_index=int(self._workload_indices[slot]),
            instance=instance,
            seed=int(instance.seed),
            elapsed_s=float(self._elapsed_s[slot]),
            steps=int(self._steps[slot]),
            terminal_reason=terminal_reason,
            terminal_snapshot=_copy_snapshot_slot(snapshot, slot),
            visible_fraction=(
                float(self._visible_count[slot] / self._sample_count[slot])
                if int(self._sample_count[slot]) > 0 else 0.0
            ),
            control_effort=float(self._control_effort[slot]),
        )

    def _commands_to_arrays(self, commands: CtbrCommandBatch | Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(commands, CtbrCommandBatch):
            thrust_n = commands.thrust_n
            body_rates_b = commands.body_rates_b
        else:
            thrust_n = commands["thrust_n"]
            body_rates_b = commands["body_rates_b"]
        return (
            np.asarray(thrust_n, dtype=np.float32).reshape(self.num_envs),
            np.asarray(body_rates_b, dtype=np.float32).reshape(self.num_envs, 3),
        )

    def _validate_instance(self, instance: SimInstance) -> None:
        if instance.config is None:
            raise ValueError("BatchSimEngineRunner requires SimInstance.config")
        dt_s = self._dt_from_instance(instance)
        if abs(dt_s - self._dt_s) > 1e-12:
            raise ValueError("BatchSimEngineRunner requires homogeneous backend_dt/action_substeps")

    @staticmethod
    def _dt_from_instance(instance: SimInstance) -> float:
        if instance.config is None:
            raise ValueError("BatchSimEngineRunner requires SimInstance.config")
        options = instance.config.options
        return float(options.backend_dt) * max(1, int(options.action_substeps))


def _copy_snapshot_slot(snapshot: dict[str, Any], slot: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in snapshot.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (snapshot["pursuer"].shape[0],):
            out[key] = value[slot].copy()
        elif isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            out[key] = value
    return out


class HoverCommandProvider:
    """Physical CTBR command provider that holds body rates at zero and hovers."""

    def __call__(self, state: BatchSimEngineRunnerState) -> CtbrCommandBatch:
        thrust_n = np.zeros(len(state.instances), dtype=np.float32)
        body_rates_b = np.zeros((len(state.instances), 3), dtype=np.float32)
        for slot, instance in enumerate(state.instances):
            if instance is None or not bool(state.active[slot]):
                continue
            params = instance.config.pursuer
            thrust_n[slot] = float(params.mass_kg * params.gravity_mps2)
        return CtbrCommandBatch(thrust_n=thrust_n, body_rates_b=body_rates_b)
