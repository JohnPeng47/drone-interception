from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backends.csim.bindings import BatchPufferSimEngineBackend, SimInstance
from backends.csim.bindings.types import SimSnapshot, SimSnapshots
from backends.csim.generator.instance_store import read_sim_instances


@dataclass(frozen=True)
class CtbrCommandBatch:
    thrust_n: np.ndarray
    body_rates_b: np.ndarray


@dataclass(frozen=True)
class MotorSpeedCommandBatch:
    motor_speeds_rpm: np.ndarray


@dataclass(frozen=True)
class CompletedSim:
    slot: int
    workload_index: int
    instance: SimInstance
    seed: int
    elapsed_s: float
    steps: int
    terminal_reason: str
    terminal_snapshot: SimSnapshot


@dataclass(frozen=True)
class SimRunnerState:
    snapshot: SimSnapshots
    active: np.ndarray
    workload_indices: np.ndarray
    instances: tuple[SimInstance | None, ...]
    elapsed_s: np.ndarray
    steps: np.ndarray


@dataclass(frozen=True)
class SimRunnerStep:
    state: SimRunnerState
    completed: tuple[CompletedSim, ...]
    commands: CtbrCommandBatch | MotorSpeedCommandBatch | None = None


@dataclass(frozen=True)
class SimRunResult:
    completed: tuple[CompletedSim, ...]
    steps: tuple[SimRunnerStep, ...]


CommandProvider = Callable[[SimRunnerState], CtbrCommandBatch | MotorSpeedCommandBatch | Mapping[str, Any]]
StepCallback = Callable[[SimRunnerStep], None]


class SimControlPolicy:
    """Base interface for controllers that emit physical CTBR commands."""

    def reset(self, state: SimRunnerState) -> None:
        pass

    def on_slots_started(
        self,
        slots: np.ndarray,
        instances: Sequence[SimInstance],
        state: SimRunnerState,
    ) -> None:
        pass

    def command(self, state: SimRunnerState) -> CtbrCommandBatch | MotorSpeedCommandBatch | Mapping[str, Any]:
        raise NotImplementedError

    def on_step(self, step: SimRunnerStep) -> None:
        pass

    def close(self) -> None:
        pass


class _CallablePolicy(SimControlPolicy):
    def __init__(self, provider: CommandProvider):
        self._provider = provider

    def command(self, state: SimRunnerState) -> CtbrCommandBatch | MotorSpeedCommandBatch | Mapping[str, Any]:
        return self._provider(state)


class SimRunner:
    """Run typed SimInstance workloads through fixed-width C SimEngine slots."""

    def __init__(
        self,
        max_envs: int,
        *,
        step_callbacks: Sequence[StepCallback] = (),
    ):
        self.step_callbacks = tuple(step_callbacks)
        self.max_envs = int(max_envs)
        if self.max_envs <= 0:
            raise ValueError("max_envs must be positive")

        self.num_envs = 0
        self.backend: BatchPufferSimEngineBackend | None = None
        self.snapshot: SimSnapshots | None = None
        self._workload: tuple[SimInstance, ...] = ()
        self._next_workload_index = 0
        self._active = np.zeros(0, dtype=bool)
        self._workload_indices = np.full(0, -1, dtype=np.int64)
        self._instances: list[SimInstance | None] = []
        self._duration_s = np.zeros(0, dtype=np.float32)
        self._elapsed_s = np.zeros(0, dtype=np.float32)
        self._steps = np.zeros(0, dtype=np.int32)
        self._step_log: list[SimRunnerStep] = []
        self._dt_s = 0.0
        self._policy: SimControlPolicy | None = None

    @property
    def active(self) -> np.ndarray:
        return self._active.copy()

    @property
    def has_active(self) -> bool:
        return bool(np.any(self._active))

    def reset(self, instances: Sequence[SimInstance]) -> SimRunnerState:
        self._workload = tuple(instances)
        if not self._workload:
            raise ValueError("SimRunner requires at least one SimInstance")

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
        self._step_log = []
        self._dt_s = self._dt_from_instance(self._workload[0])
        self._fill_slots(np.arange(self.num_envs, dtype=np.int64))
        return self.state()

    def state(self) -> SimRunnerState:
        if self.snapshot is None:
            raise RuntimeError("SimRunner has not been reset")
        return SimRunnerState(
            snapshot=self.snapshot,
            active=self._active.copy(),
            workload_indices=self._workload_indices.copy(),
            instances=tuple(self._instances),
            elapsed_s=self._elapsed_s.copy(),
            steps=self._steps.copy(),
        )

    def step(self, commands: CtbrCommandBatch | MotorSpeedCommandBatch | Mapping[str, Any]) -> SimRunnerStep:
        if self.backend is None or self.snapshot is None:
            raise RuntimeError("SimRunner has not been reset")
        if not self.has_active:
            return SimRunnerStep(state=self.state(), completed=())

        applied_commands: CtbrCommandBatch | MotorSpeedCommandBatch
        if self._is_motor_speed_command(commands):
            motor_speeds_rpm = self._motor_speed_commands_to_arrays(commands)
            motor_speeds_rpm = np.where(self._active[:, None], motor_speeds_rpm, 0.0).astype(np.float32, copy=False)
            applied_commands = MotorSpeedCommandBatch(motor_speeds_rpm=motor_speeds_rpm.copy())
            terminal_snapshot = self.backend.step_motor_speeds_many(motor_speeds_rpm)
        else:
            thrust_n, body_rates_b = self._ctbr_commands_to_arrays(commands)
            thrust_n = np.where(self._active, thrust_n, 0.0).astype(np.float32, copy=False)
            body_rates_b = np.where(self._active[:, None], body_rates_b, 0.0).astype(np.float32, copy=False)
            applied_commands = CtbrCommandBatch(thrust_n=thrust_n.copy(), body_rates_b=body_rates_b.copy())
            terminal_snapshot = self.backend.step_ctbr_commands_many(thrust_n, body_rates_b)
        self.snapshot = terminal_snapshot
        self._elapsed_s[self._active] += self._dt_s
        self._steps[self._active] += 1

        done, reasons = self._done(terminal_snapshot)
        completed = tuple(
            self._completed_from_slot(slot, reasons[slot], terminal_snapshot)
            for slot in np.flatnonzero(done)
        )

        step = SimRunnerStep(state=self.state(), completed=completed, commands=applied_commands)
        self._step_log.append(step)
        for callback in self.step_callbacks:
            callback(step)
        if self._policy is not None:
            self._policy.on_step(step)

        if completed:
            done_slots = np.flatnonzero(done)
            self._active[done_slots] = False
            self._workload_indices[done_slots] = -1
            for slot in done_slots:
                self._instances[int(slot)] = None
            started_slots, started_instances = self._fill_slots(done_slots)
            if self._policy is not None and len(started_slots):
                self._policy.on_slots_started(started_slots, started_instances, self.state())

        return step

    def run(
        self,
        instances: Sequence[SimInstance],
        policy: SimControlPolicy | CommandProvider,
    ) -> SimRunResult:
        self._policy = _coerce_policy(policy)
        try:
            state = self.reset(instances)
            self._policy.reset(state)
            initial_slots = np.flatnonzero(state.active)
            initial_instances = tuple(instance for instance in state.instances if instance is not None)
            if len(initial_slots):
                self._policy.on_slots_started(initial_slots, initial_instances, state)

            completed: list[CompletedSim] = []
            while self.has_active:
                step = self.step(self._policy.command(self.state()))
                completed.extend(step.completed)
            return SimRunResult(completed=tuple(completed), steps=tuple(self._step_log))
        finally:
            if self._policy is not None:
                self._policy.close()
            self._policy = None

    def run_file(
        self,
        path: str | Path,
        policy: SimControlPolicy | CommandProvider,
        *,
        count: int | None = None,
        offset: int = 0,
    ) -> SimRunResult:
        instances = read_sim_instances(path, count=count, offset=offset)
        if not instances:
            return SimRunResult(completed=(), steps=())
        return self.run(instances, policy)

    def _fill_slots(self, candidate_slots: np.ndarray) -> tuple[np.ndarray, tuple[SimInstance, ...]]:
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
            self._next_workload_index += 1

        if slots:
            self.snapshot = self.backend.reset_many(np.asarray(slots, dtype=np.int64), tuple(instances))
        return np.asarray(slots, dtype=np.int64), tuple(instances)

    def _done(self, snapshot: SimSnapshots) -> tuple[np.ndarray, list[str]]:
        intercepted = np.asarray([slot.metrics.intercepted for slot in snapshot], dtype=bool)
        failed, fail_reasons = self._failed(snapshot)
        timeout = self._timeout()
        done = self._active & (intercepted | failed | timeout)
        reasons = [
            "intercepted" if intercepted[i] else fail_reasons[i] if failed[i] else "timeout" if timeout[i] else ""
            for i in range(self.num_envs)
        ]
        return done, reasons

    def _failed(self, snapshot: SimSnapshots) -> tuple[np.ndarray, list[str]]:
        pos = np.asarray([slot.pursuer.position_w for slot in snapshot], dtype=np.float32)
        oob = np.zeros(self.num_envs, dtype=bool)
        for slot, instance in enumerate(self._instances):
            if instance is None or not bool(self._active[slot]):
                continue
            bounds_w = None if instance.config is None else instance.config.bounds_w
            if bounds_w is None:
                continue
            bounds = np.asarray(bounds_w, dtype=np.float32).reshape(3)
            oob[slot] = bool(np.any(np.abs(pos[slot]) > bounds))
        nonfinite = ~np.all(np.isfinite(pos), axis=1)
        failed = self._active & (oob | nonfinite)
        reasons = ["nonfinite" if nonfinite[i] else "oob" if oob[i] else "" for i in range(self.num_envs)]
        return failed, reasons

    def _timeout(self) -> np.ndarray:
        timeout = np.zeros(self.num_envs, dtype=bool)
        timeout |= (self._duration_s > 0.0) & (self._elapsed_s >= self._duration_s)
        return self._active & timeout

    def _completed_from_slot(
        self,
        slot: int,
        terminal_reason: str,
        snapshot: SimSnapshots,
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
            terminal_snapshot=snapshot[slot],
        )

    @staticmethod
    def _is_motor_speed_command(commands: CtbrCommandBatch | MotorSpeedCommandBatch | Mapping[str, Any]) -> bool:
        return isinstance(commands, MotorSpeedCommandBatch) or (
            isinstance(commands, Mapping) and "motor_speeds_rpm" in commands
        )

    def _ctbr_commands_to_arrays(self, commands: CtbrCommandBatch | Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
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

    def _motor_speed_commands_to_arrays(
        self,
        commands: MotorSpeedCommandBatch | Mapping[str, Any],
    ) -> np.ndarray:
        if isinstance(commands, MotorSpeedCommandBatch):
            motor_speeds_rpm = commands.motor_speeds_rpm
        else:
            motor_speeds_rpm = commands["motor_speeds_rpm"]
        return np.asarray(motor_speeds_rpm, dtype=np.float32).reshape(self.num_envs, 4)

    def _validate_instance(self, instance: SimInstance) -> None:
        if instance.config is None:
            raise ValueError("SimRunner requires SimInstance.config")
        duration_s = float(instance.config.options.duration_s)
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("SimRunner requires SimConfig.options.duration_s to be finite and positive")
        dt_s = self._dt_from_instance(instance)
        if abs(dt_s - self._dt_s) > 1e-12:
            raise ValueError("SimRunner requires homogeneous backend_dt/action_substeps")

    @staticmethod
    def _dt_from_instance(instance: SimInstance) -> float:
        if instance.config is None:
            raise ValueError("SimRunner requires SimInstance.config")
        options = instance.config.options
        return float(options.backend_dt) * max(1, int(options.action_substeps))


def _coerce_policy(policy: SimControlPolicy | CommandProvider) -> SimControlPolicy:
    if isinstance(policy, SimControlPolicy):
        return policy
    return _CallablePolicy(policy)
