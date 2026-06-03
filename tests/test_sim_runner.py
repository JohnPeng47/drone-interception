from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest

from backends import (
    PursuerInitialState,
    PursuerParams,
    SimConfig,
    SimInstance,
    SimOptions,
    TargetConfig,
    TargetInitialState,
    write_sim_instances,
)
from backends.csim.runner import (
    CtbrCommandBatch,
    SimControlPolicy,
    SimRunner,
    SimRunnerState,
    SimRunnerStep,
)


class RecordingHoverPolicy(SimControlPolicy):
    def __init__(self):
        self.reset_seen = False
        self.started: list[tuple[list[int], list[int]]] = []
        self.steps: list[list[int]] = []
        self.closed = False

    def reset(self, state: SimRunnerState) -> None:
        self.reset_seen = True
        assert state.active.tolist() == [True, True]

    def on_slots_started(
        self,
        slots: np.ndarray,
        instances: Sequence[SimInstance],
        state: SimRunnerState,
    ) -> None:
        self.started.append((
            [int(slot) for slot in slots],
            [int(instance.seed) for instance in instances],
        ))

    def command(self, state: SimRunnerState) -> CtbrCommandBatch:
        thrust_n = np.zeros(len(state.instances), dtype=np.float32)
        body_rates_b = np.zeros((len(state.instances), 3), dtype=np.float32)
        for slot, instance in enumerate(state.instances):
            if instance is not None and bool(state.active[slot]):
                params = instance.config.pursuer
                thrust_n[slot] = float(params.mass_kg * params.gravity_mps2)
        return CtbrCommandBatch(thrust_n=thrust_n, body_rates_b=body_rates_b)

    def on_step(self, step: SimRunnerStep) -> None:
        self.steps.append([int(item.workload_index) for item in step.completed])

    def close(self) -> None:
        self.closed = True


class DistinctCommandPolicy(SimControlPolicy):
    def command(self, state: SimRunnerState) -> CtbrCommandBatch:
        thrust_n = np.arange(1, len(state.instances) + 1, dtype=np.float32)
        body_rates_b = np.zeros((len(state.instances), 3), dtype=np.float32)
        body_rates_b[:, 0] = np.arange(10, 10 + len(state.instances), dtype=np.float32)
        return CtbrCommandBatch(thrust_n=thrust_n, body_rates_b=body_rates_b)


def test_sim_runner_run_file_invokes_policy_lifecycle_and_refills_slots(tmp_path):
    path = tmp_path / "samples.csimin"
    write_sim_instances(path, _instances(3))
    policy = RecordingHoverPolicy()
    runner = SimRunner(max_envs=2)

    result = runner.run_file(path, policy)

    assert [item.workload_index for item in result.completed] == [0, 1, 2]
    assert [item.seed for item in result.completed] == [100, 101, 102]
    assert [item.terminal_reason for item in result.completed] == ["timeout", "timeout", "timeout"]
    assert len(result.steps) == 4
    assert policy.reset_seen is True
    assert policy.started == [([0, 1], [100, 101]), ([0], [102])]
    assert policy.steps == [[], [0, 1], [], [2]]
    assert policy.closed is True


def test_sim_runner_callbacks_receive_applied_commands_before_slot_refill():
    seen: list[tuple[list[int], list[float], list[float]]] = []

    def record(step: SimRunnerStep) -> None:
        seen.append((
            [int(index) for index in step.state.workload_indices],
            [float(value) for value in step.commands.thrust_n],
            [float(value) for value in step.commands.body_rates_b[:, 0]],
        ))

    runner = SimRunner(max_envs=2, step_callbacks=(record,))

    result = runner.run(_instances(2), DistinctCommandPolicy())

    assert [item.workload_index for item in result.completed] == [0, 1]
    assert len(result.steps) == 2
    assert seen == [
        ([0, 1], [1.0, 2.0], [10.0, 11.0]),
        ([0, 1], [1.0, 2.0], [10.0, 11.0]),
    ]


def test_sim_runner_step_returns_pre_refill_state_for_completed_slots():
    runner = SimRunner(max_envs=2)
    runner.reset(_instances(3))
    command = CtbrCommandBatch(
        thrust_n=np.zeros(2, dtype=np.float32),
        body_rates_b=np.zeros((2, 3), dtype=np.float32),
    )

    runner.step(command)
    step = runner.step(command)

    assert [item.workload_index for item in step.completed] == [0, 1]
    assert step.state.active.tolist() == [True, True]
    assert step.state.workload_indices.tolist() == [0, 1]
    assert [None if item is None else item.seed for item in runner.state().instances] == [102, None]


def test_sim_runner_rejects_instances_without_positive_duration():
    instance = _instances(1)[0]
    assert instance.config is not None
    invalid = replace(
        instance,
        config=replace(
            instance.config,
            options=replace(instance.config.options, duration_s=0.0),
        ),
    )

    with pytest.raises(ValueError, match="duration_s"):
        SimRunner(max_envs=1).run((invalid,), RecordingHoverPolicy())


@pytest.mark.parametrize("duration_s", [float("nan"), float("inf")])
def test_sim_runner_rejects_nonfinite_duration(duration_s: float):
    instance = _instances(1)[0]
    assert instance.config is not None
    invalid = replace(
        instance,
        config=replace(
            instance.config,
            options=replace(instance.config.options, duration_s=duration_s),
        ),
    )

    with pytest.raises(ValueError, match="duration_s"):
        SimRunner(max_envs=1).run((invalid,), RecordingHoverPolicy())


def _instances(count: int) -> tuple[SimInstance, ...]:
    params = PursuerParams(
        mass_kg=0.027,
        ixx=3.85e-6,
        iyy=3.85e-6,
        izz=5.9675e-6,
        arm_len_m=0.0396,
        k_thrust=3.16e-10,
        k_yaw=0.005964552,
        max_rpm=21702.0,
    )
    config = SimConfig(
        pursuer=params,
        options=SimOptions(duration_s=0.02),
        targets=(TargetConfig(id="target", kind="target", radius_m=0.2),),
        intercept_radius_m=0.1,
    )
    initial = PursuerInitialState(
        position_w=np.zeros(3),
        velocity_w=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        body_rates_b=np.zeros(3),
    )
    target_initial = TargetInitialState(
        position_w=np.array([2.0, 0.0, 0.0]),
        velocity_w=np.zeros(3),
    )
    return tuple(
        SimInstance(
            seed=100 + index,
            pursuer_initial=initial,
            target_initials=(target_initial,),
            config=config,
        )
        for index in range(count)
    )
