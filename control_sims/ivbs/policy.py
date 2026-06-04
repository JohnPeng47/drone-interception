from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from backends.csim.bindings.types import SimInstance
from backends.csim.runner import CtbrCommandBatch, SimControlPolicy, SimRunnerState
from control_sims.beihang_paper_sim.controller.control_math import DEFAULT_GAINS

from .control_law import beihang_command_from_estimate, cautious_bearing_command
from .observer import VisualObserverConfig, VisualRelativeStateObserver


class IVBSControlPolicy(SimControlPolicy):
    """Beihang-style IVBS controller driven by visual relative-state estimates."""

    def __init__(
        self,
        gains: Mapping[str, float] | None = None,
        observer_config: VisualObserverConfig | Mapping[str, float] | None = None,
    ):
        self._gains = {
            **DEFAULT_GAINS,
            "cautious_closing_accel_mps2": 4.0,
            "cautious_velocity_damping": 0.25,
            **dict(gains or {}),
        }
        self._observer = VisualRelativeStateObserver(observer_config)

    def reset(self, state: SimRunnerState) -> None:
        self._observer.reset()

    def on_slots_started(
        self,
        slots: np.ndarray,
        instances: Sequence[SimInstance],
        state: SimRunnerState,
    ) -> None:
        snapshots = tuple(state.snapshot[int(slot)] for slot in np.asarray(slots, dtype=np.int64).reshape(-1))
        self._observer.start_slots(slots, instances, snapshots)

    def command(self, state: SimRunnerState) -> CtbrCommandBatch:
        thrust_n = np.zeros(len(state.instances), dtype=np.float32)
        body_rates_b = np.zeros((len(state.instances), 3), dtype=np.float32)
        for slot, instance in enumerate(state.instances):
            if instance is None or not bool(state.active[slot]):
                self._observer.stop_slot(slot)
                continue
            snapshot = state.snapshot[slot]
            estimate = self._observer.estimate(
                slot,
                instance,
                snapshot,
                t_s=float(state.elapsed_s[slot]),
            )
            if estimate.metric_confident:
                command = beihang_command_from_estimate(instance, snapshot, estimate, self._gains)
            else:
                command = cautious_bearing_command(instance, snapshot, estimate, self._gains)
            thrust_n[slot] = np.float32(command[0])
            body_rates_b[slot] = np.asarray(command[1], dtype=np.float32).reshape(3)
        return CtbrCommandBatch(thrust_n=thrust_n, body_rates_b=body_rates_b)
