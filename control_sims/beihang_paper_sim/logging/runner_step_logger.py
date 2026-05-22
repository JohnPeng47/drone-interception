"""RunnerStepLogger LeafSystem for replay/debug traces."""

from __future__ import annotations

import numpy as np
from pydrake.systems.framework import LeafSystem

from ..drake_compat import (
    capture_value,
    ctbr_value,
    measurements_value,
    observer_state_value,
    scene_value,
    vehicle_state_value,
)
from ..types import CtbrCommand, RunnerStep


class RunnerStepLogger(LeafSystem):
    def __init__(self, dt: float):
        super().__init__()
        self._dt = float(dt)
        self._log: list[RunnerStep] = []

        self.DeclareAbstractInputPort("vehicle_state_dict", vehicle_state_value())
        self.DeclareAbstractInputPort("scene", scene_value())
        self.DeclareAbstractInputPort("capture", capture_value())
        self.DeclareAbstractInputPort("measurements", measurements_value())
        self.DeclareAbstractInputPort("observer_state", observer_state_value())
        self.DeclareAbstractInputPort("ctbr_cmd", ctbr_value())
        self.DeclarePeriodicPublishEvent(
            period_sec=self._dt, offset_sec=0.0, publish=self._record,
        )

    def get_log(self) -> list[RunnerStep]:
        return list(self._log)

    def reset(self) -> None:
        self._log.clear()

    def _record(self, context):
        t = context.get_time()
        state = self.GetInputPort("vehicle_state_dict").Eval(context)
        scene = self.GetInputPort("scene").Eval(context)
        capture = self.GetInputPort("capture").Eval(context)
        measurements = self.GetInputPort("measurements").Eval(context)
        observer_state = self.GetInputPort("observer_state").Eval(context)
        cmd = self.GetInputPort("ctbr_cmd").Eval(context)

        if cmd is None:
            cmd = CtbrCommand(t=t, thrust_n=0.0, body_rates_b=np.zeros(3, dtype=float))

        rotorpy_state = {k: np.asarray(v, dtype=float).copy() for k, v in state.items()}
        self._log.append(
            RunnerStep(
                t=float(t),
                rotorpy_state=rotorpy_state,
                scene=scene,
                capture=capture,
                measurements=tuple(measurements),
                observer_state=observer_state,
                command=cmd,
            )
        )
