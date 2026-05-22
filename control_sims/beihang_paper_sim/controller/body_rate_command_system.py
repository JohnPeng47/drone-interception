"""Body-rate CTBR command LeafSystem for the paper controller."""

from __future__ import annotations

import numpy as np
from pydrake.systems.framework import LeafSystem

from ..drake_compat import ctbr_value, hover_ctbr
from ..types import CtbrCommand
from .control_math import DEFAULT_GAINS, G_VEC, vex
from .pipeline_types import thrust_plan_value


class BodyRateCommandSystem(LeafSystem):
    def __init__(self, mass_kg: float, gains: dict | None = None):
        super().__init__()
        self._m = float(mass_kg)
        g = {**DEFAULT_GAINS, **(gains or {})}
        self._omega_max = float(g["omega_max"])

        self.DeclareAbstractInputPort("thrust_plan", thrust_plan_value())
        self.DeclareAbstractOutputPort(
            "ctbr_cmd", ctbr_value, self._calc,
            prerequisites_of_calc={self.time_ticket()},
        )

    def _calc(self, context, output):
        plan = self.GetInputPort("thrust_plan").Eval(context)
        t = float(context.get_time())
        if not plan.valid:
            output.set_value(hover_ctbr(t, self._m, gravity_mps2=abs(G_VEC[2])))
            return

        S = plan.R_d.T @ plan.R_wb - plan.R_wb.T @ plan.R_d
        b_omega_2 = -vex(S)
        b_omega_d = plan.b_omega_1 + b_omega_2
        n_w = float(np.linalg.norm(b_omega_d))
        if n_w > self._omega_max:
            b_omega_d = b_omega_d * (self._omega_max / n_w)

        output.set_value(CtbrCommand(t=t, thrust_n=plan.thrust_n, body_rates_b=b_omega_d))
