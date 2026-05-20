from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from intercept_sim.types import CtbrCommand


@dataclass(frozen=True)
class KeyMap:
    thrust_up: str
    thrust_down: str
    yaw_left: str
    yaw_right: str
    pitch_forward: str
    pitch_back: str
    roll_left: str
    roll_right: str
    hover: str
    reset: str
    quit: str

    @classmethod
    def from_config(cls, raw: dict[str, str]) -> "KeyMap":
        return cls(**{key: str(value).upper() for key, value in raw.items()})


@dataclass
class ManualCtbrState:
    thrust_n: float
    body_rates_b: np.ndarray


@dataclass(frozen=True)
class ThrustControlConfig:
    min_n: float
    max_n: float
    trim_n: float
    step_n_per_s: float


@dataclass(frozen=True)
class BodyRateControlConfig:
    max_roll_rps: float
    max_pitch_rps: float
    max_yaw_rps: float
    step_rps_per_s: float
    damping: float


@dataclass
class ControlMap:
    keymap: KeyMap
    thrust: ThrustControlConfig
    body_rates: BodyRateControlConfig
    state: ManualCtbrState

    @classmethod
    def from_config(cls, control_config: dict[str, Any], *, mass_kg: float, gravity_mps2: float = 9.81) -> "ControlMap":
        hover = mass_kg * gravity_mps2
        thrust_raw = control_config["thrust"]
        body_raw = control_config["body_rates"]
        max_thrust = _auto_or_float(thrust_raw["max_n"], default=3.0 * hover)
        trim = _auto_or_float(thrust_raw["trim_n"], default=hover)
        return cls(
            keymap=KeyMap.from_config(control_config["keymap"]),
            thrust=ThrustControlConfig(
                min_n=float(thrust_raw["min_n"]),
                max_n=max_thrust,
                trim_n=trim,
                step_n_per_s=float(thrust_raw["step_n_per_s"]),
            ),
            body_rates=BodyRateControlConfig(
                max_roll_rps=float(body_raw["max_roll_rps"]),
                max_pitch_rps=float(body_raw["max_pitch_rps"]),
                max_yaw_rps=float(body_raw["max_yaw_rps"]),
                step_rps_per_s=float(body_raw["step_rps_per_s"]),
                damping=float(body_raw["damping"]),
            ),
            state=ManualCtbrState(thrust_n=trim, body_rates_b=np.zeros(3, dtype=float)),
        )

    def update(self, pressed_keys: Iterable[str], dt: float, t: float = 0.0) -> CtbrCommand:
        keys = {key.upper() for key in pressed_keys}
        if self.keymap.hover in keys:
            self.state.thrust_n = self.thrust.trim_n
            self.state.body_rates_b = np.zeros(3, dtype=float)
            return self.command(t)

        thrust_delta = 0.0
        if self.keymap.thrust_up in keys:
            thrust_delta += self.thrust.step_n_per_s * dt
        if self.keymap.thrust_down in keys:
            thrust_delta -= self.thrust.step_n_per_s * dt
        self.state.thrust_n = float(np.clip(self.state.thrust_n + thrust_delta, self.thrust.min_n, self.thrust.max_n))

        rate_delta = np.zeros(3, dtype=float)
        if self.keymap.roll_left in keys:
            rate_delta[0] += self.body_rates.step_rps_per_s * dt
        if self.keymap.roll_right in keys:
            rate_delta[0] -= self.body_rates.step_rps_per_s * dt
        if self.keymap.pitch_forward in keys:
            rate_delta[1] += self.body_rates.step_rps_per_s * dt
        if self.keymap.pitch_back in keys:
            rate_delta[1] -= self.body_rates.step_rps_per_s * dt
        if self.keymap.yaw_left in keys:
            rate_delta[2] += self.body_rates.step_rps_per_s * dt
        if self.keymap.yaw_right in keys:
            rate_delta[2] -= self.body_rates.step_rps_per_s * dt

        decay = max(0.0, 1.0 - self.body_rates.damping * dt)
        self.state.body_rates_b = self.state.body_rates_b * decay + rate_delta
        self.state.body_rates_b = np.clip(
            self.state.body_rates_b,
            [-self.body_rates.max_roll_rps, -self.body_rates.max_pitch_rps, -self.body_rates.max_yaw_rps],
            [self.body_rates.max_roll_rps, self.body_rates.max_pitch_rps, self.body_rates.max_yaw_rps],
        )
        return self.command(t)

    def command(self, t: float) -> CtbrCommand:
        return CtbrCommand(
            t=float(t),
            thrust_n=float(self.state.thrust_n),
            body_rates_b=self.state.body_rates_b.copy(),
        )

    def should_quit(self, pressed_keys: Iterable[str]) -> bool:
        return self.keymap.quit in {key.upper() for key in pressed_keys}


def _auto_or_float(value: Any, *, default: float) -> float:
    if isinstance(value, str) and value.lower() == "auto":
        return float(default)
    return float(value)
