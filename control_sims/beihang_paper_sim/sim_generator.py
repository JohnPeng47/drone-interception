"""Python scenario generation for Beihang control-sim runs.

This is the first control-sim-facing slice of the planned SimGenerator layer.
It still delegates red-balloon geometry to the existing shared builder, but it
returns an explicit SimInstance so callers stop depending directly on
`build_red_balloon_config`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backends import InitialState
from intercept_sim.experiments.red_balloon import (
    RedBalloonScenario,
    build_red_balloon_config,
    load_red_balloon_scenario,
)

from .config import ExperimentConfig


@dataclass(frozen=True)
class TargetInitialState:
    position_w: np.ndarray
    velocity_w: np.ndarray
    radius_m: float


@dataclass(frozen=True)
class SimInstance:
    seed: int
    pursuer_initial: InitialState
    target_initial: TargetInitialState
    raw_config: dict[str, Any]
    path: Path | None = None

    def to_experiment_config(self) -> ExperimentConfig:
        return ExperimentConfig(raw=copy.deepcopy(self.raw_config), path=self.path)


class RedBalloonSimGenerator:
    def __init__(self, scenario: RedBalloonScenario):
        self._scenario = scenario

    @classmethod
    def from_path(cls, path: str | Path) -> RedBalloonSimGenerator:
        return cls(load_red_balloon_scenario(path))

    def sample(
        self,
        *,
        seed: int,
        distance_m: float | None = None,
        closing_speed_mps: float | None = None,
        los_azimuth_deg: float | None = None,
        los_elevation_deg: float | None = None,
    ) -> SimInstance:
        cfg = build_red_balloon_config(
            self._scenario,
            seed=int(seed),
            distance_m=distance_m,
            closing_speed_mps=closing_speed_mps,
            los_azimuth_deg=los_azimuth_deg,
            los_elevation_deg=los_elevation_deg,
        )
        raw = copy.deepcopy(cfg.raw)
        vehicle = raw["vehicle"]
        target = raw["target"]
        pursuer_initial = InitialState(
            position_w=_array(vehicle["initial_position_w"], 3),
            velocity_w=_array(vehicle.get("initial_velocity_w", [0.0, 0.0, 0.0]), 3),
            quat_xyzw=_array(vehicle.get("initial_quat_xyzw", [0.0, 0.0, 0.0, 1.0]), 4),
            body_rates_b=_array(vehicle.get("initial_body_rates_b", [0.0, 0.0, 0.0]), 3),
            rotor_speeds=None,
            wind_w=_array(vehicle.get("wind_w", [0.0, 0.0, 0.0]), 3),
        )
        target_initial = TargetInitialState(
            position_w=_array(target["initial_position_w"], 3),
            velocity_w=_array(target.get("velocity_w", [0.0, 0.0, 0.0]), 3),
            radius_m=float(target["radius_m"]),
        )
        return SimInstance(
            seed=int(seed),
            pursuer_initial=pursuer_initial,
            target_initial=target_initial,
            raw_config=raw,
            path=cfg.path,
        )


def _array(value: Any, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"Expected shape ({length},), got {arr.shape}")
    return arr.copy()
