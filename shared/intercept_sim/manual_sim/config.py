from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ManualSimConfig:
    raw: dict[str, Any]

    @property
    def sim_dt(self) -> float:
        return float(self.raw["sim"]["dt"])

    @property
    def physics_hz(self) -> float:
        return float(self.raw["sim"]["physics_hz"])

    @property
    def render_hz(self) -> float:
        return float(self.raw["sim"]["render_hz"])

    @property
    def control(self) -> dict[str, Any]:
        return self.raw["control"]

    @property
    def vehicle(self) -> dict[str, Any]:
        return self.raw["vehicle"]

    @property
    def renderer(self) -> dict[str, Any]:
        return self.raw["renderer"]


def load_manual_sim_config(path: str | Path) -> ManualSimConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    _validate(raw)
    return ManualSimConfig(raw=raw)


def _validate(raw: dict[str, Any]) -> None:
    required_top = {"sim", "vehicle", "world", "control", "renderer"}
    missing = required_top - set(raw)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    for key in ("dt", "render_hz", "physics_hz"):
        if key not in raw["sim"]:
            raise ValueError(f"Missing sim.{key}")
    for key in ("thrust", "body_rates", "keymap"):
        if key not in raw["control"]:
            raise ValueError(f"Missing control.{key}")

