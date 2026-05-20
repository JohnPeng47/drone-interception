from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    path: Path | None = None

    @property
    def name(self) -> str:
        return str(self.raw["experiment"]["name"])

    @property
    def duration_s(self) -> float:
        return float(self.raw["sim"]["duration_s"])

    @property
    def dt(self) -> float:
        return float(self.raw["sim"]["dt"])

    @property
    def catch_radius_m(self) -> float:
        return float(self.raw["metrics"]["catch_radius_m"])


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    _validate(raw)
    return ExperimentConfig(raw=raw, path=config_path)


def _validate(raw: dict[str, Any]) -> None:
    required_top = {"experiment", "sim", "vehicle", "target", "camera", "perception", "observer", "controller", "metrics"}
    missing = required_top - set(raw)
    if missing:
        raise ValueError(f"Missing experiment config sections: {sorted(missing)}")

    for section, keys in {
        "experiment": ("name",),
        "sim": ("duration_s", "dt"),
        "vehicle": ("initial_position_w",),
        "target": ("initial_position_w", "velocity_w", "radius_m"),
        "camera": ("width_px", "height_px", "fx_px", "fy_px", "hfov_deg", "vfov_deg", "capture_rate_hz"),
        "perception": ("processing_delay_s",),
        "observer": ("type",),
        "controller": ("max_rate_rps",),
        "metrics": ("catch_radius_m",),
    }.items():
        missing_keys = [key for key in keys if key not in raw[section]]
        if missing_keys:
            raise ValueError(f"Missing {section} keys: {missing_keys}")
