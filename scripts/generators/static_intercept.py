from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from backends.csim.bindings.types import SimInstance
from scripts.generators.robust_intercept import (
    DEFAULT_ROBUST_INTERCEPT_CONFIG,
    RobustInterceptConfigGenerator,
    SampleEvaluation,
    _deep_merge,
    _label_instance,
    _resolve_instance,
    _sample_record,
)


DEFAULT_STATIC_INTERCEPT_CONFIG: dict[str, Any] = _deep_merge(
    DEFAULT_ROBUST_INTERCEPT_CONFIG,
    {
        "sampling": {
            "n_samples": 1048,
        },
        "controller": {
            "type": "static_intercept_reference",
        },
        "parameters": {
            "range_m": {"min": 10.0, "max": 10.0, "distribution": "uniform"},
        },
        "grid": None,
    },
)


class StaticInterceptConfigGenerator(RobustInterceptConfigGenerator):
    """Generate robust-intercept initial conditions with a stationary target."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(_deep_merge(DEFAULT_STATIC_INTERCEPT_CONFIG, config or {}))

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return copy.deepcopy(DEFAULT_STATIC_INTERCEPT_CONFIG)

    def _resolve_instance(self, point: Any) -> SimInstance:
        return _resolve_instance(
            self.config,
            point,
            target_velocity_override_w=np.zeros(3, dtype=float),
        )


def write_default_config(path: str | Path) -> None:
    config = copy.deepcopy(DEFAULT_STATIC_INTERCEPT_CONFIG)
    Path(path).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_instances(config: dict[str, Any]) -> list[SimInstance]:
    return [evaluation.instance for evaluation in evaluate_samples(config)]


def generate_sample_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [evaluation.record for evaluation in evaluate_samples(config)]


def evaluate_samples(config: dict[str, Any]) -> list[SampleEvaluation]:
    return list(iter_sample_evaluations(config))


def iter_sample_evaluations(config: dict[str, Any]):
    generator = StaticInterceptConfigGenerator(config)
    for point in generator._sample_points:
        instance = generator._resolve_instance(point)
        labels, label_details = _label_instance(instance)
        record = _sample_record(generator.config, point, labels=labels, label_details=label_details)
        record["scenario"] = "static_intercept"
        yield SampleEvaluation(
            instance=instance,
            record=record,
            labels=labels,
            label_details=label_details,
        )
