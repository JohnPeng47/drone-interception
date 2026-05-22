from __future__ import annotations

import copy
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...noise_config import NoiseConfig
from ...types import RunnerStep


@dataclass(frozen=True)
class ExperimentMetrics:
    min_distance_m: float
    final_distance_m: float
    catch_time_s: float | None
    target_visible_fraction: float
    image_feature_availability_fraction: float
    average_image_error_norm: float | None
    miss_distance_m: float

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class ControlSimRunResult:
    seed: int
    config: dict[str, Any]
    metrics: ExperimentMetrics
    wall_s: float
    log: list[RunnerStep]

    def row(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            **self.metrics.to_dict(),
            "wall_s": float(self.wall_s),
        }

SimRunResult = ControlSimRunResult


def run_drake_config(
    raw_config: dict[str, Any],
    *,
    seed: int,
    controller_gains: dict[str, float] | None = None,
    noise_config: NoiseConfig | None = None,
) -> ControlSimRunResult:
    from pydrake.systems.analysis import Simulator

    from ...diagram import build_diagram_from_config

    raw = copy.deepcopy(raw_config)
    diagram, logger = build_diagram_from_config(
        raw,
        controller_gains=controller_gains,
        noise_config=noise_config,
    )
    sim = Simulator(diagram)
    sim.Initialize()

    t0 = time.perf_counter()
    sim.AdvanceTo(float(raw["sim"]["duration_s"]))
    wall_s = time.perf_counter() - t0

    dt = float(raw["sim"]["dt"])
    num_steps = int(math.ceil(float(raw["sim"]["duration_s"]) / dt))
    log = logger.get_log()[:num_steps]
    metrics = compute_metrics(log, catch_radius_m=float(raw["metrics"]["catch_radius_m"]))
    return ControlSimRunResult(
        seed=int(seed),
        config=raw,
        metrics=metrics,
        wall_s=float(wall_s),
        log=log,
    )


def compute_metrics(log: list[RunnerStep], *, catch_radius_m: float) -> ExperimentMetrics:
    if not log:
        return ExperimentMetrics(
            min_distance_m=float("nan"),
            final_distance_m=float("nan"),
            catch_time_s=None,
            target_visible_fraction=0.0,
            image_feature_availability_fraction=0.0,
            average_image_error_norm=None,
            miss_distance_m=float("nan"),
        )

    distances = np.array([_distance_to_primary_target(step) for step in log], dtype=float)
    catch_indices = np.flatnonzero(distances <= catch_radius_m)
    catch_time = float(log[int(catch_indices[0])].t) if catch_indices.size else None

    captures = [step.capture for step in log if step.capture is not None]
    visible_count = sum(1 for capture in captures if capture.detected)
    target_visible_fraction = visible_count / len(captures) if captures else 0.0

    features = [step.observer_state.image_feature for step in log]
    available_features = [
        feature
        for feature in features
        if feature is not None and feature.detected and feature.uv_norm is not None
    ]
    image_feature_availability_fraction = len(available_features) / len(log)
    if available_features:
        average_image_error_norm = float(
            np.mean([
                np.linalg.norm(np.asarray(feature.uv_norm, dtype=float))
                for feature in available_features
            ])
        )
    else:
        average_image_error_norm = None

    return ExperimentMetrics(
        min_distance_m=float(np.min(distances)),
        final_distance_m=float(distances[-1]),
        catch_time_s=catch_time,
        target_visible_fraction=float(target_visible_fraction),
        image_feature_availability_fraction=float(image_feature_availability_fraction),
        average_image_error_norm=average_image_error_norm,
        miss_distance_m=float(np.min(distances)),
    )


def circular_error_probable(values: list[float] | np.ndarray, *, percentile: float = 50.0) -> float:
    distances = np.asarray(values, dtype=float)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, percentile))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_log_jsonl_gz(path: Path, result: ControlSimRunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "kind": "metadata",
            "experiment_id": str(result.config["experiment"]["name"]),
            "config": result.config,
            "metrics": result.metrics.to_dict(),
        }, sort_keys=True))
        handle.write("\n")
        for step in result.log:
            handle.write(json.dumps({"kind": "step", **_step_to_dict(step)}, sort_keys=True))
            handle.write("\n")


def _distance_to_primary_target(step: RunnerStep) -> float:
    if not step.scene.targets:
        return float("nan")
    pursuer_pos = np.asarray(step.scene.pursuer.position_w, dtype=float)
    target_pos = np.asarray(step.scene.targets[0].position_w, dtype=float)
    return float(np.linalg.norm(target_pos - pursuer_pos))


def _step_to_dict(step: RunnerStep) -> dict[str, Any]:
    target = step.scene.targets[0] if step.scene.targets else None
    return {
        "t": float(step.t),
        "pursuer_position_w": np.asarray(step.scene.pursuer.position_w, dtype=float).tolist(),
        "pursuer_velocity_w": np.asarray(step.scene.pursuer.velocity_w, dtype=float).tolist(),
        "target_position_w": (
            None if target is None else np.asarray(target.position_w, dtype=float).tolist()
        ),
        "target_velocity_w": (
            None if target is None else np.asarray(target.velocity_w, dtype=float).tolist()
        ),
        "capture_detected": None if step.capture is None else bool(step.capture.detected),
        "feature_detected": (
            step.observer_state.image_feature is not None
            and bool(step.observer_state.image_feature.detected)
        ),
        "thrust_n": float(step.command.thrust_n),
        "body_rates_b": np.asarray(step.command.body_rates_b, dtype=float).tolist(),
    }
