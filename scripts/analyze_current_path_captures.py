#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backends import SimInstance, read_sim_instances  # noqa: E402
from backends.csim.generator.generators.robust_intercept import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    SAMPLE_BINARY_NAME_TEMPLATE,
)


@dataclass(frozen=True)
class ScenarioCapture:
    seed: int
    captured: bool
    closest_time_s: float
    closest_distance_m: float
    capture_time_s: float | None
    duration_s: float
    capture_radius_m: float
    target_index: int


@dataclass(frozen=True)
class CaptureSummary:
    path: str
    total: int
    captured: int
    missed: int
    capture_fraction: float
    duration_min_s: float | None
    duration_max_s: float | None
    capture_radius_min_m: float | None
    capture_radius_max_m: float | None
    closest_distance_min_m: float | None
    closest_distance_median_m: float | None
    closest_distance_max_m: float | None
    capture_time_min_s: float | None
    capture_time_median_s: float | None
    capture_time_max_s: float | None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count generated robust-intercept scenarios whose current linear "
            "pursuer/target flight paths enter the capture radius."
        ),
        epilog=(
            "robust_intercept.py writes generated binary configs to "
            f"{DEFAULT_OUTPUT_DIR}/"
            f"{SAMPLE_BINARY_NAME_TEMPLATE.format(strategy='<strategy>')}."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more .csimin files, or directories containing *_samples.csimin files.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Override each scenario's configured duration.",
    )
    parser.add_argument(
        "--capture-radius-m",
        type=float,
        default=None,
        help="Override each scenario's configured intercept radius.",
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="Do not clamp closest approach to the scenario duration.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text summary.",
    )
    parser.add_argument(
        "--details-out",
        type=Path,
        default=None,
        help="Optional JSONL file for per-scenario closest-approach results.",
    )
    args = parser.parse_args()

    paths = _expand_inputs(args.inputs)
    summaries: list[CaptureSummary] = []
    detail_rows: list[ScenarioCapture] = []

    for path in paths:
        instances = read_sim_instances(path)
        rows = [
            analyze_instance(
                instance,
                duration_s=args.duration_s,
                capture_radius_m=args.capture_radius_m,
                unbounded=args.unbounded,
            )
            for instance in instances
        ]
        summaries.append(summarize(path, rows))
        detail_rows.extend(rows)

    if args.details_out is not None:
        _write_details(args.details_out, detail_rows)

    if args.json:
        print(json.dumps([asdict(summary) for summary in summaries], indent=2, sort_keys=True))
    else:
        _print_summaries(summaries)


def analyze_instance(
    instance: SimInstance,
    *,
    duration_s: float | None,
    capture_radius_m: float | None,
    unbounded: bool,
) -> ScenarioCapture:
    if not instance.target_initials:
        raise ValueError(f"SimInstance seed={instance.seed} has no targets")

    duration = _duration_s(instance, duration_s, unbounded)
    radius = _capture_radius_m(instance, capture_radius_m)
    pursuer_position = np.asarray(instance.pursuer_initial.position_w, dtype=float)
    pursuer_velocity = np.asarray(instance.pursuer_initial.velocity_w, dtype=float)

    target_results = [
        _analyze_target(
            seed=instance.seed,
            target_index=index,
            pursuer_position=pursuer_position,
            pursuer_velocity=pursuer_velocity,
            target_position=np.asarray(target.position_w, dtype=float),
            target_velocity=np.asarray(target.velocity_w, dtype=float),
            duration_s=duration,
            capture_radius_m=radius,
        )
        for index, target in enumerate(instance.target_initials)
    ]
    captured = [result for result in target_results if result.captured]
    if captured:
        return min(captured, key=lambda result: result.capture_time_s if result.capture_time_s is not None else math.inf)
    return min(target_results, key=lambda result: result.closest_distance_m)


def _analyze_target(
    *,
    seed: int,
    target_index: int,
    pursuer_position: np.ndarray,
    pursuer_velocity: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    duration_s: float,
    capture_radius_m: float,
) -> ScenarioCapture:
    relative_position = pursuer_position - target_position
    relative_velocity = pursuer_velocity - target_velocity
    closest_time = _closest_time(relative_position, relative_velocity, duration_s)
    closest_position = relative_position + relative_velocity * closest_time
    closest_distance = float(np.linalg.norm(closest_position))
    capture_time = _capture_time(relative_position, relative_velocity, duration_s, capture_radius_m)
    return ScenarioCapture(
        seed=int(seed),
        captured=capture_time is not None,
        closest_time_s=float(closest_time),
        closest_distance_m=closest_distance,
        capture_time_s=capture_time,
        duration_s=float(duration_s),
        capture_radius_m=float(capture_radius_m),
        target_index=target_index,
    )


def _closest_time(relative_position: np.ndarray, relative_velocity: np.ndarray, duration_s: float) -> float:
    speed_sq = float(np.dot(relative_velocity, relative_velocity))
    if speed_sq <= 1.0e-12:
        return 0.0
    unconstrained = -float(np.dot(relative_position, relative_velocity)) / speed_sq
    if math.isinf(duration_s):
        return max(0.0, unconstrained)
    return min(max(0.0, unconstrained), duration_s)


def _capture_time(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    duration_s: float,
    capture_radius_m: float,
) -> float | None:
    radius_sq = float(capture_radius_m) ** 2
    if float(np.dot(relative_position, relative_position)) <= radius_sq:
        return 0.0

    a = float(np.dot(relative_velocity, relative_velocity))
    if a <= 1.0e-12:
        return None

    b = 2.0 * float(np.dot(relative_position, relative_velocity))
    c = float(np.dot(relative_position, relative_position)) - radius_sq
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None

    sqrt_discriminant = math.sqrt(max(discriminant, 0.0))
    roots = sorted(((-b - sqrt_discriminant) / (2.0 * a), (-b + sqrt_discriminant) / (2.0 * a)))
    for root in roots:
        if root < 0.0:
            continue
        if math.isinf(duration_s) or root <= duration_s:
            return float(root)
    return None


def _duration_s(instance: SimInstance, override: float | None, unbounded: bool) -> float:
    if unbounded:
        return math.inf
    if override is not None:
        return _positive_float(override, "duration")
    if instance.config is None:
        raise ValueError(f"SimInstance seed={instance.seed} has no SimConfig; pass --duration-s")
    return _positive_float(instance.config.options.duration_s, "duration")


def _capture_radius_m(instance: SimInstance, override: float | None) -> float:
    if override is not None:
        return _positive_float(override, "capture radius")
    if instance.config is not None and instance.config.intercept_radius_m > 0.0:
        return float(instance.config.intercept_radius_m)
    if instance.config is None or not instance.config.targets:
        raise ValueError(f"SimInstance seed={instance.seed} has no target config; pass --capture-radius-m")
    return _positive_float(instance.config.targets[0].radius_m, "capture radius")


def _positive_float(value: float, label: str) -> float:
    result = float(value)
    if not result > 0.0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return result


def summarize(path: Path, rows: list[ScenarioCapture]) -> CaptureSummary:
    captured_rows = [row for row in rows if row.captured]
    return CaptureSummary(
        path=str(path),
        total=len(rows),
        captured=len(captured_rows),
        missed=len(rows) - len(captured_rows),
        capture_fraction=0.0 if not rows else len(captured_rows) / len(rows),
        duration_min_s=_min(row.duration_s for row in rows),
        duration_max_s=_max(row.duration_s for row in rows),
        capture_radius_min_m=_min(row.capture_radius_m for row in rows),
        capture_radius_max_m=_max(row.capture_radius_m for row in rows),
        closest_distance_min_m=_min(row.closest_distance_m for row in rows),
        closest_distance_median_m=_median(row.closest_distance_m for row in rows),
        closest_distance_max_m=_max(row.closest_distance_m for row in rows),
        capture_time_min_s=_min(row.capture_time_s for row in captured_rows),
        capture_time_median_s=_median(row.capture_time_s for row in captured_rows),
        capture_time_max_s=_max(row.capture_time_s for row in captured_rows),
    )


def _expand_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = item.expanduser()
        if path.is_dir():
            paths.extend(sorted(path.glob("*_samples.csimin")))
        else:
            paths.append(path)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing .csimin input(s): {', '.join(str(path) for path in missing)}")
    if not paths:
        raise FileNotFoundError("No *_samples.csimin files found in the supplied input directories")
    return paths


def _write_details(path: Path, rows: list[ScenarioCapture]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def _print_summaries(summaries: list[CaptureSummary]) -> None:
    for summary in summaries:
        print(f"{summary.path}")
        print(f"  scenarios: {summary.total}")
        print(f"  captured_on_current_path: {summary.captured}")
        print(f"  missed: {summary.missed}")
        print(f"  capture_fraction: {summary.capture_fraction:.2%}")
        print(
            "  duration_s: "
            f"{_fmt(summary.duration_min_s)}..{_fmt(summary.duration_max_s)}"
        )
        print(
            "  capture_radius_m: "
            f"{_fmt(summary.capture_radius_min_m)}..{_fmt(summary.capture_radius_max_m)}"
        )
        print(
            "  closest_distance_m min/median/max: "
            f"{_fmt(summary.closest_distance_min_m)} / "
            f"{_fmt(summary.closest_distance_median_m)} / "
            f"{_fmt(summary.closest_distance_max_m)}"
        )
        print(
            "  capture_time_s min/median/max: "
            f"{_fmt(summary.capture_time_min_s)} / "
            f"{_fmt(summary.capture_time_median_s)} / "
            f"{_fmt(summary.capture_time_max_s)}"
        )


def _values(values: Any) -> list[float]:
    return [float(value) for value in values if value is not None]


def _min(values: Any) -> float | None:
    items = _values(values)
    return None if not items else min(items)


def _max(values: Any) -> float | None:
    items = _values(values)
    return None if not items else max(items)


def _median(values: Any) -> float | None:
    items = _values(values)
    return None if not items else float(np.median(np.asarray(items, dtype=float)))


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.6g}"


if __name__ == "__main__":
    main()
