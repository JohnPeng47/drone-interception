from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

from intercept_sim.experiments.benchmark import BenchmarkResult
from intercept_sim.experiments.config import ExperimentConfig, load_experiment_config
from intercept_sim.experiments.runner import run_experiment


DEFAULT_DELAYS_S = (0.0, 0.08)
DEFAULT_CAMERA_RATES_HZ = (30.0, 50.0)
DEFAULT_OBSERVERS = ("latest", "constant_velocity", "delayed_replay", "beihang_image_ekf")


def build_delay_benchmark_configs(
    base_config: ExperimentConfig | str | Path,
    *,
    delays_s: Iterable[float] = DEFAULT_DELAYS_S,
    camera_rates_hz: Iterable[float] = DEFAULT_CAMERA_RATES_HZ,
    observers: Iterable[str] = DEFAULT_OBSERVERS,
) -> list[ExperimentConfig]:
    base = base_config if isinstance(base_config, ExperimentConfig) else load_experiment_config(base_config)
    configs: list[ExperimentConfig] = []
    for delay_s in delays_s:
        for camera_rate_hz in camera_rates_hz:
            for observer in observers:
                raw = copy.deepcopy(base.raw)
                raw["experiment"]["name"] = _variant_name(base.name, delay_s, camera_rate_hz, observer)
                raw["perception"]["processing_delay_s"] = float(delay_s)
                raw["camera"]["capture_rate_hz"] = float(camera_rate_hz)
                raw["observer"]["type"] = observer
                if observer == "constant_velocity":
                    raw["observer"].setdefault("history_size", 4)
                elif observer == "beihang_image_ekf":
                    raw["observer"].setdefault("history_size", 50)
                else:
                    raw["observer"].pop("history_size", None)
                configs.append(ExperimentConfig(raw=raw, path=base.path))
    return configs


def run_delay_benchmark(
    base_config: ExperimentConfig | str | Path,
    *,
    delays_s: Iterable[float] = DEFAULT_DELAYS_S,
    camera_rates_hz: Iterable[float] = DEFAULT_CAMERA_RATES_HZ,
    observers: Iterable[str] = DEFAULT_OBSERVERS,
    comment: str | None = None,
) -> BenchmarkResult:
    configs = build_delay_benchmark_configs(
        base_config,
        delays_s=delays_s,
        camera_rates_hz=camera_rates_hz,
        observers=observers,
    )
    return BenchmarkResult(results=[run_experiment(config, comment=comment or _generated_comment(config)) for config in configs])


def add_zero_delay_delta_rows(result: BenchmarkResult) -> list[dict[str, float | int | str | None]]:
    rows = result.rows()
    baselines = {
        _delay_key(row): row
        for row in rows
        if _delay_key(row) is not None and _delay_key(row)[2] == 0
    }
    out: list[dict[str, float | int | str | None]] = []
    for row in rows:
        key = _delay_key(row)
        if key is None:
            out.append(row)
            continue
        baseline = baselines.get((key[0], key[1], 0))
        enriched = dict(row)
        if baseline is not None:
            for metric in ("final_distance_m", "min_distance_m", "average_image_error_norm"):
                current_value = row.get(metric)
                baseline_value = baseline.get(metric)
                if current_value is not None and baseline_value is not None:
                    enriched[f"{metric}_delta_vs_zero_delay"] = float(current_value) - float(baseline_value)
        out.append(enriched)
    return out


def _variant_name(base_name: str, delay_s: float, camera_rate_hz: float, observer: str) -> str:
    delay_ms = int(round(float(delay_s) * 1000.0))
    rate_hz = int(round(float(camera_rate_hz)))
    return f"{base_name}_{observer}_{delay_ms}ms_{rate_hz}hz"


def _generated_comment(config: ExperimentConfig) -> str:
    observer = config.raw["observer"]["type"]
    delay_ms = int(round(float(config.raw["perception"]["processing_delay_s"]) * 1000.0))
    rate_hz = int(round(float(config.raw["camera"]["capture_rate_hz"])))
    return f"delay_benchmark observer={observer} delay_ms={delay_ms} camera_rate_hz={rate_hz}"


def _delay_key(row: dict[str, object]) -> tuple[str, int, int] | None:
    experiment = str(row.get("experiment", ""))
    parts = experiment.rsplit("_", 2)
    if len(parts) != 3:
        return None
    base_and_observer, delay_part, rate_part = parts
    if not delay_part.endswith("ms") or not rate_part.endswith("hz"):
        return None
    try:
        delay_ms = int(delay_part[:-2])
        rate_hz = int(rate_part[:-2])
    except ValueError:
        return None
    observer_suffixes = sorted(DEFAULT_OBSERVERS, key=len, reverse=True)
    for observer in observer_suffixes:
        suffix = f"_{observer}"
        if base_and_observer.endswith(suffix):
            base = base_and_observer[: -len(suffix)]
            return (f"{base}_{observer}", rate_hz, delay_ms)
    return None
