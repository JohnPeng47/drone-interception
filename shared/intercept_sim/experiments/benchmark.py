from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from intercept_sim.experiments.config import ExperimentConfig, load_experiment_config
from intercept_sim.experiments.runner import ExperimentResult, run_experiment


@dataclass(frozen=True)
class BenchmarkResult:
    results: list[ExperimentResult]

    def rows(self) -> list[dict[str, Any]]:
        return [_summary_row(result) for result in self.results]

    def summary_dict(self) -> dict[str, Any]:
        return {"runs": self.rows()}

    def aggregate_rows(self) -> list[dict[str, Any]]:
        return aggregate_rows(self.rows())


def run_benchmark(
    config_paths: Iterable[str | Path],
    *,
    seeds: Iterable[int] | None = None,
    comment: str | None = None,
) -> BenchmarkResult:
    seed_values = list(seeds) if seeds is not None else [None]
    results: list[ExperimentResult] = []
    for config_path in config_paths:
        base_config = load_experiment_config(config_path)
        for seed in seed_values:
            config = _config_with_seed(base_config, seed) if seed is not None else base_config
            results.append(run_experiment(config, comment=comment or _generated_comment(config, seed=seed)))
    return BenchmarkResult(results=results)


def save_benchmark_result(
    result: BenchmarkResult,
    *,
    json_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    aggregate_csv_path: str | Path | None = None,
) -> None:
    if json_path is not None:
        output_path = Path(json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result.summary_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    if csv_path is not None:
        rows = result.rows()
        output_path = Path(csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = _fieldnames(rows)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if aggregate_csv_path is not None:
        rows = result.aggregate_rows()
        output_path = Path(aggregate_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = _fieldnames(rows)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_base_experiment_name(str(row["experiment"])), []).append(row)

    out: list[dict[str, Any]] = []
    for experiment in sorted(groups):
        group = groups[experiment]
        aggregate: dict[str, Any] = {"experiment": experiment, "runs": len(group)}
        for metric in _numeric_metric_names(group):
            values = [_coerce_float(row.get(metric)) for row in group]
            values = [value for value in values if value is not None]
            if values:
                aggregate[f"{metric}_mean"] = sum(values) / len(values)
                aggregate[f"{metric}_std"] = _std(values)
        out.append(aggregate)
    return out


def _summary_row(result: ExperimentResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment": result.config.name,
        "comment": result.comment,
        "config_path": str(result.config.path) if result.config.path is not None else "",
        "seed": result.config.raw["perception"].get("rng_seed"),
        "duration_s": result.config.duration_s,
        "dt": result.config.dt,
        "steps": len(result.log),
    }
    row.update(result.metrics.to_dict())
    return row


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "experiment",
        "comment",
        "config_path",
        "seed",
        "duration_s",
        "dt",
        "steps",
        "min_distance_m",
        "final_distance_m",
        "catch_time_s",
        "target_visible_fraction",
        "image_feature_availability_fraction",
        "average_image_error_norm",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


def _numeric_metric_names(rows: list[dict[str, Any]]) -> list[str]:
    excluded = {"experiment", "config_path", "seed"}
    return sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in excluded and _coerce_float(value) is not None
        }
    )


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _base_experiment_name(experiment: str) -> str:
    marker = "_seed_"
    if marker not in experiment:
        return experiment
    return experiment.rsplit(marker, 1)[0]


def _config_with_seed(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    raw = copy.deepcopy(config.raw)
    raw["perception"]["rng_seed"] = int(seed)
    raw["experiment"]["name"] = f"{config.name}_seed_{int(seed)}"
    return ExperimentConfig(raw=raw, path=config.path)


def _generated_comment(config: ExperimentConfig, *, seed: int | None) -> str:
    config_name = config.path.name if config.path is not None else config.name
    seed_label = "default" if seed is None else str(seed)
    return f"benchmark config={config_name} seed={seed_label}"
