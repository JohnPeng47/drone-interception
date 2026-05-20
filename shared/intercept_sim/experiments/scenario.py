from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import gzip
import json
from pathlib import Path
import re
from typing import Any, Protocol, Sequence

import yaml

from intercept_sim.experiments.config import ExperimentConfig
from intercept_sim.experiments.runner import run_experiment
from intercept_sim.experiments.telemetry import ExperimentTelemetry


DEFAULT_LOG_ROOT = Path(".runs")


class ScenarioMetrics(Protocol):
    def rows(self) -> list[dict[str, Any]]: ...
    def aggregate_rows(self) -> list[dict[str, Any]]: ...


class Scenario(Protocol):
    @property
    def name(self) -> str: ...
    def build_experiment_configs(self) -> list[ExperimentConfig]: ...
    def comment_for_config(self, config: ExperimentConfig) -> str: ...
    def evaluate(self, telemetry: Sequence[ExperimentTelemetry]) -> ScenarioMetrics: ...


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    telemetry: list[ExperimentTelemetry]
    metrics: ScenarioMetrics
    path: Path | None = None


def run_scenario(scenario: Scenario, *, comment: str | None = None) -> ScenarioResult:
    telemetry: list[ExperimentTelemetry] = []
    for config in scenario.build_experiment_configs():
        result = run_experiment(config, comment=comment or scenario.comment_for_config(config))
        if result.telemetry is None:
            raise RuntimeError(f"Experiment {config.name} did not produce telemetry")
        telemetry.append(result.telemetry)
    return ScenarioResult(
        name=scenario.name,
        telemetry=telemetry,
        metrics=scenario.evaluate(telemetry),
    )


def save_scenario_result(
    result: ScenarioResult,
    *,
    log_root: str | Path = DEFAULT_LOG_ROOT,
    root_dir: str | Path | None = None,
    detailed_trace: bool = False,
    created_at: datetime | None = None,
) -> Path:
    timestamp = created_at or datetime.now()
    scenario_root = Path(root_dir) if root_dir is not None else Path(log_root) / "scenarios"
    group_dir = _next_group_dir(scenario_root, result.name, timestamp)
    group_dir.mkdir(parents=True, exist_ok=False)

    run_rows = result.metrics.rows()
    aggregate_rows = result.metrics.aggregate_rows()
    run_entries: list[dict[str, Any]] = []

    for index, telemetry in enumerate(result.telemetry, start=1):
        run_dir = group_dir / f"run_{index}"
        run_dir.mkdir(parents=True, exist_ok=False)
        run_metric = run_rows[index - 1] if index - 1 < len(run_rows) else None
        trace_path = _write_run_files(run_dir, telemetry, run_metric, detailed_trace=detailed_trace)
        run_entries.append(
            {
                "index": index,
                "run_dir": run_dir.name,
                "experiment": telemetry.experiment_id,
                "comment": telemetry.comment,
                "summary_path": f"{run_dir.name}/summary.json",
                "config_path": f"{run_dir.name}/config.yaml",
                "telemetry_path": f"{run_dir.name}/{trace_path.name}" if trace_path is not None else None,
                "metrics_path": f"{run_dir.name}/scenario_metrics.json",
            }
        )

    group = {
        "schema_version": 1,
        "scenario_name": result.name,
        "created_at": timestamp.isoformat(timespec="seconds"),
        "group_dir": group_dir.name,
        "run_count": len(result.telemetry),
        "detailed_trace": detailed_trace,
        "runs": run_entries,
        "aggregate_metrics": aggregate_rows,
    }
    _write_json(group_dir / "group.json", group)
    return group_dir


def _write_run_files(
    run_dir: Path,
    telemetry: ExperimentTelemetry,
    run_metric: dict[str, Any] | None,
    *,
    detailed_trace: bool,
) -> Path | None:
    _write_json(run_dir / "summary.json", telemetry.to_summary_dict())
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(telemetry.config, handle, sort_keys=False)
    _write_json(run_dir / "scenario_metrics.json", run_metric or {})
    if not detailed_trace:
        return None

    trace_path = run_dir / "telemetry.jsonl.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "metadata", **telemetry.to_summary_dict()}, sort_keys=True))
        handle.write("\n")
        for step in telemetry.steps:
            handle.write(json.dumps({"kind": "step", **step.to_dict()}, sort_keys=True))
            handle.write("\n")
    return trace_path


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _next_group_dir(root_dir: Path, scenario_name: str, timestamp: datetime) -> Path:
    scenario_dir = root_dir / _slug(scenario_name)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    date_suffix = timestamp.strftime("%y_%m_%d")
    max_index = 0
    pattern = re.compile(r"^(\d+)_\d{2}_\d{2}_\d{2}$")
    for child in scenario_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match is not None:
            max_index = max(max_index, int(match.group(1)))
    return scenario_dir / f"{max_index + 1}_{date_suffix}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "scenario"
