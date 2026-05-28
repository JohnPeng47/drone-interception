from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backends.csim.generator.instance_store import SIM_INSTANCE_FORMAT_VERSION, SIM_INSTANCE_MAGIC


SAMPLE_METADATA_SCHEMA_VERSION = 1


def sample_metadata_path(samples_path: str | Path) -> Path:
    return Path(samples_path).with_suffix(".json")


def write_sample_metadata(
    samples_path: str | Path,
    *,
    generator: str,
    strategy: str,
    config: dict[str, Any],
    total_samples: int,
    written_samples: int,
    invalid_samples: int = 0,
    records_path: str | Path | None = None,
    plots: list[str | Path] | None = None,
    labels: dict[str, int] | None = None,
) -> Path:
    samples = Path(samples_path)
    sampling = dict(config.get("sampling", {}))
    sim = dict(config.get("sim", {}))
    metadata: dict[str, Any] = {
        "schema_version": SAMPLE_METADATA_SCHEMA_VERSION,
        "kind": "sim_instance_table",
        "samples": {
            "path": str(samples),
            "file_size_bytes": samples.stat().st_size,
            "format_magic": SIM_INSTANCE_MAGIC.decode("ascii"),
            "format_version": SIM_INSTANCE_FORMAT_VERSION,
            "count": int(written_samples),
        },
        "generator": {
            "name": str(generator),
            "strategy": str(strategy),
        },
        "sampling": {
            "requested_samples": _optional_int(sampling.get("n_samples")),
            "total_samples": int(total_samples),
            "written_samples": int(written_samples),
            "invalid_samples": int(invalid_samples),
            "seed": _optional_int(sampling.get("seed")),
            "scramble": sampling.get("scramble"),
            "active_parameters": list(sampling.get("active_parameters", ())),
        },
        "sim": {
            "backend": sim.get("backend"),
            "duration_s": _optional_float(sim.get("duration_s")),
            "dt": _optional_float(sim.get("dt")),
        },
        "parameters": config.get("parameters", {}),
    }
    if labels is not None:
        metadata["labels"] = {str(key): int(value) for key, value in labels.items()}
    if records_path is not None:
        metadata["records"] = {"path": str(records_path)}
    if plots:
        metadata["plots"] = [str(path) for path in plots]

    path = sample_metadata_path(samples)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
