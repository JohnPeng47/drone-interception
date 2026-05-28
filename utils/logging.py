from __future__ import annotations

import csv
import datetime as dt
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Clock = Callable[[], dt.datetime]


@dataclass(frozen=True)
class RunsDirLogger:
    """Create and write run artifacts under the date-partitioned .runs tree."""

    prefix: str
    root: Path = Path(".runs")
    clock: Clock = dt.datetime.now

    def __post_init__(self) -> None:
        _validate_name_part(self.prefix, "prefix")

    def run_name(self, suffix: str | None = None) -> str:
        if suffix is None or suffix == "":
            return self.prefix
        _validate_name_part(suffix, "suffix")
        return f"{self.prefix}_{suffix}"

    def date_dir(self) -> Path:
        return Path(self.root) / self.clock().strftime("%Y-%m-%d")

    def create_run_dir(self, suffix: str | None = None, *, exist_ok: bool = False) -> Path:
        run_dir = self.date_dir() / self.run_name(suffix)
        run_dir.mkdir(parents=True, exist_ok=exist_ok)
        return run_dir

    def write_json(self, run_dir: Path, relative_path: str | Path, data: Any) -> Path:
        path = self.resolve(run_dir, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_csv(
        self,
        run_dir: Path,
        relative_path: str | Path,
        rows: Iterable[Mapping[str, Any]],
        fieldnames: list[str],
        *,
        extrasaction: str = "raise",
    ) -> Path:
        path = self.resolve(run_dir, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction=extrasaction)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def resolve(self, run_dir: Path, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"run artifact path must be relative to the run dir: {relative_path}")
        return Path(run_dir) / relative


def _validate_name_part(value: str, label: str) -> None:
    if not value:
        raise ValueError(f"{label} must not be empty")
    part = Path(value)
    if part.is_absolute() or len(part.parts) != 1 or value in {".", ".."}:
        raise ValueError(f"{label} must be a single path name: {value!r}")
