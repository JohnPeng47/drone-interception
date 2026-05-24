from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backends.csim.generator.instance_store import read_sim_instances
from backends.csim.bindings.types import SimInstance


@dataclass(frozen=True)
class ScenarioLabel:
    scenario_index: int
    cell_index: int | None
    range_m: float | None
    closing_speed_mps: float | None


class ScenarioTable:
    def __init__(
        self,
        path: str | Path,
        *,
        manifest_path: str | Path | None = None,
        max_scenarios: int | None = None,
    ):
        self.path = Path(path)
        self.instances = tuple(read_sim_instances(self.path, count=max_scenarios))
        if not self.instances:
            raise ValueError(f"{self.path} did not contain any scenarios")
        self.manifest_path = None if manifest_path is None else Path(manifest_path)
        self.manifest = _load_manifest(self.manifest_path)
        self.cells = tuple((self.manifest or {}).get("grid", {}).get("cells", ()))
        self.samples_per_cell = (self.manifest or {}).get("grid", {}).get("samples_per_cell")

    @property
    def count(self) -> int:
        return len(self.instances)

    def get(self, index: int) -> SimInstance:
        return self.instances[int(index) % len(self.instances)]

    def label(self, index: int) -> ScenarioLabel:
        index = int(index) % len(self.instances)
        if not self.cells or not self.samples_per_cell:
            return ScenarioLabel(index, None, None, None)
        cell_index = min(index // int(self.samples_per_cell), len(self.cells) - 1)
        cell = self.cells[cell_index]
        return ScenarioLabel(
            scenario_index=index,
            cell_index=int(cell["cell_index"]),
            range_m=float(cell["range_m"]),
            closing_speed_mps=float(cell["closing_speed_mps"]),
        )


def _load_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
