from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


HEADER = struct.Struct("<8sIIQ")
MAGIC = b"CSIMINST"


def write_manifest(
    *,
    scenario_table: Path,
    output: Path,
    ranges_m: list[float],
    closing_speeds_mps: list[float],
) -> dict[str, Any]:
    magic, version, count, payload_len = _read_header(scenario_table)
    if magic != MAGIC:
        raise ValueError(f"{scenario_table} is not a CSIMINST table")

    cell_count = len(ranges_m) * len(closing_speeds_mps)
    if cell_count <= 0:
        raise ValueError("grid must contain at least one cell")
    if count % cell_count != 0:
        raise ValueError(f"{count} records cannot be split evenly into {cell_count} grid cells")

    samples_per_cell = count // cell_count
    cells = []
    index = 0
    for range_m in ranges_m:
        for closing_speed_mps in closing_speeds_mps:
            cells.append(
                {
                    "cell_index": index,
                    "start_index": index * samples_per_cell,
                    "count": samples_per_cell,
                    "range_m": float(range_m),
                    "closing_speed_mps": float(closing_speed_mps),
                }
            )
            index += 1

    manifest = {
        "scenario_table": str(scenario_table),
        "scenario_table_name": scenario_table.name,
        "scenario_table_sha256": _sha256(scenario_table),
        "format": {
            "magic": magic.decode("ascii"),
            "version": int(version),
            "record_count": int(count),
            "payload_bytes": int(payload_len),
            "file_bytes": int(scenario_table.stat().st_size),
        },
        "grid": {
            "order": ["range_m", "closing_speed_mps"],
            "ranges_m": [float(value) for value in ranges_m],
            "closing_speeds_mps": [float(value) for value in closing_speeds_mps],
            "cell_count": cell_count,
            "samples_per_cell": samples_per_cell,
            "cells": cells,
        },
        "indexing": {
            "cell_index": f"record_index // {samples_per_cell}",
            "within_cell_index": f"record_index % {samples_per_cell}",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _read_header(path: Path) -> tuple[bytes, int, int, int]:
    data = path.read_bytes()[: HEADER.size]
    if len(data) != HEADER.size:
        raise ValueError(f"{path} is too small to be a SimInstance table")
    return HEADER.unpack(data)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Write grid labels for a generated CSIM scenario table.")
    parser.add_argument("scenario_table", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ranges-m", default="5,8,20")
    parser.add_argument("--closing-speeds-mps", default="0.5,2,8")
    args = parser.parse_args()

    scenario_table = args.scenario_table
    output = args.output or scenario_table.with_name(f"{scenario_table.stem}_grid_manifest.json")
    manifest = write_manifest(
        scenario_table=scenario_table,
        output=output,
        ranges_m=_parse_float_list(args.ranges_m),
        closing_speeds_mps=_parse_float_list(args.closing_speeds_mps),
    )
    print(json.dumps({"output": str(output), "record_count": manifest["format"]["record_count"]}, indent=2))


if __name__ == "__main__":
    _main()
