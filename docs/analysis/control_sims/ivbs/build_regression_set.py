from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from backends.csim.generator.instance_store import write_sim_instances
from backends.csim.generator.instance_store import read_sim_instances


def main() -> int:
    parser = argparse.ArgumentParser(description="Build IVBS scenario-class analysis and regression subset.")
    parser.add_argument("--scenario-table", type=Path, required=True)
    parser.add_argument("--ivbs-trials", type=Path, required=True)
    parser.add_argument("--beihang-trials", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path("scripts/generators/sim_instances/ivbs_regression_20260604/sobol_samples.csimin"),
    )
    args = parser.parse_args()

    instances = read_sim_instances(args.scenario_table)
    ivbs_trials = _read_trials(args.ivbs_trials)
    beihang_trials = _read_trials(args.beihang_trials)
    rows = _joined_rows(instances, ivbs_trials, beihang_trials)
    if len(rows) != len(instances):
        raise ValueError(
            f"joined {len(rows)} scenarios, expected {len(instances)}; "
            "check that both trial files cover the scenario table seeds"
        )
    selected = _select_regression_rows(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    selected_instances = [instances[int(row["sample_index"])] for row in selected]
    write_sim_instances(args.output_table, selected_instances)

    joined_path = args.out_dir / "joined_scenarios.csv"
    slice_path = args.out_dir / "slice_summary.csv"
    selected_path = args.out_dir / "selected_regression_rows.csv"
    _write_csv(joined_path, rows)
    _write_csv(slice_path, _slice_summary(rows))
    _write_csv(selected_path, selected)
    _write_metadata(args, selected, selected_path)

    summary = {
        "scenario_table": str(args.scenario_table),
        "ivbs_trials": str(args.ivbs_trials),
        "beihang_trials": str(args.beihang_trials),
        "joined_scenarios": str(joined_path),
        "slice_summary": str(slice_path),
        "selected_rows": str(selected_path),
        "output_table": str(args.output_table),
        "n": len(rows),
        "selected_n": len(selected),
        "selected_counts": _counts(row["selection_class"] for row in selected),
        "ivbs_caught": sum(1 for row in rows if _bool(row["ivbs_caught"])),
        "beihang_caught": sum(1 for row in rows if _bool(row["beihang_caught"])),
        "ivbs_only": sum(1 for row in rows if _bool(row["ivbs_caught"]) and not _bool(row["beihang_caught"])),
        "beihang_only": sum(1 for row in rows if _bool(row["beihang_caught"]) and not _bool(row["ivbs_caught"])),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _read_trials(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["seed"]): row for row in csv.DictReader(handle)}


def _joined_rows(instances, ivbs_trials, beihang_trials) -> list[dict[str, Any]]:
    rows = []
    for index, instance in enumerate(instances):
        seed = int(instance.seed)
        if seed not in ivbs_trials or seed not in beihang_trials:
            continue
        features = _instance_features(index, instance)
        ivbs = ivbs_trials[seed]
        beihang = beihang_trials[seed]
        ivbs_caught = _bool(ivbs["caught"])
        beihang_caught = _bool(beihang["caught"])
        rows.append(
            {
                **features,
                "ivbs_caught": ivbs_caught,
                "beihang_caught": beihang_caught,
                "outcome_class": _outcome_class(ivbs_caught, beihang_caught),
                "ivbs_min_distance_m": _float(ivbs["min_distance_m"]),
                "beihang_min_distance_m": _float(beihang["min_distance_m"]),
                "ivbs_visible_fraction": _float(ivbs["visible_fraction"]),
                "beihang_visible_fraction": _float(beihang["visible_fraction"]),
                "ivbs_control_effort": _float(ivbs["control_effort"]),
                "beihang_control_effort": _float(beihang["control_effort"]),
                "visibility_bucket": _visibility_bucket(_float(ivbs["visible_fraction"])),
                "closing_bucket": _closing_bucket(features["initial_closing_speed_mps"]),
            }
        )
    rows.sort(key=lambda row: int(row["seed"]))
    return rows


def _instance_features(index: int, instance) -> dict[str, Any]:
    pursuer_p = np.asarray(instance.pursuer_initial.position_w, dtype=float).reshape(3)
    pursuer_v = np.asarray(instance.pursuer_initial.velocity_w, dtype=float).reshape(3)
    target_p = np.asarray(instance.target_initial.position_w, dtype=float).reshape(3)
    target_v = np.asarray(instance.target_initial.velocity_w, dtype=float).reshape(3)
    los_w = target_p - pursuer_p
    range_m = float(np.linalg.norm(los_w))
    los_unit = los_w / max(range_m, 1.0e-12)
    rel_v = pursuer_v - target_v
    closing_speed = float(rel_v @ los_unit)
    r_wb = _quat_xyzw_to_rot(np.asarray(instance.pursuer_initial.quat_xyzw, dtype=float).reshape(4))
    camera = instance.config.cameras[0]
    r_b2c = np.asarray(camera.body_to_camera, dtype=float).reshape(3, 3)
    r_wc = r_wb @ r_b2c.T
    camera_forward_w = r_wc @ np.array([1.0, 0.0, 0.0], dtype=float)
    camera_azimuth_deg, camera_elevation_deg = _azimuth_elevation_deg(camera_forward_w)
    camera_p = pursuer_p + r_wb @ np.asarray(camera.position_b, dtype=float).reshape(3)
    target_c = r_b2c @ (r_wb.T @ (target_p - camera_p))
    uv_norm = np.array([target_c[1] / max(target_c[0], 1.0e-12), target_c[2] / max(target_c[0], 1.0e-12)])
    intrinsics = camera.intrinsics
    u_fraction = float(uv_norm[0] / math.tan(float(intrinsics.hfov_rad) / 2.0))
    v_fraction = float(uv_norm[1] / math.tan(float(intrinsics.vfov_rad) / 2.0))
    return {
        "seed": int(instance.seed),
        "sample_index": int(index),
        "initial_range_m": range_m,
        "initial_closing_speed_mps": closing_speed,
        "camera_azimuth_deg": camera_azimuth_deg,
        "camera_elevation_deg": camera_elevation_deg,
        "camera_u_fraction": u_fraction,
        "camera_v_fraction": v_fraction,
        "image_plane_bucket": _image_plane_bucket(u_fraction, v_fraction),
        "camera_elevation_bucket": _camera_elevation_bucket(camera_elevation_deg),
        "camera_azimuth_bucket": _azimuth_bucket(camera_azimuth_deg),
    }


def _select_regression_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("shared_catch", lambda row: row["outcome_class"] == "shared_catch", 4),
        ("ivbs_only", lambda row: row["outcome_class"] == "ivbs_only", 4),
        ("beihang_only", lambda row: row["outcome_class"] == "beihang_only", 4),
        (
            "high_visibility_miss",
            lambda row: (not _bool(row["ivbs_caught"])) and float(row["ivbs_visible_fraction"]) >= 0.1,
            6,
        ),
        (
            "low_visibility_miss",
            lambda row: (not _bool(row["ivbs_caught"])) and float(row["ivbs_visible_fraction"]) < 0.02,
            6,
        ),
    ]
    selected_by_seed: dict[int, dict[str, Any]] = {}
    for label, predicate, limit in specs:
        candidates = [row for row in rows if predicate(row)]
        candidates.sort(key=lambda row: (float(row["ivbs_min_distance_m"]), int(row["seed"])))
        for row in candidates[:limit]:
            item = dict(row)
            item["selection_class"] = label
            selected_by_seed.setdefault(int(row["seed"]), item)
    selected = list(selected_by_seed.values())
    selected.sort(key=lambda row: (str(row["selection_class"]), int(row["seed"])))
    return selected


def _slice_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for field in (
        "visibility_bucket",
        "closing_bucket",
        "image_plane_bucket",
        "camera_elevation_bucket",
        "camera_azimuth_bucket",
        "outcome_class",
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[field])].append(row)
        for bucket, bucket_rows in sorted(grouped.items()):
            out.append(
                {
                    "slice": field,
                    "bucket": bucket,
                    "n": len(bucket_rows),
                    "ivbs_caught": sum(1 for row in bucket_rows if _bool(row["ivbs_caught"])),
                    "beihang_caught": sum(1 for row in bucket_rows if _bool(row["beihang_caught"])),
                    "ivbs_catch_fraction": _mean(_bool(row["ivbs_caught"]) for row in bucket_rows),
                    "beihang_catch_fraction": _mean(_bool(row["beihang_caught"]) for row in bucket_rows),
                    "ivbs_min_distance_p50_m": _percentile(
                        (row["ivbs_min_distance_m"] for row in bucket_rows),
                        50,
                    ),
                    "ivbs_visible_fraction_mean": _mean(row["ivbs_visible_fraction"] for row in bucket_rows),
                }
            )
    return out


def _write_metadata(args, selected: list[dict[str, Any]], selected_path: Path) -> None:
    metadata = {
        "schema_version": 1,
        "kind": "sim_instance_table",
        "generator": {
            "name": "IVBSRegressionSubsetAnalysis",
            "strategy": "selected_from_existing_table",
        },
        "samples": {
            "path": str(args.output_table),
            "count": len(selected),
        },
        "source": {
            "scenario_table": str(args.scenario_table),
            "ivbs_trials": str(args.ivbs_trials),
            "beihang_trials": str(args.beihang_trials),
        },
        "records": {
            "path": str(selected_path),
        },
        "selection_classes": _counts(row["selection_class"] for row in selected),
    }
    args.output_table.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _outcome_class(ivbs_caught: bool, beihang_caught: bool) -> str:
    if ivbs_caught and beihang_caught:
        return "shared_catch"
    if ivbs_caught:
        return "ivbs_only"
    if beihang_caught:
        return "beihang_only"
    return "shared_miss"


def _visibility_bucket(value: float) -> str:
    if value < 0.02:
        return "00_lt_0.02"
    if value < 0.05:
        return "01_0.02_0.05"
    if value < 0.10:
        return "02_0.05_0.10"
    return "03_ge_0.10"


def _closing_bucket(value: float) -> str:
    if value < 0.35:
        return "00_lt_0.35"
    if value < 0.45:
        return "01_0.35_0.45"
    return "02_ge_0.45"


def _image_plane_bucket(u_fraction: float, v_fraction: float) -> str:
    radius = max(abs(float(u_fraction)), abs(float(v_fraction)))
    if radius < 0.35:
        return "center"
    if radius < 0.70:
        return "mid"
    return "edge"


def _camera_elevation_bucket(value: float) -> str:
    if value < -30.0:
        return "down"
    if value > 30.0:
        return "up"
    return "level"


def _azimuth_bucket(value: float) -> str:
    azimuth = float(value) % 360.0
    if azimuth < 90.0:
        return "front_right"
    if azimuth < 180.0:
        return "rear_right"
    if azimuth < 270.0:
        return "rear_left"
    return "front_left"


def _azimuth_elevation_deg(vector: np.ndarray) -> tuple[float, float]:
    unit = np.asarray(vector, dtype=float).reshape(3)
    unit /= max(float(np.linalg.norm(unit)), 1.0e-12)
    return (
        float(math.degrees(math.atan2(unit[1], unit[0])) % 360.0),
        float(math.degrees(math.asin(np.clip(unit[2], -1.0, 1.0)))),
    )


def _quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1.0e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def _mean(values) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _percentile(values, percentile: float) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.percentile(finite, percentile)) if finite else float("nan")


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
