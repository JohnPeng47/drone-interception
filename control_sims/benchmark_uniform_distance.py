"""Benchmark control_sims controllers on robust uniform-distance datapoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backends.csim.generator.instance_store import read_sim_instances
from backends.csim.generator.generators.robust_intercept_uniform_distance import (
    RobustInterceptUniformDistanceConfigGenerator,
)

from control_sims.beihang_minimal_sim.config import (
    CameraConfig as MinimalCameraConfig,
    TargetConfig as MinimalTargetConfig,
    TrialConfig as MinimalTrialConfig,
    VehicleConfig as MinimalVehicleConfig,
)
from control_sims.beihang_minimal_sim.replay import run_trial as run_minimal_trial
from control_sims.beihang_paper_sim._paths import ensure_paths


ensure_paths()

from pydrake.systems.analysis import Simulator  # noqa: E402

from control_sims.beihang_paper_sim.diagram import build_diagram_from_config  # noqa: E402
from control_sims.beihang_paper_sim.noise_config import NoiseConfig  # noqa: E402


SIMS = ("beihang_minimal", "beihang_paper")
_GENERATOR: RobustInterceptUniformDistanceConfigGenerator | None = None


@dataclass(frozen=True)
class _PaperMetrics:
    catch_time_s: float | None
    min_distance_m: float
    final_distance_m: float
    target_visible_fraction: float


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--scenario-table", type=Path, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sims", default=",".join(SIMS))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of scenario workers. Defaults to min(samples, CPU count - 1). Use 1 for serial.",
    )
    args = parser.parse_args()

    sims = tuple(item.strip() for item in args.sims.split(",") if item.strip())
    unknown = sorted(set(sims) - set(SIMS))
    if unknown:
        raise ValueError(f"Unknown sim names: {unknown}; expected {SIMS}")

    run_dir = args.out_dir or Path(".runs/control_sims_uniform_distance") / dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    global _GENERATOR
    source: str
    generator = RobustInterceptUniformDistanceConfigGenerator()
    if args.scenario_table is None:
        sample_count = int(args.samples) if args.samples is not None else 1500
        _GENERATOR = generator
        tasks = list(range(int(args.seed_start), int(args.seed_start) + sample_count))
        source = "robust_intercept_uniform_distance"
        duration_s = float(generator.config["sim"]["duration_s"])
        dt_s = float(generator.config["sim"]["dt"])
    else:
        scenario_table = Path(args.scenario_table)
        if not scenario_table.exists():
            raise FileNotFoundError(f"scenario table not found: {scenario_table}")
        instances = read_sim_instances(
            scenario_table,
            count=None if args.samples is None else int(args.samples),
            offset=int(args.offset),
        )
        tasks = list(instances)
        sample_count = len(tasks)
        source = str(scenario_table)
        if tasks:
            first_config = tasks[0].config
            duration_s = float(first_config.options.duration_s)
            dt_s = float(first_config.options.backend_dt)
        else:
            duration_s = math.nan
            dt_s = math.nan

    workers = _resolve_workers(args.workers, sample_count)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    print(f"running {sample_count} scenarios x {len(sims)} sims with {workers} worker(s)", flush=True)
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            rows.extend(_run_task(task, sims))
            _print_progress(index, sample_count, len(sims), int(args.progress_every), start)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_task, task, sims): task
                for task in tasks
            }
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                task = futures[future]
                try:
                    rows.extend(future.result())
                except Exception as exc:  # noqa: BLE001
                    seed = int(task if isinstance(task, int) else task.seed)
                    rows.extend(_seed_error_rows(seed, sims, str(exc)))
                _print_progress(index, sample_count, len(sims), int(args.progress_every), start)

    sim_order = {sim_name: index for index, sim_name in enumerate(sims)}
    rows.sort(key=lambda row: (int(row["seed"]), sim_order.get(str(row["sim"]), len(sim_order))))

    _write_rows(run_dir / "trials.csv", rows)
    summary = {
        "run_dir": str(run_dir),
        "source": source,
        "sims": list(sims),
        "num_scenarios": int(sample_count),
        "seed_start": int(args.seed_start),
        "offset": int(args.offset),
        "workers": int(workers),
        "duration_s": duration_s,
        "dt": dt_s,
        "elapsed_wall_s": time.perf_counter() - start,
        "summary": _summarize(rows),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_workers(requested: int | None, samples: int) -> int:
    if requested is not None:
        return max(1, min(int(samples), int(requested)))
    cpu_count = os.cpu_count() or 1
    return max(1, min(int(samples), max(1, cpu_count - 1)))


def _print_progress(
    completed: int,
    total: int,
    sim_count: int,
    progress_every: int,
    start: float,
) -> None:
    if progress_every <= 0:
        return
    if completed % progress_every != 0 and completed != total:
        return
    elapsed = time.perf_counter() - start
    print(f"completed {completed}/{total} scenarios x {sim_count} sims in {elapsed:.1f}s", flush=True)


def _run_task(task: Any, sims: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(task, int):
        return _run_scenario(int(task), sims)
    return _run_instance(task, sims)


def _run_scenario(seed: int, sims: tuple[str, ...]) -> list[dict[str, Any]]:
    generator = _get_generator()
    instance = generator.sample(seed=seed)
    point = generator._by_seed[int(seed)]
    scenario_fields = _scenario_fields(point)
    return _run_instance(instance, sims, scenario_fields=scenario_fields)


def _run_instance(instance, sims: tuple[str, ...], scenario_fields: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if scenario_fields is None:
        scenario_fields = _scenario_fields_from_instance(instance)
    rows: list[dict[str, Any]] = []
    for sim_name in sims:
        trial_start = time.perf_counter()
        try:
            row = _run_one(sim_name, instance)
            row["error"] = None
        except Exception as exc:  # noqa: BLE001
            row = _error_row(sim_name, instance, str(exc))
        row.update(scenario_fields)
        row["wall_s"] = time.perf_counter() - trial_start
        rows.append(row)
    return rows


def _get_generator() -> RobustInterceptUniformDistanceConfigGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = RobustInterceptUniformDistanceConfigGenerator()
    return _GENERATOR


def _seed_error_rows(seed: int, sims: tuple[str, ...], error: str) -> list[dict[str, Any]]:
    rows = []
    for sim_name in sims:
        row = _error_row_for_seed(sim_name, seed, error)
        row.update({
            "stratum": "unknown",
            "range_m": math.nan,
            "closing_speed_mps": math.nan,
            "wall_s": math.nan,
        })
        rows.append(row)
    return rows


def _run_one(sim_name: str, instance) -> dict[str, Any]:
    if sim_name == "beihang_minimal":
        return _run_minimal(instance)
    if sim_name == "beihang_paper":
        return _run_paper(instance)
    raise ValueError(sim_name)


def _run_minimal(instance) -> dict[str, Any]:
    target = instance.config.targets[0]
    target_initial = instance.target_initials[0]
    camera = instance.config.cameras[0]
    vehicle = MinimalVehicleConfig(
        mass_kg=float(instance.config.pursuer.mass_kg),
        max_thrust_n=float(instance.config.max_thrust_n),
        max_body_rate_rad_s=float(instance.config.max_rate_rps),
        initial_position_w=tuple(float(x) for x in instance.pursuer_initial.position_w),
        initial_velocity_w=tuple(float(x) for x in instance.pursuer_initial.velocity_w),
        initial_quat_xyzw=tuple(float(x) for x in instance.pursuer_initial.quat_xyzw),
    )
    target_cfg = MinimalTargetConfig(
        radius_m=float(target.radius_m),
        initial_position_w=tuple(float(x) for x in target_initial.position_w),
        base_velocity_w=tuple(float(x) for x in target_initial.velocity_w),
        weave_amplitude_m=(0.0, 0.0),
        weave_frequency_hz=(0.0, 0.0),
    )
    camera_cfg = MinimalCameraConfig(
        body_to_camera=tuple(tuple(float(v) for v in row) for row in camera.body_to_camera),
        max_uv_norm=max(
            math.tan(float(camera.intrinsics.hfov_rad) / 2.0),
            math.tan(float(camera.intrinsics.vfov_rad) / 2.0),
        ),
        min_depth_m=0.1,
    )
    config = MinimalTrialConfig(
        dt=float(instance.config.options.backend_dt),
        duration_s=float(instance.config.options.duration_s),
        capture_radius_m=float(instance.config.intercept_radius_m),
        arena_min_w=(-100.0, -100.0, -100.0),
        arena_max_w=(100.0, 100.0, 100.0),
        vehicle=vehicle,
        target=target_cfg,
        camera=camera_cfg,
    )
    metrics, samples = run_minimal_trial(config)
    if metrics is None:
        raise RuntimeError("minimal sim produced no metrics")
    return {
        "sim": "beihang_minimal",
        "seed": int(instance.seed),
        "caught": bool(metrics.captured),
        "catch_time_s": metrics.capture_time_s,
        "min_distance_m": float(metrics.min_distance_m),
        "final_distance_m": float(metrics.distance_m),
        "visible_fraction": _minimal_visible_fraction(samples),
        "control_effort": float(metrics.control_effort),
        "steps": int(len(samples)),
        "crashed": bool(metrics.crashed),
        "out_of_bounds": bool(metrics.out_of_bounds),
    }


def _run_paper(instance) -> dict[str, Any]:
    raw = _paper_raw_config(instance)
    diagram, logger = build_diagram_from_config(
        raw,
        controller_gains=raw["controller"].get("gains"),
        noise_config=NoiseConfig(rng_seed=int(instance.seed)),
    )
    sim = Simulator(diagram)
    sim.set_target_realtime_rate(0.0)
    sim.Initialize()
    duration_s = float(raw["sim"]["duration_s"])
    dt_s = float(raw["sim"]["dt"])
    catch_radius_m = float(raw["metrics"]["catch_radius_m"])
    sim.AdvanceTo(duration_s)
    log = logger.get_log()[: int(math.ceil(duration_s / dt_s))]
    metrics = _compute_paper_metrics(log, catch_radius_m=catch_radius_m)
    return {
        "sim": "beihang_paper",
        "seed": int(instance.seed),
        "caught": metrics.catch_time_s is not None,
        "catch_time_s": metrics.catch_time_s,
        "min_distance_m": float(metrics.min_distance_m),
        "final_distance_m": float(metrics.final_distance_m),
        "visible_fraction": float(metrics.target_visible_fraction),
        "control_effort": _paper_control_effort(log, dt_s),
        "steps": int(len(log)),
        "crashed": False,
        "out_of_bounds": False,
    }


def _paper_raw_config(instance) -> dict[str, Any]:
    target = instance.config.targets[0]
    target_initial = instance.target_initials[0]
    camera = instance.config.cameras[0]
    intr = camera.intrinsics
    return {
        "experiment": {"name": "robust_uniform_distance_beihang_paper"},
        "sim": {
            "backend": "puffer_c",
            "duration_s": float(instance.config.options.duration_s),
            "dt": float(instance.config.options.backend_dt),
        },
        "vehicle": {
            "model": "x500",
            "initial_position_w": _list(instance.pursuer_initial.position_w),
            "initial_velocity_w": _list(instance.pursuer_initial.velocity_w),
            "initial_quat_xyzw": _list(instance.pursuer_initial.quat_xyzw),
            "initial_body_rates_b": _list(instance.pursuer_initial.body_rates_b),
            "wind_w": _list(np.zeros(3) if instance.pursuer_initial.wind_w is None else instance.pursuer_initial.wind_w),
            "initial_pitch_offset_deg": 0.0,
        },
        "target": {
            "id": target.id,
            "kind": target.kind,
            "initial_state": {
                "position_w": _list(target_initial.position_w),
                "velocity_w": _list(target_initial.velocity_w),
            },
            "radius_m": float(target.radius_m),
        },
        "camera": {
            "id": camera.id,
            "parent_id": camera.parent_id,
            "position_b": _list(camera.position_b),
            "body_to_camera": np.asarray(camera.body_to_camera, dtype=float).tolist(),
            "width_px": int(intr.width_px),
            "height_px": int(intr.height_px),
            "fx_px": float(intr.fx_px),
            "fy_px": float(intr.fy_px),
            "cx_px": float(intr.cx_px),
            "cy_px": float(intr.cy_px),
            "hfov_deg": math.degrees(float(intr.hfov_rad)),
            "vfov_deg": math.degrees(float(intr.vfov_rad)),
            "capture_rate_hz": float(camera.capture_rate_hz),
        },
        "perception": {
            "processing_delay_s": float(instance.config.noise.processing_delay_s),
            "pixel_noise_std_px": [0.0, 0.0],
            "dropout_probability": 0.0,
            "rng_seed": int(instance.seed),
        },
        "observer": {"type": "beihang_image_ekf", "history_size": 50},
        "controller": {
            "type": "beihang_paper_stack",
            "max_rate_rps": float(instance.config.max_rate_rps),
            "max_thrust_n": float(instance.config.max_thrust_n),
        },
        "metrics": {"catch_radius_m": float(instance.config.intercept_radius_m)},
    }


def _scenario_fields(point) -> dict[str, Any]:
    return {
        "stratum": str(point.stratum),
        "range_m": float(point.values["range_m"]),
        "closing_speed_mps": float(point.values["closing_speed_mps"]),
    }


def _scenario_fields_from_instance(instance) -> dict[str, Any]:
    target_initial = instance.target_initials[0]
    rel_pos = np.asarray(target_initial.position_w, dtype=float) - np.asarray(instance.pursuer_initial.position_w, dtype=float)
    rel_vel = np.asarray(instance.pursuer_initial.velocity_w, dtype=float) - np.asarray(target_initial.velocity_w, dtype=float)
    range_m = float(np.linalg.norm(rel_pos))
    los_w = rel_pos / max(range_m, 1e-12)
    metadata = getattr(instance, "metadata", {}) or {}
    return {
        "stratum": str(metadata.get("stratum", "pregenerated")),
        "range_m": range_m,
        "closing_speed_mps": float(np.dot(rel_vel, los_w)),
    }


def _error_row(sim_name: str, instance, error: str) -> dict[str, Any]:
    return _error_row_for_seed(sim_name, int(instance.seed), error)


def _error_row_for_seed(sim_name: str, seed: int, error: str) -> dict[str, Any]:
    return {
        "sim": sim_name,
        "seed": int(seed),
        "caught": False,
        "catch_time_s": None,
        "min_distance_m": math.nan,
        "final_distance_m": math.nan,
        "visible_fraction": math.nan,
        "control_effort": math.nan,
        "steps": 0,
        "crashed": False,
        "out_of_bounds": False,
        "error": error,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sim",
        "seed",
        "stratum",
        "range_m",
        "closing_speed_mps",
        "caught",
        "catch_time_s",
        "min_distance_m",
        "final_distance_m",
        "visible_fraction",
        "control_effort",
        "steps",
        "crashed",
        "out_of_bounds",
        "wall_s",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        sim_name: {
            **_summarize_subset([row for row in rows if row["sim"] == sim_name]),
            "by_closing_speed_mps": {
                str(speed): _summarize_subset([
                    row for row in rows
                    if row["sim"] == sim_name and float(row["closing_speed_mps"]) == speed
                ])
                for speed in sorted({float(row["closing_speed_mps"]) for row in rows if row["sim"] == sim_name})
            },
        }
        for sim_name in sorted({row["sim"] for row in rows})
    }


def _summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    min_distance = _finite_array(row["min_distance_m"] for row in valid)
    final_distance = _finite_array(row["final_distance_m"] for row in valid)
    visible = _finite_array(row["visible_fraction"] for row in valid)
    effort = _finite_array(row["control_effort"] for row in valid)
    caught = np.array([bool(row["caught"]) for row in valid], dtype=bool)
    return {
        "n": int(len(rows)),
        "valid": int(len(valid)),
        "errors": int(len(rows) - len(valid)),
        "catch_fraction": float(np.mean(caught)) if caught.size else math.nan,
        "min_distance_p50_m": _percentile(min_distance, 50),
        "min_distance_p90_m": _percentile(min_distance, 90),
        "final_distance_p50_m": _percentile(final_distance, 50),
        "visible_fraction_mean": _mean(visible),
        "control_effort_mean": _mean(effort),
    }


def _minimal_visible_fraction(samples) -> float:
    if not samples:
        return 0.0
    return sum(1 for sample in samples if sample.feature.detected) / len(samples)


def _paper_control_effort(log, dt_s: float) -> float:
    total = 0.0
    for step in log:
        command = step.command
        total += (float(np.linalg.norm(command.body_rates_b)) + 0.02 * abs(float(command.thrust_n))) * float(dt_s)
    return total


def _compute_paper_metrics(log, *, catch_radius_m: float) -> _PaperMetrics:
    if not log:
        return _PaperMetrics(
            catch_time_s=None,
            min_distance_m=math.nan,
            final_distance_m=math.nan,
            target_visible_fraction=math.nan,
        )

    distances: list[float] = []
    visible_count = 0
    catch_time_s: float | None = None
    for step in log:
        target = step.scene.targets[0] if step.scene.targets else None
        if target is None:
            distances.append(math.nan)
            continue
        distance = float(np.linalg.norm(step.scene.pursuer.position_w - target.position_w))
        distances.append(distance)
        if step.capture is not None and step.capture.detected:
            visible_count += 1
        if catch_time_s is None and distance <= float(catch_radius_m):
            catch_time_s = float(step.t)

    finite = _finite_array(distances)
    return _PaperMetrics(
        catch_time_s=catch_time_s,
        min_distance_m=float(np.min(finite)) if finite.size else math.nan,
        final_distance_m=float(distances[-1]) if math.isfinite(float(distances[-1])) else math.nan,
        target_visible_fraction=visible_count / max(len(log), 1),
    )


def _finite_array(values) -> np.ndarray:
    array = np.array(list(values), dtype=float)
    return array[np.isfinite(array)]


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else math.nan


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def _list(value) -> list[float]:
    return [float(x) for x in np.asarray(value, dtype=float).reshape(-1)]


if __name__ == "__main__":
    raise SystemExit(main())
