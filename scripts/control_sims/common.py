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
from control_sims.beihang_minimal_sim.config import (
    CameraConfig as MinimalCameraConfig,
    TargetConfig as MinimalTargetConfig,
    TrialConfig as MinimalTrialConfig,
    VehicleConfig as MinimalVehicleConfig,
)
from control_sims.beihang_minimal_sim.replay import run_trial as run_minimal_trial
from control_sims.beihang_paper_sim._paths import ensure_paths
from control_sims.beihang_paper_sim.noise_config import NoiseConfig
from scripts.generators.robust_intercept_uniform_distance import (
    RobustInterceptUniformDistanceConfigGenerator,
)


ensure_paths()

from pydrake.systems.analysis import Simulator  # noqa: E402

from control_sims.beihang_paper_sim.diagram import build_diagram_from_config  # noqa: E402


SIMS = ("beihang_minimal", "beihang_paper")
_GENERATOR: RobustInterceptUniformDistanceConfigGenerator | None = None


@dataclass(frozen=True)
class PaperMetrics:
    catch_time_s: float | None
    min_distance_m: float
    final_distance_m: float
    target_visible_fraction: float


def run_cli(sim_name: str, description: str) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--scenario-table", type=Path, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of scenario workers. Defaults to min(samples, CPU count - 1). Use 1 for serial.",
    )
    parser.add_argument("--log-snapshots", action="store_true")
    parser.add_argument(
        "--snapshot-log-rate",
        type=int,
        default=100,
        help="Write one snapshot row every N sim ticks when --log-snapshots is enabled.",
    )
    args = parser.parse_args()

    if sim_name not in SIMS:
        raise ValueError(f"unknown control sim {sim_name!r}")
    if int(args.snapshot_log_rate) <= 0:
        raise ValueError("--snapshot-log-rate must be positive")

    run_dir = args.out_dir or Path(".runs") / sim_name / dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks, source, duration_s, dt_s = _load_tasks(args)
    workers = _resolve_workers(args.workers, len(tasks))
    print(f"running {len(tasks)} scenarios with {workers} worker(s) for {sim_name}", flush=True)

    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    start = time.perf_counter()
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            result = _run_task(task, sim_name, bool(args.log_snapshots), int(args.snapshot_log_rate))
            rows.append(result["row"])
            snapshots.extend(result["snapshots"])
            _print_progress(index, len(tasks), int(args.progress_every), start)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_task,
                    task,
                    sim_name,
                    bool(args.log_snapshots),
                    int(args.snapshot_log_rate),
                ): task
                for task in tasks
            }
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                task = futures[future]
                try:
                    result = future.result()
                    rows.append(result["row"])
                    snapshots.extend(result["snapshots"])
                except Exception as exc:  # noqa: BLE001
                    seed = int(task if isinstance(task, int) else task.seed)
                    row = _error_row_for_seed(sim_name, seed, str(exc))
                    row.update(_scenario_fields_for_task(task))
                    row["wall_s"] = math.nan
                    rows.append(row)
                _print_progress(index, len(tasks), int(args.progress_every), start)

    rows.sort(key=lambda row: int(row["seed"]))
    snapshots.sort(key=lambda row: (int(row["seed"]), int(row["tick"])))
    _write_rows(run_dir / "trials.csv", rows)
    snapshot_path = None
    if args.log_snapshots:
        snapshot_path = run_dir / "snapshots" / f"{sim_name}.csv"
        _write_snapshots(snapshot_path, snapshots)
        (snapshot_path.parent / "logging_config.json").write_text(
            json.dumps(
                {
                    "every_n_ticks": int(args.snapshot_log_rate),
                    "output_dir": str(snapshot_path.parent),
                    "sim": sim_name,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    summary = {
        "run_dir": str(run_dir),
        "source": source,
        "sim": sim_name,
        "num_scenarios": int(len(tasks)),
        "seed_start": int(args.seed_start),
        "offset": int(args.offset),
        "workers": int(workers),
        "duration_s": duration_s,
        "dt": dt_s,
        "elapsed_wall_s": time.perf_counter() - start,
        "snapshot_log": {
            "enabled": bool(args.log_snapshots),
            "every_n_ticks": int(args.snapshot_log_rate),
            "path": None if snapshot_path is None else str(snapshot_path),
        },
        "summary": _summarize_subset(rows),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_tasks(args: argparse.Namespace) -> tuple[list[Any], str, float, float]:
    global _GENERATOR
    generator = RobustInterceptUniformDistanceConfigGenerator()
    if args.scenario_table is None:
        sample_count = int(args.samples) if args.samples is not None else 100
        _GENERATOR = generator
        return (
            list(range(int(args.seed_start), int(args.seed_start) + sample_count)),
            "robust_intercept_uniform_distance",
            float(generator.config["sim"]["duration_s"]),
            float(generator.config["sim"]["dt"]),
        )

    scenario_table = Path(args.scenario_table)
    if not scenario_table.exists():
        raise FileNotFoundError(f"scenario table not found: {scenario_table}")
    instances = read_sim_instances(
        scenario_table,
        count=None if args.samples is None else int(args.samples),
        offset=int(args.offset),
    )
    if not instances:
        return [], str(scenario_table), math.nan, math.nan
    first_config = instances[0].config
    return (
        list(instances),
        str(scenario_table),
        float(first_config.options.duration_s),
        float(first_config.options.backend_dt),
    )


def _resolve_workers(requested: int | None, samples: int) -> int:
    if samples <= 0:
        return 1
    if requested is not None:
        return max(1, min(int(samples), int(requested)))
    cpu_count = os.cpu_count() or 1
    return max(1, min(int(samples), max(1, cpu_count - 1)))


def _print_progress(completed: int, total: int, progress_every: int, start: float) -> None:
    if progress_every <= 0:
        return
    if completed % progress_every != 0 and completed != total:
        return
    elapsed = time.perf_counter() - start
    print(f"completed {completed}/{total} scenarios in {elapsed:.1f}s", flush=True)


def _run_task(task: Any, sim_name: str, log_snapshots: bool, snapshot_log_rate: int) -> dict[str, Any]:
    instance, scenario_fields = _materialize_task(task)
    start = time.perf_counter()
    snapshots: list[dict[str, Any]] = []
    try:
        if sim_name == "beihang_minimal":
            row, snapshots = _run_minimal(instance, log_snapshots=log_snapshots, snapshot_log_rate=snapshot_log_rate)
        elif sim_name == "beihang_paper":
            row, snapshots = _run_paper(instance, log_snapshots=log_snapshots, snapshot_log_rate=snapshot_log_rate)
        else:
            raise ValueError(sim_name)
        row["error"] = None
    except Exception as exc:  # noqa: BLE001
        row = _error_row_for_seed(sim_name, int(instance.seed), str(exc))
    row.update(scenario_fields)
    row["wall_s"] = time.perf_counter() - start
    return {"row": row, "snapshots": snapshots}


def _materialize_task(task: Any):
    if isinstance(task, int):
        generator = _get_generator()
        instance = generator.sample(seed=int(task))
        point = generator._by_seed[int(task)]
        return instance, _scenario_fields(point)
    return task, _scenario_fields_from_instance(task)


def _get_generator() -> RobustInterceptUniformDistanceConfigGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = RobustInterceptUniformDistanceConfigGenerator()
    return _GENERATOR


def _run_minimal(instance, *, log_snapshots: bool, snapshot_log_rate: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    row = {
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
    snapshot_rows = _minimal_snapshot_rows(instance.seed, samples, snapshot_log_rate) if log_snapshots else []
    return row, snapshot_rows


def _run_paper(instance, *, log_snapshots: bool, snapshot_log_rate: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    row = {
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
    snapshot_rows = _paper_snapshot_rows(instance.seed, log, snapshot_log_rate) if log_snapshots else []
    return row, snapshot_rows


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


def _minimal_snapshot_rows(seed: int, samples: list[Any], every_n_ticks: int) -> list[dict[str, Any]]:
    rows = []
    for tick, sample in enumerate(samples, start=1):
        if tick % every_n_ticks != 0:
            continue
        row = _base_snapshot_row("beihang_minimal", seed, tick, float(sample.t))
        _add_vector(row, "pursuer", "position_w", sample.vehicle.position_w)
        _add_vector(row, "pursuer", "velocity_w", sample.vehicle.velocity_w)
        _add_vector(row, "target", "position_w", sample.target.position_w)
        _add_vector(row, "target", "velocity_w", sample.target.velocity_w)
        row.update({
            "distance_m": float(sample.metrics.distance_m),
            "min_distance_m": float(sample.metrics.min_distance_m),
            "intercepted": bool(sample.metrics.captured),
            "camera_detected": bool(sample.feature.detected),
            "camera_u_norm": _maybe_index(sample.feature.uv_norm, 0),
            "camera_v_norm": _maybe_index(sample.feature.uv_norm, 1),
            "command_thrust_n": float(sample.command.thrust_n),
            "command_body_rate_x_rad_s": float(sample.command.body_rates_b[0]),
            "command_body_rate_y_rad_s": float(sample.command.body_rates_b[1]),
            "command_body_rate_z_rad_s": float(sample.command.body_rates_b[2]),
        })
        rows.append(row)
    return rows


def _paper_snapshot_rows(seed: int, log: list[Any], every_n_ticks: int) -> list[dict[str, Any]]:
    rows = []
    for tick, step in enumerate(log, start=1):
        if tick % every_n_ticks != 0:
            continue
        row = _base_snapshot_row("beihang_paper", seed, tick, float(step.t))
        state = step.rotorpy_state
        _add_vector(row, "pursuer", "position_w", state.get("x"))
        _add_vector(row, "pursuer", "velocity_w", state.get("v"))
        quat = state.get("q")
        if quat is not None:
            row.update({
                "pursuer_qx": float(quat[0]),
                "pursuer_qy": float(quat[1]),
                "pursuer_qz": float(quat[2]),
                "pursuer_qw": float(quat[3]),
            })
        body_rates = state.get("w")
        if body_rates is not None:
            row.update({
                "pursuer_p_b_rad_s": float(body_rates[0]),
                "pursuer_q_b_rad_s": float(body_rates[1]),
                "pursuer_r_b_rad_s": float(body_rates[2]),
            })
        rotor_speeds = state.get("rotor_speeds")
        if rotor_speeds is not None:
            for i, rpm in enumerate(np.asarray(rotor_speeds, dtype=float).reshape(-1)[:4]):
                row[f"motor_{i}_rpm"] = float(rpm)
        target = step.scene.targets[0] if step.scene.targets else None
        if target is not None:
            _add_vector(row, "target", "position_w", target.position_w)
            _add_vector(row, "target", "velocity_w", target.velocity_w)
            distance = float(np.linalg.norm(step.scene.pursuer.position_w - target.position_w))
            row["distance_m"] = distance
        row.update({
            "intercepted": False,
            "camera_detected": bool(step.capture is not None and step.capture.detected),
            "camera_u_norm": "" if step.capture is None else _maybe_index(step.capture.uv_norm, 0),
            "camera_v_norm": "" if step.capture is None else _maybe_index(step.capture.uv_norm, 1),
            "command_thrust_n": float(step.command.thrust_n),
            "command_body_rate_x_rad_s": float(step.command.body_rates_b[0]),
            "command_body_rate_y_rad_s": float(step.command.body_rates_b[1]),
            "command_body_rate_z_rad_s": float(step.command.body_rates_b[2]),
        })
        rows.append(row)
    return rows


def _base_snapshot_row(sim: str, seed: int, tick: int, t_s: float) -> dict[str, Any]:
    return {"sim": sim, "seed": int(seed), "tick": int(tick), "t_s": float(t_s)}


def _add_vector(row: dict[str, Any], entity: str, name: str, value: Any) -> None:
    if value is None:
        return
    arr = np.asarray(value, dtype=float).reshape(-1)
    suffixes = ("x", "y", "z")
    unit = "m" if "position" in name else "mps"
    for suffix, scalar in zip(suffixes, arr[:3]):
        row[f"{entity}_{suffix}_w_{unit}"] = float(scalar)


def _maybe_index(value: Any, index: int) -> float | str:
    if value is None:
        return ""
    return float(np.asarray(value, dtype=float).reshape(-1)[index])


def _scenario_fields(point) -> dict[str, Any]:
    return {
        "stratum": str(point.stratum),
        "range_m": float(point.values["range_m"]),
        "closing_speed_mps": float(point.values["closing_speed_mps"]),
    }


def _scenario_fields_for_task(task: Any) -> dict[str, Any]:
    if isinstance(task, int):
        return {"stratum": "unknown", "range_m": math.nan, "closing_speed_mps": math.nan}
    return _scenario_fields_from_instance(task)


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


SNAPSHOT_FIELDNAMES = [
    "sim",
    "seed",
    "tick",
    "t_s",
    "pursuer_x_w_m",
    "pursuer_y_w_m",
    "pursuer_z_w_m",
    "pursuer_x_w_mps",
    "pursuer_y_w_mps",
    "pursuer_z_w_mps",
    "pursuer_qx",
    "pursuer_qy",
    "pursuer_qz",
    "pursuer_qw",
    "pursuer_p_b_rad_s",
    "pursuer_q_b_rad_s",
    "pursuer_r_b_rad_s",
    "motor_0_rpm",
    "motor_1_rpm",
    "motor_2_rpm",
    "motor_3_rpm",
    "target_x_w_m",
    "target_y_w_m",
    "target_z_w_m",
    "target_x_w_mps",
    "target_y_w_mps",
    "target_z_w_mps",
    "distance_m",
    "min_distance_m",
    "intercepted",
    "camera_detected",
    "camera_u_norm",
    "camera_v_norm",
    "command_thrust_n",
    "command_body_rate_x_rad_s",
    "command_body_rate_y_rad_s",
    "command_body_rate_z_rad_s",
]


def _write_snapshots(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def _compute_paper_metrics(log, *, catch_radius_m: float) -> PaperMetrics:
    if not log:
        return PaperMetrics(
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
    return PaperMetrics(
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
