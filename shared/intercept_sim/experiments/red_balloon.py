from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from intercept_sim.analysis import circular_error_probable
from intercept_sim.experiments.benchmark import BenchmarkResult
from intercept_sim.experiments.config import ExperimentConfig
from intercept_sim.experiments.runner import ExperimentResult, run_experiment
from intercept_sim.experiments.scenario import ScenarioMetrics
from intercept_sim.experiments.telemetry import ExperimentTelemetry


@dataclass(frozen=True)
class RedBalloonScenario:
    raw: dict[str, Any]
    path: Path | None = None
    distances_m: tuple[float, ...] | None = None
    closing_speeds_mps: tuple[float, ...] | None = None
    seeds: tuple[int, ...] | None = None
    los_azimuths_deg: tuple[float, ...] | None = None
    los_elevations_deg: tuple[float, ...] | None = None

    @property
    def name(self) -> str:
        return str(self.raw["experiment"]["name"])

    def with_sweep(
        self,
        *,
        distances_m: Iterable[float],
        closing_speeds_mps: Iterable[float],
        seeds: Iterable[int],
        los_azimuths_deg: Iterable[float] | None = None,
        los_elevations_deg: Iterable[float] | None = None,
    ) -> RedBalloonScenario:
        return RedBalloonScenario(
            raw=copy.deepcopy(self.raw),
            path=self.path,
            distances_m=tuple(float(value) for value in distances_m),
            closing_speeds_mps=tuple(float(value) for value in closing_speeds_mps),
            seeds=tuple(int(value) for value in seeds),
            los_azimuths_deg=None if los_azimuths_deg is None else tuple(float(value) for value in los_azimuths_deg),
            los_elevations_deg=(
                None if los_elevations_deg is None else tuple(float(value) for value in los_elevations_deg)
            ),
        )

    def build_experiment_configs(self) -> list[ExperimentConfig]:
        distances = self.distances_m or (float(self.raw["scenario"]["distance_m"]),)
        speeds = self.closing_speeds_mps or (float(self.raw["scenario"]["closing_speed_mps"]),)
        seeds = self.seeds or (int(self.raw["scenario"]["seed"]),)
        azimuths = self.los_azimuths_deg or (float(self.raw["scenario"].get("los_azimuth_deg", 0.0)),)
        elevations = self.los_elevations_deg or (float(self.raw["scenario"].get("los_elevation_deg", 0.0)),)
        return [
            build_red_balloon_config(
                self,
                seed=seed,
                distance_m=distance_m,
                closing_speed_mps=speed,
                los_azimuth_deg=azimuth,
                los_elevation_deg=elevation,
            )
            for distance_m in distances
            for speed in speeds
            for seed in seeds
            for azimuth in azimuths
            for elevation in elevations
        ]

    def comment_for_config(self, config: ExperimentConfig) -> str:
        scenario = config.raw["scenario"]
        return _generated_comment(
            config,
            seed=int(scenario["seed"]),
            distance_m=float(scenario["distance_m"]),
            closing_speed_mps=float(scenario["closing_speed_mps"]),
            los_azimuth_deg=float(scenario.get("los_azimuth_deg", 0.0)),
            los_elevation_deg=float(scenario.get("los_elevation_deg", 0.0)),
        )

    def evaluate(self, telemetry: Iterable[ExperimentTelemetry]) -> RedBalloonScenarioMetrics:
        return red_balloon_scenario_metrics_from_telemetry(telemetry)


@dataclass(frozen=True)
class RedBalloonRunMetric:
    experiment: str
    comment: str
    seed: int
    distance_m: float
    closing_speed_mps: float
    balloon_speed_mps: float
    los_azimuth_deg: float
    los_elevation_deg: float
    duration_s: float
    dt: float
    steps: int
    min_distance_m: float
    final_distance_m: float
    catch_time_s: float | None
    target_visible_fraction: float
    image_feature_availability_fraction: float
    average_image_error_norm: float | None
    miss_distance_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RedBalloonAggregateMetric:
    distance_m: float
    closing_speed_mps: float
    runs: int
    cep50_m: float
    cep90_m: float
    min_miss_distance_m: float
    mean_miss_distance_m: float
    catch_fraction: float
    mean_visible_fraction: float
    mean_feature_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RedBalloonScenarioMetrics(ScenarioMetrics):
    run_metrics: list[RedBalloonRunMetric]
    aggregate_metrics: list[RedBalloonAggregateMetric]

    def rows(self) -> list[dict[str, Any]]:
        return [metric.to_dict() for metric in self.run_metrics]

    def aggregate_rows(self) -> list[dict[str, Any]]:
        return [metric.to_dict() for metric in self.aggregate_metrics]


def load_red_balloon_scenario(path: str | Path) -> RedBalloonScenario:
    scenario_path = Path(path)
    with scenario_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    _validate_red_balloon_scenario(raw)
    return RedBalloonScenario(raw=raw, path=scenario_path)


def build_red_balloon_config(
    scenario_or_path: RedBalloonScenario | str | Path,
    *,
    seed: int | None = None,
    distance_m: float | None = None,
    closing_speed_mps: float | None = None,
    los_azimuth_deg: float | None = None,
    los_elevation_deg: float | None = None,
) -> ExperimentConfig:
    scenario = (
        scenario_or_path
        if isinstance(scenario_or_path, RedBalloonScenario)
        else load_red_balloon_scenario(scenario_or_path)
    )
    raw = copy.deepcopy(scenario.raw)
    scenario_raw = raw["scenario"]
    if seed is not None:
        scenario_raw["seed"] = int(seed)
    if distance_m is not None:
        scenario_raw["distance_m"] = float(distance_m)
    if closing_speed_mps is not None:
        scenario_raw["closing_speed_mps"] = float(closing_speed_mps)
    if los_azimuth_deg is not None:
        scenario_raw["los_azimuth_deg"] = float(los_azimuth_deg)
    if los_elevation_deg is not None:
        scenario_raw["los_elevation_deg"] = float(los_elevation_deg)

    rng = np.random.default_rng(int(scenario_raw["seed"]))
    fixed_vehicle_origin_w = (
        _array(scenario_raw["fixed_vehicle_origin_w"], length=3)
        if "fixed_vehicle_origin_w" in scenario_raw
        else None
    )
    fixed_vehicle_velocity_w = (
        _array(scenario_raw["fixed_vehicle_velocity_w"], length=3)
        if "fixed_vehicle_velocity_w" in scenario_raw
        else None
    )
    balloon_position_w = _array(
        scenario_raw.get("balloon_origin_w", scenario_raw.get("balloon_position_w", [0.0, 0.0, 3.0])),
        length=3,
    )
    nominal_los_w = (
        _unit(_array(scenario_raw["los_w"], length=3)) if "los_w" in scenario_raw else _sample_unit_vector(rng)
    )
    los_w = _offset_los(
        nominal_los_w,
        azimuth_deg=float(scenario_raw.get("los_azimuth_deg", 0.0)),
        elevation_deg=float(scenario_raw.get("los_elevation_deg", 0.0)),
    )
    drift_dir_w = (
        _unit(_array(scenario_raw["balloon_drift_dir_w"], length=3))
        if "balloon_drift_dir_w" in scenario_raw
        else (
            _sample_uniform_unit_vector(rng)
            if scenario_raw.get("balloon_velocity_sampling") == "uniform_sphere"
            else _sample_unit_vector(rng)
        )
    )

    distance = float(scenario_raw["distance_m"])
    closing_speed = float(scenario_raw["closing_speed_mps"])
    balloon_speed = float(scenario_raw["balloon_speed_mps"])
    if fixed_vehicle_origin_w is not None:
        copter_position_w = fixed_vehicle_origin_w
        if "balloon_origin_w" not in scenario_raw:
            balloon_position_w = fixed_vehicle_origin_w + distance * los_w
        los_w = _unit(balloon_position_w - copter_position_w)
    balloon_velocity_w = balloon_speed * drift_dir_w
    if fixed_vehicle_origin_w is None:
        copter_position_w = balloon_position_w - distance * los_w
    if fixed_vehicle_velocity_w is not None:
        copter_velocity_w = fixed_vehicle_velocity_w
    elif fixed_vehicle_origin_w is not None:
        copter_velocity_w = closing_speed * los_w
    else:
        copter_velocity_w = balloon_velocity_w + closing_speed * los_w
    copter_quat_xyzw = _quat_body_x_to_world_vector(los_w)

    raw["experiment"]["name"] = _experiment_name(
        scenario.name,
        int(scenario_raw["seed"]),
        distance,
        closing_speed,
        float(scenario_raw.get("los_azimuth_deg", 0.0)),
        float(scenario_raw.get("los_elevation_deg", 0.0)),
    )
    raw["vehicle"]["initial_position_w"] = copter_position_w.tolist()
    raw["vehicle"]["initial_velocity_w"] = copter_velocity_w.tolist()
    raw["vehicle"]["initial_quat_xyzw"] = copter_quat_xyzw.tolist()
    raw["target"]["kind"] = "red_balloon"
    raw["target"]["initial_position_w"] = balloon_position_w.tolist()
    raw["target"]["velocity_w"] = balloon_velocity_w.tolist()
    raw["perception"]["rng_seed"] = int(scenario_raw["seed"])
    return ExperimentConfig(raw=raw, path=scenario.path)


def run_red_balloon_sweep(
    scenario_or_path: RedBalloonScenario | str | Path,
    *,
    distances_m: Iterable[float],
    closing_speeds_mps: Iterable[float],
    seeds: Iterable[int],
    los_azimuths_deg: Iterable[float] | None = None,
    los_elevations_deg: Iterable[float] | None = None,
    comment: str | None = None,
) -> BenchmarkResult:
    scenario = (
        scenario_or_path
        if isinstance(scenario_or_path, RedBalloonScenario)
        else load_red_balloon_scenario(scenario_or_path)
    ).with_sweep(
        distances_m=distances_m,
        closing_speeds_mps=closing_speeds_mps,
        seeds=seeds,
        los_azimuths_deg=los_azimuths_deg,
        los_elevations_deg=los_elevations_deg,
    )
    results = [
        run_experiment(config, comment=comment or scenario.comment_for_config(config))
        for config in scenario.build_experiment_configs()
    ]
    return BenchmarkResult(results=results)


def red_balloon_sweep_rows(result: BenchmarkResult) -> list[dict[str, Any]]:
    return red_balloon_scenario_metrics(result.results).rows()


def red_balloon_aggregate_rows(result: BenchmarkResult) -> list[dict[str, Any]]:
    return red_balloon_scenario_metrics(result.results).aggregate_rows()


def red_balloon_scenario_metrics(results: Iterable[ExperimentResult]) -> RedBalloonScenarioMetrics:
    telemetry = [_telemetry_from_result(result) for result in results]
    return red_balloon_scenario_metrics_from_telemetry(telemetry)


def red_balloon_scenario_metrics_from_telemetry(
    telemetry: Iterable[ExperimentTelemetry],
) -> RedBalloonScenarioMetrics:
    run_metrics = [_run_metric_from_telemetry(item) for item in telemetry]
    groups: dict[tuple[float, float], list[RedBalloonRunMetric]] = {}
    for metric in run_metrics:
        groups.setdefault((metric.distance_m, metric.closing_speed_mps), []).append(metric)

    aggregate_metrics: list[RedBalloonAggregateMetric] = []
    for (distance_m, closing_speed_mps), group in sorted(groups.items()):
        miss_distances = [metric.miss_distance_m for metric in group]
        aggregate_metrics.append(
            RedBalloonAggregateMetric(
                distance_m=distance_m,
                closing_speed_mps=closing_speed_mps,
                runs=len(group),
                cep50_m=circular_error_probable(miss_distances, percentile=50.0),
                cep90_m=circular_error_probable(miss_distances, percentile=90.0),
                min_miss_distance_m=min(miss_distances),
                mean_miss_distance_m=float(np.mean(miss_distances)),
                catch_fraction=sum(metric.catch_time_s is not None for metric in group) / len(group),
                mean_visible_fraction=float(np.mean([metric.target_visible_fraction for metric in group])),
                mean_feature_fraction=float(np.mean([metric.image_feature_availability_fraction for metric in group])),
            )
        )
    return RedBalloonScenarioMetrics(run_metrics=run_metrics, aggregate_metrics=aggregate_metrics)


def save_red_balloon_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _quat_body_x_to_world_vector(x_axis_w: np.ndarray) -> np.ndarray:
    x_axis = _unit(x_axis_w)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(x_axis, world_up))) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
    y_axis = _unit(np.cross(world_up, x_axis))
    z_axis = _unit(np.cross(x_axis, y_axis))
    rotation_wb = np.column_stack((x_axis, y_axis, z_axis))
    return Rotation.from_matrix(rotation_wb).as_quat()


def _sample_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(0.0, 1.0, size=3)
    vector[2] *= 0.35
    return _unit(vector)


def _sample_uniform_unit_vector(rng: np.random.Generator) -> np.ndarray:
    return _unit(rng.normal(0.0, 1.0, size=3))


def _offset_los(nominal_los_w: np.ndarray, *, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    forward = _unit(nominal_los_w)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(forward, world_up))) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
    right = _unit(np.cross(world_up, forward))
    up = _unit(np.cross(forward, right))
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    horizontal = np.cos(azimuth) * forward + np.sin(azimuth) * right
    return _unit(np.cos(elevation) * horizontal + np.sin(elevation) * up)


def _experiment_name(
    base_name: str,
    seed: int,
    distance_m: float,
    closing_speed_mps: float,
    los_azimuth_deg: float | None = None,
    los_elevation_deg: float | None = None,
) -> str:
    distance_label = str(float(distance_m)).replace(".", "p")
    speed_label = str(float(closing_speed_mps)).replace(".", "p")
    name = f"{base_name}_d{distance_label}_v{speed_label}_seed_{seed}"
    if los_azimuth_deg is not None or los_elevation_deg is not None:
        azimuth_label = _angle_label(float(los_azimuth_deg or 0.0))
        elevation_label = _angle_label(float(los_elevation_deg or 0.0))
        name = f"{name}_az{azimuth_label}_el{elevation_label}"
    return name


def _angle_label(value: float) -> str:
    sign = "m" if value < 0.0 else "p"
    return f"{sign}{str(abs(float(value))).replace('.', 'p')}"


def _generated_comment(
    config: ExperimentConfig,
    *,
    seed: int,
    distance_m: float,
    closing_speed_mps: float,
    los_azimuth_deg: float,
    los_elevation_deg: float,
) -> str:
    config_name = config.path.name if config.path is not None else config.name
    observer = config.raw["observer"]["type"]
    controller = config.raw["controller"].get("type", "ibvs")
    return (
        f"red_balloon config={config_name} observer={observer} controller={controller} "
        f"distance_m={distance_m:g} closing_speed_mps={closing_speed_mps:g} "
        f"los_azimuth_deg={los_azimuth_deg:g} los_elevation_deg={los_elevation_deg:g} seed={seed}"
    )


def _telemetry_from_result(result: ExperimentResult) -> ExperimentTelemetry:
    telemetry = result.telemetry
    if telemetry is None:
        telemetry = ExperimentTelemetry(
            experiment_id=result.config.name,
            comment=result.comment,
            config=result.config.raw,
            summary_metrics=result.metrics,
            steps=[],
        )
    return telemetry


def _run_metric_from_telemetry(telemetry: ExperimentTelemetry) -> RedBalloonRunMetric:
    scenario = telemetry.config["scenario"]
    metrics = telemetry.summary_metrics
    return RedBalloonRunMetric(
        experiment=telemetry.experiment_id,
        comment=telemetry.comment,
        seed=int(scenario["seed"]),
        distance_m=float(scenario["distance_m"]),
        closing_speed_mps=float(scenario["closing_speed_mps"]),
        balloon_speed_mps=float(scenario["balloon_speed_mps"]),
        los_azimuth_deg=float(scenario.get("los_azimuth_deg", 0.0)),
        los_elevation_deg=float(scenario.get("los_elevation_deg", 0.0)),
        duration_s=float(telemetry.config["sim"]["duration_s"]),
        dt=float(telemetry.config["sim"]["dt"]),
        steps=len(telemetry.steps),
        min_distance_m=metrics.min_distance_m,
        final_distance_m=metrics.final_distance_m,
        catch_time_s=metrics.catch_time_s,
        target_visible_fraction=metrics.target_visible_fraction,
        image_feature_availability_fraction=metrics.image_feature_availability_fraction,
        average_image_error_norm=metrics.average_image_error_norm,
        miss_distance_m=metrics.miss_distance_m,
    )


def _validate_red_balloon_scenario(raw: dict[str, Any]) -> None:
    required_top = {
        "experiment",
        "sim",
        "scenario",
        "vehicle",
        "target",
        "camera",
        "perception",
        "observer",
        "controller",
        "metrics",
    }
    missing = required_top - set(raw)
    if missing:
        raise ValueError(f"Missing red-balloon config sections: {sorted(missing)}")
    for key in ("seed", "distance_m", "closing_speed_mps", "balloon_speed_mps"):
        if key not in raw["scenario"]:
            raise ValueError(f"Missing scenario.{key}")


def _array(value: Any, *, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,):
        raise ValueError(f"Expected array of shape ({length},), got {array.shape}")
    return array


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm
