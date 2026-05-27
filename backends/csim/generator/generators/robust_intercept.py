from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from backends.csim.bindings.types import (
    PursuerInitialState,
    SimInstance,
    TargetInitialState,
)
from backends.csim.generator.generator import SimGenerator, get_config
from backends.csim.generator.instance_store import write_sim_instances


DEFAULT_OUTPUT_DIR = Path(".runs/csim_generator_sampling")
SAMPLE_BINARY_NAME_TEMPLATE = "{strategy}_samples.csimin"


DEFAULT_ROBUST_INTERCEPT_CONFIG: dict[str, Any] = {
    "sampling": {
        "strategy": "sobol",
        "seed": 1,
        "n_samples": 7200,
        "scramble": True,
        "active_parameters": [
            "camera_azimuth_deg",
            "camera_elevation_deg",
            "camera_u_fraction",
            "camera_v_fraction",
        ],
    },
    "scenario": {
        "target_origin_w": [0.0, 0.0, 3.0],
        "target_radius_m": 0.2,
        "intercept_radius_m": 0.5,
    },
    "sim": {
        "backend": "puffer_c",
        "duration_s": 3.0,
        "dt": 0.005,
    },
    "controller": {
        "type": "robust_intercept_reference",
        "max_rate_rps": 8.0,
        "max_thrust_n": 40.0,
    },
    "perception": {
        "processing_delay_s": 0.0,
        "pixel_noise_std_px": [0.0, 0.0],
        "dropout_probability": 0.0,
        "rng_seed": 1,
    },
    "camera": {
        "id": "front",
        "parent_id": "interceptor",
        "position_b": [0.0, 0.0, 0.0],
        "body_to_camera": np.eye(3).tolist(),
        "width_px": 1920,
        "height_px": 1080,
        "hfov_deg": 90.0,
        "vfov_deg": 60.0,
        "capture_rate_hz": 30.0,
    },
    "parameters": {
        "range_m": {"min": 8.0, "max": 8.0, "distribution": "uniform"},
        "camera_azimuth_deg": {"min": 0.0, "max": 360.0, "distribution": "uniform"},
        "camera_elevation_deg": {"min": -90.0, "max": 90.0, "distribution": "uniform_sin"},
        "camera_u_fraction": {"min": -0.9, "max": 0.9, "distribution": "uniform"},
        "camera_v_fraction": {"min": -0.9, "max": 0.9, "distribution": "uniform"},
        "camera_roll_deg": {"min": 0.0, "max": 0.0, "distribution": "uniform"},
        "forward_speed_mps": {"min": 2.0, "max": 2.0, "distribution": "uniform"},
        "target_speed_mps": {"min": 0.0, "max": 0.0, "distribution": "uniform"},
        "target_azimuth_rad": {"min": 0.0, "max": 0.0, "distribution": "uniform"},
        "target_elevation_deg": {"min": 0.0, "max": 0.0, "distribution": "uniform_sin"},
        "body_rate_x_radps": {"min": 0.0, "max": 0.0, "distribution": "uniform"},
        "body_rate_y_radps": {"min": 0.0, "max": 0.0, "distribution": "uniform"},
        "body_rate_z_radps": {"min": 0.0, "max": 0.0, "distribution": "uniform"},
        "wind_horizontal_speed_mps": {"min": 0.0, "max": 0.0, "distribution": "uniform"},
        "wind_direction_rad": {"min": 0.0, "max": 0.0, "distribution": "uniform"},
        "wind_vertical_mps": {"min": 0.0, "max": 0.0, "distribution": "uniform"},
    },
    "grid": {
        "range_m": [5.0, 8.0, 20.0],
        "forward_speed_mps": [0.5, 2.0, 8.0],
    },
}


@dataclass(frozen=True)
class _SamplePoint:
    index: int
    seed: int
    stratum: str
    values: dict[str, float]


@dataclass(frozen=True)
class SampleEvaluation:
    instance: SimInstance
    record: dict[str, Any]
    labels: dict[str, bool]
    label_details: dict[str, str]


class RobustInterceptConfigGenerator(SimGenerator):
    """Generate robust interception initial conditions from strategy-agnostic distributions.

    The sampler backend produces unit-cube samples. Parameter specs transform
    those unit values to physical variables, and the resolver maps the variables
    to concrete `SimInstance` initial conditions.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = _deep_merge(DEFAULT_ROBUST_INTERCEPT_CONFIG, config or {})
        self.parameters = tuple(self.config["parameters"].keys())
        self._sample_points = _build_sample_points(self.config, self.parameters)
        self._by_seed = {point.seed: point for point in self._sample_points}

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return copy.deepcopy(DEFAULT_ROBUST_INTERCEPT_CONFIG)

    def sample_many(self, *, count: int, seed_start: int = 1, **kwargs: Any) -> list[SimInstance]:
        if kwargs:
            raise TypeError(f"{type(self).__name__}.sample_many does not accept kwargs")
        count = int(count)
        seed_start = int(seed_start)
        return [self.sample(seed=seed) for seed in range(seed_start, seed_start + count)]

    def _sample_once(self, *, seed: int, **kwargs: Any) -> SimInstance:
        if kwargs:
            raise TypeError(f"{type(self).__name__}.sample does not accept kwargs")
        point = self._by_seed.get(int(seed))
        if point is None:
            raise KeyError(f"No robust-intercept sample for seed {seed}")
        return _resolve_instance(self.config, point)


def write_default_config(path: str | Path) -> None:
    config = copy.deepcopy(DEFAULT_ROBUST_INTERCEPT_CONFIG)
    Path(path).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_instances(config: dict[str, Any]) -> list[SimInstance]:
    return [evaluation.instance for evaluation in evaluate_samples(config)]


def generate_sample_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [evaluation.record for evaluation in evaluate_samples(config)]


def evaluate_samples(config: dict[str, Any]) -> list[SampleEvaluation]:
    return list(iter_sample_evaluations(config))


def iter_sample_evaluations(config: dict[str, Any]):
    generator = RobustInterceptConfigGenerator(config)
    for point in generator._sample_points:
        instance = _resolve_instance(generator.config, point)
        labels, label_details = _label_instance(instance)
        record = _sample_record(generator.config, point, labels=labels, label_details=label_details)
        yield SampleEvaluation(
            instance=instance,
            record=record,
            labels=labels,
            label_details=label_details,
        )


def plot_sample_records(records_by_strategy: dict[str, list[dict[str, Any]]], out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for strategy, rows in records_by_strategy.items():
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        fig.suptitle(f"Robust intercept samples: {strategy}")
        _scatter(
            axes[0, 0],
            rows,
            "los_azimuth_deg",
            "los_elevation_deg",
            "LOS azimuth deg",
            "LOS elevation deg",
            "LOS sphere coverage",
        )
        _scatter(
            axes[0, 1],
            rows,
            "camera_u_fraction",
            "camera_v_fraction",
            "camera u / FOV",
            "camera v / FOV",
            "Initial image bearing",
        )
        _scatter(
            axes[0, 2],
            rows,
            "los_closing_speed_mps",
            "los_lateral_speed_mps",
            "closing speed m/s",
            "lateral speed m/s",
            "Derived relative velocity",
        )
        _scatter(
            axes[1, 0],
            rows,
            "range_m",
            "forward_speed_mps",
            "range m",
            "forward speed m/s",
            "Range vs forward speed",
        )
        _scatter(
            axes[1, 1],
            rows,
            "wind_horizontal_speed_mps",
            "wind_vertical_mps",
            "wind horizontal m/s",
            "wind vertical m/s",
            "Wind",
        )
        axes[1, 2].hist([row["stratum"] for row in rows], bins=len(set(row["stratum"] for row in rows)))
        axes[1, 2].set_title("Stratum counts")
        axes[1, 2].tick_params(axis="x", rotation=25)
        axes[1, 2].grid(True, alpha=0.25)
        png = out_path / f"{strategy}_sampling.png"
        fig.savefig(png, dpi=160)
        plt.close(fig)
        written.append(png)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, (strategy, rows) in zip(axes, records_by_strategy.items()):
        _scatter(
            ax,
            rows,
            "camera_u_fraction",
            "camera_v_fraction",
            "camera u / FOV",
            "camera v / FOV",
            strategy,
        )
    png = out_path / "strategy_camera_bearing_comparison.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)
    written.append(png)
    return written


def _build_sample_points(config: dict[str, Any], parameters: tuple[str, ...]) -> list[_SamplePoint]:
    sampling = config["sampling"]
    seed_start = int(sampling.get("seed", 1))
    n_samples = int(sampling["n_samples"])
    strategy = str(sampling.get("strategy", "sobol")).lower()
    grid = config.get("grid") or {}
    if grid:
        return _build_grid_sample_points(config, parameters, grid, n_samples, seed_start, strategy)

    strata = config.get("strata") or {"default": {"weight": 1.0, "parameters": {}}}
    quotas = _stratum_quotas(strata, n_samples)
    points: list[_SamplePoint] = []
    cursor = 0
    for stratum_name, count in quotas:
        if count <= 0:
            continue
        stratum = strata[stratum_name]
        active_parameters = tuple(stratum.get("active_parameters", parameters))
        active_indexes = {name: index for index, name in enumerate(active_parameters)}
        cube = _unit_cube(strategy, count, len(active_parameters), seed_start + cursor, bool(sampling.get("scramble", True)))
        for local_index, unit_values in enumerate(cube):
            specs = _merged_parameter_specs(config["parameters"], stratum.get("parameters", {}))
            values = {
                name: _transform_parameter(
                    float(unit_values[active_indexes[name]]) if name in active_indexes else 0.5,
                    specs[name],
                )
                for name in parameters
            }
            edge = stratum.get("edge_fov")
            if edge:
                if "camera_u_fraction" not in active_indexes or "camera_v_fraction" not in active_indexes:
                    raise ValueError("edge_fov strata require active camera_u_fraction and camera_v_fraction")
                values["camera_u_fraction"], values["camera_v_fraction"] = _edge_fov_bearing(
                    float(unit_values[active_indexes["camera_u_fraction"]]),
                    float(unit_values[active_indexes["camera_v_fraction"]]),
                    float(edge.get("min_radius_fraction", 0.70)),
                    float(edge.get("max_radius_fraction", 0.95)),
                )
            points.append(
                _SamplePoint(
                    index=cursor + local_index,
                    seed=seed_start + cursor + local_index,
                    stratum=str(stratum_name),
                    values=values,
                )
            )
        cursor += count
    return points


def _build_grid_sample_points(
    config: dict[str, Any],
    parameters: tuple[str, ...],
    grid: dict[str, list[float]],
    n_samples: int,
    seed_start: int,
    strategy: str,
) -> list[_SamplePoint]:
    sampling = config["sampling"]
    active_parameters = tuple(sampling.get("active_parameters", parameters))
    active_indexes = {name: index for index, name in enumerate(active_parameters)}
    cells = _grid_cells(grid)
    quotas = _stratum_quotas({name: {"weight": 1.0} for name, _values in cells}, n_samples)
    specs = _merged_parameter_specs(config["parameters"], {})
    points: list[_SamplePoint] = []
    cursor = 0
    for (cell_name, grid_values), (_quota_name, count) in zip(cells, quotas):
        if count <= 0:
            continue
        cube = _unit_cube(strategy, count, len(active_parameters), seed_start + cursor, bool(sampling.get("scramble", True)))
        for local_index, unit_values in enumerate(cube):
            values = {
                name: _transform_parameter(
                    float(unit_values[active_indexes[name]]) if name in active_indexes else 0.5,
                    specs[name],
                )
                for name in parameters
            }
            for name, value in grid_values.items():
                values[name] = float(value)
            points.append(
                _SamplePoint(
                    index=cursor + local_index,
                    seed=seed_start + cursor + local_index,
                    stratum=cell_name,
                    values=values,
                )
            )
        cursor += count
    return points


def _resolve_instance(config: dict[str, Any], point: _SamplePoint) -> SimInstance:
    values = point.values
    camera_cfg = config["camera"]
    scenario = config["scenario"]

    body_to_camera = np.asarray(camera_cfg.get("body_to_camera", np.eye(3)), dtype=float).reshape(3, 3)
    target_dir_c = _camera_ray_from_fov_fraction(camera_cfg, values)
    if "camera_azimuth_deg" in values and "camera_elevation_deg" in values:
        camera_forward_w = _unit(_spherical_deg(values["camera_azimuth_deg"], values["camera_elevation_deg"]))
        rotation_wc = _camera_rotation_from_forward(camera_forward_w)
        los_w = _unit(rotation_wc @ target_dir_c)
        rotation_wb = Rotation.from_matrix(rotation_wc @ body_to_camera)
    else:
        los_w = _unit(_spherical_deg(values["los_azimuth_deg"], values["los_elevation_deg"]))
        target_dir_b = _unit(body_to_camera.T @ target_dir_c)
        rotation_wb = Rotation.align_vectors([los_w], [target_dir_b])[0]
        if abs(values["camera_roll_deg"]) > 1e-12:
            rotation_wb = Rotation.from_rotvec(math.radians(values["camera_roll_deg"]) * los_w) * rotation_wb
        rotation_wc = rotation_wb.as_matrix() @ body_to_camera.T
        camera_forward_w = _unit(rotation_wc @ np.array([1.0, 0.0, 0.0], dtype=float))

    target_position_w = _array(scenario["target_origin_w"], length=3)
    pursuer_position_w = target_position_w - float(values["range_m"]) * los_w

    relative_velocity_w = values["forward_speed_mps"] * camera_forward_w
    target_dir_w = _spherical_rad(values["target_azimuth_rad"], math.radians(values["target_elevation_deg"]))
    target_velocity_w = values["target_speed_mps"] * target_dir_w
    pursuer_velocity_w = target_velocity_w + relative_velocity_w

    wind_w = np.array([
        values["wind_horizontal_speed_mps"] * math.cos(values["wind_direction_rad"]),
        values["wind_horizontal_speed_mps"] * math.sin(values["wind_direction_rad"]),
        values["wind_vertical_mps"],
    ])
    body_rates_b = np.array([
        values["body_rate_x_radps"],
        values["body_rate_y_radps"],
        values["body_rate_z_radps"],
    ])

    return SimInstance(
        seed=point.seed,
        pursuer_initial=PursuerInitialState(
            position_w=pursuer_position_w,
            velocity_w=pursuer_velocity_w,
            quat_xyzw=rotation_wb.as_quat(),
            body_rates_b=body_rates_b,
            wind_w=wind_w,
        ),
        target_initials=(
            TargetInitialState(position_w=target_position_w, velocity_w=target_velocity_w),
        ),
        config=get_config("base"),
    )


def _sample_record(
    config: dict[str, Any],
    point: _SamplePoint,
    *,
    labels: dict[str, bool] | None = None,
    label_details: dict[str, str] | None = None,
) -> dict[str, Any]:
    derived_values = _derived_record_values(config, point.values)
    return {
        "scenario": "robust_intercept",
        "strategy": str(config["sampling"].get("strategy", "sobol")).lower(),
        "stratum": point.stratum,
        "sample_index": point.index,
        "seed": point.seed,
        "labels": {} if labels is None else dict(labels),
        "label_details": {} if label_details is None else dict(label_details),
        **{name: float(value) for name, value in point.values.items()},
        **derived_values,
    }


def _label_instance(instance: SimInstance) -> tuple[dict[str, bool], dict[str, str]]:
    from backends.csim.generator.validations import (
        validate_kinematic_intercept,
        validate_no_straight_path_capture,
        validate_target_in_fov,
    )

    labels: dict[str, bool] = {}
    details: dict[str, str] = {}

    labels["target_in_fov"] = _passes_label(validate_target_in_fov, instance, details, "target_in_fov")
    labels["straight_path_capture"] = not _passes_label(
        validate_no_straight_path_capture,
        instance,
        details,
        "straight_path_capture",
    )
    labels["kinematic_intercept_feasible"] = _passes_label(
        validate_kinematic_intercept,
        instance,
        details,
        "kinematic_intercept_feasible",
    )
    return labels, details


def _passes_label(validate: Any, instance: SimInstance, details: dict[str, str], name: str) -> bool:
    try:
        validate(instance)
    except ValueError as exc:
        details[name] = str(exc)
        return False
    return True


def _grid_cells(grid: dict[str, list[float]]) -> list[tuple[str, dict[str, float]]]:
    names = list(grid.keys())
    cells: list[tuple[str, dict[str, float]]] = []

    def visit(index: int, values: dict[str, float]) -> None:
        if index == len(names):
            label = "_".join(f"{name}_{_format_grid_value(values[name])}" for name in names)
            cells.append((label, dict(values)))
            return
        name = names[index]
        for value in grid[name]:
            values[name] = float(value)
            visit(index + 1, values)

    visit(0, {})
    return cells


def _format_grid_value(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "neg").replace(".", "p")


def _derived_record_values(config: dict[str, Any], values: dict[str, float]) -> dict[str, float]:
    if "camera_azimuth_deg" not in values or "camera_elevation_deg" not in values:
        return {}
    camera_cfg = config["camera"]
    target_dir_c = _camera_ray_from_fov_fraction(camera_cfg, values)
    camera_forward_w = _unit(_spherical_deg(values["camera_azimuth_deg"], values["camera_elevation_deg"]))
    rotation_wc = _camera_rotation_from_forward(camera_forward_w)
    los_w = _unit(rotation_wc @ target_dir_c)
    los_azimuth_deg, los_elevation_deg = _azimuth_elevation_deg(los_w)
    relative_velocity_w = float(values["forward_speed_mps"]) * camera_forward_w
    los_closing_speed = float(np.dot(relative_velocity_w, los_w))
    los_lateral_velocity_w = relative_velocity_w - los_closing_speed * los_w
    los_lateral_speed = float(np.linalg.norm(los_lateral_velocity_w))
    return {
        "los_azimuth_deg": los_azimuth_deg,
        "los_elevation_deg": los_elevation_deg,
        "los_closing_speed_mps": los_closing_speed,
        "los_lateral_speed_mps": los_lateral_speed,
        "trajectory_angle_deg": math.degrees(
            math.atan2(los_lateral_speed, max(los_closing_speed, 1.0e-12))
        ),
    }


def _camera_ray_from_fov_fraction(camera: dict[str, Any], values: dict[str, float]) -> np.ndarray:
    h_limit = math.tan(math.radians(float(camera["hfov_deg"])) / 2.0)
    v_limit = math.tan(math.radians(float(camera["vfov_deg"])) / 2.0)
    return _unit(np.array([
        1.0,
        float(values["camera_u_fraction"]) * h_limit,
        float(values["camera_v_fraction"]) * v_limit,
    ]))


def _camera_rotation_from_forward(forward_w: np.ndarray) -> np.ndarray:
    x_w = _unit(forward_w)
    up_w = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(x_w, up_w))) > 0.98:
        up_w = np.array([0.0, 1.0, 0.0], dtype=float)
    y_w = _unit(np.cross(up_w, x_w))
    z_w = _unit(np.cross(x_w, y_w))
    return np.column_stack((x_w, y_w, z_w))


def _unit_cube(strategy: str, count: int, dim: int, seed: int, scramble: bool) -> np.ndarray:
    if strategy == "uniform":
        return np.random.default_rng(seed).random((count, dim))
    if strategy == "latin":
        return qmc.LatinHypercube(d=dim, seed=seed).random(count)
    if strategy == "sobol":
        return qmc.Sobol(d=dim, scramble=scramble, seed=seed).random(count)
    raise ValueError(f"Unknown sampling strategy {strategy!r}; expected sobol, latin, or uniform")


def _transform_parameter(unit_value: float, spec: dict[str, Any]) -> float:
    lo = float(spec["min"])
    hi = float(spec["max"])
    u = min(max(float(unit_value), 0.0), np.nextafter(1.0, 0.0))
    distribution = str(spec.get("distribution", "uniform"))
    if distribution == "uniform":
        return lo + u * (hi - lo)
    if distribution == "log_uniform":
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError("log_uniform requires positive min and max")
        return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    if distribution == "uniform_sin":
        sin_lo = math.sin(math.radians(lo))
        sin_hi = math.sin(math.radians(hi))
        return math.degrees(math.asin(sin_lo + u * (sin_hi - sin_lo)))
    raise ValueError(f"Unsupported distribution {distribution!r}")


def _stratum_quotas(strata: dict[str, Any], n_samples: int) -> list[tuple[str, int]]:
    names = list(strata.keys())
    weights = np.array([float(strata[name].get("weight", 1.0)) for name in names], dtype=float)
    weights = weights / np.sum(weights)
    exact = weights * int(n_samples)
    quotas = np.floor(exact).astype(int)
    remainder = int(n_samples) - int(np.sum(quotas))
    for idx in np.argsort(-(exact - quotas))[:remainder]:
        quotas[idx] += 1
    return [(name, int(count)) for name, count in zip(names, quotas)]


def _merged_parameter_specs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    specs = copy.deepcopy(base)
    for name, values in override.items():
        specs[name] = {**specs[name], **values}
    return specs


def _edge_fov_bearing(u_radius: float, u_angle: float, r_min: float, r_max: float) -> tuple[float, float]:
    radius = r_min + float(u_radius) * (r_max - r_min)
    angle = 2.0 * math.pi * float(u_angle)
    return radius * math.cos(angle), radius * math.sin(angle)


def _scatter(ax: Any, rows: list[dict[str, Any]], x_key: str, y_key: str, x_label: str, y_label: str, title: str) -> None:
    strata = sorted(set(str(row["stratum"]) for row in rows))
    for stratum in strata:
        straight_path_subset = [
            row for row in rows if row["stratum"] == stratum and row.get("labels", {}).get("straight_path_capture", False)
        ]
        ordinary_subset = [
            row for row in rows if row["stratum"] == stratum and not row.get("labels", {}).get("straight_path_capture", False)
        ]
        if ordinary_subset:
            ax.scatter(
                [row[x_key] for row in ordinary_subset],
                [row[y_key] for row in ordinary_subset],
                s=8,
                alpha=0.6,
                label=stratum,
            )
        if straight_path_subset:
            ax.scatter(
                [row[x_key] for row in straight_path_subset],
                [row[y_key] for row in straight_path_subset],
                s=18,
                alpha=0.85,
                marker="x",
                c="#dc2626",
                label=f"{stratum} straight path",
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if len(strata) <= 6:
        ax.legend(markerscale=2, fontsize=7)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _spherical_deg(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    return _spherical_rad(math.radians(azimuth_deg), math.radians(elevation_deg))


def _spherical_rad(azimuth_rad: float, elevation_rad: float) -> np.ndarray:
    return np.array([
        math.cos(elevation_rad) * math.cos(azimuth_rad),
        math.cos(elevation_rad) * math.sin(azimuth_rad),
        math.sin(elevation_rad),
    ], dtype=float)


def _azimuth_elevation_deg(direction: np.ndarray) -> tuple[float, float]:
    unit = _unit(direction)
    azimuth = math.degrees(math.atan2(float(unit[1]), float(unit[0])))
    if azimuth < 0.0:
        azimuth += 360.0
    elevation = math.degrees(math.asin(np.clip(float(unit[2]), -1.0, 1.0)))
    return azimuth, elevation


def _orthonormal_perpendicular_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(_unit(direction), axis))) > 0.95:
        axis = np.array([0.0, 1.0, 0.0], dtype=float)
    first = _unit(np.cross(axis, direction))
    second = _unit(np.cross(direction, first))
    return first, second


def _array(value: Any, *, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"Expected array of shape ({length},), got {arr.shape}")
    return arr.copy()


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize zero vector")
    return np.asarray(value, dtype=float) / norm


def _load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_ROBUST_INTERCEPT_CONFIG)
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    return _deep_merge(DEFAULT_ROBUST_INTERCEPT_CONFIG, loaded)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate robust-intercept samples and strategy visualizations.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--strategies", default="sobol,latin,uniform")
    parser.add_argument("--write-default-config", type=Path, default=None)
    parser.add_argument("--skip-records", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    if args.write_default_config is not None:
        write_default_config(args.write_default_config)
        return

    base_config = _load_config(args.config)
    if args.n_samples is not None:
        base_config["sampling"]["n_samples"] = int(args.n_samples)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records_by_strategy: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {"out_dir": str(out_dir), "strategies": {}}
    for strategy in [item.strip() for item in args.strategies.split(",") if item.strip()]:
        config = _deep_merge(base_config, {"sampling": {"strategy": strategy}})
        records: list[dict[str, Any]] = []
        total_count = 0
        label_counts: dict[str, int] = {}
        should_label = not args.skip_records or not args.skip_plots

        def labeled_instances():
            nonlocal total_count, label_counts, records
            if should_label:
                for evaluation in iter_sample_evaluations(config):
                    total_count += 1
                    for name, value in evaluation.labels.items():
                        if value:
                            label_counts[name] = label_counts.get(name, 0) + 1
                    records.append(evaluation.record)
                    yield evaluation.instance
                return

            generator = RobustInterceptConfigGenerator(config)
            for point in generator._sample_points:
                total_count += 1
                yield _resolve_instance(generator.config, point)

        samples_path = out_dir / SAMPLE_BINARY_NAME_TEMPLATE.format(strategy=strategy)
        write_sim_instances(samples_path, labeled_instances())
        if not args.skip_records:
            (out_dir / f"{strategy}_sample_records.json").write_text(
                json.dumps(records, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if not args.skip_plots:
            records_by_strategy[strategy] = records
        summary["strategies"][strategy] = {
            "total": total_count,
            "written": total_count,
            "samples": str(samples_path),
        }
        if should_label:
            summary["strategies"][strategy]["labels"] = label_counts
        if not args.skip_records:
            summary["strategies"][strategy]["records"] = str(out_dir / f"{strategy}_sample_records.json")

    if not args.skip_plots:
        written = plot_sample_records(records_by_strategy, out_dir)
        summary["pngs"] = [str(path) for path in written]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
