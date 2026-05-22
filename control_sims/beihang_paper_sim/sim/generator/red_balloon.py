from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from backends import InitialState, SimGenerator, SimInstance, TargetInitialState


RED_BALLOON_X500_CONFIG: dict[str, Any] = {
    "experiment": {"name": "red_balloon_beihang_ekf"},
    "sim": {
        "backend": "puffer_c",
        "duration_s": 3.0,
        "dt": 0.005,
    },
    "scenario": {
        "seed": 1,
        "distance_m": 8.0,
        "closing_speed_mps": 1.0,
        "balloon_speed_mps": 0.5,
        "balloon_position_w": [0.0, 0.0, 3.0],
        "los_w": [1.0, 0.0, 0.0],
    },
    "vehicle": {
        "model": "x500",
        "initial_position_w": [0.0, 0.0, 0.0],
    },
    "target": {
        "id": "red_balloon",
        "kind": "red_balloon",
        "initial_position_w": [0.0, 0.0, 0.0],
        "velocity_w": [0.0, 0.0, 0.0],
        "radius_m": 0.2,
    },
    "camera": {
        "id": "front",
        "parent_id": "interceptor",
        "position_b": [0.0, 0.0, 0.0],
        "width_px": 1920,
        "height_px": 1080,
        "fx_px": 800.0,
        "fy_px": 800.0,
        "hfov_deg": 90.0,
        "vfov_deg": 60.0,
        "capture_rate_hz": 30.0,
    },
    "perception": {
        "processing_delay_s": 0.08,
        "pixel_noise_std_px": [0.0, 0.0],
        "dropout_probability": 0.0,
        "rng_seed": 1,
    },
    "observer": {
        "type": "beihang_image_ekf",
        "history_size": 50,
    },
    "controller": {
        "type": "beihang_backstepping",
        "max_rate_rps": 8.0,
        "max_thrust_n": 40.0,
    },
    "metrics": {
        "catch_radius_m": 0.5,
    },
}


class RedBalloonConfigGenerator(SimGenerator):
    def __init__(self, base_config: dict[str, Any] | None = None):
        self._base_config = copy.deepcopy(base_config or RED_BALLOON_X500_CONFIG)

    def _sample_once(
        self,
        *,
        seed: int,
        distance_m: float | None = None,
        closing_speed_mps: float | None = None,
        los_azimuth_deg: float | None = None,
        los_elevation_deg: float | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> SimInstance:
        raw = copy.deepcopy(self._base_config)
        scenario = raw["scenario"]
        scenario["seed"] = int(seed)
        if distance_m is not None:
            scenario["distance_m"] = float(distance_m)
        if closing_speed_mps is not None:
            scenario["closing_speed_mps"] = float(closing_speed_mps)
        if los_azimuth_deg is not None:
            scenario["los_azimuth_deg"] = float(los_azimuth_deg)
        if los_elevation_deg is not None:
            scenario["los_elevation_deg"] = float(los_elevation_deg)
        _deep_update(raw, overrides or {})
        self._resolve_red_balloon_geometry(raw)
        return SimInstance(
            seed=int(seed),
            pursuer_initial=_pursuer_initial(raw["vehicle"]),
            target_initial=_target_initial(raw["target"]),
            raw_config=raw,
            metadata={
                "scenario": "red_balloon",
                "distance_m": float(raw["scenario"]["distance_m"]),
                "closing_speed_mps": float(raw["scenario"]["closing_speed_mps"]),
                "los_azimuth_deg": float(raw["scenario"].get("los_azimuth_deg", 0.0)),
                "los_elevation_deg": float(raw["scenario"].get("los_elevation_deg", 0.0)),
            },
        )

    def _resolve_red_balloon_geometry(self, raw: dict[str, Any]) -> None:
        scenario = raw["scenario"]
        rng = np.random.default_rng(int(scenario["seed"]))
        fixed_vehicle_origin_w = (
            _array(scenario["fixed_vehicle_origin_w"], length=3)
            if "fixed_vehicle_origin_w" in scenario
            else None
        )
        fixed_vehicle_velocity_w = (
            _array(scenario["fixed_vehicle_velocity_w"], length=3)
            if "fixed_vehicle_velocity_w" in scenario
            else None
        )
        balloon_position_w = _array(
            scenario.get("balloon_origin_w", scenario.get("balloon_position_w", [0.0, 0.0, 3.0])),
            length=3,
        )
        nominal_los_w = (
            _unit(_array(scenario["los_w"], length=3))
            if "los_w" in scenario
            else _sample_unit_vector(rng)
        )
        los_w = _offset_los(
            nominal_los_w,
            azimuth_deg=float(scenario.get("los_azimuth_deg", 0.0)),
            elevation_deg=float(scenario.get("los_elevation_deg", 0.0)),
        )
        drift_dir_w = (
            _unit(_array(scenario["balloon_drift_dir_w"], length=3))
            if "balloon_drift_dir_w" in scenario
            else (
                _sample_uniform_unit_vector(rng)
                if scenario.get("balloon_velocity_sampling") == "uniform_sphere"
                else _sample_unit_vector(rng)
            )
        )

        distance = float(scenario["distance_m"])
        closing_speed = float(scenario["closing_speed_mps"])
        balloon_velocity_w = float(scenario["balloon_speed_mps"]) * drift_dir_w
        if fixed_vehicle_origin_w is not None:
            pursuer_position_w = fixed_vehicle_origin_w
            if "balloon_origin_w" not in scenario:
                balloon_position_w = fixed_vehicle_origin_w + distance * los_w
            los_w = _unit(balloon_position_w - pursuer_position_w)
        else:
            pursuer_position_w = balloon_position_w - distance * los_w

        if fixed_vehicle_velocity_w is not None:
            pursuer_velocity_w = fixed_vehicle_velocity_w
        elif fixed_vehicle_origin_w is not None:
            pursuer_velocity_w = closing_speed * los_w
        else:
            pursuer_velocity_w = balloon_velocity_w + closing_speed * los_w

        raw["experiment"]["name"] = _experiment_name(
            str(self._base_config["experiment"]["name"]),
            int(scenario["seed"]),
            distance,
            closing_speed,
            float(scenario.get("los_azimuth_deg", 0.0)),
            float(scenario.get("los_elevation_deg", 0.0)),
        )
        raw["vehicle"]["initial_position_w"] = pursuer_position_w.tolist()
        raw["vehicle"]["initial_velocity_w"] = pursuer_velocity_w.tolist()
        raw["vehicle"]["initial_quat_xyzw"] = _quat_body_x_to_world_vector(los_w).tolist()
        raw["target"]["kind"] = "red_balloon"
        raw["target"]["initial_position_w"] = balloon_position_w.tolist()
        raw["target"]["velocity_w"] = balloon_velocity_w.tolist()
        raw["perception"]["rng_seed"] = int(scenario["seed"])


def _pursuer_initial(vehicle: dict[str, Any]) -> InitialState:
    return InitialState(
        position_w=_array(vehicle["initial_position_w"], length=3),
        velocity_w=_array(vehicle.get("initial_velocity_w", [0.0, 0.0, 0.0]), length=3),
        quat_xyzw=_array(vehicle.get("initial_quat_xyzw", [0.0, 0.0, 0.0, 1.0]), length=4),
        body_rates_b=_array(vehicle.get("initial_body_rates_b", [0.0, 0.0, 0.0]), length=3),
        rotor_speeds=None,
        wind_w=_array(vehicle.get("wind_w", [0.0, 0.0, 0.0]), length=3),
    )


def _target_initial(target: dict[str, Any]) -> TargetInitialState:
    return TargetInitialState(
        position_w=_array(target["initial_position_w"], length=3),
        velocity_w=_array(target.get("velocity_w", [0.0, 0.0, 0.0]), length=3),
        radius_m=float(target["radius_m"]),
    )


def _quat_body_x_to_world_vector(x_axis_w: np.ndarray) -> np.ndarray:
    x_axis = _unit(x_axis_w)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(x_axis, world_up))) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
    y_axis = _unit(np.cross(world_up, x_axis))
    z_axis = _unit(np.cross(x_axis, y_axis))
    return Rotation.from_matrix(np.column_stack((x_axis, y_axis, z_axis))).as_quat()


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
    los_azimuth_deg: float,
    los_elevation_deg: float,
) -> str:
    distance_label = str(float(distance_m)).replace(".", "p")
    speed_label = str(float(closing_speed_mps)).replace(".", "p")
    azimuth_label = _angle_label(los_azimuth_deg)
    elevation_label = _angle_label(los_elevation_deg)
    return (
        f"{base_name}_d{distance_label}_v{speed_label}_seed_{seed}"
        f"_az{azimuth_label}_el{elevation_label}"
    )


def _angle_label(value: float) -> str:
    sign = "m" if value < 0.0 else "p"
    return f"{sign}{str(abs(float(value))).replace('.', 'p')}"


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


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
