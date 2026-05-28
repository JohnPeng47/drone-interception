from __future__ import annotations

from typing import Any

import numpy as np

from .sensing import FeaturePerceptionModel
from .targets import KinematicTarget
from .types import CameraIntrinsics, CameraRig


def validate_experiment_config(raw: dict[str, Any]) -> None:
    required_top = {
        "experiment",
        "sim",
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
        raise ValueError(f"Missing experiment config sections: {sorted(missing)}")

    for section, keys in {
        "experiment": ("name",),
        "sim": ("duration_s", "dt"),
        "vehicle": ("initial_position_w",),
        "target": ("radius_m",),
        "camera": (
            "width_px",
            "height_px",
            "fx_px",
            "fy_px",
            "hfov_deg",
            "vfov_deg",
            "capture_rate_hz",
        ),
        "perception": ("camera_image_delay_s",),
        "observer": ("type",),
        "controller": ("max_rate_rps",),
        "metrics": ("catch_radius_m",),
    }.items():
        missing_keys = [key for key in keys if key not in raw[section]]
        if missing_keys:
            raise ValueError(f"Missing {section} keys: {missing_keys}")

    target = raw["target"]
    target_initial = target.get("initial_state", target)
    if "position_w" not in target_initial and "initial_position_w" not in target:
        raise ValueError("Missing target initial position")
    if "velocity_w" not in target_initial and "velocity_w" not in target:
        raise ValueError("Missing target velocity")


def initial_rotorpy_state(vehicle_config: dict[str, Any], quad_params: dict[str, Any]) -> dict[str, np.ndarray]:
    hover_speed = np.sqrt(
        quad_params["mass"] * 9.81 / (quad_params["num_rotors"] * quad_params["k_eta"])
    )
    return {
        "x": array(vehicle_config["initial_position_w"], length=3),
        "v": array(vehicle_config.get("initial_velocity_w", [0.0, 0.0, 0.0]), length=3),
        "q": array(vehicle_config.get("initial_quat_xyzw", [0.0, 0.0, 0.0, 1.0]), length=4),
        "w": array(vehicle_config.get("initial_body_rates_b", [0.0, 0.0, 0.0]), length=3),
        "wind": array(vehicle_config.get("wind_w", [0.0, 0.0, 0.0]), length=3),
        "rotor_speeds": np.full(quad_params["num_rotors"], hover_speed, dtype=float),
    }


def target_from_config(target_config: dict[str, Any]) -> KinematicTarget:
    initial = target_config.get("initial_state", target_config)
    return KinematicTarget(
        target_id=str(target_config.get("id", "target")),
        kind=str(target_config.get("kind", "target")),
        initial_position_w=array(initial["position_w"] if "position_w" in initial else target_config["initial_position_w"], length=3),
        velocity_w=array(initial["velocity_w"] if "velocity_w" in initial else target_config["velocity_w"], length=3),
        radius_m=float(target_config["radius_m"]),
    )


def camera_from_config(camera_config: dict[str, Any]) -> CameraRig:
    width_px = int(camera_config["width_px"])
    height_px = int(camera_config["height_px"])
    return CameraRig(
        id=str(camera_config.get("id", "front")),
        parent_id=str(camera_config.get("parent_id", "interceptor")),
        position_b=array(camera_config.get("position_b", [0.0, 0.0, 0.0]), length=3),
        body_to_camera=np.asarray(camera_config.get("body_to_camera", np.eye(3)), dtype=float).reshape(3, 3),
        intrinsics=CameraIntrinsics(
            width_px=width_px,
            height_px=height_px,
            fx_px=float(camera_config["fx_px"]),
            fy_px=float(camera_config["fy_px"]),
            cx_px=float(camera_config.get("cx_px", width_px / 2.0)),
            cy_px=float(camera_config.get("cy_px", height_px / 2.0)),
            hfov_rad=np.deg2rad(float(camera_config["hfov_deg"])),
            vfov_rad=np.deg2rad(float(camera_config["vfov_deg"])),
        ),
        capture_rate_hz=float(camera_config["capture_rate_hz"]),
    )


def perception_from_config(perception_config: dict[str, Any]) -> FeaturePerceptionModel:
    return FeaturePerceptionModel(
        camera_image_delay_s=float(perception_config["camera_image_delay_s"]),
        pixel_noise_std_px=array(perception_config.get("pixel_noise_std_px", [0.0, 0.0]), length=2),
        dropout_probability=float(perception_config.get("dropout_probability", 0.0)),
        rng=np.random.default_rng(int(perception_config.get("rng_seed", 1))),
    )


def array(value: Any, *, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"Expected array of shape ({length},), got {arr.shape}")
    return arr.copy()
