from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from intercept_sim.analysis import ExperimentMetrics, compute_metrics
from intercept_sim.controllers import BeihangBacksteppingController, GeometricImageFeatureController, ImageFeatureIbvsController
from intercept_sim.experiments.config import ExperimentConfig, load_experiment_config
from intercept_sim.experiments.telemetry import ExperimentTelemetry, build_experiment_telemetry
from intercept_sim.observers import (
    BeihangImageEkfObserver,
    ConstantVelocityFeatureObserver,
    DelayedFeatureReplayObserver,
    LatestFeatureObserver,
    TruthRelativeFeatureObserver,
)
from intercept_sim.runner import InterceptionRunner, RunnerStep
from intercept_sim.sensors import FeaturePerceptionModel, GeometryCamera
from intercept_sim.targets import KinematicTarget
from intercept_sim.types import CameraIntrinsics, CameraRig


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    log: list[RunnerStep]
    metrics: ExperimentMetrics
    comment: str = ""
    telemetry: ExperimentTelemetry | None = None

    def summary_dict(self) -> dict[str, Any]:
        if self.telemetry is not None:
            return self.telemetry.to_summary_dict()
        return _summary_dict(self.config, self.metrics, len(self.log), self.comment)


def run_experiment(config_or_path: ExperimentConfig | str | Path, *, comment: str = "") -> ExperimentResult:
    config = config_or_path if isinstance(config_or_path, ExperimentConfig) else load_experiment_config(config_or_path)
    runner = build_runner(config)
    log = runner.run(config.duration_s)
    metrics = compute_metrics(log, catch_radius_m=config.catch_radius_m)
    telemetry = build_experiment_telemetry(
        experiment_id=config.name,
        comment=str(comment),
        config=config.raw,
        metrics=metrics,
        log=log,
    )
    return ExperimentResult(config=config, log=log, metrics=metrics, comment=str(comment), telemetry=telemetry)


def save_experiment_result(result: ExperimentResult, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result.summary_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_compact_log(result: ExperimentResult, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for step in result.log:
            handle.write(json.dumps(_compact_step(step), sort_keys=True))
            handle.write("\n")


def save_experiment_telemetry(result: ExperimentResult, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry = result.telemetry
    if telemetry is None:
        telemetry = build_experiment_telemetry(
            experiment_id=result.config.name,
            comment=result.comment,
            config=result.config.raw,
            metrics=result.metrics,
            log=result.log,
        )
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "metadata", **telemetry.to_summary_dict()}, sort_keys=True))
        handle.write("\n")
        for step in telemetry.steps:
            handle.write(json.dumps({"kind": "step", **step.to_dict()}, sort_keys=True))
            handle.write("\n")


def build_runner(config: ExperimentConfig) -> InterceptionRunner:
    from rotorpy.sensors.imu import Imu
    from rotorpy.vehicles.multirotor import Multirotor

    quad_params = _vehicle_params_from_config(config.raw["vehicle"])
    initial_state = _initial_rotorpy_state(config.raw["vehicle"], quad_params)
    vehicle = Multirotor(
        quad_params,
        initial_state=initial_state,
        control_abstraction="cmd_ctbr",
        aero=bool(config.raw["vehicle"].get("aero", False)),
        integrator_kwargs=config.raw["vehicle"].get(
            "integrator_kwargs", {"method": "RK45", "rtol": 1e-6, "atol": 1e-9}
        ),
    )

    return InterceptionRunner(
        vehicle=vehicle,
        imu=Imu(sampling_rate=float(config.raw["vehicle"].get("imu_rate_hz", 200.0))),
        target=_target_from_config(config.raw["target"]),
        camera=GeometryCamera(_camera_from_config(config.raw["camera"])),
        perception=_perception_from_config(config.raw["perception"]),
        observer=_observer_from_config(config.raw["observer"]),
        controller=_controller_from_config(config.raw["controller"], config.raw["camera"], mass_kg=float(quad_params["mass"])),
        dt=config.dt,
        initial_state=initial_state,
    )


def _initial_rotorpy_state(vehicle_config: dict[str, Any], quad_params: dict[str, Any]) -> dict[str, np.ndarray]:
    hover_speed = np.sqrt(quad_params["mass"] * 9.81 / (quad_params["num_rotors"] * quad_params["k_eta"]))
    return {
        "x": _array(vehicle_config["initial_position_w"], length=3),
        "v": _array(vehicle_config.get("initial_velocity_w", [0.0, 0.0, 0.0]), length=3),
        "q": _array(vehicle_config.get("initial_quat_xyzw", [0.0, 0.0, 0.0, 1.0]), length=4),
        "w": _array(vehicle_config.get("initial_body_rates_b", [0.0, 0.0, 0.0]), length=3),
        "wind": _array(vehicle_config.get("wind_w", [0.0, 0.0, 0.0]), length=3),
        "rotor_speeds": np.full(quad_params["num_rotors"], hover_speed, dtype=float),
    }


def _vehicle_params_from_config(vehicle_config: dict[str, Any]) -> dict[str, Any]:
    model = str(vehicle_config.get("model", "hummingbird"))
    module_names = {
        "hummingbird": "rotorpy.vehicles.hummingbird_params",
        "crazyflie": "rotorpy.vehicles.crazyflie_params",
        "crazyfliebrushless": "rotorpy.vehicles.crazyfliebrushless_params",
        "px4_sihsim_quadx": "rotorpy.vehicles.px4_sihsim_quadx_params",
    }
    if model not in module_names:
        raise ValueError(f"Unsupported vehicle model: {model}")
    module = importlib.import_module(module_names[model])
    return dict(module.quad_params)


def _target_from_config(target_config: dict[str, Any]) -> KinematicTarget:
    return KinematicTarget(
        target_id=str(target_config.get("id", "target")),
        kind=str(target_config.get("kind", "target")),
        initial_position_w=_array(target_config["initial_position_w"], length=3),
        velocity_w=_array(target_config["velocity_w"], length=3),
        radius_m=float(target_config["radius_m"]),
    )


def _camera_from_config(camera_config: dict[str, Any]) -> CameraRig:
    width_px = int(camera_config["width_px"])
    height_px = int(camera_config["height_px"])
    return CameraRig(
        id=str(camera_config.get("id", "front")),
        parent_id=str(camera_config.get("parent_id", "interceptor")),
        position_b=_array(camera_config.get("position_b", [0.0, 0.0, 0.0]), length=3),
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


def _perception_from_config(perception_config: dict[str, Any]) -> FeaturePerceptionModel:
    return FeaturePerceptionModel(
        processing_delay_s=float(perception_config["processing_delay_s"]),
        pixel_noise_std_px=_array(perception_config.get("pixel_noise_std_px", [0.0, 0.0]), length=2),
        dropout_probability=float(perception_config.get("dropout_probability", 0.0)),
        rng=np.random.default_rng(int(perception_config.get("rng_seed", 1))),
    )


def _observer_from_config(
    observer_config: dict[str, Any],
) -> LatestFeatureObserver | ConstantVelocityFeatureObserver | DelayedFeatureReplayObserver | TruthRelativeFeatureObserver | BeihangImageEkfObserver:
    observer_type = str(observer_config["type"])
    if observer_type == "latest":
        return LatestFeatureObserver()
    if observer_type == "constant_velocity":
        return ConstantVelocityFeatureObserver(history_size=int(observer_config.get("history_size", 4)))
    if observer_type in {"delayed_replay", "dkf_scaffold"}:
        return DelayedFeatureReplayObserver(history_size=int(observer_config.get("history_size", 16)))
    if observer_type in {"beihang_image_ekf", "beihang_dkf"}:
        return BeihangImageEkfObserver(history_size=int(observer_config.get("history_size", 50)))
    if observer_type == "truth_relative":
        return TruthRelativeFeatureObserver()
    raise ValueError(f"Unsupported observer type: {observer_type}")


def _controller_from_config(
    controller_config: dict[str, Any],
    camera_config: dict[str, Any],
    *,
    mass_kg: float,
) -> ImageFeatureIbvsController | GeometricImageFeatureController | BeihangBacksteppingController:
    controller_type = str(controller_config.get("type", "ibvs"))
    if controller_type == "ibvs":
        return ImageFeatureIbvsController(
            mass_kg=mass_kg,
            k_yaw=float(controller_config.get("k_yaw", 2.0)),
            k_pitch=float(controller_config.get("k_pitch", 2.0)),
            max_rate_rps=float(controller_config["max_rate_rps"]),
        )
    if controller_type == "geometric":
        body_to_camera = np.asarray(camera_config.get("body_to_camera", np.eye(3)), dtype=float).reshape(3, 3)
        return GeometricImageFeatureController(
            mass_kg=mass_kg,
            k_align=float(controller_config.get("k_align", 2.0)),
            max_rate_rps=float(controller_config["max_rate_rps"]),
            camera_to_body=body_to_camera.T,
            desired_los_b=_array(controller_config.get("desired_los_b", [1.0, 0.0, 0.0]), length=3),
        )
    if controller_type == "beihang_backstepping":
        body_to_camera = np.asarray(camera_config.get("body_to_camera", np.eye(3)), dtype=float).reshape(3, 3)
        return BeihangBacksteppingController(
            mass_kg=mass_kg,
            gravity_mps2=float(controller_config.get("gravity_mps2", 9.801)),
            barrier_k=float(controller_config.get("barrier_k", 0.25)),
            c1=float(controller_config.get("c1", 1.0)),
            alpha1_gain=float(controller_config.get("alpha1_gain", 10.0)),
            c2=float(controller_config.get("c2", 0.8)),
            c3=float(controller_config.get("c3", 0.3)),
            max_rate_rps=float(controller_config["max_rate_rps"]),
            max_thrust_n=(
                None if "max_thrust_n" not in controller_config else float(controller_config["max_thrust_n"])
            ),
            camera_to_body=body_to_camera.T,
            camera_optical_axis_c=_array(controller_config.get("camera_optical_axis_c", [1.0, 0.0, 0.0]), length=3),
        )
    raise ValueError(f"Unsupported controller type: {controller_type}")


def _array(value: Any, *, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,):
        raise ValueError(f"Expected array of shape ({length},), got {array.shape}")
    return array


def _compact_step(step: RunnerStep) -> dict[str, Any]:
    feature = step.observer_state.image_feature
    capture = step.capture
    return {
        "t": step.t,
        "pursuer_position_w": np.asarray(step.rotorpy_state["x"], dtype=float).tolist(),
        "pursuer_velocity_w": np.asarray(step.rotorpy_state["v"], dtype=float).tolist(),
        "target_position_w": np.asarray(step.scene.targets[0].position_w, dtype=float).tolist() if step.scene.targets else None,
        "capture_detected": capture.detected if capture is not None else None,
        "capture_uv_norm": capture.uv_norm.tolist() if capture is not None and capture.uv_norm is not None else None,
        "feature_detected": feature.detected if feature is not None else None,
        "feature_uv_norm": feature.uv_norm.tolist() if feature is not None and feature.uv_norm is not None else None,
        "command_thrust_n": step.command.thrust_n,
        "command_body_rates_b": np.asarray(step.command.body_rates_b, dtype=float).tolist(),
    }


def _summary_dict(
    config: ExperimentConfig,
    metrics: ExperimentMetrics,
    steps: int,
    comment: str,
) -> dict[str, Any]:
    return {
        "experiment": config.name,
        "comment": comment,
        "duration_s": config.duration_s,
        "dt": config.dt,
        "steps": steps,
        "metrics": metrics.to_dict(),
    }
