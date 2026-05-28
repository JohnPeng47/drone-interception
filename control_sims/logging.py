from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .sim_runner import BatchSimEngineStep, CtbrCommandBatch


@dataclass(frozen=True)
class LoggingConfig:
    output_dir: Path
    every_n_ticks: int = 1
    log_pursuer_state: bool = True
    log_target_state: bool = True
    log_metrics: bool = True
    log_camera: bool = True
    log_commands: bool = True
    log_motor_state: bool = True
    # Future fields once the snapshot exposes them:
    # log_position_setpoint: bool = False
    # log_velocity_setpoint: bool = False
    # log_attitude_setpoint: bool = False
    # log_rate_setpoint: bool = False
    # log_desired_torques: bool = False
    # log_motor_saturation: bool = False
    # log_imu: bool = False
    # log_estimator_state: bool = False

    def __post_init__(self) -> None:
        if int(self.every_n_ticks) <= 0:
            raise ValueError("every_n_ticks must be positive")


class SnapshotLogger:
    """Append normalized sim snapshots under a run output directory."""

    def __init__(self, sim_name: str, config: LoggingConfig):
        self.sim_name = str(sim_name)
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.path = self.output_dir / f"{self.sim_name}.csv"
        self._handle = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> SnapshotLogger:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def open(self) -> None:
        if self._handle is not None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logging_config.json").write_text(
            json.dumps(_config_to_json(self.config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=_fieldnames(self.config))
        self._writer.writeheader()

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None
        self._writer = None

    def log_snapshots(self, step: BatchSimEngineStep) -> None:
        self.open()
        assert self._writer is not None
        rows = list(_rows_from_batch_step(self.sim_name, self.config, step))
        if rows:
            self._writer.writerows(rows)


def _rows_from_batch_step(
    sim_name: str,
    config: LoggingConfig,
    step: BatchSimEngineStep,
) -> list[dict[str, Any]]:
    state = step.state
    snapshot = state.snapshot
    commands = _commands_to_arrays(step.commands)
    rows: list[dict[str, Any]] = []
    for slot in np.flatnonzero(state.active):
        slot_i = int(slot)
        tick = int(state.steps[slot_i])
        if tick <= 0 or tick % int(config.every_n_ticks) != 0:
            continue
        instance = state.instances[slot_i]
        row: dict[str, Any] = {
            "sim": sim_name,
            "slot": slot_i,
            "workload_index": int(state.workload_indices[slot_i]),
            "seed": "" if instance is None else int(instance.seed),
            "tick": tick,
            "t_s": float(state.elapsed_s[slot_i]),
        }
        if config.log_pursuer_state:
            _add_pursuer(row, np.asarray(snapshot["pursuer"])[slot_i])
        if config.log_motor_state:
            _add_motor_state(row, np.asarray(snapshot["pursuer"])[slot_i])
        if config.log_target_state and "target" in snapshot:
            _add_target(row, np.asarray(snapshot["target"])[slot_i])
        if config.log_metrics and "metrics" in snapshot:
            _add_metrics(row, np.asarray(snapshot["metrics"])[slot_i])
        if config.log_camera and "camera" in snapshot:
            _add_camera(row, np.asarray(snapshot["camera"])[slot_i])
        if config.log_commands and commands is not None:
            thrust_n, body_rates_b = commands
            row.update({
                "command_thrust_n": float(thrust_n[slot_i]),
                "command_body_rate_x_rad_s": float(body_rates_b[slot_i, 0]),
                "command_body_rate_y_rad_s": float(body_rates_b[slot_i, 1]),
                "command_body_rate_z_rad_s": float(body_rates_b[slot_i, 2]),
            })
        rows.append(row)
    return rows


def _commands_to_arrays(commands: CtbrCommandBatch | Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray] | None:
    if commands is None:
        return None
    if isinstance(commands, CtbrCommandBatch):
        thrust_n = commands.thrust_n
        body_rates_b = commands.body_rates_b
    else:
        thrust_n = commands["thrust_n"]
        body_rates_b = commands["body_rates_b"]
    return (
        np.asarray(thrust_n, dtype=float).reshape(-1),
        np.asarray(body_rates_b, dtype=float).reshape(-1, 3),
    )


def _add_pursuer(row: dict[str, Any], pursuer: np.ndarray) -> None:
    row.update({
        "pursuer_x_w_m": float(pursuer[0]),
        "pursuer_y_w_m": float(pursuer[1]),
        "pursuer_z_w_m": float(pursuer[2]),
        "pursuer_vx_w_mps": float(pursuer[3]),
        "pursuer_vy_w_mps": float(pursuer[4]),
        "pursuer_vz_w_mps": float(pursuer[5]),
        "pursuer_qx": float(pursuer[6]),
        "pursuer_qy": float(pursuer[7]),
        "pursuer_qz": float(pursuer[8]),
        "pursuer_qw": float(pursuer[9]),
        "pursuer_p_b_rad_s": float(pursuer[10]),
        "pursuer_q_b_rad_s": float(pursuer[11]),
        "pursuer_r_b_rad_s": float(pursuer[12]),
    })


def _add_motor_state(row: dict[str, Any], pursuer: np.ndarray) -> None:
    if len(pursuer) >= 17:
        row.update({
            "motor_0_rpm": float(pursuer[13]),
            "motor_1_rpm": float(pursuer[14]),
            "motor_2_rpm": float(pursuer[15]),
            "motor_3_rpm": float(pursuer[16]),
        })


def _add_target(row: dict[str, Any], target: np.ndarray) -> None:
    row.update({
        "target_x_w_m": float(target[0]),
        "target_y_w_m": float(target[1]),
        "target_z_w_m": float(target[2]),
        "target_vx_w_mps": float(target[3]),
        "target_vy_w_mps": float(target[4]),
        "target_vz_w_mps": float(target[5]),
    })


def _add_metrics(row: dict[str, Any], metrics: np.ndarray) -> None:
    row.update({
        "distance_m": float(metrics[0]),
        "min_distance_m": float(metrics[1]),
        "intercepted": bool(metrics[2] > 0.5),
        "intercept_time_s": float(metrics[3]),
        "target_index": int(metrics[4]),
    })


def _add_camera(row: dict[str, Any], camera: np.ndarray) -> None:
    row.update({
        "camera_detected": bool(camera[0] > 0.5),
        "camera_u_norm": float(camera[1]),
        "camera_v_norm": float(camera[2]),
    })


def _fieldnames(config: LoggingConfig) -> list[str]:
    fields = ["sim", "slot", "workload_index", "seed", "tick", "t_s"]
    if config.log_pursuer_state:
        fields.extend([
            "pursuer_x_w_m",
            "pursuer_y_w_m",
            "pursuer_z_w_m",
            "pursuer_vx_w_mps",
            "pursuer_vy_w_mps",
            "pursuer_vz_w_mps",
            "pursuer_qx",
            "pursuer_qy",
            "pursuer_qz",
            "pursuer_qw",
            "pursuer_p_b_rad_s",
            "pursuer_q_b_rad_s",
            "pursuer_r_b_rad_s",
        ])
    if config.log_motor_state:
        fields.extend(["motor_0_rpm", "motor_1_rpm", "motor_2_rpm", "motor_3_rpm"])
    if config.log_target_state:
        fields.extend([
            "target_x_w_m",
            "target_y_w_m",
            "target_z_w_m",
            "target_vx_w_mps",
            "target_vy_w_mps",
            "target_vz_w_mps",
        ])
    if config.log_metrics:
        fields.extend(["distance_m", "min_distance_m", "intercepted", "intercept_time_s", "target_index"])
    if config.log_camera:
        fields.extend(["camera_detected", "camera_u_norm", "camera_v_norm"])
    if config.log_commands:
        fields.extend([
            "command_thrust_n",
            "command_body_rate_x_rad_s",
            "command_body_rate_y_rad_s",
            "command_body_rate_z_rad_s",
        ])
    return fields


def _config_to_json(config: LoggingConfig) -> dict[str, Any]:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    return data
