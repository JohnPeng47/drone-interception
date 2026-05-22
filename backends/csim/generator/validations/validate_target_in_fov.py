from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_INITIAL_PITCH_OFFSET_DEG = 20.0


def validate_target_in_fov(raw_config: dict[str, Any]) -> None:
    """Require the primary target to start inside the pursuer camera FOV.

    Camera convention matches the C sim / control sim:
    camera x is optical depth, uv_norm = [y_c / x_c, z_c / x_c].
    """

    vehicle = raw_config["vehicle"]
    target = raw_config["target"].get("initial_state", raw_config["target"])
    camera = raw_config["camera"]

    pursuer_position_w = _array(vehicle["initial_position_w"], length=3)
    target_position_w = _array(
        target["position_w"] if "position_w" in target else raw_config["target"]["initial_position_w"],
        length=3,
    )
    q_xyzw = _array(vehicle.get("initial_quat_xyzw", [0.0, 0.0, 0.0, 1.0]), length=4)
    q_xyzw = apply_initial_pitch_offset(q_xyzw, vehicle)
    rotation_wb = Rotation.from_quat(q_xyzw).as_matrix()

    camera_position_b = _array(camera.get("position_b", [0.0, 0.0, 0.0]), length=3)
    body_to_camera = np.asarray(camera.get("body_to_camera", np.eye(3)), dtype=float).reshape(3, 3)
    p_wc = pursuer_position_w + rotation_wb @ camera_position_b
    p_target_b = rotation_wb.T @ (target_position_w - p_wc)
    p_target_c = body_to_camera @ p_target_b

    forward = float(p_target_c[0])
    if forward <= 1e-9:
        raise ValueError(
            f"target starts behind camera: target_pos_c={p_target_c.tolist()}"
        )

    u_norm = float(p_target_c[1] / forward)
    v_norm = float(p_target_c[2] / forward)
    h_limit = float(np.tan(np.deg2rad(float(camera["hfov_deg"])) / 2.0))
    v_limit = float(np.tan(np.deg2rad(float(camera["vfov_deg"])) / 2.0))
    if abs(u_norm) > h_limit or abs(v_norm) > v_limit:
        raise ValueError(
            "target starts outside camera FOV: "
            f"uv_norm={[u_norm, v_norm]}, limits={[h_limit, v_limit]}, "
            f"target_pos_c={p_target_c.tolist()}"
        )


def apply_initial_pitch_offset(q_xyzw: np.ndarray, vehicle: dict[str, Any]) -> np.ndarray:
    pitch_deg = float(vehicle.get("initial_pitch_offset_deg", DEFAULT_INITIAL_PITCH_OFFSET_DEG))
    if abs(pitch_deg) < 1e-9:
        return q_xyzw
    theta = np.deg2rad(pitch_deg)
    q_pitch = np.array([0.0, np.sin(theta / 2.0), 0.0, np.cos(theta / 2.0)], dtype=float)
    q_new = _quat_mul(q_xyzw, q_pitch)
    return q_new / max(float(np.linalg.norm(q_new)), 1e-12)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=float)


def _array(value: Any, *, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"Expected array of shape ({length},), got {arr.shape}")
    return arr.copy()
