from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


OBS_SIZE = 26


def observation_from_snapshot(snapshot: dict, *, max_vel_mps: float = 20.0, max_rate_rps: float = 20.0) -> np.ndarray:
    vehicle = snapshot["vehicle_state"]
    target = snapshot["target_states"][0]
    q_xyzw = np.asarray(vehicle["q"], dtype=float)
    rotation_wb = Rotation.from_quat(q_xyzw).as_matrix()
    rotation_bw = rotation_wb.T

    pos = np.asarray(vehicle["x"], dtype=float)
    vel = np.asarray(vehicle["v"], dtype=float)
    omega = np.asarray(vehicle["w"], dtype=float)
    target_pos = np.asarray(target["position_w"], dtype=float)
    target_vel = np.asarray(target["velocity_w"], dtype=float)
    rel_pos_w = target_pos - pos
    rel_vel_w = target_vel - vel

    vel_b = rotation_bw @ vel
    rel_pos_b = rotation_bw @ rel_pos_w
    rel_vel_b = rotation_bw @ rel_vel_w
    gravity_b = rotation_bw @ np.array([0.0, 0.0, -1.0], dtype=float)
    rpms = np.asarray(vehicle["rotor_speeds"], dtype=float)
    max_rpm = _max_rpm(snapshot)

    camera_output = _latest_camera_output(snapshot)
    if camera_output is not None and camera_output.get("uv_norm") is not None:
        uv_norm = np.asarray(camera_output["uv_norm"], dtype=float)
    else:
        uv_norm = np.zeros(2, dtype=float)

    distance = float(snapshot["metrics"]["distance_m"])
    closing_speed = _closing_speed(rel_pos_w, rel_vel_w)
    denom = max(float(max_vel_mps), 1e-6) * np.sqrt(3.0)
    obs = np.concatenate(
        [
            vel_b / denom,
            omega / max(float(max_rate_rps), 1e-6),
            gravity_b,
            np.tanh(rel_pos_b * 0.1),
            np.tanh(rel_pos_b * 10.0),
            rel_vel_b / denom,
            uv_norm,
            np.array([distance / 20.0, closing_speed / 8.0], dtype=float),
            rpms / max(max_rpm, 1e-6),
        ]
    )
    if obs.shape != (OBS_SIZE,):
        raise RuntimeError(f"expected observation shape {(OBS_SIZE,)}, got {obs.shape}")
    return obs.astype(np.float32, copy=False)


def _latest_camera_output(snapshot: dict) -> dict | None:
    outputs = snapshot.get("camera_outputs", ())
    if not outputs:
        return None
    return outputs[-1]


def _closing_speed(rel_pos_w: np.ndarray, rel_vel_w: np.ndarray) -> float:
    norm = float(np.linalg.norm(rel_pos_w))
    if norm <= 1e-9:
        return 0.0
    los_w = rel_pos_w / norm
    return float(np.dot(rel_vel_w, los_w))


def _max_rpm(snapshot: dict) -> float:
    # SimEngine snapshots do not include params. Generated tables use X500
    # params with max_rpm stored in SimConfig, so env.py passes a better value
    # by scaling commands. Observation only needs a stable normalization.
    return 21702.0
