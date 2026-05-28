from __future__ import annotations

import numpy as np


OBS_SIZE = 26


def observation_from_batch_snapshot(snapshot: dict[str, np.ndarray]) -> np.ndarray:
    pursuer = np.asarray(snapshot["pursuer"], dtype=np.float32)
    target = np.asarray(snapshot["target"], dtype=np.float32)
    metrics = np.asarray(snapshot["metrics"], dtype=np.float32)
    camera = np.asarray(snapshot["camera"], dtype=np.float32)
    n = pursuer.shape[0]

    pos = pursuer[:, 0:3]
    vel = pursuer[:, 3:6]
    quat_xyzw = _normalize_quat(pursuer[:, 6:10])
    omega = pursuer[:, 10:13]
    rpms = pursuer[:, 13:17]
    target_pos = target[:, 0:3]
    target_vel = target[:, 3:6]

    rel_pos_w = target_pos - pos
    rel_vel_w = target_vel - vel
    vel_b = _rotate_world_to_body(quat_xyzw, vel)
    rel_pos_b = _rotate_world_to_body(quat_xyzw, rel_pos_w)
    rel_vel_b = _rotate_world_to_body(quat_xyzw, rel_vel_w)
    gravity_b = _rotate_world_to_body(
        quat_xyzw,
        np.broadcast_to(np.array([0.0, 0.0, -1.0], dtype=np.float32), (n, 3)),
    )

    detected = camera[:, 0:1] > 0.5
    uv_norm = np.where(detected, camera[:, 1:3], 0.0)
    distance = metrics[:, 0]
    range_norm = np.maximum(np.linalg.norm(rel_pos_w, axis=1), 1e-9)
    closing_speed = np.sum(rel_vel_w * rel_pos_w, axis=1) / range_norm
    max_rate = np.maximum(np.asarray(snapshot["max_rate_rps"], dtype=np.float32), 1e-6)
    max_rpm = np.maximum(np.asarray(snapshot["max_rpm"], dtype=np.float32), 1e-6)
    vel_denom = np.float32(20.0 * np.sqrt(3.0))

    obs = np.concatenate(
        [
            vel_b / vel_denom,
            omega / max_rate[:, None],
            gravity_b,
            np.tanh(rel_pos_b * np.float32(0.1)),
            np.tanh(rel_pos_b * np.float32(10.0)),
            rel_vel_b / vel_denom,
            uv_norm,
            np.stack([distance / 20.0, closing_speed / 8.0], axis=1),
            rpms / max_rpm[:, None],
        ],
        axis=1,
    )
    if obs.shape != (n, OBS_SIZE):
        raise RuntimeError(f"expected observation shape {(n, OBS_SIZE)}, got {obs.shape}")
    return obs.astype(np.float32, copy=False)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.maximum(norm, 1e-9)


def _rotate_world_to_body(q_xyzw: np.ndarray, vec_w: np.ndarray) -> np.ndarray:
    qv = -q_xyzw[:, 0:3]
    qw = q_xyzw[:, 3:4]
    t = 2.0 * np.cross(qv, vec_w)
    return vec_w + qw * t + np.cross(qv, t)
