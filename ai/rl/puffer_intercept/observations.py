from __future__ import annotations

import numpy as np

from backends.csim.bindings.types import SimSnapshotArrays, SimSnapshots


OBS_SIZE = 25


def observation_from_batch_snapshot(
    snapshot: SimSnapshots,
    previous_action: np.ndarray | None = None,
) -> np.ndarray:
    return observation_from_batch_arrays(snapshot.arrays, previous_action=previous_action)


def observation_from_batch_arrays(
    arrays: SimSnapshotArrays,
    previous_action: np.ndarray | None = None,
) -> np.ndarray:
    pursuer = arrays.pursuer
    target = arrays.target
    n = pursuer.shape[0]

    pos = pursuer[:, 0:3]
    vel = pursuer[:, 3:6]
    quat_xyzw = _normalize_quat(pursuer[:, 6:10])
    target_pos = target[:, 0:3]
    target_vel = target[:, 3:6]
    prev = (
        np.zeros((n, 4), dtype=np.float32)
        if previous_action is None
        else np.asarray(previous_action, dtype=np.float32).reshape(n, 4)
    )

    obs = np.concatenate(
        [
            pos,
            vel,
            _quat_xyzw_to_matrix(quat_xyzw).reshape(n, 9),
            prev,
            target_pos,
            target_vel,
        ],
        axis=1,
    )
    if obs.shape != (n, OBS_SIZE):
        raise RuntimeError(f"expected observation shape {(n, OBS_SIZE)}, got {obs.shape}")
    return obs.astype(np.float32, copy=False)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.maximum(norm, 1e-9)


def _quat_xyzw_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    x = q_xyzw[:, 0]
    y = q_xyzw[:, 1]
    z = q_xyzw[:, 2]
    w = q_xyzw[:, 3]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    out = np.empty((q_xyzw.shape[0], 3, 3), dtype=np.float32)
    out[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    out[:, 0, 1] = 2.0 * (xy - wz)
    out[:, 0, 2] = 2.0 * (xz + wy)
    out[:, 1, 0] = 2.0 * (xy + wz)
    out[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    out[:, 1, 2] = 2.0 * (yz - wx)
    out[:, 2, 0] = 2.0 * (xz - wy)
    out[:, 2, 1] = 2.0 * (yz + wx)
    out[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return out
