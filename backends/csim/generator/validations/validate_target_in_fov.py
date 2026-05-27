from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from backends.csim.bindings.types import SimInstance


def validate_target_in_fov(instance: SimInstance) -> None:
    """Require the primary target to start inside the pursuer camera FOV.

    Camera convention matches the C sim / control sim:
    camera x is optical depth, uv_norm = [y_c / x_c, z_c / x_c].
    """
    if instance.config is None:
        raise ValueError("target FOV validation requires SimInstance.config")
    if not instance.target_initials:
        raise ValueError("target FOV validation requires at least one target")
    if not instance.config.cameras:
        raise ValueError("target FOV validation requires at least one camera")

    initial = instance.pursuer_initial
    target = instance.target_initials[0]
    camera = instance.config.cameras[0]

    rotation_wb = Rotation.from_quat(np.asarray(initial.quat_xyzw, dtype=float)).as_matrix()
    p_wc = np.asarray(initial.position_w, dtype=float) + rotation_wb @ np.asarray(camera.position_b, dtype=float)
    p_target_b = rotation_wb.T @ (np.asarray(target.position_w, dtype=float) - p_wc)
    p_target_c = np.asarray(camera.body_to_camera, dtype=float).reshape(3, 3) @ p_target_b

    forward = float(p_target_c[0])
    if forward <= 1e-9:
        raise ValueError(f"target starts behind camera: target_pos_c={p_target_c.tolist()}")

    u_norm = float(p_target_c[1] / forward)
    v_norm = float(p_target_c[2] / forward)
    h_limit = float(np.tan(float(camera.intrinsics.hfov_rad) / 2.0))
    v_limit = float(np.tan(float(camera.intrinsics.vfov_rad) / 2.0))
    if abs(u_norm) > h_limit or abs(v_norm) > v_limit:
        raise ValueError(
            "target starts outside camera FOV: "
            f"uv_norm={[u_norm, v_norm]}, limits={[h_limit, v_limit]}, "
            f"target_pos_c={p_target_c.tolist()}"
        )
