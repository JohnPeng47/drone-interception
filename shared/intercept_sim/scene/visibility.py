from __future__ import annotations

import numpy as np

from intercept_sim.types import CameraCapture, CameraRig, SceneSnapshot, SimulationTarget


def target_position_camera(
    pursuer: SimulationTarget,
    target: SimulationTarget,
    camera: CameraRig,
) -> np.ndarray:
    p_wc = pursuer.position_w + pursuer.rotation_wb @ camera.position_b
    p_target_b = pursuer.rotation_wb.T @ (target.position_w - p_wc)
    return camera.body_to_camera @ p_target_b


def project_target(scene: SceneSnapshot, camera: CameraRig, target: SimulationTarget) -> CameraCapture:
    p_c = target_position_camera(scene.pursuer, target, camera)
    forward = float(p_c[0])
    range_m = float(np.linalg.norm(target.position_w - scene.pursuer.position_w))

    if forward <= 1e-9:
        return _miss(scene.t, camera.id, target.id)

    u_norm = float(p_c[1] / forward)
    v_norm = float(p_c[2] / forward)
    intr = camera.intrinsics
    if abs(u_norm) > np.tan(intr.hfov_rad / 2.0) or abs(v_norm) > np.tan(intr.vfov_rad / 2.0):
        return _miss(scene.t, camera.id, target.id)

    uv_px = np.array(
        [
            intr.fx_px * u_norm + intr.cx_px,
            intr.fy_px * v_norm + intr.cy_px,
        ],
        dtype=float,
    )
    apparent_radius_px = intr.fx_px * target.radius_m / max(range_m, 1e-9)
    return CameraCapture(
        t_capture=scene.t,
        camera_id=camera.id,
        target_id=target.id,
        detected=True,
        uv_px=uv_px,
        uv_norm=np.array([u_norm, v_norm], dtype=float),
        target_pos_c=p_c,
        range_m=range_m,
        apparent_radius_px=float(apparent_radius_px),
    )


def _miss(t: float, camera_id: str, target_id: str | None) -> CameraCapture:
    return CameraCapture(
        t_capture=float(t),
        camera_id=camera_id,
        target_id=target_id,
        detected=False,
        uv_px=None,
        uv_norm=None,
        target_pos_c=None,
        range_m=None,
        apparent_radius_px=None,
    )

