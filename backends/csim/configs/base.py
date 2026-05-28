from __future__ import annotations

import math

import numpy as np

from backends.csim.bindings.types import (
    CameraConfig,
    CameraIntrinsics,
    NoiseConfig,
    PursuerParams,
    SimConfig,
    SimOptions,
    TargetConfig,
)


_ARM_M = 0.174
_CAMERA_WIDTH_PX = 1920
_CAMERA_HEIGHT_PX = 1080
_CAMERA_HFOV_RAD = math.radians(90.0)
_CAMERA_VFOV_RAD = math.radians(60.0)

SIM_CONFIG = SimConfig(
    pursuer=PursuerParams(
        mass_kg=2.064,
        ixx=0.0217,
        iyy=0.0217,
        izz=0.0400,
        arm_len_m=_ARM_M,
        k_thrust=8.54858e-6,
        k_yaw=0.016,
        max_rpm=21702.0,
        rotor_positions_b=np.array([
            [_ARM_M, _ARM_M, 0.0],
            [-_ARM_M, _ARM_M, 0.0],
            [-_ARM_M, -_ARM_M, 0.0],
            [_ARM_M, -_ARM_M, 0.0],
        ]),
        rotor_directions=np.array([1.0, -1.0, 1.0, -1.0]),
    ),
    options=SimOptions(
        backend_dt=0.005,
        action_substeps=1,
        duration_s=3.0,
        validation_dt=None,
    ),
    targets=(
        TargetConfig(
            id="target",
            kind="target",
            radius_m=0.2,
        ),
    ),
    cameras=(
        CameraConfig(
            id="front",
            parent_id="interceptor",
            position_b=np.array([0.0, 0.0, 0.0]),
            body_to_camera=np.eye(3),
            intrinsics=CameraIntrinsics(
                width_px=_CAMERA_WIDTH_PX,
                height_px=_CAMERA_HEIGHT_PX,
                fx_px=_CAMERA_WIDTH_PX / (2.0 * math.tan(_CAMERA_HFOV_RAD / 2.0)),
                fy_px=_CAMERA_HEIGHT_PX / (2.0 * math.tan(_CAMERA_VFOV_RAD / 2.0)),
                cx_px=_CAMERA_WIDTH_PX / 2.0,
                cy_px=_CAMERA_HEIGHT_PX / 2.0,
                hfov_rad=_CAMERA_HFOV_RAD,
                vfov_rad=_CAMERA_VFOV_RAD,
            ),
            capture_rate_hz=30.0,
        ),
    ),
    intercept_radius_m=0.5,
    max_thrust_n=40.0,
    max_rate_rps=8.0,
    noise=NoiseConfig(
        camera_image_delay_s=0.8,
        pixel_noise_std_px=(0.0, 0.0),
        dropout_probability=0.0,
        rng_seed=1,
    ),
)
