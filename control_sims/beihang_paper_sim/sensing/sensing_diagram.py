"""add_sensing: camera + perception + paper-faithful IMU.

Paper §VI-A-2 says the controller "does not need to use GPS position data" —
GPS was previously instantiated here but never wired downstream. Removed.

Pixel measurement noise on the image is configured via
`perception.pixel_noise_std_px`; FeaturePerceptionModel already supports it.
"""

from __future__ import annotations

from pydrake.systems.framework import DiagramBuilder

from ..noise_config import NoiseConfig
from . import FeaturePerceptionModel, GeometryCamera
from .camera_system import CameraCaptureSystem
from .imu_system import ImuSystem
from .perception_system import FeatureDetectionSystem


def add_sensing(
    builder: DiagramBuilder,
    *,
    camera: GeometryCamera | None,
    perception: FeaturePerceptionModel,
    dt: float,
    noise_config: NoiseConfig | None = None,
) -> dict:
    perception_sys = builder.AddSystem(FeatureDetectionSystem(perception, dt))
    imu = builder.AddSystem(ImuSystem(dt=dt, noise_config=noise_config))

    out = {
        "perception": perception_sys,
        "imu": imu,
    }
    if camera is not None:
        camera_sys = builder.AddSystem(CameraCaptureSystem(camera, dt))
        builder.Connect(camera_sys.GetOutputPort("capture"),
                        perception_sys.GetInputPort("capture"))
        out["camera"] = camera_sys

    return out
