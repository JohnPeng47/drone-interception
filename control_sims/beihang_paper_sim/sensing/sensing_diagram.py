"""add_sensing: camera + perception + paper-faithful IMU.

Paper §VI-A-2 says the controller "does not need to use GPS position data" —
GPS was previously instantiated here but never wired downstream. Removed.

Pixel measurement noise on the image is configured via the existing
intercept_sim YAML (`perception.pixel_noise_std_px`) — FeaturePerceptionModel
already supports it; nothing extra needed here.
"""

from __future__ import annotations

from pydrake.systems.framework import DiagramBuilder

from intercept_sim.sensors import FeaturePerceptionModel, GeometryCamera

from .camera_system import CameraCaptureSystem
from .perception_system import FeatureDetectionSystem
from ..noise_config import NoiseConfig
from .imu_system import ImuSystem


def add_sensing(
    builder: DiagramBuilder,
    *,
    camera: GeometryCamera,
    perception: FeaturePerceptionModel,
    dt: float,
    noise_config: NoiseConfig | None = None,
) -> dict:
    camera_sys = builder.AddSystem(CameraCaptureSystem(camera, dt))
    perception_sys = builder.AddSystem(FeatureDetectionSystem(perception, dt))
    imu = builder.AddSystem(ImuSystem(dt=dt, noise_config=noise_config))

    builder.Connect(camera_sys.GetOutputPort("capture"),
                    perception_sys.GetInputPort("capture"))

    return {
        "camera": camera_sys,
        "perception": perception_sys,
        "imu": imu,
    }
