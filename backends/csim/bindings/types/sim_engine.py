from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .camera_sim import CameraConfig
from .sim_types import PUFFER_ACTION_SUBSTEPS, PUFFER_DT, PursuerInitialState, PursuerParams
from .target_sim import TargetConfig, TargetInitialState


@dataclass(frozen=True)
class NoiseConfig:
    """Sensor/perception noise owned by a typed sim configuration."""

    camera_image_delay_s: float = 0.0
    pixel_noise_std_px: tuple[float, float] = (0.0, 0.0)
    dropout_probability: float = 0.0
    sigma_img: float = 0.0
    sigma_gyr: float = 0.0
    sigma_acc: float = 0.0
    sigma_b_gyr: float = 0.0
    sigma_b_acc: float = 0.0
    bias_init_std: float = 0.0
    rng_seed: int = 0


@dataclass(frozen=True)
class SimOptions:
    backend_dt: float = PUFFER_DT
    action_substeps: int = PUFFER_ACTION_SUBSTEPS
    duration_s: float = 0.0
    validation_dt: float | None = None
    command_mode: str = "ctbr"
    ctbr_rate_gain: float = 0.08
    randomize_params: bool = False


@dataclass(frozen=True)
class RenderConfig:
    camera_id: str | None = None
    backend: str = "software"
    platform: str = "auto"
    scene_id: str = "liftoff_fpv_0"
    timeout_ms: int = 16
    fail_on_error: bool = False


@dataclass(frozen=True)
class SimConfig:
    pursuer: PursuerParams
    options: SimOptions = field(default_factory=SimOptions)
    targets: tuple[TargetConfig, ...] = ()
    cameras: tuple[CameraConfig, ...] = ()
    intercept_radius_m: float = 0.0
    max_thrust_n: float = 0.0
    max_rate_rps: float = 0.0
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    rendering: bool = False
    render: RenderConfig = field(default_factory=RenderConfig)


@dataclass(frozen=True)
class SimInstance:
    seed: int
    pursuer_initial: PursuerInitialState
    target_initials: tuple[TargetInitialState, ...]
    config: SimConfig | None = None
    raw_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_initial(self) -> TargetInitialState:
        if not self.target_initials:
            raise AttributeError("SimInstance has no target_initials")
        return self.target_initials[0]
