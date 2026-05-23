from __future__ import annotations

from dataclasses import dataclass, field
from .camera_sim import CameraConfig
from .sim_types import PUFFER_ACTION_SUBSTEPS, PUFFER_DT, PursuerInitialState, PursuerParams
from .target_sim import TargetConfig, TargetInitialState


@dataclass(frozen=True)
class NoiseConfig:
    """Sensor/perception noise owned by a typed sim configuration.

    The C sim currently owns camera geometry/projection and detection gating.
    These fields describe measurement noise/delay/dropout for callers that add
    noisy sensor/perception layers around those geometric observations.
    """

    processing_delay_s: float = 0.0
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
class SimConfig:
    pursuer: PursuerParams
    options: SimOptions = field(default_factory=SimOptions)
    intercept_radius_m: float = 0.0
    max_thrust_n: float = 0.0
    max_rate_rps: float = 0.0
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    render_frames: bool = False
    render_camera_id: str | None = None
    render_endpoint: str = "tcp://127.0.0.1:47391"


@dataclass(frozen=True)
class SimInstance:
    seed: int
    pursuer_initial: PursuerInitialState
    targets: tuple[TargetConfig, ...]
    cameras: tuple[CameraConfig, ...] = ()
    config: SimConfig | None = None

    @property
    def target_initial(self) -> TargetInitialState:
        if not self.targets:
            raise AttributeError("SimInstance has no targets")
        target = self.targets[0]
        return TargetInitialState(
            position_w=target.initial.position_w,
            velocity_w=target.initial.velocity_w,
            radius_m=target.radius_m,
        )
