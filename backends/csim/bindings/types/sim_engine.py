from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator, Sequence

import numpy as np

from .camera_sim import CameraConfig, CameraObservation
from .sim_types import PUFFER_ACTION_SUBSTEPS, PUFFER_DT, PursuerInitialState, PursuerParams, PursuerState
from .target_sim import TargetConfig, TargetInitialState, TargetState


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
    bounds_w: tuple[float, float, float] | None = None
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    rendering: bool = False
    render: RenderConfig = field(default_factory=RenderConfig)


@dataclass(frozen=True)
class SimInstance:
    seed: int
    pursuer_initial: PursuerInitialState
    target_initials: tuple[TargetInitialState, ...]
    config: SimConfig | None = None

    @property
    def target_initial(self) -> TargetInitialState:
        if not self.target_initials:
            raise AttributeError("SimInstance has no target_initials")
        return self.target_initials[0]


@dataclass(frozen=True)
class InterceptMetrics:
    distance_m: float
    min_distance_m: float
    intercepted: bool
    intercept_time_s: float
    target_index: int


@dataclass(frozen=True)
class SimSnapshot:
    """Typed snapshot for one SimEngine slot."""

    pursuer: PursuerState
    target: TargetState
    metrics: InterceptMetrics
    camera: CameraObservation
    max_rate_rps: float
    max_rpm: float
    body_rates_b: np.ndarray | None = None
    thrust_n: float | None = None

    @classmethod
    def from_arrays(
        cls,
        pursuer: Sequence[float] | np.ndarray,
        target: Sequence[float] | np.ndarray,
        metrics: Sequence[float] | np.ndarray,
        camera: Sequence[float] | np.ndarray,
        *,
        max_rate_rps: float,
        max_rpm: float,
        body_rates_b: Sequence[float] | np.ndarray | None = None,
        thrust_n: float | None = None,
    ) -> "SimSnapshot":
        pursuer_arr = np.asarray(pursuer, dtype=np.float32).reshape(17)
        target_arr = np.asarray(target, dtype=np.float32).reshape(6)
        metrics_arr = np.asarray(metrics, dtype=np.float32).reshape(5)
        camera_arr = np.asarray(camera, dtype=np.float32).reshape(3)
        return cls(
            pursuer=PursuerState(
                position_w=pursuer_arr[0:3].copy(),
                velocity_w=pursuer_arr[3:6].copy(),
                quat_xyzw=pursuer_arr[6:10].copy(),
                body_rates_b=pursuer_arr[10:13].copy(),
                rotor_speeds=pursuer_arr[13:17].copy(),
            ),
            target=TargetState(
                position_w=target_arr[0:3].copy(),
                velocity_w=target_arr[3:6].copy(),
            ),
            metrics=InterceptMetrics(
                distance_m=float(metrics_arr[0]),
                min_distance_m=float(metrics_arr[1]),
                intercepted=bool(metrics_arr[2] > 0.5),
                intercept_time_s=float(metrics_arr[3]),
                target_index=int(metrics_arr[4]),
            ),
            camera=CameraObservation(
                detected=bool(camera_arr[0] > 0.5),
                uv_norm=camera_arr[1:3].copy(),
            ),
            max_rate_rps=float(max_rate_rps),
            max_rpm=float(max_rpm),
            body_rates_b=(
                None
                if body_rates_b is None
                else np.asarray(body_rates_b, dtype=np.float32).reshape(3).copy()
            ),
            thrust_n=None if thrust_n is None else float(thrust_n),
        )

    def with_command(self, thrust_n: float, body_rates_b: Sequence[float] | np.ndarray) -> "SimSnapshot":
        return SimSnapshot(
            pursuer=self.pursuer,
            target=self.target,
            metrics=self.metrics,
            camera=self.camera,
            max_rate_rps=self.max_rate_rps,
            max_rpm=self.max_rpm,
            body_rates_b=np.asarray(body_rates_b, dtype=np.float32).reshape(3).copy(),
            thrust_n=float(thrust_n),
        )


@dataclass(frozen=True)
class SimSnapshotArrays:
    """Vectorized batch view for high-throughput consumers."""

    pursuer: np.ndarray
    target: np.ndarray
    metrics: np.ndarray
    camera: np.ndarray
    max_rate_rps: np.ndarray
    max_rpm: np.ndarray
    body_rates_b: np.ndarray | None = None
    thrust_n: np.ndarray | None = None

    @classmethod
    def from_arrays(
        cls,
        pursuer: np.ndarray,
        target: np.ndarray,
        metrics: np.ndarray,
        camera: np.ndarray,
        max_rate_rps: np.ndarray,
        max_rpm: np.ndarray,
        *,
        body_rates_b: np.ndarray | None = None,
        thrust_n: np.ndarray | None = None,
    ) -> "SimSnapshotArrays":
        pursuer_arr = np.asarray(pursuer, dtype=np.float32).reshape(-1, 17).copy()
        count = int(pursuer_arr.shape[0])
        target_arr = np.asarray(target, dtype=np.float32).reshape(count, 6).copy()
        metrics_arr = np.asarray(metrics, dtype=np.float32).reshape(count, 5).copy()
        camera_arr = np.asarray(camera, dtype=np.float32).reshape(count, 3).copy()
        max_rate_arr = np.asarray(max_rate_rps, dtype=np.float32).reshape(count).copy()
        max_rpm_arr = np.asarray(max_rpm, dtype=np.float32).reshape(count).copy()
        rates_arr = (
            None
            if body_rates_b is None
            else np.asarray(body_rates_b, dtype=np.float32).reshape(count, 3).copy()
        )
        thrust_arr = (
            None
            if thrust_n is None
            else np.asarray(thrust_n, dtype=np.float32).reshape(count).copy()
        )
        return cls(
            pursuer=pursuer_arr,
            target=target_arr,
            metrics=metrics_arr,
            camera=camera_arr,
            max_rate_rps=max_rate_arr,
            max_rpm=max_rpm_arr,
            body_rates_b=rates_arr,
            thrust_n=thrust_arr,
        )

    def with_commands(self, thrust_n: np.ndarray, body_rates_b: np.ndarray) -> "SimSnapshotArrays":
        return SimSnapshotArrays(
            pursuer=self.pursuer,
            target=self.target,
            metrics=self.metrics,
            camera=self.camera,
            max_rate_rps=self.max_rate_rps,
            max_rpm=self.max_rpm,
            body_rates_b=np.asarray(body_rates_b, dtype=np.float32).reshape(len(self), 3).copy(),
            thrust_n=np.asarray(thrust_n, dtype=np.float32).reshape(len(self)).copy(),
        )

    def __len__(self) -> int:
        return int(self.pursuer.shape[0])


@dataclass(frozen=True)
class SimSnapshots:
    """Typed snapshots for a fixed-width batch of SimEngine slots."""

    arrays: SimSnapshotArrays

    @classmethod
    def from_arrays(
        cls,
        pursuer: np.ndarray,
        target: np.ndarray,
        metrics: np.ndarray,
        camera: np.ndarray,
        max_rate_rps: np.ndarray,
        max_rpm: np.ndarray,
        *,
        body_rates_b: np.ndarray | None = None,
        thrust_n: np.ndarray | None = None,
    ) -> "SimSnapshots":
        arrays = SimSnapshotArrays.from_arrays(
            pursuer,
            target,
            metrics,
            camera,
            max_rate_rps,
            max_rpm,
            body_rates_b=body_rates_b,
            thrust_n=thrust_n,
        )
        return cls(arrays)

    def with_commands(self, thrust_n: np.ndarray, body_rates_b: np.ndarray) -> "SimSnapshots":
        thrust = np.asarray(thrust_n, dtype=np.float32).reshape(len(self))
        body_rates = np.asarray(body_rates_b, dtype=np.float32).reshape(len(self), 3)
        return SimSnapshots(self.arrays.with_commands(thrust, body_rates))

    def __len__(self) -> int:
        return len(self.arrays)

    def __iter__(self) -> Iterator[SimSnapshot]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> SimSnapshot:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return SimSnapshot.from_arrays(
            self.arrays.pursuer[index],
            self.arrays.target[index],
            self.arrays.metrics[index],
            self.arrays.camera[index],
            max_rate_rps=float(self.arrays.max_rate_rps[index]),
            max_rpm=float(self.arrays.max_rpm[index]),
            body_rates_b=None if self.arrays.body_rates_b is None else self.arrays.body_rates_b[index],
            thrust_n=None if self.arrays.thrust_n is None else float(self.arrays.thrust_n[index]),
        )
