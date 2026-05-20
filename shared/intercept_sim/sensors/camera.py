from __future__ import annotations

from dataclasses import dataclass, field

from intercept_sim.scene.visibility import project_target
from intercept_sim.types import CameraCapture, CameraRig, SceneSnapshot


@dataclass
class GeometryCamera:
    rig: CameraRig
    _next_capture_t: float = field(default=0.0, init=False, repr=False)

    @property
    def period_s(self) -> float:
        return 1.0 / self.rig.capture_rate_hz

    def maybe_capture(self, scene: SceneSnapshot) -> CameraCapture | None:
        if scene.t + 1e-12 < self._next_capture_t:
            return None
        self._next_capture_t += self.period_s
        if not scene.targets:
            return CameraCapture(
                t_capture=scene.t,
                camera_id=self.rig.id,
                target_id=None,
                detected=False,
                uv_px=None,
                uv_norm=None,
                target_pos_c=None,
                range_m=None,
                apparent_radius_px=None,
            )
        # Single-target first pass: choose the first visible target candidate.
        return project_target(scene, self.rig, scene.targets[0])

    def reset(self) -> None:
        self._next_capture_t = 0.0

