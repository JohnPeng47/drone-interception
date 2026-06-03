from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..types import CameraCapture, ImageFeatureMeasurement


@dataclass
class FeaturePerceptionModel:
    camera_image_delay_s: float
    pixel_noise_std_px: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    pixel_to_norm: np.ndarray | None = None
    dropout_probability: float = 0.0
    default_confidence: float = 1.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    _pending: deque[ImageFeatureMeasurement] = field(default_factory=deque, init=False, repr=False)

    def submit_capture(self, capture: CameraCapture) -> None:
        detected = capture.detected and self.rng.random() >= self.dropout_probability
        uv_px = None
        uv_norm = None
        confidence = 0.0

        if detected and capture.uv_px is not None and capture.uv_norm is not None:
            noise_px = self.rng.normal(0.0, self.pixel_noise_std_px, size=2)
            uv_px = capture.uv_px + noise_px
            if self.pixel_to_norm is not None:
                uv_norm = capture.uv_norm + noise_px * np.asarray(self.pixel_to_norm, dtype=float)
            else:
                uv_norm = capture.uv_norm.copy()
            confidence = self.default_confidence

        self._pending.append(
            ImageFeatureMeasurement(
                t_capture=capture.t_capture,
                t_available=capture.t_capture + self.camera_image_delay_s,
                camera_id=capture.camera_id,
                target_id=capture.target_id,
                detected=detected,
                uv_px=uv_px,
                uv_norm=uv_norm,
                confidence=confidence,
            )
        )

    def pop_available(self, t_now: float) -> list[ImageFeatureMeasurement]:
        out: list[ImageFeatureMeasurement] = []
        while self._pending and self._pending[0].t_available <= t_now + 1e-12:
            out.append(self._pending.popleft())
        return out

    def reset(self) -> None:
        self._pending.clear()
