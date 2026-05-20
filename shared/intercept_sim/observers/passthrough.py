from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np

from intercept_sim.types import ImageFeatureMeasurement, ObserverState


@dataclass
class LatestFeatureObserver:
    latest_feature: ImageFeatureMeasurement | None = None

    def predict(self, t: float, vehicle_state: dict[str, np.ndarray]) -> ObserverState:
        return ObserverState(t=float(t), vehicle_state=vehicle_state, image_feature=self.latest_feature)

    def update_image_feature(self, measurement: ImageFeatureMeasurement) -> None:
        self.latest_feature = measurement

    def reset(self) -> None:
        self.latest_feature = None


@dataclass
class ConstantVelocityFeatureObserver:
    history_size: int = 4
    latest_feature: ImageFeatureMeasurement | None = None
    _history: deque[ImageFeatureMeasurement] | None = None

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.history_size)

    def predict(self, t: float, vehicle_state: dict[str, np.ndarray]) -> ObserverState:
        if self.latest_feature is None or not self.latest_feature.detected or self.latest_feature.uv_norm is None:
            return ObserverState(t=float(t), vehicle_state=vehicle_state, image_feature=self.latest_feature)

        prediction = self.latest_feature
        history = [m for m in (self._history or []) if m.detected and m.uv_norm is not None]
        if len(history) >= 2:
            prev = history[-2]
            curr = history[-1]
            dt_capture = curr.t_capture - prev.t_capture
            if dt_capture > 1e-9:
                uv_rate = (curr.uv_norm - prev.uv_norm) / dt_capture
                uv_pred = curr.uv_norm + uv_rate * (float(t) - curr.t_capture)
                prediction = ImageFeatureMeasurement(
                    t_capture=curr.t_capture,
                    t_available=float(t),
                    camera_id=curr.camera_id,
                    target_id=curr.target_id,
                    detected=True,
                    uv_px=None,
                    uv_norm=uv_pred,
                    confidence=curr.confidence,
                )

        return ObserverState(t=float(t), vehicle_state=vehicle_state, image_feature=prediction)

    def update_image_feature(self, measurement: ImageFeatureMeasurement) -> None:
        self.latest_feature = measurement
        if self._history is None:
            self._history = deque(maxlen=self.history_size)
        self._history.append(measurement)

    def reset(self) -> None:
        self.latest_feature = None
        if self._history is not None:
            self._history.clear()
