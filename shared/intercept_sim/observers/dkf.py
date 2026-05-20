from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from intercept_sim.types import ImageFeatureMeasurement, ObserverState


@dataclass
class DelayedFeatureReplayObserver:
    history_size: int = 16
    latest_feature: ImageFeatureMeasurement | None = None
    _history: list[ImageFeatureMeasurement] = field(default_factory=list, init=False, repr=False)

    def predict(self, t: float, vehicle_state: dict[str, np.ndarray]) -> ObserverState:
        valid_history = [m for m in self._history if m.detected and m.uv_norm is not None and m.t_capture <= t + 1e-12]
        if not valid_history:
            return ObserverState(t=float(t), vehicle_state=vehicle_state, image_feature=self.latest_feature)

        if len(valid_history) == 1:
            prediction = valid_history[-1]
        else:
            prev, curr = valid_history[-2], valid_history[-1]
            dt_capture = curr.t_capture - prev.t_capture
            if dt_capture <= 1e-12:
                prediction = curr
            else:
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
        self._history.append(measurement)
        self._history.sort(key=lambda m: m.t_capture)
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size :]

    def reset(self) -> None:
        self.latest_feature = None
        self._history.clear()
