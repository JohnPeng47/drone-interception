from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from intercept_sim.types import ImageFeatureMeasurement, ObserverState, SceneSnapshot


@dataclass
class TruthRelativeFeatureObserver:
    latest_feature: ImageFeatureMeasurement | None = None
    relative_position_w: np.ndarray | None = None
    relative_velocity_w: np.ndarray | None = None
    target_acceleration_w: np.ndarray | None = None
    vehicle_rotation_wb: np.ndarray | None = None

    def update_scene(self, scene: SceneSnapshot) -> None:
        if not scene.targets:
            self.relative_position_w = None
            self.relative_velocity_w = None
            self.target_acceleration_w = None
            self.vehicle_rotation_wb = scene.pursuer.rotation_wb.copy()
            return

        target = scene.targets[0]
        self.relative_position_w = scene.pursuer.position_w - target.position_w
        self.relative_velocity_w = scene.pursuer.velocity_w - target.velocity_w
        self.target_acceleration_w = np.zeros(3, dtype=float)
        self.vehicle_rotation_wb = scene.pursuer.rotation_wb.copy()

    def predict(self, t: float, vehicle_state: dict[str, np.ndarray]) -> ObserverState:
        return ObserverState(
            t=float(t),
            vehicle_state=vehicle_state,
            image_feature=self.latest_feature,
            relative_position_w=None if self.relative_position_w is None else self.relative_position_w.copy(),
            relative_velocity_w=None if self.relative_velocity_w is None else self.relative_velocity_w.copy(),
            target_acceleration_w=None if self.target_acceleration_w is None else self.target_acceleration_w.copy(),
            vehicle_rotation_wb=None if self.vehicle_rotation_wb is None else self.vehicle_rotation_wb.copy(),
        )

    def update_image_feature(self, measurement: ImageFeatureMeasurement) -> None:
        self.latest_feature = measurement

    def reset(self) -> None:
        self.latest_feature = None
        self.relative_position_w = None
        self.relative_velocity_w = None
        self.target_acceleration_w = None
        self.vehicle_rotation_wb = None
