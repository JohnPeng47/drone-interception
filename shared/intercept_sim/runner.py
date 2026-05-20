from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from intercept_sim.rotorpy_adapter import ctbr_to_rotorpy, hover_ctbr, rotorpy_state_to_target
from intercept_sim.scene import make_scene_snapshot
from intercept_sim.sensors import FeaturePerceptionModel, GeometryCamera
from intercept_sim.targets import KinematicTarget
from intercept_sim.types import CameraCapture, CtbrCommand, ImageFeatureMeasurement, ObserverState, SceneSnapshot


class FeatureObserver(Protocol):
    def predict(self, t: float, vehicle_state: dict[str, np.ndarray]) -> ObserverState: ...
    def update_image_feature(self, measurement: ImageFeatureMeasurement) -> None: ...
    def reset(self) -> None: ...


class FeatureController(Protocol):
    def update(self, t: float, observer_state: ObserverState) -> CtbrCommand: ...


@dataclass(frozen=True)
class RunnerStep:
    t: float
    rotorpy_state: dict[str, np.ndarray]
    scene: SceneSnapshot
    capture: CameraCapture | None
    measurements: tuple[ImageFeatureMeasurement, ...]
    observer_state: ObserverState
    command: CtbrCommand


@dataclass
class InterceptionRunner:
    vehicle: object
    imu: object
    target: KinematicTarget
    camera: GeometryCamera
    perception: FeaturePerceptionModel
    observer: FeatureObserver
    controller: FeatureController
    dt: float
    initial_state: dict[str, np.ndarray]
    log: list[RunnerStep] = field(default_factory=list, init=False)

    def run(self, t_final: float) -> list[RunnerStep]:
        state = {key: np.asarray(value, dtype=float).copy() for key, value in self.initial_state.items()}
        command = hover_ctbr(0.0, self.vehicle.mass)
        self.log.clear()
        self.camera.reset()
        self.perception.reset()
        self.observer.reset()
        controller_reset = getattr(self.controller, "reset", None)
        if controller_reset is not None:
            controller_reset()

        num_steps = int(np.ceil(t_final / self.dt))
        for step in range(num_steps):
            t = step * self.dt
            pursuer = rotorpy_state_to_target(state)
            target_state = self.target.state_at(t)
            scene = make_scene_snapshot(t, pursuer, [target_state], [self.camera.rig])
            update_scene = getattr(self.observer, "update_scene", None)
            if update_scene is not None:
                update_scene(scene)
            capture = self.camera.maybe_capture(scene)
            if capture is not None:
                self.perception.submit_capture(capture)
            measurements = tuple(self.perception.pop_available(t))
            for measurement in measurements:
                self.observer.update_image_feature(measurement)

            observer_state = self.observer.predict(t, state)
            command = self.controller.update(t, observer_state)
            state = self.vehicle.step(state, ctbr_to_rotorpy(command), self.dt)
            self.log.append(
                RunnerStep(
                    t=t,
                    rotorpy_state={key: np.asarray(value).copy() for key, value in state.items()},
                    scene=scene,
                    capture=capture,
                    measurements=measurements,
                    observer_state=observer_state,
                    command=command,
                )
            )
        return self.log
