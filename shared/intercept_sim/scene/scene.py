from __future__ import annotations

from intercept_sim.types import CameraRig, SceneSnapshot, SimulationTarget


def make_scene_snapshot(
    t: float,
    pursuer: SimulationTarget,
    targets: list[SimulationTarget] | tuple[SimulationTarget, ...],
    cameras: list[CameraRig] | tuple[CameraRig, ...],
) -> SceneSnapshot:
    return SceneSnapshot(
        t=float(t),
        pursuer=pursuer,
        targets=tuple(targets),
        cameras=tuple(cameras),
    )

