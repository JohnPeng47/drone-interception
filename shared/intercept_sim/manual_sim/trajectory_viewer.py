from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from intercept_sim.manual_sim.renderer import BasicOpenGlRenderer, RenderFrame
from intercept_sim.runner import RunnerStep


@dataclass(frozen=True)
class PlaybackConfig:
    trail_length: int = 200


def frame_from_runner_step(
    step: RunnerStep,
    history: Sequence[RunnerStep],
    config: PlaybackConfig | None = None,
) -> RenderFrame:
    cfg = PlaybackConfig() if config is None else config
    trail_steps = history[-cfg.trail_length :]
    pursuer_trail = tuple(np.asarray(item.rotorpy_state["x"], dtype=float).copy() for item in trail_steps)
    target_positions = tuple(target.position_w.copy() for target in step.scene.targets)
    return RenderFrame.from_rotorpy_state(
        step.t,
        step.rotorpy_state,
        target_positions_w=target_positions,
        pursuer_trail_w=pursuer_trail,
    )


def playback_runner_log(
    renderer: BasicOpenGlRenderer,
    log: Sequence[RunnerStep],
    *,
    playback_hz: float = 60.0,
    config: PlaybackConfig | None = None,
) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("Trajectory playback requires pygame. Install intercept-sim[manual].") from exc

    clock = pygame.time.Clock()
    history: list[RunnerStep] = []
    for step in log:
        history.append(step)
        renderer.render(frame_from_runner_step(step, history, config))
        clock.tick(playback_hz)
