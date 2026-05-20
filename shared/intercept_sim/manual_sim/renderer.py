from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from intercept_sim.manual_sim.geometry import drone_segments, transform_point


@dataclass
class RendererConfig:
    ground_size_m: float
    arm_length_m: float


@dataclass(frozen=True)
class RenderFrame:
    t: float
    pursuer_state: dict[str, np.ndarray]
    target_positions_w: tuple[np.ndarray, ...] = ()
    pursuer_trail_w: tuple[np.ndarray, ...] = ()

    @classmethod
    def from_rotorpy_state(
        cls,
        t: float,
        state: dict[str, np.ndarray],
        *,
        target_positions_w: tuple[np.ndarray, ...] = (),
        pursuer_trail_w: tuple[np.ndarray, ...] = (),
    ) -> "RenderFrame":
        return cls(
            t=float(t),
            pursuer_state={key: np.asarray(value).copy() for key, value in state.items()},
            target_positions_w=tuple(np.asarray(p, dtype=float).copy() for p in target_positions_w),
            pursuer_trail_w=tuple(np.asarray(p, dtype=float).copy() for p in pursuer_trail_w),
        )


class BasicOpenGlRenderer:
    def __init__(self, config: RendererConfig) -> None:
        try:
            import pygame
            from OpenGL import GL, GLU
        except ImportError as exc:
            raise RuntimeError("Manual rendering requires pygame and PyOpenGL. Install intercept-sim[manual].") from exc
        self.pygame = pygame
        self.GL = GL
        self.GLU = GLU
        self.config = config

    def initialize(self, width: int = 1280, height: int = 720) -> None:
        pygame = self.pygame
        GL = self.GL
        GLU = self.GLU
        pygame.init()
        pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.05, 0.06, 0.07, 1.0)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(70.0, width / height, 0.05, 200.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def render(self, frame: RenderFrame) -> None:
        pygame = self.pygame
        GL = self.GL
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        position = np.asarray(frame.pursuer_state["x"], dtype=float)
        eye = position + np.array([-6.0, -6.0, 3.0], dtype=float)
        center = position
        self.GLU.gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], 0.0, 0.0, 1.0)
        self._draw_ground()
        self._draw_trail(frame.pursuer_trail_w)
        for target_position in frame.target_positions_w:
            self._draw_target(target_position)
        self._draw_drone(frame.pursuer_state)
        pygame.display.flip()

    def _draw_ground(self) -> None:
        GL = self.GL
        s = self.config.ground_size_m / 2.0
        GL.glColor3f(0.20, 0.22, 0.20)
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex3f(-s, -s, 0.0)
        GL.glVertex3f(s, -s, 0.0)
        GL.glVertex3f(s, s, 0.0)
        GL.glVertex3f(-s, s, 0.0)
        GL.glEnd()

    def _draw_target(self, position_w: np.ndarray) -> None:
        GL = self.GL
        p = np.asarray(position_w, dtype=float)
        r = 0.12
        GL.glColor3f(1.0, 0.15, 0.1)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(p[0] - r, p[1], p[2])
        GL.glVertex3f(p[0] + r, p[1], p[2])
        GL.glVertex3f(p[0], p[1] - r, p[2])
        GL.glVertex3f(p[0], p[1] + r, p[2])
        GL.glVertex3f(p[0], p[1], p[2] - r)
        GL.glVertex3f(p[0], p[1], p[2] + r)
        GL.glEnd()

    def _draw_trail(self, trail_w: tuple[np.ndarray, ...]) -> None:
        if len(trail_w) < 2:
            return
        GL = self.GL
        GL.glLineWidth(2.0)
        GL.glColor3f(0.2, 0.65, 1.0)
        GL.glBegin(GL.GL_LINE_STRIP)
        for point in trail_w:
            p = np.asarray(point, dtype=float)
            GL.glVertex3f(p[0], p[1], p[2])
        GL.glEnd()

    def _draw_drone(self, state: dict[str, np.ndarray]) -> None:
        GL = self.GL
        position = np.asarray(state["x"], dtype=float)
        rotation_wb = Rotation.from_quat(state["q"]).as_matrix()
        GL.glLineWidth(4.0)
        GL.glColor3f(0.95, 0.95, 0.90)
        GL.glBegin(GL.GL_LINES)
        for a_b, b_b in drone_segments(self.config.arm_length_m):
            a_w = transform_point(rotation_wb, position, a_b)
            b_w = transform_point(rotation_wb, position, b_b)
            GL.glVertex3f(a_w[0], a_w[1], a_w[2])
            GL.glVertex3f(b_w[0], b_w[1], b_w[2])
        GL.glEnd()
