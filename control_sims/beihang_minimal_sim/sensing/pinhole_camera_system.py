"""Deterministic pinhole projection from scene state to image feature."""

from __future__ import annotations

import numpy as np
from pydrake.systems.framework import LeafSystem

from ..config import CameraConfig
from ..drake_values import image_feature_value, scene_state_value
from ..types import ImageFeature


class PinholeCameraSystem(LeafSystem):
    def __init__(self, config: CameraConfig):
        super().__init__()
        self._R_b2c = np.asarray(config.body_to_camera, dtype=float)
        self._max_uv = float(config.max_uv_norm)
        self._min_depth = float(config.min_depth_m)
        self.DeclareAbstractInputPort("scene", scene_state_value())
        self.DeclareAbstractOutputPort(
            "image_feature",
            image_feature_value,
            self._calc,
            prerequisites_of_calc={self.time_ticket()},
        )

    def _calc(self, context, output) -> None:
        scene = self.GetInputPort("scene").Eval(context)
        rel_w = scene.target.position_w - scene.vehicle.position_w
        rel_b = scene.vehicle.rotation_wb.T @ rel_w
        rel_c = self._R_b2c @ rel_b
        depth = float(rel_c[0])
        if depth <= self._min_depth:
            output.set_value(
                ImageFeature(
                    t=float(context.get_time()),
                    detected=False,
                    uv_norm=None,
                    depth_m=depth,
                    bearing_c=None,
                )
            )
            return

        uv = np.array([rel_c[1] / depth, rel_c[2] / depth], dtype=float)
        detected = bool(np.all(np.abs(uv) <= self._max_uv))
        bearing = rel_c / max(float(np.linalg.norm(rel_c)), 1e-12)
        output.set_value(
            ImageFeature(
                t=float(context.get_time()),
                detected=detected,
                uv_norm=uv if detected else None,
                depth_m=depth,
                bearing_c=bearing if detected else None,
            )
        )

