from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import LIFTOFF_RENDER_OK, ctypes_config, status_name
from .ctypes_api import (
    LiftoffRenderCameraState,
    LiftoffRenderDroneState,
    LiftoffRenderFrame,
    LiftoffRenderFrameRequest,
    LiftoffRenderQuatXyzw,
    LiftoffRenderTargetState,
    LiftoffRenderVec3,
    load_library,
)


class RenderError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)


@dataclass(frozen=True)
class RenderFrameResult:
    status: int
    status_name: str
    sequence_id: int
    width_px: int
    height_px: int
    channels: int
    stride_bytes: int
    pixels: bytes | None

    @property
    def has_frame(self) -> bool:
        return self.status == LIFTOFF_RENDER_OK and self.pixels is not None


class NativeRenderEngine:
    def __init__(self, config: Any, library_path: str | None = None):
        self._lib = load_library(library_path)
        self._engine = C.c_void_p()
        self._config = config
        status = int(
            self._lib.liftoff_render_engine_create(
                C.byref(ctypes_config(config)),
                C.byref(self._engine),
            )
        )
        if status != LIFTOFF_RENDER_OK:
            raise RenderError(status, self.status_message(status))

    def close(self) -> None:
        if self._engine:
            self._lib.liftoff_render_engine_destroy(self._engine)
            self._engine = C.c_void_p()

    def __enter__(self) -> NativeRenderEngine:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def status_message(self, status: int) -> str:
        raw = self._lib.liftoff_render_status_string(int(status))
        return raw.decode("utf-8") if raw else status_name(status)

    def render_frame(
        self,
        *,
        drone: dict[str, Any],
        camera: dict[str, Any],
        targets: tuple[dict[str, Any], ...] | list[dict[str, Any]],
        sequence_id: int,
    ) -> RenderFrameResult:
        if not self._engine:
            raise RenderError(-1, "render engine is closed")

        c_drone = _drone_state(drone, sequence_id)
        c_camera = _camera_state(camera)
        target_values = [_target_state(target, i) for i, target in enumerate(targets)]
        target_arr = (LiftoffRenderTargetState * len(target_values))(*target_values)
        request = LiftoffRenderFrameRequest(
            C.pointer(c_drone),
            C.pointer(c_camera),
            target_arr if target_values else None,
            len(target_values),
        )
        frame = LiftoffRenderFrame()
        status = int(self._lib.liftoff_render_frame(self._engine, C.byref(request), C.byref(frame)))
        pixels = None
        if status == LIFTOFF_RENDER_OK and frame.pixels and frame.pixel_bytes:
            pixels = C.string_at(frame.pixels, int(frame.pixel_bytes))
        result = RenderFrameResult(
            status=status,
            status_name=status_name(status),
            sequence_id=int(frame.sequence_id),
            width_px=int(frame.width_px),
            height_px=int(frame.height_px),
            channels=int(frame.channels),
            stride_bytes=int(frame.stride_bytes),
            pixels=pixels,
        )
        self._lib.liftoff_render_release_frame(self._engine, C.byref(frame))
        return result


def _drone_state(drone: dict[str, Any], sequence_id: int) -> LiftoffRenderDroneState:
    quat = np.asarray(drone["q"], dtype=float).reshape(4)
    return LiftoffRenderDroneState(
        float(drone.get("t", 0.0)),
        int(sequence_id),
        _vec3(drone["x"]),
        _vec3(drone["v"]),
        LiftoffRenderQuatXyzw(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        _vec3(drone["w"]),
    )


def _camera_state(camera: dict[str, Any]) -> LiftoffRenderCameraState:
    body_to_camera = np.asarray(camera["body_to_camera"], dtype=float).reshape(3, 3)
    return LiftoffRenderCameraState(
        int(camera["c_id"]),
        _vec3(camera["position_b"]),
        (C.c_double * 9)(*body_to_camera.reshape(9).tolist()),
        int(camera["width_px"]),
        int(camera["height_px"]),
        float(camera["fx_px"]),
        float(camera["fy_px"]),
        float(camera["cx_px"]),
        float(camera["cy_px"]),
        float(camera["hfov_rad"]),
        float(camera["vfov_rad"]),
    )


def _target_state(target: dict[str, Any], fallback_id: int) -> LiftoffRenderTargetState:
    return LiftoffRenderTargetState(
        int(target.get("c_id", fallback_id)),
        _vec3(target["position_w"]),
        _vec3(target["velocity_w"]),
        float(target.get("radius_m", 0.0)),
    )


def _vec3(value: Any) -> LiftoffRenderVec3:
    arr = np.asarray(value, dtype=float).reshape(3)
    return LiftoffRenderVec3(float(arr[0]), float(arr[1]), float(arr[2]))
