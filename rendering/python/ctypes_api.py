from __future__ import annotations

import ctypes as C
from pathlib import Path

from .build_native import build_native


class LiftoffRenderConfig(C.Structure):
    _fields_ = [
        ("backend", C.c_int),
        ("platform", C.c_int),
        ("timeout_ms", C.c_uint32),
        ("flags", C.c_uint32),
        ("scene_id", C.c_char * 256),
    ]


class LiftoffRenderVec3(C.Structure):
    _fields_ = [("x", C.c_double), ("y", C.c_double), ("z", C.c_double)]


class LiftoffRenderQuatXyzw(C.Structure):
    _fields_ = [("x", C.c_double), ("y", C.c_double), ("z", C.c_double), ("w", C.c_double)]


class LiftoffRenderDroneState(C.Structure):
    _fields_ = [
        ("t", C.c_double),
        ("sequence_id", C.c_uint64),
        ("position_w", LiftoffRenderVec3),
        ("velocity_w", LiftoffRenderVec3),
        ("quat_xyzw", LiftoffRenderQuatXyzw),
        ("body_rates_b", LiftoffRenderVec3),
    ]


class LiftoffRenderCameraState(C.Structure):
    _fields_ = [
        ("camera_id", C.c_uint32),
        ("position_b", LiftoffRenderVec3),
        ("body_to_camera", C.c_double * 9),
        ("width_px", C.c_uint32),
        ("height_px", C.c_uint32),
        ("fx_px", C.c_double),
        ("fy_px", C.c_double),
        ("cx_px", C.c_double),
        ("cy_px", C.c_double),
        ("hfov_rad", C.c_double),
        ("vfov_rad", C.c_double),
    ]


class LiftoffRenderTargetState(C.Structure):
    _fields_ = [
        ("target_id", C.c_uint32),
        ("position_w", LiftoffRenderVec3),
        ("velocity_w", LiftoffRenderVec3),
        ("radius_m", C.c_double),
    ]


class LiftoffRenderFrameRequest(C.Structure):
    _fields_ = [
        ("drone", C.POINTER(LiftoffRenderDroneState)),
        ("camera", C.POINTER(LiftoffRenderCameraState)),
        ("targets", C.POINTER(LiftoffRenderTargetState)),
        ("target_count", C.c_uint32),
    ]


class LiftoffRenderFrame(C.Structure):
    _fields_ = [
        ("sequence_id", C.c_uint64),
        ("width_px", C.c_uint32),
        ("height_px", C.c_uint32),
        ("channels", C.c_uint32),
        ("stride_bytes", C.c_uint32),
        ("pixels", C.POINTER(C.c_uint8)),
        ("pixel_bytes", C.c_size_t),
    ]


def load_library(path: str | Path | None = None) -> C.CDLL:
    lib_path = Path(path) if path is not None else build_native()
    lib = C.CDLL(str(lib_path))
    lib.liftoff_render_engine_create.argtypes = [
        C.POINTER(LiftoffRenderConfig),
        C.POINTER(C.c_void_p),
    ]
    lib.liftoff_render_engine_create.restype = C.c_int
    lib.liftoff_render_engine_destroy.argtypes = [C.c_void_p]
    lib.liftoff_render_engine_destroy.restype = None
    lib.liftoff_render_frame.argtypes = [
        C.c_void_p,
        C.POINTER(LiftoffRenderFrameRequest),
        C.POINTER(LiftoffRenderFrame),
    ]
    lib.liftoff_render_frame.restype = C.c_int
    lib.liftoff_render_release_frame.argtypes = [C.c_void_p, C.POINTER(LiftoffRenderFrame)]
    lib.liftoff_render_release_frame.restype = None
    lib.liftoff_render_status_string.argtypes = [C.c_int]
    lib.liftoff_render_status_string.restype = C.c_char_p
    return lib
