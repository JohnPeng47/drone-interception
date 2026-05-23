from __future__ import annotations

from typing import Any

from .ctypes_api import LiftoffRenderConfig


LIFTOFF_RENDER_BACKEND_NONE = 0
LIFTOFF_RENDER_BACKEND_UNITY = 1
LIFTOFF_RENDER_BACKEND_SOFTWARE = 2

LIFTOFF_RENDER_PLATFORM_AUTO = 0
LIFTOFF_RENDER_PLATFORM_WINDOWS = 1
LIFTOFF_RENDER_PLATFORM_LINUX = 2

LIFTOFF_RENDER_OK = 0
LIFTOFF_RENDER_DISABLED = 1
LIFTOFF_RENDER_BACKEND_UNAVAILABLE = 2
LIFTOFF_RENDER_TIMEOUT = 3
LIFTOFF_RENDER_INVALID_REQUEST = 4
LIFTOFF_RENDER_FRAME_DROPPED = 5
LIFTOFF_RENDER_INTERNAL_ERROR = 6

_BACKENDS = {
    "none": LIFTOFF_RENDER_BACKEND_NONE,
    "disabled": LIFTOFF_RENDER_BACKEND_NONE,
    "unity": LIFTOFF_RENDER_BACKEND_UNITY,
    "software": LIFTOFF_RENDER_BACKEND_SOFTWARE,
}
_PLATFORMS = {
    "auto": LIFTOFF_RENDER_PLATFORM_AUTO,
    "windows": LIFTOFF_RENDER_PLATFORM_WINDOWS,
    "win32": LIFTOFF_RENDER_PLATFORM_WINDOWS,
    "linux": LIFTOFF_RENDER_PLATFORM_LINUX,
}
_STATUS_NAMES = {
    LIFTOFF_RENDER_OK: "ok",
    LIFTOFF_RENDER_DISABLED: "disabled",
    LIFTOFF_RENDER_BACKEND_UNAVAILABLE: "backend_unavailable",
    LIFTOFF_RENDER_TIMEOUT: "timeout",
    LIFTOFF_RENDER_INVALID_REQUEST: "invalid_request",
    LIFTOFF_RENDER_FRAME_DROPPED: "frame_dropped",
    LIFTOFF_RENDER_INTERNAL_ERROR: "internal_error",
}


def backend_kind(value: str | int) -> int:
    if isinstance(value, int):
        return value
    key = str(value).strip().lower()
    if key not in _BACKENDS:
        raise ValueError(f"Unsupported render backend: {value}")
    return _BACKENDS[key]


def platform_kind(value: str | int) -> int:
    if isinstance(value, int):
        return value
    key = str(value).strip().lower()
    if key not in _PLATFORMS:
        raise ValueError(f"Unsupported render platform: {value}")
    return _PLATFORMS[key]


def status_name(status: int) -> str:
    return _STATUS_NAMES.get(int(status), "unknown")


def ctypes_config(config: Any) -> LiftoffRenderConfig:
    scene_id = str(_get(config, "scene_id", "liftoff_fpv_0")).encode("utf-8")
    if len(scene_id) >= 256:
        raise ValueError("Render scene_id must encode to fewer than 256 bytes")
    c_scene_id = (scene_id + b"\0" * 256)[:256]
    return LiftoffRenderConfig(
        backend_kind(_get(config, "backend", "software")),
        platform_kind(_get(config, "platform", "auto")),
        int(_get(config, "timeout_ms", 16)),
        0,
        c_scene_id,
    )


def _get(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
