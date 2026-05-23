"""Client for a running Liftoff Unity render bridge."""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import struct
from typing import Any

import numpy as np


_HEADER = struct.Struct("<II")


class RenderUnavailableError(RuntimeError):
    """Raised when SimEngine asked for Liftoff frames but the bridge is unavailable."""


@dataclass(frozen=True)
class RenderFrameRequest:
    t: float
    camera_id: str
    vehicle_state: dict[str, np.ndarray]
    target_states: tuple[dict[str, Any], ...]
    camera_state: dict[str, Any]


@dataclass(frozen=True)
class RenderFrameResponse:
    width_px: int
    height_px: int
    channels: int
    rgb: np.ndarray


class LiftoffRenderEngine:
    """IPC client for Liftoff's real Unity camera/render stack.

    The server side is expected to run inside Liftoff and drive the existing
    first-person camera path before reading pixels from a RenderTexture.
    """

    def __init__(self, endpoint: str, *, timeout_s: float = 1.0):
        self.host, self.port = _parse_tcp_endpoint(endpoint)
        self.timeout_s = float(timeout_s)

    def render_frame(self, request: RenderFrameRequest) -> RenderFrameResponse:
        payload = json.dumps(_request_to_json(request), separators=(",", ":")).encode("utf-8")
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as sock:
                sock.settimeout(self.timeout_s)
                sock.sendall(_HEADER.pack(len(payload), 0))
                sock.sendall(payload)
                meta_len, frame_len = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
                frame_info = json.loads(_recv_exact(sock, meta_len).decode("utf-8"))
                frame = _recv_exact(sock, frame_len)
        except OSError as exc:
            raise RenderUnavailableError(
                f"Liftoff render bridge unavailable at tcp://{self.host}:{self.port}"
            ) from exc

        width = int(frame_info["width_px"])
        height = int(frame_info["height_px"])
        channels = int(frame_info.get("channels", 3))
        expected = width * height * channels
        if len(frame) != expected:
            raise RenderUnavailableError(
                f"Liftoff render bridge returned {len(frame)} bytes, expected {expected}"
            )
        rgb = np.frombuffer(frame, dtype=np.uint8).reshape((height, width, channels)).copy()
        return RenderFrameResponse(width, height, channels, rgb)


def _parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    prefix = "tcp://"
    if not endpoint.startswith(prefix):
        raise ValueError(f"Unsupported Liftoff render endpoint: {endpoint!r}")
    host_port = endpoint[len(prefix):]
    host, sep, port = host_port.rpartition(":")
    if not sep or not host:
        raise ValueError(f"Invalid Liftoff render endpoint: {endpoint!r}")
    return host, int(port)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise RenderUnavailableError("Liftoff render bridge closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def _request_to_json(request: RenderFrameRequest) -> dict[str, Any]:
    return {
        "t": float(request.t),
        "camera_id": request.camera_id,
        "vehicle_state": _state_to_json(request.vehicle_state),
        "targets": tuple(_target_to_json(target) for target in request.target_states),
        "camera": _camera_to_json(request.camera_state),
    }


def _state_to_json(state: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "position_w": np.asarray(state["x"], dtype=float).reshape(3).tolist(),
        "velocity_w": np.asarray(state["v"], dtype=float).reshape(3).tolist(),
        "quat_xyzw": np.asarray(state["q"], dtype=float).reshape(4).tolist(),
        "body_rates_b": np.asarray(state["w"], dtype=float).reshape(3).tolist(),
    }


def _target_to_json(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(target.get("id", target.get("c_id", ""))),
        "kind": str(target.get("kind", "target")),
        "position_w": np.asarray(target["position_w"], dtype=float).reshape(3).tolist(),
        "velocity_w": np.asarray(target["velocity_w"], dtype=float).reshape(3).tolist(),
        "radius_m": float(target.get("radius_m", 0.0)),
    }


def _camera_to_json(camera: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(camera.get("id", camera.get("c_id", ""))),
        "position_b": np.asarray(camera["position_b"], dtype=float).reshape(3).tolist(),
        "body_to_camera": np.asarray(camera["body_to_camera"], dtype=float).reshape(3, 3).tolist(),
        "width_px": int(camera["width_px"]),
        "height_px": int(camera["height_px"]),
        "fx_px": float(camera["fx_px"]),
        "fy_px": float(camera["fy_px"]),
        "cx_px": float(camera["cx_px"]),
        "cy_px": float(camera["cy_px"]),
        "hfov_rad": float(camera["hfov_rad"]),
        "vfov_rad": float(camera["vfov_rad"]),
    }
