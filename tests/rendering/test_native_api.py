from __future__ import annotations

import numpy as np

from backends import RenderConfig
from rendering.python import (
    LIFTOFF_RENDER_BACKEND_UNAVAILABLE,
    LIFTOFF_RENDER_DISABLED,
    LIFTOFF_RENDER_OK,
    NativeRenderEngine,
)


def test_native_backend_none_returns_disabled():
    with NativeRenderEngine(RenderConfig(enabled=True, backend="none")) as renderer:
        result = renderer.render_frame(
            drone=_drone(),
            camera=_camera(),
            targets=(_target(),),
            sequence_id=1,
        )

    assert result.status == LIFTOFF_RENDER_DISABLED
    assert result.status_name == "disabled"
    assert result.has_frame is False
    assert result.pixels is None


def test_native_unity_stub_returns_backend_unavailable():
    with NativeRenderEngine(RenderConfig(enabled=True, backend="unity")) as renderer:
        result = renderer.render_frame(
            drone=_drone(),
            camera=_camera(),
            targets=(_target(),),
            sequence_id=2,
        )

    assert result.status == LIFTOFF_RENDER_BACKEND_UNAVAILABLE
    assert result.status_name == "backend_unavailable"
    assert result.has_frame is False
    assert result.pixels is None


def test_native_software_backend_returns_rgb_frame(monkeypatch, tmp_path):
    monkeypatch.setenv("LIFTOFF_RENDER_DRONE_MESH", str(tmp_path / "missing_target_drone.obj"))
    with NativeRenderEngine(RenderConfig(enabled=True, backend="software")) as renderer:
        result = renderer.render_frame(
            drone=_drone(),
            camera=_camera(),
            targets=(_target(),),
            sequence_id=3,
        )

    assert result.status == LIFTOFF_RENDER_OK
    assert result.status_name == "ok"
    assert result.has_frame is True
    assert result.width_px == 640
    assert result.height_px == 480
    assert result.channels == 3
    assert result.stride_bytes == 640 * 3
    assert result.pixels is not None
    assert len(result.pixels) == 640 * 480 * 3
    frame = np.frombuffer(result.pixels, dtype=np.uint8).reshape((480, 640, 3))
    assert int(frame.max()) > int(frame.min())
    assert np.unique(frame.reshape((-1, 3)), axis=0).shape[0] > 64
    red_marker_pixels = np.sum(
        (frame[:, :, 0] > 180) & (frame[:, :, 1] > 45) & (frame[:, :, 1] < 130) & (frame[:, :, 2] < 90)
    )
    cyan_marker_pixels = np.sum(
        (frame[:, :, 0] > 45) & (frame[:, :, 0] < 130) & (frame[:, :, 1] > 130) & (frame[:, :, 2] > 150)
    )
    dark_airframe_pixels = np.sum(
        (frame[:, :, 0] > 20) & (frame[:, :, 0] < 120) &
        (frame[:, :, 1] > 25) & (frame[:, :, 1] < 130) &
        (frame[:, :, 2] > 25) & (frame[:, :, 2] < 135)
    )
    assert red_marker_pixels > 0
    assert cyan_marker_pixels > 0
    assert dark_airframe_pixels > 128


def test_native_software_backend_loads_target_mesh(monkeypatch, tmp_path):
    mesh_path = tmp_path / "target_drone.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "o body",
                "usemtl vortex_frame",
                "v 0.06 -0.09 -0.015",
                "v 0.06 0.09 -0.015",
                "v -0.08 0.09 -0.015",
                "v -0.08 -0.09 -0.015",
                "v 0.06 -0.09 0.015",
                "v 0.06 0.09 0.015",
                "v -0.08 0.09 0.015",
                "v -0.08 -0.09 0.015",
                "f 1 2 3",
                "f 1 3 4",
                "f 5 7 6",
                "f 5 8 7",
                "usemtl prop",
                "v 0.08 -0.14 0.01",
                "v 0.08 -0.04 0.01",
                "v -0.08 -0.14 0.01",
                "v -0.08 -0.04 0.01",
                "v 0.08 0.04 0.01",
                "v 0.08 0.14 0.01",
                "v -0.08 0.04 0.01",
                "v -0.08 0.14 0.01",
                "f 9 10 11",
                "f 10 12 11",
                "f 13 14 15",
                "f 14 16 15",
            ]
        ),
        encoding="ascii",
    )
    monkeypatch.setenv("LIFTOFF_RENDER_DRONE_MESH", str(mesh_path))

    with NativeRenderEngine(RenderConfig(enabled=True, backend="software")) as renderer:
        result = renderer.render_frame(
            drone=_drone(),
            camera=_camera(),
            targets=(_target(),),
            sequence_id=4,
        )

    assert result.status == LIFTOFF_RENDER_OK
    assert result.pixels is not None
    frame = np.frombuffer(result.pixels, dtype=np.uint8).reshape((480, 640, 3))
    mesh_pixels = np.sum(
        (frame[:, :, 0] > 25) & (frame[:, :, 0] < 145) &
        (frame[:, :, 1] > 30) & (frame[:, :, 1] < 150) &
        (frame[:, :, 2] > 30) & (frame[:, :, 2] < 155)
    )
    red_marker_pixels = np.sum(
        (frame[:, :, 0] > 180) & (frame[:, :, 1] > 45) & (frame[:, :, 1] < 130) & (frame[:, :, 2] < 90)
    )
    assert mesh_pixels > 500
    assert red_marker_pixels < 256


def test_native_software_backend_depends_on_camera_attitude(monkeypatch, tmp_path):
    monkeypatch.setenv("LIFTOFF_RENDER_DRONE_MESH", str(tmp_path / "missing_target_drone.obj"))
    with NativeRenderEngine(RenderConfig(enabled=True, backend="software")) as renderer:
        level = renderer.render_frame(
            drone=_drone(),
            camera=_camera(),
            targets=(_target(),),
            sequence_id=10,
        )
        pitched = renderer.render_frame(
            drone={**_drone(), "q": _quat_y(np.deg2rad(28.0))},
            camera=_camera(),
            targets=(_target(),),
            sequence_id=11,
        )

    assert level.pixels is not None
    assert pitched.pixels is not None
    level_frame = np.frombuffer(level.pixels, dtype=np.uint8)
    pitched_frame = np.frombuffer(pitched.pixels, dtype=np.uint8)
    mean_abs_delta = np.mean(np.abs(level_frame.astype(float) - pitched_frame.astype(float)))
    assert mean_abs_delta > 4.0


def _drone() -> dict:
    return {
        "t": 0.0,
        "x": np.zeros(3),
        "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
    }


def _camera() -> dict:
    return {
        "c_id": 0,
        "position_b": np.zeros(3),
        "body_to_camera": np.eye(3),
        "width_px": 640,
        "height_px": 480,
        "fx_px": 320.0,
        "fy_px": 320.0,
        "cx_px": 320.0,
        "cy_px": 240.0,
        "hfov_rad": 1.0,
        "vfov_rad": 0.8,
    }


def _target() -> dict:
    return {
        "c_id": 0,
        "position_w": np.array([2.0, 0.0, 0.0]),
        "velocity_w": np.zeros(3),
        "radius_m": 0.2,
    }


def _quat_y(theta: float) -> np.ndarray:
    return np.array([0.0, np.sin(theta / 2.0), 0.0, np.cos(theta / 2.0)])
