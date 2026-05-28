from __future__ import annotations

import json

import numpy as np

from backends.csim.rendering.python.episode import generate_puffer_pov_episode


def test_render_integration_generates_video_frame_sequence(tmp_path):
    episode = generate_puffer_pov_episode(
        tmp_path / "render_integration",
        steps=6,
        dt=1.0 / 24.0,
        width_px=96,
        height_px=72,
    )

    assert len(episode.frame_paths) == 6
    assert [path.name for path in episode.frame_paths] == [
        f"frame_{index:04d}.ppm" for index in range(6)
    ]

    metadata = json.loads(episode.metadata_path.read_text(encoding="utf-8"))
    assert metadata["renderer"] == "software"
    assert metadata["frame_count"] == 6
    assert metadata["width_px"] == 96
    assert metadata["height_px"] == 72
    assert metadata["frames_dir"] == "frames"
    assert [sample["render_status"] for sample in metadata["samples"]] == ["ok"] * 6

    frames = [_read_ppm_rgb(path, width=96, height=72) for path in episode.frame_paths]
    assert all(frame.max() > frame.min() for frame in frames)
    assert all(np.unique(frame.reshape((-1, 3)), axis=0).shape[0] > 32 for frame in frames)

    frame_deltas = [
        np.mean(np.abs(frames[index + 1].astype(float) - frames[index].astype(float)))
        for index in range(len(frames) - 1)
    ]
    assert max(frame_deltas) > 0.1


def _read_ppm_rgb(path, *, width: int, height: int) -> np.ndarray:
    data = path.read_bytes()
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    assert data.startswith(header)
    payload = data[len(header):]
    assert len(payload) == width * height * 3
    return np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3))
