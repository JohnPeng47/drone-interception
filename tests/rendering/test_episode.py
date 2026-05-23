from __future__ import annotations

import json

from rendering.python.episode import generate_puffer_pov_episode


def test_generate_puffer_pov_episode_writes_frames(tmp_path):
    episode = generate_puffer_pov_episode(
        tmp_path / "episode",
        steps=3,
        width_px=64,
        height_px=48,
    )

    assert len(episode.frame_paths) == 3
    assert episode.metadata_path.exists()
    metadata = json.loads(episode.metadata_path.read_text(encoding="utf-8"))
    assert metadata["frame_count"] == 3
    for frame_path in episode.frame_paths:
        data = frame_path.read_bytes()
        assert data.startswith(b"P6\n64 48\n255\n")
        assert len(data) > 64 * 48 * 3
