from __future__ import annotations

import json

from backends.csim.rendering.python.episode import generate_puffer_pov_episode
from backends.csim.rendering.python.liftoff_assets import variant_names


def test_liftoff_asset_variants_are_named():
    names = variant_names()

    assert names == (
        "vortex_dal_xnova_runcam",
        "vortex_racekraft_xnova_hs1177",
        "vortex_gemfan_xnova_actioncam",
        "vortex_dal_heavy_actioncam",
        "vortex_racekraft_low_cam",
    )


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
    assert metadata["target_visual"] in {"liftoff_mesh_quadcopter", "procedural_quadcopter"}
    if metadata["target_visual"] == "liftoff_mesh_quadcopter":
        assert metadata["target_variant"] == "vortex_dal_xnova_runcam"
        assert metadata["target_mesh_path"] == "assets/variants/vortex_dal_xnova_runcam.obj"
    for frame_path in episode.frame_paths:
        data = frame_path.read_bytes()
        assert data.startswith(b"P6\n64 48\n255\n")
        assert len(data) > 64 * 48 * 3
