from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backends import (
    RenderConfig,
    SimConfig,
    SimOptions,
    PursuerInitialState,
    PursuerParams,
    PufferSimEngineBackend,
)


@dataclass(frozen=True)
class RenderedEpisode:
    out_dir: Path
    frame_paths: tuple[Path, ...]
    metadata_path: Path


def generate_puffer_pov_episode(
    out_dir: str | Path,
    *,
    steps: int = 90,
    dt: float = 1.0 / 30.0,
    width_px: int = 320,
    height_px: int = 240,
    camera_id: str = "front",
) -> RenderedEpisode:
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    params = _default_pursuer_params()
    backend = PufferSimEngineBackend(
        SimConfig(
            pursuer=params,
            options=SimOptions(backend_dt=0.002, action_substeps=5),
            render=RenderConfig(enabled=True, camera_id=camera_id, backend="software"),
        )
    )
    snapshot = backend.reset(
        PursuerInitialState(
            position_w=np.zeros(3),
            velocity_w=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            body_rates_b=np.zeros(3),
        ),
        targets=(
            {
                "id": "target",
                "position_w": np.array([5.0, -0.55, 0.15]),
                "velocity_w": np.array([-0.85, 0.16, -0.04]),
                "radius_m": 0.25,
            },
        ),
        cameras=(_pov_camera(camera_id, width_px, height_px, capture_rate_hz=1.0 / dt),),
    )

    frame_paths: list[Path] = []
    samples: list[dict[str, Any]] = []
    for frame_index in range(int(steps)):
        frame_path = frames_dir / f"frame_{frame_index:04d}.ppm"
        output = _selected_frame(snapshot, camera_id)
        _write_ppm(
            frame_path,
            output["frame_rgb"],
            width=int(output["frame_width_px"]),
            height=int(output["frame_height_px"]),
        )
        frame_paths.append(frame_path)
        samples.append({
            "frame": frame_index,
            "path": str(frame_path.relative_to(out_dir)),
            "render_status": output["render_status_name"],
            "vehicle_position_w": _array(snapshot["vehicle_state"]["x"]),
            "target_position_w": _array(snapshot["target_states"][0]["position_w"]),
            "distance_m": float(snapshot["metrics"]["distance_m"]),
        })

        hover = {
            "thrust_n": params.mass_kg * params.gravity_mps2,
            "body_rates_b": np.zeros(3),
        }
        snapshot = backend.step_ctbr(snapshot, hover, dt=dt)

    metadata = {
        "renderer": "software",
        "camera_id": camera_id,
        "frame_count": len(frame_paths),
        "width_px": int(width_px),
        "height_px": int(height_px),
        "dt": float(dt),
        "frames_dir": "frames",
        "samples": samples,
    }
    metadata_path = out_dir / "episode.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return RenderedEpisode(
        out_dir=out_dir,
        frame_paths=tuple(frame_paths),
        metadata_path=metadata_path,
    )


def _selected_frame(snapshot: dict[str, Any], camera_id: str) -> dict[str, Any]:
    outputs = snapshot.get("camera_outputs", ())
    for output in outputs:
        if output.get("camera_id") == camera_id:
            if output.get("frame_rgb") is None:
                raise RuntimeError(f"Camera {camera_id} did not produce a frame: {output}")
            return output
    raise RuntimeError(f"Snapshot has no output for camera {camera_id}")


def _write_ppm(path: Path, pixels: bytes, *, width: int, height: int) -> None:
    expected = int(width) * int(height) * 3
    if len(pixels) != expected:
        raise ValueError(f"Expected {expected} RGB bytes, got {len(pixels)}")
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(pixels)


def _default_pursuer_params() -> PursuerParams:
    return PursuerParams(
        mass_kg=0.027,
        ixx=3.85e-6,
        iyy=3.85e-6,
        izz=5.9675e-6,
        arm_len_m=0.0396,
        k_thrust=3.16e-10,
        k_yaw=0.005964552,
        max_rpm=21702.0,
    )


def _pov_camera(camera_id: str, width: int, height: int, *, capture_rate_hz: float) -> dict[str, Any]:
    hfov = np.deg2rad(90.0)
    vfov = 2.0 * np.arctan(np.tan(hfov * 0.5) * float(height) / float(width))
    fx = float(width) / (2.0 * np.tan(hfov * 0.5))
    fy = float(height) / (2.0 * np.tan(vfov * 0.5))
    return {
        "id": camera_id,
        "position_b": np.zeros(3),
        "body_to_camera": np.eye(3),
        "capture_rate_hz": float(capture_rate_hz),
        "intrinsics": {
            "width_px": int(width),
            "height_px": int(height),
            "fx_px": fx,
            "fy_px": fy,
            "cx_px": float(width) / 2.0,
            "cy_px": float(height) / 2.0,
            "hfov_rad": hfov,
            "vfov_rad": vfov,
        },
    }


def _array(value: Any) -> list[float]:
    return [float(x) for x in np.asarray(value, dtype=float).reshape(-1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(".runs/rendered_pov_episode"))
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--width-px", type=int, default=320)
    parser.add_argument("--height-px", type=int, default=240)
    args = parser.parse_args()
    episode = generate_puffer_pov_episode(
        args.out_dir,
        steps=args.steps,
        dt=args.dt,
        width_px=args.width_px,
        height_px=args.height_px,
    )
    print(episode.metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
