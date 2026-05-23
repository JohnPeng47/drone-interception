from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any

import numpy as np

DEFAULT_DATASET = Path(".runs/csim_generator_sampling/whole_sphere_fixed_camera_roll_1024/sobol_samples.csimin")
DEFAULT_RECORDS = Path(".runs/csim_generator_sampling/whole_sphere_fixed_camera_roll_1024/sobol_sample_records.json")
DEFAULT_TARGET_W = np.array([0.0, 0.0, 3.0], dtype=float)
TITLE = "Sobol initial positions and world LOS"


def create_figure(*, data: str | Path | None = None, records: str | Path | None = None, **_: Any):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("This graph requires plotly. Install it with `python -m pip install plotly`.") from exc

    dataset = Path(data) if data is not None else DEFAULT_DATASET
    records_path = Path(records) if records is not None else DEFAULT_RECORDS
    if records_path.exists():
        samples = _load_record_samples(records_path)
        title_source = records_path
    else:
        samples = _load_instance_samples(dataset)
        title_source = dataset

    if not samples:
        raise ValueError(f"{title_source} did not contain any samples")

    pursuer = np.array([sample["pursuer_position_w"] for sample in samples], dtype=float)
    target_start = np.asarray(samples[0]["target_position_w"], dtype=float)
    valid = np.array([bool(sample["valid"]) for sample in samples], dtype=bool)

    valid_pursuer = pursuer[valid]
    invalid_pursuer = pursuer[~valid]
    fig = go.Figure()
    # Plotly uses one line style per trace, so split valid/invalid LOS segments.
    for is_valid, name, color in (
        (True, "valid LOS to target", "rgba(37, 99, 235, 0.14)"),
        (False, "invalid LOS to target", "rgba(220, 38, 38, 0.22)"),
    ):
        subset_x: list[float | None] = []
        subset_y: list[float | None] = []
        subset_z: list[float | None] = []
        for start in pursuer[valid == is_valid]:
            subset_x.extend([float(start[0]), float(target_start[0]), None])
            subset_y.extend([float(start[1]), float(target_start[1]), None])
            subset_z.extend([float(start[2]), float(target_start[2]), None])
        if subset_x:
            fig.add_trace(
                go.Scatter3d(
                    x=subset_x,
                    y=subset_y,
                    z=subset_z,
                    mode="lines",
                    name=name,
                    line={"color": color, "width": 1},
                    hoverinfo="skip",
                )
            )

    if len(valid_pursuer):
        fig.add_trace(
            go.Scatter3d(
                x=valid_pursuer[:, 0],
                y=valid_pursuer[:, 1],
                z=valid_pursuer[:, 2],
                mode="markers",
                name=f"valid pursuer starts ({len(valid_pursuer)})",
                marker={"size": 3, "color": valid_pursuer[:, 2], "colorscale": "Viridis", "opacity": 0.82},
                text=[_sample_hover(sample) for sample in samples if sample["valid"]],
                hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        )
    if len(invalid_pursuer):
        fig.add_trace(
            go.Scatter3d(
                x=invalid_pursuer[:, 0],
                y=invalid_pursuer[:, 1],
                z=invalid_pursuer[:, 2],
                mode="markers",
                name=f"invalid candidates ({len(invalid_pursuer)})",
                marker={"size": 5, "color": "#dc2626", "symbol": "x", "opacity": 0.92},
                text=[_sample_hover(sample) for sample in samples if not sample["valid"]],
                hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=[target_start[0]],
            y=[target_start[1]],
            z=[target_start[2]],
            mode="markers+text",
            name="target start",
            marker={"size": 9, "color": "#facc15", "line": {"color": "black", "width": 2}},
            text=["Target"],
            textposition="top center",
            hovertemplate="target<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{TITLE}<br><sup>{title_source}; valid={int(valid.sum())}, invalid={int((~valid).sum())}</sup>",
        scene={
            "xaxis_title": "world x [m]",
            "yaxis_title": "world y [m]",
            "zaxis_title": "world z [m]",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        legend={"x": 0.02, "y": 0.98},
        template="plotly_white",
    )
    return fig


def _load_record_samples(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for row in rows:
        los_w = _spherical_deg(float(row["los_azimuth_deg"]), float(row["los_elevation_deg"]))
        target_w = DEFAULT_TARGET_W.copy()
        pursuer_w = target_w - float(row["range_m"]) * los_w
        samples.append(
            {
                "seed": int(row["seed"]),
                "stratum": str(row["stratum"]),
                "valid": bool(row.get("valid", True)),
                "validation_error": row.get("validation_error"),
                "pursuer_position_w": pursuer_w,
                "target_position_w": target_w,
            }
        )
    return samples


def _load_instance_samples(path: Path) -> list[dict[str, Any]]:
    from backends.csim.generator.instance_store import read_sim_instances

    instances = read_sim_instances(path)
    return [
        {
            "seed": int(instance.seed),
            "stratum": "valid",
            "valid": True,
            "validation_error": None,
            "pursuer_position_w": np.asarray(instance.pursuer_initial.position_w, dtype=float),
            "target_position_w": np.asarray(instance.targets[0].initial.position_w, dtype=float),
        }
        for instance in instances
    ]


def _sample_hover(sample: dict[str, Any]) -> str:
    text = f"seed={sample['seed']}<br>stratum={sample['stratum']}<br>valid={sample['valid']}"
    if sample.get("validation_error"):
        text += f"<br>{sample['validation_error']}"
    return text


def _spherical_deg(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=float,
    )
