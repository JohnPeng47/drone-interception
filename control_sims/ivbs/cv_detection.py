from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TraditionalCvMeasurement:
    detected: bool
    uv_norm: np.ndarray
    apparent_radius_px: float
    confidence: float


@dataclass(frozen=True)
class TraditionalCvConfig:
    min_area_px: int = 6
    max_area_fraction: float = 0.2
    dark_threshold: int = 80
    contrast_margin: int = 25


def detect_dark_blob(
    frame_rgb: np.ndarray,
    *,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    config: TraditionalCvConfig | None = None,
) -> TraditionalCvMeasurement:
    """Detect a dark target blob in RGB pixels using deterministic CV only."""

    if not _valid_intrinsics(fx_px, fy_px, cx_px, cy_px):
        return missed_measurement()
    cfg = config or TraditionalCvConfig()
    frame = np.asarray(frame_rgb)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return missed_measurement()
    rgb = frame[:, :, 0:3].astype(np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    threshold = min(float(cfg.dark_threshold), float(np.median(gray) - cfg.contrast_margin))
    mask = gray <= threshold
    component = _largest_component(mask, min_area=int(cfg.min_area_px))
    if component is None:
        return missed_measurement()
    ys, xs = component
    area = int(len(xs))
    max_area = float(frame.shape[0] * frame.shape[1]) * float(cfg.max_area_fraction)
    if area <= 0 or float(area) > max_area:
        return missed_measurement()
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    radius = float(np.sqrt(float(area) / np.pi))
    uv = np.array([(cx - float(cx_px)) / float(fx_px), (cy - float(cy_px)) / float(fy_px)], dtype=float)
    confidence = float(np.clip(area / max(float(cfg.min_area_px) * 4.0, 1.0), 0.0, 1.0))
    return TraditionalCvMeasurement(
        detected=True,
        uv_norm=uv,
        apparent_radius_px=radius,
        confidence=confidence,
    )


def _valid_intrinsics(fx_px: float, fy_px: float, cx_px: float, cy_px: float) -> bool:
    return bool(
        np.isfinite(float(fx_px))
        and np.isfinite(float(fy_px))
        and np.isfinite(float(cx_px))
        and np.isfinite(float(cy_px))
        and float(fx_px) > 0.0
        and float(fy_px) > 0.0
    )


def missed_measurement() -> TraditionalCvMeasurement:
    return TraditionalCvMeasurement(
        detected=False,
        uv_norm=np.zeros(2, dtype=float),
        apparent_radius_px=0.0,
        confidence=0.0,
    )


def _largest_component(mask: np.ndarray, *, min_area: int) -> tuple[np.ndarray, np.ndarray] | None:
    mask_bool = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask_bool.shape, dtype=bool)
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_area = 0
    height, width = mask_bool.shape
    for y0, x0 in zip(*np.nonzero(mask_bool)):
        if visited[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        visited[y0, x0] = True
        ys: list[int] = []
        xs: list[int] = []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or not mask_bool[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))
        area = len(xs)
        if area >= int(min_area) and area > best_area:
            best_area = area
            best = (np.asarray(ys, dtype=int), np.asarray(xs, dtype=int))
    return best
