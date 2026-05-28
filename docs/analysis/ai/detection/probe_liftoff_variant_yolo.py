from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backends import RenderConfig
from backends.csim.rendering.python import LIFTOFF_RENDER_OK, NativeRenderEngine
from backends.csim.rendering.python.liftoff_assets import export_target_drone_variants, variant_names


DEFAULT_MODEL = (
    REPO_ROOT
    / "ai"
    / "detection"
    / "models"
    / "doguilmak__Drone-Detection-YOLOv11x"
    / "weight"
    / "best.pt"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render one centered Liftoff drone variant image and run the local YOLO drone detector."
    )
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR / "liftoff_variant_yolo_probe")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--width-px", type=int, default=640)
    parser.add_argument("--height-px", type=int, default=480)
    parser.add_argument("--target-distance-m", type=float, default=1.1)
    parser.add_argument("--target-radius-m", type=float, default=0.34)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = args.out_dir
    images_dir = out_dir / "images"
    annotated_dir = out_dir / "annotated"
    assets_dir = out_dir / "assets"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    variant_paths = export_target_drone_variants(assets_dir)
    variant_to_mesh = {path.stem: path for path in variant_paths}
    rendered = [
        render_variant(
            variant,
            variant_to_mesh[variant],
            images_dir,
            width_px=args.width_px,
            height_px=args.height_px,
            target_distance_m=args.target_distance_m,
            target_radius_m=args.target_radius_m,
        )
        for variant in variant_names()
    ]

    model_path = args.model.resolve()
    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(path) for path in rendered],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )

    rows: list[dict[str, Any]] = []
    for image_path, result in zip(rendered, results, strict=True):
        detections = []
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                confidence = float(box.conf.item())
                class_name = model.names.get(class_id, str(class_id))
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
        best = max((d["confidence"] for d in detections), default=None)
        rows.append(
            {
                "variant": image_path.stem,
                "image": str(image_path),
                "detections": len(detections),
                "best_confidence": best if best is not None else "",
                "boxes": detections,
            }
        )
        annotate_image(image_path, annotated_dir / image_path.name, detections)

    write_csv(out_dir / "detections.csv", rows)
    summary = {
        "model": str(model_path),
        "out_dir": str(out_dir.resolve()),
        "image_count": len(rendered),
        "detected_images": sum(1 for row in rows if int(row["detections"]) > 0),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def render_variant(
    variant: str,
    mesh_path: Path,
    images_dir: Path,
    *,
    width_px: int,
    height_px: int,
    target_distance_m: float,
    target_radius_m: float,
) -> Path:
    old_mesh_path = os.environ.get("LIFTOFF_RENDER_DRONE_MESH")
    os.environ["LIFTOFF_RENDER_DRONE_MESH"] = str(mesh_path)
    try:
        with NativeRenderEngine(RenderConfig(backend="software")) as renderer:
            result = renderer.render_frame(
                drone=_drone(),
                camera=_camera(width_px, height_px),
                targets=(
                    {
                        "c_id": 0,
                        "position_w": np.array([target_distance_m, 0.0, 0.0]),
                        "velocity_w": np.array([-0.05, 0.0, 0.0]),
                        "radius_m": target_radius_m,
                    },
                ),
                sequence_id=1,
            )
    finally:
        if old_mesh_path is None:
            os.environ.pop("LIFTOFF_RENDER_DRONE_MESH", None)
        else:
            os.environ["LIFTOFF_RENDER_DRONE_MESH"] = old_mesh_path

    if result.status != LIFTOFF_RENDER_OK or result.pixels is None:
        raise RuntimeError(f"Render failed for {variant}: {result.status_name}")

    frame = np.frombuffer(result.pixels, dtype=np.uint8).reshape((height_px, width_px, 3))
    path = images_dir / f"{variant}.png"
    Image.fromarray(frame, mode="RGB").save(path)
    return path


def annotate_image(image_path: Path, out_path: Path, detections: list[dict[str, Any]]) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1 = float(det["x1"])
        y1 = float(det["y1"])
        x2 = float(det["x2"])
        y2 = float(det["y2"])
        label = f"{det['class_name']} {float(det['confidence']):.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 80, 40), width=2)
        draw.text((x1 + 2, max(0.0, y1 - 14)), label, fill=(255, 230, 210))
    image.save(out_path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "image", "detections", "best_confidence"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row["variant"],
                    "image": row["image"],
                    "detections": row["detections"],
                    "best_confidence": row["best_confidence"],
                }
            )


def _drone() -> dict[str, Any]:
    return {
        "t": 0.0,
        "x": np.zeros(3),
        "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
    }


def _camera(width_px: int, height_px: int) -> dict[str, Any]:
    hfov = np.deg2rad(90.0)
    vfov = 2.0 * np.arctan(np.tan(hfov * 0.5) * float(height_px) / float(width_px))
    fx = float(width_px) / (2.0 * np.tan(hfov * 0.5))
    fy = float(height_px) / (2.0 * np.tan(vfov * 0.5))
    return {
        "c_id": 0,
        "position_b": np.zeros(3),
        "body_to_camera": np.eye(3),
        "width_px": int(width_px),
        "height_px": int(height_px),
        "fx_px": fx,
        "fy_px": fy,
        "cx_px": float(width_px) / 2.0,
        "cy_px": float(height_px) / 2.0,
        "hfov_rad": float(hfov),
        "vfov_rad": float(vfov),
    }


if __name__ == "__main__":
    raise SystemExit(main())
