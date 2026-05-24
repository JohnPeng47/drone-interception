from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO

from common import DETECTION_ROOT


DEFAULT_MODEL = DETECTION_ROOT / "models" / "doguilmak__Drone-Detection-YOLOv11x" / "weight" / "best.pt"
DEFAULT_IMAGES = DETECTION_ROOT / "data" / "validation_sample" / "images"
DEFAULT_OUT = DETECTION_ROOT / "runs" / "hf_yolov11x_validation_sample"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a folder of images.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_path = args.model.resolve()
    images_path = args.images.resolve()
    out_path = args.out.resolve()

    out_path.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(images_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        project=str(out_path.parent),
        name=out_path.name,
        exist_ok=True,
        verbose=False,
    )

    rows: list[dict[str, object]] = []
    for result in results:
        image_path = Path(result.path)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            rows.append(
                {
                    "image": str(image_path),
                    "class_id": "",
                    "class_name": "",
                    "confidence": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                }
            )
            continue

        for box in boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            rows.append(
                {
                    "image": str(image_path),
                    "class_id": class_id,
                    "class_name": model.names.get(class_id, str(class_id)),
                    "confidence": float(box.conf.item()),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    csv_path = out_path / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": str(model_path),
        "images": str(images_path),
        "output": str(out_path),
        "image_count": len(results),
        "prediction_rows": len(rows),
        "detections": sum(1 for row in rows if row["confidence"] != ""),
    }
    summary_path = out_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote predictions to {csv_path}")


if __name__ == "__main__":
    main()
