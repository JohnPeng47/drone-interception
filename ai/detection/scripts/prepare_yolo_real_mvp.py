from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from common import DETECTION_ROOT, DUT_ROOT, UAV_EAGLE_ROOT, iter_images, reset_dir, symlink_or_replace


DEFAULT_OUT = DETECTION_ROOT / "data" / "yolo_real_mvp"


def convert_box_to_yolo(width: int, height: int, xmin: float, ymin: float, xmax: float, ymax: float) -> str:
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def convert_dut_xml(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing size block in {xml_path}")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    labels: list[str] = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        labels.append(convert_box_to_yolo(width, height, xmin, ymin, xmax, ymax))
    return labels


def add_dut_split(split: str, out: Path) -> int:
    count = 0
    for image in iter_images(DUT_ROOT / split / "img"):
        stem = f"dut_{split}_{image.stem}"
        target_image = out / "images" / split / f"{stem}{image.suffix.lower()}"
        target_label = out / "labels" / split / f"{stem}.txt"
        symlink_or_replace(image, target_image)
        labels = convert_dut_xml(DUT_ROOT / split / "xml" / f"{image.stem}.xml")
        target_label.write_text("\n".join(labels) + ("\n" if labels else ""))
        count += 1
    return count


def add_uav_eagle_train(out: Path) -> int:
    count = 0
    for image in iter_images(UAV_EAGLE_ROOT / "images"):
        stem = f"uav_eagle_{image.stem}"
        target_image = out / "images" / "train" / f"{stem}{image.suffix.lower()}"
        target_label = out / "labels" / "train" / f"{stem}.txt"
        source_label = UAV_EAGLE_ROOT / "labels" / f"{image.stem}.txt"
        if not source_label.exists():
            continue
        symlink_or_replace(image, target_image)
        shutil.copyfile(source_label, target_label)
        count += 1
    return count


def verify_images(out: Path) -> None:
    for split in ("train", "val", "test"):
        for image in iter_images(out / "images" / split):
            with Image.open(image) as im:
                im.verify()


def write_data_yaml(out: Path) -> None:
    yaml_text = f"""path: {out}
train: images/train
val: images/val
test: images/test
names:
  0: drone
"""
    (out / "data.yaml").write_text(yaml_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a canonical YOLO dataset from local real-drone data.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()

    reset_dir(args.out)
    for split in ("train", "val", "test"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {
        "dut_train": add_dut_split("train", args.out),
        "dut_val": add_dut_split("val", args.out),
        "dut_test": add_dut_split("test", args.out),
        "uav_eagle_train": add_uav_eagle_train(args.out),
    }
    write_data_yaml(args.out)
    if args.verify_images:
        verify_images(args.out)

    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"Wrote YOLO dataset to {args.out}")
    print(f"Wrote data config to {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
