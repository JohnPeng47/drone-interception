from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import DETECTION_ROOT, DUT_ROOT, UAV_EAGLE_ROOT, iter_images, reset_dir, symlink_or_replace


DEFAULT_OUT = DETECTION_ROOT / "data" / "validation_sample"


def pick_evenly(items: list[Path], count: int) -> list[Path]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return items
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def add_split(rows: list[dict[str, str]], source_name: str, images: list[Path], out_dir: Path) -> None:
    for image in images:
        target_name = f"{source_name}_{image.name}"
        target = out_dir / "images" / target_name
        symlink_or_replace(image, target)
        rows.append(
            {
                "source": source_name,
                "source_image": str(image),
                "image": str(target),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small validation sample for detector smoke tests.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dut-val-count", type=int, default=40)
    parser.add_argument("--dut-test-count", type=int, default=20)
    parser.add_argument("--uav-eagle-count", type=int, default=20)
    args = parser.parse_args()

    reset_dir(args.out)

    rows: list[dict[str, str]] = []
    add_split(
        rows,
        "dut_val",
        pick_evenly(iter_images(DUT_ROOT / "val" / "img"), args.dut_val_count),
        args.out,
    )
    add_split(
        rows,
        "dut_test",
        pick_evenly(iter_images(DUT_ROOT / "test" / "img"), args.dut_test_count),
        args.out,
    )
    add_split(
        rows,
        "uav_eagle",
        pick_evenly(iter_images(UAV_EAGLE_ROOT / "images"), args.uav_eagle_count),
        args.out,
    )

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "source_image", "image"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} validation images to {args.out / 'images'}")
    print(f"Wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
