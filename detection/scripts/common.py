from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_ROOT = REPO_ROOT / "detection"
DATASETS_ROOT = DETECTION_ROOT / "datasets"

DUT_ROOT = DATASETS_ROOT / "dut_anti_uav"
UAV_EAGLE_ROOT = DATASETS_ROOT / "uav_eagle" / "UAV-Eagle"


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                reset_dir(child)
                child.rmdir()
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def symlink_or_replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))


def iter_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
