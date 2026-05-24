from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, model_info

from common import DETECTION_ROOT


DEFAULT_REPO = "doguilmak/Drone-Detection-YOLOv11x"
DEFAULT_FILE = "weight/best.pt"


def safe_repo_dir(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model checkpoint for baseline inference.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--filename", default=DEFAULT_FILE)
    parser.add_argument("--out-root", type=Path, default=DETECTION_ROOT / "models")
    args = parser.parse_args()

    out_dir = args.out_root / safe_repo_dir(args.repo_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )

    info = model_info(args.repo_id)
    metadata = {
        "repo_id": args.repo_id,
        "revision": info.sha,
        "filename": args.filename,
        "local_path": str(Path(path)),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(path)
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
