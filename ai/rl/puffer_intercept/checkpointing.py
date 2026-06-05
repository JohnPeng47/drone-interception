from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch


CHECKPOINT_TYPE = "puffer_intercept_training"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SavedCheckpoint:
    checkpoint_path: Path
    latest_path: Path | None


def save_training_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ppo_epoch: int,
    generator_state: dict[str, Any],
    checkpoint_dir: Path,
    global_step: int,
    args: Any,
    save_latest: bool = True,
) -> SavedCheckpoint:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ppo_epoch": int(ppo_epoch),
        "generator": generator_state,
        "global_step": int(global_step),
        "args": vars(args),
        "rng": _rng_state(),
    }
    checkpoint_path = checkpoint_dir / f"{int(global_step):012d}.pt"
    _atomic_torch_save(payload, checkpoint_path)
    latest_path = None
    if save_latest:
        latest_path = checkpoint_dir / "latest.pt"
        _atomic_copy(checkpoint_path, latest_path)
    return SavedCheckpoint(checkpoint_path=checkpoint_path, latest_path=latest_path)


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError(f"{path} is not a {CHECKPOINT_TYPE} checkpoint")
    if int(checkpoint.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema in {path}: {checkpoint.get('schema_version')!r}")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    _restore_rng_state(checkpoint.get("rng", {}))
    return checkpoint


def download_s3_uri(s3_uri: str, destination: Path) -> Path:
    bucket, key = _parse_s3_uri(s3_uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _boto3_client().download_file(bucket, key, str(destination))
    return destination


def upload_checkpoint_to_s3(saved: SavedCheckpoint, s3_prefix: str) -> list[str]:
    uploaded = []
    for path in (saved.checkpoint_path, saved.latest_path):
        if path is None:
            continue
        uri = _upload_file_to_s3(path, s3_prefix)
        uploaded.append(uri)
    return uploaded


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    temp_path = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, destination)


def _atomic_copy(source: Path, destination: Path) -> None:
    temp_path = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temp_path)
    os.replace(temp_path, destination)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["torch_cuda"]])


def _upload_file_to_s3(path: Path, s3_prefix: str) -> str:
    bucket, prefix = _parse_s3_uri(s3_prefix)
    key = "/".join(part for part in (prefix.rstrip("/"), path.name) if part)
    _boto3_client().upload_file(str(path), bucket, key)
    return f"s3://{bucket}/{key}"


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3://bucket/key URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _boto3_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 checkpoint transfer") from exc
    return boto3.client("s3")
