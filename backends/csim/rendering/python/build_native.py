from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def native_library_path() -> Path:
    suffix = ".dll" if os.name == "nt" else ".so"
    return Path(__file__).resolve().parents[1] / "native" / "_build" / f"libliftoff_render_native{suffix}"


def build_native(force: bool = False) -> Path:
    root = repo_root()
    rendering_dir = root / "backends" / "csim" / "rendering"
    native_dir = rendering_dir / "native"
    platform_name = platform.system().lower()
    platform_source = (
        native_dir / "platform" / "win32" / "render_platform_win32.cpp"
        if platform_name == "windows"
        else native_dir / "platform" / "linux" / "render_platform_linux.cpp"
    )
    sources = [
        native_dir / "src" / "render_engine.cpp",
        platform_source,
    ]
    headers = [
        rendering_dir / "include" / "liftoff_render_api.h",
        rendering_dir / "include" / "liftoff_render_errors.h",
        rendering_dir / "include" / "liftoff_render_types.h",
        native_dir / "platform" / "render_platform.h",
    ]
    out = native_library_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    newest_input = max(path.stat().st_mtime for path in [*sources, *headers])
    if not force and out.exists() and out.stat().st_mtime >= newest_input:
        return out

    cmd = [
        "c++",
        "-std=c++17",
        "-O2",
        "-shared",
        f"-I{rendering_dir / 'include'}",
        f"-I{native_dir / 'platform'}",
        f"-I{native_dir / 'src'}",
        *(str(src) for src in sources),
        "-o",
        str(out),
    ]
    if os.name != "nt":
        cmd.insert(3, "-fPIC")
    subprocess.run(cmd, check=True)
    return out


if __name__ == "__main__":
    print(build_native())
