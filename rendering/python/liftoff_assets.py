from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_LIFTOFF_DATA_DIR = Path("/mnt/c/Program Files (x86)/Steam/steamapps/common/Liftoff/Liftoff_Data")


@dataclass(frozen=True)
class MeshSpec:
    asset_file: str
    path_id: int
    material: str
    offset_u: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0


@dataclass(frozen=True)
class DroneVariant:
    name: str
    description: str
    meshes: tuple[MeshSpec, ...]


FRAME_MESHES: tuple[MeshSpec, ...] = (
    MeshSpec("resources.assets", 305, "vortex_frame"),
    MeshSpec("resources.assets", 312, "vortex_frame"),
    MeshSpec("resources.assets", 319, "vortex_frame"),
    MeshSpec("resources.assets", 323, "vortex_frame"),
    MeshSpec("resources.assets", 327, "vortex_frame"),
    MeshSpec("resources.assets", 302, "battery", (0.0, 0.027, -0.015)),
    MeshSpec("resources.assets", 310, "strap", (-0.018, 0.024, -0.003)),
    MeshSpec("resources.assets", 314, "strap", (0.018, 0.024, -0.003)),
)

MOTOR_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (0.076, 0.021, -0.055),
    (-0.076, 0.021, -0.055),
    (0.076, 0.021, 0.065),
    (-0.076, 0.021, 0.065),
)

PROP_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (0.076, 0.028, -0.055),
    (-0.076, 0.028, -0.055),
    (0.076, 0.028, 0.065),
    (-0.076, 0.028, 0.065),
)


def _xnova_motor_specs(
    offsets: Iterable[tuple[float, float, float]],
    *,
    scale: float,
) -> tuple[MeshSpec, ...]:
    specs: list[MeshSpec] = []
    for x, y, z in offsets:
        specs.append(MeshSpec("sharedassets5.assets", 37, "motor", (x, y, z), scale))
        specs.append(MeshSpec("sharedassets5.assets", 43, "motor", (x, y - 0.004, z), scale))
    return tuple(specs)


def _prop_pair_specs(
    asset_file: str,
    right_path_id: int,
    left_path_id: int,
    material: str,
    offsets: tuple[tuple[float, float, float], ...],
    *,
    scale: float,
) -> tuple[MeshSpec, ...]:
    return (
        MeshSpec(asset_file, right_path_id, material, offsets[0], scale),
        MeshSpec(asset_file, left_path_id, material, offsets[1], scale),
        MeshSpec(asset_file, left_path_id, material, offsets[2], scale),
        MeshSpec(asset_file, right_path_id, material, offsets[3], scale),
    )


DRONE_VARIANTS: tuple[DroneVariant, ...] = (
    DroneVariant(
        "vortex_dal_xnova_runcam",
        "Vortex frame with DAL tri-blades, XNova motors, and compact Runcam-style FPV camera.",
        FRAME_MESHES
        + (MeshSpec("resources.assets", 303, "camera", (0.0, 0.012, -0.085)),)
        + _prop_pair_specs("resources.assets", 301, 309, "prop", PROP_OFFSETS, scale=1.0)
        + _xnova_motor_specs(MOTOR_OFFSETS, scale=8.0),
    ),
    DroneVariant(
        "vortex_racekraft_xnova_hs1177",
        "Vortex frame with broader RaceKraft tri-blades, XNova motors, and HS1177 box camera.",
        FRAME_MESHES
        + (MeshSpec("sharedassets5.assets", 40, "camera_blue", (0.0, 0.012, -0.088), 1.1),)
        + _prop_pair_specs("sharedassets5.assets", 41, 45, "prop_cyan", PROP_OFFSETS, scale=1.07)
        + _xnova_motor_specs(MOTOR_OFFSETS, scale=8.8),
    ),
    DroneVariant(
        "vortex_gemfan_xnova_actioncam",
        "Vortex frame with long Gemfan bullnose props, XNova motors, and a tall action camera block.",
        FRAME_MESHES
        + (MeshSpec("resources.assets", 316, "camera_red", (0.0, 0.035, -0.025), 1.0),)
        + tuple(MeshSpec("resources.assets", 308, "prop_orange", offset, 1.08) for offset in PROP_OFFSETS)
        + _xnova_motor_specs(MOTOR_OFFSETS, scale=8.4),
    ),
    DroneVariant(
        "vortex_dal_heavy_actioncam",
        "Vortex frame with DAL props, oversized motor bells, and a top-mounted action camera profile.",
        FRAME_MESHES
        + (MeshSpec("resources.assets", 316, "camera_red", (0.0, 0.045, -0.005), 1.15),)
        + _prop_pair_specs("resources.assets", 301, 309, "prop_white", PROP_OFFSETS, scale=1.12)
        + _xnova_motor_specs(MOTOR_OFFSETS, scale=10.5),
    ),
    DroneVariant(
        "vortex_racekraft_low_cam",
        "Vortex frame with RaceKraft props, smaller motor bells, and low forward camera mount.",
        FRAME_MESHES
        + (MeshSpec("resources.assets", 306, "camera_blue", (0.0, 0.004, -0.095), 0.9),)
        + _prop_pair_specs("sharedassets5.assets", 41, 45, "prop_green", PROP_OFFSETS, scale=0.96)
        + _xnova_motor_specs(MOTOR_OFFSETS, scale=7.2),
    ),
)

DRONE_MESHES: tuple[MeshSpec, ...] = DRONE_VARIANTS[0].meshes


def export_target_drone(
    out_dir: str | Path,
    *,
    liftoff_data_dir: str | Path | None = None,
    variant: str = "vortex_dal_xnova_runcam",
) -> Path:
    selected = _variant_by_name(variant)
    data_dir = Path(
        liftoff_data_dir
        or os.environ.get("LIFTOFF_DATA_DIR", "")
        or DEFAULT_LIFTOFF_DATA_DIR
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return _export_variant(out_dir, data_dir, selected, stem="target_drone")


def export_target_drone_variants(
    out_dir: str | Path,
    *,
    liftoff_data_dir: str | Path | None = None,
) -> tuple[Path, ...]:
    data_dir = Path(
        liftoff_data_dir
        or os.environ.get("LIFTOFF_DATA_DIR", "")
        or DEFAULT_LIFTOFF_DATA_DIR
    )
    out_dir = Path(out_dir)
    variants_dir = out_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(_export_variant(variants_dir, data_dir, variant, stem=variant.name) for variant in DRONE_VARIANTS)
    manifest = {
        "source_data_dir": str(data_dir),
        "variant_count": len(DRONE_VARIANTS),
        "variants": [
            {
                "name": variant.name,
                "description": variant.description,
                "mesh_path": str(path),
            }
            for variant, path in zip(DRONE_VARIANTS, paths, strict=True)
        ],
    }
    (out_dir / "target_drone_variants.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if paths:
        default_path = out_dir / "target_drone.obj"
        default_json = out_dir / "target_drone.json"
        default_path.write_text(paths[0].read_text(encoding="ascii"), encoding="ascii", newline="\n")
        default_json.write_text((variants_dir / f"{DRONE_VARIANTS[0].name}.json").read_text(encoding="utf-8"), encoding="utf-8")
    return paths


def variant_names() -> tuple[str, ...]:
    return tuple(variant.name for variant in DRONE_VARIANTS)


def _variant_by_name(name: str) -> DroneVariant:
    for variant in DRONE_VARIANTS:
        if variant.name == name:
            return variant
    raise ValueError(f"Unknown Liftoff drone variant {name!r}; expected one of {', '.join(variant_names())}")


def _export_variant(out_dir: Path, data_dir: Path, variant: DroneVariant, *, stem: str) -> Path:
    out_path = out_dir / f"{stem}.obj"
    metadata_path = out_dir / f"{stem}.json"

    meshes = _load_mesh_exports(data_dir, variant.meshes)
    vertex_offset = 0
    parts: list[dict[str, object]] = []
    with out_path.open("w", encoding="ascii", newline="\n") as f:
        f.write(f"# Liftoff-derived target drone mesh cache: {variant.name}\n")
        f.write("# Generated from local Steam assets; do not commit extracted asset payloads.\n")
        for index, spec in enumerate(variant.meshes):
            mesh_name, vertices, faces = meshes[(spec.asset_file, spec.path_id)]
            f.write(f"o part_{index:02d}_{_safe_name(mesh_name)}\n")
            f.write(f"usemtl {spec.material}\n")
            for vertex in vertices:
                x, y, z = _unity_to_renderer(vertex, spec.offset_u, spec.scale)
                f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            for face in faces:
                i0, i1, i2 = face
                f.write(f"f {i0 + vertex_offset + 1} {i1 + vertex_offset + 1} {i2 + vertex_offset + 1}\n")
            parts.append(
                {
                    "asset_file": spec.asset_file,
                    "path_id": spec.path_id,
                    "name": mesh_name,
                    "material": spec.material,
                    "vertex_count": len(vertices),
                    "triangle_count": len(faces),
                    "offset_u": spec.offset_u,
                    "scale": spec.scale,
                }
            )
            vertex_offset += len(vertices)

    metadata = {
        "variant": variant.name,
        "description": variant.description,
        "source_data_dir": str(data_dir),
        "mesh_path": str(out_path),
        "format": "obj_subset_v1",
        "coordinate_system": "x forward, y right, z up; converted from Unity x lateral, y up, z longitudinal",
        "parts": parts,
        "vertex_count": vertex_offset,
        "triangle_count": sum(int(part["triangle_count"]) for part in parts),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_path


def _load_mesh_exports(
    data_dir: Path,
    specs: Iterable[MeshSpec],
) -> dict[tuple[str, int], tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]]]]:
    try:
        import UnityPy
    except ImportError as exc:
        raise RuntimeError("UnityPy is required only for Liftoff asset export; install it for this dev command") from exc

    requested: dict[str, set[int]] = {}
    for spec in specs:
        requested.setdefault(spec.asset_file, set()).add(spec.path_id)

    loaded: dict[tuple[str, int], tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]]]] = {}
    for asset_file, path_ids in requested.items():
        asset_path = data_dir / asset_file
        if not asset_path.exists():
            raise FileNotFoundError(asset_path)
        env = UnityPy.load(str(asset_path))
        for obj in env.objects:
            if obj.path_id not in path_ids:
                continue
            data = obj.read()
            if obj.type.name != "Mesh":
                raise TypeError(f"{asset_file}:{obj.path_id} is {obj.type.name}, expected Mesh")
            loaded[(asset_file, obj.path_id)] = _parse_unity_obj(data.m_Name, data.export())

    missing = [(spec.asset_file, spec.path_id) for spec in specs if (spec.asset_file, spec.path_id) not in loaded]
    if missing:
        raise RuntimeError(f"Missing requested Liftoff meshes: {missing}")
    return loaded


def _parse_unity_obj(
    name: str,
    text: str,
) -> tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in text.splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            indices = [_obj_index(token, len(vertices)) for token in line.split()[1:]]
            for i in range(1, len(indices) - 1):
                faces.append((indices[0], indices[i], indices[i + 1]))
    if not vertices or not faces:
        raise RuntimeError(f"Mesh {name} did not export vertices and faces")
    return name, vertices, faces


def _obj_index(token: str, vertex_count: int) -> int:
    value = int(token.split("/", 1)[0])
    if value < 0:
        return vertex_count + value
    return value - 1


def _unity_to_renderer(
    vertex: tuple[float, float, float],
    offset_u: tuple[float, float, float],
    scale: float,
) -> tuple[float, float, float]:
    xu = (vertex[0] * scale) + offset_u[0]
    yu = (vertex[1] * scale) + offset_u[1]
    zu = (vertex[2] * scale) + offset_u[2]
    return (-zu, xu, yu)


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export local Liftoff drone meshes into the renderer asset cache.")
    parser.add_argument("--liftoff-data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(".runs/liftoff_assets"))
    parser.add_argument("--variant", choices=variant_names(), default="vortex_dal_xnova_runcam")
    parser.add_argument("--all-variants", action="store_true")
    args = parser.parse_args()
    if args.all_variants:
        for mesh_path in export_target_drone_variants(args.out_dir, liftoff_data_dir=args.liftoff_data_dir):
            print(mesh_path)
    else:
        mesh_path = export_target_drone(args.out_dir, liftoff_data_dir=args.liftoff_data_dir, variant=args.variant)
        print(mesh_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
