from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

import numpy as np

from backends.csim.bindings.types import (
    CameraConfig,
    CameraIntrinsics,
    PursuerInitialState,
    PursuerParams,
    SimConfig,
    SimInstance,
    SimOptions,
    TargetBehaviorConfig,
    TargetConfig,
    TargetControllerConfig,
    TargetState,
)


SIM_INSTANCE_MAGIC = b"CSIMINST"
SIM_INSTANCE_FORMAT_VERSION = 2
_HEADER = struct.Struct("<8sIIQ")
_U8 = struct.Struct("<B")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_I64 = struct.Struct("<q")
_F32 = struct.Struct("<f")


def write_sim_instances(path: str | Path, instances: Iterable[SimInstance]) -> None:
    records = list(instances)
    payload = bytearray()
    for instance in records:
        _write_sim_instance(payload, instance)
    header = _HEADER.pack(
        SIM_INSTANCE_MAGIC,
        SIM_INSTANCE_FORMAT_VERSION,
        len(records),
        len(payload),
    )
    Path(path).write_bytes(header + payload)


def read_sim_instances(path: str | Path) -> list[SimInstance]:
    data = Path(path).read_bytes()
    if len(data) < _HEADER.size:
        raise ValueError(f"{path} is too small to be a SimInstance table")
    magic, version, count, payload_len = _HEADER.unpack_from(data)
    if magic != SIM_INSTANCE_MAGIC:
        raise ValueError(f"{path} is not a SimInstance table")
    if version != SIM_INSTANCE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported SimInstance table version {version}; "
            f"expected {SIM_INSTANCE_FORMAT_VERSION}"
        )
    payload_start = _HEADER.size
    payload_end = payload_start + int(payload_len)
    if payload_end != len(data):
        raise ValueError(f"{path} has an invalid SimInstance table length")

    cursor = _Cursor(data[payload_start:payload_end])
    instances = [_read_sim_instance(cursor) for _ in range(int(count))]
    cursor.expect_finished(path)
    return instances


class _Cursor:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def read(self, size: int) -> bytes:
        end = self.offset + int(size)
        if end > len(self.data):
            raise ValueError("Unexpected end of SimInstance table")
        chunk = self.data[self.offset:end]
        self.offset = end
        return chunk

    def expect_finished(self, path: str | Path) -> None:
        if self.offset != len(self.data):
            raise ValueError(f"{path} has trailing bytes after SimInstance records")


def _write_u8(buf: bytearray, value: int) -> None:
    buf.extend(_U8.pack(int(value)))


def _read_u8(cursor: _Cursor) -> int:
    return int(_U8.unpack(cursor.read(_U8.size))[0])


def _write_u16(buf: bytearray, value: int) -> None:
    if not 0 <= int(value) <= 0xFFFF:
        raise ValueError(f"uint16 value out of range: {value}")
    buf.extend(_U16.pack(int(value)))


def _read_u16(cursor: _Cursor) -> int:
    return int(_U16.unpack(cursor.read(_U16.size))[0])


def _write_u32(buf: bytearray, value: int) -> None:
    buf.extend(_U32.pack(int(value)))


def _read_u32(cursor: _Cursor) -> int:
    return int(_U32.unpack(cursor.read(_U32.size))[0])


def _write_i64(buf: bytearray, value: int) -> None:
    buf.extend(_I64.pack(int(value)))


def _read_i64(cursor: _Cursor) -> int:
    return int(_I64.unpack(cursor.read(_I64.size))[0])


def _write_f32(buf: bytearray, value: float) -> None:
    buf.extend(_F32.pack(float(value)))


def _read_f32(cursor: _Cursor) -> float:
    return float(_F32.unpack(cursor.read(_F32.size))[0])


def _write_string(buf: bytearray, value: str) -> None:
    encoded = str(value).encode("utf-8")
    _write_u16(buf, len(encoded))
    buf.extend(encoded)


def _read_string(cursor: _Cursor) -> str:
    length = _read_u16(cursor)
    return cursor.read(length).decode("utf-8")


def _write_f32_array(buf: bytearray, value: np.ndarray, shape: tuple[int, ...]) -> None:
    array = np.asarray(value, dtype=np.float32).reshape(shape)
    buf.extend(array.astype("<f4", copy=False).tobytes(order="C"))


def _read_f32_array(cursor: _Cursor, shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape))
    data = cursor.read(count * 4)
    return np.frombuffer(data, dtype="<f4").astype(float).reshape(shape).copy()


def _write_optional_array(buf: bytearray, value: np.ndarray | None, shape: tuple[int, ...]) -> None:
    _write_u8(buf, int(value is not None))
    if value is not None:
        _write_f32_array(buf, value, shape)


def _read_optional_array(cursor: _Cursor, shape: tuple[int, ...]) -> np.ndarray | None:
    if not _read_u8(cursor):
        return None
    return _read_f32_array(cursor, shape)


def _write_pursuer_params(buf: bytearray, params: PursuerParams) -> None:
    for value in (
        params.mass_kg,
        params.ixx,
        params.iyy,
        params.izz,
        params.arm_len_m,
        params.k_thrust,
        params.k_yaw,
        params.k_ang_damp,
        params.b_drag,
        params.gravity_mps2,
        params.max_rpm,
        params.max_vel_mps,
        params.max_omega_rps,
        params.motor_tau_s,
        params.k_w,
    ):
        _write_f32(buf, value)
    _write_u8(buf, int(params.rpm_min is not None))
    if params.rpm_min is not None:
        _write_f32(buf, params.rpm_min)
    _write_optional_array(buf, params.rotor_positions_b, (4, 3))
    _write_optional_array(buf, params.rotor_directions, (4,))


def _read_pursuer_params(cursor: _Cursor) -> PursuerParams:
    values = [_read_f32(cursor) for _ in range(15)]
    rpm_min = _read_f32(cursor) if _read_u8(cursor) else None
    return PursuerParams(
        mass_kg=values[0],
        ixx=values[1],
        iyy=values[2],
        izz=values[3],
        arm_len_m=values[4],
        k_thrust=values[5],
        k_yaw=values[6],
        k_ang_damp=values[7],
        b_drag=values[8],
        gravity_mps2=values[9],
        max_rpm=values[10],
        max_vel_mps=values[11],
        max_omega_rps=values[12],
        motor_tau_s=values[13],
        k_w=values[14],
        rpm_min=rpm_min,
        rotor_positions_b=_read_optional_array(cursor, (4, 3)),
        rotor_directions=_read_optional_array(cursor, (4,)),
    )


def _write_pursuer_initial(buf: bytearray, initial: PursuerInitialState) -> None:
    _write_f32_array(buf, initial.position_w, (3,))
    _write_f32_array(buf, initial.velocity_w, (3,))
    _write_f32_array(buf, initial.quat_xyzw, (4,))
    _write_f32_array(buf, initial.body_rates_b, (3,))
    _write_optional_array(buf, initial.rotor_speeds, (4,))
    _write_optional_array(buf, initial.wind_w, (3,))


def _read_pursuer_initial(cursor: _Cursor) -> PursuerInitialState:
    return PursuerInitialState(
        position_w=_read_f32_array(cursor, (3,)),
        velocity_w=_read_f32_array(cursor, (3,)),
        quat_xyzw=_read_f32_array(cursor, (4,)),
        body_rates_b=_read_f32_array(cursor, (3,)),
        rotor_speeds=_read_optional_array(cursor, (4,)),
        wind_w=_read_optional_array(cursor, (3,)),
    )


def _write_target(buf: bytearray, target: TargetConfig) -> None:
    _write_string(buf, target.id)
    _write_string(buf, target.kind)
    _write_f32(buf, target.radius_m)
    _write_f32_array(buf, target.initial.position_w, (3,))
    _write_f32_array(buf, target.initial.velocity_w, (3,))
    _write_string(buf, target.behavior.kind)
    _write_u16(buf, len(target.behavior.waypoints))
    for waypoint in target.behavior.waypoints:
        _write_f32_array(buf, waypoint, (3,))
    _write_f32(buf, target.behavior.duration_s)
    _write_u8(buf, int(target.behavior.loop))
    _write_string(buf, target.controller.kind)
    _write_f32(buf, target.controller.kp)
    _write_f32(buf, target.controller.kv)
    _write_f32(buf, target.controller.max_accel_mps2)


def _read_target(cursor: _Cursor) -> TargetConfig:
    target_id = _read_string(cursor)
    kind = _read_string(cursor)
    radius_m = _read_f32(cursor)
    initial = TargetState(
        position_w=_read_f32_array(cursor, (3,)),
        velocity_w=_read_f32_array(cursor, (3,)),
    )
    behavior_kind = _read_string(cursor)
    waypoints = tuple(_read_f32_array(cursor, (3,)) for _ in range(_read_u16(cursor)))
    behavior = TargetBehaviorConfig(
        kind=behavior_kind,
        waypoints=waypoints,
        duration_s=_read_f32(cursor),
        loop=bool(_read_u8(cursor)),
    )
    controller = TargetControllerConfig(
        kind=_read_string(cursor),
        kp=_read_f32(cursor),
        kv=_read_f32(cursor),
        max_accel_mps2=_read_f32(cursor),
    )
    return TargetConfig(
        id=target_id,
        kind=kind,
        radius_m=radius_m,
        initial=initial,
        behavior=behavior,
        controller=controller,
    )


def _write_camera(buf: bytearray, camera: CameraConfig) -> None:
    _write_string(buf, camera.id)
    _write_string(buf, camera.parent_id)
    _write_f32_array(buf, camera.position_b, (3,))
    _write_f32_array(buf, camera.body_to_camera, (3, 3))
    _write_u32(buf, camera.intrinsics.width_px)
    _write_u32(buf, camera.intrinsics.height_px)
    for value in (
        camera.intrinsics.fx_px,
        camera.intrinsics.fy_px,
        camera.intrinsics.cx_px,
        camera.intrinsics.cy_px,
        camera.intrinsics.hfov_rad,
        camera.intrinsics.vfov_rad,
        camera.capture_rate_hz,
    ):
        _write_f32(buf, value)


def _read_camera(cursor: _Cursor) -> CameraConfig:
    camera_id = _read_string(cursor)
    parent_id = _read_string(cursor)
    position_b = _read_f32_array(cursor, (3,))
    body_to_camera = _read_f32_array(cursor, (3, 3))
    intrinsics = CameraIntrinsics(
        width_px=_read_u32(cursor),
        height_px=_read_u32(cursor),
        fx_px=_read_f32(cursor),
        fy_px=_read_f32(cursor),
        cx_px=_read_f32(cursor),
        cy_px=_read_f32(cursor),
        hfov_rad=_read_f32(cursor),
        vfov_rad=_read_f32(cursor),
    )
    return CameraConfig(
        id=camera_id,
        parent_id=parent_id,
        position_b=position_b,
        body_to_camera=body_to_camera,
        intrinsics=intrinsics,
        capture_rate_hz=_read_f32(cursor),
    )


def _write_sim_config(buf: bytearray, config: SimConfig | None) -> None:
    _write_u8(buf, int(config is not None))
    if config is None:
        return
    _write_pursuer_params(buf, config.pursuer)
    _write_f32(buf, config.options.backend_dt)
    _write_u32(buf, config.options.action_substeps)
    _write_string(buf, config.options.command_mode)
    _write_f32(buf, config.options.ctbr_rate_gain)
    _write_u8(buf, int(config.options.randomize_params))
    _write_f32(buf, config.intercept_radius_m)


def _read_sim_config(cursor: _Cursor) -> SimConfig | None:
    if not _read_u8(cursor):
        return None
    return SimConfig(
        pursuer=_read_pursuer_params(cursor),
        options=SimOptions(
            backend_dt=_read_f32(cursor),
            action_substeps=_read_u32(cursor),
            command_mode=_read_string(cursor),
            ctbr_rate_gain=_read_f32(cursor),
            randomize_params=bool(_read_u8(cursor)),
        ),
        intercept_radius_m=_read_f32(cursor),
    )


def _write_sim_instance(buf: bytearray, instance: SimInstance) -> None:
    _write_i64(buf, instance.seed)
    _write_pursuer_initial(buf, instance.pursuer_initial)
    _write_u16(buf, len(instance.targets))
    for target in instance.targets:
        _write_target(buf, target)
    _write_u16(buf, len(instance.cameras))
    for camera in instance.cameras:
        _write_camera(buf, camera)
    _write_sim_config(buf, instance.config)


def _read_sim_instance(cursor: _Cursor) -> SimInstance:
    seed = _read_i64(cursor)
    pursuer_initial = _read_pursuer_initial(cursor)
    targets = tuple(_read_target(cursor) for _ in range(_read_u16(cursor)))
    cameras = tuple(_read_camera(cursor) for _ in range(_read_u16(cursor)))
    config = _read_sim_config(cursor)
    return SimInstance(
        seed=seed,
        pursuer_initial=pursuer_initial,
        targets=targets,
        cameras=cameras,
        config=config,
    )
