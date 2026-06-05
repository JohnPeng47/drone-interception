from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

import numpy as np

from backends.csim.bindings.types import (
    CameraConfig,
    CameraIntrinsics,
    NoiseConfig,
    PursuerInitialState,
    PursuerParams,
    RenderConfig,
    SimConfig,
    SimInstance,
    SimOptions,
    TargetBehaviorConfig,
    TargetConfig,
    TargetControllerConfig,
    TargetInitialState,
)


SIM_INSTANCE_MAGIC = b"CSIMINST"
SIM_INSTANCE_FORMAT_VERSION = 8
SIM_INSTANCE_WRITE_BLOCK_BYTES = 8 * 1024 * 1024
_HEADER = struct.Struct("<8sIIQ")
_U8 = struct.Struct("<B")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_I64 = struct.Struct("<q")
_F32 = struct.Struct("<f")


def write_sim_instances(path: str | Path, instances: Iterable[SimInstance]) -> None:
    out_path = Path(path)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    count = 0
    payload_len = 0
    block = bytearray()
    try:
        with tmp_path.open("wb") as handle:
            handle.write(_HEADER.pack(SIM_INSTANCE_MAGIC, SIM_INSTANCE_FORMAT_VERSION, 0, 0))
            for instance in instances:
                record = bytearray()
                _write_sim_instance(record, instance)
                block.extend(record)
                count += 1
                payload_len += len(record)
                if len(block) >= SIM_INSTANCE_WRITE_BLOCK_BYTES:
                    handle.write(block)
                    block.clear()
                    handle.flush()
            if block:
                handle.write(block)
                block.clear()
                handle.flush()
            handle.seek(0)
            handle.write(
                _HEADER.pack(
                    SIM_INSTANCE_MAGIC,
                    SIM_INSTANCE_FORMAT_VERSION,
                    count,
                    payload_len,
                )
            )
        tmp_path.replace(out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def read_sim_instances(
    path: str | Path,
    *,
    count: int | None = None,
    offset: int = 0,
) -> list[SimInstance]:
    offset = int(offset)
    if offset < 0:
        raise ValueError("offset must be non-negative")
    requested_count = None if count is None else int(count)
    if requested_count is not None and requested_count < 0:
        raise ValueError("count must be non-negative")

    in_path = Path(path)
    with in_path.open("rb") as handle:
        header = handle.read(_HEADER.size)
        if len(header) < _HEADER.size:
            raise ValueError(f"{path} is too small to be a SimInstance table")
        magic, version, total_count, payload_len = _HEADER.unpack(header)
        if magic != SIM_INSTANCE_MAGIC:
            raise ValueError(f"{path} is not a SimInstance table")
        if version != SIM_INSTANCE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported SimInstance table version {version}; "
                f"expected {SIM_INSTANCE_FORMAT_VERSION}"
            )
        payload_end = _HEADER.size + int(payload_len)
        if in_path.stat().st_size != payload_end:
            raise ValueError(f"{path} has an invalid SimInstance table length")

        cursor = _FileCursor(handle, payload_end)
        for _ in range(min(offset, int(total_count))):
            _read_sim_instance(cursor)
        remaining = max(int(total_count) - offset, 0)
        read_count = remaining if requested_count is None else min(requested_count, remaining)
        instances = [_read_sim_instance(cursor) for _ in range(read_count)]
        if requested_count is None and offset == 0:
            cursor.expect_finished(path)
        return instances


def read_sim_instances_by_index(path: str | Path, indices: list[int] | tuple[int, ...]) -> tuple[dict[int, SimInstance], int]:
    requested = sorted({int(index) for index in indices})
    if any(index < 0 for index in requested):
        raise ValueError("indices must be non-negative")
    if not requested:
        return {}, 0

    in_path = Path(path)
    with in_path.open("rb") as handle:
        header = handle.read(_HEADER.size)
        if len(header) < _HEADER.size:
            raise ValueError(f"{path} is too small to be a SimInstance table")
        magic, version, total_count, payload_len = _HEADER.unpack(header)
        if magic != SIM_INSTANCE_MAGIC:
            raise ValueError(f"{path} is not a SimInstance table")
        if version != SIM_INSTANCE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported SimInstance table version {version}; "
                f"expected {SIM_INSTANCE_FORMAT_VERSION}"
            )
        payload_end = _HEADER.size + int(payload_len)
        if in_path.stat().st_size != payload_end:
            raise ValueError(f"{path} has an invalid SimInstance table length")

        cursor = _FileCursor(handle, payload_end)
        selected: dict[int, SimInstance] = {}
        requested_pos = 0
        max_index = min(requested[-1], int(total_count) - 1)
        for index in range(max_index + 1):
            instance = _read_sim_instance(cursor)
            if requested_pos < len(requested) and index == requested[requested_pos]:
                selected[index] = instance
                requested_pos += 1
                while requested_pos < len(requested) and requested[requested_pos] >= int(total_count):
                    requested_pos += 1
        return selected, int(total_count)


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


class _FileCursor:
    def __init__(self, handle, payload_end: int):
        self.handle = handle
        self.payload_end = int(payload_end)

    def read(self, size: int) -> bytes:
        end = self.handle.tell() + int(size)
        if end > self.payload_end:
            raise ValueError("Unexpected end of SimInstance table")
        chunk = self.handle.read(int(size))
        if len(chunk) != int(size):
            raise ValueError("Unexpected end of SimInstance table")
        return chunk

    def expect_finished(self, path: str | Path) -> None:
        if self.handle.tell() != self.payload_end:
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


def _write_optional_string(buf: bytearray, value: str | None) -> None:
    _write_u8(buf, int(value is not None))
    if value is not None:
        _write_string(buf, value)


def _read_optional_string(cursor: _Cursor) -> str | None:
    if not _read_u8(cursor):
        return None
    return _read_string(cursor)


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
        behavior=behavior,
        controller=controller,
    )


def _write_target_initial(buf: bytearray, initial: TargetInitialState) -> None:
    _write_f32_array(buf, initial.position_w, (3,))
    _write_f32_array(buf, initial.velocity_w, (3,))


def _read_target_initial(cursor: _Cursor) -> TargetInitialState:
    return TargetInitialState(
        position_w=_read_f32_array(cursor, (3,)),
        velocity_w=_read_f32_array(cursor, (3,)),
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


def _write_noise_config(buf: bytearray, noise: NoiseConfig) -> None:
    for value in (
        noise.camera_image_delay_s,
        noise.pixel_noise_std_px[0],
        noise.pixel_noise_std_px[1],
        noise.dropout_probability,
        noise.sigma_img,
        noise.sigma_gyr,
        noise.sigma_acc,
        noise.sigma_b_gyr,
        noise.sigma_b_acc,
        noise.bias_init_std,
    ):
        _write_f32(buf, value)
    _write_i64(buf, noise.rng_seed)


def _read_noise_config(cursor: _Cursor) -> NoiseConfig:
    values = [_read_f32(cursor) for _ in range(10)]
    return NoiseConfig(
        camera_image_delay_s=values[0],
        pixel_noise_std_px=(values[1], values[2]),
        dropout_probability=values[3],
        sigma_img=values[4],
        sigma_gyr=values[5],
        sigma_acc=values[6],
        sigma_b_gyr=values[7],
        sigma_b_acc=values[8],
        bias_init_std=values[9],
        rng_seed=_read_i64(cursor),
    )


def _write_sim_config(buf: bytearray, config: SimConfig | None) -> None:
    _write_u8(buf, int(config is not None))
    if config is None:
        return
    _write_pursuer_params(buf, config.pursuer)
    _write_f32(buf, config.options.backend_dt)
    _write_u32(buf, config.options.action_substeps)
    _write_f32(buf, config.options.duration_s)
    _write_u8(buf, int(config.options.validation_dt is not None))
    if config.options.validation_dt is not None:
        _write_f32(buf, config.options.validation_dt)
    _write_string(buf, config.options.command_mode)
    _write_f32(buf, config.options.ctbr_rate_gain)
    _write_u8(buf, int(config.options.randomize_params))
    _write_u16(buf, len(config.targets))
    for target in config.targets:
        _write_target(buf, target)
    _write_u16(buf, len(config.cameras))
    for camera in config.cameras:
        _write_camera(buf, camera)
    _write_f32(buf, config.intercept_radius_m)
    _write_f32(buf, config.max_thrust_n)
    _write_f32(buf, config.max_rate_rps)
    _write_optional_array(
        buf,
        None if config.bounds_w is None else np.asarray(config.bounds_w, dtype=float),
        (3,),
    )
    _write_noise_config(buf, config.noise)
    _write_u8(buf, int(config.rendering))
    _write_render_config(buf, config.render)


def _read_sim_config(cursor: _Cursor) -> SimConfig | None:
    if not _read_u8(cursor):
        return None
    pursuer = _read_pursuer_params(cursor)
    backend_dt = _read_f32(cursor)
    action_substeps = _read_u32(cursor)
    duration_s = _read_f32(cursor)
    validation_dt = _read_f32(cursor) if _read_u8(cursor) else None
    command_mode = _read_string(cursor)
    ctbr_rate_gain = _read_f32(cursor)
    randomize_params = bool(_read_u8(cursor))
    targets = tuple(_read_target(cursor) for _ in range(_read_u16(cursor)))
    cameras = tuple(_read_camera(cursor) for _ in range(_read_u16(cursor)))
    intercept_radius_m = _read_f32(cursor)
    max_thrust_n = _read_f32(cursor)
    max_rate_rps = _read_f32(cursor)
    bounds_array = _read_optional_array(cursor, (3,))
    noise = _read_noise_config(cursor)
    rendering = bool(_read_u8(cursor))
    return SimConfig(
        pursuer=pursuer,
        options=SimOptions(
            backend_dt=backend_dt,
            action_substeps=action_substeps,
            duration_s=duration_s,
            validation_dt=validation_dt,
            command_mode=command_mode,
            ctbr_rate_gain=ctbr_rate_gain,
            randomize_params=randomize_params,
        ),
        targets=targets,
        cameras=cameras,
        intercept_radius_m=intercept_radius_m,
        max_thrust_n=max_thrust_n,
        max_rate_rps=max_rate_rps,
        bounds_w=None if bounds_array is None else tuple(float(x) for x in bounds_array),
        noise=noise,
        rendering=rendering,
        render=_read_render_config(cursor),
    )


def _write_render_config(buf: bytearray, config: RenderConfig) -> None:
    _write_optional_string(buf, config.camera_id)
    _write_string(buf, config.backend)
    _write_string(buf, config.platform)
    _write_string(buf, config.scene_id)
    _write_u32(buf, config.timeout_ms)
    _write_u8(buf, int(config.fail_on_error))


def _read_render_config(cursor: _Cursor) -> RenderConfig:
    return RenderConfig(
        camera_id=_read_optional_string(cursor),
        backend=_read_string(cursor),
        platform=_read_string(cursor),
        scene_id=_read_string(cursor),
        timeout_ms=_read_u32(cursor),
        fail_on_error=bool(_read_u8(cursor)),
    )


def _write_sim_instance(buf: bytearray, instance: SimInstance) -> None:
    _write_i64(buf, instance.seed)
    _write_pursuer_initial(buf, instance.pursuer_initial)
    _write_u16(buf, len(instance.target_initials))
    for initial in instance.target_initials:
        _write_target_initial(buf, initial)
    _write_sim_config(buf, instance.config)


def _read_sim_instance(cursor: _Cursor) -> SimInstance:
    seed = _read_i64(cursor)
    pursuer_initial = _read_pursuer_initial(cursor)
    target_initials = tuple(_read_target_initial(cursor) for _ in range(_read_u16(cursor)))
    config = _read_sim_config(cursor)
    return SimInstance(
        seed=seed,
        pursuer_initial=pursuer_initial,
        target_initials=target_initials,
        config=config,
    )
