# Rendering Architecture

This document maps the current renderer setup after moving rendering under
`backends/csim`.

## Directory Layout

```text
backends/csim/
|-- sim_engine.c/.h              C SimEngine owns render invocation
|-- camera_sim.h                 CameraOutput includes render frame metadata
|-- bindings/
|   |-- puffer_c.py              Python binding configures SimEngine rendering
|   `-- types/sim_engine.py      SimConfig.rendering + RenderConfig
`-- rendering/
    |-- include/                 Public C ABI used by SimEngine
    |   |-- liftoff_render_api.h
    |   |-- liftoff_render_types.h
    |   `-- liftoff_render_errors.h
    |-- native/
    |   |-- src/render_engine.cpp
    |   `-- platform/
    |       |-- linux/
    |       `-- win32/
    `-- python/
        |-- build_native.py      Builds libliftoff_render_native.so
        |-- ctypes_api.py        Python mirror of render C ABI
        |-- engine.py            Direct Python NativeRenderEngine wrapper
        |-- config.py            Backend/platform/status enum mapping
        |-- episode.py           Generates saved PPM frame sequences
        `-- liftoff_assets.py    Optional Liftoff mesh export/cache helpers
```

## Ownership Boundaries

```text
Python app/test code
        |
        v
backends.csim.bindings
        |
        |  typed objects only:
        |  SimConfig, SimInstance, TargetConfig, CameraConfig
        v
SimEngine C API
        |
        |  only renderer dependency visible to csim:
        |  backends/csim/rendering/include/liftoff_render_api.h
        v
Renderer native ABI
        |
        v
Renderer backend implementation
```

Important rule: Python callers should not call `sim_engine_*` directly outside
`backends/csim/bindings`, and `csim` should not depend on renderer internals
outside the public C ABI.

## Config Shape

```text
SimConfig
|-- rendering: bool
|   `-- False: SimEngine returns geometric camera observations only
|   `-- True:  SimEngine also asks renderer for selected camera frames
|
`-- render: RenderConfig
    |-- camera_id: str | None
    |   `-- None means render every captured camera output
    |   `-- "front" selects the matching configured CameraConfig id
    |-- backend: "software" | "unity" | "none"
    |-- platform: "auto" | "linux" | "windows"
    |-- scene_id: str
    |-- timeout_ms: int
    `-- fail_on_error: bool
```

Example:

```text
SimConfig(
    pursuer=...,
    targets=(TargetConfig(...),),
    cameras=(CameraConfig(id="front", ...),),
    rendering=True,
    render=RenderConfig(camera_id="front", backend="software"),
)
```

## Runtime Flow

```text
PufferSimEngineBackend.reset(...)
        |
        | converts typed Python config to C structs
        v
sim_engine_init
sim_engine_set_targets
sim_engine_set_cameras
sim_engine_configure_rendering
        |
        | creates LiftoffRenderEngine when SimConfig.rendering is True
        v
liftoff_render_engine_create
```

At capture time:

```text
PufferSimEngineBackend._collect_camera_outputs
        |
        v
sim_engine_collect_camera_outputs
        |
        | for each due CameraSim:
        |   1. compute CameraObservation
        |   2. if selected for rendering:
        |      build LiftoffRenderFrameRequest
        v
liftoff_render_frame
        |
        | renderer returns borrowed LiftoffRenderFrame
        v
SimEngine copies RGB bytes into owned frame buffer
        |
        v
CameraOutput
|-- observation
|-- render_status
|-- has_frame
|-- frame_width_px
|-- frame_height_px
|-- frame_channels
|-- frame_stride_bytes
|-- frame_byte_count
`-- frame_rgb
```

## Frame Request Shape

```text
LiftoffRenderFrameRequest
|-- drone: LiftoffRenderDroneState
|   |-- t
|   |-- sequence_id
|   |-- position_w
|   |-- velocity_w
|   |-- quat_xyzw
|   `-- body_rates_b
|
|-- camera: LiftoffRenderCameraState
|   |-- camera_id
|   |-- position_b
|   |-- body_to_camera[9]
|   |-- width_px / height_px
|   |-- fx_px / fy_px
|   |-- cx_px / cy_px
|   `-- hfov_rad / vfov_rad
|
`-- targets: LiftoffRenderTargetState[]
    |-- target_id
    |-- position_w
    |-- velocity_w
    `-- radius_m
```

## Native Renderer Backends

```text
RenderConfig.backend
|
|-- "none"
|   `-- liftoff_render_frame -> LIFTOFF_RENDER_DISABLED
|
|-- "software"
|   `-- render_engine.cpp rasterizes RGB bytes in-process
|
`-- "unity"
    `-- currently returns LIFTOFF_RENDER_BACKEND_UNAVAILABLE
```

The software renderer is intentionally in the native renderer, not in
`SimEngine`. `SimEngine` prepares sim state and stores output bytes; rendering
logic stays behind `liftoff_render_frame`.

## Build Flow

```text
Python binding load
        |
        v
backends/csim/rendering/python/build_native.py
        |
        | builds:
        v
backends/csim/rendering/native/_build/libliftoff_render_native.so
        |
        v
backends/csim/bindings/puffer_c.py
        |
        | links csim shared library against native renderer
        v
backends/csim/bindings/_build/libpuffer_sim_core.so
```

Generated `_build/` directories are local build artifacts.

## Episode Generation

```text
python -m backends.csim.rendering.python.episode --out-dir .runs/example
        |
        v
.runs/example/
|-- episode.json
`-- frames/
    |-- frame_0000.ppm
    |-- frame_0001.ppm
    `-- ...
```

`episode.py` is a convenience producer for saved frame sequences. It uses
`PufferSimEngineBackend` with `SimConfig.rendering=True`, then writes each
`CameraOutput.frame_rgb` payload as a PPM file.

To make an MP4 from the PPM sequence:

```text
ffmpeg -y -framerate 30 \
  -i .runs/example/frames/frame_%04d.ppm \
  -c:v libx264 -pix_fmt yuv420p \
  .runs/example/episode.mp4
```

## Test Coverage Map

```text
tests/rendering/test_native_api.py
    direct NativeRenderEngine ctypes calls

tests/test_puffer_backend_smoke.py
    SimEngine rendering through Python binding

tests/rendering/test_episode.py
    saved PPM episode generation

tests/rendering/render_integration_test.py
    generated frame sequence metadata and frame-content checks
```
