# Rendering Handoff Proposal

Current committed baseline:

`317841c Add native rendering API scaffold`

This note started from the original `rendering/` scaffold. Rendering now lives under
`backends/csim/rendering/`, and `SimEngine` owns the native render call.

The existing dirty worktree has unrelated and older changes from prior attempts. A new worktree should ideally start from `317841c` or cherry-pick only that commit.

## Current Rendering Structure

```text
backends/csim/rendering/
  README.md
  include/
    liftoff_render_api.h
    liftoff_render_errors.h
    liftoff_render_types.h
  native/
    CMakeLists.txt
    src/
      render_engine.cpp
    platform/
      README.md
      render_platform.h
      linux/render_platform_linux.cpp
      win32/render_platform_win32.cpp
```

Purpose:

- `backends/csim/rendering/include/*` is the public C ABI that `csim` should depend on.
- `backends/csim/rendering/native/src/render_engine.cpp` currently implements a no-op engine.
- `backends/csim/rendering/native/platform/*` boxes Windows/Linux transport details.
- No Unity project exists yet.
- `csim` integration exists through `SimConfig.rendering` and `RenderConfig`.

## Important Config Note

These fields currently exist in `SimConfig`, but they are from the abandoned TCP/BepInEx bridge path:

```python
rendering: bool = False
render: RenderConfig = field(default_factory=RenderConfig)
```

Do not build the implementation around TCP endpoints. The native-renderer config is:

```python
@dataclass(frozen=True)
class RenderConfig:
    camera_id: str | None = None
    backend: str = "software"
    platform: str = "auto"
    scene_id: str = "liftoff_fpv_0"
    timeout_ms: int = 16
    fail_on_error: bool = False
```

Then use:

```python
@dataclass(frozen=True)
class SimConfig:
    ...
    rendering: bool = False
    render: RenderConfig = field(default_factory=RenderConfig)
```

Given the project instruction against temporary compatibility, prefer the nested `RenderConfig` and update call sites directly instead of maintaining both old and new render settings.

## Target Architecture

```text
SimEngine / csim
  -> backends/csim/rendering/include/liftoff_render_api.h
  -> backends/csim/rendering/native/src/render_engine.cpp
  -> backends/csim/rendering/native/platform/{win32,linux}
  -> Unity native plugin / transport
  -> Unity C# thin glue
  -> Unity Camera + shaders + RenderTexture
  -> native frame buffer
  -> csim CameraOutput
```

Rules:

- `csim` includes only `liftoff_render_api.h`.
- Platform code is boxed under `backends/csim/rendering/native/platform`.
- Unity-specific code should live under `backends/csim/rendering/unity`.
- C# should be thin glue, not the core API.
- Native C++ owns transport, frame lifetime, status codes, and ABI.
- `csim` should continue to produce geometric camera observations even if rendering fails.

## Recommended Next Steps

### 1. Normalize Render Config

Update the Python `SimConfig` away from old bridge semantics.

Files likely affected:

```text
backends/csim/bindings/types/sim_engine.py
backends/csim/generator/instance_store.py
tests/test_sim_instance_store.py
tests/test_puffer_backend_smoke.py
```

### 2. Add ctypes Bindings For New Render API

Create Python ctypes mirrors for `backends/csim/rendering/include/liftoff_render_types.h`.

Suggested files:

```text
backends/csim/rendering/python/
  __init__.py
  ctypes_api.py
  engine.py
  config.py
```

The initial implementation should load `backends/csim/rendering/native/_build/libliftoff_render_native.so` and call:

```c
liftoff_render_engine_create
liftoff_render_frame
liftoff_render_release_frame
liftoff_render_engine_destroy
```

For now it will return `LIFTOFF_RENDER_BACKEND_UNAVAILABLE` unless backend is `LIFTOFF_RENDER_BACKEND_NONE`.

### 3. Build Native Library From Python

The repo currently builds `csim` ad hoc in `backends/csim/bindings/puffer_c.py` using `cc`. `cmake` is not installed on this machine.

Add a direct `c++` build helper for the native renderer, similar to the existing `csim` build style.

Possible file:

```text
backends/csim/rendering/python/build_native.py
```

It should compile:

```text
backends/csim/rendering/native/src/render_engine.cpp
backends/csim/rendering/native/platform/linux/render_platform_linux.cpp
```

on Linux/WSL, and the Win32 file when building on Windows.

Use generated outputs under:

```text
backends/csim/rendering/native/_build/
```

Do not commit generated binaries.

### 4. Wire `csim` To Renderer Interface

There are two viable integration stages.

Stage A, Python-side only:

- Keep `csim` C untouched.
- After `sim_engine_collect_camera_outputs`, Python builds `LiftoffRenderFrameRequest` and calls the renderer through ctypes.
- This is fastest and avoids changing `SimEngine` layout immediately.

Stage B, true C-level integration:

- Add a `LiftoffRenderEngine* render_engine` pointer or opaque render callback to `SimEngine`.
- `sim_engine_collect_camera_outputs` calls `liftoff_render_frame`.
- Extend `CameraOutput` with render status and frame info.

Stage B is the production target. If Stage A is used first, keep it clearly named as a test harness, not the production path.

For Stage B, files:

```text
backends/csim/sim_engine.h
backends/csim/sim_engine.c
backends/csim/camera_sim.h
backends/csim/bindings/puffer_c.py
```

Add to `CameraOutput`:

```c
int render_status;
int has_frame;
int frame_width_px;
int frame_height_px;
int frame_channels;
int frame_stride_bytes;
const unsigned char* frame_rgb;
```

It already has most frame fields. Add `render_status` and stride.

Add engine methods:

```c
void sim_engine_set_render_engine(SimEngine* engine, LiftoffRenderEngine* render_engine);
void sim_engine_set_render_camera_id(SimEngine* engine, int camera_id);
```

Or use a small config struct:

```c
typedef struct {
    int enabled;
    int camera_id;
    int fail_on_error;
} SimRenderConfig;
```

Prefer a config struct if more fields are coming.

### 5. Keep Render Failure Non-Fatal By Default

Rendering must not break physics stepping unless explicitly requested.

Default behavior:

```text
render ok -> CameraOutput.has_frame = 1
render unavailable/timeout -> CameraOutput.has_frame = 0, render_status set
```

Only if `fail_on_error=True` should Python raise.

### 6. Add Tests Before Unity

Add tests around the native no-op renderer and `csim` integration before starting Unity.

Suggested tests:

```text
tests/rendering/test_native_api.py
tests/test_puffer_backend_render_config.py
tests/test_puffer_backend_smoke.py
```

Test cases:

- `LIFTOFF_RENDER_BACKEND_NONE` returns disabled.
- `LIFTOFF_RENDER_BACKEND_UNITY` returns backend unavailable in stub.
- `SimEngine` only requests render for selected camera id.
- Capture output still exists when render backend unavailable.
- `fail_on_error=True` raises at Python boundary.
- `frame_rgb is None` when no frame is available.

### 7. Run Required Verification

Per `AGENTS.md`, after significant sim logic changes run:

```bash
python -m pytest -q tests/test_sim_instance_store.py tests/test_puffer_backend_smoke.py
```

Also run focused tests:

```bash
python -m pytest tests/test_puffer_backend_smoke.py tests/test_sim_instance_store.py
```

If adding rendering tests:

```bash
python -m pytest tests/rendering tests/test_puffer_backend_smoke.py tests/test_sim_instance_store.py
```

### 8. Unity Implementation After C Boundary Is Stable

Once `csim` can call the native API and receive no-op/unavailable statuses correctly, add:

```text
backends/csim/rendering/unity/
  LiftoffFpvRenderer/
    Assets/
      LiftoffFpv/
        Scripts/
          NativeRenderBridge.cs
          RenderLoop.cs
          CameraController.cs
          FrameReadback.cs
        Shaders/
          Fisheye.shader
          CameraNoise.shader
          FpvPostProcess.shader
        Scenes/
          FpvRenderScene.unity
```

Unity responsibilities:

- Create scene/camera.
- Pull latest request from native plugin/transport.
- Apply drone pose, camera body offset, FOV.
- Render target/world objects.
- Apply fisheye/noise/postprocess.
- Write frame to native buffer.
- Signal completion.

Native responsibilities:

- Own request/response protocol.
- Own platform transport.
- Own frame memory lifetime.
- Normalize errors.

### 9. Windows Boundary Design

Under `backends/csim/rendering/native/platform/win32`, implement one transport type first. Recommended initial Windows transport:

```text
Named shared memory + named events
```

Why:

- Fast enough for frame buffers.
- Works well with Unity and native plugin.
- Easier to make fixed ABI than TCP JSON.
- Can be mirrored later on Linux with POSIX shared memory plus `eventfd`, futex, or `poll`.

Windows files to add later:

```text
backends/csim/rendering/native/platform/win32/
  win32_handles.h
  win32_shared_memory.cpp
  win32_events.cpp
  win32_transport.cpp
```

Keep all Win32 includes, including `windows.h`, in this directory only.

### 10. Avoid

Do not:

- resurrect `tools/liftoff_bridge` as the production path
- keep TCP endpoint semantics as the main API
- expose Unity class names in `SimConfig`
- add Windows headers to `csim`
- copy decompiled Liftoff source verbatim
- make multiple rendering implementations with different behavior

## Current Validation State

Before commit, the native scaffold was verified with direct compile because `cmake` is missing:

```bash
c++ -std=c++17 -O2 -fPIC -shared \
  -Ibackends/csim/rendering/include \
  -Ibackends/csim/rendering/native/platform \
  -Ibackends/csim/rendering/native/src \
  backends/csim/rendering/native/src/render_engine.cpp \
  backends/csim/rendering/native/platform/linux/render_platform_linux.cpp \
  -o /tmp/liftoff-render-check/libliftoff_render_native.so
```

That passed.

`git diff --cached --check` passed before commit.

## Suggested First Prompt For Next Agent

Start from commit `317841c`. Implement the next integration step for the new `rendering/` API: replace old `SimConfig` bridge fields with a native `RenderConfig`, add Python ctypes bindings/build helper for `liftoff_render_api.h`, and wire `PufferSimEngineBackend` to call the no-op renderer without Unity. Keep Windows/Linux platform logic boxed in `backends/csim/rendering/native/platform`. Run focused tests and the red balloon replay required by `AGENTS.md`.
