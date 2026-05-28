# Rendering Integration

This package owns the FPV rendering stack and its native interface to `csim`.

The boundary is intentionally narrow:

- `csim` includes only `backends/csim/rendering/include/liftoff_render_api.h`.
- `backends/csim/rendering/native` owns renderer backends, frame memory, and platform APIs.
- Higher-level renderer frontends should call into the native layer, not into
  `csim`.
- Windows-specific code lives under `backends/csim/rendering/native/platform/win32`.
- Linux-specific code lives under `backends/csim/rendering/native/platform/linux`.

## Boundary

`csim` submits a `LiftoffRenderFrameRequest` containing sim time, drone state,
one selected camera, and target states. The native renderer returns a borrowed
`LiftoffRenderFrame`; `SimEngine` copies the pixels into its own camera-output
buffers before releasing the frame.

The API does not expose Unity scene names, C# types, Windows handles, shared
memory names, or platform-specific packet layouts.

## Initial Build Order

1. Build the no-op native engine and link it into tests.
2. Wire `csim` to call the API when `SimConfig.rendering` is enabled.
3. Grow the repo-owned renderer toward the Liftoff FPV camera model.
4. Add platform-specific shared-memory/event backends behind `platform/*` only
   if a separate renderer process becomes necessary.
