# Rendering Integration

This package owns the Unity FPV rendering stack and its native interface to
`csim`.

The boundary is intentionally narrow:

- `csim` includes only `rendering/include/liftoff_render_api.h`.
- `rendering/native` owns transports, shared memory, platform APIs, and Unity
  plugin handshakes.
- Unity C# code should call into the native layer, not into `csim`.
- Windows-specific code lives under `rendering/native/platform/win32`.
- Linux-specific code lives under `rendering/native/platform/linux`.

## Boundary

`csim` submits a `LiftoffRenderFrameRequest` containing sim time, drone state,
one selected camera, and target states. The renderer returns a borrowed
`LiftoffRenderFrame` whose pixel memory is owned by the render engine until the
next render call or explicit release.

The API does not expose Unity scene names, C# types, Windows handles, shared
memory names, or transport-specific packet layouts.

## Initial Build Order

1. Build the no-op native engine and link it into tests.
2. Wire `csim` to call the API when `render_frames` is enabled.
3. Add Windows Unity transport behind `platform/win32`.
4. Add Linux transport behind `platform/linux` if/when Unity moves there.
5. Add the Unity project and C# bridge against the same native API.
