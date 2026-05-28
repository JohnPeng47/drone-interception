# Rendering Diagrams (v2)

A second pass at applying templates from `docs/diagram_types.md` to
`backends/csim/rendering`. Sibling file `rendering_diagrams.md` covers the
ownership boundary, capture-time swimlane, C ABI map, frame schema,
backend adapter, software render pipeline, target-visual escape hatch,
resource lifecycle, build DAG, and episode flow. This file deliberately
picks diagram types not used there so the two files compose.

## 1. Rendering Subsystem Stack

Diagram type: 3. Layered Stack.

```text
+-- Tooling / scripts --------------------------------------------+
|  backends.csim.rendering.python.episode                         |
|  backends.csim.rendering.python.liftoff_assets                  |
+--------------------+--------------------------------------------+
                     |
                     v
+-- Python wrappers --------------------------------------------+
|  engine.py        NativeRenderEngine, RenderFrameResult       |
|  config.py        backend/platform/status enums               |
|  ctypes_api.py    LiftoffRender* Structures, load_library     |
|  build_native.py  compiles libliftoff_render_native            |
+--------------------+--------------------------------------------+
                     |
                     v
+-- Public C ABI -----------------------------------------------+
|  include/liftoff_render_api.h        engine/frame entrypoints |
|  include/liftoff_render_types.h      LiftoffRender* structs   |
|  include/liftoff_render_errors.h     LiftoffRenderStatus       |
+--------------------+--------------------------------------------+
                     |
                     v
+-- Native implementation --------------------------------------+
|  native/src/render_engine.cpp        software renderer        |
|  native/platform/render_platform.h   PlatformTransport iface  |
|  native/platform/linux/  win32/      stub backend transports  |
+---------------------------------------------------------------+
```

## 2. C ABI Vs Python Ctypes Mirror

Diagram type: 10. Side-by-Side Comparison.

```text
include/liftoff_render_types.h                python/ctypes_api.py
------------------------------                ---------------------
+--- LiftoffRenderConfig ---------+           +--- LiftoffRenderConfig ----------+
| backend     : enum int          |           | backend     : c_int              |
| platform    : enum int          |           | platform    : c_int              |
| timeout_ms  : uint32_t          |  <---->   | timeout_ms  : c_uint32           |
| flags       : uint32_t          |           | flags       : c_uint32           |
| scene_id    : char[256]         |           | scene_id    : c_char * 256       |
+---------------------------------+           +----------------------------------+

+--- LiftoffRenderCameraState ---+            +--- LiftoffRenderCameraState ---+
| camera_id     : uint32_t       |            | camera_id     : c_uint32       |
| position_b    : Vec3           |            | position_b    : Vec3           |
| body_to_camera: double[9]      |  <---->    | body_to_camera: c_double * 9   |
| width/height  : uint32_t       |            | width/height  : c_uint32       |
| fx/fy/cx/cy   : double         |            | fx/fy/cx/cy   : c_double       |
| h/vfov_rad    : double         |            | h/vfov_rad    : c_double       |
+--------------------------------+            +--------------------------------+

+--- LiftoffRenderFrame ----------+           +--- LiftoffRenderFrame -----------+
| sequence_id  : uint64_t         |           | sequence_id  : c_uint64          |
| width/height : uint32_t         |  <---->   | width/height : c_uint32          |
| channels     : uint32_t         |           | channels     : c_uint32          |
| stride_bytes : uint32_t         |           | stride_bytes : c_uint32          |
| pixels       : const uint8_t*   |           | pixels       : POINTER(c_uint8)  |
| pixel_bytes  : size_t           |           | pixel_bytes  : c_size_t          |
+---------------------------------+           +----------------------------------+

Rule: any field added to the C side must be mirrored field-for-field
in ctypes_api.py and converted in engine.py before render_frame.
```

## 3. LiftoffRenderStatus Codes

Diagram type: 4. Definition List / Glossary.

```text
+- LIFTOFF_RENDER_OK = 0 --------------------------------------+
|  Frame produced. pixels and pixel_bytes are valid until      |
|  liftoff_render_release_frame returns.                        |
+--------------------------------------------------------------+
+- LIFTOFF_RENDER_DISABLED = 1 --------------------------------+
|  Engine configured with backend=NONE. liftoff_render_frame    |
|  zeroes the frame and returns immediately.                    |
+--------------------------------------------------------------+
+- LIFTOFF_RENDER_BACKEND_UNAVAILABLE = 2 ---------------------+
|  Selected backend is not built in this binary. Today the      |
|  Unity backend always returns this; Linux/Win32 transports    |
|  also stub-return it.                                         |
+--------------------------------------------------------------+
+- LIFTOFF_RENDER_TIMEOUT = 3 ---------------------------------+
|  Renderer did not complete within LiftoffRenderConfig.        |
|  timeout_ms. Not produced by the software backend.            |
+--------------------------------------------------------------+
+- LIFTOFF_RENDER_INVALID_REQUEST = 4 -------------------------+
|  Null engine/request/drone/camera/frame pointer, or camera    |
|  width/height is zero or exceeds 8192.                        |
+--------------------------------------------------------------+
+- LIFTOFF_RENDER_FRAME_DROPPED = 5 ---------------------------+
|  Reserved for backends that intentionally skip frames under   |
|  back-pressure. Not produced by software backend.             |
+--------------------------------------------------------------+
+- LIFTOFF_RENDER_INTERNAL_ERROR = 6 --------------------------+
|  Catch-all for renderer-internal failure paths.               |
+--------------------------------------------------------------+
```

## 4. Drone Variant Catalog

Diagram type: 5. Table / Matrix.

```text
+----------------------------------+---------------------+----------------+---------------+
| variant.name                     | camera mesh          | prop family    | motor scale   |
+----------------------------------+---------------------+----------------+---------------+
| vortex_dal_xnova_runcam          | runcam (id 303)      | DAL tri        | 8.0           |
| vortex_racekraft_xnova_hs1177    | HS1177 box (id 40)   | RaceKraft tri  | 8.8           |
| vortex_gemfan_xnova_actioncam    | action cam (id 316)  | Gemfan bullnose| 8.4           |
| vortex_dal_heavy_actioncam       | action cam (id 316)  | DAL heavy      | 10.5          |
| vortex_racekraft_low_cam         | low FPV cam (id 306) | RaceKraft tri  | 7.2           |
+----------------------------------+---------------------+----------------+---------------+

All variants share FRAME_MESHES (5 frame parts, battery, 2 straps) plus
4 motors at MOTOR_OFFSETS and 4 props at PROP_OFFSETS.
```

## 5. Episode Frame Sequence

Diagram type: 6. Sequence Diagram.

```text
  episode.py        PufferSim          sim_engine.c        render_engine.cpp
  ----------        ---------          ------------        -----------------
     |                  |                   |                      |
     | reset(...)       |                   |                      |
     |----------------->|                   |                      |
     |                  | sim_engine_init    |                      |
     |                  |------------------>|                      |
     |                  | configure_rendering                       |
     |                  |------------------>| engine_create        |
     |                  |                   |--------------------->|
     |                  |                   |<---------------------|
     |                  |                   |     LiftoffRenderEngine*
     |                  |                   |                      |
     |  for each step: step_ctbr / read snapshot                   |
     |----------------->|                   |                      |
     |                  | collect_camera_outputs                    |
     |                  |------------------>|                      |
     |                  |                   | build FrameRequest    |
     |                  |                   | render_frame          |
     |                  |                   |--------------------->|
     |                  |                   |                      | render_software
     |                  |                   |<---------------------| frame.pixels
     |                  |                   | copy into owned buf   |
     |                  |                   | release_frame         |
     |                  |                   |--------------------->|
     |                  |<------------------| CameraOutput          |
     |<-----------------|  frame_rgb         |                      |
     | write PPM file   |                   |                      |
     |                  |                   |                      |
```

## 6. liftoff_render_frame Backend Fork

Diagram type: 7. Fork / Decision Point.

```text
liftoff_render_frame(engine, request, frame)
   |
   |-- request/drone/camera/frame null? --YES--> LIFTOFF_RENDER_INVALID_REQUEST
   |
   |-- backend == BACKEND_NONE? ---------YES--> LIFTOFF_RENDER_DISABLED
   |                                            (frame already zeroed)
   |
   |-- backend == BACKEND_SOFTWARE? -----YES--> render_software
   |                                              |-- width/height 0 or > 8192?
   |                                              |     YES -> INVALID_REQUEST
   |                                              |
   |                                              `-- rasterize, return OK
   |                                                  frame.pixels borrowed
   |
   `-- otherwise (BACKEND_UNITY, ...) --------- > LIFTOFF_RENDER_BACKEND_UNAVAILABLE
```

## 7. Drone Variant Composition Tree

Diagram type: 9. Tree / Hierarchy.

```text
DroneVariant("vortex_dal_xnova_runcam")
  |
  +-- FRAME_MESHES (shared) ----- resources.assets:305  vortex_frame
  |                               resources.assets:312  vortex_frame
  |                               resources.assets:319  vortex_frame
  |                               resources.assets:323  vortex_frame
  |                               resources.assets:327  vortex_frame
  |                               resources.assets:302  battery
  |                               resources.assets:310  strap (L)
  |                               resources.assets:314  strap (R)
  |
  +-- camera mesh --------------- resources.assets:303  camera
  |
  +-- _prop_pair_specs ---------- resources.assets:301  prop (front-right)
  |                               resources.assets:309  prop (front-left)
  |                               resources.assets:309  prop (rear-left)
  |                               resources.assets:301  prop (rear-right)
  |
  `-- _xnova_motor_specs -------- sharedassets5.assets:37  motor   (x4 positions)
                                  sharedassets5.assets:43  motor   (x4 positions)
```

## 8. NativeRenderEngine Lifecycle

Diagram type: 8. State Machine.

```text
                           NativeRenderEngine(config)
                                  |
                                  v
                            +------------+
              render_frame  | UNINIT     |
            <----- err -----| _engine = 0|
                            +-----+------+
                                  |
            liftoff_render_engine_create -> OK
                                  |
                                  v
                            +------------+   render_frame      +-------------+
                            |  OPEN      |------------------>  | RENDERING   |
                            | _engine ok |                     | frame buf   |
                            |            |<------------------  | borrowed    |
                            +-----+------+   release_frame     +-------------+
                                  |
                          close() / __del__ / __exit__
                                  |
                                  v
                            +------------+
                            |  CLOSED    |
                            | _engine = 0|
                            +------------+
                                  |
                       render_frame -> RenderError("closed")
```

## 9. Software Frame Composition Onion

Diagram type: 23. Onion / Concentric Rings.

```text
+----------------------------------------------------------------------+
|  HUD: gate corner ticks + center crosshair                            |
|  +----------------------------------------------------------------+   |
|  |  Postprocess: vignette + scanlines + chroma + grain             |   |
|  |  +----------------------------------------------------------+   |   |
|  |  |  Target raster pass (per request->targets[i])             |   |   |
|  |  |    +----------------------------------------------------+ |   |   |
|  |  |    |  Z-buffered mesh triangles (if mesh loaded)         | |   |   |
|  |  |    |     fallback: draw_drone_target procedural quad     | |   |   |
|  |  |    +----------------------------------------------------+ |   |   |
|  |  +----------------------------------------------------------+   |   |
|  |                                                                  |   |
|  |  Sky/ground pass: per-pixel ray cast through camera intrinsics  |   |
|  +----------------------------------------------------------------+   |
+----------------------------------------------------------------------+

Innermost layer runs first (sky/ground fills the buffer), and each
enclosing layer composes on top. Postprocess and HUD see the final
composited pixels, including target rasterization.
```

## 10. Per-Target Fan-Out In render_software

Diagram type: 17. Fan-Out / Fan-In.

```text
                                         +------------------------------+
                                  +----->| target[0]: draw_mesh_target  |---+
                                  |      | (fallback: drone procedural) |   |
                                  |      +------------------------------+   |
                                  |                                          |
   request->targets[]             |      +------------------------------+   |    shared
   target_count = N        -------+----->| target[1]: draw_mesh_target  |---+--> engine->pixels
   request->drone, camera         |      +------------------------------+   |    z-buffer
   camera_pos_w (precomputed)     |                                          |
                                  |      +------------------------------+   |
                                  +----->| target[N-1]: ...             |---+
                                         +------------------------------+

Each target is projected into camera space independently using the same
drone pose, camera intrinsics, and camera_pos_w. The shared z-buffer
and pixel vector serialize the per-target writes; no parallelism today,
but the spatial fan-out is the structural shape.
```
