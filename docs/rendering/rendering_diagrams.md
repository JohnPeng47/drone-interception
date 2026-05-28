# Rendering Diagrams

These diagrams apply the spatial templates from `docs/diagram_types.md` to
`backends/csim/rendering`.

## 1. Renderer Ownership Boundary

Diagram type: 14. Boundary / Containment Region.

```text
+-------------------------------------------------------------------------+
| PYTHON SIM CALLERS                                                       |
|                                                                         |
|  PufferSimEngineBackend                                                  |
|  SimConfig(rendering=True, render=RenderConfig(...))                     |
|  CameraOutput.frame_rgb consumers                                        |
|                                                                         |
|  Rule: callers use backends/csim/bindings typed objects.                 |
+================================== API ===================================+
|                                                                         |
| BACKENDS/CSIM/BINDINGS                                                   |
|                                                                         |
|  puffer_c.py                                                             |
|  - converts RenderConfig to LiftoffRenderConfig                          |
|  - calls sim_engine_configure_rendering                                  |
|  - exposes render status and copied RGB bytes                            |
|                                                                         |
+========================== SIMENGINE C BOUNDARY ==========================+
|                                                                         |
| BACKENDS/CSIM                                                            |
|                                                                         |
|  sim_engine.c/.h                                                         |
|  - owns camera capture timing                                            |
|  - builds LiftoffRenderFrameRequest                                      |
|  - copies borrowed renderer pixels into CameraOutput storage             |
|                                                                         |
|  Only renderer include visible here:                                     |
|  backends/csim/rendering/include/liftoff_render_api.h                    |
|                                                                         |
+========================== RENDERER C ABI ================================+
|                                                                         |
| BACKENDS/CSIM/RENDERING                                                  |
|                                                                         |
|  include/      stable C ABI                                              |
|  native/       renderer implementation and platform transports           |
|  python/       direct renderer tools, ctypes mirror, build helper        |
+-------------------------------------------------------------------------+
```

## 2. Capture-Time Runtime Flow

Diagram type: 19. Swimlane.

```text
+-------------------+---------------+------------------+------------------+-------------------+
| Python binding    | collect        | convert C output | fail_on_error    | snapshot exposes  |
| puffer_c.py       | camera outputs | to dicts/bytes   | check            | CameraOutput data |
+---------+---------+---------------+------------------+------------------+-------------------+
          |
+---------v---------+---------------+------------------+------------------+-------------------+
| SimEngine         | camera due?    | geometric        | selected for     | copy renderer     |
| sim_engine.c      | per CameraSim  | observation      | rendering?       | RGB to owned buf  |
+---------+---------+-------+-------+---------+--------+---------+--------+----------+--------+
          |                 |                 |                  |                  ^
          |                 |                 v                  |                  |
          |                 |       LiftoffRenderFrameRequest     |                  |
          |                 |       drone + camera + targets      |                  |
          |                 |                 |                  |                  |
+---------v---------+-------+---------+-------v--------+---------v--------+----------+--------+
| Renderer C ABI    | liftoff_render_frame                  borrowed LiftoffRenderFrame        |
| liftoff_render_*  | returns status + pixels                liftoff_render_release_frame       |
+---------+---------+--------------------------+--------------------------+--------------------+
          |
+---------v---------+--------------------------+--------------------------+--------------------+
| Native backend    | backend dispatch          software renderer fills RGB vector             |
| render_engine.cpp | none/software/unity       status if disabled or unavailable              |
+-------------------+--------------------------+--------------------------+--------------------+
```

## 3. Public C ABI Map

Diagram type: 12. Namespace / API Map.

```text
backends/csim/rendering/include
+----------------------------------------------------------------------------+
| liftoff_render_api.h                                                        |
|                                                                            |
|  liftoff_render_engine_create(config, engine**)                             |
|  liftoff_render_engine_destroy(engine*)                                     |
|  liftoff_render_frame(engine*, request*, frame*)                            |
|  liftoff_render_release_frame(engine*, frame*)                              |
|                                                                            |
| liftoff_render_types.h                                                      |
|                                                                            |
|  LiftoffRenderConfig                                                        |
|  LiftoffRenderDroneState                                                    |
|  LiftoffRenderCameraState                                                   |
|  LiftoffRenderTargetState                                                   |
|  LiftoffRenderFrameRequest                                                  |
|  LiftoffRenderFrame                                                         |
|                                                                            |
| liftoff_render_errors.h                                                     |
|                                                                            |
|  LIFTOFF_RENDER_OK                                                          |
|  LIFTOFF_RENDER_DISABLED                                                    |
|  LIFTOFF_RENDER_BACKEND_UNAVAILABLE                                         |
|  LIFTOFF_RENDER_TIMEOUT                                                     |
|  LIFTOFF_RENDER_INVALID_REQUEST                                             |
|  LIFTOFF_RENDER_FRAME_DROPPED                                               |
|  LIFTOFF_RENDER_INTERNAL_ERROR                                              |
|  liftoff_render_status_string(status)                                       |
+----------------------------------------------------------------------------+
```

## 4. Frame Request And Response Shape

Diagram type: 2. Record / Schema Box.

```text
+-- LiftoffRenderFrameRequest ----------------------------------------------+
| drone   : LiftoffRenderDroneState*                                         |
| camera  : LiftoffRenderCameraState*                                        |
| targets : LiftoffRenderTargetState*                                        |
| count   : uint32_t                                                         |
+-----------------------------------+----------------------------------------+
                                    |
                                    v
+-- LiftoffRenderFrame ------------------------------------------------------+
| sequence_id  : uint64_t                                                    |
| width_px     : uint32_t                                                    |
| height_px    : uint32_t                                                    |
| channels     : uint32_t     current software renderer: 3 RGB channels      |
| stride_bytes : uint32_t     width_px * 3                                   |
| pixels       : const uint8_t* borrowed from LiftoffRenderEngine::pixels     |
| pixel_bytes  : size_t       copied by SimEngine before release             |
+----------------------------------------------------------------------------+
```

## 5. Backend Dispatch

Diagram type: 22. Adapter / Port-and-Plug.

```text
                              +--------------------------------------+
                              | liftoff_render_frame                 |
                              | stable C renderer port               |
                              +-------------------+------------------+
                                                  |
              +-----------------------------------+-----------------------------------+
              |                                   |                                   |
              v                                   v                                   v
   +----------------------+            +----------------------+            +----------------------+
   | backend=none         |            | backend=software     |            | backend=unity        |
   |                      |            |                      |            |                      |
   | returns              |            | render_software      |            | returns              |
   | DISABLED             |            | fills RGB in-process |            | BACKEND_UNAVAILABLE |
   +----------------------+            +----------+-----------+            +----------------------+
                                                  |
                                                  v
                                       +----------------------+
                                       | LiftoffRenderFrame   |
                                       | borrowed pixels      |
                                       +----------------------+
```

## 6. Native Software Render Pipeline

Diagram type: 15. Horizontal Processing Pipeline.

```text
+------------------+    +------------------+    +------------------+    +------------------+
| validate request |--->| allocate buffers |--->| sky/ground pass  |--->| target pass      |
| camera dimensions|    | pixels + zbuffer |    | per pixel ray    |    | mesh or fallback |
+------------------+    +------------------+    +------------------+    +--------+---------+
                                                                                |
                                                                                v
+------------------+    +------------------+    +------------------+    +-------+----------+
| fill frame       |<---| HUD overlay      |<---| postprocess      |<---| target raster   |
| metadata/pointer |    | gate + crosshair |    | vignette/grain   |    | zbuffered mesh  |
+------------------+    +------------------+    +------------------+    +------------------+
```

## 7. Target Visual Selection

Diagram type: 27. Escape Hatch / Exception Path.

```text
render_software target loop
  |
  v
draw_mesh_target
  |
  +----x mesh unavailable or did not write pixels ----> procedural quadcopter
  |                                                     |
  |                                                     v
  |                                          project target center
  |                                          draw arms, rotors, body
  |
  v
load_target_mesh once per engine
  |
  v
resolve OBJ path:
  LIFTOFF_RENDER_DRONE_MESH
  LIFTOFF_RENDER_ASSET_DIR/target_drone.obj
  .runs/liftoff_assets/target_drone.obj
  |
  v
parse v/usemtl/f records
  |
  v
raster triangles with zbuffer
```

## 8. Renderer Resource Lifecycle

Diagram type: 28. Resource Lifecycle.

```text
+-- acquire ----------------------------------------------------------------+
| sim_engine_configure_rendering                                            |
|   sim_engine_close_rendering                                              |
|   liftoff_render_engine_create                                            |
|   engine->render_engine = LiftoffRenderEngine*                            |
|                                                                           |
|  +-- use ---------------------------------------------------------------+  |
|  | sim_engine_collect_camera_outputs                                    |  |
|  |   sim_engine_render_camera_output                                    |  |
|  |     liftoff_render_frame                                             |  |
|  |     renderer returns borrowed frame.pixels                           |  |
|  |     SimEngine reallocs/copies into render_frame_buffers[output_idx]  |  |
|  |     CameraOutput.frame_rgb points at SimEngine-owned bytes           |  |
|  |     liftoff_render_release_frame zeros the borrowed frame struct     |  |
|  +---------------------------------------------------------------------+  |
|                                                                           |
| finally:                                                                  |
|   sim_engine_close_rendering                                              |
|     liftoff_render_engine_destroy                                         |
|     free render_frame_buffers                                             |
+---------------------------------------------------------------------------+
```

## 9. Build And Load Dependency Graph

Diagram type: 20. Dependency Graph (DAG).

```text
                         +-------------------------------+
                         | rendering/include/*.h         |
                         +---------------+---------------+
                                         |
              +--------------------------+--------------------------+
              |                                                     |
              v                                                     v
   +----------------------+                             +----------------------+
   | native/src           |                             | csim/sim_engine.c    |
   | render_engine.cpp    |                             | includes render API  |
   +----------+-----------+                             +----------+-----------+
              |                                                     |
              v                                                     v
   +----------------------+        links against        +----------------------+
   | build_native.py      |---------------------------->| puffer_c.py build    |
   | libliftoff_render_*  |                             | libpuffer_sim_core   |
   +----------+-----------+                             +----------+-----------+
              |                                                     |
              +--------------------------+--------------------------+
                                         v
                              +--------------------+
                              | Python test/tools  |
                              | load shared libs   |
                              +--------------------+
```

## 10. Episode Generation Flow

Diagram type: 1. Vertical Sequential Flow.

```text
python -m backends.csim.rendering.python.episode
  |
  v
ensure optional Liftoff target mesh cache
  |
  v
create PufferSimEngineBackend with SimConfig.rendering=True
  |
  v
reset with one target and one POV camera
  |
  v
for each frame:
  |
  +--> read selected CameraOutput.frame_rgb
  |    write frames/frame_NNNN.ppm
  |    append metadata sample
  |
  v
step_ctbr hover command
  |
  v
write episode.json
```
