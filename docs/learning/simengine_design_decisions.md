# SimEngine Design Decisions

These are the most consequential SimEngine design decisions surfaced during the
stepping and rendering review.

## 1. C Stepping And Snapshot Collection Are Separate

`sim_engine_step_motor_speeds_dt()` advances the C `SimEngine` in place. It does
not return state. State is exported separately through `sim_engine_get_snapshot()`.

The current Python scalar and batch bindings choose to wrap those operations so
one Python step returns a fresh snapshot:

```text
Python step -> C step -> C snapshot -> Python snapshot object
```

Consequence: the C API supports sparse inspection or future Python APIs that
advance multiple steps before sampling. Current public Python callers still get
one snapshot per Python step.

Rendering consequence: camera capture/rendering is triggered during snapshot or
camera-output collection, not during C stepping. Skipping snapshots would skip
capture checks unless a caller explicitly collects camera outputs.

## 2. SimEngine Has Three Runtime Frequencies

Runtime behavior is governed by three separate cadences:

```text
public step frequency       = 1 / (backend_dt * action_substeps)
pursuer integration freq    = 1 / backend_dt
camera capture/render freq  = camera.capture_rate_hz
```

The public step frequency is the command/environment cadence. The pursuer
integration frequency is the RK4 substep cadence inside the public step. The
camera frequency controls when camera outputs and rendered frames are due.

Consequence: changing `action_substeps` changes the public command interval
relative to pursuer integration. It does not create additional target, metrics,
or rendering substeps.

## 3. Rendering Produces Per-Capture Frames, Not Trajectories

`RenderConfig` configures the render backend, selected camera, scene, timeout,
and error behavior. It does not make `SimEngine` output a rendered trajectory,
video, or frame sequence.

When rendering is enabled, each due camera output can include one rendered frame.
Trajectory rendering is a caller responsibility: a caller must repeatedly step
the sim, collect snapshots, extract frame bytes, and write or assemble the
sequence.

Consequence: `SimEngine` remains a state/camera-output engine. Episode-level
artifacts such as rendered trajectories should live in runner or rendering
episode utilities, not in the core C step function.
