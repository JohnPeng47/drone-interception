# SimEngine Frequencies

This document describes the clocks that affect `SimEngine` runtime behavior.
There are three runtime frequencies to keep distinct:

1. The public control step frequency.
2. The internal pursuer integration frequency.
3. The camera capture/rendering frequency.

No other frequency currently drives C `SimEngine` state advancement or rendering.

## Public Control Step

Python callers advance `SimEngine` by calling the binding layer, for example
`PufferSimEngineBackend.step_ctbr()`, `BatchPufferSimEngineBackend.step_ctbr_many()`,
or `SimRunner.step()`.

The public step duration is:

```text
step_dt = SimOptions.backend_dt * max(1, SimOptions.action_substeps)
step_hz = 1 / step_dt
```

One public step consumes one command and advances the C engine once via
`sim_engine_step_motor_speeds_dt(..., dt=step_dt, substeps=action_substeps)`.

`SimRunner` and the RL environments treat this public step as their environment
step. Episode elapsed time increments by `step_dt`; `SimRunner` evaluates
`duration_s` termination at this cadence, and RL environments may additionally
count their own `max_episode_steps` horizon at this cadence.

## Pursuer Integration

Inside `pursuer_sim_step_motor_dt()` and `pursuer_sim_step_motor_speeds_dt()`,
the pursuer dynamics split the public step into integration substeps:

```text
pursuer_sub_dt = step_dt / max(1, action_substeps)
               = backend_dt
pursuer_hz     = 1 / backend_dt
```

Each pursuer substep runs RK4 and applies velocity, body-rate, and RPM clamps.
The command is held constant across all substeps in that public step.

Important boundary: this substepping is only for the pursuer dynamics. Targets
are stepped once per public SimEngine step with the full `step_dt`, and
intercept metrics are updated once after that full step.

## Camera Capture And Rendering

Each camera has its own `CameraConfig.capture_rate_hz`. The C camera schedule is:

```text
capture_period = 1 / capture_rate_hz
capture due when engine.t >= camera.next_capture_t
after capture: camera.next_capture_t += capture_period
```

Cameras reset with `next_capture_t = 0`, so the first collected snapshot can
emit a capture at `t = 0` when `capture_rate_hz > 0`.

Rendering is tied to camera capture, not directly to the physics step. C
`sim_engine_step_*()` only advances state and time. Camera observations and
render frames are produced when camera outputs are collected:

- Scalar snapshots call `sim_engine_get_snapshot()`, which calls
  `sim_engine_collect_camera_outputs()`.
- `sim_engine_collect_camera_outputs()` checks each camera schedule.
- If a capture is due and rendering is enabled for that camera,
  `sim_engine_render_camera_output()` calls `liftoff_render_frame()`.

In normal scalar Python use, `PufferSimEngineBackend.reset()` and
`PufferSimEngineBackend.step_ctbr()` both return snapshots, so camera capture
and rendering are checked once on reset and once after each public step.
If `capture_rate_hz` is lower than `step_hz`, most returned snapshots have no
camera output. If `capture_rate_hz` is higher than `step_hz`, the current code
emits at most one capture per camera per output collection and advances
`next_capture_t` by one capture period.

Batched snapshots currently expose compact first-camera observations through
`sim_engine_batch_get_snapshots()`. They do not carry rendered frame bytes.

## Example

With:

```text
backend_dt = 0.002
action_substeps = 5
capture_rate_hz = 30
```

the frequencies are:

```text
public step_dt       = 0.010 s  => 100 Hz commands/environment steps
pursuer_sub_dt       = 0.002 s  => 500 Hz pursuer RK4 integration
camera capture/render           => 30 Hz when snapshots collect outputs
```

With the base csim config:

```text
backend_dt = 0.005
action_substeps = 1
capture_rate_hz = 30
```

the frequencies are:

```text
public step_dt       = 0.005 s  => 200 Hz commands/environment steps
pursuer_sub_dt       = 0.005 s  => 200 Hz pursuer RK4 integration
camera capture/render           => 30 Hz when snapshots collect outputs
```

## Related Timing That Is Not A SimEngine Runtime Clock

These settings exist in or near SimEngine workflows, but they are not additional
C `SimEngine` frequencies:

- `SimOptions.validation_dt`: only used by generator validation checks such as
  kinematic intercept validation. It does not affect runtime stepping.
- `SimOptions.duration_s`: an episode horizon used by runners/environments. It
  does not change integration cadence.
- `max_episode_steps`: an RL environment horizon counted in public steps.
- `RenderConfig.timeout_ms`: a renderer call timeout/configuration value, not a
  frame rate.
- `NoiseConfig.camera_image_delay_s`: a perception delay used by control-sim
  sensing models, not a SimEngine render cadence.
- `PursuerParams.motor_tau_s`: a motor response time constant inside the
  dynamics, not a scheduler frequency.
- Control-sim Drake `period_sec` values, logger periods, and training
  log/checkpoint intervals: adapter or tooling cadences outside C `SimEngine`.

## Source Pointers

- `backends/csim/bindings/types/sim_engine.py`: `SimOptions`, `SimConfig`,
  `RenderConfig`.
- `backends/csim/bindings/puffer_c.py`: Python binding step durations and batch
  homogeneity checks.
- `backends/csim/runner/runner.py`: `SimRunner.step()` elapsed-time accounting.
- `backends/csim/sim_engine.c`: target stepping, metrics update, camera output
  collection, and render calls.
- `backends/csim/pursuer_sim.c`: RK4 substeps inside pursuer dynamics.
- `backends/csim/camera_sim.c`: capture schedule and `next_capture_t`.
