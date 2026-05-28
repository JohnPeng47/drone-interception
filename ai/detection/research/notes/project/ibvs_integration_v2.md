# IBVS + PNG Integration Plan v2

Concrete implementation plan for integrating the Yan/Yang analytical control
logic into the existing simulator stack, then exporting trajectory-conditioned
samples for the synthetic rendering pipeline.

This document replaces the earlier plan in
[`ibvs_integration.md`](./ibvs_integration.md) where that plan simplified the
controller too aggressively and blurred the boundary between the runnable
simulator and the downstream renderer.

## Goal

Produce a Phase 1 trajectory generator that:

- runs the Yan/Yang-style analytical controller in the existing
  `drones/interception` simulator,
- emits trajectory samples in the schema expected by
  [`synthetic/pipeline.md`](./synthetic/pipeline.md),
- provides a clean handoff to a rendered scene component downstream.

## Key corrections from v1

1. The runnable simulator today is the sibling `drones/interception` module,
   not `swarm/`.
2. The controller should preserve the actual paper structure:
   PNG guidance + FOV-holding + lift/attitude conversion, not only a reactive
   "point camera at target" kernel.
3. Rendering is required for the overall Phase 1 pipeline, but it should remain
   downstream of the control sim. The controller integration does not need a
   renderer in-loop.

## Scope

In scope:

- New analytical controller in `drones/interception`
- Camera model sufficient for ideal image-plane measurements
- Scripted evader trajectories with deterministic preview support
- Canonical driver and smoke tests
- Trajectory export for the synthetic pipeline
- Renderer-facing sample schema and one smoke-tested handoff path

Out of scope for this phase:

- Delayed Kalman filter
- Delayed image measurements
- IMU bias / realistic sensor corruption
- Full Blender scene generation implementation
- Learned controller integration into `swarm/`

## Existing code boundary

The implementation target is the simulator in:

- [`../interception/src/interception/dynamics.py`](../interception/src/interception/dynamics.py)
- [`../interception/src/interception/evader.py`](../interception/src/interception/evader.py)
- [`../interception/src/interception/mpc_ctbr.py`](../interception/src/interception/mpc_ctbr.py)
- [`../interception/src/interception/simulator.py`](../interception/src/interception/simulator.py)
- [`../interception/scripts/run_canonical.py`](../interception/scripts/run_canonical.py)

The downstream consumer is:

- [`synthetic/pipeline.md`](./synthetic/pipeline.md)

Important implication:

- `swarm/` currently contains docs, papers, and planning material.
- `interception/` contains the executable sim interfaces we can integrate with
  immediately.
- The v2 plan therefore uses `interception/` as the control-validation harness
  and `swarm/docs/synthetic/` as the output contract.

## Control architecture to implement

Implement the controller in three layers that mirror the papers:

1. Image measurement layer
   - Use ground-truth target world position and pursuer pose from the simulator.
   - Transform target position into the camera frame.
   - Compute normalized image coordinates and image error `e = [ex, ey]`.

2. Guidance layer
   - Recover LOS angles from the target bearing.
   - Compute desired velocity angles via PNG-style updates as described in
     Yan 2024 / Yang 2024.
   - Maintain a desired speed / acceleration law through the paper's `ka`,
     `Ky`, and `Kz` style gains.

3. FOV-hold + command layer
   - Apply FOV-holding control on image-plane error, especially `ex`.
   - Convert desired velocity / acceleration into desired lift direction.
   - Convert lift direction into thrust + desired body rates.
   - Reuse the existing CTBR inner-loop allocation path for rotor thrusts.

This is the minimum structure that still deserves to be called "Yan/Yang in the
sim." A pure collinearity controller is acceptable as a fallback mode for early
debugging, but not as the main implementation target.

## Measurement model for Phase 1

Use idealized measurements:

- target world pose from the sim,
- pursuer world pose from the sim,
- deterministic camera projection,
- no DKF, no latency, no pixel noise.

Reason:

- Phase 1 needs trajectory-conditioned pose distributions for synthetic data.
- The downstream renderer and detector pipeline own realism at this stage.
- Estimator realism is important later, but it is not load-bearing for the
  trajectory generator milestone.

## Files to add or extend

All code changes below are in `drones/interception` unless explicitly noted.

### New files

- `src/interception/camera.py`
  - `CameraParams` dataclass
  - projection helpers
  - FOV checks
  - YAML loader matching the style of `params.py`

- `src/interception/ibvs_png.py`
  - `IBVSPNGController`
  - controller-compatible surface:
    - `params`
    - `horizon_steps`
    - `solve(state, target_positions)`
    - `command_to_rotor_thrusts(state, command, dt)`
    - `reset_execution_state()`
  - debug fields expected by `simulator.py`:
    - `last_predicted_states`
    - `last_predicted_controls`
    - `last_realization`
    - `last_stats`

- `scripts/run_ibvs_png_canonical.py`
  - driver for one or two fixed scenarios
  - saves plots / metrics alongside existing canonical outputs

- `tests/test_ibvs_png.py`
  - controller unit tests
  - simulator-level smoke tests

- `configs/camera.yaml`
  - deployment-like monocular camera intrinsics and orientation

- `scripts/export_ibvs_trajectories.py`
  - runs batches of engagements
  - writes renderer-facing trajectory records

### Existing files to extend

- `src/interception/evader.py`
  - add `ScriptedTrajectoryEvader`
  - preserve `EvaderPolicy` compatibility

No edits should be required to:

- `dynamics.py`
- `simulator.py`
- `mpc_ctbr.py`
- `mpc_srt.py`

If the controller cannot be integrated without touching these files, treat that
as a design regression and stop to reassess.

## Controller interface details

The new controller must conform to the expectations already encoded in
[`../interception/src/interception/simulator.py`](../interception/src/interception/simulator.py):

- `solve()` receives:
  - current 22D state
  - target trajectory shaped `(horizon_steps + 1, 3)`
- `command_to_rotor_thrusts()` returns rotor thrusts for the SRT dynamics path
- `last_predicted_states` and `last_predicted_controls` must be populated even
  if the controller is reactive

Implementation rule:

- set `horizon_steps = 1`
- return a 2-step predicted state rollout placeholder
- return a 1-step predicted control rollout placeholder

That keeps `simulator.py` debug logging happy without pretending this is MPC.

## Concrete control law mapping

The implementation should map paper concepts to sim quantities as follows:

- camera-frame target vector:
  - from pursuer world pose and body rotation
  - then through `body_to_camera`

- image-plane error:
  - normalized projection error relative to image center

- LOS angles:
  - computed from the target bearing vector

- desired velocity-angle update:
  - maintain prior-step `sigma_y`, `sigma_z`
  - update from LOS-angle deltas using PNG gains

- desired velocity vector:
  - derive from desired speed magnitude and desired velocity angles

- FOV holding:
  - yaw-rate PD from `ex`
  - pitch / roll body-rate contribution from desired lift direction

- thrust command:
  - from desired acceleration projected against gravity
  - clipped to total feasible thrust

- rotor allocation:
  - reuse the same inner-loop conversion style as CTBR

The exact equations should follow the paper closely where possible, but the
first version may omit terms that depend on unavailable onboard estimator state
as long as the guidance / FOV / lift decomposition is preserved.

## Camera assumptions

Default assumptions:

- monocular pinhole camera
- fixed mount
- optical axis aligned horizontally
- configuration carried by a `body_to_camera` rotation

Recommended default:

- camera optical axis aligned with body `+x`

Reason:

- it matches the "camera forward, vehicle flies forward" interpretation better
  than the v1 note's provisional `+z` camera axis assumption
- it makes the image model easier to reason about for a rendering handoff

## Evader design

Add `ScriptedTrajectoryEvader` to
[`../interception/src/interception/evader.py`](../interception/src/interception/evader.py).

Requirements:

- piecewise-linear or spline-backed motion
- deterministic `step()`
- `predict_positions()` consistent with the same internal motion model
- support for:
  - straight crossing
  - constant-acceleration escape
  - sinusoidal / weaving target

Reason:

- the paper scenarios cover CV, CA, and sinusoidal maneuver families
- the renderer pipeline needs richer pose distributions than constant-velocity
  crossing only

## Export schema

The export script is the bridge from control to rendering.

Each record should include at least:

- `engagement_id`
- `frame_idx`
- `t`
- `pursuer_position_w`
- `pursuer_rotation_wb`
- `target_position_w`
- `target_velocity_w`
- `range_m`
- `closing_rate_mps`
- `normalized_pixel`
- `target_size_px`
- `target_type`
- `lighting_id`
- `outcome`

Notes:

- `lighting_id` can be a placeholder at export time if the renderer owns the
  actual lighting choice later.
- `target_size_px` should be estimated from range and target physical size so
  Stage 1 coverage filtering can bin by apparent scale.
- `target_type` should be included now even if fixed to one model initially.

## Renderer handoff

Yes, a rendered component is required overall.

But the boundary should be explicit:

1. `export_ibvs_trajectories.py` emits trajectory-conditioned camera/target
   poses and metadata.
2. A separate renderer consumes those records to build scenes and images.

This v2 plan requires one concrete smoke path:

- one exported trajectory sample can be consumed by a renderer-facing adapter
  without inventing extra control-side state.

That means the control export must already include the camera pose ingredients
the renderer needs.

## Implementation sequence

### Phase A: controller foundation

1. Add `camera.py`
2. Add `IBVSPNGController`
3. Populate reactive placeholder debug rollouts
4. Reuse CTBR allocation path

Deliverable:

- controller compiles and runs in one stationary-target scenario

### Phase B: scenario coverage

1. Add `ScriptedTrajectoryEvader`
2. Add canonical driver
3. Add simulator smoke tests

Deliverable:

- successful runs for stationary, crossing, and turning-target cases

### Phase C: trajectory export

1. Add export script
2. Define stable record schema
3. Save batched engagement outputs

Deliverable:

- trajectory dataset usable by Stage 1 of the synthetic pipeline

### Phase D: renderer boundary check

1. Validate exported camera/target pose fields against
   [`synthetic/pipeline.md`](./synthetic/pipeline.md)
2. Add one smoke-tested adapter or sample contract for renderer consumption

Deliverable:

- confirmed control-to-renderer handoff with no missing pose metadata

## Tests

Add tests in `tests/test_ibvs_png.py` covering:

- stationary target in FOV
- crossing target catch or stable close-in pursuit
- target initially behind pursuer
- image-error sign producing the correct yaw command direction
- `predict_positions()` consistency for `ScriptedTrajectoryEvader`
- export record shape / required fields

Prefer two test layers:

- controller-level unit tests
- one or two end-to-end sim smoke tests

## Acceptance criteria

The implementation is done when:

1. `pytest tests/test_ibvs_png.py` passes in `drones/interception`.
2. `python scripts/run_ibvs_png_canonical.py` produces at least one successful
   intercept plot for a moving target case.
3. `python scripts/export_ibvs_trajectories.py --n 100` writes a batch file with
   renderer-facing records matching the Stage 1 contract.
4. Exported records contain enough information to reconstruct camera pose and
   target pose in a renderer without hidden simulator state.
5. No core simulator interfaces were rewritten to special-case IBVS.

## Open decisions

These should be resolved before or during implementation:

1. Camera mount convention
   - default recommendation: optical axis along body `+x`

2. Target physical size model
   - default recommendation: store a target bounding radius per record

3. Catch radius
   - default recommendation: use a less brittle value than the current
     `0.15 m` when validating paper-like scenarios

4. Export format
   - `npz` is fine for early batches
   - a tabular format may be better once the renderer starts consuming records

## Non-goals for this doc

This doc does not specify:

- the Blender implementation details
- the RL policy architecture
- the eventual `swarm/` production simulator integration path

Those are separate workstreams. This plan is strictly the shortest defensible
path from Yan/Yang control logic to renderer-consumable trajectory data.
