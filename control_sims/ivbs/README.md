# IVBS Beihang-Style Controller

This package reimplements the Beihang paper controller as an image-based visual
servoing controller whose command path does not read target truth from
`SimEngine`.

## SimEngine Observation Contract

The `SimSnapshot` object contains both observation-like state and simulator
truth. The IVBS controller may only use fields that would be available to an
onboard controller or to the detector output.

Allowed command-path inputs:

- `snapshot.pursuer.position_w`
- `snapshot.pursuer.velocity_w`
- `snapshot.pursuer.quat_xyzw`
- `snapshot.pursuer.body_rates_b`
- `snapshot.camera.detected`
- `snapshot.camera.uv_norm`
- `instance.config.cameras[0]`
- pursuer configuration: mass, gravity, thrust/rate limits
- camera configuration: extrinsics, intrinsics, capture rate
- previous IVBS observer state and previous IVBS commands

Forbidden command-path inputs:

- `snapshot.target.position_w`
- `snapshot.target.velocity_w`
- `snapshot.metrics.distance_m`
- `snapshot.metrics.min_distance_m`
- `snapshot.metrics.intercepted`
- `snapshot.metrics.intercept_time_s`
- `instance.target_initial.position_w`
- `instance.target_initial.velocity_w`
- C-layer camera truth such as `target_pos_c`, `range_m`, or apparent target
  radius unless a future detector computes it from image pixels and exposes it
  as a measurement.

Target truth and metrics may still be used by runners, logging, tests, and
offline analysis. They must not enter `IVBSControlPolicy.command`.

## Milestone 1: Non-Cheating Controller Boundary

The first milestone establishes the permanent command boundary:

```text
allowed SimSnapshot fields
        -> VisualRelativeStateObserver
        -> RelativeStateEstimate(p_r_w, v_r_w, covariance, valid, stale_s)
        -> Beihang control law
        -> CtbrCommandBatch
```

The controller extracts the Beihang command math into a function that consumes
estimated relative position and relative velocity. The observer bootstraps from
the image bearing plus a configured range prior, initializes target velocity
from a configured target-motion prior, and carries high covariance on range and
velocity. The policy does not enter the full metric Beihang law until the
observer has processed enough image measurements for the estimate to be treated
as a visual estimate rather than only an initialization prior.

Acceptance checks:

- commands are identical when only `snapshot.target.position_w` changes
- commands are identical when only `snapshot.target.velocity_w` changes
- the policy can run generated control scenarios without command-path target
  access

## Milestone 2: Deterministic Visual Observer

The second milestone replaces the prior-only observer with a deterministic
monocular EKF. The observer state is:

```text
x = [p_target_w, v_target_w]
```

Prediction uses a constant-velocity target model:

```text
p_target_w[k+1] = p_target_w[k] + v_target_w[k] * dt
v_target_w[k+1] = v_target_w[k] + process_noise
```

Measurement uses only the normalized image point:

```text
uv_norm = project(camera, p_target_w - p_camera_w)
```

The EKF prediction uses pursuer pose history and camera extrinsics to account
for ego-motion. Image residuals correct target position and, through
cross-covariance, target velocity. If the estimate covariance is too high, the
controller runs a cautious bearing-centered command instead of pretending the
full metric law is reliable.

Validation compares estimate error against simulator truth only outside the
command path.
