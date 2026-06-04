# IVBS Improvement Plan

Baseline as of `docs/analysis/control_sims/ivbs_512_20260604_final2`:

- Scenario table: `scripts/generators/sim_instances/sobol_samples_512.csimin`
- Catch fraction: `0.037109375` (`19/512`)
- Median closest approach: `3.936667799949646 m`
- Mean visible fraction: `0.0660484649154619`
- Mean control effort: `8.833645771367644`

Cheating Beihang baseline on the same table:

- Catch fraction: `0.037109375` (`19/512`)
- Median closest approach: `3.1408482789993286 m`
- Mean visible fraction: `0.05715725693761922`
- Mean control effort: `9.271325817755395`

The primary failure mode is target loss. On the 512 run, IVBS caught `14/66`
scenarios with `visible_fraction >= 0.1`, but only `5/446` with
`visible_fraction < 0.1`. Improve visibility and uncertainty handling before
adding more aggressive metric interception.

The command path must continue to obey the IVBS observation contract in
`README.md`: no `snapshot.target.*`, no `snapshot.metrics.*`, no
`instance.target_initial*`, and no C-layer `target_pos_c`/`range_m`.

## Milestone 1: Observer And Command Telemetry

Goal: make IVBS failures diagnosable without using target truth in control.

Implementation:

- Add an IVBS telemetry record for each active slot/tick:
  - command mode: `metric`, `bearing_fallback`, `hover`
  - `estimate.valid`
  - `estimate.metric_confident`
  - `estimate.stale_s`
  - detection count
  - estimated range
  - range-direction std
  - position std
  - velocity std
  - bearing error between optical axis and selected LOS
- Keep telemetry outside the command decision surface. Target truth may be used
  only by analysis scripts after rollout completion.
- Add a docs/analysis script under `docs/analysis/control_sims/ivbs/` that
  summarizes telemetry by caught/missed, visibility bucket, and command mode.

Acceptance:

- Existing IVBS non-cheating tests pass.
- Static scan of `control_sims/ivbs` still finds no forbidden target/metrics
  reads outside README text.
- Analysis output identifies whether misses are dominated by stale estimates,
  fallback mode, metric mode, or target invisibility.
- On milestone completion, request a `gpt-5.5` `xhigh` subagent review.

## Milestone 2: FOV Retention And Bearing Fallback Tuning

Goal: improve target visibility before relying on range estimates.

Implementation:

- Parameterize fallback behavior:
  - LOS centering gain
  - `cautious_closing_accel_mps2`
  - `cautious_velocity_damping`
  - body-rate saturation behavior
  - minimum detections before metric mode
  - range std threshold for metric mode
- Add a sweep script under `docs/analysis/control_sims/ivbs/` that runs bounded
  candidate sets on the 512 table and writes artifacts in that folder.
- Prefer configurations that increase `visible_fraction_mean` and reduce
  `min_distance_p50_m` without reducing catch fraction.

Acceptance:

- Best candidate catches at least `19/512`.
- Best candidate improves either:
  - `visible_fraction_mean` by at least 10 percent relative, or
  - `min_distance_p50_m` below `3.5 m`.
- No new command-path target-truth reads.
- Run:

```bash
python -m pytest -q tests/test_sim_instance_store.py tests/test_puffer_backend_smoke.py tests/controllers/test_ivbs_policy.py
python scripts/runners/control_sim/ivbs.py --scenario-table scripts/generators/sim_instances/sobol_samples_512.csimin --samples 512 --workers 8 --max-envs 64 --progress-every 64 --out-dir docs/analysis/control_sims/ivbs_512_<label>
```

- On milestone completion, request a `gpt-5.5` `xhigh` subagent review.

## Milestone 3: Traditional-CV Apparent Size Range Cue

Goal: add a real visual range cue instead of relying on fixed range prior.

Implementation:

- Do not modify SimEngine, the C API, or typed SimEngine observation structs for
  this work.
- Implement deterministic image processing outside SimEngine using traditional
  CV techniques. No learned detector is required for this milestone.
- Candidate techniques:
  - background subtraction or frame differencing where a stable background is
    available
  - thresholding/segmentation against rendered target contrast
  - contour extraction and connected components
  - morphology to remove small artifacts
  - blob/ellipse fitting to estimate centroid and apparent target size
  - temporal filtering of centroid and size measurements
- The CV module should output an image measurement object containing:
  - detected/not detected
  - normalized centroid `uv_norm`
  - apparent size in pixels, preferably radius or equivalent area radius
  - confidence/quality score based on contour stability and size bounds
- This measurement must come from pixels or an existing image stream, not from
  SimEngine truth fields such as `range_m`, `target_pos_c`, or target state.
- Extend the IVBS observer measurement model with:

```text
range_estimate ~= target_radius_m * focal_px / apparent_radius_px
```

- Fuse apparent-size range as a noisy scalar measurement in the EKF.
- Add tests that mutating `snapshot.target`, `snapshot.metrics`, and
  `instance.target_initials` does not change commands, while mutating the
  traditional-CV apparent-size measurement can change commands.
- Document detector progress in `docs/detection/worklog.md`.

Acceptance:

- Range-direction covariance falls faster on visible trajectories.
- Catch fraction improves over the current `19/512` baseline or
  `min_distance_p50_m` improves below the cheating Beihang baseline of
  `3.1408482789993286 m`.
- No command-path use of simulator truth.
- On milestone completion, request a `gpt-5.5` `xhigh` subagent review.

## Milestone 4: Observer Model And Terminal Behavior

Goal: improve interception after the target is kept visible and range is less
prior-driven.

Implementation:

- Compare target process models:
  - stationary target prior
  - constant velocity
  - damped velocity
- Add terminal visual mode for close/uncertain estimates:
  - keep LOS centered
  - cap closing speed
  - reduce thrust when bearing error grows
  - avoid switching into full metric control with high range covariance
- Evaluate separate behavior for low-visibility, mid-visibility, and
  high-visibility buckets.

Acceptance:

- Catch fraction improves above `19/512`, or median closest approach improves
  while preserving catch fraction.
- Terminal mode does not increase out-of-bounds or nonfinite failures.
- Run the standard sim tests and the 512 benchmark.
- On milestone completion, request a `gpt-5.5` `xhigh` subagent review.

## Milestone 5: Scenario-Class Analysis And Regression Set

Goal: prevent overfitting one aggregate number.

Implementation:

- Build scenario slices from existing generated metadata:
  - camera azimuth/elevation
  - image-plane `u/v`
  - visibility bucket
  - closing-speed bucket
- Track IVBS-only catches and Beihang-only catches.
- Create a small fixed regression subset containing:
  - shared catches
  - IVBS-only catches
  - Beihang-only catches
  - high-visibility misses
  - low-visibility misses
- Add a fast smoke benchmark for this subset.

Acceptance:

- Each future IVBS change reports both full 512 results and regression-subset
  results.
- Regression subset lives under generated simulation sample files or analysis
  artifacts according to repo conventions.
- On milestone completion, request a `gpt-5.5` `xhigh` subagent review.
