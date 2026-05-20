# Intercept Sim Worklog

## 2026-05-08: Beihang Controller and EKF Progress

### Progress So Far

- Built the first RotorPy-backed interception simulator package around:
  - kinematic red-balloon target scenarios
  - camera/FOV projection
  - delayed image-feature perception
  - observer/controller abstractions
  - CTBR command output into RotorPy
  - benchmark metrics including miss distance and CEP
- Implemented the red-balloon test scenario:
  - balloon drifts on a seeded straight-line trajectory
  - copter starts with the balloon centered in FOV
  - copter initial velocity uses relative closing speed toward the balloon
- Ported the Beihang/Rfly `BacksteppingController` into
  `BeihangBacksteppingController`.
  - The controller uses image LOS plus relative target state:
    `p_r`, `v_r`, target acceleration, and vehicle attitude.
  - Current implementation emits RotorPy CTBR commands.
- Added a truth-relative observer for controller bring-up.
- Added a first-pass Beihang-style image/IMU EKF observer:
  - 18D state layout: quaternion, position, velocity, image feature,
    gyro bias, accel bias
  - image measurement update
  - delayed replay over stored prediction snapshots
  - config support via `observer.type: beihang_image_ekf`
- Added no-known-target-state future work items to `HANDOFF.md`:
  - estimate range from apparent balloon size and keep the controller
  - test image-only controllers
  - investigate bearing-only control laws

Current full test status after the Beihang EKF work:

```text
pytest -q
48 passed
```

### Paper-Derived Static Red-Balloon Sweep

Paper-derived setup used for this sweep:

- Static red-balloon real-flight paper coordinates imply an initial distance of
  about `10.4 m`:
  - interceptor start approximately `(-1 m, -1.8 m, 3 m)`
  - target approximately `(-3 m, -12 m, 4 m)`
- Sweep parameters:
  - `distance_m = 10.4`
  - `duration_s = 5.0`
  - speeds: `5, 10, 15, 19, 21 m/s`
  - scenario config: `configs/experiments/red_balloon_beihang_ekf.yaml`
  - controller: `beihang_backstepping`
  - observer: `beihang_image_ekf`
  - perception delay: `80 ms`
  - camera rate: `30 Hz`
  - catch radius: `0.5 m`

Important sweep caveat:

- A 50-seed run was started but stopped because it took too long and printed no
  progress.
- The rerun used 5 seeds per speed with progress printing.
- Results were identical across seeds because the current scenario fixes
  initial LOS and uses relative-closing initialization. Random balloon drift
  mostly cancels from the initial relative motion.
- Treat the table below as a deterministic sweep, not a meaningful CEP
  distribution yet.

Beihang image EKF observer results:

| distance m | speed m/s | runs | CEP50 m | CEP90 m | min miss m | mean miss m | catch frac | visible | feature | avg image err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10.4 | 5 | 5 | 0.631 | 0.631 | 0.631 | 0.631 | 0.00 | 0.55 | 0.98 | 1.340 |
| 10.4 | 10 | 5 | 1.885 | 1.885 | 1.885 | 1.885 | 0.00 | 0.33 | 0.98 | 0.794 |
| 10.4 | 15 | 5 | 2.282 | 2.282 | 2.282 | 2.282 | 0.00 | 0.31 | 0.98 | 0.599 |
| 10.4 | 19 | 5 | 2.135 | 2.135 | 2.135 | 2.135 | 0.00 | 0.31 | 0.98 | 0.596 |
| 10.4 | 21 | 5 | 2.044 | 2.044 | 2.044 | 2.044 | 0.00 | 0.32 | 0.98 | 0.596 |

Same conditions, comparing truth image/relative state against Beihang image EKF:

| observer | speed m/s | miss m | catch | visible | feature | avg image err |
|---|---:|---:|---:|---:|---:|---:|
| truth | 5 | 0.055 | 1 | 0.50 | 0.50 | 0.117 |
| truth | 10 | 1.016 | 0 | 0.35 | 0.35 | 0.161 |
| truth | 15 | 1.657 | 0 | 0.27 | 0.27 | 0.177 |
| truth | 19 | 1.635 | 0 | 0.13 | 0.11 | 0.183 |
| truth | 21 | 1.564 | 0 | 0.08 | 0.08 | 0.094 |
| ekf | 5 | 0.631 | 0 | 0.55 | 0.98 | 1.340 |
| ekf | 10 | 1.885 | 0 | 0.33 | 0.98 | 0.794 |
| ekf | 15 | 2.282 | 0 | 0.31 | 0.98 | 0.599 |
| ekf | 19 | 2.135 | 0 | 0.31 | 0.98 | 0.596 |
| ekf | 21 | 2.044 | 0 | 0.32 | 0.98 | 0.596 |

### Interpretation

* Reminder: we should test with data from flight paths known
- With the paper-derived `10.4 m` start, current simulator performance does not
  match the paper's high-speed static-target results.
- The truth-image path catches at `5 m/s` but misses at `10 m/s` and above.
- The Beihang image EKF path currently performs worse than the truth-image path,
  which is expected against a perfect-image upper bound but still indicates that
  the EKF port needs tuning/validation.
- Since even the truth-image path fails at paper-like speeds, the largest current
  gap is likely vehicle/controller tuning and high-speed authority mismatch
  between RotorPy and the paper's platform.
- The EKF implementation is useful for delay/noise comparisons, but it is not yet
  enough to reproduce paper performance.

### Next Useful Steps

- Add a proper sweep script with progress output and CSV output before rerunning
  50-seed paper-style sweeps.
- Randomize initial LOS if CEP across seeds is desired; current seeds are nearly
  deterministic under fixed LOS and relative-closing initialization.
- Tune RotorPy vehicle/controller limits against the paper platform:
  - thrust-to-weight ratio about `3`
  - reported max acceleration about `5 m/s^2`
  - controller/inner loop about `200 Hz`
  - camera about `30 Hz`
- Compare Beihang image EKF against `latest`, `constant_velocity`, and
  `delayed_replay` under the same delay/noise/dropout conditions.
- Validate the EKF image propagation against synthetic image-motion cases before
  relying on closed-loop CEP.
