# Iteration 4 Notes

## Observation From Review Agent

- The remaining failures are not policy plumbing. Cached motor execution matches direct RPM rollout.
- Time scale, linear interpolation, post-plan tails, dynamics substeps, and global rate scaling all showed seed-dependent tradeoffs.
- The full `rate=0.5` run solved all six plans to `0.1 m`, but caught only seeds `1,3,6`; seeds `2,4,5` still missed.
- Reviewer recommendation: add real planner smoothness/robustness instead of continuing scalar sweeps.

## Local Diagnostics

- Added smoothness metrics to `planner_metrics.csv`.
- Comparing `rate=0.5` seeds 1 and 4:
  - seed 1 catches and has max motor command jump about `10205 rpm`, max total-thrust jump about `17.7 N`;
  - seed 4 misses badly and has max motor command jump about `13252 rpm`, max total-thrust jump about `31.0 N`;
  - seed 4 direct rollout body-rate tracking error is much larger (`4.48 rad/s` mean) than seed 1 (`0.48 rad/s` mean).

## Suggested Step

- Add generic planner robustness knobs:
  - command/body-rate smoothness penalties in the OCP objective;
  - terminal capture window constraints over the last few nodes.
- Evaluate first on seed 4 because it is the clearest degradation case after rate limiting.

## Results

- Implemented OCP smoothness and terminal capture-window knobs:
  - `command_smoothness_weight`
  - `body_rate_smoothness_weight`
  - `terminal_capture_window_nodes`
- Seed 4 with `rate=0.5`, smoothness weights `0.05`, window `5` improved from `4.59 m` miss to `1.30 m`.
- Seed 4 with smoothness weights `0.2`, window `8` caught cleanly at `0.486 m`.
- Full six with `rate=0.5`, smoothness weights `0.2`, window `8` remained 3/6 but changed which seeds caught:
  - caught: seeds `2,3,4`;
  - missed: seed `1` at `0.530 m`, seed `5` at `0.664 m`, seed `6` at `2.526 m`.
- This is still useful: seeds 1 and 5 are now close, and seed 4 is fixed, but seed 6 regressed badly versus the previous rate-margin run.
