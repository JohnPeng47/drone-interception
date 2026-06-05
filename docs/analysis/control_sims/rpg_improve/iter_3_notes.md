# Iteration 3 Notes

## Observation From Review Agent

- Cached motor execution now matches direct planned-RPM rollout, so the remaining misses are not policy plumbing.
- Linear interpolation, post-horizon hold/hover tails, and a single global time scale are not robust fixes.
- The likely failure is planner/execution grid fidelity and aggressive limit-saturated plans.

## Implemented This Iteration

- `RpgTimeOptimalPlan` now carries planned actual motor RPM state.
- Direct rollout diagnostics now report actual-RPM tracking and body-rate tracking against planned states.
- Added a generic planner body-rate margin via `planner_rate_limit_scale`.

## Results

- Seed 1 baseline tracking showed actual RPM tracking was close near closest approach, but body-rate tracking was off by about `3 rad/s`; this implicated angular-state replay fidelity rather than command extraction.
- `planner_rate_limit_scale=0.75` slightly improved seed 1 but did not solve it.
- `planner_rate_limit_scale=0.5` made seed 1 catch cleanly with low tracking error.
- Hard seeds `1,4,5,6` with `rate=0.5` caught seeds 1 and 6, but seed 4 worsened badly and seed 5 still missed.
- Full six with `cpc=0.1`, `nodes=60`, `ipopt=300`, `rate=0.5`, motor-only cached execution reached 3/6:
  - caught: seeds 1, 3, 6;
  - missed: seed 2 at `1.012 m`, seed 4 at `4.587 m`, seed 5 at `1.929 m`.

## Suggestions

- A single global rate margin is not sufficient; it trades off seed 2 and seed 4.
- Next diagnostics should compare planned body-rate and thrust profiles for seeds that improve under `rate=0.5` versus seeds that degrade.
- Next implementation should target a smoother/more faithful angular trajectory rather than simply lowering all rate limits:
  - penalize rate/thrust changes,
  - add multi-tick terminal capture window,
  - or align the OCP to SimEngine action ticks.
