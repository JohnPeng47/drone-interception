# Iteration 1 Notes

## Observations

- The corrected baseline (`0.5 m` terminal tolerance, 30 nodes) produced 1/6 motor-feedforward catches.
- Tightening the terminal tolerance to `0.1 m` and doubling to 60 nodes improved motor-feedforward to 2/6, but one seed became a planner failure.
- The current diagnostics did not directly answer whether the planned RPM sequence replays through SimEngine, so policy misses could still be planner extraction, command hold timing, or horizon behavior.
- The planner summary reported body-rate norm, but constraints are per-component. This made some values look like violations when they were not necessarily invalid.

## Implemented This Iteration

- Added IPOPT status, success, max constraint violation, requested terminal tolerance, terminal tolerance satisfaction, per-component rate max, and explicit limit violations to `planner_metrics.csv`.
- Added direct planned-RPM SimEngine replay diagnostics:
  - `plan_rollout_metrics.csv`
  - `plan_rollout_trajectories.csv`
  - `summary.json` `plan_rollout` section
- Added `--ipopt-max-iter` and `--seeds` CLI options to enable focused solver sweeps.

## Suggested Next Steps

- Re-run seed 6 with `cpc_tolerance_m=0.1`, `terminal_nodes=60`, and higher IPOPT iteration limits to determine whether the previous planner miss was a solve-budget issue.
- Compare direct plan rollout catch fraction against motor-feedforward catch fraction. If direct rollout misses the same seeds, focus on plan discretization/timing/model issues. If direct rollout catches but policy misses, focus on policy command sampling or horizon behavior.
- If higher IPOPT iterations recover seed 6, run a compact sweep over `cpc_tolerance_m={0.1,0.25}`, `terminal_nodes={45,60}`, and `ipopt_max_iter={300}`.

## Results

- Seed 6 with `cpc_tolerance_m=0.1`, `terminal_nodes=60`, `ipopt_max_iter=300` solved successfully to approximately `0.1 m`, so the previous seed-6 planner failure was mostly solve budget.
- Direct SimEngine replay of that seed-6 RPM plan still missed by `2.55 m`, while RPM tracking at closest approach was near zero. This points to dynamics/discretization fidelity rather than command extraction.
- Adding `dynamics_substeps=3` reduced seed-6 direct replay miss from `2.55 m` to `1.07 m` and motor-feedforward min distance from `2.55 m` to `0.98 m`, but the run took about `598 s` end to end and had max constraint violation around `0.023`.
- Dynamics substepping is directionally useful but too expensive for broad sweeps unless paired with better initialization, fewer nodes, or a narrower hard-seed workflow.
