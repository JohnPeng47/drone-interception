# Iteration 10 Notes

## Reviewer Observation

Hypatia recommended making the final iteration a margin-gated early-accept and diagnostics pass rather than adding a new candidate or doing another broad sweep.

The recommended generic early-accept thresholds were:

- `rollout_min_distance_m <= 0.45`
- `rollout_position_tracking_error_mean_m <= 0.5`
- keep the existing clean/caught/max-consecutive-capture requirement

Rationale:

- The portfolio already catches 6/6.
- The actual runner stops at first intercept, so runner min distance remains close to 0.5m and should not be interpreted as full plan margin.
- Direct replay margin and consecutive dwell are better robustness signals.
- A fourth candidate would increase runtime without clear general evidence.

## Implementation

Updated `control_sims/rpg_time_optimal/portfolio_policy.py`:

- `RpgTimeOptimalPortfolioMotorPolicy` now accepts:
  - `early_accept_max_min_distance_m=0.45`
  - `early_accept_max_tracking_error_mean_m=0.5`
- `solve_portfolio_plan()` threads those thresholds into `_early_accept()`.
- `_early_accept()` now requires:
  - clean score
  - replay caught radius
  - max consecutive capture steps >= 15
  - replay min distance <= 0.45m
  - replay mean tracking error <= 0.5m

Updated `run_portfolio_validation.py` diagnostics:

- selected rows now include `intercept_radius_m`
- `replay_margin_m = intercept_radius_m - rollout_min_distance_m`
- `rollout_max_consecutive_capture_duration_s`
- candidate rows also include replay margin and consecutive capture duration

Updated tests:

- early-accept tests now reject poor replay margin and poor tracking error.

## Replay Check

Ran:

```bash
python docs/analysis/control_sims/rpg_improve/replay_portfolio_pruning.py \
  --trace-csv docs/analysis/control_sims/rpg_improve/iter_8_warm_portfolio_validation/portfolio_candidates.csv \
  --output-dir docs/analysis/control_sims/rpg_improve/iter_10_margin_gate_replay \
  --min-consecutive-capture-steps 15 \
  --max-min-distance-m 0.45 \
  --max-tracking-error-m 0.5
```

Result:

- Catch fraction: 6/6.
- Simulated solved candidates: 9/18.
- Simulated skipped candidates: 9/18.
- Worst selected replay min distance: 0.398m.

## Validation

Production replay validation:

```bash
python docs/analysis/control_sims/rpg_improve/run_portfolio_validation.py \
  --workers 6 \
  --output-dir docs/analysis/control_sims/rpg_improve/iter_10_margin_gate_validation
```

Result:

- Catch fraction: 6/6.
- Worst selected replay min distance: 0.398m.
- Worst selected replay margin: 0.102m.
- Selected candidates unchanged:
  - Seed 1: `rate0p5_body0p2_win8`
  - Seed 2: `rate0p5_body0p2_win8`
  - Seed 3: `rate0p5_body0p2_win8`
  - Seed 4: `rate0p5_body0p2_win8`
  - Seed 5: `rate0p5_body0p2_win6`
  - Seed 6: `rate0p5_cmd0p2_body0p05_win8`

Selected replay margins:

- Seed 1: 0.102m.
- Seed 2: 0.287m.
- Seed 3: 0.317m.
- Seed 4: 0.403m.
- Seed 5: 0.218m.
- Seed 6: 0.397m.

Actual policy runner:

```bash
python scripts/runners/control_sim/rpg_time_optimal_motor_portfolio.py \
  --scenario-table scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin \
  --workers 6 \
  --max-envs 1 \
  --progress-every 1 \
  --out-dir docs/analysis/control_sims/rpg_improve/iter_10_margin_gate_policy_runner
```

Result:

- Catch fraction: 6/6.
- Errors: 0.
- Elapsed wall time: 141.2s.
- Median runner min distance: 0.494m.
- P90 runner min distance: 0.498m.

Runner distances remain close to the 0.5m threshold because the runner stops on first intercept. The direct replay margins above are the more useful measure of full-plan robustness.

## Tests

```bash
python -m pytest -q tests/controllers/test_rpg_time_optimal.py
```

Result: `6 passed in 48.37s`.

```bash
python -m pytest -q tests/test_sim_instance_store.py tests/test_puffer_backend_smoke.py
```

Result: `19 passed in 0.96s`.

## Final State

The final production path is a seed-agnostic, replay-scored motor portfolio with:

- fixed 3-candidate set;
- clean-plan gating;
- SimEngine replay scoring;
- x0-only warm starts;
- early accept based on consecutive replay dwell, replay margin, and tracking error;
- explicit skipped-candidate traces.

The final verified controller catches all 6 regression seeds through actual `SimRunner` execution.
