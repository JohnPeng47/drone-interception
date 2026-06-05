# Iteration 7 Notes

## Reviewer Observation

Goodall identified three gaps after iteration 6:

- The production validation called `solve_portfolio_plan()` directly, but did not exercise `RpgTimeOptimalPortfolioMotorPolicy` through `SimRunner`.
- Solve cost is now the main bottleneck. The production portfolio solves three 60-node NLPs per scenario.
- Sweep reporting did not match production selection semantics: the sweep best-by-seed report chose the lowest min distance, while production prioritizes catch, capture dwell, min distance, tracking error, and plan time.

Recommended next steps were to add a true `SimRunner` validation artifact, add timing breakdowns before optimizing cost, avoid implicit nested parallelism, and align sweep reports with production scoring.

## Implementation

Added/changed:

- `run_diagnostics.py`
  - Added `rollout_capture_steps` to direct planned-RPM rollout metrics.
  - Added planner timing fields to `planner_metrics.csv`: `plan_nlp_build_wall_s` and `plan_optimizer_wall_s`.

- `run_candidate_sweep.py`
  - Changed `best_by_seed.csv` selection to match production semantics:
    catch first, then capture dwell, then min distance, tracking error, and plan time.

- `RpgTimeOptimalPlan`
  - Added `nlp_build_wall_s` and `optimizer_wall_s` timing metadata.

- `RpgPlanReplayScore`
  - Added `replay_wall_s`.

- `run_portfolio_validation.py`
  - Added `replay_wall_s` to validation CSV output.

No objective, constraint, or command-generation behavior was changed in this iteration.

## Runner Validation

Ran the actual portfolio policy through `SimRunner`:

```bash
python scripts/runners/control_sim/rpg_time_optimal_motor_portfolio.py \
  --scenario-table scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin \
  --workers 6 \
  --max-envs 1 \
  --progress-every 1 \
  --out-dir docs/analysis/control_sims/rpg_improve/iter_7_portfolio_policy_runner
```

Result:

- Catch fraction: 6/6.
- Errors: 0.
- Elapsed wall time: 193.1s.
- Median min distance: 0.488m.
- P90 min distance: 0.497m.

Per-seed execution min distances:

- Seed 1: 0.499m.
- Seed 2: 0.481m.
- Seed 3: 0.456m.
- Seed 4: 0.472m.
- Seed 5: 0.495m.
- Seed 6: 0.494m.

This closes the previous validation gap: the portfolio works through the actual policy lifecycle, not only through direct plan replay scoring.

## Scoring Smoke

Ran a two-seed `portfolio3` sweep to verify the updated sweep reporting:

```bash
python docs/analysis/control_sims/rpg_improve/run_candidate_sweep.py \
  --preset portfolio3 \
  --seeds 2,3 \
  --workers 3 \
  --progress-every 1 \
  --output-dir docs/analysis/control_sims/rpg_improve/iter_7_sweep_scoring_smoke
```

Result:

- 6 tasks in 63.0s.
- Best per-seed portfolio: 2/2.
- `best_by_seed.csv` now includes `rollout_capture_steps` and selects by production-like scoring.

For seed 2, the updated report chooses `rate0p5_cmd0p2_body0p05_win8` because it has more capture dwell, even though `rate0p5_body0p2_win6` has a slightly smaller min distance. That matches the production selector's intended semantics.

## Tests

```bash
python -m py_compile \
  control_sims/rpg_time_optimal/planner.py \
  control_sims/rpg_time_optimal/portfolio_policy.py \
  docs/analysis/control_sims/rpg_improve/run_diagnostics.py \
  docs/analysis/control_sims/rpg_improve/run_candidate_sweep.py \
  docs/analysis/control_sims/rpg_improve/run_portfolio_validation.py

python -m pytest -q tests/controllers/test_rpg_time_optimal.py
```

Result: `5 passed in 30.78s`.

## Interpretation

We now have three levels of 6/6 evidence:

- clean direct replay-selected production candidate set;
- production `solve_portfolio_plan()` validation;
- actual `RpgTimeOptimalPortfolioMotorPolicy` execution through `SimRunner`.

The remaining issue is margin and cost. The actual runner catches are close to the 0.5m threshold, with P90 min distance at 0.497m. The portfolio is real, but it is expensive and has limited execution slack.

## Next Steps

- Use the new timing fields to measure build vs IPOPT solve vs replay cost across the production portfolio.
- Implement same-layout warm starts so the second and third portfolio candidates start from the first candidate's solution.
- Prototype a time-capped robust second-stage solve, but keep the current portfolio as the regression baseline until the single-solve path proves 6/6 with comparable or better margin.
