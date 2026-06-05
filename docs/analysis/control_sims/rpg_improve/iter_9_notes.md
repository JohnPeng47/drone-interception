# Iteration 9 Notes

## Reviewer Observation

Hooke recommended testing early acceptance with trace replay before changing production behavior.

Key points:

- `solve_portfolio_plan()` still solved and replayed every candidate, so skipping later candidates is the biggest remaining cost lever.
- Candidate 0, `rate0p5_body0p2_win8`, catches 5/6 and should remain first.
- Do not accept on `caught` alone. Seed 5's candidate-0 catch is weak: 3 capture steps, 0.489m replay min distance, and 0.838m mean tracking error.
- A general gate like `clean && caught && rollout_capture_steps >= 15` looked promising from the iteration-8 trace.
- Any gate that changes seed 2's selected candidate must pass actual `SimRunner` execution, because replay margin and runner margin are not identical.

## Implementation

Added trace fields:

- `RpgPlanReplayScore.rollout_max_consecutive_capture_steps`
- `RpgPortfolioCandidateTrace.rollout_max_consecutive_capture_steps`
- `RpgPortfolioCandidateTrace.skipped`
- `RpgPortfolioCandidateTrace.stop_reason`
- `RpgPortfolioCandidateTrace.optimizer_iterations`
- `RpgTimeOptimalPlan.optimizer_iterations`

Added production early accept:

- `RpgTimeOptimalPortfolioMotorPolicy(..., early_accept_min_consecutive_capture_steps=15)`
- `solve_portfolio_plan(..., early_accept_min_consecutive_capture_steps=15)`
- The default gate is seed-agnostic:
  - clean replay score
  - replay caught radius
  - max consecutive capture steps >= 15
- When early accept triggers, remaining candidates are written as skipped trace rows with `stop_reason=early_accept`.

Added analysis replay:

- `docs/analysis/control_sims/rpg_improve/replay_portfolio_pruning.py`
- It replays candidate traces against seed-agnostic gates and writes:
  - `pruning_summary.csv`
  - `pruning_selections.csv`

Added tests:

- Portfolio early-accept predicate rejects dirty scores.
- It rejects caught-but-weak-dwell scores.
- It rejects missed scores.

## Pruning Replay

Ran:

```bash
python docs/analysis/control_sims/rpg_improve/replay_portfolio_pruning.py \
  --trace-csv docs/analysis/control_sims/rpg_improve/iter_8_warm_portfolio_validation/portfolio_candidates.csv \
  --output-dir docs/analysis/control_sims/rpg_improve/iter_9_pruning_replay
```

Best replayed gate:

- `consec10` and `consec15` both kept 6/6.
- Simulated solved candidates: 9/18.
- Simulated skipped candidates: 9/18.
- Worst selected replay min distance: 0.398m.

The `consec15` gate is the production default because it rejects seed 5's weak candidate-0 catch while still accepting strong candidate-0 plans.

## Validation

Production early-accept validation:

```bash
python docs/analysis/control_sims/rpg_improve/run_portfolio_validation.py \
  --workers 6 \
  --output-dir docs/analysis/control_sims/rpg_improve/iter_9_early_accept_validation
```

Result:

- Catch fraction: 6/6.
- Elapsed wall time: 93.7s.
- Candidate solve wall sum: 382.1s.
- Candidate optimizer wall sum: 259.1s.
- Candidate replay wall sum: 32.6s.
- Worst selected replay min distance: 0.398m.
- Solved candidates: 9.
- Skipped candidates: 9.

Selected candidates:

- Seed 1: `rate0p5_body0p2_win8`, 0.398m replay min, 20 consecutive capture steps.
- Seed 2: `rate0p5_body0p2_win8`, 0.213m replay min, 24 consecutive capture steps.
- Seed 3: `rate0p5_body0p2_win8`, 0.183m replay min, 19 consecutive capture steps.
- Seed 4: `rate0p5_body0p2_win8`, 0.097m replay min, 44 consecutive capture steps.
- Seed 5: `rate0p5_body0p2_win6`, 0.282m replay min, 17 consecutive capture steps.
- Seed 6: `rate0p5_cmd0p2_body0p05_win8`, 0.103m replay min, 34 consecutive capture steps.

Actual policy runner:

```bash
python scripts/runners/control_sim/rpg_time_optimal_motor_portfolio.py \
  --scenario-table scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin \
  --workers 6 \
  --max-envs 1 \
  --progress-every 1 \
  --out-dir docs/analysis/control_sims/rpg_improve/iter_9_early_accept_policy_runner
```

Result:

- Catch fraction: 6/6.
- Errors: 0.
- Elapsed wall time: 116.3s.
- Median min distance: 0.494m.
- P90 min distance: 0.498m.

Per-seed actual runner min distances:

- Seed 1: 0.499m.
- Seed 2: 0.498m.
- Seed 3: 0.456m.
- Seed 4: 0.472m.
- Seed 5: 0.495m.
- Seed 6: 0.494m.

## Tests

```bash
python -m pytest -q tests/controllers/test_rpg_time_optimal.py
```

Result: `6 passed in 39.52s`.

```bash
python -m pytest -q tests/test_sim_instance_store.py tests/test_puffer_backend_smoke.py
```

Result: `18 passed in 0.70s`.

## Interpretation

Early acceptance preserved the 6/6 replay-selected portfolio and the 6/6 actual policy runner result while cutting validation wall time from 153.3s to 93.7s. The actual runner also improved from about 190.6s to 116.3s.

The main caveat is execution margin. The actual runner stops on first intercept, and the terminal distances remain close to 0.5m. Direct replay dwell and replay min distance are better robustness signals than the runner's terminal distance.

## Next Steps

- Stop here per user request; do not start iteration 10 automatically.
- If resumed, iteration 10 should focus on increasing execution margin rather than only reducing solve cost.
- The best candidate for iteration 10 is an analysis-only two-stage robust solve or a stricter replay gate that includes more margin, compared against this early-accept portfolio baseline.
