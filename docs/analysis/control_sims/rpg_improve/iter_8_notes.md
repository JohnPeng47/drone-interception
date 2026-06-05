# Iteration 8 Notes

## Reviewer Observation

Newton recommended the safest cost-reduction path:

- Add per-candidate portfolio traces before changing behavior.
- Implement x0-only same-layout warm starts.
- Do not warm-start IPOPT multipliers because candidates have different constraint counts.
- Avoid candidate pruning until the trace data shows a safe global rule.
- Keep the two-stage robust solve analysis-only until it beats the current portfolio on catch rate, dwell, margin, and cost.

## Implementation

Added planner warm-start support:

- `RpgTimeOptimalPlan` now stores the solved `decision_vector`.
- `RpgTimeOptimalPlanner.solve()` accepts `initial_guess`.
- `_solve_terminal_ocp()` uses a finite, length-matched decision vector as IPOPT `x0`.
- Warm starts are primal-vector only; no `lam_g0` or multiplier reuse.

Added portfolio trace support:

- `RpgSelectedPortfolioPlan.traces`
- `RpgPortfolioCandidateTrace`
- per-candidate fields for selected, clean, warm-started, replay catch, dwell, min distance, solve/build/optimizer/replay timing, and constraint status.

Updated production portfolio selection:

- `solve_portfolio_plan()` solves candidate 1 cold.
- Later candidates warm-start from the last successful same-layout plan.
- Same-layout means matching terminal node count and decision-vector length.

Updated analysis output:

- `run_portfolio_validation.py` now writes:
  - `portfolio_validation.csv`
  - `portfolio_candidates.csv`
- Its summary includes aggregate candidate solve, optimizer, and replay wall time.
- `run_candidate_sweep.py` now aggregates NLP build and optimizer time in `candidate_summary.csv`.

## Validation

Focused tests:

```bash
python -m pytest -q tests/controllers/test_rpg_time_optimal.py
```

Result: `5 passed in 40.79s`.

Required sim-store/backend smoke tests:

```bash
python -m pytest -q tests/test_sim_instance_store.py tests/test_puffer_backend_smoke.py
```

Result: `18 passed in 0.55s`.

Warm production portfolio validation:

```bash
python docs/analysis/control_sims/rpg_improve/run_portfolio_validation.py \
  --workers 6 \
  --output-dir docs/analysis/control_sims/rpg_improve/iter_8_warm_portfolio_validation
```

Result:

- Catch fraction: 6/6.
- Elapsed wall time: 153.3s.
- Worst selected replay min distance: 0.398m.
- Candidate solve wall sum: 756.5s.
- Candidate optimizer wall sum: 508.1s.
- Candidate replay wall sum: 81.2s.

Warm-start trace:

- 6 cold candidate solves.
- 12 warm-started candidate solves.
- Selected candidates stayed consistent with the previous production validation:
  - Seed 1: `rate0p5_body0p2_win8`
  - Seed 2: `rate0p5_cmd0p2_body0p05_win8`
  - Seed 3: `rate0p5_body0p2_win8`
  - Seed 4: `rate0p5_body0p2_win8`
  - Seed 5: `rate0p5_body0p2_win6`
  - Seed 6: `rate0p5_cmd0p2_body0p05_win8`

Aggregate solve time by candidate:

- `rate0p5_body0p2_win8`: 279.2s solve, 203.4s optimizer, selected 3 times, caught 5/6.
- `rate0p5_body0p2_win6`: 233.4s solve, 143.7s optimizer, selected 1 time, caught 3/6.
- `rate0p5_cmd0p2_body0p05_win8`: 243.9s solve, 161.0s optimizer, selected 2 times, caught 3/6.

End-to-end warm policy runner:

```bash
python scripts/runners/control_sim/rpg_time_optimal_motor_portfolio.py \
  --scenario-table scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin \
  --workers 6 \
  --max-envs 1 \
  --progress-every 1 \
  --out-dir docs/analysis/control_sims/rpg_improve/iter_8_warm_portfolio_policy_runner
```

Result:

- Catch fraction: 6/6.
- Errors: 0.
- Elapsed wall time: 190.6s.
- Median min distance: 0.488m.
- P90 min distance: 0.497m.

## Interpretation

Warm starts preserved the 6/6 behavior and reduced direct validation wall time from about 178s to 153s. The end-to-end policy runner stayed around the same wall time as iteration 7, likely because it is dominated by the slowest worker and process overhead rather than aggregate candidate solve time.

The trace data shows the biggest costs are still CasADi/IPOPT solve time, not replay. Build time is also significant, so further speedups may need either solver reuse, fewer candidates, or a single robust solve formulation.

## Next Steps

- Use the trace data to evaluate safe global candidate pruning, especially whether `rate0p5_body0p2_win8` can be accepted early when it has strong dwell/margin.
- Prototype a two-stage robust solve in analysis and compare against the portfolio baseline on replay catch, dwell, min distance, and solve wall time.
- Consider solver/NLP object reuse only if CasADi construction time remains a bottleneck after pruning or two-stage experiments.
