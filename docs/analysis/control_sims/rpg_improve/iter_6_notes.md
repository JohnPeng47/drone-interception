# Iteration 6 Notes

## Reviewer Observation

Mencius recommended implementing the candidate-portfolio motor path before the two-stage time-capped robust solve.

Key points:

- Iteration 5 already proved a general 6/6 route: solve a fixed generic candidate set, replay each candidate through SimEngine, then select by replay robustness.
- A clean 3-candidate subset appeared sufficient:
  - `rate0p5_body0p2_win8`
  - `rate0p5_body0p2_win6`
  - `rate0p5_cmd0p2_body0p05_win8`
- The selector must reject dirty plans before scoring. The previous sweep helper could choose `Maximum_Iterations_Exceeded` plans when ranking by min distance alone.
- Production code should not import from `docs/analysis`; replay/scoring belongs under `control_sims/rpg_time_optimal`.
- The existing CLI runner uses the CTBR policy, while the 6/6 evidence is motor feedforward, so the portfolio motor path should be explicit.

## Implementation

Added production portfolio code:

- `control_sims/rpg_time_optimal/portfolio_policy.py`
  - `RpgTimeOptimalPortfolioMotorPolicy`
  - `solve_portfolio_plan`
  - `score_plan_replay`
  - `select_best_scored_plan`
  - `DEFAULT_PORTFOLIO_CANDIDATES`

The portfolio policy:

- solves the fixed generic candidate set without seed-specific branching;
- rejects plans that are not solver-successful, terminal-tolerance-satisfied, planned-feasible, finite, and under the constraint-violation tolerance;
- replays clean plans through `BatchPufferSimEngineBackend`;
- selects by replay catch first, then capture dwell, min distance, tracking error, and plan time;
- executes the selected plan as direct motor RPM feedforward.

Also:

- promoted the motor command sampler in `motor_feedforward_policy.py` to `sample_motor_speed_command`;
- added package exports in `control_sims/rpg_time_optimal/__init__.py`;
- added `scripts/runners/control_sim/rpg_time_optimal_motor_portfolio.py`;
- added selector tests in `tests/controllers/test_rpg_time_optimal.py`;
- tightened `run_candidate_sweep.py` so future sweep summaries rank only clean plans;
- added `portfolio3` preset wired to production defaults;
- added `run_portfolio_validation.py`, which calls production `solve_portfolio_plan()` in parallel and writes validation artifacts.

## Validation

Fast checks:

```bash
python -m py_compile \
  control_sims/rpg_time_optimal/portfolio_policy.py \
  control_sims/rpg_time_optimal/motor_feedforward_policy.py \
  docs/analysis/control_sims/rpg_improve/run_candidate_sweep.py

python -m pytest -q tests/controllers/test_rpg_time_optimal.py
```

Result: `5 passed in 31.09s`.

Required sim logic smoke tests:

```bash
python -m pytest -q tests/test_sim_instance_store.py tests/test_puffer_backend_smoke.py
```

Result: `18 passed in 0.18s`.

Production candidate-set sweep:

```bash
python docs/analysis/control_sims/rpg_improve/run_candidate_sweep.py \
  --preset portfolio3 \
  --workers 6 \
  --progress-every 3 \
  --output-dir docs/analysis/control_sims/rpg_improve/iter_6_portfolio3_sweep
```

Result:

- 18 tasks in 180.7s.
- Best single candidate: `rate0p5_body0p2_win8`, 5/6.
- Best clean per-seed portfolio: 6/6.
- Worst selected replay min distance: 0.398m.

Production portfolio validation:

```bash
python docs/analysis/control_sims/rpg_improve/run_portfolio_validation.py \
  --workers 6 \
  --output-dir docs/analysis/control_sims/rpg_improve/iter_6_production_portfolio_validation
```

Result:

- `solve_portfolio_plan()` selected 6/6 catches.
- Worst selected replay min distance: 0.398m.
- All selected plans had clean solver success and constraint violation below `1e-8`.

Selected production candidates:

- Seed 1: `rate0p5_body0p2_win8`, 0.398m.
- Seed 2: `rate0p5_cmd0p2_body0p05_win8`, 0.057m.
- Seed 3: `rate0p5_body0p2_win8`, 0.183m.
- Seed 4: `rate0p5_body0p2_win8`, 0.097m.
- Seed 5: `rate0p5_body0p2_win6`, 0.282m.
- Seed 6: `rate0p5_cmd0p2_body0p05_win8`, 0.103m.

## Interpretation

This iteration gives a production-side, seed-agnostic 6/6 replay-selected motor portfolio. The single best global candidate improved to 5/6, but the reliable route is portfolio selection by SimEngine replay margin.

The major remaining weakness is cost: production validation took about 178s with six workers because each seed solves three 60-node NLPs. The next iteration should focus on reducing portfolio solve cost without losing the 6/6 behavior.

## Next Steps

- Add warm-start support or a two-stage time-capped robust solve to reduce per-candidate solve time.
- Consider parallelizing candidate solves inside `solve_portfolio_plan()` only when it is not already running inside a process pool, to avoid oversubscription.
- Add a slow/manual integration test or runner command that executes `RpgTimeOptimalPortfolioMotorPolicy` through `SimRunner`, not just replay-selected plan validation.
