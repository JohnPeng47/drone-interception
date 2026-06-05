# Iteration 5 Notes

## Reviewer Observation

The reviewer identified that iteration 4's smoothness/window objective was useful but traded directly against time. Smoothness was added to the same objective as total time, so the solver could buy smoother trajectories by increasing intercept time. That helped seeds 2 and 4, but regressed seed 6 by stretching its plan from about 0.887s to about 1.221s.

The reviewer suggested:

- Add better divergence/closest-approach diagnostics.
- Run a generic candidate grid containing the known rate-only and smooth/window catch families.
- Score candidates by direct SimEngine RPM replay.
- If seed 5 still missed, move to a two-stage time-optimal then robustness solve with a bounded time-inflation cap.

## Implementation

Added `run_candidate_sweep.py` under this analysis folder. It runs independent `(candidate, seed)` planner solves and direct SimEngine RPM rollouts through a process pool, writes compact ranking artifacts, and now streams progress plus `candidate_results.partial.csv` while long sweeps are still running.

Updated `run_diagnostics.py` so normal diagnostics also parallelize the expensive independent work:

- `--planner-workers` controls process workers for planner solves and direct planned-RPM rollouts.
- `--max-envs` controls SimRunner batch width for policy execution.
- CTBR diagnostics now reuse cached planner results instead of solving plans again inside the policy.

Smoke checks:

- `python -m py_compile docs/analysis/control_sims/rpg_improve/run_diagnostics.py docs/analysis/control_sims/rpg_improve/run_candidate_sweep.py`
- One-seed candidate sweep smoke completed.
- Two-seed parallel diagnostics smoke completed with `--planner-workers 2 --max-envs 2`.

## Sweep Results

Ran:

```bash
python docs/analysis/control_sims/rpg_improve/run_candidate_sweep.py \
  --preset iter5 \
  --ipopt-max-iter 120 \
  --workers 6 \
  --progress-every 3 \
  --output-dir docs/analysis/control_sims/rpg_improve/iter_5_candidate_sweep_iter120
```

The sweep evaluated 11 generic candidates across all 6 seeds: 66 planner/replay tasks in 611.6s with 6 workers.

Best single candidate:

- `n60_rate0p5_cmd0p2_body0p05_win8_scale1_sub1`
- Direct RPM rollout catch count: 4/6
- Worst rollout min distance: 1.157m

Best per-seed portfolio:

- Direct RPM rollout catch count: 6/6
- Worst selected min distance: 0.282m
- Mean selected min distance: 0.132m

Clean per-seed catch evidence from the sweep:

- Seed 1: `cmd0_body0.2_win8`, 0.398m, clean solve.
- Seed 2: multiple clean catches; best observed 0.031m.
- Seed 3: multiple clean catches; best observed 0.007m.
- Seed 4: `cmd0_body0.2_win8`, 0.097m, clean solve.
- Seed 5: `cmd0_body0.2_win6`, 0.282m, clean solve.
- Seed 6: `cmd0.2_body0.05_win8`, 0.103m, clean solve.

Seed 5 was the main new result: it was previously not caught by any stored candidate.

Focused high-cap validation for seed 5:

```bash
python docs/analysis/control_sims/rpg_improve/run_diagnostics.py \
  --seeds 5 \
  --terminal-nodes 60 \
  --ipopt-max-iter 300 \
  --planner-rate-limit-scale 0.5 \
  --cpc-tolerance-m 0.1 \
  --body-rate-smoothness-weight 0.2 \
  --terminal-capture-window-nodes 6 \
  --policies motor \
  --planner-workers 1 \
  --max-envs 1 \
  --output-dir docs/analysis/control_sims/rpg_improve/iter_5_seed5_body0p2_window6_iter300
```

Result:

- Planner solve succeeded with max constraint violation `9.12e-09`.
- Direct RPM rollout caught with min distance 0.282m.
- Motor policy execution caught with min/final distance 0.495m.

## Interpretation

Parallelism is now in place for broad data collection. The candidate sweep showed that the planner has enough generic trajectory families to catch all six seeds when plans are selected by direct SimEngine replay, but no single global candidate in this grid catches all six.

The strongest general optimization remains a candidate selector or two-stage robust planner:

- A fixed generic candidate set plus SimEngine replay scoring already gives a 6/6 portfolio in analysis.
- A production-quality version should avoid seed-specific branching and select plans by model-predicted replay robustness.
- The next core planner change should reduce solve cost and improve single-candidate robustness: solve time-optimal first, then optimize smoothness/dwell with `T <= T* * 1.05` or `1.10`.

## Next Steps

- Add a fixed candidate-portfolio planner/policy path that solves a generic set of candidate configs, replays each candidate through SimEngine, and selects the lowest rollout minimum distance.
- Add progress and incremental outputs to any future broad analysis scripts by default.
- Implement warm-started two-stage optimization to reduce per-candidate cost and avoid smoothness-induced time inflation.
