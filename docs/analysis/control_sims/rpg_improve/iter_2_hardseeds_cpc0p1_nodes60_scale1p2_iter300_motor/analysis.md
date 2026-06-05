# RPG Time-Optimal Diagnostics

Scenario table: `scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin`
Scenarios: 4

## Planner

- Ideal feasible fraction: 1.000
- Terminal tolerance satisfied fraction: 1.000
- Mean planned min distance: 0.100 m
- Worst planned min distance: 0.100 m
- Max constraint violation: 9.715e-09

## Plan Rollout

- SimEngine RPM rollout catch fraction: 0.000
- Mean rollout min distance: 1.937 m
- Worst rollout min distance: 3.279 m
- Mean rollout tracking error: 1.248 m

## Execution

### rpg_time_optimal_motor_feedforward

- Catch fraction: 0.000
- Errors: 0
- Median min distance: 1.653 m
- Mean visible fraction: 0.015
- Mean tracking error: 8.891 m
- Max tracking error: 35.841 m
- Classifications: `{'execution_tracking_or_model_mismatch': 4}`

## Interpretation

The planner reaches the target under its own model, but execution usually misses. The next work should focus on plan tracking, online replanning, or reducing model mismatch.
