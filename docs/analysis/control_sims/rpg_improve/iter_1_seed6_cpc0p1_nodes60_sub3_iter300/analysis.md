# RPG Time-Optimal Diagnostics

Scenario table: `scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin`
Scenarios: 1

## Planner

- Ideal feasible fraction: 1.000
- Terminal tolerance satisfied fraction: 0.000
- Mean planned min distance: 0.102 m
- Worst planned min distance: 0.102 m
- Max constraint violation: 2.321e-02

## Plan Rollout

- SimEngine RPM rollout catch fraction: 0.000
- Mean rollout min distance: 1.072 m
- Worst rollout min distance: 1.072 m
- Mean rollout tracking error: 0.245 m

## Execution

### rpg_time_optimal_ctbr

- Catch fraction: 0.000
- Errors: 0
- Median min distance: 5.176 m
- Mean visible fraction: 0.018
- Mean tracking error: 13.022 m
- Max tracking error: 34.828 m
- Classifications: `{'execution_tracking_or_model_mismatch': 1}`

### rpg_time_optimal_motor_feedforward

- Catch fraction: 0.000
- Errors: 0
- Median min distance: 0.978 m
- Mean visible fraction: 0.037
- Mean tracking error: 7.651 m
- Max tracking error: 28.311 m
- Classifications: `{'execution_tracking_or_model_mismatch': 1}`

## Interpretation

The planner reaches the target under its own model, but execution usually misses. The next work should focus on plan tracking, online replanning, or reducing model mismatch.
