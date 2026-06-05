# RPG Time-Optimal Diagnostics

Scenario table: `scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin`
Scenarios: 1

## Planner

- Ideal feasible fraction: 1.000
- Terminal tolerance satisfied fraction: 1.000
- Mean planned min distance: 0.030 m
- Worst planned min distance: 0.030 m
- Max constraint violation: 8.661e-09

## Plan Rollout

- SimEngine RPM rollout catch fraction: 0.000
- Mean rollout min distance: 3.496 m
- Worst rollout min distance: 3.496 m
- Mean rollout tracking error: 6.102 m
- Mean rollout body-rate tracking error: 3.677 rad/s

## Execution

### rpg_time_optimal_motor_feedforward

- Catch fraction: 0.000
- Errors: 0
- Median min distance: 3.496 m
- Mean visible fraction: 0.033
- Mean tracking error: 6.599 m
- Max tracking error: 23.519 m
- Classifications: `{'execution_tracking_or_model_mismatch': 1}`

## Interpretation

The planner reaches the target under its own model, but execution usually misses. The next work should focus on plan tracking, online replanning, or reducing model mismatch.
