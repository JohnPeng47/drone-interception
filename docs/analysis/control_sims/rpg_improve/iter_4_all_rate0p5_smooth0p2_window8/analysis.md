# RPG Time-Optimal Diagnostics

Scenario table: `scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin`
Scenarios: 6

## Planner

- Ideal feasible fraction: 1.000
- Terminal tolerance satisfied fraction: 1.000
- Mean planned min distance: 0.031 m
- Worst planned min distance: 0.037 m
- Max constraint violation: 8.918e-09

## Plan Rollout

- SimEngine RPM rollout catch fraction: 0.500
- Mean rollout min distance: 0.677 m
- Worst rollout min distance: 2.526 m
- Mean rollout tracking error: 0.414 m
- Mean rollout body-rate tracking error: 0.864 rad/s

## Execution

### rpg_time_optimal_motor_feedforward

- Catch fraction: 0.500
- Errors: 0
- Median min distance: 0.508 m
- Mean visible fraction: 0.042
- Mean tracking error: 3.578 m
- Max tracking error: 24.505 m
- Classifications: `{'execution_tracking_or_model_mismatch': 3, 'caught': 3}`

## Interpretation

The planner reaches the target under its own model, but execution usually misses. The next work should focus on plan tracking, online replanning, or reducing model mismatch.
