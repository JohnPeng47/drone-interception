# RPG Time-Optimal Diagnostics

Scenario table: `scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin`
Scenarios: 4

## Planner

- Ideal feasible fraction: 1.000
- Terminal tolerance satisfied fraction: 1.000
- Mean planned min distance: 0.100 m
- Worst planned min distance: 0.100 m
- Max constraint violation: 4.038e-09

## Plan Rollout

- SimEngine RPM rollout catch fraction: 0.500
- Mean rollout min distance: 1.789 m
- Worst rollout min distance: 4.632 m
- Mean rollout tracking error: 0.458 m
- Mean rollout body-rate tracking error: 2.185 rad/s

## Execution

### rpg_time_optimal_motor_feedforward

- Catch fraction: 0.500
- Errors: 0
- Median min distance: 1.207 m
- Mean visible fraction: 0.014
- Mean tracking error: 5.089 m
- Max tracking error: 36.033 m
- Classifications: `{'caught': 2, 'execution_tracking_or_model_mismatch': 2}`

## Interpretation

The planner reaches the target under its own model, but execution usually misses. The next work should focus on plan tracking, online replanning, or reducing model mismatch.
