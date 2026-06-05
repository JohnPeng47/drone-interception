# RPG Time-Optimal Diagnostics

Scenario table: `scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin`
Scenarios: 6

## Planner

- Ideal feasible fraction: 1.000
- Terminal tolerance satisfied fraction: 1.000
- Mean planned min distance: 0.100 m
- Worst planned min distance: 0.100 m
- Max constraint violation: 5.796e-09

## Plan Rollout

- SimEngine RPM rollout catch fraction: 0.500
- Mean rollout min distance: 1.423 m
- Worst rollout min distance: 4.632 m
- Mean rollout tracking error: 0.343 m
- Mean rollout body-rate tracking error: 2.363 rad/s

## Execution

### rpg_time_optimal_motor_feedforward

- Catch fraction: 0.500
- Errors: 0
- Median min distance: 0.748 m
- Mean visible fraction: 0.020
- Mean tracking error: 4.724 m
- Max tracking error: 36.033 m
- Classifications: `{'caught': 3, 'execution_tracking_or_model_mismatch': 3}`

## Interpretation

The planner reaches the target under its own model, but execution usually misses. The next work should focus on plan tracking, online replanning, or reducing model mismatch.
