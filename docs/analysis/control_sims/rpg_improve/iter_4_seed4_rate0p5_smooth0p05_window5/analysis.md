# RPG Time-Optimal Diagnostics

Scenario table: `scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin`
Scenarios: 1

## Planner

- Ideal feasible fraction: 1.000
- Terminal tolerance satisfied fraction: 1.000
- Mean planned min distance: 0.045 m
- Worst planned min distance: 0.045 m
- Max constraint violation: 9.283e-09

## Plan Rollout

- SimEngine RPM rollout catch fraction: 0.000
- Mean rollout min distance: 1.298 m
- Worst rollout min distance: 1.298 m
- Mean rollout tracking error: 0.453 m
- Mean rollout body-rate tracking error: 1.227 rad/s

## Execution

### rpg_time_optimal_motor_feedforward

- Catch fraction: 0.000
- Errors: 0
- Median min distance: 1.298 m
- Mean visible fraction: 0.007
- Mean tracking error: 6.172 m
- Max tracking error: 24.322 m
- Classifications: `{'execution_tracking_or_model_mismatch': 1}`

## Interpretation

The planner reaches the target under its own model, but execution usually misses. The next work should focus on plan tracking, online replanning, or reducing model mismatch.
