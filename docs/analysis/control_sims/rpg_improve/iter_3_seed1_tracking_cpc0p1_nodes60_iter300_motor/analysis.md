# RPG Time-Optimal Diagnostics

Scenario table: `scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin`
Scenarios: 1

## Planner

- Ideal feasible fraction: 1.000
- Terminal tolerance satisfied fraction: 1.000
- Mean planned min distance: 0.100 m
- Worst planned min distance: 0.100 m
- Max constraint violation: 9.715e-09

## Plan Rollout

- SimEngine RPM rollout catch fraction: 0.000
- Mean rollout min distance: 2.417 m
- Worst rollout min distance: 2.417 m
- Mean rollout tracking error: 0.643 m
- Mean rollout body-rate tracking error: 2.710 rad/s

## Execution

### rpg_time_optimal_motor_feedforward

- Catch fraction: 0.000
- Errors: 0
- Median min distance: 2.417 m
- Mean visible fraction: 0.006
- Mean tracking error: 9.966 m
- Max tracking error: 31.538 m
- Classifications: `{'execution_tracking_or_model_mismatch': 1}`

## Interpretation

The planner reaches the target under its own model, but execution usually misses. The next work should focus on plan tracking, online replanning, or reducing model mismatch.
