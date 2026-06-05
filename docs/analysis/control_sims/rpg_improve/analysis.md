# RPG Time-Optimal Diagnostics

Scenario table: `scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin`
Scenarios: 6

## Planner

- Ideal feasible fraction: 1.000
- Mean planned min distance: 0.500 m
- Worst planned min distance: 0.500 m

## Execution

### rpg_time_optimal_ctbr

- Catch fraction: 0.000
- Errors: 0
- Median min distance: 2.200 m
- Mean visible fraction: 0.059
- Mean tracking error: 10.095 m
- Max tracking error: 41.890 m
- Classifications: `{'execution_tracking_or_model_mismatch': 6}`

### rpg_time_optimal_motor_feedforward

- Catch fraction: 0.167
- Errors: 0
- Median min distance: 1.723 m
- Mean visible fraction: 0.016
- Mean tracking error: 8.087 m
- Max tracking error: 45.504 m
- Classifications: `{'execution_tracking_or_model_mismatch': 5, 'caught': 1}`

## Interpretation

The planner reaches the target under its own model, but execution usually misses. The next work should focus on plan tracking, online replanning, or reducing model mismatch.
