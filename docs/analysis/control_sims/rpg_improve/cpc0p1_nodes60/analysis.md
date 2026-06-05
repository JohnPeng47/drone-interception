# RPG Time-Optimal Diagnostics

Scenario table: `scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin`
Planner config: cpc_tolerance_m=0.1, terminal_nodes=60
Scenarios: 6

## Planner

- Ideal feasible fraction: 0.833
- Mean planned min distance: 0.920 m
- Worst planned min distance: 4.949 m
- Mean planner wall time: 24.983 s

## Execution

### rpg_time_optimal_ctbr

- Catch fraction: 0.000
- Errors: 0
- Median min distance: 1.950 m
- p90 min distance: 4.468 m
- Mean visible fraction: 0.049
- Mean tracking error: 9.787 m
- Max tracking error: 42.966 m
- Classifications: `{'execution_tracking_or_model_mismatch': 5, 'planner_ideal_misses': 1}`

### rpg_time_optimal_motor_feedforward

- Catch fraction: 0.333
- Errors: 0
- Median min distance: 1.267 m
- p90 min distance: 3.548 m
- Mean visible fraction: 0.029
- Mean tracking error: 5.375 m
- Max tracking error: 31.319 m
- Classifications: `{'execution_tracking_or_model_mismatch': 3, 'caught': 2, 'planner_ideal_misses': 1}`
