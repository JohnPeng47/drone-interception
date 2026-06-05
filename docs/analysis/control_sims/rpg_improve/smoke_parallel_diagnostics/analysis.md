# RPG Time-Optimal Diagnostics

Scenario table: `scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin`
Scenarios: 2

## Planner

- Ideal feasible fraction: 0.000
- Terminal tolerance satisfied fraction: 0.000
- Mean planned min distance: 4.561 m
- Worst planned min distance: 4.988 m
- Max constraint violation: 4.438e+01

## Plan Rollout

- SimEngine RPM rollout catch fraction: 0.000
- Mean rollout min distance: 3.920 m
- Worst rollout min distance: 4.541 m
- Mean rollout tracking error: 0.745 m
- Mean rollout body-rate tracking error: 0.607 rad/s

## Execution

### rpg_time_optimal_motor_feedforward

- Catch fraction: 0.000
- Errors: 0
- Median min distance: 3.057 m
- Mean visible fraction: 0.041
- Mean tracking error: 9.681 m
- Max tracking error: 28.245 m
- Classifications: `{'planner_ideal_misses': 2}`

## Interpretation

The planner itself does not produce an ideal trajectory inside the capture radius for every seed. Fix the terminal OCP before tuning tracking.
