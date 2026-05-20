# RL Training

This directory owns reinforcement-learning training configuration and training
infrastructure.

Keep simulator and backend code outside this directory:

- `intercept_env/` contains the Puffer C environment implementation.
- `backends/` contains Python simulator/backend adapters.
- `shared/intercept_sim/` contains simulator-independent interception logic,
  scenarios, metrics, and telemetry.

Current contents:

- `config/` stores PufferLib training configs for the intercept task.
- `scripts/` stores RunPod upload, bootstrap, sync, and teardown helpers for
  training runs.
