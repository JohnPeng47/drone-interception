# RL Training

This directory owns reinforcement-learning training configuration and training
infrastructure.

Keep simulator and backend code outside this directory:

- `intercept_env/` contains the Puffer C environment implementation.
- `backends/` contains Python simulator/backend adapters.
- `control_sims/beihang_paper_sim/sim/generator/` contains the current Python
  scenario generators for deterministic control-sim runs.

Current contents:

- `config/` stores PufferLib training configs for the intercept task.
- `scripts/` stores RunPod upload, bootstrap, sync, and teardown helpers for
  training runs.
