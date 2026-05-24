# SimEngine RL Runner

This RL path uses `backends/csim` `SimEngine` through the Python bindings.
It does not use `intercept_env`.

The runner loads generated `.csimin` scenario tables, creates parallel worker
processes, resets each environment from a table row, and trains a continuous
CTBR policy over SimEngine rollouts.

## Smoke

```bash
python -m rl.simengine_env.train \
  --scenario-table .runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples.csimin \
  --manifest .runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples_grid_manifest.json \
  --max-scenarios 64 \
  --num-workers 2 \
  --envs-per-worker 4 \
  --total-timesteps 2048 \
  --horizon 32
```

## Large Dataset

```bash
python -m rl.simengine_env.train \
  --scenario-table .runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples.csimin \
  --manifest .runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples_grid_manifest.json \
  --num-workers 16 \
  --envs-per-worker 64 \
  --horizon 128 \
  --total-timesteps 500000000
```
