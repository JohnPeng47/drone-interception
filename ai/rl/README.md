# SimEngine RL Runner

This RL path uses `backends/csim` `SimEngine` through the Python bindings.
It does not use `intercept_env`.

The batch runner loads generated `.csimin` scenario tables, keeps a fixed-width
set of C `SimEngine` slots filled from those scenarios, and trains one shared
continuous CTBR policy over batched rollouts.

The PPO implementation in `ai/rl/simengine_batch/puffer_ppo.py` is adapted from
PufferLib's PyTorch trainer (`puffer/pufferlib/torch_pufferl.py`): unsquashed
Normal continuous actions, Puffer advantage/V-trace recurrence, prioritized
trajectory minibatch sampling, clipped value loss, replay ratio, and optional
Muon optimizer.

## Smoke

```bash
python -m ai.rl.simengine_batch.train \
  --scenario-table .runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples.csimin \
  --manifest .runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples_grid_manifest.json \
  --max-scenarios 64 \
  --num-envs 8 \
  --total-timesteps 2048 \
  --horizon 32
```

## Throughput Benchmark

```bash
python -m ai.rl.simengine_batch.benchmark \
  --scenario-table .runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples.csimin \
  --manifest .runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples_grid_manifest.json \
  --num-envs 64 128 256 512 1024 \
  --steps 256
```

## Large Dataset

```bash
python -m ai.rl.simengine_batch.train \
  --scenario-table .runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples.csimin \
  --manifest .runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples_grid_manifest.json \
  --num-envs 1024 \
  --horizon 128 \
  --total-timesteps 500000000
```

## Restartable Remote Training

Batch training checkpoints include the policy weights, PPO optimizer state,
PPO epoch, global step, RNG state, and scenario generator sampling state.
Live SimEngine slots are not serialized; a resumed process starts fresh slots
from the saved generator state and continues training from the checkpoint step.

To upload each checkpoint and `latest.pt` as training runs:

```bash
S3_PREFIX=s3://drone-interception-rl-checkpoints-241810645840-us-east-2/runs/my-run \
CHECKPOINT_INTERVAL_STEPS=1000000 \
./ai/rl/scripts/remote_train_simengine.sh
```

To cold-start from S3 on a new pod:

```bash
RESUME_S3_URI=s3://drone-interception-rl-checkpoints-241810645840-us-east-2/runs/my-run/checkpoints/latest.pt \
S3_PREFIX=s3://drone-interception-rl-checkpoints-241810645840-us-east-2/runs/my-run \
./ai/rl/scripts/remote_train_simengine.sh
```

For local resume from an already downloaded checkpoint:

```bash
python -m ai.rl.simengine_batch.train \
  --scenario-table .runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples.csimin \
  --manifest .runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples_grid_manifest.json \
  --resume-from checkpoints/simengine_batch/latest.pt
```
