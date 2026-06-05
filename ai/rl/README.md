# Puffer Intercept RL Runner

This RL path trains against the native `ai/rl/puffer_intercept` C vecenv.
It consumes generated `.csimin` scenario tables.

The PPO implementation in `ai/rl/puffer_intercept/puffer_ppo.py` is adapted from
PufferLib's PyTorch trainer (`puffer/pufferlib/torch_pufferl.py`): unsquashed
Normal continuous actions, Puffer advantage/V-trace recurrence, prioritized
trajectory minibatch sampling, clipped value loss, replay ratio, and optional
Muon optimizer.

## Smoke

```bash
python scripts/runners/rl/puffer_intercept_runner.py \
  --scenario-table scripts/generators/sim_instances/sobol_samples_512.csimin \
  --num-envs 8 \
  --total-timesteps 2048 \
  --horizon 32
```

## Throughput Benchmark

```bash
python scripts/runners/benchmark_intercept_envs.py \
  --scenario-file scripts/generators/sim_instances/sobol_samples_512.csimin \
  --num-envs 64 128 256 512 1024 \
  --steps 256
```

## Large Dataset

```bash
python scripts/runners/rl/puffer_intercept_runner.py \
  --scenario-table scripts/generators/sim_instances/sobol_samples_512.csimin \
  --num-envs 1024 \
  --horizon 128 \
  --total-timesteps 500000000
```

## Restartable Remote Training

Training checkpoints include policy weights, PPO optimizer state, PPO epoch,
global step, RNG state, and native training metadata.

To upload each checkpoint and `latest.pt` as training runs:

```bash
S3_PREFIX=s3://drone-interception-rl-checkpoints-241810645840-us-east-2/runs/my-run \
CHECKPOINT_INTERVAL_STEPS=1000000 \
python ai/rl/scripts/runpod_puffer_intercept_training.py --run-name my-run
```

To cold-start from S3 on a new pod:

```bash
RESUME_S3_URI=s3://drone-interception-rl-checkpoints-241810645840-us-east-2/runs/my-run/checkpoints/latest.pt \
S3_PREFIX=s3://drone-interception-rl-checkpoints-241810645840-us-east-2/runs/my-run \
python ai/rl/scripts/runpod_puffer_intercept_training.py --run-name my-run-resume
```

For local resume from an already downloaded checkpoint:

```bash
python scripts/runners/rl/puffer_intercept_runner.py \
  --scenario-table scripts/generators/sim_instances/sobol_samples_512.csimin \
  --resume-from checkpoints/puffer_intercept/latest.pt
```
