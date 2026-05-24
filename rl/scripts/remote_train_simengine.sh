#!/usr/bin/env bash
set -euo pipefail

cd /workspace/drone-interception

SCENARIO_TABLE=${SCENARIO_TABLE:-/workspace/drone-interception/data/scenarios/sobol_samples.csimin}
SCENARIO_MANIFEST=${SCENARIO_MANIFEST:-/workspace/drone-interception/data/scenarios/sobol_samples_grid_manifest.json}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-500000000}
NUM_WORKERS=${NUM_WORKERS:-16}
ENVS_PER_WORKER=${ENVS_PER_WORKER:-64}
HORIZON=${HORIZON:-128}
WANDB_PROJECT=${WANDB_PROJECT:-drone-interception}
WANDB_GROUP=${WANDB_GROUP:-simengine-scenarios-4090}
WANDB_FLAG=${WANDB_FLAG:-}
if [ -f /root/.wandb_key ]; then
    WANDB_FLAG=${WANDB_FLAG:---wandb}
fi

mkdir -p logs checkpoints/simengine

echo "=== run metadata ===" | tee logs/simengine_train.log
git rev-parse HEAD 2>/dev/null | tee -a logs/simengine_train.log || true
sha256sum "$SCENARIO_TABLE" "$SCENARIO_MANIFEST" | tee -a logs/simengine_train.log
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a logs/simengine_train.log

python3 -m rl.simengine_env.train \
    --scenario-table "$SCENARIO_TABLE" \
    --manifest "$SCENARIO_MANIFEST" \
    --num-workers "$NUM_WORKERS" \
    --envs-per-worker "$ENVS_PER_WORKER" \
    --horizon "$HORIZON" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --checkpoint-dir checkpoints/simengine \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_GROUP" \
    $WANDB_FLAG \
    2>&1 | tee -a logs/simengine_train.log
