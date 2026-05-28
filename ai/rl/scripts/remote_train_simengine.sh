#!/usr/bin/env bash
set -euo pipefail

cd /workspace/drone-interception

SCENARIO_TABLE=${SCENARIO_TABLE:-/workspace/drone-interception/data/scenarios/sobol_samples.csimin}
SCENARIO_MANIFEST=${SCENARIO_MANIFEST:-}
MAX_SCENARIOS=${MAX_SCENARIOS:-}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-500000000}
NUM_ENVS=${NUM_ENVS:-1024}
HORIZON=${HORIZON:-128}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-8192}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
REPLAY_RATIO=${REPLAY_RATIO:-1.0}
OPTIMIZER=${OPTIMIZER:-adam}
CHECKPOINT_INTERVAL_STEPS=${CHECKPOINT_INTERVAL_STEPS:-1000000}
LOG_INTERVAL_STEPS=${LOG_INTERVAL_STEPS:-8192}
RESUME_FROM=${RESUME_FROM:-}
RESUME_S3_URI=${RESUME_S3_URI:-}
S3_PREFIX=${S3_PREFIX:-}
S3_CHECKPOINT_PREFIX=${S3_CHECKPOINT_PREFIX:-}
WANDB_PROJECT=${WANDB_PROJECT:-drone-interception}
WANDB_GROUP=${WANDB_GROUP:-simengine-batch-4090}
WANDB_NAME=${WANDB_NAME:-}
WANDB_FLAG=${WANDB_FLAG:-}
if [ -f /root/.wandb_key ]; then
    WANDB_FLAG=${WANDB_FLAG:---wandb}
fi
WANDB_NAME_ARGS=()
if [ -n "$WANDB_NAME" ]; then
    WANDB_NAME_ARGS=(--wandb-name "$WANDB_NAME")
fi
MAX_SCENARIO_ARGS=()
if [ -n "$MAX_SCENARIOS" ]; then
    MAX_SCENARIO_ARGS=(--max-scenarios "$MAX_SCENARIOS")
fi
RESUME_ARGS=()
if [ -n "$RESUME_FROM" ]; then
    RESUME_ARGS+=(--resume-from "$RESUME_FROM")
fi
if [ -n "$RESUME_S3_URI" ]; then
    RESUME_ARGS+=(--resume-s3-uri "$RESUME_S3_URI")
fi
if [ -n "$S3_PREFIX" ] && [ -z "$S3_CHECKPOINT_PREFIX" ]; then
    S3_CHECKPOINT_PREFIX="$S3_PREFIX/checkpoints"
fi
S3_ARGS=()
if [ -n "$S3_CHECKPOINT_PREFIX" ]; then
    S3_ARGS=(--s3-checkpoint-prefix "$S3_CHECKPOINT_PREFIX")
fi
MANIFEST_ARGS=()
if [ -n "$SCENARIO_MANIFEST" ]; then
    MANIFEST_ARGS=(--manifest "$SCENARIO_MANIFEST")
fi

mkdir -p logs checkpoints/simengine_batch

echo "=== run metadata ===" | tee logs/simengine_train.log
git rev-parse HEAD 2>/dev/null | tee -a logs/simengine_train.log || true
if [ -n "$SCENARIO_MANIFEST" ]; then
    sha256sum "$SCENARIO_TABLE" "$SCENARIO_MANIFEST" | tee -a logs/simengine_train.log
else
    sha256sum "$SCENARIO_TABLE" | tee -a logs/simengine_train.log
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a logs/simengine_train.log

python3 -m ai.rl.simengine_batch.train \
    --scenario-table "$SCENARIO_TABLE" \
    "${MANIFEST_ARGS[@]}" \
    "${MAX_SCENARIO_ARGS[@]}" \
    --num-envs "$NUM_ENVS" \
    --horizon "$HORIZON" \
    --minibatch-size "$MINIBATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --replay-ratio "$REPLAY_RATIO" \
    --optimizer "$OPTIMIZER" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --checkpoint-dir checkpoints/simengine_batch \
    --checkpoint-interval-steps "$CHECKPOINT_INTERVAL_STEPS" \
    --log-interval-steps "$LOG_INTERVAL_STEPS" \
    "${RESUME_ARGS[@]}" \
    "${S3_ARGS[@]}" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_GROUP" \
    "${WANDB_NAME_ARGS[@]}" \
    $WANDB_FLAG \
    2>&1 | tee -a logs/simengine_train.log
