#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
META="$ROOT/ai/rl/scripts/.runpod_pod.json"
[ -f "$META" ] || { echo "no $META - run ai/rl/scripts/runpod_setup.sh first"; exit 1; }

SCENARIO_TABLE=${SCENARIO_TABLE:-.runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples.csimin}
SCENARIO_MANIFEST=${SCENARIO_MANIFEST:-.runs/csim_generator_sampling/camera_basis_grid_589824/sobol_samples_grid_manifest.json}

IP=$(jq -r '.runtime.ports[] | select(.privatePort==22) | .ip' "$META")
PORT=$(jq -r '.runtime.ports[] | select(.privatePort==22) | .publicPort' "$META")
SSH_OPTS="-p $PORT -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null"
SSH="ssh $SSH_OPTS"

$SSH root@$IP 'command -v rsync >/dev/null || (export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; apt-get install -y -qq rsync)'

echo ">> Uploading repo to /workspace/drone-interception"
rsync -az --delete \
    -e "ssh $SSH_OPTS" \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='.runs/' \
    --exclude='.cache/' \
    --exclude='checkpoints/' \
    --exclude='detection/' \
    --exclude='logs/' \
    --exclude='papers/' \
    --exclude='renders/' \
    --exclude='wandb/' \
    --exclude='.wandb_key' \
    "$ROOT/" "root@$IP:/workspace/drone-interception/"

echo ">> Uploading scenario table"
$SSH root@$IP 'mkdir -p /workspace/drone-interception/data/scenarios'
rsync -az -e "ssh $SSH_OPTS" "$ROOT/$SCENARIO_TABLE" \
    "root@$IP:/workspace/drone-interception/data/scenarios/sobol_samples.csimin"
rsync -az -e "ssh $SSH_OPTS" "$ROOT/$SCENARIO_MANIFEST" \
    "root@$IP:/workspace/drone-interception/data/scenarios/sobol_samples_grid_manifest.json"

if [ -f "$ROOT/.wandb_key" ]; then
    scp -q -P "$PORT" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null \
        "$ROOT/.wandb_key" root@$IP:/root/.wandb_key
    $SSH root@$IP 'chmod 600 /root/.wandb_key'
elif [ -n "${WANDB_API_KEY:-}" ]; then
    printf '%s' "$WANDB_API_KEY" | $SSH root@$IP 'cat > /root/.wandb_key && chmod 600 /root/.wandb_key'
fi

UPLOAD_AWS_CREDENTIALS=${UPLOAD_AWS_CREDENTIALS:-0}
AWS_CREDENTIALS_DIR=${AWS_CREDENTIALS_DIR:-$HOME/.aws}
if [ "$UPLOAD_AWS_CREDENTIALS" = "1" ] && [ -d "$AWS_CREDENTIALS_DIR" ]; then
    echo ">> Uploading AWS credentials"
    $SSH root@$IP 'mkdir -p /root/.aws && chmod 700 /root/.aws'
    rsync -az -e "ssh $SSH_OPTS" "$AWS_CREDENTIALS_DIR/" "root@$IP:/root/.aws/"
    $SSH root@$IP 'chmod -R go-rwx /root/.aws'
fi

echo ">> Done. SSH: ssh -p $PORT root@$IP"
