#!/usr/bin/env bash
# Rsync the gavin_puffer dir up to the running pod, excluding build artifacts.
# Reads rl/scripts/.runpod_pod.json for SSH endpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
META="$ROOT/rl/scripts/.runpod_pod.json"
[ -f "$META" ] || { echo "no $META - run rl/scripts/runpod_setup.sh first"; exit 1; }

IP=$(jq -r '.runtime.ports[] | select(.privatePort==22) | .ip' "$META")
PORT=$(jq -r '.runtime.ports[] | select(.privatePort==22) | .publicPort' "$META")

SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null"

# Wait up to 90s for sshd to bind after pod start.
for i in $(seq 1 18); do
    $SSH -o ConnectTimeout=5 -o BatchMode=yes root@$IP 'echo ready' >/dev/null 2>&1 && break
    sleep 5
done

# rsync isn't in the runpod/pytorch image — install it before transfer.
# Need apt-get update first (fresh container has empty apt cache).
$SSH root@$IP 'command -v rsync >/dev/null || \
    (export DEBIAN_FRONTEND=noninteractive; \
     apt-get update -qq 2>&1 | tail -2; \
     apt-get install -y -qq rsync 2>&1 | tail -2)'

# Push wandb key to pod if present locally.
if [ -f "$ROOT/.wandb_key" ]; then
    scp -q -P $PORT -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null \
        "$ROOT/.wandb_key" root@$IP:/root/.wandb_key
    $SSH root@$IP 'chmod 600 /root/.wandb_key'
    echo ">> wandb key staged on pod"
fi

echo ">> Uploading to root@$IP:$PORT:/workspace/gavin_puffer"
rsync -az --delete \
    -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null" \
    --exclude='build/' \
    --exclude='raylib-5.5_*/' \
    --exclude='__pycache__/' \
    --exclude='*.so' \
    --exclude='*.o' \
    --exclude='.runs/' \
    --exclude='/experiments/' \
    --exclude='checkpoints/' \
    --exclude='logs/' \
    --exclude='wandb/' \
    --exclude='rl/scripts/.runpod_pod.json' \
    --exclude='.wandb_key' \
    "$ROOT/" "root@$IP:/workspace/gavin_puffer/"

echo ">> Done. SSH in with:"
echo "   ssh -p $PORT root@$IP"
