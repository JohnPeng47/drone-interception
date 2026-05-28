#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
META="$ROOT/ai/rl/scripts/.runpod_pod.json"
[ -f "$META" ] || { echo "no $META"; exit 1; }

POD_ID=$(jq -r '.id' "$META")
IP=$(jq -r '.runtime.ports[] | select(.privatePort==22) | .ip' "$META")
PORT=$(jq -r '.runtime.ports[] | select(.privatePort==22) | .publicPort' "$META")
STAMP=$(date +%Y%m%d_%H%M%S)
DST="$ROOT/runs/${POD_ID}_${STAMP}"
mkdir -p "$DST"
SSH_OPTS="-p $PORT -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null"

for src in logs checkpoints wandb; do
    rsync -az -e "ssh $SSH_OPTS" "root@$IP:/workspace/drone-interception/$src/" "$DST/$src/" \
        && echo "ok: $src" || echo "missing/fail: $src"
done
find "$DST" -type d -empty -delete
du -sh "$DST"
