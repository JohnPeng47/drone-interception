#!/usr/bin/env bash
# Pull checkpoints/, logs/, wandb/, and raw train stdout back from the pod
# BEFORE teardown. Saved under runs/<pod_id>_<timestamp>/.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
META="$ROOT/rl/scripts/.runpod_pod.json"
[ -f "$META" ] || { echo "no $META - nothing to sync"; exit 1; }

POD_ID=$(jq -r '.id' "$META")
IP=$(jq -r '[.runtime.ports[] | select(.privatePort==22)][0].ip' "$META")
PORT=$(jq -r '[.runtime.ports[] | select(.privatePort==22)][0].publicPort' "$META")

STAMP=$(date +%Y%m%d_%H%M%S)
DST="$ROOT/runs/${POD_ID}_${STAMP}"
mkdir -p "$DST"/{checkpoints,logs,wandb}

SSH_OPTS="-p $PORT -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null"
echo ">> Syncing back to $DST"

# Each rsync surfaces its own status; don't suppress errors silently.
for src in checkpoints logs wandb; do
    rsync -az -e "ssh $SSH_OPTS" \
        "root@$IP:/workspace/gavin_puffer/$src/" "$DST/$src/" \
        && echo "   ok: $src" \
        || echo "   FAIL: $src (rsync exit $?)"
done

# Train stdout log from /tmp
rsync -az -e "ssh $SSH_OPTS" \
    "root@$IP:/tmp/intercept_train.log" "$DST/intercept_train.log" \
    && echo "   ok: intercept_train.log" \
    || echo "   (no intercept_train.log)"

# Drop the empty dirs we pre-created so the listing is honest.
find "$DST" -type d -empty -delete

echo ">> Synced. Size: $(du -sh "$DST" | cut -f1)"
find "$DST" -type f -printf "   %p\n" 2>/dev/null | head -10
