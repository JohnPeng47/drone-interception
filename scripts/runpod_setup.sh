#!/usr/bin/env bash
# Provision an RTX 4090 community-cloud pod on Runpod via GraphQL.
# Writes pod metadata to scripts/.runpod_pod.json so other scripts can read it.
#
# Env vars:
#   RUNPOD_API_KEY       (required) your rpa_... token
#   POD_NAME             default: gavin-puffer
#   POD_IMAGE            default: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   POD_DISK_GB          default: 25
#   POD_MIN_VCPU         default: 16
#   POD_MIN_MEM_GB       default: 24
set -euo pipefail

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"

POD_NAME=${POD_NAME:-gavin-puffer}
POD_IMAGE=${POD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}
POD_DISK_GB=${POD_DISK_GB:-25}
POD_MIN_VCPU=${POD_MIN_VCPU:-8}
POD_MIN_MEM_GB=${POD_MIN_MEM_GB:-16}
# RTX 4090 community pool is often resource-starved → falls back to 3090 if unavailable.
# Override with GPU_TYPE_ID=... to force a specific card.
GPU_TYPE_ID=${GPU_TYPE_ID:-NVIDIA GeForce RTX 4090}
GPU_FALLBACK=${GPU_FALLBACK:-NVIDIA GeForce RTX 3090}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
META="$ROOT/scripts/.runpod_pod.json"

deploy_attempt() {
    local gpu="$1"
    local query
    read -r -d '' query <<EOF || true
mutation {
  podFindAndDeployOnDemand(input: {
    cloudType: COMMUNITY,
    gpuCount: 1,
    gpuTypeId: "$gpu",
    minVcpuCount: $POD_MIN_VCPU,
    minMemoryInGb: $POD_MIN_MEM_GB,
    containerDiskInGb: $POD_DISK_GB,
    volumeInGb: 0,
    name: "$POD_NAME",
    imageName: "$POD_IMAGE",
    ports: "22/tcp",
    startSsh: true
  }) {
    id
    machineId
    costPerHr
    desiredStatus
  }
}
EOF
    curl -sS -X POST 'https://api.runpod.io/graphql' \
        -H 'Content-Type: application/json' \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "$(jq -nc --arg q "$query" '{query: $q}')"
}

# Try preferred GPU; if pool is starved, fall back automatically.
echo ">> Provisioning $GPU_TYPE_ID community pod..."
RESP=$(deploy_attempt "$GPU_TYPE_ID")
if echo "$RESP" | jq -e '.errors[]?.message | test("does not have the resources")' >/dev/null 2>&1; then
    echo ">> $GPU_TYPE_ID pool unavailable, falling back to $GPU_FALLBACK..."
    RESP=$(deploy_attempt "$GPU_FALLBACK")
    GPU_TYPE_ID="$GPU_FALLBACK"
fi

if echo "$RESP" | jq -e '.errors' >/dev/null 2>&1; then
    echo "ERROR from Runpod:"
    echo "$RESP" | jq .
    exit 1
fi

POD_ID=$(echo "$RESP" | jq -r '.data.podFindAndDeployOnDemand.id')
COST=$(echo "$RESP" | jq -r '.data.podFindAndDeployOnDemand.costPerHr')
echo ">> Pod $POD_ID created at \$$COST/hr. Waiting for SSH..."

# Poll until runtime.ports has a public mapping for privatePort=22.
for i in $(seq 1 60); do
    STATUS=$(curl -sS -X POST 'https://api.runpod.io/graphql' \
        -H 'Content-Type: application/json' \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "$(jq -nc --arg id "$POD_ID" '{query: "query { pod(input: {podId: \"\($id)\"}) { id desiredStatus runtime { ports { privatePort publicPort ip isIpPublic type } } } }"}')")

    READY=$(echo "$STATUS" | jq -r '.data.pod.runtime.ports // [] | map(select(.privatePort == 22)) | length')
    if [ "$READY" = "1" ]; then
        echo "$STATUS" | jq '.data.pod' > "$META"
        IP=$(jq -r '.runtime.ports[] | select(.privatePort==22) | .ip' "$META")
        PORT=$(jq -r '.runtime.ports[] | select(.privatePort==22) | .publicPort' "$META")
        echo ">> Ready. SSH: ssh -p $PORT root@$IP"
        echo ">> Pod info: $META"
        echo ">> Tear down: scripts/runpod_teardown.sh"
        exit 0
    fi
    sleep 5
    echo "   ... still waiting (${i}/60)"
done

echo "ERROR: pod did not become SSH-ready within 5 min."
echo "$STATUS" | jq .
exit 1
