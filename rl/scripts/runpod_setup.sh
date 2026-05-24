#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"

POD_NAME=${POD_NAME:-drone-simengine-rl}
POD_IMAGE=${POD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}
POD_DISK_GB=${POD_DISK_GB:-80}
POD_MIN_VCPU=${POD_MIN_VCPU:-16}
POD_MIN_MEM_GB=${POD_MIN_MEM_GB:-32}
POD_CLOUD_TYPE=${POD_CLOUD_TYPE:-COMMUNITY}
GPU_TYPE_ID=${GPU_TYPE_ID:-NVIDIA GeForce RTX 4090}

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
META="$ROOT/rl/scripts/.runpod_pod.json"

read -r -d '' QUERY <<EOF || true
mutation {
  podFindAndDeployOnDemand(input: {
    cloudType: $POD_CLOUD_TYPE,
    gpuCount: 1,
    gpuTypeId: "$GPU_TYPE_ID",
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

echo ">> Provisioning $GPU_TYPE_ID $POD_CLOUD_TYPE pod..."
RESP=$(curl -sS -X POST 'https://api.runpod.io/graphql' \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d "$(jq -nc --arg q "$QUERY" '{query: $q}')")

if echo "$RESP" | jq -e '.errors' >/dev/null 2>&1; then
    echo "$RESP" | jq .
    exit 1
fi

POD_ID=$(echo "$RESP" | jq -r '.data.podFindAndDeployOnDemand.id')
echo ">> Pod $POD_ID created. Waiting for SSH..."

for i in $(seq 1 90); do
    STATUS=$(curl -sS -X POST 'https://api.runpod.io/graphql' \
        -H 'Content-Type: application/json' \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "$(jq -nc --arg id "$POD_ID" '{query: "query { pod(input: {podId: \"\($id)\"}) { id desiredStatus runtime { ports { privatePort publicPort ip isIpPublic type } } } }"}')")
    READY=$(echo "$STATUS" | jq -r '.data.pod.runtime.ports // [] | map(select(.privatePort == 22)) | length')
    if [ "$READY" = "1" ]; then
        mkdir -p "$(dirname "$META")"
        echo "$STATUS" | jq '.data.pod' > "$META"
        IP=$(jq -r '.runtime.ports[] | select(.privatePort==22) | .ip' "$META")
        PORT=$(jq -r '.runtime.ports[] | select(.privatePort==22) | .publicPort' "$META")
        echo ">> Ready: ssh -p $PORT root@$IP"
        exit 0
    fi
    sleep 5
done

echo "ERROR: pod did not become SSH-ready"
echo "$STATUS" | jq .
exit 1
