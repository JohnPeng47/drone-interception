#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
META="$ROOT/ai/rl/scripts/.runpod_pod.json"
[ -f "$META" ] || { echo "no $META"; exit 0; }
POD_ID=$(jq -r '.id' "$META")

curl -sS -X POST 'https://api.runpod.io/graphql' \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d "$(jq -nc --arg id "$POD_ID" '{query: "mutation { podTerminate(input: {podId: \"\($id)\"}) }"}')" \
    | jq .
mv "$META" "$META.terminated.$(date +%s)"
