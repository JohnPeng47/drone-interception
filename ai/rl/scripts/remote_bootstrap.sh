#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH=/root/.local/bin:$PATH

cd /workspace/drone-interception

echo "=== nvidia-smi ==="
nvidia-smi

echo "=== system deps ==="
apt-get update -qq
apt-get install -y -qq jq rsync ccache clang libomp-dev 2>&1 | tail -5

echo "=== python deps ==="
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet numpy scipy torch wandb rich boto3

if [ -f /root/.wandb_key ]; then
    wandb login --relogin "$(cat /root/.wandb_key)"
fi

echo "=== smoke import ==="
python3 - <<'PY'
from ai.rl.simengine_env import ScenarioTable
p = "/workspace/drone-interception/data/scenarios/sobol_samples.csimin"
m = "/workspace/drone-interception/data/scenarios/sobol_samples_grid_manifest.json"
t = ScenarioTable(p, manifest_path=m, max_scenarios=32)
print("scenarios", t.count)
PY
