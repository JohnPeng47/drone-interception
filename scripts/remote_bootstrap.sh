#!/usr/bin/env bash
# Runs INSIDE the Runpod pod after upload.sh. Installs build deps, builds the
# CUDA bindings, and runs a short smoke train to measure SPS.
#
# Lessons from the first attempt baked in:
#   - DEBIAN_FRONTEND=noninteractive (apt prompts hang the install otherwise)
#   - PATH must include /usr/local/cuda/bin (nvcc not in default PATH)
#   - The runpod/pytorch image has CUDA but no cuDNN/NCCL system libs;
#     the build's autodetect falls back to nvidia-cudnn-cu12 / nvidia-nccl-cu12
#     pip wheels — install them BEFORE building.
#   - Skip gpytorch/scikit-learn/wandb at install time (heavy, only needed for
#     sweep mode); install them lazily if you actually `puffer sweep`.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH=/root/.local/bin:/usr/local/cuda/bin:$PATH

cd /workspace/gavin_puffer

echo "=== nvidia-smi ==="
nvidia-smi -L
echo
echo "=== nvcc ==="
nvcc --version | tail -2

echo
echo "=== installing system build deps (apt) ==="
apt-get update -qq
apt-get install -y -qq rsync clang libomp-dev jq ccache 2>&1 | tail -3

echo
echo "=== installing python build deps (uv, minimal set) ==="
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Don't pull gpytorch/scikit-learn — they pin SciPy/torch and stall for minutes.
# build.sh's cudnn/nccl autodetect falls back to these wheels.
uv pip install --system --quiet \
    pybind11 rich rich_argparse wandb \
    nvidia-cudnn-cu12 nvidia-nccl-cu12

echo
echo "=== fixing cuDNN/nvidia-ml unversioned symlinks (wheels ship only .so.9) ==="
CUDNN_LIB=$(python3 -c "import nvidia.cudnn, os; print(os.path.join(nvidia.cudnn.__path__[0], 'lib'))")
(cd "$CUDNN_LIB" && for f in libcudnn*.so.9; do
    [ -e "$f" ] && ln -sf "$f" "${f%.so.*}.so"
done)
ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so 2>/dev/null || true

echo
echo "=== wandb login (if key present) ==="
if [ -f /root/.wandb_key ]; then
    wandb login --relogin "$(cat /root/.wandb_key)" 2>&1 | tail -2
    WANDB_FLAG="--wandb"
else
    echo "  (no /root/.wandb_key — running without wandb)"
    WANDB_FLAG=""
fi

echo
echo "=== building CUDA backend (./build.sh intercept --float) ==="
./build.sh intercept --float 2>&1 | tail -15

echo
echo "=== verifying _C.so loads ==="
python3 -c "from pufferlib import _C; print('env=', _C.env_name, 'gpu=', _C.gpu, 'precision_bytes=', _C.precision_bytes)"

echo
echo "=== smoke train (180 sec wall, 200M timestep cap) ==="
timeout 180 python3 -m pufferlib.pufferl train intercept \
    --train.total-timesteps 200000000 \
    --wandb-project gavin_puffer --wandb-group smoke \
    $WANDB_FLAG \
    2>&1 | tee /tmp/intercept_train.log || true

echo
echo "=== extracting metrics ==="
sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' /tmp/intercept_train.log \
    | grep -E "^\│  (Steps|SPS|Uptime|episode_return|episode_length|ema_dist|entropy)" \
    | tail -40
