#!/usr/bin/env bash
# Driver for the RunPod A/B test of the ernie45_vl_moe AutoWeightsLoader change.
# Keeps the pod alive at the end so logs stay readable.
set -x

BASE_SHA=ed0f475          # commit immediately before the loader change
FILE=vllm/model_executor/models/ernie45_vl_moe.py
export HF_HOME=/workspace/hf
export VLLM_LOGGING_LEVEL=INFO
# Build from source against the host's CUDA 12.8: the published precompiled
# wheels link libcudart.so.13, which a 12.8 driver cannot load.
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="9.0"   # H100/H200 are sm90; one arch keeps the build short
export CCACHE_DIR=/workspace/ccache

mark() { echo "@@@@ $* @@@@"; }

mark STAGE_SETUP_START
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
nvidia-smi | head -5 || true
df -h /workspace || true

apt-get update -y && apt-get install -y git curl build-essential ccache
export MAX_JOBS=$(nproc)
echo "MAX_JOBS=$MAX_JOBS  CUDA_HOME=$CUDA_HOME"
nvcc --version || true
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace
git clone --depth 50 -b claude/runpod-ab-test https://github.com/jinminghe950/vllm.git repo
cd /workspace/repo
git log --oneline -3

mark STAGE_INSTALL_START
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/build/cuda.txt --torch-backend=auto 2>&1 | tail -15
python -c "import torch; print('TORCH', torch.__version__, 'CUDA', torch.version.cuda)"
mark STAGE_COMPILE_START
uv pip install -e . --no-build-isolation --torch-backend=auto 2>&1 | tail -40
uv pip install pillow 2>&1 | tail -3
mark STAGE_INSTALL_DONE

# Preflight: abort before the ~56GB model download if the runtime is broken.
if python -c "import vllm, torch; print('VLLM', vllm.__version__, 'TORCH', torch.__version__, 'TORCH_CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"; then
  mark PREFLIGHT_OK
else
  mark VERDICT_INSTALL_FAILED
  sleep infinity
fi

mark STAGE_NEW_START
python runpod_ab_test.py /workspace/new.json 2>&1 | tail -60
mark STAGE_NEW_DONE

mark STAGE_OLD_START
git checkout $BASE_SHA -- $FILE
git diff --stat HEAD -- $FILE
python runpod_ab_test.py /workspace/old.json 2>&1 | tail -60
mark STAGE_OLD_DONE

mark STAGE_COMPARE
if [ -f /workspace/new.json ] && [ -f /workspace/old.json ]; then
  if diff -u /workspace/old.json /workspace/new.json; then
    mark VERDICT_IDENTICAL_OUTPUTS
  else
    mark VERDICT_OUTPUTS_DIFFER
  fi
else
  mark VERDICT_INCOMPLETE_ONE_OR_BOTH_RUNS_FAILED
fi

mark ALL_DONE
sleep infinity
