#!/usr/bin/env bash
# Driver for the RunPod A/B test of the ernie45_vl_moe AutoWeightsLoader change.
#
# The published vLLM wheels link libcudart.so.13, but RunPod's H200 hosts hand
# out a CUDA 12.8 driver. H200 is a datacenter GPU, so NVIDIA's forward
# compatibility package lets the CUDA 13 binaries run on the older driver.
# Keeps the pod alive at the end so logs stay readable.
set -x

BASE_SHA=ed0f475          # commit immediately before the loader change
FILE=vllm/model_executor/models/ernie45_vl_moe.py
export HF_HOME=/workspace/hf
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_PRECOMPILED=1

mark() { echo "@@@@ $* @@@@"; }

mark STAGE_SETUP_START
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
df -h /workspace || true

apt-get update -y && apt-get install -y git curl wget

mark STAGE_CUDA_COMPAT
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update -y
apt-get install -y cuda-compat-13-0
ls -d /usr/local/cuda-13.0/compat || true
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace
rm -rf /workspace/repo
git clone --depth 50 -b claude/runpod-ab-test https://github.com/jinminghe950/vllm.git repo
cd /workspace/repo
git log --oneline -3

mark STAGE_INSTALL_START
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . --torch-backend=cu130 2>&1 | tail -20
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
