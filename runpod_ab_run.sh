#!/usr/bin/env bash
# Driver for the RunPod A/B test of the ernie45_vl_moe AutoWeightsLoader change.
# Keeps the pod alive at the end so logs stay readable.
set -x

BASE_SHA=ed0f475          # commit immediately before the loader change
FILE=vllm/model_executor/models/ernie45_vl_moe.py
export HF_HOME=/workspace/hf
export VLLM_USE_PRECOMPILED=1
export VLLM_LOGGING_LEVEL=INFO

mark() { echo "@@@@ $* @@@@"; }

mark STAGE_SETUP_START
nvidia-smi || true
df -h /workspace || true

apt-get update -y && apt-get install -y git curl build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace
git clone --depth 50 -b claude/runpod-ab-test https://github.com/jinminghe950/vllm.git repo
cd /workspace/repo
git log --oneline -3

mark STAGE_INSTALL_START
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . --torch-backend=auto 2>&1 | tail -30
uv pip install pillow 2>&1 | tail -3
python -c "import vllm, torch; print('VLLM', vllm.__version__, 'TORCH', torch.__version__, torch.cuda.get_device_name(0))"
mark STAGE_INSTALL_DONE

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
