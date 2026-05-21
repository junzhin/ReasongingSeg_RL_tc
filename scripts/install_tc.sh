#!/bin/bash
# Install script for papo_tc conda environment
# - No GPU on current node; PyTorch installed as cu124 wheel (no compile)
# - flash-attn skipped here; install manually on GPU node later
# - No setup.py in repo; use PYTHONPATH instead of pip install -e .

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="papo_tc"
PYTHON_VERSION="3.10"

echo "=== papo_tc Install Script ==="
echo "Repo root: ${REPO_ROOT}"
echo "Target env: ${ENV_NAME}"

# Step 1: Create conda env
echo ""
echo "Step 1/5: Creating conda environment '${ENV_NAME}' (python=${PYTHON_VERSION})..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Env '${ENV_NAME}' already exists. Skipping creation."
else
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    echo "  Created."
fi

# Detect conda base and activate
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "  Active env: ${CONDA_DEFAULT_ENV}"

# Step 2: Base build tools
echo ""
echo "Step 2/5: Installing build tools..."
pip install --upgrade pip setuptools wheel ninja packaging

# Step 3: PyTorch cu124 (MUST be cu124, not CPU-only)
echo ""
echo "Step 3/5: Installing PyTorch 2.6.0 + CUDA 12.4..."
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

TORCH_VER=$(python -c "import torch; print(torch.__version__)")
echo "  Installed: ${TORCH_VER}"
if [[ "${TORCH_VER}" != *"cu124"* ]]; then
    echo "WARNING: PyTorch version '${TORCH_VER}' does not contain 'cu124'."
    echo "         GPU training may not work. Check your install URL."
fi

# Step 4: Core dependencies (no flash-attn)
echo ""
echo "Step 4/5: Installing core dependencies..."

# Set flash-attn disabled globally for this env
export FLASH_ATTENTION_FORCE_DISABLE=1

pip install transformers accelerate
pip install "ray[default]"
pip install omegaconf hydra-core
pip install jinja2 pydantic

# vLLM (may take a while)
echo "  Installing vLLM 0.8.4..."
if ! pip install vllm==0.8.4 --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6/ --quiet; then
    echo "  vLLM from flashinfer failed, trying PyPI..."
    pip install vllm==0.8.4
fi

# Logging and misc
pip install wandb tensorboard
pip install datasets huggingface_hub

# Optional: liger-kernel
if pip install liger-kernel --quiet 2>/dev/null; then
    echo "  liger-kernel installed."
else
    echo "  liger-kernel skipped (not critical)."
fi

# Step 5: PYTHONPATH setup (replaces pip install -e . since no setup.py)
echo ""
echo "Step 5/5: Setting up PYTHONPATH..."
ACTIVATE_SCRIPT="${CONDA_BASE}/envs/${ENV_NAME}/etc/conda/activate.d/papo_tc_env.sh"
DEACTIVATE_SCRIPT="${CONDA_BASE}/envs/${ENV_NAME}/etc/conda/deactivate.d/papo_tc_env.sh"

mkdir -p "$(dirname ${ACTIVATE_SCRIPT})"
mkdir -p "$(dirname ${DEACTIVATE_SCRIPT})"

cat > "${ACTIVATE_SCRIPT}" << EOF
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH}"
export FLASH_ATTENTION_FORCE_DISABLE=1
EOF

cat > "${DEACTIVATE_SCRIPT}" << EOF
export PYTHONPATH="\$(echo \$PYTHONPATH | tr ':' '\n' | grep -v "^${REPO_ROOT}$" | tr '\n' ':' | sed 's/:$//')"
unset FLASH_ATTENTION_FORCE_DISABLE
EOF

echo "  PYTHONPATH will be set to include repo root on conda activate."

# Verification
echo ""
echo "=== Verification ==="
python - << 'PYEOF'
import sys
print(f"Python: {sys.version}")

ok = True

try:
    import torch
    print(f"  torch: {torch.__version__}")
    if "cu" not in torch.__version__:
        print("  WARNING: torch is CPU-only!")
        ok = False
except ImportError as e:
    print(f"  torch FAILED: {e}"); ok = False

try:
    import transformers
    print(f"  transformers: {transformers.__version__}")
except ImportError as e:
    print(f"  transformers FAILED: {e}"); ok = False

try:
    import vllm
    print(f"  vllm: {vllm.__version__}")
except ImportError as e:
    print(f"  vllm FAILED: {e}"); ok = False

try:
    import ray
    print(f"  ray: {ray.__version__}")
except ImportError as e:
    print(f"  ray FAILED: {e}"); ok = False

try:
    import omegaconf
    print(f"  omegaconf: {omegaconf.__version__}")
except ImportError as e:
    print(f"  omegaconf FAILED: {e}"); ok = False

# verl import (needs PYTHONPATH set)
try:
    import verl
    print(f"  verl: import OK")
except ImportError as e:
    print(f"  verl: FAILED (check PYTHONPATH) - {e}")
    print(f"    Hint: re-activate conda env or run:")
    print(f"    export PYTHONPATH={sys.path[0]}:$PYTHONPATH")

if ok:
    print("\nCore deps OK. Re-activate env then run verl import check.")
else:
    print("\nSome packages failed. Review above.")
PYEOF

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. conda activate ${ENV_NAME}"
echo "  2. cd ${REPO_ROOT}"
echo "  3. python -c \"from verl.trainer import main; print('verl OK')\""
echo ""
echo "Flash-attn (GPU node only, run after conda activate ${ENV_NAME}):"
echo "  export CUDA_HOME=/usr/local/cuda"
echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
echo "  pip install 'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.7/flash_attn-2.7.4.post1+cu124torch2.6-cp310-cp310-linux_x86_64.whl'"
