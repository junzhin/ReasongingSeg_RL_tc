#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Medical Evidence RL Training — DAPO + PAPO
# Hardware: 4 × H200 (141 GB HBM3e each)
# Model:    Lingshu-7B (Qwen2.5-VL-7B medical fine-tune)
# Data:     train_cepo_lite_80_10_10_papo.jsonl (26731 samples)
#           80% original + 10% background_perturbed + 10% evidence_deleted
# Reward:   medical_evidence.py — bbox IoU matching + format check
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage (from anywhere):
#   bash /path/to/examples/ours_medical/qwen2_5_vl_7b_dapo_papo_4xH200.sh
#
# Checkpoints saved to: checkpoints/easy_r1/${EXP_NAME}/
# ═══════════════════════════════════════════════════════════════════════════

set -euxo pipefail

# ── 1. Locate repo root (auto-detect from script location) ─────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "Repo root: ${REPO_ROOT}"
cd "${REPO_ROOT}"

# ── 2. Activate conda environment ──────────────────────────────────────
ENV_NAME="papo_tc"

# Try common conda locations
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found. Install conda or update this script."
    exit 1
fi

conda activate "${ENV_NAME}"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo "Python: $(python --version)"

# ── 3. Set PYTHONPATH (verl needs repo root importable) ────────────────
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ── 4. Environment variables ──────────────────────────────────────────
export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# ── 5. GPU Configuration ─────────────────────────────────────────────
CUDA_IDS=0,1,2,3
N_GPU=4

# ── 6. Paths ──────────────────────────────────────────────────────────
MODEL_PATH=/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/20260519_medsegreasoner_tc/project/qproject-multimedicine/public/share_models/Lingshu-7B

CONGI_FILE="examples/config.yaml"
TRAIN_FILE="data/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_papo.jsonl"
VAL_FILE="data/eval/benchmark_medreasoner_evidence_eval_400_balanced_papo.jsonl"
FORMAT_PROMPT="examples/format_prompt/math_perception.jinja"
REWARD_FUNCTION="examples/reward_function/medical_evidence.py:compute_score"

# ── 7. Sanity checks ─────────────────────────────────────────────────
for f in "${CONGI_FILE}" "${TRAIN_FILE}" "${VAL_FILE}" "${FORMAT_PROMPT}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: Missing file: ${REPO_ROOT}/${f}"
        exit 1
    fi
done

if [ ! -d "$(dirname "${MODEL_PATH}")" ]; then
    echo "WARNING: Model parent dir not found: $(dirname "${MODEL_PATH}")"
    echo "         Update MODEL_PATH if model is elsewhere."
fi

echo "Train data: $(wc -l < "${TRAIN_FILE}") samples"
echo "Val data:   $(wc -l < "${VAL_FILE}") samples"

# ── 8. Training Hyperparameters ───────────────────────────────────────
TOTAL_EPOCHES=10
SAVE_FREQ=2
SAVE_LIMIT=6
VAL_FREQ=2

GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
MINI_ROLLOUT_BATCH_SIZE=128
MAX_PROMPT_LENGTH=4096

# ── 9. PAPO Parameters ───────────────────────────────────────────────
KL_PRCP_COEF=0.01

USE_AUG_ENTROPY_LOSS=true
AUG_ENTROPY_LOSS_COEF=0.03
USE_ORI_ENTROPY_LOSS=true
ORI_ENTROPY_LOSS_COEF=0.03

# ── 10. Experiment Name ──────────────────────────────────────────────
EXP_NAME="medical_evidence__dapo_papo__7b__ep${TOTAL_EPOCHES}_rb${ROLLOUT_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}"

echo ""
echo "=========================================="
echo "  Experiment: ${EXP_NAME}"
echo "  Algorithm:  DAPO + PAPO"
echo "  GPUs:       ${N_GPU} × H200"
echo "  Epochs:     ${TOTAL_EPOCHES}"
echo "  Save every: ${SAVE_FREQ} steps"
echo "=========================================="
echo ""

# ── 11. Launch Training ──────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.mini_rollout_batch_size=${MINI_ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.adv_estimator=dapo \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    algorithm.online_filtering=true \
    algorithm.filter_key=overall \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    algorithm.kl_prcp_coef=${KL_PRCP_COEF} \
    algorithm.use_aug_entropy_loss=${USE_AUG_ENTROPY_LOSS} \
    algorithm.aug_entropy_loss_coef=${AUG_ENTROPY_LOSS_COEF} \
    algorithm.use_ori_entropy_loss=${USE_ORI_ENTROPY_LOSS} \
    algorithm.ori_entropy_loss_coef=${ORI_ENTROPY_LOSS_COEF} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.save_limit=${SAVE_LIMIT} \
    trainer.val_freq=${VAL_FREQ} \
    trainer.save_best_checkpoint=true \
    worker.reward.reward_function=${REWARD_FUNCTION}
