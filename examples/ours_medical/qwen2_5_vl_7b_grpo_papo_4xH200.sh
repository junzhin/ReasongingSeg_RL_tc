#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Medical Evidence RL Training — GRPO + PAPO
# Hardware: 4 × H200 (141 GB HBM3e each)
# Model:    Lingshu-7B (Qwen2.5-VL-7B medical fine-tune)
# Data:     train_cepo_lite_80_10_10_grpo_papo.jsonl
# Reward:   medical_evidence.py — bbox IoU matching + format check
# ═══════════════════════════════════════════════════════════════════════════

# set -euxo pipefail

source /inspire/hdd/global_user/hejunjun-24017/junzhin/.bashrc
conda env list

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "Repo root: ${REPO_ROOT}"
cd "${REPO_ROOT}"

ENV_NAME="papo_tc"

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

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export RAY_TMPDIR=/tmp/ray_grpo_papo_tc

CUDA_IDS=0,1,2,3
N_GPU=4

MODEL_PATH=/inspire/hdd/project/qproject-multimedicine/public/share_models/Lingshu-7B

CONFIG_FILE="examples/config.yaml"
TRAIN_FILE="data/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_grpo_papo.jsonl"
VAL_FILE="data/eval/benchmark_medreasoner_evidence_eval_400_balanced_papo.jsonl"
FORMAT_PROMPT="examples/format_prompt/medical_evidence.jinja"
REWARD_FUNCTION="examples/reward_function/medical_evidence.py:compute_score"

for f in "${CONFIG_FILE}" "${TRAIN_FILE}" "${VAL_FILE}" "${FORMAT_PROMPT}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: Missing file: ${REPO_ROOT}/${f}"
        exit 1
    fi
done

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model dir not found: ${MODEL_PATH}"
    exit 1
fi

echo "Train data: $(wc -l < "${TRAIN_FILE}") samples"
echo "Val data:   $(wc -l < "${VAL_FILE}") samples"

TOTAL_EPOCHES=10
SAVE_FREQ=2
SAVE_LIMIT=6
VAL_FREQ=2

GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
MINI_ROLLOUT_BATCH_SIZE=128
MAX_PROMPT_LENGTH=4096
MAX_TRY_MAKE_BATCH=50

KL_PRCP_COEF=0.01
USE_AUG_ENTROPY_LOSS=true
AUG_ENTROPY_LOSS_COEF=0.03
USE_ORI_ENTROPY_LOSS=true
ORI_ENTROPY_LOSS_COEF=0.03

EXP_NAME="medical_evidence__grpo_papo__7b__ep${TOTAL_EPOCHES}_rb${ROLLOUT_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}"

LOAD_CHECKPOINT_PATH=null

CKPT_BASE="checkpoints/easy_r1/${EXP_NAME}"
mkdir -p "${CKPT_BASE}"
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${CKPT_BASE}/training_${RUN_TIMESTAMP}.log"
echo "Log file: ${LOG_FILE}"

echo ""
echo "=========================================="
echo "  Experiment: ${EXP_NAME}"
echo "  Algorithm:  GRPO + PAPO"
echo "  GPUs:       ${N_GPU} × H200"
echo "  Epochs:     ${TOTAL_EPOCHES}"
echo "  Save every: ${SAVE_FREQ} steps"
echo "  Log:        ${LOG_FILE}"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONFIG_FILE} \
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
    algorithm.adv_estimator=grpo \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    algorithm.online_filtering=false \
    algorithm.use_kl_prcp=true \
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
    trainer.max_try_make_batch=${MAX_TRY_MAKE_BATCH} \
    trainer.load_checkpoint_path=${LOAD_CHECKPOINT_PATH} \
    trainer.save_best_checkpoint=true \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    2>&1 | tee "${LOG_FILE}"
