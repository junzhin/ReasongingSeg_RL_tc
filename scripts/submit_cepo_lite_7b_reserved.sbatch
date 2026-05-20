#!/bin/bash
#SBATCH -p Medisco
#SBATCH --gres=gpu:4
#SBATCH -c 8
#SBATCH --quotatype=reserved
#SBATCH -J rg_cepo_80_10_10
#SBATCH -o logs/rg_cepo_80_10_10_%j.out

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

export N_GPU=4
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty; Slurm did not expose allocated GPUs."
    exit 1
fi
export CUDA_IDS="${CUDA_VISIBLE_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export EXP_NAME="${EXP_NAME:-rg_cepo_lite_80_10_10_lingshu_grpo_lr1e-6_kl1e-2}"
export TOTAL_EPOCHES="${TOTAL_EPOCHES:-3}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
export ROLLOUT_N="${ROLLOUT_N:-5}"
export VAL_N="${VAL_N:-8}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export VAL_FREQ="${VAL_FREQ:-5}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Using CUDA_IDS=${CUDA_IDS}"
nvidia-smi || true

unset RAY_ADDRESS
bash examples/ours_grpo/qwen2_5_vl_7b_cepo_lite_grpo_portable.sh
