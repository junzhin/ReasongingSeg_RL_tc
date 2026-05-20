# ReasongingSeg RL Bundle

This directory is a trimmed, portable training bundle derived from `PAPO`.

## Included

- RL training code from `PAPO`, excluding `.git`, `PAPO-Eval`, `wandb`, `logs`, and `checkpoints`
- Training JSONL: `data/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_papo.jsonl`
- Evaluation JSONLs kept under `data/eval/`
- All images referenced by the copied training and evaluation JSONLs, rewritten to repo-relative paths under `data/images/`
- Pretrained model weights: `models/Lingshu-7B`

## Current Layout

- Code root: this directory
- Train file: `data/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_papo.jsonl`
- Eval file used by the portable script: `data/eval/benchmark_medreasoner_evidence_eval_400_balanced_papo.jsonl`
- Image root: `data/images/`
- Default model path in the portable training script: `models/Lingshu-7B`
- Portable training script: `examples/ours_grpo/qwen2_5_vl_7b_cepo_lite_grpo_portable.sh`
- Slurm template: `scripts/submit_cepo_lite_7b_reserved.sbatch`

## What To Change On Another Cluster

1. Update the conda setup at the top of `examples/ours_grpo/qwen2_5_vl_7b_cepo_lite_grpo_portable.sh`.
   It currently uses `/mnt/petrelfs/tangcheng/miniconda3/etc/profile.d/conda.sh` and `conda activate papo`.
2. Make sure the environment has the PAPO dependencies installed.
3. If you want to start from another checkpoint or model, override `MODEL_PATH`.
4. If your Slurm partition or quota names differ, edit `scripts/submit_cepo_lite_7b_reserved.sbatch`.
5. If you put the repo somewhere else, no JSONL image path change is needed as long as you run from the repo root; the copied JSONLs already use repo-relative image paths like `data/images/...`.

## Typical Training Command

```bash
cd /mnt/petrelfs/tangcheng/tangcheng/ReasongingSeg/ReasongingSeg_RL
bash examples/ours_grpo/qwen2_5_vl_7b_cepo_lite_grpo_portable.sh
```

## Typical Slurm Submission

```bash
cd /mnt/petrelfs/tangcheng/tangcheng/ReasongingSeg/ReasongingSeg_RL
sbatch scripts/submit_cepo_lite_7b_reserved.sbatch
```

## Notes

- The portable script auto-resumes from the latest checkpoint under `checkpoints/${WANDB_PROJECT}/${EXP_NAME}`.
- The bundle does not include `PAPO-Eval`.
- The bundle does not include previous training checkpoints.
- The bundle does not include `final_sft_lingshu`; only `models/Lingshu-7B` is copied.

## Bundle Summary

- Training JSONLs copied: 26731
- Eval JSONLs copied: 800
- Unique images copied: 26204
- Missing images detected during build: 0
