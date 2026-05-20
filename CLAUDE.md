# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ReasongingSeg RL Bundle** — portable RL training bundle for multimodal VLMs, derived from PAPO (Perception-Augmented Policy Optimization). Trains Qwen2.5-VL models with GRPO/DAPO + PAPO perceptual augmentation on medical evidence reasoning tasks.

## Training Entry Point

```bash
# Run from repo root
python -m verl.trainer.main config=examples/config.yaml \
    data.train_files=<path> \
    worker.actor.model.model_path=<model> \
    trainer.experiment_name=<name>
```

Config uses OmegaConf dot-notation CLI overrides — all YAML values can be overridden on the command line.

## Common Commands

```bash
# Portable single-node training (4 GPU)
bash examples/ours_grpo/qwen2_5_vl_7b_cepo_lite_grpo_portable.sh

# Slurm submission
sbatch scripts/submit_cepo_lite_7b_reserved.sbatch

# GRPO baseline (no PAPO)
bash examples/papo_grpo/qwen2_5_vl_7b_grpo.sh

# GRPO + PAPO
bash examples/papo_grpo/qwen2_5_vl_7b_grpo_papo.sh

# DAPO + PAPO
bash examples/papo_dapo/qwen2_5_vl_7b_dapo_papo.sh
```

## Architecture

```
verl/
├── trainer/
│   ├── main.py          # Entry point — parses config, builds trainer
│   ├── ray_trainer.py   # Core RayPPOTrainer — orchestrates all workers via Ray
│   ├── core_algos.py    # PPO/GRPO/DAPO advantage estimators, KL controllers
│   ├── config.py        # Dataclass configs (DataConfig, AlgorithmConfig, TrainerConfig)
│   ├── data_loader.py   # Dataset + collate for multimodal JSONL
│   └── papo_utils.py    # random_patch_blackening — PAPO image augmentation
├── workers/
│   ├── actor/           # FSDP actor (policy update)
│   ├── critic/          # FSDP critic (value estimation)
│   ├── rollout/         # vLLM rollout (generation)
│   ├── reward/          # Pluggable reward functions
│   └── sharding_manager/# FSDP↔vLLM weight sync (Ulysses/SPMD)
├── single_controller/
│   └── ray/             # Ray worker group management, resource pools
├── models/
│   └── transformers/    # Qwen2-VL monkey-patches, flash-attn utils
└── utils/               # Tokenizer, checkpointing, seqlen balancing, metrics
```

**Data flow**: `data_loader` → `RayPPOTrainer` → rollout worker generates responses → reward worker scores → actor/critic update with GRPO/DAPO advantages.

**PAPO extensions** (vs vanilla GRPO):
- `algorithm.use_kl_prcp`: KL loss between original and patch-blackened image representations
- `algorithm.use_aug_entropy_loss` / `use_ori_entropy_loss`: entropy regularization on augmented vs original
- `papo_utils.random_patch_blackening`: randomly zeros 14×14 patches (60% prob by default)

## Reward Functions

Pluggable via `worker.reward.reward_function=path/to/file.py:function_name`.

- `examples/reward_function/math.py` — format + answer correctness for math/perception
- `examples/reward_function/medical_evidence.py` — bounding box IoU + parse correctness for evidence grounding

## Config Key Parameters

| Key | Default | Notes |
|-----|---------|-------|
| `algorithm.adv_estimator` | `dapo` | `grpo` or `dapo` |
| `algorithm.disable_kl` | `true` | reference KL in actor loss |
| `algorithm.use_kl_prcp` | `true` | PAPO perception KL |
| `algorithm.kl_prcp_coef` | `0.01` | perception KL weight |
| `worker.rollout.n` | `5` | samples per prompt |
| `worker.actor.fsdp.torch_dtype` | `bf16` | training precision |
| `data.max_pixels` | `1003520` | 1280×28×28 |

## Data Layout

```
data/
├── images/                          # 26,204 PNG/JPG — 13 GB, NOT for Git
├── rl_3_evidence_papo_jsonl/
│   └── train_cepo_lite_80_10_10_papo.jsonl  # 45 MB train set
└── eval/
    └── benchmark_medreasoner_evidence_eval_400_balanced_papo.jsonl
```

JSONL schema: `{"problem": "...", "answer": "...", "images": ["data/images/..."]}`

Image paths in JSONL are repo-relative — run training from repo root.

## Conda Environment

Scripts expect conda env `papo`. The portable script sources conda from a hardcoded path; update the conda init line at the top of `examples/ours_grpo/qwen2_5_vl_7b_cepo_lite_grpo_portable.sh` when deploying to a new cluster.

## Checkpoint Resume

Training auto-resumes from `checkpoints/${WANDB_PROJECT}/${EXP_NAME}/` (latest step). Override with `trainer.load_checkpoint_path`.
