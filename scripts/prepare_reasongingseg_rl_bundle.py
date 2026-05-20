#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SOURCE_REPO = Path("/mnt/petrelfs/tangcheng/tangcheng/ReasongingSeg/PAPO")
SOURCE_DATA_ROOT = Path("/mnt/petrelfs/tangcheng/tangcheng/ReasongingSeg/data")
SOURCE_MODEL_DIR = Path("/mnt/petrelfs/tangcheng/tangcheng/ReasongingSeg/model/Lingshu-7B")
DEFAULT_DEST = Path("/mnt/petrelfs/tangcheng/tangcheng/ReasongingSeg/ReasongingSeg_RL")

TRAIN_JSONL = SOURCE_DATA_ROOT / "sft_bench_rl_v2/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_papo.jsonl"
EVAL_JSONL_DIR = SOURCE_REPO / "data/eval"

CODE_EXCLUDES = [
    ".git",
    "PAPO-Eval",
    "checkpoints",
    "logs",
    "wandb",
    "__pycache__",
    "*.pyc",
    "*.pyo",
]


def run(cmd):
    subprocess.run(cmd, check=True)


def rsync_dir(src: Path, dst: Path, extra_excludes=None):
    cmd = ["rsync", "-a", f"{src}/", f"{dst}/"]
    for pattern in CODE_EXCLUDES + (extra_excludes or []):
        cmd.extend(["--exclude", pattern])
    run(cmd)


def copy_model(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-a", f"{src}/", f"{dst}/"])


def rewrite_and_collect(src_jsonl: Path, dst_jsonl: Path, source_data_root: Path):
    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)
    image_paths = set()
    line_count = 0

    with src_jsonl.open() as fin, dst_jsonl.open("w") as fout:
        for raw in fin:
            if not raw.strip():
                continue
            obj = json.loads(raw)
            new_images = []
            for image in obj.get("images", []):
                image_path = Path(image)
                if not str(image_path).startswith(str(source_data_root) + "/"):
                    raise ValueError(f"Image path not under source data root: {image}")
                rel = image_path.relative_to(source_data_root)
                new_path = Path("data/images") / rel
                new_images.append(new_path.as_posix())
                image_paths.add(image_path)
            obj["images"] = new_images
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            line_count += 1
    return line_count, image_paths


def copy_one_image(src: Path, source_data_root: Path, dest_root: Path):
    rel = src.relative_to(source_data_root)
    dst = dest_root / "data/images" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def write_readme(dest_root: Path, summary: dict):
    readme = f"""# ReasongingSeg RL Bundle

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
cd {dest_root}
bash examples/ours_grpo/qwen2_5_vl_7b_cepo_lite_grpo_portable.sh
```

## Typical Slurm Submission

```bash
cd {dest_root}
sbatch scripts/submit_cepo_lite_7b_reserved.sbatch
```

## Notes

- The portable script auto-resumes from the latest checkpoint under `checkpoints/${{WANDB_PROJECT}}/${{EXP_NAME}}`.
- The bundle does not include `PAPO-Eval`.
- The bundle does not include previous training checkpoints.
- The bundle does not include `final_sft_lingshu`; only `models/Lingshu-7B` is copied.

## Bundle Summary

- Training JSONLs copied: {summary["train_jsonl_count"]}
- Eval JSONLs copied: {summary["eval_jsonl_count"]}
- Unique images copied: {summary["image_count"]}
- Missing images detected during build: {summary["missing_images"]}
"""
    (dest_root / "README.md").write_text(readme)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    dest_root = args.dest.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    rsync_dir(SOURCE_REPO, dest_root)

    train_dst = dest_root / "data/rl_3_evidence_papo_jsonl" / TRAIN_JSONL.name
    train_line_count, train_images = rewrite_and_collect(TRAIN_JSONL, train_dst, SOURCE_DATA_ROOT)

    eval_jsonls = sorted(EVAL_JSONL_DIR.glob("*.jsonl"))
    eval_line_count = 0
    all_images = set(train_images)
    for eval_jsonl in eval_jsonls:
        dst_jsonl = dest_root / "data/eval" / eval_jsonl.name
        lines, images = rewrite_and_collect(eval_jsonl, dst_jsonl, SOURCE_DATA_ROOT)
        eval_line_count += lines
        all_images.update(images)

    missing = [str(path) for path in all_images if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing images: {missing[:10]}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(lambda p: copy_one_image(p, SOURCE_DATA_ROOT, dest_root), sorted(all_images)))

    copy_model(SOURCE_MODEL_DIR, dest_root / "models/Lingshu-7B")

    summary = {
        "train_jsonl_count": train_line_count,
        "eval_jsonl_count": eval_line_count,
        "image_count": len(all_images),
        "missing_images": len(missing),
    }
    (dest_root / "bundle_manifest.json").write_text(json.dumps(summary, indent=2))
    write_readme(dest_root, summary)


if __name__ == "__main__":
    main()
