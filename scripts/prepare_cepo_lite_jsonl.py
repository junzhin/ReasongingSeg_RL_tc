#!/usr/bin/env python3
"""Build PAPO-loader JSONL files for CEPO-lite medical evidence RL."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIR = Path(
    "/mnt/petrelfs/tangcheng/tangcheng/ReasongingSeg/data/sft_bench_rl_v2/rl_3_evidence_jsonl"
)
DEFAULT_BENCHMARK = Path(
    "/mnt/petrelfs/tangcheng/tangcheng/ReasongingSeg/data/sft_bench_rl_v2/benchmark_medreasoner_evidence_eval.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/petrelfs/tangcheng/tangcheng/ReasongingSeg/data/sft_bench_rl_v2/rl_3_evidence_papo_jsonl"
)
DEFAULT_EVAL_OUTPUT_DIR = Path(
    "/mnt/petrelfs/tangcheng/tangcheng/ReasongingSeg/PAPO/data/eval"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare converted CEPO-lite JSONL files for PAPO/verl.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--benchmark-file", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-output-dir", type=Path, default=DEFAULT_EVAL_OUTPUT_DIR)
    parser.add_argument("--eval-subset-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def first_user_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for turn in prompt:
            if isinstance(turn, dict) and turn.get("role") == "user":
                return str(turn.get("content", "")).strip()
    if isinstance(prompt, str):
        return prompt.strip()

    for turn in row.get("conversations") or []:
        if isinstance(turn, dict) and turn.get("from") == "human":
            return str(turn.get("value", "")).strip()
    raise ValueError(f"sample {row.get('id')} has no user prompt")


def infer_sample_type(row: dict[str, Any]) -> str:
    gt = (row.get("reward_model") or {}).get("ground_truth") or {}
    meta = row.get("meta") or row.get("extra_info") or {}
    return (
        row.get("sample_type")
        or gt.get("sample_type")
        or (meta.get("cf_info") or {}).get("type")
        or "original"
    )


def converted_row(row: dict[str, Any]) -> dict[str, Any]:
    gt = (row.get("reward_model") or {}).get("ground_truth")
    if not isinstance(gt, dict):
        raise ValueError(f"sample {row.get('id')} missing reward_model.ground_truth")
    images = row.get("images")
    if not isinstance(images, list):
        raise ValueError(f"sample {row.get('id')} missing images list")

    sample_type = infer_sample_type(row)
    return {
        "id": row.get("id"),
        "problem": first_user_text(row),
        "images": images,
        "answer": {
            "sample_type": sample_type,
            "ground_truth": gt,
        },
        "meta": row.get("meta") or row.get("extra_info") or {},
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            out = converted_row(row)
            counts[out["answer"]["sample_type"]] += 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return {"path": str(path), "num_rows": len(rows), "sample_type_counts": dict(counts)}


def sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"requested {count} rows from only {len(rows)} rows")
    return rng.sample(rows, count)


def benchmark_group_key(row: dict[str, Any]) -> tuple[str, int]:
    gt = (row.get("reward_model") or {}).get("ground_truth") or {}
    extra = row.get("extra_info") or row.get("meta") or {}
    modality = str(gt.get("modality") or extra.get("modality") or "unknown")
    try:
        num_boxes = int(gt.get("num_boxes", 0))
    except (TypeError, ValueError):
        num_boxes = 0
    return modality, num_boxes


def balanced_benchmark_subset(rows: list[dict[str, Any]], size: int, rng: random.Random) -> list[dict[str, Any]]:
    if size <= 0 or size >= len(rows):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled

    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(benchmark_group_key(row), []).append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    keys = sorted(groups, key=lambda item: (item[0], item[1]))
    selected: list[dict[str, Any]] = []
    while len(selected) < size and keys:
        next_keys = []
        for key in keys:
            if groups[key] and len(selected) < size:
                selected.append(groups[key].pop())
            if groups[key]:
                next_keys.append(key)
        keys = next_keys

    rng.shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    original = read_jsonl(args.source_dir / "original.jsonl")
    deleted = read_jsonl(args.source_dir / "evidence_deleted.jsonl")
    background = read_jsonl(args.source_dir / "background_perturbed.jsonl")
    benchmark = read_jsonl(args.benchmark_file)

    n_original = len(original)
    n_80 = round(n_original * 0.10 / 0.80)
    n_70 = round(n_original * 0.15 / 0.70)

    outputs = []
    outputs.append(write_jsonl(args.output_dir / "train_original_only_papo.jsonl", original))

    rows_80 = original + sample_rows(deleted, n_80, rng) + sample_rows(background, n_80, rng)
    rng.shuffle(rows_80)
    outputs.append(write_jsonl(args.output_dir / "train_cepo_lite_80_10_10_papo.jsonl", rows_80))

    rows_70 = original + sample_rows(deleted, n_70, rng) + sample_rows(background, n_70, rng)
    rng.shuffle(rows_70)
    outputs.append(write_jsonl(args.output_dir / "train_cepo_lite_70_15_15_papo.jsonl", rows_70))

    outputs.append(write_jsonl(args.output_dir / "benchmark_medreasoner_evidence_eval_papo.jsonl", benchmark))

    eval_subset = balanced_benchmark_subset(benchmark, args.eval_subset_size, rng)
    args.eval_output_dir.mkdir(parents=True, exist_ok=True)
    raw_subset_path = args.eval_output_dir / "benchmark_medreasoner_evidence_eval_400_balanced.jsonl"
    with raw_subset_path.open("w", encoding="utf-8") as f:
        for row in eval_subset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    outputs.append(write_jsonl(args.eval_output_dir / "benchmark_medreasoner_evidence_eval_400_balanced_papo.jsonl", eval_subset))

    summary = {
        "source_dir": str(args.source_dir),
        "benchmark_file": str(args.benchmark_file),
        "seed": args.seed,
        "output_dir": str(args.output_dir),
        "eval_output_dir": str(args.eval_output_dir),
        "eval_subset_size": len(eval_subset),
        "eval_subset_group_counts": {
            f"{modality}|{num_boxes}": count
            for (modality, num_boxes), count in sorted(Counter(benchmark_group_key(row) for row in eval_subset).items())
        },
        "outputs": outputs,
    }
    summary_path = args.output_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
