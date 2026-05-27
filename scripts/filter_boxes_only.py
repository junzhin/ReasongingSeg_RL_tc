"""
filter_boxes_only.py
过滤训练数据，只保留"有合法框"的样本（双保险：标签 + 内容）。

过滤条件：
  1. sample_type != "evidence_deleted"   （标签层）
  2. gt.num_boxes > 0                    （内容层）
  3. 至少一个坐标合法的框存在           （内容层）

默认参数在脚本顶部定义，也可通过命令行覆盖。
"""

import argparse
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

# ── 默认参数（可在此处直接修改，或通过命令行覆盖）─────────────────────────────
DEFAULT_INPUT  = "data/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_grpo.jsonl"
DEFAULT_OUTPUT = "data/rl_3_evidence_papo_jsonl/train_boxes_only_grpo.jsonl"
EXCLUDE_TYPES  = {"evidence_deleted"}
# ─────────────────────────────────────────────────────────────────────────────

Box = Tuple[float, float, float, float]


def _valid_box(box: Any) -> Optional[Box]:
    """校验单个 bbox 坐标合法性，与 reward function 保持一致。"""
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000:
        return x1, y1, x2, y2
    return None


def _extract_valid_boxes(gt: dict) -> List[Box]:
    """从 gt 中提取坐标合法的框列表。"""
    raw = gt.get("boxes") or gt.get("objects") or []
    boxes = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                parsed = _valid_box(item.get("bbox"))
                if parsed is not None:
                    boxes.append(parsed)
    return boxes


def _parse_answer(entry: dict) -> tuple:
    """
    返回 (sample_type, gt_dict)。
    gt_dict 已解包到最内层的 ground_truth 字典。
    """
    answer = entry.get("answer", {})
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except json.JSONDecodeError:
            return "unknown", {}

    sample_type = answer.get("sample_type", "unknown")

    gt = answer.get("ground_truth", {})
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except json.JSONDecodeError:
            gt = {}

    return sample_type, (gt if isinstance(gt, dict) else {})


def filter_jsonl(input_path: str, output_path: str, exclude_types: set) -> None:
    input_p  = Path(input_path)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    total            = 0
    skip_label       = 0   # 被标签过滤掉
    skip_content     = 0   # 标签通过但内容脏（num_boxes=0 或无合法框）
    skip_parse_error = 0   # JSON 解析失败
    kept             = 0

    with input_p.open("r", encoding="utf-8") as fin, \
         output_p.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] 第 {total} 行 JSON 解析失败，跳过")
                skip_parse_error += 1
                continue

            sample_type, gt = _parse_answer(entry)

            # 层1：标签过滤
            if sample_type in exclude_types:
                skip_label += 1
                continue

            # 层2：内容校验
            num_boxes_declared = int(gt.get("num_boxes", -1))
            valid_boxes = _extract_valid_boxes(gt)

            if num_boxes_declared == 0 or len(valid_boxes) == 0:
                print(f"[WARN] 第 {total} 行 sample_type={sample_type} 但内容无合法框"
                      f"（declared={num_boxes_declared}, valid={len(valid_boxes)}），跳过")
                skip_content += 1
                continue

            fout.write(line + "\n")
            kept += 1

    print()
    print("=" * 50)
    print(f"输入文件  : {input_p}")
    print(f"总条数    : {total}")
    print(f"跳过(标签): {skip_label}  （{exclude_types}）")
    print(f"跳过(内容): {skip_content} （标签通过但无合法框）")
    print(f"跳过(解析): {skip_parse_error}")
    print(f"最终保留  : {kept}")
    print(f"输出文件  : {output_p}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="过滤 evidence_deleted 及无框样本（双保险）")
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="输入 JSONL 路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 JSONL 路径")
    args = parser.parse_args()

    filter_jsonl(args.input, args.output, EXCLUDE_TYPES)


if __name__ == "__main__":
    main()
