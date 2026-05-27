import json
from typing import Any, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# 医学证据定位 Reward Function — boxes_only 版本
#
# 【与原版 medical_evidence.py 的区别】
# 本版本用于"强制有框"的对照实验，假设数据中所有样本都有框（已过滤掉
# evidence_deleted 样本），因此：
#   - 移除了 evidence_deleted 打分分支
#   - 移除了 _max_iou / _max_dice（仅 anti 场景使用）
#   - 移除了 empty_rate 返回字段
#   - overall 改为纯 matched_IoU（不再加 0.1×num_ok）
#
# 【评分逻辑（简化为3条）】
#   格式解析失败             → -0.2
#   gt有框但pred为空（漏检） → -0.2
#   pred有框 + gt有框        → overall = matched_IoU
# ══════════════════════════════════════════════════════════════════════════════

Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2)，坐标范围 0–1000


def _tag_content(text: str, tag: str) -> Optional[str]:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start < 0 or end < 0 or end <= start:
        return None
    return text[start + len(start_tag) : end].strip()


def _valid_box(box: Any) -> Optional[Box]:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000:
        return x1, y1, x2, y2
    return None


def parse_evidence(response: str) -> Dict[str, Any]:
    content = _tag_content(response or "", "evidence")
    if content is None:
        return {"parse_ok": False, "boxes": [], "num_boxes": 0}
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {"parse_ok": False, "boxes": [], "num_boxes": 0}
    if not isinstance(payload, dict):
        return {"parse_ok": False, "boxes": [], "num_boxes": 0}

    raw_boxes = payload.get("boxes")
    if raw_boxes is None:
        raw_boxes = payload.get("objects", [])
    if not isinstance(raw_boxes, list):
        return {"parse_ok": False, "boxes": [], "num_boxes": 0}

    boxes: List[Box] = []
    for item in raw_boxes:
        if not isinstance(item, dict):
            return {"parse_ok": False, "boxes": [], "num_boxes": 0}
        parsed = _valid_box(item.get("bbox"))
        if parsed is None:
            return {"parse_ok": False, "boxes": [], "num_boxes": 0}
        boxes.append(parsed)

    num_boxes = payload.get("num_boxes", len(boxes))
    if not isinstance(num_boxes, int) or num_boxes != len(boxes):
        return {"parse_ok": False, "boxes": [], "num_boxes": 0}
    return {"parse_ok": True, "boxes": boxes, "num_boxes": num_boxes}


def _extract_gt_boxes(gt: Dict[str, Any]) -> List[Box]:
    """提取 ground truth 中的主要证据框（boxes / objects）。"""
    raw: Any = []
    for key in ("boxes", "objects"):
        if key in gt:
            raw = gt.get(key)
            break
    boxes: List[Box] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                parsed = _valid_box(item.get("bbox"))
                if parsed is not None:
                    boxes.append(parsed)
    return boxes


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def _dice(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b
    return 0.0 if denom <= 0 else 2.0 * inter / denom


def _matched_iou(pred_boxes: List[Box], gt_boxes: List[Box]) -> float:
    if not pred_boxes or not gt_boxes:
        return 0.0
    unused = set(range(len(pred_boxes)))
    matched = []
    for gt_box in gt_boxes:
        best_idx = None
        best_iou = 0.0
        for idx in unused:
            cur = _iou(pred_boxes[idx], gt_box)
            if cur > best_iou:
                best_idx = idx
                best_iou = cur
        if best_idx is not None:
            unused.remove(best_idx)
        matched.append(best_iou)
    return sum(matched) / len(gt_boxes)


def _matched_dice(pred_boxes: List[Box], gt_boxes: List[Box]) -> float:
    if not pred_boxes or not gt_boxes:
        return 0.0
    unused = set(range(len(pred_boxes)))
    matched = []
    for gt_box in gt_boxes:
        best_idx = None
        best_dice = 0.0
        for idx in unused:
            cur = _dice(pred_boxes[idx], gt_box)
            if cur > best_dice:
                best_idx = idx
                best_dice = cur
        if best_idx is not None:
            unused.remove(best_idx)
        matched.append(best_dice)
    return sum(matched) / len(gt_boxes)


def _normalize_ground_truth(ground_truth: Any) -> Dict[str, Any]:
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return {}
    if not isinstance(ground_truth, dict):
        return {}
    gt = ground_truth.get("ground_truth") if "ground_truth" in ground_truth else ground_truth
    return gt if isinstance(gt, dict) else {}


def _score_one(reward_input: Dict[str, Any]) -> Dict[str, float]:
    gt = _normalize_ground_truth(reward_input.get("ground_truth"))
    pred = parse_evidence(str(reward_input.get("response", "")))

    parse_ok = 1.0 if pred["parse_ok"] else 0.0
    pred_boxes: List[Box] = pred["boxes"]
    pred_num = int(pred["num_boxes"])
    gt_boxes = _extract_gt_boxes(gt)
    gt_num = int(gt.get("num_boxes", len(gt_boxes)) or 0)
    num_ok = 1.0 if pred_num == gt_num else 0.0

    # 情况1：格式解析失败 → -0.2
    if not pred["parse_ok"]:
        return {
            "overall": -0.2,
            "bbox_iou": 0.0,
            "bbox_dice": 0.0,
            "parse_ok": 0.0,
            "num_ok": 0.0,
        }

    # 情况1.5：GT 无框 — boxes-only 数据集中不应出现，属于脏数据
    # 返回固定负分并打印警告，便于训练时发现并排查数据问题
    if gt_num == 0 or len(gt_boxes) == 0:
        import warnings
        warnings.warn(
            f"[medical_evidence_boxes_only] gt_num=0 detected in boxes-only dataset. "
            f"This sample should have been filtered out. gt={gt}",
            RuntimeWarning, stacklevel=3,
        )
        return {
            "overall": -0.2,
            "bbox_iou": 0.0,
            "bbox_dice": 0.0,
            "parse_ok": parse_ok,
            "num_ok": 0.0,
        }

    # 情况2：gt 有框但 pred 为空（漏检）→ -0.2
    if gt_num > 0 and pred_num == 0:
        return {
            "overall": -0.2,
            "bbox_iou": 0.0,
            "bbox_dice": 0.0,
            "parse_ok": parse_ok,
            "num_ok": num_ok,
        }

    # 情况3：pred 和 gt 都有框 → overall = matched_IoU（纯定位信号）
    bbox_iou = _matched_iou(pred_boxes, gt_boxes)
    bbox_dice = _matched_dice(pred_boxes, gt_boxes)
    return {
        "overall": bbox_iou,
        "bbox_iou": bbox_iou,
        "bbox_dice": bbox_dice,
        "parse_ok": parse_ok,
        "num_ok": num_ok,
    }


def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for medical evidence reward function.")
    return [_score_one(item) for item in reward_inputs]
