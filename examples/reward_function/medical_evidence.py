import json
from typing import Any, Dict, List, Optional, Tuple


Box = Tuple[float, float, float, float]


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


def _extract_gt_boxes(gt: Dict[str, Any], primary: bool) -> List[Box]:
    keys = ("boxes", "objects") if primary else ("target_boxes", "masked_objects", "boxes", "objects")
    raw: Any = []
    for key in keys:
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


def _max_iou(pred_boxes: List[Box], target_boxes: List[Box]) -> float:
    if not pred_boxes or not target_boxes:
        return 0.0
    return max(_iou(pred, target) for pred in pred_boxes for target in target_boxes)


def _max_dice(pred_boxes: List[Box], target_boxes: List[Box]) -> float:
    if not pred_boxes or not target_boxes:
        return 0.0
    return max(_dice(pred, target) for pred in pred_boxes for target in target_boxes)


def _normalize_ground_truth(ground_truth: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return "original", {}
    if not isinstance(ground_truth, dict):
        return "original", {}

    gt = ground_truth.get("ground_truth") if "ground_truth" in ground_truth else ground_truth
    if not isinstance(gt, dict):
        gt = {}
    sample_type = ground_truth.get("sample_type") or gt.get("sample_type") or "original"
    if sample_type == "positive":
        sample_type = "original"
    return str(sample_type), gt


def _score_one(reward_input: Dict[str, Any]) -> Dict[str, float]:
    sample_type, gt = _normalize_ground_truth(reward_input.get("ground_truth"))
    pred = parse_evidence(str(reward_input.get("response", "")))
    parse_ok = 1.0 if pred["parse_ok"] else 0.0
    pred_boxes: List[Box] = pred["boxes"]
    pred_num = int(pred["num_boxes"])
    gt_num = int(gt.get("num_boxes", len(_extract_gt_boxes(gt, primary=True))) or 0)
    gt_boxes = _extract_gt_boxes(gt, primary=True)
    target_boxes = _extract_gt_boxes(gt, primary=False)

    if not pred["parse_ok"]:
        return {
            "overall": -0.2,
            "bbox_iou": 0.0,
            "bbox_dice": 0.0,
            "parse_ok": 0.0,
            "num_ok": 0.0,
            "empty_rate": 0.0,
        }

    empty_rate = 1.0 if pred_num == 0 else 0.0
    num_ok = 1.0 if pred_num == gt_num else 0.0

    if sample_type == "evidence_deleted":
        max_iou_to_target = _max_iou(pred_boxes, target_boxes)
        max_dice_to_target = _max_dice(pred_boxes, target_boxes)
        overall = 0.5 if pred_num == 0 else -0.1 - 0.4 * max_iou_to_target
        return {
            "overall": overall,
            "bbox_iou": max_iou_to_target,
            "bbox_dice": max_dice_to_target,
            "parse_ok": parse_ok,
            "num_ok": num_ok,
            "empty_rate": empty_rate,
        }

    if gt_num > 0 and pred_num == 0:
        return {
            "overall": -0.2,
            "bbox_iou": 0.0,
            "bbox_dice": 0.0,
            "parse_ok": parse_ok,
            "num_ok": num_ok,
            "empty_rate": empty_rate,
        }

    bbox_iou = _matched_iou(pred_boxes, gt_boxes)
    bbox_dice = _matched_dice(pred_boxes, gt_boxes)
    overall = 0.90 * bbox_iou + 0.10 * num_ok
    return {
        "overall": overall,
        "bbox_iou": bbox_iou,
        "bbox_dice": bbox_dice,
        "parse_ok": parse_ok,
        "num_ok": num_ok,
        "empty_rate": empty_rate,
    }


def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for medical evidence reward function.")
    return [_score_one(item) for item in reward_inputs]
