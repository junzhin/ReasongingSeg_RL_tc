import json
from typing import Any, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# 医学证据定位 Reward Function
#
# 【整体职责】
# 这个文件是 RL 训练的"评分裁判"——模型生成一段文字回答后，
# 这里的函数决定这个回答得几分（reward）。
# reward 信号是 RL 训练的唯一监督来源，模型会朝着得分更高的方向进化。
#
# 【任务定义】
# 模型输入：医学图像 + 问题（"图中的证据区域在哪里？"）
# 模型输出：包含 bounding box 的 JSON，格式为：
#   <evidence>{"boxes": [{"bbox": [x1,y1,x2,y2]}, ...], "num_boxes": N}</evidence>
#   坐标范围 0–1000（归一化，0=左上角，1000=右下角）
#
# 【评分逻辑总览】
#   解析失败（格式错）          → -0.2
#   解析成功但漏检（gt有框pred空）→ -0.2
#   evidence_deleted 样本（应无框）：
#       pred 也是空框           → +0.5  （正确识别"无证据"）
#       pred 有框               → -0.1 ~ -0.5 （框越重叠惩罚越重）
#   正常样本：
#       overall = 0.9 × matched_IoU + 0.1 × (pred框数 == gt框数)
#
# 【为什么这样设计分数？】
# - 主要用 IoU（框的重叠度）而非像素分割指标，是因为 VLM 输出的是坐标，
#   不是像素 mask；IoU 对坐标的偏差有连续的惩罚梯度，适合 RL 优化。
# - 格式错误给 -0.2（负分）而非 0，是为了给模型强信号：
#   输出格式不对是比输出空结果更差的行为。
# - evidence_deleted 样本单独处理：该类样本的 ground truth 是"无证据"，
#   正确答案是输出空框（num_boxes=0）。如果模型画了框反而扣分。
# ══════════════════════════════════════════════════════════════════════════════

Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2)，坐标范围 0–1000


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数：从文本中提取 XML 标签内容
# 模型输出的 <evidence>...</evidence> 就靠这个函数提取
# ──────────────────────────────────────────────────────────────────────────────
def _tag_content(text: str, tag: str) -> Optional[str]:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start < 0 or end < 0 or end <= start:
        return None
    return text[start + len(start_tag) : end].strip()


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数：验证并提取单个 bounding box
# 合法 box 必须：长度为4、全是数字、坐标在 [0,1000] 范围内、x1<x2 且 y1<y2
# 坐标不合法的框一律视为解析失败（整个回答得 -0.2）
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# 解析模型输出的 evidence 标签
# 返回 parse_ok（是否成功解析）、boxes（框列表）、num_boxes（框数量）
#
# 任何一个环节失败（标签缺失、JSON 格式错、box 坐标非法、num_boxes 与实际不符）
# 都返回 parse_ok=False，最终得分 -0.2。
# 这种严格校验的目的：强迫模型学习严格的输出格式，避免输出格式"差不多对"的捷径。
# ──────────────────────────────────────────────────────────────────────────────
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
        raw_boxes = payload.get("objects", [])  # 兼容旧版本字段名
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

    # num_boxes 必须与实际 boxes 数量完全一致，否则认为格式错误
    # 这个约束防止模型输出 num_boxes=0 但实际有框的情况
    num_boxes = payload.get("num_boxes", len(boxes))
    if not isinstance(num_boxes, int) or num_boxes != len(boxes):
        return {"parse_ok": False, "boxes": [], "num_boxes": 0}
    return {"parse_ok": True, "boxes": boxes, "num_boxes": num_boxes}


# ──────────────────────────────────────────────────────────────────────────────
# 从 ground truth 字典中提取标准答案框
# primary=True：提取主要证据框（boxes/objects）
# primary=False：提取"被删除"的目标框（target_boxes/masked_objects），
#   用于 evidence_deleted 样本的惩罚计算
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# IoU（Intersection over Union）：衡量两个框的重叠程度
# 值域 [0, 1]：0 = 完全不重叠，1 = 完全重合
# 这是目标检测领域最常用的框匹配指标
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# Dice 系数：另一种框重叠度量，对小框更敏感（分母是两框面积之和而非并集）
# 值域 [0, 1]，同 IoU 方向一致（越大越好）
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# 匹配 IoU：贪心一对一匹配 pred 框和 gt 框，计算平均 IoU
#
# 【为什么用贪心匹配而不是全排列最优匹配？】
# 一个 gt 框只能被一个 pred 框匹配（用掉后从候选集移除）。
# 这防止了一个 pred 框"包揽"所有 gt 框的情况——
# 如果用简单的"每个 gt 找最近的 pred"，一个大框可能和所有 gt 都有高 IoU，
# 模型就学会只输出一个大框来刷分。贪心匹配消除了这个作弊路径。
#
# 最终分数 = 所有 gt 框匹配到的最佳 IoU 之和 / gt 框数量
# （gt 框没被匹配到就算 0）
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# 匹配 Dice：与 _matched_iou 逻辑相同，但用 Dice 指标替代 IoU
# 两者都计算，IoU 作为主要 reward 信号（权重0.9），Dice 只记录到日志
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# 最大 IoU/Dice：不做一对一匹配，取任意 pred-gt 对的最大值
# 专门用于 evidence_deleted 样本的惩罚计算：
# 如果模型画了框，看它和"被删除的目标框"最多重叠多少——重叠越多惩罚越重
# （说明模型在"已知无证据"的区域还在乱画框）
# ──────────────────────────────────────────────────────────────────────────────
def _max_iou(pred_boxes: List[Box], target_boxes: List[Box]) -> float:
    if not pred_boxes or not target_boxes:
        return 0.0
    return max(_iou(pred, target) for pred in pred_boxes for target in target_boxes)


def _max_dice(pred_boxes: List[Box], target_boxes: List[Box]) -> float:
    if not pred_boxes or not target_boxes:
        return 0.0
    return max(_dice(pred, target) for pred in pred_boxes for target in target_boxes)


# ──────────────────────────────────────────────────────────────────────────────
# 解析 ground truth 字段
# ground_truth 可能是字符串（JSON序列化）或字典，这里统一处理
# sample_type 决定打分逻辑：
#   "original" / "positive" → 正常样本，比框的 IoU
#   "evidence_deleted"       → 无证据样本，pred 应为空框
# ──────────────────────────────────────────────────────────────────────────────
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
        sample_type = "original"  # "positive" 是 "original" 的别名，统一处理
    return str(sample_type), gt


# ──────────────────────────────────────────────────────────────────────────────
# 单个样本打分核心逻辑
# 返回字典：overall 是 RL 优化目标，其余字段记录到 TensorBoard/wandb 供诊断
# ──────────────────────────────────────────────────────────────────────────────
def _score_one(reward_input: Dict[str, Any]) -> Dict[str, float]:
    sample_type, gt = _normalize_ground_truth(reward_input.get("ground_truth"))
    pred = parse_evidence(str(reward_input.get("response", "")))
    parse_ok = 1.0 if pred["parse_ok"] else 0.0
    pred_boxes: List[Box] = pred["boxes"]
    pred_num = int(pred["num_boxes"])
    gt_num = int(gt.get("num_boxes", len(_extract_gt_boxes(gt, primary=True))) or 0)
    gt_boxes = _extract_gt_boxes(gt, primary=True)
    target_boxes = _extract_gt_boxes(gt, primary=False)

    # 情况1：格式解析失败 → 直接 -0.2，无需进一步比较
    if not pred["parse_ok"]:
        return {
            "overall": -0.2,
            "bbox_iou": 0.0,
            "bbox_dice": 0.0,
            "parse_ok": 0.0,
            "num_ok": 0.0,
            "empty_rate": 0.0,
        }

    empty_rate = 1.0 if pred_num == 0 else 0.0      # 记录模型输出空框的频率（诊断用）
    num_ok = 1.0 if pred_num == gt_num else 0.0      # 框数量是否预测正确

    # 情况2：evidence_deleted 样本（标注时人工删除了图中的目标区域）
    # 这类样本的正确答案是"无证据"，模型应输出空框
    if sample_type == "evidence_deleted":
        max_iou_to_target = _max_iou(pred_boxes, target_boxes)   # 预测框与删除区域的重叠度
        max_dice_to_target = _max_dice(pred_boxes, target_boxes)
        # 输出空框：+0.5 奖励（难度较高，特别奖励）
        # 输出有框：-0.1 基础惩罚 - 0.4×重叠度（越重叠说明越"看到了不存在的东西"）
        overall = 0.5 if pred_num == 0 else -0.1 - 0.4 * max_iou_to_target
        return {
            "overall": overall,
            "bbox_iou": max_iou_to_target,
            "bbox_dice": max_dice_to_target,
            "parse_ok": parse_ok,
            "num_ok": num_ok,
            "empty_rate": empty_rate,
        }

    # 情况3：正常样本但预测为空框（漏检）→ -0.2
    # 格式正确但什么都没预测，和格式错误一样严重
    if gt_num > 0 and pred_num == 0:
        return {
            "overall": -0.2,
            "bbox_iou": 0.0,
            "bbox_dice": 0.0,
            "parse_ok": parse_ok,
            "num_ok": num_ok,
            "empty_rate": empty_rate,
        }

    # 情况4：正常样本，pred 和 gt 都有框 → 计算匹配 IoU/Dice
    # overall = 0.9 × IoU + 0.1 × 框数正确
    # IoU 是主信号，框数正确是辅助约束（防止模型输出过多/过少的框）
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


# ──────────────────────────────────────────────────────────────────────────────
# 对外接口：批量打分
#
# 【调用方式】
# 在 config.yaml 中指定：
#   worker.reward.reward_function: examples/reward_function/medical_evidence.py:compute_score
#
# 【输入格式】
# reward_inputs: List[Dict]，每个 dict 包含：
#   - "response": 模型生成的回答字符串
#   - "ground_truth": 标准答案（字典或 JSON 字符串）
#
# 【输出格式】
# List[Dict]，每个 dict 包含：
#   - "overall": float，RL 优化的目标分数
#   - "bbox_iou", "bbox_dice", "parse_ok", "num_ok", "empty_rate": 诊断指标
# ──────────────────────────────────────────────────────────────────────────────
def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for medical evidence reward function.")
    return [_score_one(item) for item in reward_inputs]
