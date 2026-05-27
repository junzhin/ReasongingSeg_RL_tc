# Boxes-Only Ablation 实验记录

**创建日期**：2026-05-27  
**共两组实验**：Exp01（GRPO）、Exp02（GRPO + PAPO）  
**核心前提**：两组实验共享相同的数据过滤规则和 reward function，唯一受控变量是是否启用 PAPO。

---

## 背景：为什么需要这两组实验

在前序实验（含 `evidence_deleted` 样本的完整数据集训练）中，观察到以下异常：

- `anti_ray reward` 从训练初期快速攀升至 ≈ 1.0
- `bbox_iou` 和 `bbox_dice` 全程接近 0
- `overall reward` 随 `anti_ray` 一起增长，而非随定位精度增长

**根本原因 — Reward Hacking**：数据集中 10%（2,673/26,731 条）为 `evidence_deleted` 样本，这类样本输出空框即得 +0.5。模型找到捷径后对所有输入均输出空框，完全放弃 bounding box 定位。

此外，前序实验还使用了错误的 prompt 模板（`math_perception.jinja`，强制输出 `\boxed{}`），与医学 reward 的 `<evidence>{...}</evidence>` 格式冲突，导致 reward 直接判格式错给 -0.2。

**因此前序实验结果不可用于对比**，需要重新设计干净的对照实验。

---

## 数据改动（两组实验共用）

### 原始数据分布

| sample_type | 数量 | 说明 |
|---|---|---|
| `original` | 21,385 | 正常有框样本 |
| `background_perturbed` | 2,673 | 背景扰动，仍有框 |
| `evidence_deleted` | 2,673 | 人工删除图中目标区域，GT 为空框 |
| **合计** | **26,731** | |

### 过滤规则（双保险：标签 + 内容）

**保留条件**（需同时满足）：
1. `sample_type != "evidence_deleted"`（标签层）
2. `gt.num_boxes > 0`（内容层）
3. 至少一个坐标合法的 bbox 存在，即 `x1 < x2`，`y1 < y2`，坐标在 `[0, 1000]`（内容层）

**过滤后数据**：

| sample_type | 数量 |
|---|---|
| `original` | 21,385 |
| `background_perturbed` | 2,673 |
| **合计** | **24,058** |

跳过（标签）：2,673 条；跳过（内容脏）：0 条（当前数据集干净）

| 用途 | 输入文件 | 输出文件 |
|---|---|---|
| Exp01 (GRPO) | `train_cepo_lite_80_10_10_grpo.jsonl` | `train_boxes_only_grpo.jsonl` |
| Exp02 (GRPO+PAPO) | `train_cepo_lite_80_10_10_grpo_papo.jsonl` | `train_boxes_only_grpo_papo.jsonl` |

**过滤脚本**：`scripts/filter_boxes_only.py`

---

## Reward Function 改动（两组实验共用）

**新文件**：`examples/reward_function/medical_evidence_boxes_only.py`  
**原文件**：`examples/reward_function/medical_evidence.py`（未修改，保留对照）

### 原版 vs Boxes-Only 版对比

| 组件 | 原版 `medical_evidence.py` | 本实验 `medical_evidence_boxes_only.py` |
|---|---|---|
| `evidence_deleted` 打分分支 | ✅ pred空框→+0.5，有框→-0.1~-0.5 | ❌ 删除 |
| `_max_iou` / `_max_dice` 函数 | ✅ 计算 pred 与被删区域重叠度 | ❌ 删除 |
| `empty_rate` 返回字段 | ✅ 记录空框输出频率 | ❌ 删除 |
| `overall` 公式 | `0.9 × matched_IoU + 0.1 × num_ok` | `matched_IoU`（纯 IoU） |
| `num_ok` 字段 | 进入 overall（权重 0.1） | 保留为诊断字段，不进 overall |
| `parse_ok` | ✅ 保留 | ✅ 保留 |
| `bbox_iou` | ✅ 保留 | ✅ 保留 |
| `bbox_dice` | ✅ 保留（诊断） | ✅ 保留（诊断） |
| 脏数据 fail-fast | ❌ 无 | ✅ gt_num=0 → RuntimeWarning + overall=-0.2 |

### 简化后打分逻辑（3条）

```
情况1：格式解析失败（无 <evidence> 标签 / JSON 错 / 坐标非法）→ overall = -0.2
情况2：GT 有框但预测为空框（漏检）                             → overall = -0.2
情况3：GT 有框，预测也有框                                     → overall = matched_IoU
```

### 为什么删掉 `0.1 × num_ok`

- 实验目标是验证**纯 IoU 信号**能否驱动定位学习，num_ok 是额外约束会混淆消融结论
- 多数样本 num_boxes=1，num_ok 梯度贡献极小
- num_ok 保留为诊断字段，可在后续实验中单独评估其作用

---

## Exp01：GRPO Boxes-Only

**实验标识**：`exp01_grpo_boxes_only`  
**训练脚本**：`examples/ours_medical/qwen2_5_vl_7b_grpo_boxes_only_4xH200.sh`

### 配置

| 参数 | 值 |
|---|---|
| 算法 | GRPO |
| PAPO | 全部关闭（`use_kl_prcp=false`） |
| 训练集 | `train_boxes_only_grpo.jsonl`（24,058 条） |
| Reward | `medical_evidence_boxes_only.py` |
| Prompt 模板 | `medical_evidence.jinja` |
| Epochs | 10 |
| Rollout batch | 384 |
| Global batch | 128 |

### 与原版 GRPO 实验的唯一区别

| 变量 | 原版 GRPO | Exp01 |
|---|---|---|
| 训练数据 | 含 evidence_deleted（26,731 条） | 仅有框（24,058 条） |
| Reward | `medical_evidence.py` | `medical_evidence_boxes_only.py` |
| Prompt 模板 | `math_perception.jinja`（❌ 格式冲突） | `medical_evidence.jinja`（✅ 正确） |

### 预期行为

- `anti_ray` 指标消失
- `bbox_iou` 从 0 开始有真实梯度，随训练上升
- `overall` 反映真实定位精度

---

## Exp02：GRPO + PAPO Boxes-Only

**实验标识**：`exp02_grpo_papo_boxes_only`  
**训练脚本**：`examples/ours_medical/qwen2_5_vl_7b_grpo_papo_boxes_only_4xH200.sh`

### 配置

| 参数 | 值 |
|---|---|
| 算法 | GRPO + PAPO |
| `use_kl_prcp` | true，coef=0.01 |
| `use_aug_entropy_loss` | true，coef=0.03 |
| `use_ori_entropy_loss` | true，coef=0.03 |
| 训练集 | `train_boxes_only_grpo_papo.jsonl`（24,058 条） |
| Reward | `medical_evidence_boxes_only.py` |
| Prompt 模板 | `medical_evidence.jinja` |
| Epochs | 10 |
| Rollout batch | 384 |
| Global batch | 128 |

### 与 Exp01 的唯一区别

| 变量 | Exp01 (GRPO) | Exp02 (GRPO+PAPO) |
|---|---|---|
| PAPO loss 项 | 无 | `kl_prcp` + `aug_entropy` + `ori_entropy` |
| 训练数据文件 | `train_boxes_only_grpo.jsonl` | `train_boxes_only_grpo_papo.jsonl` |

> **注**：两个数据文件经过相同的过滤规则处理，内容等价（24,058 条），仅文件名区分来源。Reward function 和所有超参数完全相同。

### 预期行为

与 Exp01 相同的 reward 收敛趋势，额外观察 PAPO loss 项（`actor/kl_prcp_loss`、`actor/aug_entropy_loss`、`actor/ori_entropy_loss`）是否带来定位精度的进一步提升。

---

## 实验对照关系总结（用于论文）

| 实验 | 数据 | Reward | PAPO | 用途 |
|---|---|---|---|---|
| 原始 GRPO（无效） | 含 evidence_deleted | 含 anti 分支 | ❌ | 作废，存在 reward hacking |
| **Exp01** | boxes-only | 纯 IoU | ❌ | GRPO 干净基线 |
| **Exp02** | boxes-only | 纯 IoU | ✅ | GRPO+PAPO 消融 |

---

## 文件清单

| 文件 | 说明 |
|---|---|
| `scripts/filter_boxes_only.py` | 数据过滤脚本（双保险） |
| `data/rl_3_evidence_papo_jsonl/train_boxes_only_grpo.jsonl` | Exp01 训练集（24,058 条） |
| `data/rl_3_evidence_papo_jsonl/train_boxes_only_grpo_papo.jsonl` | Exp02 训练集（24,058 条） |
| `examples/reward_function/medical_evidence_boxes_only.py` | 简化版 reward（两组共用） |
| `examples/ours_medical/qwen2_5_vl_7b_grpo_boxes_only_4xH200.sh` | Exp01 训练脚本 |
| `examples/ours_medical/qwen2_5_vl_7b_grpo_papo_boxes_only_4xH200.sh` | Exp02 训练脚本 |
| `examples/reward_function/medical_evidence.py` | 原版 reward（未修改，保留对照） |
