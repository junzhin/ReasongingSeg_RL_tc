# Experiment 03: GRPO + PAPO Full-Data Prompt-Fixed

**创建日期**：2026-05-27  
**实验标识**：`exp03_grpo_papo_full_data_prompt_fixed`  
**对应训练脚本**：`examples/ours_medical/qwen2_5_vl_7b_grpo_papo_4xH200.sh`

---

## 1. 实验动机

本实验是在 `Exp02` 纯 GRPO 基线之上，加入 PAPO 感知一致性约束后的受控对照实验。

实验目的不是改变 reward function，而是回答下面这个问题：

> 在**完全相同的医学 reward、相同训练数据、相同 Prompt 模板**下，仅通过加入 PAPO 的感知 KL 与双路 entropy 正则，是否能改善策略更新质量？

需要特别强调：

- 本实验与 `Exp02` 的 reward function 完全相同
- 本实验与 `Exp02` 的训练数据当前也完全相同
- 两者的主要差别只在 **loss function 结构**

---

## 2. 数据与 Prompt 配置

### 2.1 训练/验证数据

| 项目 | 文件 | 说明 |
|---|---|---|
| 训练集 | `data/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_grpo_papo.jsonl` | 26,731 条 |
| 验证集 | `data/eval/benchmark_medreasoner_evidence_eval_400_balanced_papo.jsonl` | 400 条 |

当前 `train_cepo_lite_80_10_10_grpo_papo.jsonl` 与 `train_cepo_lite_80_10_10_grpo.jsonl` 是**内容等价副本**。独立命名的目的不是改数据，而是为了隔离不同实验的输入路径与命名。

### 2.2 Prompt 修正

与 `Exp02` 一样，本实验使用：

- `examples/format_prompt/medical_evidence.jinja`

从而保证模型遵循数据样本中自带的 `<think> + <evidence>` 格式说明，而不是被错误模板引导输出 `\boxed{}`。

---

## 3. 算法配置

### 3.1 相对 Exp02 新增的 PAPO 开关

| 配置项 | Exp02 | Exp03 |
|---|---|---|
| `algorithm.adv_estimator` | `grpo` | `grpo` |
| `algorithm.online_filtering` | `false` | `false` |
| `algorithm.use_kl_prcp` | `false` | `true` |
| `algorithm.kl_prcp_coef` | `-` | `0.01` |
| `algorithm.use_aug_entropy_loss` | `false` | `true` |
| `algorithm.aug_entropy_loss_coef` | `-` | `0.03` |
| `algorithm.use_ori_entropy_loss` | `false` | `true` |
| `algorithm.ori_entropy_loss_coef` | `-` | `0.03` |

### 3.2 Loss 结构解释

本实验的优化目标可理解为：

```text
L_total ≈ L_pg
        + kl_prcp_coef × L_kl_prcp
        + aug_entropy_loss_coef × L_aug_entropy
        + ori_entropy_loss_coef × L_ori_entropy
```

因此在 TensorBoard 中，预期会出现 `Exp02` 没有的标量：

- `actor/kl_prcp_loss`
- `actor/aug_entropy_loss`
- `actor/ori_entropy_loss`

如果只盯着 `actor/pg_loss` 或 `reward/overall`，会误以为两者“几乎一样”；但 PAPO 的差异实际上体现在**附加正则项**而非 reward function 本身。

---

## 4. 关键解释：为什么 reward 可能和 GRPO 很像

本实验与 `Exp02` 的以下部分完全一致：

- reward function：`medical_evidence.py`
- validation set：同一 400 条
- 训练数据内容：当前等价
- 初始模型：同一 Lingshu-7B

因此短期内若观察到：

- `val/reward_score` 接近
- `reward/overall` 接近
- `parse_ok`、`bbox_iou` 曲线相近

这是**正常现象**，并不代表 PAPO 没生效。

真正需要对比的是：

1. `actor/kl_prcp_loss` 是否持续存在  
2. `actor/aug_entropy_loss` / `actor/ori_entropy_loss` 是否稳定  
3. 中后期 `val/overall_reward`、`reward/bbox_iou` 是否出现分叉  

---

## 5. 有效性说明

和 `Exp02` 一样，本实验也必须基于修正后的医学 Prompt 模板重新运行。修正前的旧日志由于使用了错误模板，不应作为 `GRPO+PAPO` 的正式实验结果。

因此，本实验的有效结果应满足：

- Prompt 模板为 `medical_evidence.jinja`
- 模型输出主要为 `<evidence>{...}</evidence>`
- TensorBoard 中明确可见 PAPO 相关 loss tag

---

## 6. 文件清单

| 文件 | 说明 |
|---|---|
| `examples/ours_medical/qwen2_5_vl_7b_grpo_papo_4xH200.sh` | GRPO + PAPO 启动脚本 |
| `examples/format_prompt/medical_evidence.jinja` | 医学任务专用 Prompt 模板 |
| `examples/reward_function/medical_evidence.py` | 医学 evidence reward |
| `data/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_grpo_papo.jsonl` | GRPO+PAPO 独立数据入口 |
| `data/eval/benchmark_medreasoner_evidence_eval_400_balanced_papo.jsonl` | 验证集 |
