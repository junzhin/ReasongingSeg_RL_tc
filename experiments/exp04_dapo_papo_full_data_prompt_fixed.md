# Experiment 04: DAPO + PAPO Full-Data Prompt-Fixed

**创建日期**：2026-05-27  
**实验标识**：`exp04_dapo_papo_full_data_prompt_fixed`  
**对应训练脚本**：`examples/ours_medical/qwen2_5_vl_7b_dapo_papo_4xH200.sh`

---

## 1. 实验动机

本实验用于测试 `DAPO + PAPO` 在完整医学 80/10/10 训练集上的表现。

与纯 GRPO 相比，DAPO 的核心思想是：在组内 reward 全对或全错时，这组样本对优势归一化几乎不提供有效学习信号，因此训练时直接过滤掉这些组，保留“有区分度”的样本组进行更新。

在医学 evidence 任务里，这一策略的意义在于：

- 减少“整组全部格式错”或“整组全部空框”的无效更新
- 将有限训练步集中到真正有 reward 差异的样本组上
- 与 PAPO 的视觉一致性约束结合，尝试同时改善稳定性与视觉对齐

---

## 2. 数据与 Prompt 配置

### 2.1 训练/验证数据

| 项目 | 文件 | 说明 |
|---|---|---|
| 训练集 | `data/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_papo.jsonl` | 26,731 条 |
| 验证集 | `data/eval/benchmark_medreasoner_evidence_eval_400_balanced_papo.jsonl` | 400 条 |

训练集构成为：

| sample_type | 占比 |
|---|---|
| `original` | 80% |
| `background_perturbed` | 10% |
| `evidence_deleted` | 10% |

### 2.2 Prompt 修正

本实验同样使用：

- `examples/format_prompt/medical_evidence.jinja`

因此也属于修正后的有效医学实验版本，而非旧版错误 Prompt 结果。

---

## 3. 算法配置

### 3.1 DAPO 与 PAPO 关键开关

| 配置项 | 值 |
|---|---|
| `algorithm.adv_estimator` | `dapo` |
| `algorithm.online_filtering` | `true` |
| `algorithm.filter_key` | `overall` |
| `algorithm.filter_low` | `0.01` |
| `algorithm.filter_high` | `0.99` |
| `algorithm.disable_kl` | `true` |
| `algorithm.use_kl_loss` | `false` |
| `algorithm.kl_prcp_coef` | `0.01` |
| `algorithm.use_aug_entropy_loss` | `true` |
| `algorithm.use_ori_entropy_loss` | `true` |

### 3.2 一个容易忽略的点

脚本里没有显式写：

```text
algorithm.use_kl_prcp=true
```

但由于公共配置 `examples/config.yaml` 默认开启了 `use_kl_prcp: true`，因此本实验**仍然是 PAPO 开启状态**。脚本额外覆盖的是：

- `kl_prcp_coef`
- `use_aug_entropy_loss`
- `use_ori_entropy_loss`

换言之，本实验实际上是：

```text
DAPO + PAPO + 双路 entropy regularization
```

---

## 4. 与 Exp02 / Exp03 的核心区别

| 实验 | 优势估计 | Online Filtering | PAPO |
|---|---|---|---|
| `Exp02` | `grpo` | `false` | `false` |
| `Exp03` | `grpo` | `false` | `true` |
| `Exp04` | `dapo` | `true` | `true` |

因此 `Exp04` 相对 `Exp03` 多出来的不是新的 reward function，而是：

1. 使用 `dapo` advantage estimator  
2. 对 `overall` 过低或过高的组做在线过滤  
3. 通过 `max_try_make_batch=50` 允许反复补样，尽量凑满 batch

预期表现为：

- 更少无效梯度更新
- 更高的训练时间波动（因为 filtering 后需要补 batch）
- 如果过滤有效，`reward/overall` 与 `bbox_iou` 后期可能更稳

---

## 5. 有效性说明

和 `Exp02` / `Exp03` 一样，本实验也应视为对旧版医学训练的**修正后重跑版本**。旧版由于 Prompt 模板错误，不应作为 DAPO+PAPO 的正式对比结果。

本实验的观察重点包括：

- `reward/overall` 是否摆脱长期 `-0.2` 格式失败状态
- `val/parse_ok_reward` 是否明显提升
- DAPO filtering 后训练是否出现更稳定的 `bbox_iou` 增长

---

## 6. 文件清单

| 文件 | 说明 |
|---|---|
| `examples/ours_medical/qwen2_5_vl_7b_dapo_papo_4xH200.sh` | DAPO + PAPO 启动脚本 |
| `examples/format_prompt/medical_evidence.jinja` | 医学任务专用 Prompt 模板 |
| `examples/reward_function/medical_evidence.py` | 医学 evidence reward |
| `data/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_papo.jsonl` | 原始 80/10/10 医学训练集 |
| `data/eval/benchmark_medreasoner_evidence_eval_400_balanced_papo.jsonl` | 验证集 |
