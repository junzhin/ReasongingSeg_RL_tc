# Experiment 02: GRPO Full-Data Prompt-Fixed Baseline

**创建日期**：2026-05-27  
**实验标识**：`exp02_grpo_full_data_prompt_fixed`  
**对应训练脚本**：`examples/ours_medical/qwen2_5_vl_7b_grpo_4xH200.sh`

---

## 1. 实验动机

本实验用于建立一个**有效的医学 evidence 定位 GRPO 基线**。

在此前的医学 RL 训练中，虽然训练流程本身可以正常跑通，但 Prompt 模板错误地复用了数学任务模板 `math_perception.jinja`。该模板强制模型以 `\boxed{}` 输出答案，而医学 reward `medical_evidence.py` 只接受：

```text
<evidence>{"num_boxes":N,"boxes":[{"bbox":[x1,y1,x2,y2]}]}</evidence>
```

因此旧版实验会出现：

- 模型按模板输出 `\boxed{5}` 等文本
- reward parser 视为格式错误
- `overall_reward` / `reward_score` 长时间卡在 `-0.2`

本实验的目标是：

1. 修正 Prompt-Reward 格式不一致问题  
2. 在完整 80/10/10 医学训练集上建立纯 `GRPO` 对照基线  
3. 为后续 `GRPO+PAPO`、`DAPO+PAPO` 提供可比较的参考点

---

## 2. 数据与 Prompt 配置

### 2.1 训练/验证数据

| 项目 | 文件 | 说明 |
|---|---|---|
| 训练集 | `data/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_grpo.jsonl` | 26,731 条 |
| 验证集 | `data/eval/benchmark_medreasoner_evidence_eval_400_balanced_papo.jsonl` | 400 条平衡验证集 |

当前 `train_cepo_lite_80_10_10_grpo.jsonl` 是从 `train_cepo_lite_80_10_10_papo.jsonl` 复制出的**等价数据副本**，目的是给 GRPO 实验保留独立输入路径，避免后续不同算法实验互相覆盖命名。

### 2.2 Prompt 修正

**旧版错误配置**：

- `examples/format_prompt/math_perception.jinja`
- 会额外追加 `Then, provide your final answer enclosed within \boxed{}` 指令

**新版正确配置**：

- `examples/format_prompt/medical_evidence.jinja`
- 内容仅为：

```jinja
{{ content | trim }}
```

也就是说，模板只透传数据样本中自带的医学输出说明，不再额外引入 `\boxed{}` 干扰。

---

## 3. 算法配置

### 3.1 本实验核心开关

| 配置项 | 值 |
|---|---|
| `algorithm.adv_estimator` | `grpo` |
| `algorithm.online_filtering` | `false` |
| `algorithm.use_kl_prcp` | `false` |
| `algorithm.use_aug_entropy_loss` | `false` |
| `algorithm.use_ori_entropy_loss` | `false` |
| `algorithm.disable_kl` | `true` |
| `algorithm.use_kl_loss` | `false` |

### 3.2 训练超参

| 超参 | 值 |
|---|---|
| GPU | `4 × H200` |
| Epochs | `10` |
| `rollout_batch_size` | `384` |
| `mini_rollout_batch_size` | `128` |
| `global_batch_size` | `128` |
| `max_prompt_length` | `4096` |
| `save_freq` / `val_freq` | `2 / 2` |

---

## 4. 与其他实验的关系

> **Exp02 vs Exp03**：训练数据、reward function、Prompt 模板、超参数完全一致；唯一受控变量是 PAPO 附加 loss 是否开启。  
> **Exp02 vs Exp04**：Exp02 为纯 GRPO；Exp04 在此基础上加入 DAPO 的 online filtering，并同时保留 PAPO 项。

本实验应在 TensorBoard 中呈现：

- 有 `actor/pg_loss`
- 有 `actor/entropy_loss`
- **没有** `actor/kl_prcp_loss`
- **没有** `actor/aug_entropy_loss`
- **没有** `actor/ori_entropy_loss`

---

## 5. 有效性说明

本实验是对旧版无效医学训练的**修正版重跑基线**。旧版使用错误 Prompt 模板，结果不应与本实验混合分析。

因此：

- 旧版 `medical_evidence__grpo__7b__...` 日志只能作为“发现 Prompt mismatch 问题”的排错记录
- 本实验重新运行后产出的日志与 TensorBoard 曲线，才可用于正式比较

---

## 6. 文件清单

| 文件 | 说明 |
|---|---|
| `examples/ours_medical/qwen2_5_vl_7b_grpo_4xH200.sh` | 纯 GRPO 启动脚本 |
| `examples/format_prompt/medical_evidence.jinja` | 医学任务专用 Prompt 模板 |
| `examples/reward_function/medical_evidence.py` | 医学 evidence reward |
| `data/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_grpo.jsonl` | GRPO 实验独立数据入口 |
| `data/eval/benchmark_medreasoner_evidence_eval_400_balanced_papo.jsonl` | 验证集 |
