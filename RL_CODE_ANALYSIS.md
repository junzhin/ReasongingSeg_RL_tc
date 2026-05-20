# ReasongingSeg RL 训练代码完整分析

## 1. 项目概览

这是一个基于 **VERL**（Versatile Reinforcement Learning）框架的医学影像证据定位 RL 训练管线，
从字节跳动的 PAPO（Perception-aware Policy Optimization）项目裁剪而来。

**核心任务**：给定医学影像（CT/MR/X光）和自然语言问题，模型输出 `<think>` 推理过程 + `<evidence>` 边界框坐标。

**训练目标**：通过 RL（GRPO/DAPO）优化模型在医学影像上定位解剖结构/病变的能力，
奖励信号来自边界框 IoU/Dice 匹配。

---

## 2. 目录结构

```
ReasongingSeg_RL/
├── examples/                          # 训练脚本和配置
│   ├── config.yaml                    # 主训练配置（完整参数）
│   ├── runtime_env.yaml               # Ray 运行时环境变量
│   ├── papo_grpo/                     # GRPO 训练脚本
│   │   ├── qwen2_5_vl_7b_grpo_papo.sh # 7B GRPO + PAPO 数据
│   │   ├── qwen2_5_vl_3b_grpo_papo.sh # 3B GRPO + PAPO 数据
│   │   ├── qwen2_5_vl_7b_grpo.sh      # 7B GRPO + 原始数据
│   │   └── ...                        # 更多变体
│   ├── papo_dapo/                     # DAPO 训练脚本
│   ├── format_prompt/
│   │   └── math_perception.jinja      # Jinja2 prompt 模板
│   └── reward_function/
│       └── medical_evidence.py        # 核心奖励函数（234行）
│
├── verl/                              # VERL RL 框架
│   ├── trainer/
│   │   ├── main.py                    # 训练入口点
│   │   ├── ray_trainer.py             # Ray 分布式训练编排（861行）
│   │   ├── core_algos.py              # 核心 RL 算法（GRPO/DAPO/GAE等）
│   │   ├── config.py                  # 配置数据类
│   │   ├── data_loader.py             # 数据加载器创建
│   │   ├── metrics.py                 # 训练指标计算
│   │   └── papo_utils.py              # PAPO 特有工具（图像增强）
│   ├── workers/
│   │   ├── actor/dp_actor.py          # DataParallel Actor（416行）
│   │   ├── rollout/vllm_rollout_spmd.py # vLLM rollout 推理
│   │   ├── reward/function.py         # 奖励函数管理器
│   │   ├── critic/dp_critic.py        # Critic 模型（GAE用）
│   │   └── sharding_manager/          # FSDP/vLLM 权重分片
│   ├── models/
│   │   ├── monkey_patch.py            # Qwen2-VL 模型 monkey patch
│   │   └── transformers/qwen2_vl.py   # Qwen2-VL 前向传播重写
│   ├── single_controller/ray/         # Ray 分布式 Worker 管理
│   └── utils/
│       ├── dataset.py                 # RLHF 数据集类
│       ├── tokenizer.py               # Tokenizer/Processor 加载
│       ├── torch_functional.py        # PyTorch 工具函数
│       └── checkpoint/                # 检查点管理
│
├── data/
│   ├── rl_3_evidence_papo_jsonl/      # 训练数据（26731条）
│   ├── eval/                          # 验证数据
│   └── images/                        # 图片（26204张，repo相对路径）
│
├── models/Lingshu-7B/                 # 预训练模型权重
└── scripts/
    ├── install.sh                     # 依赖安装脚本
    └── submit_cepo_lite_7b_reserved.sbatch  # Slurm 提交模板
```

---

## 3. 训练数据格式

### 3.1 数据结构

训练数据为 JSONL 格式，每行一个样本，共 **26731** 条。

```json
{
  "id": "umrg14k__train__001208__...",
  "problem": "<image>\nWhat might be the central part of the lower brain...\n\nPlease answer with:\n<think>...</think>\n<evidence>{...}</evidence>\n\nRules:\n- Use normalized xyxy integer coordinates in [0,1000].\n- ...",
  "images": ["data/images/sft_qwen_final/..."],
  "answer": {
    "sample_type": "original",
    "ground_truth": {
      "box_format": "xyxy_norm_1000",
      "num_boxes": 1,
      "boxes": [{"label": "brainstem", "bbox": [469, 563, 526, 608]}],
      "target_boxes": [{"label": "brainstem", "bbox": [469, 563, 526, 608]}]
    }
  },
  "meta": {
    "dataset": "u_mrg_14k",
    "modality": "ct",
    "category": "brainstem",
    "problem_type": "reasoning_grounding",
    "cf_info": {"type": "none", "method": "none", "params": {}}
  }
}
```

**关键字段**：
- `problem`：完整的用户 prompt，包含 `<image>` 占位符和输出格式指令
- `images`：图片路径列表（repo相对路径 `data/images/...`）
- `answer.sample_type`：样本类型（见下文）
- `answer.ground_truth.boxes`：需要定位的主要边界框（用于正向奖励）
- `answer.ground_truth.target_boxes`：counterfactual 样本中被修改的目标框（用于惩罚）
- `meta.cf_info`：counterfactual 扰动信息

### 3.2 三种样本类型

| 类型 | 数量 | 说明 |
|------|------|------|
| `original` | 21,385 (80%) | 正常样本，模型需正确预测边界框 |
| `background_perturbed` | 2,673 (10%) | 背景被高斯模糊扰动的样本（仍含目标），测试鲁棒性 |
| `evidence_deleted` | 2,673 (10%) | 目标已被删除的样本，模型应输出空框 |

这种 80/10/10 划分是 PAPO 方法的核心设计——通过 counterfactual 样本让模型学会"不确定时不说"。

### 3.3 Prompt 格式

Prompt 通过 Jinja2 模板 `math_perception.jinja` 渲染：

```
{{ content | trim }}

You first think through the reasoning process as an internal monologue, enclosed within <think> </think> tags. Then, provide your final answer enclosed within \boxed{}.
```

注意：这里的模板文本说 `\boxed{}`，但实际训练数据中的 prompt 直接指定了 `<evidence>` 格式。实际模型输出的解析使用的是 `<evidence>...</evidence>` 标签，而不是 `\boxed{}`。这是因为此模板继承自数学推理任务，医学任务在 problem 字段中直接指定了输出格式。

---

## 4. 奖励函数设计 (`medical_evidence.py`)

奖励函数是整个 RL 训练的核心，共 234 行，实现了基于边界框匹配的奖励计算。

### 4.1 解析模型输出

```python
def parse_evidence(response: str) -> Dict[str, Any]:
    content = _tag_content(response, "<evidence>")  # 提取 <evidence> 标签内容
    payload = json.loads(content)                     # 解析 JSON
    # 验证格式：boxes 数组 + num_boxes 一致性
    # 验证每个 bbox：4个坐标，范围 [0,1000], x1<x2, y1<y2
```

### 4.2 奖励计算逻辑

```python
def _score_one(reward_input):
    # 1. 解析失败 → overall = -0.2 (惩罚)
    if not parse_ok:
        return {"overall": -0.2, ...}

    # 2. evidence_deleted 样本（目标已被删除）
    if sample_type == "evidence_deleted":
        if pred_num == 0:    # 正确输出空框
            overall = 0.5     # 中等奖励
        else:                 # 错误输出了框
            overall = -0.1 - 0.4 * max_iou_to_target  # 惩罚，框越准罚越重

    # 3. GT有框但预测无框
    if gt_num > 0 and pred_num == 0:
        overall = -0.2  # 漏检惩罚

    # 4. 正常样本：IoU/Dice 匹配
    bbox_iou = greedy_matched_iou(pred_boxes, gt_boxes)
    overall = 0.90 * bbox_iou + 0.10 * num_ok
```

### 4.3 匹配算法

- **贪心匹配**：对每个 GT 框，找未匹配的预测框中 IoU 最大的
- **IoU (Intersection over Union)**：`inter / union`
- **Dice 系数**：`2*inter / (area_a + area_b)`
- 输出 6 个指标：`overall`, `bbox_iou`, `bbox_dice`, `parse_ok`, `num_ok`, `empty_rate`

### 4.4 奖励类型

配置中使用 `reward_type=batch`，对应 `BatchFunctionRewardManager`：
- 对整个 batch 的响应一起调用 `compute_score()`
- 奖励值放在每个序列的最后一个 token 位置（outcome supervision）
- 最终只用一个标量 `overall` 作为 token-level reward

---

## 5. 训练配置详解 (`config.yaml`)

### 5.1 数据配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `train_files` | `PAPOGalaxy/PAPO_ViRL39K_train` | HF dataset（shell 中覆盖） |
| `prompt_key` | `problem` | 数据中的 prompt 字段 |
| `answer_key` | `answer` | 数据中的答案字段 |
| `image_key` | `images` | 图片字段 |
| `max_prompt_length` | 4096 | prompt 最大长度 |
| `max_response_length` | 2048 | 模型回答最大长度 |
| `max_pixels` | 1003520 (1280×28×28) | 图片最大像素数 |
| `min_pixels` | 200704 (256×28×28) | 图片最小像素数 |
| `rollout_batch_size` | 512 | 每次 rollout 的样本数 |
| `mini_rollout_batch_size` | 128 | DataLoader 的 batch size |

### 5.2 算法配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `adv_estimator` | `dapo` | 优势估计算法（DAPO/GRPO/GAE） |
| `disable_kl` | `true` | 是否禁用 KL 散度（GRPO/DAPO不需要 ref 模型） |
| `use_kl_loss` | `true` | 是否在 loss 中使用 KL 惩罚 |
| `kl_coef` | 0.01 | KL 惩罚系数 |

#### KL-PRCP（Perception-aware Regularization via Contrastive Preference）

| 参数 | 值 | 说明 |
|------|-----|------|
| `use_kl_prcp` | `true` | 启用对比 KL 正则化 |
| `contrastive_type` | `augmented` | 对比类型：用增强图像 |
| `kl_prcp_coef` | 0.02 | 对比 KL 系数 |
| `kl_prcp_apply_mode` | `all` | 对所有样本应用 |
| `aug_config.patch_size` | 14 | 随机黑块大小 |
| `aug_config.black_prob` | 0.6 | 每块被涂黑的概率 |

#### 双熵损失

| 参数 | 值 | 说明 |
|------|-----|------|
| `use_aug_entropy_loss` | `true` | 对增强图像使用熵损失 |
| `aug_entropy_loss_coef` | 0.05 | 增强熵损失系数 |
| `use_ori_entropy_loss` | `true` | 对原始图像使用熵损失 |
| `ori_entropy_loss_coef` | 0.05 | 原始熵损失系数 |

#### 在线过滤（DAPO Over-Long Filtering）

| 参数 | 值 | 说明 |
|------|-----|------|
| `online_filtering` | `true` | 启用在线样本过滤 |
| `filter_key` | `overall` | 按 overall score 过滤 |
| `filter_low` | 0.01 | 去掉 score 太低的组 |
| `filter_high` | 0.99 | 去掉 score 太高的组（防止过拟合） |

### 5.3 Worker 配置

#### Actor

| 参数 | 值 | 说明 |
|------|-----|------|
| `global_batch_size` | 128 | PPO mini-batch size |
| `micro_batch_size_per_device_for_update` | 4 | 每 GPU 更新时 micro batch |
| `micro_batch_size_per_device_for_experience` | 16 | 每 GPU log_prob 计算 micro batch |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `padding_free` | `true` | 使用 padding-free 训练（flash_attn varien） |
| `freeze_vision_tower` | `false` | 不冻结视觉编码器 |
| `lr` | 1e-6 | 学习率 |
| `fsdp.torch_dtype` | `bf16` | bfloat16 训练 |
| `offload_params` | `true` | 参数 offload 到 CPU |

#### Rollout（vLLM 推理引擎）

| 参数 | 值 | 说明 |
|------|-----|------|
| `n` | 5 | 每个 prompt 采样 5 个回答 |
| `temperature` | 1.0 | 采样温度 |
| `top_p` | 0.99 | nucleus sampling |
| `tensor_parallel_size` | 2 | 张量并行度 |
| `gpu_memory_utilization` | 0.6 | vLLM 显存占用率 |
| `val_override_config.n` | 8 | 验证时采样 8 个回答 |

### 5.4 训练器配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `total_epochs` | 15 | 总训练轮数 |
| `nnodes` | 1 | 单机训练 |
| `n_gpus_per_node` | 8 | 8 GPU |
| `val_freq` | 5 | 每 5 步验证一次 |
| `save_freq` | 5 | 每 5 步保存检查点 |
| `save_limit` | 3 | 最多保留 3 个检查点 |

---

## 6. 核心算法详解

### 6.1 DAPO（Decoupled Alignment from Policy Optimization）

DAPO 是 GRPO 的改进版本。两者的核心区别在于优势计算的归一化方式：

**GRPO 优势计算**（`compute_grpo_outcome_advantage`）：
```python
# 对同一 prompt 的多个 rollout 做组内归一化
for each prompt group:
    mean = mean(scores_in_group)
    std = std(scores_in_group)
    advantage = (score - mean) / (std + eps)
```

**DAPO 优势计算**（`compute_dapo_outcome_advantage`）：
与 GRPO 相同的组内归一化逻辑，但 DAPO 额外引入了在线过滤（`online_filtering`）机制来去除过于简单或过于困难的样本。

两种算法的核心思想都是：**同一 prompt 的多个回答之间比较**，而不是学习绝对的价值函数。
因此都不需要 Critic 模型（`disable_kl=true`）。

### 6.2 KL-PRCP（Perception-aware Regularization via Contrastive Preference）

这是 PAPO 的核心创新，通过图像增强实现感知对比正则化。

**流程**：
1. **Rollout 阶段**：对原始图像生成回答
2. **增强阶段**：对同一图像进行随机 patch 涂黑（`random_patch_blackening`）
3. **计算 log_probs 差异**：
   - `old_log_probs`：原始图像下当前策略的 log 概率
   - `aug_log_probs`：增强图像下当前策略的 log 概率
4. **KL 散度作为正则项**：
   ```python
   kld_contrastive = KL(old_log_probs || aug_log_probs)
   # 加入 reward：奖励在图像扰动下仍保持一致预测的行为
   updated_rewards = current_rewards + kl_coef * kld_contrastive
   ```

**直觉**：如果模型在被涂黑的图像上仍然输出相似的 log_probs（KL 小），说明模型依赖的是全局语义而非局部噪声——这是好的行为，应该被奖励。

**增强方法**（`papo_utils.py`）：
```python
def random_patch_blackening(pil_img, patch_size=14, black_prob=0.6):
    # 将图像分成 14×14 的 patch
    # 每个 patch 以 60% 概率被涂黑
    # 模拟信息缺失，迫使模型学习鲁棒特征
```

### 6.3 双熵损失（Double Entropy Loss）

为防止策略坍缩（policy collapse），引入两种熵损失：

1. **原始熵损失**（`use_ori_entropy_loss`）：鼓励模型在原始图像上保持探索性
   ```python
   ori_entropy_loss = -masked_mean(log_probs, response_mask)
   ```

2. **增强熵损失**（`use_aug_entropy_loss`）：鼓励模型在增强图像上也保持探索性
   ```python
   aug_entropy_loss = -masked_mean(aug_log_probs, response_mask)
   ```

两者系数均为 0.05，通过 `discount_ratio` 与 KL-PRCP 的 annealing 联动。

### 6.4 策略梯度损失

最终的 actor loss 由以下部分组成：

```
pg_loss = clipped_policy_gradient_loss
        + kl_loss * kl_coef                          # KL 惩罚（可选）
        - kl_prcp_loss * kl_prcp_coef                 # 对比 KL 奖励
        + aug_entropy_coef * aug_entropy_loss         # 增强熵正则
        + ori_entropy_coef * ori_entropy_loss         # 原始熵正则
        + sft_coef * sft_loss                         # SFT 辅助损失（可选）
```

---

## 7. 训练工作流

### 7.1 入口点 (`verl/trainer/main.py`)

```python
def main():
    # 1. 从 CLI + YAML 合并配置
    cli_args = OmegaConf.from_cli()
    config = OmegaConf.merge(default_config, file_config, cli_args)

    # 2. 初始化 Ray 集群
    ray.init(runtime_env={...})

    # 3. 创建 Runner（Ray Actor），避免 driver 进程负载过重
    runner = Runner.remote()
    ray.get(runner.run.remote(config))
```

### 7.2 Runner.run() 流程

```python
class Runner:
    def run(self, config):
        # 1. 加载 tokenizer 和 processor（图像处理器）
        tokenizer = get_tokenizer(model_path)
        processor = get_processor(model_path)

        # 2. 定义 Worker 角色和资源池
        role_worker_mapping = {
            Role.ActorRolloutRef: FSDPWorker,  # Actor + Rollout + Ref 三合一
            Role.Critic: FSDPWorker,            # 仅 GAE 用
            Role.RefPolicy: FSDPWorker,         # 仅非 GRPO 用
        }

        # 3. 创建远程 Reward Manager
        reward_fn = RemoteRewardManager.remote(config, tokenizer)

        # 4. 创建 DataLoader
        train_dataloader, val_dataloader = create_dataloader(...)

        # 5. 初始化 RayPPOTrainer
        trainer = RayPPOTrainer(...)
        trainer.init_workers()
        trainer.fit()
```

### 7.3 训练主循环 (`RayPPOTrainer.fit()`)

```
每个 training step:

1. 生成 batch（_make_batch_data）
   ├─ 从 DataLoader 读取 mini-batch
   ├─ 如果启用 KL-PRCP：生成增强图像
   ├─ vLLM rollout 生成回答（每个 prompt 生成 n=5 个回答）
   ├─ 如果启用 online_filtering：
   │  ├─ 计算 reward
   │  ├─ 按 uid 分组求平均 score
   │  └─ 过滤 score 在 [0.01, 0.99] 之外的组
   └─ 累积直到达到 rollout_batch_size

2. 序列平衡（_balance_batch）
   └─ 按序列长度重新排序，让各 GPU 负载均衡

3. 计算 reward
   ├─ 调用 reward_fn.compute_reward(batch)
   └─ reward 放在每个序列最后一个有效 token 位置

4. 计算 old_log_probs（当前策略对旧回答的 log 概率）

5. 计算 aug_log_probs（对增强图像的 log 概率）

6. [可选] 计算 ref_log_probs（参考模型的 log 概率）

7. 计算优势函数（DAPO/GRPO）
   └─ 组内归一化：(score - group_mean) / group_std

8. 更新 Actor（update_policy）
   ├─ 多轮 PPO epoch
   ├─ 计算 policy gradient loss + KL loss + KL-PRCP loss + entropy losses
   └─ FSDP 梯度累积 + 反向传播

9. [可选] 更新 Critic（仅 GAE）

10. 验证（每 val_freq 步）
    └─ 在验证集上 rollout + 计算 reward

11. 保存检查点（每 save_freq 步）
```

### 7.4 vLLM Rollout 流程

```python
class vLLMRollout:
    def generate_sequences(self, prompts):
        # 1. 处理多模态数据（图片 resize + 转 RGB）
        multi_modal_data = _process_multi_modal_data(...)

        # 2. 构建 vLLM 输入 {"prompt_token_ids": [...], "multi_modal_data": {...}}
        vllm_inputs = [...]

        # 3. vLLM 批量推理
        completions = self.inference_engine.generate(vllm_inputs, sampling_params)

        # 4. 后处理：拼接 prompt+response, 构建 attention_mask, position_ids
        return DataProto({prompts, responses, input_ids, attention_mask, ...})
```

vLLM 引擎设置了 `enable_sleep_mode=True`，在 rollout 完成后会 offload 模型参数释放显存。

### 7.5 模型 Monkey Patch (`monkey_patch.py`)

为了支持 Ulysses 序列并行和 flash attention，对 Qwen2-VL 模型做了以下修改：

1. **Attention 前向传播替换**：用自定义的 `qwen2_vl_attn_forward` 替换原始 attention，支持 flash attention varien
2. **模型前向传播替换**：根据 transformers 版本选择不同的 forward 实现
   - 新版 (>=4.52.0)：`qwen2_vl_forward_new`——先通过 visual encoder 获取 image embeddings，再送入 language model
   - 旧版：`qwen2_vl_forward_old`——手动做 embeddings 的 masked_scatter
3. **位置编码**：实现 3D RoPE（mrope），为图像 token 分配 3 维位置编码 (t, h, w)

---

## 8. GRPO vs DAPO 训练脚本对比

### GRPO 训练脚本（`qwen2_5_vl_7b_grpo_papo.sh`）

```bash
CUDA_IDS=0,1,2,3
N_GPU=4
TOTAL_EPOCHES=2
ROLLOUT_BATCH_SIZE=384
GLOBAL_BATCH_SIZE=128

# 启用 KL-PRCP
KL_PRCP_COEF=0.02

# 启用双熵损失
USE_AUG_ENTROPY_LOSS=true
AUG_ENTROPY_LOSS_COEF=0.05
USE_ORI_ENTROPY_LOSS=true
ORI_ENTROPY_LOSS_COEF=0.05

python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    worker.actor.model.model_path=${MODEL_PATH} \
    algorithm.kl_prcp_coef=${KL_PRCP_COEF} \
    algorithm.use_aug_entropy_loss=${USE_AUG_ENTROPY_LOSS} \
    ...
```

### DAPO 训练脚本（`qwen2_5_vl_7b_dapo_papo.sh`）

关键区别：
- `adv_estimator=dapo`（在 config.yaml 中设定）
- 启用 `online_filtering=true`
- 可选 `overlong_buffer` 机制（处理超长序列）

### 带/不带 KL Ref 的变体

- `_no_kl_ref.sh`：设置 `algorithm.use_kl_loss=false`，不计算 ref_log_probs
- 这是充分消融实验的标准做法

---

## 9. 数据流总结

```
JSONL 文件
  │
  ▼
RLHFDataset.__getitem__()
  ├─ 读取 problem + images
  ├─ 应用 Jinja2 格式模板
  ├─ Processor 处理图片 + 文本 → input_ids, pixel_values, image_grid_thw
  ├─ Qwen2-VL mrope position_ids
  ├─ 左侧 padding 到 max_prompt_length
  └─ 返回 {input_ids, attention_mask, position_ids, raw_prompt_ids, ground_truth, multi_modal_data}
  │
  ▼
DataLoader (StatefulDataLoader, batch_size=mini_rollout_batch_size)
  │
  ▼
RayPPOTrainer._make_batch_data()
  ├─ [可选] 生成增强图像 (KL-PRCP)
  ├─ vLLM rollout → responses
  ├─ [可选] online_filtering (DAPO)
  └─ 重复 n=5 次，累积到 rollout_batch_size
  │
  ▼
Trainer.fit() 主循环
  ├─ compute reward (via Ray remote call)
  ├─ compute old_log_probs
  ├─ compute aug_log_probs
  ├─ compute advantages (DAPO/GRPO)
  └─ update_actor (policy gradient)
```

---

## 10. 如何运行

### 10.1 单机直接运行

```bash
cd /path/to/ReasongingSeg_RL
conda activate papo

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=data/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_papo.jsonl \
    data.val_files=data/eval/benchmark_medreasoner_evidence_eval_400_balanced_papo.jsonl \
    data.rollout_batch_size=384 \
    data.format_prompt=examples/format_prompt/math_perception.jinja \
    worker.actor.model.model_path=models/Lingshu-7B \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=128 \
    trainer.experiment_name=my_exp \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=2 \
    worker.reward.reward_function=examples/reward_function/medical_evidence.py:compute_score \
    data.max_prompt_length=4096 \
    algorithm.kl_prcp_coef=0.02 \
    algorithm.use_aug_entropy_loss=true \
    algorithm.aug_entropy_loss_coef=0.05 \
    algorithm.use_ori_entropy_loss=true \
    algorithm.ori_entropy_loss_coef=0.05
```

### 10.2 Slurm 提交

```bash
sbatch scripts/submit_cepo_lite_7b_reserved.sbatch
```

### 10.3 关键环境变量

```yaml
TOKENIZERS_PARALLELISM: "true"
NCCL_DEBUG: "WARN"
VLLM_LOGGING_LEVEL: "WARN"
TORCH_NCCL_AVOID_RECORD_STREAMS: "1"
PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:False"
```

---

## 11. 关键设计要点

1. **Counterfactual 样本训练**：10% `background_perturbed` + 10% `evidence_deleted` 让模型学会区分"看不到"和"不确定"
2. **KL-PRCP 惩罚机制**：对 `evidence_deleted` 样本中模型错误地预测了框的行为，惩罚力度与预测框和已删除目标的 IoU 成正比（`-0.1 - 0.4 * max_iou`）
3. **Outcome supervision**：奖励只放在序列最后一个 token，不进行逐 token 奖励
4. **Group-based advantage**：GRPO/DAPO 在同一 prompt 的 n=5 个回答间做组内归一化，消除 prompt 难度差异
5. **Padding-free 训练**：使用 flash_attn 的 varien 模式去除 padding token，提高训练效率
6. **vLLM sleep mode**：rollout 完成后 offload 模型到 CPU，释放 GPU 显存给 FSDP 训练
