# Batch Size 配置單位理解

| 項目 | 內容 |
|------|------|
| **筆記名稱** | RL 訓練中 Prompt / Sequence 兩種計算單位 |
| **日期** | 2026-05-25 |
| **核心問題** | 各 batch size 配置的單位是 prompt 還是 sequence？外循環和內循環各代表什麼？ |
| **代碼範圍** | `examples/config.yaml`、`verl/trainer/ray_trainer.py` |

---

## 兩個計算單位

| 術語 | 定義 |
|------|------|
| **Prompt** | 一條輸入問題，對應一個「組」（group） |
| **Sequence** | 一條模型生成的回答，1 個 prompt 對應 n=5 條 sequence |

---

## 配置參數單位對照

| 配置參數 | 單位 | 值 | 說明 |
|---|---|---|---|
| `rollout_batch_size` | **prompts** | 384 | 一個 step 目標收集的有效 prompt 數（DAPO 過濾後） |
| `mini_rollout_batch_size` | **prompts** | 128 | vLLM 生成時的分批大小，純顯存管理，384/128=3 批送給 vLLM |
| `worker.rollout.n` | — | 5 | 每個 prompt 生成幾條 response |
| `global_batch_size` | **prompts** | 128 | 每次 optimizer.step() 消費的 prompt 數 |
| `micro_batch_size_per_device_for_update` | **sequences** | 4 | 每張卡每次 forward+backward 處理的 sequence 數 |
| `micro_batch_size_per_device_for_experience` | **sequences** | 16 | 重算 log_prob（無梯度）時的 batch 大小，可比 update 大 |

**核心規律：高層邏輯用 prompt 計數；GPU 執行層用 sequence 計數。**

---

## 一個 Training Step 的完整計算鏈

```
【生成階段】
rollout_batch_size = 384 prompts
vLLM 分 3 批（mini_rollout=128）生成
每個 prompt × n=5 response = 1920 sequences 總量

【DAPO 過濾】
丟掉「全對」或「全錯」的 prompt 組
保留有對有錯的 prompts，補充直到 ≥ 384

【更新階段】
外循環（Train mini-batches）= 1920 / (128×5) = 3 次 optimizer.step()
    每次消費：128 prompts × 5 = 640 sequences

    內循環（Update policy / 梯度累積）= 640 / (4卡 × 4 sequences/卡) = 40 次
        每次 forward+backward：4 sequences/卡 × 4 卡 = 16 sequences
        40 次累積完梯度 → 1 次 optimizer.step()
```

---

## 模型實際更新次數

| 週期 | 模型更新次數 |
|------|------|
| 1 個 training step | **3 次**（外循環 3 次 optimizer.step()） |
| 40 次內循環 | **0 次更新**（純梯度累積，等效於一次處理 640 sequences） |
| 690 個 training steps | **2070 次** |

---

## 為什麼高層用 Prompt 計數？

GRPO/DAPO 的 advantage 在 **group 內** 歸一化（同一 prompt 的 5 條 response 互相比較）。
必須保持同一 prompt 的所有 response 在同一個 mini-batch 裡，所以外層以 prompt 為粒度管理，到 GPU 執行時才展開成 sequence。

---

## 梯度累積的本質

顯存放不下 640 個 sequences 同時做 backward，拆成 40 份、每份 16 個 sequences 分批計算梯度、累加在一起。**效果等價於一次性用 640 個 sequences 更新，顯存佔用只需 1/40。**

```python
# 偽代碼
for mini_batch in 3:           # 外循環：3 次真實更新
    optimizer.zero_grad()
    for micro_batch in 40:     # 內循環：梯度累積
        loss.backward()        # 只積累，不更新
    optimizer.step()           # 真實更新：這裡才動模型權重
```
