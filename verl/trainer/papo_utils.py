# Copyright 2025 - PAPO
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# PAPO 核心创新：图像随机 Patch 遮掩（Patch Blackening）
#
# 【为什么需要这个函数？】
# 在医学图像理解任务中，模型有可能"走捷径"——例如靠图像背景纹理、
# 图像边缘区域、或某个固定位置的视觉特征来猜答案，而不是真正理解
# 图像内容。这种捷径在训练集上有效，但在新样本上会失败。
#
# 解决方案：在训练时生成两份图像——一份原图，一份随机遮掩版。
# 让模型对两份图像的输出尽量一致（通过 KL 散度约束）。
# 这样模型就无法依赖某个特定区域，被迫学习整体语义理解。
#
# 【patch_size=14 的含义】
# Qwen2.5-VL 的视觉编码器使用 14×14 像素作为基本处理单元（patch）。
# 遮掩粒度对齐这个单元，确保遮掩后的图像在视觉 token 层面产生有意义的干扰，
# 而不是次像素级的噪声（视觉编码器感知不到）。
#
# 【black_prob=0.6 的含义】
# 每个 patch 有 60% 概率被遮掉。这个比例经过调参——
# 太低（如10%）干扰太小，KL 损失接近0，约束失效；
# 太高（如90%）图像几乎全黑，模型输出差异太大，KL 损失主导训练，
# 反而压制了 RL reward 信号。60% 是两者的平衡点。
# ──────────────────────────────────────────────────────────────────────────────
def random_patch_blackening(pil_img, patch_size=14, black_prob=0.6):
    """Randomly blacken square patches in a PIL image."""
    img = np.array(pil_img).astype(np.float32)
    h, w = img.shape[:2]

    # 以 patch_size 为步长扫描整张图，每个 14×14 块独立决定是否遮掉
    # 遮掩是"随机"的，每次前向传播遮掩位置不同，防止模型记住固定模式
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if np.random.rand() < black_prob:
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                # 将该 patch 内所有像素置零（对 RGB 三通道都置零）
                # 置零而非高斯噪声：零值在归一化后是一个明确的"无信息"信号
                if img.ndim == 3:
                    img[y:y_end, x:x_end, :] = 0
                else:
                    img[y:y_end, x:x_end] = 0

    # 返回值是遮掩后的 PIL 图像，后续会被视觉编码器处理成 token
    # 在 ray_trainer.py 的 _aug_img_for_kl_prcp() 中被调用
    return Image.fromarray(img.astype(np.uint8))
