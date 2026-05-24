# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .core_algos import AdvantageEstimator, FixedKLController, KLController, compute_kl, get_kl_controller
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics

from PIL import Image
from ..utils.dataset import collate_fn
from .papo_utils import random_patch_blackening


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    token_level_rewards = token_level_scores - kl_ctrl.kl_coef * kld
    if "token_level_rewards" in data.batch:
        data.batch.set_("token_level_rewards", token_level_rewards)
    else:
        was_locked = data.batch.is_locked
        if was_locked:
            data.batch.unlock_()
        data.batch["token_level_rewards"] = token_level_rewards
        if was_locked:
            data.batch.lock_()

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics

# ──────────────────────────────────────────────────────────────────────────────
# PAPO 感知 KL（kl_prcp / contrastive KL）注入
#
# 【核心思想】
# PAPO 想让模型对"原图"和"随机遮掩图"的输出分布尽量一致——
# 若模型真正理解图像语义，遮掉部分 patch 不应大幅改变它的回答分布；
# 若模型靠局部捷径作答，遮掩会让分布剧烈漂移。把这个漂移（KL 散度）
# 当作惩罚加进 reward，就能逼模型学整体语义、抗遮掩。
#
# 【实现方式】
# 用 old_log_probs（原图前向）和 aug_log_probs（遮掩图前向，由 papo_utils
# 的 random_patch_blackening 生成）算逐 token KL，乘 response_mask 后
# 按系数 kl_coef 累加到 token_level_rewards 上。系数可随训练步数 annealing。
# GRPO baseline 不调用此函数（仅在 use_kl_prcp=true 时启用）。
# ──────────────────────────────────────────────────────────────────────────────
def apply_kl_contrastive(
    data: DataProto,
    kl_ctrl_contrastive: core_algos.KLController,
    kl_penalty_contrastive="kl",
    kl_prcp_apply_mode="all",
):
    # not used in GRPO
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    if kl_prcp_apply_mode == "correct_only":
        raise NotImplementedError("correct_only mode is not implemented yet.")

    # 原图 log_prob vs 遮掩图 log_prob 的逐 token KL 散度 = 感知漂移量
    kld_contrastive = core_algos.compute_kl(
        data.batch["old_log_probs"], data.batch["aug_log_probs"], kl_penalty=kl_penalty_contrastive
    )
    kld_contrastive = kld_contrastive * response_mask
    # 把感知 KL 按系数加进 token_level_rewards：漂移越大，惩罚越大
    if "token_level_rewards" in data.batch:
        current_rewards = data.batch["token_level_rewards"]
        updated_rewards = current_rewards + kl_ctrl_contrastive.kl_coef * kld_contrastive
        data.batch.set_("token_level_rewards", updated_rewards)
    else:
        was_locked = data.batch.is_locked
        if was_locked:
            data.batch.unlock_()
        data.batch["token_level_rewards"] = kl_ctrl_contrastive.kl_coef * kld_contrastive
        if was_locked:
            data.batch.lock_()
    
    current_kl_contrastive = VF.masked_mean(kld_contrastive, mask=response_mask, dim=-1)
    current_kl_contrastive = torch.mean(current_kl_contrastive, dim=0).item()
    metrics = {"kl_contrastive/kl": current_kl_contrastive, "kl_contrastive/kl_coef": kl_ctrl_contrastive.kl_coef}
    kl_ctrl_contrastive.update(current_kl=current_kl_contrastive, n_steps=batch_size)
    
    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    return data, metrics

def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    # advantage 分发器：按 adv_estimator 选择 core_algos.py 里对应的估计器。
    # 注意：感知 KL（PAPO）已在 apply_kl_contrastive 中累加进 token_level_rewards，
    # 所以这里各估计器拿到的 reward 已经包含了感知惩罚，无需特殊处理。
    # uid 即每个回答所属 prompt 的 index，用于 GRPO/DAPO/RLOO 的组内归一化。
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.DAPO:
        advantages, returns = core_algos.compute_dapo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    if "advantages" in data.batch:
        data.batch.set_("advantages", advantages)
    else:
        was_locked = data.batch.is_locked
        if was_locked:
            data.batch.unlock_()
        data.batch["advantages"] = advantages
        if was_locked:
            data.batch.lock_()

    if "returns" in data.batch:
        data.batch.set_("returns", returns)
    else:
        was_locked = data.batch.is_locked
        if was_locked:
            data.batch.unlock_()
        data.batch["returns"] = returns
        if was_locked:
            data.batch.lock_()
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    @staticmethod
    def safe_set_tensordict(tensor_dict, key, value):
        """Safely set a value in a potentially locked TensorDict."""
        if key in tensor_dict:
            tensor_dict.set_(key, value)
        else:
            was_locked = tensor_dict.is_locked
            if was_locked:
                tensor_dict.unlock_()
            tensor_dict[key] = value
            if was_locked:
                tensor_dict.lock_()

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.DAPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        # define KL schedular
        if config.algorithm.use_kl_prcp:
            if config.algorithm.kl_prcp_schedule == "fixed":
                self.kl_ctrl_contrastive = core_algos.FixedKLController(init_kl_coef=config.algorithm.kl_prcp_coef)
            elif config.algorithm.kl_prcp_schedule == "annealing":
                start_value = config.algorithm.kl_prcp_schedule_args.get("start_value", config.algorithm.kl_prcp_coef)
                end_value = config.algorithm.kl_prcp_schedule_args.get("end_value", 0.0)
                annealing_ratio = config.algorithm.kl_prcp_schedule_args.get("annealing_ratio", 0.5)
                annealing_steps = int(self.training_steps * annealing_ratio)
                print(f"Using annealing KL schedule with start value {start_value}, end value {end_value}, and total steps {annealing_steps}.")
                self.kl_ctrl_contrastive = core_algos.AnnealingKLController(
                    start_value=start_value,
                    end_value=end_value,
                    total_steps=annealing_steps,
                )

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        # 验证流程：跑验证集生成回答 → 打分 → 汇总平均 reward。不更新参数，纯评估。
        reward_tensor_lst = []  # 收集每个验证 batch 的 reward 张量
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []  # 收集样本（输入/输出/标签/分数），用于日志展示表格
        reward_metrics_lst = defaultdict(list)  # 收集各项打分指标（accuracy 等），按 key 累积
        print("Start validation...")  # 提示开始验证
        self.actor_rollout_ref_wg.prepare_rollout_engine()  # 同步 actor 权重到 vLLM，准备生成
        for batch_dict in self.val_dataloader:  # 遍历验证集每个 batch
            test_batch = DataProto.from_single_dict(batch_dict)  # 原始 dict → DataProto 结构
            test_gen_batch = test_batch.pop(  # 抽出生成所需字段，单独组成生成输入
                batch_keys=["input_ids", "attention_mask", "position_ids"],  # 张量字段：token、注意力掩码、位置编码
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],  # 非张量字段：原始 prompt、图像等多模态数据
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)  # 验证时每个 prompt 生成几个回答（默认 1）
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config  # 用验证专用的生成配置覆盖（如 temperature）
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels  # 图像最小像素约束
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels  # 图像最大像素约束
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps  # 视频帧率（若有视频输入）

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)  # 补 pad 使 batch 能被 GPU 数整除（分布式均分）
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)  # vLLM 生成回答
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)  # 去掉之前补的 pad（×repeat 因生成翻倍）

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)  # 原始 batch 也复制 n 份，与多回答对齐
            test_batch = test_batch.union(test_output_gen_batch)  # 合并生成结果（回答）回 batch

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))  # 对回答打分（与训练同一套 reward_fn）

            # store generations
            input_ids = test_batch.batch["prompts"]  # 取 prompt token id
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]  # 解码成可读 prompt 文本
            output_ids = test_batch.batch["responses"]  # 取回答 token id
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]  # 解码成可读回答文本
            scores = reward_tensor.sum(-1).cpu().tolist()  # 每条回答的标量分（token 维求和）
            sample_inputs.extend(input_texts)  # 累积输入文本
            sample_outputs.extend(output_texts)  # 累积输出文本
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())  # 累积标准答案
            sample_scores.extend(scores)  # 累积分数

            reward_tensor_lst.append(reward_tensor)  # 缓存本 batch reward 张量
            for key, value in reward_metrics.items():  # 遍历本 batch 各项指标
                reward_metrics_lst[key].extend(value)  # 按 key 累积到全局列表

        self.actor_rollout_ref_wg.release_rollout_engine()  # 验证完释放 vLLM 显存
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)  # 按需把样本生成结果记录到日志（表格）
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()  # 全验证集平均 reward（核心指标）
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}  # 聚合各项指标，加 val/ 前缀
        print("Finish validation.")  # 提示验证结束
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics}  # 返回总分 + 各项指标

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _aug_img_for_kl_prcp(self, original_images_pil: List[Image.Image]) -> List[Image.Image]:
        """
        Perform augmentation on the original images for contrastive KL.
        This function should be implemented based on the specific augmentation method used.
        """
        aug_config = self.config.algorithm.aug_config
        if self.config.algorithm.contrastive_type == "augmented":
            augmented_images = []
            for img in original_images_pil:
                aug_img = random_patch_blackening(img, **aug_config)
                augmented_images.append(aug_img)
            return augmented_images
        else:
            raise NotImplementedError(f"Unknown contrastive KL type: {self.config.algorithm.contrastive_type}.")

    def _get_kl_prcp_weights(self, batch: DataProto, reward_metrics: Dict[str, Any]) -> DataProto:
        if self.config.algorithm.kl_prcp_apply_mode == "all":
            batch.non_tensor_batch["kl_prcp_weighting"] = np.array([1.0] * len(batch.batch))
        else:
            raise NotImplementedError(f"Unknown contrastive KL apply mode: {self.config.algorithm.kl_prcp_apply_mode}.")
        return batch
    
    def _get_correctness_mult_mask(self, batch: DataProto, reward_metrics: Dict[str, Any]) -> DataProto:
        weights = []
        for i in range(len(batch.batch)):
            if reward_metrics['accuracy'][i] > 0.1:
                weights.append(1.0)
            else:
                weights.append(0.0)
        batch.non_tensor_batch["correctness_mult_mask"] = np.array(weights) # len(batch.batch); 0.0 or 1.0
        return batch

    def _get_kl_prcp_coef(self, batch: DataProto) -> DataProto:
        batch.non_tensor_batch["kl_prcp_coef"] = np.array([self.kl_ctrl_contrastive.kl_coef] * len(batch.batch))
        # update
        self.kl_ctrl_contrastive.update(current_kl=None, n_steps=1)
        return batch

    # ──────────────────────────────────────────────────────────────────────────
    # 训练 Step 第一阶段：生成有效 Batch
    #
    # 【为什么需要一个 while 循环？】
    # DAPO 的在线过滤（online_filtering）会丢弃"全对/全错"的样本组。
    # 过滤后剩余样本数可能不足 rollout_batch_size，需要继续从数据集取数据补充。
    # 这个循环持续生成，直到积累了足够多的有效样本。
    #
    # 【PAPO 的图像增强在这里发生】
    # 如果开启了 use_kl_prcp，在这里对每个样本的原图生成遮掩版本，
    # 随 batch 一起传给后续的 compute_log_probs_aug()。
    # ──────────────────────────────────────────────────────────────────────────
    def _make_batch_data(self, metrics: Dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                # 数据集遍历完一轮，从头开始（epoch 继续）
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)

            # 【PAPO 感知增强】：为每张图生成遮掩版本，供后续 KL 对比使用
            # 遮掩在这里提前生成，而不是在 actor 那边，原因：
            # 遮掩操作是 CPU 上的 PIL 图像处理，在数据准备阶段做比在 GPU worker 上做更高效
            if self.config.algorithm.use_kl_prcp and "multi_modal_data" in new_batch.non_tensor_batch.keys():
                # take the raw PIL images
                aug_multi_modal_data = []
                for item in new_batch.non_tensor_batch["multi_modal_data"]:
                    if "image_aug" in item:
                        # 离线预增强：数据集中已提前存好了遮掩图（语义感知遮掩，比随机更精准）
                        aug_images_pil = item.pop('image_aug')  # a list
                    else:
                        # 在线随机遮掩：实时生成（papo_utils.random_patch_blackening）
                        original_images_pil = item['images'] # a list
                        aug_images_pil = self._aug_img_for_kl_prcp(original_images_pil)
                    aug_multi_modal_data.append({"images": aug_images_pil})
                # 遮掩图和原图一起进入 batch，后续 compute_log_probs_aug() 会用到遮掩图
                new_batch.non_tensor_batch["aug_multi_modal_data"] = aug_multi_modal_data

            # 分离"用于生成的字段"：只把 prompt token 发给 vLLM rollout worker
            # 其他字段（如 answer、aug_multi_modal_data）先保留在 new_batch，生成结束后合并
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # 调用 vLLM rollout worker，对每个 prompt 生成 n 个回答
            # generate_sequences 是 Ray remote call，在 rollout worker 上异步执行
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                # REMAX 算法需要一个贪心解码的基线（temperature=0），用于计算相对优势
                # 本项目用 GRPO/DAPO，这段可以忽略
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                RayPPOTrainer.safe_set_tensordict(new_batch.batch, "reward_baselines", reward_baseline_tensor)
                del gen_baseline_batch, gen_baseline_output

            # 给每个 prompt 分配唯一 ID，用于后续 DAPO 过滤时按组聚合
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            # 把 prompt 数据复制 n 份（对齐 n 个生成回答），再与生成结果合并
            # 这样 batch 中每条数据 = (prompt + 第i个回答)，共 batch_size × n 条
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # 【DAPO 在线过滤】：丢弃"全对/全错"的样本组，只保留有学习信号的组
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                RayPPOTrainer.safe_set_tensordict(new_batch.batch, "token_level_scores", reward_tensor)
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)
                filter_scores = reward_metrics[self.config.algorithm.filter_key]  # 用 overall 字段
                assert len(filter_scores) != 0, "Filter scores should not be empty."
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                # 计算每组的平均分，过滤掉全对（>0.99）和全错（<0.01）的组
                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    # 极端情况：所有组都被过滤了，保留全部（避免空 batch）
                    kept_sample_idxs = list(range(len(uids)))
                new_batch = new_batch[kept_sample_idxs]

            # 累积有效样本，直到达到 rollout_batch_size
            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                if len(batch) == 0:
                    print("Warning: Generated batch is empty, continuing to generate more data...")
                    continue
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise ValueError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
                # 截取精确的 rollout_batch_size 数量返回（多余的丢弃）
                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    # ──────────────────────────────────────────────────────────────────────────
    # 主训练循环：fit()
    #
    # 【整体流程图（每个 step）】
    #
    #  ┌─ 阶段1：生成（gen）─────────────────────────────────────┐
    #  │  _make_batch_data()                                      │
    #  │  → vLLM rollout：每个 prompt 生成 n 个回答              │
    #  │  → [DAPO] 过滤全对/全错的组                             │
    #  │  → [PAPO] 生成遮掩图版本                                │
    #  └──────────────────────────────────────────────────────────┘
    #           ↓
    #  ┌─ 阶段2：打分（reward）──────────────────────────────────┐
    #  │  reward_fn.compute_reward()                              │
    #  │  → medical_evidence.py:compute_score()                  │
    #  │  → 得到每个回答的 overall 分（用于 advantage 计算）     │
    #  └──────────────────────────────────────────────────────────┘
    #           ↓
    #  ┌─ 阶段3：计算 log_prob ──────────────────────────────────┐
    #  │  compute_log_probs()：当前策略对生成文本的对数概率       │
    #  │  [PAPO] compute_log_probs_aug()：遮掩图的对数概率        │
    #  │  [可选] compute_ref_log_probs()：参考策略的对数概率      │
    #  └──────────────────────────────────────────────────────────┘
    #           ↓
    #  ┌─ 阶段4：计算 advantage（adv）──────────────────────────┐
    #  │  compute_advantage()：调用 core_algos.py 中的           │
    #  │  compute_grpo/dapo_outcome_advantage()                   │
    #  │  → 组内归一化 reward → advantage                        │
    #  └──────────────────────────────────────────────────────────┘
    #           ↓
    #  ┌─ 阶段5：更新参数（update_actor）───────────────────────┐
    #  │  actor_rollout_ref_wg.update_actor()                     │
    #  │  → PPO clip loss + KL loss + [PAPO] kl_prcp loss         │
    #  │  → FSDP 反向传播，梯度同步，参数更新                   │
    #  └──────────────────────────────────────────────────────────┘
    #
    # 【driver 进程 vs worker 进程】
    # fit() 运行在 driver 进程（主进程），所有实际计算通过 Ray RPC 分发给 worker。
    # driver 本身只做轻量逻辑：数据调度、advantage 计算、日志记录。
    # ──────────────────────────────────────────────────────────────────────────
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())  # 初始化日志记录器（wandb/tensorboard 等）
        self.global_step = 0  # 全局训练步计数器，从 0 开始
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)  # 进度条
        val_metrics: Optional[Dict[str, Any]] = None  # 验证指标缓存，初始为空

        # load checkpoint before doing anything
        self._load_checkpoint()  # 断点续训：若有 checkpoint，恢复模型/优化器/step
        main_tqdm.update(self.global_step)  # 进度条同步到恢复后的 step

        # 训练前先做一次验证，记录初始性能基准（便于判断训练是否在提升）
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:  # 有验证函数且配置要求训练前验证
            val_metrics = self._validate()  # 跑一遍验证集，得到初始指标
            self.logger.log(data=val_metrics, step=self.global_step)  # 记录到日志
            if self.config.trainer.val_only:  # 若只验证不训练（val_only 模式）
                return  # 直接退出

        self.data_iterator = iter(self.train_dataloader)  # 构造训练数据迭代器
        while self.global_step < self.training_steps:  # 主训练循环：直到达到总步数
            self.global_step += 1  # 步数自增

            metrics, timing_raw = {}, {}  # 本步的指标字典 + 计时字典
            with timer("step", timing_raw):  # 给整个 step 计时
                # ── 阶段1：生成 ──────────────────────────────────────────
                # prepare_rollout_engine()：把 FSDP actor 权重同步给 vLLM rollout worker
                # 每次 actor 更新后，rollout worker 需要拿到最新权重才能生成"当前策略"的回答
                with timer("gen", timing_raw):  # 给生成阶段计时
                    self.actor_rollout_ref_wg.prepare_rollout_engine()  # 把最新 actor 权重同步给 vLLM rollout 引擎
                    batch = self._make_batch_data(metrics=metrics)  # vLLM 生成回答，组成训练 batch（DAPO 模式下还含过滤+打分）
                    # 生成完成后释放 vLLM 显存，让 actor 更新阶段有足够显存
                    self.actor_rollout_ref_wg.release_rollout_engine()  # 卸载 vLLM，回收显存

                # 各 GPU 卡的有效 token 数可能差异很大（不同长度的序列）
                # 重新排序 batch，使每张卡分到的 token 总量尽量均衡，提高 GPU 利用率
                # 注意：reorder 后组内的相对顺序会被打乱，
                #       但 GRPO advantage 计算用 uid 分组，不依赖绝对顺序，所以安全
                self._balance_batch(batch, metrics=metrics)  # 跨卡负载均衡：按 token 数重排序

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()  # 统计每条序列的有效 token 数，存入 meta（后续吞吐量/loss 归一化用）

                # ── 阶段2：打分 ──────────────────────────────────────────
                # online_filtering=True 时，_make_batch_data 里已经打过分了（存在 token_level_scores）
                # 否则这里打分（非 DAPO 模式）
                reward_metrics = None  # 打分指标缓存，None 表示本步还没打过分
                if "token_level_scores" not in batch.batch:  # batch 里还没有分数（非 DAPO 在线过滤模式）
                    with timer("reward", timing_raw):  # 给打分计时
                        reward_ref = self.reward_fn.compute_reward.remote(batch)  # 异步 RPC：reward worker 对回答文本打分
                        reward_tensor, reward_metrics = ray.get(reward_ref)  # 阻塞取回打分结果（标量分 + accuracy 等指标）

                # 【PAPO 专用】：把每个样本的 kl_prcp 权重和系数注入 batch
                # 这些值在 actor update 阶段用于调整 kl_prcp 损失的大小
                if self.config.algorithm.use_kl_prcp:  # 开启了 PAPO 感知 KL
                    if reward_metrics is None:  # 守卫：若上面没打过分（DAPO 已打则跳过），这里补打一次
                        reward_ref = self.reward_fn.compute_reward.remote(batch)
                        reward_tensor, reward_metrics = ray.get(reward_ref)

                    # store acc reward in batch
                    batch = self._get_kl_prcp_weights(batch, reward_metrics)  # 计算每条样本的 kl_prcp 权重，写入 batch
                    # store kl_prcp coef in batch（可能随训练步数 annealing）
                    batch = self._get_kl_prcp_coef(batch)  # 写入当前 kl_prcp 系数（支持 annealing 衰减）

                if self.config.algorithm.use_sft_loss:  # 开启了辅助 SFT 损失
                    if reward_metrics is None:  # 同样的守卫：确保有打分结果
                        reward_ref = self.reward_fn.compute_reward.remote(batch)
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                    # compute correctness mask
                    batch = self._get_correctness_mult_mask(batch, reward_metrics)  # 按 accuracy 生成正确性掩码（只对正确样本加 SFT 损失）

                # ── 阶段3：计算 log_prob ─────────────────────────────────
                # 用当前策略重新计算生成文本的 log_prob（"旧策略" log_prob，用于 PPO clip 比值）
                # 为什么要"重算"？rollout 时用的是 vLLM 推理，不输出 log_prob；
                # 现在用 FSDP actor forward 精确计算
                with timer("old", timing_raw):  # 给 old_log_prob 计算计时
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)  # actor forward 精确算出生成文本的 log_prob
                    batch = batch.union(old_log_probs)  # 合并进 batch（作为 PPO clip 比值的分母基准）

                # 【PAPO 专用】：用遮掩图重新做 forward，得到遮掩图的 log_prob
                # 原图 log_prob 和遮掩图 log_prob 的 KL 散度就是 kl_prcp 损失
                if self.config.algorithm.use_kl_prcp and "aug_multi_modal_data" in batch.non_tensor_batch.keys():  # 开启 PAPO 且 batch 含遮掩图
                    with timer("aug_probs", timing_raw):  # 给遮掩图 forward 计时
                        aug_log_probs = self.actor_rollout_ref_wg.compute_log_probs_aug(batch)  # 用遮掩图做 forward 算 log_prob
                        batch = batch.union(aug_log_probs)  # 合并进 batch（与原图 log_prob 算感知 KL）

                # 参考策略的 log_prob（用于 KL 惩罚项，disable_kl=true 时不参与 reward）
                if self.use_reference_policy:  # 启用了参考模型
                    with timer("ref", timing_raw):  # 给 ref forward 计时
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)  # 参考模型 forward 算 log_prob
                        batch = batch.union(ref_log_probs)  # 合并进 batch（防策略偏离基座的 KL 基准）

                # critic（价值网络）：GRPO/DAPO 不用 critic，这段忽略
                if self.use_critic:  # 仅 PPO/GAE 模式为 True
                    with timer("values", timing_raw):  # 给 critic 计时
                        values = self.critic_wg.compute_values(batch)  # critic forward 估计每个状态的价值 V(s)
                        batch = batch.union(values)  # 合并进 batch（GAE advantage 的 baseline）

                # ── 阶段4：计算 advantage ────────────────────────────────
                with timer("adv", timing_raw):  # 给 advantage 计算计时
                    if "token_level_scores" not in batch.batch:  # 分数还没落到 batch（之前是异步 remote 调用）
                        # 异步等待 reward 计算完成（如果之前是 remote call 还没取到结果）
                        reward_tensor, reward_metrics = ray.get(reward_ref)  # 取回打分结果
                        RayPPOTrainer.safe_set_tensordict(batch.batch, "token_level_scores", reward_tensor)  # 写入原始分 token_level_scores
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}  # 聚合打分指标，加 reward/ 前缀
                        metrics.update(reward_metrics)  # 并入本步指标

                    # KL 惩罚模式（disable_kl=true 时跳过）：把 KL 散度从 reward 中扣除
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:  # 用"reward 内扣 KL"而非"loss 项 KL"，且有参考模型
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)  # 从 token_level_scores 扣 ref KL，得 token_level_rewards
                        metrics.update(kl_metrics)  # 记录 KL 指标
                    else:
                        # 不加 KL 惩罚，直接用原始 reward 作为 token-level reward
                        RayPPOTrainer.safe_set_tensordict(batch.batch, "token_level_rewards", batch.batch["token_level_scores"])  # token_level_rewards = 原始分（不扣 KL）

                    # 调用 core_algos.py 中的 GRPO/DAPO advantage 计算
                    # 这步在 driver 进程上运行（CPU），计算量轻（只是归一化）
                    batch = compute_advantage(  # 把 reward 转成 advantage（组内归一化或 GAE）
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,  # 估计器类型：grpo/dapo/gae 等
                        gamma=self.config.algorithm.gamma,  # 折扣因子（GRPO/DAPO 不用）
                        lam=self.config.algorithm.lam,  # GAE 的 lambda（GRPO/DAPO 不用）
                    )

                # ── 阶段5：更新参数 ──────────────────────────────────────
                if self.use_critic:  # 有 critic 才更新 critic（GRPO/DAPO 跳过）
                    with timer("update_critic", timing_raw):  # 给 critic 更新计时
                        critic_output = self.critic_wg.update_critic(batch)  # RPC：critic worker 做反向传播更新价值网络

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)  # 聚合多卡 critic 指标
                    metrics.update(critic_metrics)  # 并入本步指标

                # actor update：PPO clip loss + KL loss + [PAPO] kl_prcp loss
                # 内部用 FSDP 做分布式反向传播，梯度在各卡间 all-reduce 后更新参数
                if self.config.trainer.critic_warmup <= self.global_step:  # critic 预热期已过才更 actor（GRPO 下 warmup=0，恒成立）
                    with timer("update_actor", timing_raw):  # 给 actor 更新计时
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)  # RPC：actor worker 算 PPO/KL/kl_prcp 损失并反向更新策略（真正的 loss 在此 worker 内部）

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)  # 聚合多卡 actor 指标（pg_loss/kl/clipfrac 等）
                    metrics.update(actor_metrics)  # 并入本步指标

                # ── 验证 & 保存 ──────────────────────────────────────────
                if (
                    self.val_reward_fn is not None  # 有验证函数
                    and self.config.trainer.val_freq > 0  # 配置了验证频率
                    and self.global_step % self.config.trainer.val_freq == 0  # 到达验证步
                ):
                    with timer("validation", timing_raw):  # 给验证计时
                        val_metrics = self._validate()  # 跑验证集

                    metrics.update(val_metrics)  # 并入本步指标

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:  # 到达保存步
                    with timer("save_checkpoint", timing_raw):  # 给存档计时
                        self._save_checkpoint()  # 保存 checkpoint

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()  # 获取 GPU 总数（算吞吐量用）
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))  # 数据相关指标（reward/advantage 分布等）
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))  # 各阶段耗时指标
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))  # 吞吐量指标（token/s 等）

            self.logger.log(data=metrics, step=self.global_step)  # 把本步所有指标写入日志
            main_tqdm.update()  # 进度条 +1

        # perform validation after training
        if self.val_reward_fn is not None:  # 训练结束后补一次最终验证
            if (
                val_metrics is None  # 从没验证过
                or self.config.trainer.val_freq <= 0  # 或没开周期验证
                or self.global_step % self.config.trainer.val_freq != 0  # 或最后一步不是验证步
            ):
                val_metrics = self._validate()  # 补跑验证
                self.logger.log(data=val_metrics, step=self.global_step)  # 记录

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")  # 打印最终验证结果

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:  # 若最后一步还没存档
            self._save_checkpoint()  # 补存最终 checkpoint
