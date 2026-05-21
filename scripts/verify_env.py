"""
GPU 节点环境验证脚本
用法: conda activate papo_tc && python scripts/verify_env.py
"""

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

PASS = "\033[92m  OK  \033[0m"
FAIL = "\033[91m  FAIL\033[0m"
WARN = "\033[93m  WARN\033[0m"

results = []

def check(name, fn):
    try:
        msg = fn()
        print(f"{PASS} {name}" + (f" — {msg}" if msg else ""))
        results.append((name, True))
    except Exception as e:
        print(f"{FAIL} {name} => {e}")
        results.append((name, False))


# ── 1. CUDA ────────────────────────────────────────────────────────────────
def check_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    return f"{n} GPU(s): {names}"

check("torch CUDA available", check_cuda)


def check_torch_version():
    import torch
    return f"torch {torch.__version__}, cuda {torch.version.cuda}"

check("torch version", check_torch_version)


# ── 2. flash-attn ──────────────────────────────────────────────────────────
def check_flash_attn():
    import flash_attn
    return f"flash_attn {flash_attn.__version__}"

check("flash_attn import", check_flash_attn)


def check_flash_attn_func():
    import torch
    from flash_attn import flash_attn_func
    # 小规模前向测试
    B, T, H, D = 1, 16, 8, 64
    q = torch.randn(B, T, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, T, H, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, T, H, D, dtype=torch.float16, device="cuda")
    out = flash_attn_func(q, k, v)
    assert out.shape == (B, T, H, D)
    return f"output shape {out.shape}"

check("flash_attn_func forward", check_flash_attn_func)


# ── 3. verl 核心模块 ────────────────────────────────────────────────────────
def check_verl_protocol():
    from verl.protocol import DataProto
    return f"DataProto ok"

check("verl.protocol.DataProto", check_verl_protocol)


def check_verl_py_functional():
    from verl.utils.py_functional import timeout_limit, list_of_dict_to_dict_of_list
    return None

check("verl.utils.py_functional", check_verl_py_functional)


def check_verl_torch_functional():
    from verl.utils.torch_functional import allgather_dict_tensors, log_probs_from_logits
    return None

check("verl.utils.torch_functional", check_verl_torch_functional)


def check_verl_device():
    from verl.utils.device import get_device_name
    return get_device_name()

check("verl.utils.device", check_verl_device)


def check_verl_tensordict_utils():
    from verl.utils.tensordict_utils import pad_to_divisor
    return None

check("verl.utils.tensordict_utils", check_verl_tensordict_utils)


def check_verl_models():
    from verl.models.transformers import qwen2_vl
    return None

check("verl.models.transformers", check_verl_models)


# ── 4. vLLM ────────────────────────────────────────────────────────────────
def check_vllm():
    import vllm
    return f"vllm {vllm.__version__}"

check("vllm import", check_vllm)


# ── 5. Ray ─────────────────────────────────────────────────────────────────
def check_ray():
    import ray
    return f"ray {ray.__version__}"

check("ray import", check_ray)


# ── 6. 模型路径 ────────────────────────────────────────────────────────────
MODEL_PATH = "/inspire/hdd/project/qproject-multimedicine/public/share_models/Lingshu-7B"

def check_model_path():
    assert os.path.isdir(MODEL_PATH), f"not found: {MODEL_PATH}"
    files = os.listdir(MODEL_PATH)
    config = [f for f in files if "config" in f.lower()]
    return f"{len(files)} files, config: {config[:3]}"

check("model path exists", check_model_path)


# ── 7. 数据路径 ────────────────────────────────────────────────────────────
TRAIN_FILE = os.path.join(REPO_ROOT, "data/rl_3_evidence_papo_jsonl/train_cepo_lite_80_10_10_papo.jsonl")
VAL_FILE   = os.path.join(REPO_ROOT, "data/eval/benchmark_medreasoner_evidence_eval_400_balanced_papo.jsonl")

def check_train_data():
    assert os.path.isfile(TRAIN_FILE), f"not found: {TRAIN_FILE}"
    size_mb = os.path.getsize(TRAIN_FILE) / 1024 / 1024
    return f"{size_mb:.1f} MB"

def check_val_data():
    assert os.path.isfile(VAL_FILE), f"not found: {VAL_FILE}"
    size_mb = os.path.getsize(VAL_FILE) / 1024 / 1024
    return f"{size_mb:.1f} MB"

check("train data file", check_train_data)
check("val data file", check_val_data)


# ── 汇总 ───────────────────────────────────────────────────────────────────
total = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print()
print("=" * 50)
print(f"结果: {passed}/{total} 通过" + (f"，{failed} 失败" if failed else " — 全部OK"))
print("=" * 50)

if failed:
    sys.exit(1)
