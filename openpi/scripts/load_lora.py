"""从 export_lora.py 导出的 npz 加载/合并 LoRA 增量。

示例：
    from openpi.scripts.load_lora import load_lora_npz, merge_lora_into_params

    lora_params = load_lora_npz("loras/thermal_cycler_close-lora5k.npz")
    # lora_params: {"PaliGemma/llm/.../lora_a": jnp.ndarray(bfloat16), ...}

    base_params = ...  # 用 openpi.models.model.restore_params(pi0_base) 加载的基座参数
    merged = merge_lora_into_params(base_params, lora_params)
    # merged 等价于训练产出的完整 checkpoint（W + lora_b @ lora_a * scaling）。

注意：scaling = alpha / rank。AutoBio 的 thermal_cycler_close-lora 用
gemma_2b_lora / gemma_300m_lora（alpha == rank），故 scaling == 1.0。
其他模型配置请按 openpi/models/gemma.py 中的 LoRAConfig 传入 scaling。
"""
import json
import pathlib

import numpy as np


def load_lora_npz(path: str) -> dict:
    """加载 npz 并把 uint16 位视图还原为 jax bfloat16 数组。"""
    import jax.numpy as jnp

    path = pathlib.Path(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    raw = np.load(path)
    out = {}
    for key, info in meta.items():
        arr = np.ascontiguousarray(raw[key], dtype=np.uint16)
        assert list(arr.shape) == info["shape"], f"shape mismatch for {key}"
        out[key] = jnp.asarray(arr).view(jnp.bfloat16)
    return out


def merge_lora_into_params(base_params: dict, lora_params: dict, scaling: float = 1.0) -> dict:
    """把 LoRA 增量合并进基座参数，返回合并后的 params（输入不被修改）。

    按 openpi.models.lora.Einsum 的前向定义：out = x @ w + (x @ lora_a) @ lora_b * scaling，
    等价于 w_new = w + tensordot(lora_a, lora_b, 收缩 rank 维) * scaling。
    key 约定：".../lora_a" 与 ".../lora_b" 同目录配对，目标权重为 ".../w"。
    """
    params = dict(base_params)
    a_keys = [k for k in lora_params if k.endswith("lora_a")]
    for ak in a_keys:
        prefix = ak[: -len("lora_a")]
        bk, wk = prefix + "lora_b", prefix + "w"
        if wk not in params:
            raise KeyError(f"base params missing {wk}")
        a = np.asarray(lora_params[ak], dtype=np.float32)
        b = np.asarray(lora_params[bk], dtype=np.float32)
        # lora_a 形状 [..., in, rank]，lora_b 形状 [..., rank, out]（axes=(-2,-1)）
        delta = np.tensordot(a, b, axes=([-1], [-2]))
        w = np.asarray(params[wk], dtype=np.float32)
        params[wk] = w + delta * scaling
    return params
