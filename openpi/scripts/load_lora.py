"""从 export_lora.py 导出的 npz 加载 LoRA 增量（还原 bfloat16）。

示例：
    from openpi.scripts.load_lora import load_lora_npz

    lora_params = load_lora_npz("loras/thermal_cycler_close-lora5k.npz")
    # lora_params: {"PaliGemma/llm/.../lora_a": jnp.ndarray(bfloat16), ...}
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
