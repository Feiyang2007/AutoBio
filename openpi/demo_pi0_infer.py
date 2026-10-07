"""pi0_base 纯推理演示（无仿真、无画面）。

运行：
  cd openpi
  XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" uv run demo_pi0_infer.py --ckpt /path/to/pi0_base
  # --ckpt 默认取 openpi-assets 的标准缓存位置，可指向任意 pi0_base 检查点目录。
"""
import argparse
import os
import time

import numpy as np

from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

DEFAULT_CKPT = os.environ.get(
    "OPENPI_ASSETS_HOME", os.path.expanduser("~/.cache/openpi/openpi-assets")
) + "/checkpoints/pi0_base"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="pi0_base checkpoint directory")
    CKPT = p.parse_args().ckpt

    t0 = time.perf_counter()
    config = _config.get_config("pi0_aloha")
    policy = _policy_config.create_trained_policy(config, CKPT)
    print(f"[1] 加载 pi0_base : {time.perf_counter() - t0:.2f} s")

    example = aloha_policy.make_aloha_example()

    t = time.perf_counter()
    result = policy.infer(example)
    print(f"[2] 首次推理(含编译): {time.perf_counter() - t:.2f} s")

    actions = np.asarray(result["actions"])
    print(f"[3] 输出: shape={actions.shape} dtype={actions.dtype}")
    print(f"    数值范围: min={actions.min():.3f}  max={actions.max():.3f}  mean={actions.mean():.3f}")
    print(f"    第 1 步动作(前 7 维): {np.round(actions[0, :7], 4).tolist()}")

    t = time.perf_counter()
    for _ in range(5):
        policy.infer(example)
    steady = (time.perf_counter() - t) / 5
    print(f"[4] 稳定推理延迟: {steady:.3f} s/infer  (每次输出 {actions.shape[0]} 步动作)")
    print(f"[5] 全程合计: {time.perf_counter() - t0:.2f} s")