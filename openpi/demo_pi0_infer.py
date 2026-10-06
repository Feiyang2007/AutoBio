"""pi0_base 纯推理演示（无仿真、无画面）。

运行：
  cd /home/bio/AutoBio/openpi
  XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" .venv/bin/python demo_pi0_infer.py
"""
import time

import numpy as np

from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

CKPT = "/home/bio/.cache/openpi/openpi-assets/checkpoints/pi0_base"

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