"""导出 LoRA 增量为 npz（bf16 以 uint16 位视图存储）。

usage: cd /home/bio/AutoBio/openpi && JAX_PLATFORMS=cpu .venv/bin/python scripts/export_lora.py \
    --ckpt checkpoints/thermal_cycler_close-lora/lora5k/4999/params \
    --out loras/thermal_cycler_close-lora5k.npz
"""
import argparse
import pathlib
import sys

import numpy as np
from flax import traverse_util

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from openpi.models import model as _model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    params = _model.restore_params(args.ckpt, restore_type=np.ndarray)
    flat = traverse_util.flatten_dict(params)
    lora = {"/".join(k): v for k, v in flat.items() if "lora" in "/".join(k).lower()}
    assert lora, "No lora params found"

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    meta = {}
    for k, v in lora.items():
        assert v.dtype == np.dtype("bfloat16"), f"unexpected dtype {v.dtype} for {k}"
        # bfloat16 在 numpy 中无原生类型，按位存为 uint16，读取时还原。
        payload[k] = v.view(np.uint16)
        meta[k] = {"shape": list(v.shape), "dtype": "bfloat16"}
    np.savez(out, **payload)
    meta_path = out.with_suffix(".meta.json")
    import json

    meta_path.write_text(json.dumps(meta, indent=1))
    size = out.stat().st_size
    print(f"导出 {len(lora)} tensors -> {out} ({size/1e6:.1f} MB), meta -> {meta_path}")


if __name__ == "__main__":
    main()
