# Reproduction record: thermal_cycler_close LoRA

All numbers in this file were produced on a single RTX 5090 D v2 (24 GB) in WSL2 (Ubuntu 24.04),
2026-10-07. Raw artifacts: `eval_thermal_cycler_close_lora5k.json` (per-episode results) and
`train_loss_curve.csv` (loss every 100 steps).

## Training command

```bash
cd openpi
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
JAX_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run scripts/train.py thermal_cycler_close-lora \
    --exp-name lora5k \
    --batch-size 16 \
    --save-interval 1000 \
    --no-wandb-enabled
```

- Base model: `pi0_base` (PaliGemma 2B + 300M action expert, `gemma_2b_lora` / `gemma_300m_lora`
  variants, LoRA rank=16 / alpha=16 on the backbone and rank=32 / alpha=32 on the action expert,
  scaling = alpha/rank = 1.0)
- Data: `autobio-bench/thermal_cycler_close-mujoco` (100 episodes, LeRobot v2.0), converted to
  `~/.cache/huggingface/lerobot/data/thermal_cycler_close`, norm stats computed per task
- Steps: 5000 (checkpoints every 1000), batch 16, ~5.3 s/step, ~7.5 h wall clock
- Peak GPU memory: 22.2 GB
- Loss: 0.1061 (step 0) -> 0.0041 (step 4999), see `train_loss_curve.csv`

## Evaluation command

```bash
# terminal 1
cd openpi
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" uv run scripts/serve_policy.py --port 8000 \
    policy:checkpoint --policy.config thermal_cycler_close-lora \
    --policy.dir checkpoints/thermal_cycler_close-lora/lora5k/4999

# terminal 2
cd autobio
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python evaluate.py --host 127.0.0.1 --port 8000 \
    --task thermal_cycler_close --num_episodes 20 --seed 0 --save eval_lora5k.json
```

## Results

| Model | Episodes | Success | Rate |
|---|---|---|---|
| pi0_base zero-shot (smoke check) | 6 | 0 | 0% |
| **This LoRA (step 4999)** | **20 (seed 0)** | **18** | **90%** |
| Upstream paper, pi0 full finetune, 30k steps, H800 | 100 x 3 seeds | — | 99.7 ± 0.3% |

Caveats: single task (the easiest of 12), single seed, 20 episodes -> wide confidence interval
(~68-98% Wilson 95%). The zero-shot row is a 6-episode smoke check, not a proper estimate.
This is NOT an apples-to-apples comparison with the upstream paper (full finetune, 30k steps,
1500 GPU-hours on H800 vs 7.5 h LoRA on a consumer GPU).

## Known environment quirks

- `XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"` is required on RTX 5090 (sm_120); without it
  JAX/Triton compilation crashes the process.
- Training and `serve_policy.py` must not run concurrently: the server reserves ~2 GB VRAM which
  pushes the training step time from 5.3 s to ~29 s.
- `openpi/scripts/train.py` upstream raised `KeyError: SLURM_JOB_ID` after the final checkpoint
  save on non-SLURM machines (missing `return` in `wrap()`); fixed in this fork, results unaffected.
