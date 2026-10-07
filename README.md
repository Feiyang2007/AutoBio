# AutoBio — Reproduction Fork

This repository is a **reproduction and study fork** of
[autobio-bench/AutoBio](https://github.com/autobio-bench/AutoBio) (2025 pre-release snapshot).

**All credit for the AutoBio benchmark — its simulation assets, tasks, datasets, codebase, and
paper — belongs to the original AutoBio team.** This fork does not claim any ownership of the
upstream project; it exists only to (a) repair a few breakages in the frozen upstream snapshot and
(b) document our own small-scale reproduction of one fine-tuning experiment on top of it.

- Upstream repo: <https://github.com/autobio-bench/AutoBio>
- Upstream paper: [AutoBio: A Simulation and Benchmark for Robotic Automation in Digital Biology Laboratory](https://arxiv.org/abs/2505.14030)
- Upstream note (quoted): *"This is currently a preliminary version of AutoBio ... The project is
  in development, and the codebase is undergoing structural improvements."*

## What we changed in this fork

1. `autobio/task.py` — fixed 6 stale import paths so that all 12 tasks import again
   (`insert`→`transfer_centrifuge_tube`, `screw_loose`→`screw_loosen`,
   `screw_tighten`→`screw_tighten`, `insert_centrifuge_5430`→`load_centrifuge_5430`,
   `vortex_mixer`→`mani_vortex_mixer`).
2. `openpi/scripts/train.py` — added a missing `return` in `wrap()` which raised
   `KeyError: SLURM_JOB_ID` on non-SLURM machines (after checkpoint save; results unaffected).
3. `openpi/uv.lock` — bumped `nvidia-cuda-nvcc-cu12` to 12.9.86, required for RTX 5090 (sm_120).
4. Added `openpi/scripts/export_lora.py` + `openpi/scripts/load_lora.py` (portable LoRA delta
   export/load) and `openpi/demo_pi0_infer.py` (standalone π0 inference demo).

## Our reproduction result (this fork only, single easy task)

π0 LoRA fine-tuned on the MuJoCo flavor of `thermal_cycler_close` (100 episodes, 5000 steps,
batch 16, one RTX 5090 24GB, ~7.5 h):

| Model | Success rate |
|---|---|
| π0_base zero-shot | 0/6 (0%) |
| **This LoRA (5000 steps)** | **18/20 (90%)** |
| Upstream paper, π0 full finetune (30k steps, H800 ×1500 GPU-hours) | 99.7% |

Fine-tuned weights:

- **Full orbax checkpoint (~4.9 GB)**:
  [`CloudTugWind/AutoBio-pi0-thermal-cycler-close-lora5k`](https://huggingface.co/CloudTugWind/AutoBio-pi0-thermal-cycler-close-lora5k) on HuggingFace

  ```bash
  huggingface-cli download CloudTugWind/AutoBio-pi0-thermal-cycler-close-lora5k \
      autobio_pi0_tc_close_lora5k_ckpt.tar --local-dir .
  tar -xf autobio_pi0_tc_close_lora5k_ckpt.tar   # -> ./4999/
  ```

- **LoRA-only delta (100 MB)**: attached to this repo's
  [GitHub Releases](https://github.com/Feiyang2007/AutoBio/releases/tag/thermal_cycler_close-lora5k),
  load with `openpi/scripts/load_lora.py` (extraction code in `openpi/scripts/export_lora.py`).

Serve & evaluate (RTX 5090 needs the `XLA_FLAGS` workaround for a JAX/Triton crash):

```bash
cd openpi
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" uv run scripts/serve_policy.py --port 8000 \
    policy:checkpoint --policy.config thermal_cycler_close-lora --policy.dir /path/to/4999
```

```bash
cd autobio
MUJOCO_GL=egl python evaluate.py --host 127.0.0.1 --port 8000 \
    --task thermal_cycler_close --num_episodes 20 --seed 0 --save result.json
```

Everything below is from the **upstream** README, kept here for convenience (all first-person
statements belong to the AutoBio team).

---

## Upstream repository layout

- `autobio`: The AutoBio codebase. See `autobio/README.md` for install and usage instructions.
- `openpi`: The modified openpi (π0) codebase adapted from
  [OpenPI](https://github.com/Physical-Intelligence/openpi), containing the AutoBio team's code to
  convert autobio data to LeRobot format and to reproduce the π0 experiment results. See
  `openpi/README.md`.
- `RoboticsDiffusionTransformer`: The modified RDT codebase adapted from
  [RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer), used to
  reproduce the RDT experiment results. The main modification is in
  `RoboticsDiffusionTransformer/data/lerobot_vla_dataset.py` (LeRobot dataset loading). See its
  `README.md`.

## Datasets (hosted by the upstream AutoBio team on HuggingFace)

Two rendering flavors (MuJoCo and Blender Cycles), LeRobot v2.0 format, videos at 224x224@50fps:

- [MuJoCo dataset collection](https://huggingface.co/collections/autobio-bench/autobio-mujoco-68219f844a4650a970b307bd):
  [Close thermal cycler lid](https://huggingface.co/datasets/autobio-bench/thermal_cycler_close-mujoco) ·
  [Open thermal cycler lid](https://huggingface.co/datasets/autobio-bench/thermal_cycler_open-mujoco) ·
  [Pick up centrifuge tube](https://huggingface.co/datasets/autobio-bench/pickup-mujoco) ·
  [Unscrew centrifuge tube cap](https://huggingface.co/datasets/autobio-bench/screw_loose-mujoco) ·
  [Aspirate with pipette](https://huggingface.co/datasets/autobio-bench/pipette-mujoco) ·
  [Transfer centrifuge tube](https://huggingface.co/datasets/autobio-bench/insert-mujoco) ·
  [Screw on centrifuge tube cap](https://huggingface.co/datasets/autobio-bench/screw_tighten-mujoco) ·
  [Operate thermal mixer panel](https://huggingface.co/datasets/autobio-bench/thermal_mixer-mujoco) ·
  [Load centrifuge rotor](https://huggingface.co/datasets/autobio-bench/insert_centrifuge_5430-mujoco)
- [Blender Cycles dataset collection](https://huggingface.co/collections/autobio-bench/autobio-blender-6824b2fbd77b18fe7a00595d):
  same 9 tasks with `-blender` suffix.

## Upstream workflow (quoted/paraphrased from upstream docs)

### Environment setup

Follow `autobio/README.md`. The codebase assumes Linux (tested on Ubuntu 20.04 / 24.04).
The `openpi` and `RoboticsDiffusionTransformer` dependencies are fragile and should be installed
in separate environments; evaluation of fine-tuned models runs via remote inference.

### Data collection & conversion (optional — upstream datasets can be downloaded directly)

Trajectories are collected by executing task files in the `autobio` conda environment; MuJoCo
rendering via `bash render.bash logs/<task_name>`, Blender rendering via `blender --background
--python render_blender.py`. Conversion to LeRobot format + normalization stats:

```bash
cd openpi
LEROBOT_HOME=$PWD uv run scripts/convert.py --data_dir '../autobio/logs/<task_name>' --repo_id 'data/<task_name>'
LEROBOT_HOME=$PWD JAX_PLATFORMS=cpu uv run python scripts/compute_norm_stats.py --config-name '<task_name>'
```

### Training

See `openpi/slurm/train.bash` and `RoboticsDiffusionTransformer/train.bash` for the upstream
launchers. Upstream full training (batch 32) uses one 80 GiB GPU (A100/H100/H800); for smaller
GPUs, openpi supports LoRA training via config name `<task_name>-lora` (this is what our
reproduction above used).

### Evaluation

Upstream evaluation protocol:

```bash
# policy server (openpi directory)
XLA_PYTHON_CLIENT_MEM_FRACTION=.6 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    policy:checkpoint --policy.config '<task_name>' --policy.dir 'checkpoints/<task_name>/<exp_name>/29999'
# evaluation (autobio directory)
python evaluate.py --port 8000 --task '<task_name>' --num_episodes 100 --image_history 0 \
    --num_workers 0 --render_device_id 0 --save result.json
```

## License & attribution

The upstream AutoBio code retains its original licensing. Our additions listed in
"What we changed in this fork" are provided under the same terms for reproduction purposes.
