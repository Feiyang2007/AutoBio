# Reproduction record: screw_loose & screw_tighten LoRA (5000 steps) — **unsuccessful**

All numbers were produced on a single RTX 5090 D v2 (24 GB) in WSL2 (Ubuntu 24.04), 2026-10-10/11.
Raw artifacts: `eval_screw_loose_lora5k.json`, `eval_screw_tighten_lora5k.json` (100 episodes each,
seed 0), `eval_screw_loose_base_zeroshot.json` (10 episodes, seed 0), and the two loss-curve CSVs.

**Honest summary: at 5000 LoRA steps both tasks scored 0/100 in closed-loop evaluation, and the
π0_base zero-shot baseline also scored 0/10. We did NOT obtain a working policy for these two
tasks.** The rest of this file documents the setup and the diagnostics that rule out harness bugs.

## Training command (identical for both tasks)

```bash
cd openpi
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
JAX_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run scripts/train.py <task>-lora \
    --exp-name lora5k --batch-size 16 --save-interval 1000 --no-wandb-enabled
```

- Base model: `pi0_base`; LoRA config `<task>-lora` (derived automatically by `add_config()`).
- Data: `autobio-bench/screw_loose-mujoco` and `autobio-bench/screw_tighten-mujoco`
  (MuJoCo flavor, 100 episodes each, LeRobot v2.0, robot_type=aloha, state/action dim 14).
- Steps: 5000 (checkpoints every 1000), batch 16, ~5.3 s/step, ~7.4 h wall clock per task.
- Loss: screw_loose 0.2961 (step 0) -> 0.0175 (step 4900); screw_tighten 0.2425 -> 0.0146.
  Loss curves look healthy, so the failure is **not** a training-loop bug.

## Evaluation command

```bash
# terminal 1 (policy server)
cd openpi
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" uv run scripts/serve_policy.py --port 8000 \
    policy:checkpoint --policy.config screw_loose-lora \
    --policy.dir checkpoints/screw_loose-lora/lora5k/4999
# terminal 2 (evaluation)
cd autobio
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python evaluate.py --host 127.0.0.1 --port 8000 \
    --task screw_loose --num_episodes 100 --seed 0 --save eval.json
```

## Results

| Model | Task | Episodes | Success | Rate |
|---|---|---|---|---|
| This LoRA (step 4999) | screw_loose | 100 (seed 0) | 0 | 0% |
| This LoRA (step 4999) | screw_tighten | 100 (seed 0) | 0 | 0% |
| pi0_base zero-shot | screw_loose | 10 (seed 0) | 0 | 0% |

For reference, the same 5000-step LoRA recipe on the 7-DoF single-arm `thermal_cycler_close`
reached 18/20 (90%) — see `TRAINING_CONFIG.md`. So the recipe itself works; these two dual-arm
contact-rich tasks did not.

## Diagnosis (why we believe this is a policy-capability gap, not a harness bug)

1. **The evaluation pipeline is correct.** With `--seed 0`, the evaluator's initial joint state is
   *bit-identical* to training episode 0's first frame (max elementwise diff = 0.0000), and the
   live-rendered images match that episode's first video frame (mean pixel diff 3–11). Prompt,
   camera mapping, action dimension (14) and render settings (shadows/reflections/sitegroup off in
   both `autobio/render.py` and `autobio/evaluator.py`) are all consistent.
2. **The tasks are solvable in this harness.** The repo's own scripted `Expert`
   (`ScrewLoose.Expert` / `ScrewTighten.Expert`) drives the task to `check() == True`.
3. **The policy did learn the data.** Fed real dataset observations, the LoRA reproduces the expert
   action closely (~90% accurate, including large-magnitude motions; e.g. at frame 50 the expert
   action deviates 0.24 from the current state and the policy tracks it to within 0.025). Zeroing
   one or more camera streams changes the output by 0.07–0.14, so the policy does use vision.
4. **But it fails in closed loop.** Starting from the *exact training-episode initial state*, the
   policy's first action chunk undershoots the expert (~40% of the expert's motion magnitude); the
   error compounds across chunks (0.13 joint-space divergence after 50 frames) and the gripper
   never reaches the tube cap — the cap height stays constant at 0.1196 for the whole episode.
5. **Shortening the open-loop action horizon does not reliably fix it.** Re-planning every 25 or 10
   steps instead of executing the full 50-action chunk did not produce successes across seeds.

Conclusion: at 5000 LoRA steps the policy is accurate on in-distribution single-step predictions but
not dynamically stable enough for these contact-rich dual-arm screw tasks. The upstream config
default for these tasks is `num_train_steps=30_000`; we stopped at 5000 and have **not** yet
validated whether longer training closes the gap. We publish these artifacts as an honest negative
result rather than silently dropping them.

## Known environment quirks

Same as `TRAINING_CONFIG.md` (RTX 5090 `XLA_FLAGS` requirement, train/serve mutual exclusion,
`SLURM_JOB_ID` fix).
