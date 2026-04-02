# r2dreamer

PyTorch implementation of R2-Dreamer and DreamerV3 for continuous/discrete control.

## Agent Status

- **Status:** 🟢 active
- **Last session:** 2026-03-27
- **Current branch:** main
- **What happened:** Added periodic checkpointing (best.pt + latest.pt), auxiliary decoder for imagined rollout visualization (`model.aux_decoder.enabled=true`), and `compare.sh` script that trains R2Dreamer then DreamerV3 back-to-back. Smoke tested successfully — both modes run sequentially and checkpoint correctly.
- **What's next:** Launch a full comparison run via `./compare.sh`. Obtain demo dataset from cluster for demo-conditioned runs.
- **Blocked on:** Demo dataset not available locally (lives on cluster at `../confound/dreamerv3-torch/maniskill_single_goal_easy/`).
- **Key decisions made:** Buffer storage overridden to CPU (`buffer.storage_device=cpu`) and max_size=5e4 for local GPU (12 GB). ManiSkill workers raise on teardown causing non-zero exit — compare.sh treats this as success if `latest.pt` was saved. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` needed to avoid fragmentation OOM. Checkpointing saves best (by eval_score) and latest every `eval_every` steps.

## Active runs

| Run | PID | Log |
|---|---|---|
| Demo comparison v1 (R2D→DV3) | 2263642 | `logdir/comparison_demo_v1.log` |

## Environment

```bash
source .venv/bin/activate
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python train.py env=maniskill ...
```

## Comparison commands

```bash
# R2Dreamer (default)
python train.py env=maniskill logdir=./logdir/ms_r2dreamer buffer.storage_device=cpu buffer.max_size=5e4

# DreamerV3 baseline
python train.py env=maniskill model.rep_loss=dreamer logdir=./logdir/ms_dreamer buffer.storage_device=cpu buffer.max_size=5e4

# With demos (requires dataset on cluster)
python train.py env=maniskill logdir=./logdir/ms_r2dreamer_demo \
  env.demodir=../confound/dreamerv3-torch/maniskill_single_goal_easy/ir0.025_d5_n3
```
