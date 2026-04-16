# r2dreamer

PyTorch implementation of R2-Dreamer and DreamerV3 for continuous/discrete control.

## Agent Status

- **Status:** 🟢 active
- **Last session:** 2026-04-16
- **Current branch:** sigreg
- **What happened:** Ran ManiSkill comparison of dreamerv3 / r2dreamer / sigreg (`compare.sh`). DreamerV3 and R2Dreamer both reach >100 reward regularly by end of 2e5 steps. SIGReg still underperforms even after two integration fixes: (1) sigreg target changed from projector output to encoder `embed` so the encoder actually receives the isotropy gradient (matches le-wm), (2) `loss_scales.sigreg: 0.05 → 1.0` to align effective lambda with le-wm's reference. Earlier r2dreamer crash at 20k was replay-buffer fill; rerun completes.
- **What's next:** Decide whether to keep investigating sigreg (batch-size-compensated lambda ~0.7 is the next knob, since we train at B=16 vs le-wm's B=128) or set it aside. Merge sigreg branch if done.
- **Blocked on:** nothing
- **Key decisions made:** SIGReg must regularize the encoder embedding, not the projector output — otherwise the detached alignment target leaves the encoder without an isotropy gradient. The `loss_scales.sigreg * sigreg_lambd` double-multiply was masking the effective regularization weight.

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
