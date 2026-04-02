# r2dreamer

PyTorch implementation of R2-Dreamer and DreamerV3 for continuous/discrete control.

## Agent Status

- **Status:** 🟢 active
- **Last session:** 2026-04-02
- **Current branch:** feat/sailor-r2-packaging
- **What happened:** Made r2dreamer an installable Python package (`pip install -e .`) with thin `r2dreamer/__init__.py` re-export shim and `pyproject.toml`. Built new `sailor-r2` repo at `../sailor-r2/` that depends on this package — combines SAILOR's training pipeline with r2dreamer's world model via an `RSSMAdapter` bridge. All core components verified: adapter, world model, distributional critic ensemble pass shape tests.
- **What's next:** End-to-end test of sailor-r2 on a simple task (e.g. pusht_state). Merge packaging branch to main.
- **Blocked on:** nothing
- **Key decisions made:** Thin shim approach for packaging (avoid rewriting all imports). r2dreamer/\_\_init\_\_.py adds repo root to sys.path so existing relative imports still work. sailor-r2 inherits from SAILOR's SAILORTrainer to minimize code duplication.

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
