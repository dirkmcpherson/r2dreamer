#!/bin/bash
# compare.sh — Train R2Dreamer then DreamerV3 back-to-back with identical conditions.
#
# Usage:
#   ./compare.sh                             # default: 2e5 steps
#   STEPS=5000 EVAL_EVERY=2500 ./compare.sh  # quick smoke test
#   LOGDIR=./logdir/run2 ./compare.sh        # custom output dir
#
# Extra Hydra overrides are forwarded to both runs:
#   ./compare.sh model.compile=False

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

: "${ENV:=maniskill}"
: "${SEED:=0}"
: "${STEPS:=2e5}"
: "${EVAL_EVERY:=1e4}"
: "${BUFFER_DEVICE:=cpu}"
: "${BUFFER_SIZE:=5e4}"
: "${ENV_NUM:=2}"
: "${EVAL_NUM:=2}"
: "${LOGDIR:=${SCRIPT_DIR}/logdir/comparison}"

source "${SCRIPT_DIR}/.venv/bin/activate"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
    local name=$1
    local run_logdir=$2
    shift 2
    echo ""
    echo "=========================================="
    echo "  ${name}"
    echo "=========================================="
    python "${SCRIPT_DIR}/train.py" \
        env=${ENV} \
        seed=${SEED} \
        trainer.steps=${STEPS} \
        trainer.eval_every=${EVAL_EVERY} \
        buffer.storage_device=${BUFFER_DEVICE} \
        buffer.max_size=${BUFFER_SIZE} \
        env.env_num=${ENV_NUM} \
        env.eval_episode_num=${EVAL_NUM} \
        logdir="${run_logdir}" \
        "$@" || {
        # ManiSkill subprocess workers can raise on teardown, causing non-zero exit.
        # Treat as success only if the final checkpoint was saved.
        if [[ -f "${run_logdir}/latest.pt" ]]; then
            echo "WARNING: ${name} exited non-zero (likely subprocess cleanup) but latest.pt was saved — continuing."
        else
            echo "ERROR: ${name} failed — latest.pt not found in ${run_logdir}"
            exit 1
        fi
    }
}

run "R2Dreamer" "${LOGDIR}/r2dreamer" model.rep_loss=r2dreamer "$@"
run "SIGReg"    "${LOGDIR}/sigreg"    model.rep_loss=sigreg    "$@"
run "DreamerV3"  "${LOGDIR}/dreamerv3" model.rep_loss=dreamer  "$@"

echo ""
echo "All done. Results in ${LOGDIR}"
echo "  tensorboard --logdir ${LOGDIR}"
