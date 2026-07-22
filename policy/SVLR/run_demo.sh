#!/usr/bin/env bash
set -euo pipefail

# Run from RMBENCH repo root or from policy/SVLR.
# Start SVLR first in another terminal:
#   cd <svlr_repo> && bash run_svlr_rmbench.sh
#
# Default command mirrors the manual command used for the SVLR/RMBench demo:
#   press_button + demo_clean_franka + global instruction "press the button"
#
# Usage:
#   bash policy/SVLR/run_demo.sh
#   bash policy/SVLR/run_demo.sh press_button demo_clean_franka svlr_debug 0 "press the button"

cd "$(dirname "$0")/../.."

TASK_NAME="${1:-press_button}"
TASK_CONFIG="${2:-demo_clean_franka}"
CKPT_SETTING="${3:-svlr_debug}"
SEED="${4:-0}"
GLOBAL_TASK="${5:-press the button}"
SIM_CAMERA_KEY="${SIM_CAMERA_KEY:-right_camera}"
SIM_SAVE_DEBUG_IMAGES="${SIM_SAVE_DEBUG_IMAGES:-true}"
SIM_DRIVE="${SIM_DRIVE:-true}"
SIM_HOME="${SIM_HOME:-false}"
SIM_MIRROR_SINGLE_ARM="${SIM_MIRROR_SINGLE_ARM:-auto}"
SIM_KEEP_ALIVE_AFTER_ACTIONS="${SIM_KEEP_ALIVE_AFTER_ACTIONS:-false}"
RENDER_FREQ="${RENDER_FREQ:-30}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
EPISODE_NUM="${EPISODE_NUM:-1}"
INSTANCE_ID="${RMBENCH_INSTANCE_ID:-$TASK_NAME}"
SIM_PORT="${SIM_PORT:-65500}"
SVLR_URL="${SVLR_URL:-http://127.0.0.1:7860}"
RUNTIME_DIR="${RMBENCH_RUNTIME_DIR:-/tmp/svlr-rmbench/$INSTANCE_ID/rmbench}"
SIM_DEBUG_DIR="${SIM_DEBUG_DIR:-$RUNTIME_DIR/debug_images}"
EVAL_OUTPUT_DIR="${RMBENCH_EVAL_OUTPUT_DIR:-$RUNTIME_DIR/eval_result}"

mkdir -p "$RUNTIME_DIR/tmp" "$SIM_DEBUG_DIR" "$EVAL_OUTPUT_DIR"
export TMPDIR="${RMBENCH_TMPDIR:-$RUNTIME_DIR/tmp}"

if command -v fuser >/dev/null 2>&1 && fuser -s "${SIM_PORT}/tcp"; then
  echo "Port ${SIM_PORT} is already in use. Stop the matching RMBench bridge before re-running." >&2
  exit 1
fi

echo "[RMBench instance] id=$INSTANCE_ID sim_port=$SIM_PORT svlr_url=$SVLR_URL runtime=$RUNTIME_DIR"

pixi run -e svlr python script/eval_svlr.py --config policy/SVLR/deploy_policy.yml --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --policy_name SVLR \
  --ckpt_setting "${CKPT_SETTING}" \
  --seed "${SEED}" \
  --instruction_type "${INSTRUCTION_TYPE}" \
  --episode_num "${EPISODE_NUM}" \
  --global_task "${GLOBAL_TASK}" \
  --sim_port "${SIM_PORT}" \
  --svlr_url "${SVLR_URL}" \
  --sim_debug_dir "${SIM_DEBUG_DIR}" \
  --eval_output_dir "${EVAL_OUTPUT_DIR}" \
  --sim_camera_key "${SIM_CAMERA_KEY}" \
  --sim_save_debug_images "${SIM_SAVE_DEBUG_IMAGES}" \
  --sim_drive "${SIM_DRIVE}" \
  --sim_home "${SIM_HOME}" \
  --sim_mirror_single_arm "${SIM_MIRROR_SINGLE_ARM}" \
  --sim_keep_alive_after_actions "${SIM_KEEP_ALIVE_AFTER_ACTIONS}" \
  --render_freq "${RENDER_FREQ}"
