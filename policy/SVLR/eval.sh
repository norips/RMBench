#!/usr/bin/env bash
set -euo pipefail

# RMBench-style wrapper, aligned with the SVLR demo command.
# Usage from policy/SVLR:
#   bash eval.sh press_button demo_clean_franka svlr_debug 0 0 "press the button"

TASK_NAME="${1:-press_button}"
TASK_CONFIG="${2:-demo_clean_franka}"
CKPT_SETTING="${3:-svlr_debug}"
SEED="${4:-0}"
GPU_ID="${5:-0}"
GLOBAL_TASK="${6:-press the button}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo -e "\033[33mgpu id (to use): ${GPU_ID}\033[0m"

cd "$(dirname "$0")/../.."

if command -v fuser >/dev/null 2>&1 && fuser -s 65500/tcp; then
  echo "Port 65500 is already in use. Stop the previous RMBench bridge (Ctrl-C or POST /stop) before re-running." >&2
  exit 1
fi
rm -f svlr_bridge_*.png

pixi run -e svlr python script/eval_svlr.py --config policy/SVLR/deploy_policy.yml --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --policy_name SVLR \
  --ckpt_setting "${CKPT_SETTING}" \
  --seed "${SEED}" \
  --instruction_type unseen \
  --episode_num 1 \
  --global_task "${GLOBAL_TASK}" \
  --sim_camera_key right_camera \
  --sim_save_debug_images true \
  --sim_drive true \
  --sim_home false \
  --render_freq 1
