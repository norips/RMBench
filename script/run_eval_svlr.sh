#!/bin/bash
set -euo pipefail

# Minimal SVLR <-> RMBench smoke test.
# Start SVLR in another terminal first, for example:
#   cd /path/to/svlr
#   python main.py --robot_name RMBENCH --http_server 127.0.0.1 --port 65500 \
#     --llm_provider Ollama --llm_name qwen3.5:2b \
#     --vlm_provider Ollama --vlm_name granite3.2-vision \
#     --disable-startup-init-pose

python script/eval_svlr.py --config policy/SVLR/deploy_policy.yml --overrides \
    --task_name press_button \
    --task_config demo_clean_franka \
    --policy_name SVLR \
    --ckpt_setting svlr_debug \
    --seed 0 \
    --instruction_type unseen \
    --episode_num 1 \
    --global_task "press the button" \
    --sim_camera_key right_camera \
    --sim_save_debug_images true \
    --sim_drive true \
    --sim_finish_idle_s 5.0 \
    --sim_home false \
    --sim_mirror_single_arm auto \
    --sim_keep_alive_after_actions false \
    --sim_min_substeps 8
