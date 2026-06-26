python script/eval_svlr.py --config policy/SVLR/deploy_policy.yml --overrides \
    --task_name        swap_blocks \
    --global_task      "There are three traies on the table, and two blocks are placed in two different traies. You may move only one block at a time, and each tray can hold at most one block. Swap the positions of the two blocks. Finally press the button." \
    --task_config      demo_clean_franka