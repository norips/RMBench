python script/eval_policy.py --config policy/Mem-0/deploy_policy.yml --overrides \
    --task_name        swap_blocks \
    --execution_ckpt   policy/Mem-0/checkpoints/Mem-0-m1mix-RMBench/checkpoint/m1_mix_final_step50000.pt \
    --state_stats_path policy/Mem-0/checkpoints/Mem-0-m1mix-RMBench/norm_stats/norm_stats.json \
    --ckpt_setting     m1mix \
    --global_task      "There are three traies on the table, and two blocks are placed in two different traies. You may move only one block at a time, and each tray can hold at most one block. Swap the positions of the two blocks. Finally press the button." \
    --action_horizon   30