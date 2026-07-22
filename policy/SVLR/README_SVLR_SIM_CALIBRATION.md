# RMBench bridge: clean SVLR simulation calibration

This bridge now sends a real simulation calibration payload to SVLR:

- `depth_npy_b64` is converted to meters.
- `world_xyz_npy_b64` is added when SAPIEN camera access is available.
- `world_xyz_npy_b64` is an HxWx3 float32 array, one RMBench/world-frame XYZ point
  per image pixel.

This avoids reusing the real Panda wrist-camera `dx/dy/dz` calibration inside
simulation. It also works with `right_camera` because the dense world XYZ is
recomputed every frame from the current SAPIEN wrist-camera pose.

Other fixes kept:

- `/process_vlm` is called before `/process_llm_command`.
- gripper-only SVLR actions no longer crash the bridge.
- Panda-width gripper values are mapped to normalized sim gripper values.
- all RGB cameras are saved when `--sim_save_debug_images true`.
- no action timeout is used; a stuck run remains visible until `/stop` or Ctrl-C.

Recommended first test:

```bash
pixi run -e svlr python script/eval_svlr.py --config policy/SVLR/deploy_policy.yml --overrides \
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
  --sim_home false
```
