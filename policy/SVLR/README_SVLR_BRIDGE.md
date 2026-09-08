# SVLR bridge for RMBench

This folder exposes RMBench/RoboTwin as an HTTP robot server that SVLR can control without changing SVLR's core architecture.

## Clean setup for the aloha-agilex target

From a clean RMBench clone on `svlr-bridge-current`:

```bash
cd ~/RMBench
pixi install -e svlr
pixi run -e svlr setup
pixi run -e svlr validate-franka
```

`setup` is idempotent. It downloads:

- RoboTwin2.0 embodiments from `TianxingChen/RoboTwin2.0/embodiments.zip` and extracts them under `assets/embodiments/`.
- RMBench object assets from `TianxingChen/RMBench` under `assets/objects/`.
- RMBench `swap_blocks/demo_clean` evaluation data from `TianxingChen/RMBench` under `data/data/swap_blocks/demo_clean/`.

Then it generates `curobo.yml`, `curobo_left.yml`, and `curobo_right.yml` from `*_tmp.yml` templates with absolute paths for the current checkout, and validates that the aloha-agilex and Franka URDF/SRDF/meshes/Curobo files and swap-block object/data files exist.

## Embodiment layout

The SVLR default is `task_config/demo_clean_aloha.yml`, which uses `embodiment: ["aloha-agilex"]` — RMBench's default dual-arm robot, with two real arms behind the logical left/right 16D action layout.

For this mode:

- `--sim_arm right` selects the physical right arm; the left arm holds its reset pose.
- `--sim_camera_key right_camera` is the right wrist camera of that arm.
- `--sim_mirror_single_arm auto` resolves to *off*: the two logical slots are two different articulations, so no mirroring is needed or wanted.
- `HOME_CONTROLLED` in `deploy_policy.py` is the ready pose driven before the first perception call; override with `--sim_home_controlled` / `SIM_HOME_CONTROLLED` when a task needs a different one. Its quaternion is wxyz in the ee frame, which for aloha-agilex is the world rotation of `fr_link6`: `(0.5, -0.5, 0.5, 0.5)` points the fingers down, closes the jaws along world X and rolls the wrist so the picture is the same way up as the head camera's. Keep it equal to `init_pose` in SVLR's `actions/RMBENCH_ALOHA_action.json`.

`task_config/demo_clean_franka.yml` (`embodiment: ["franka-panda"]`) remains available for the single-Panda setup. That one is *one* physical articulation exposed through both logical slots, so:

- `--sim_mirror_single_arm auto` mirrors the controlled logical slot to both 16D slots before `env.take_action(..., "ee")`, avoiding one logical slot trying to hold while the other moves the same articulation.
- RMBench keeps left/right gripper state aliases synchronized so success checks that look at the right gripper remain valid.
- The pose constants in `deploy_policy.py` (`DEFAULT_ENDPOSE`, `HOME_CONTROLLED`) are now aloha values and would need to be switched back for a Franka run.

## Start SVLR

Terminal 1, in the SVLR repo (`~/svlr`, not `~/svlr-pr6`):

```bash
cd ~/svlr
SVLR_SEGMENTATION_USE_VLM_IMAGE=1 SVLR_DEBUG_BBOX=1 bash run_svlr_rmbench.sh
```

The launcher is expected to connect to the RMBench robot server at `127.0.0.1:65500`; the RMBench bridge itself defaults to `http://127.0.0.1:7860` for the SVLR Gradio app.

## Target RMBench command

Terminal 2, in RMBench:

```bash
cd ~/RMBench
RMBENCH_SWAP_DEBUG_SUCCESS=1 pixi run -e svlr python script/eval_svlr.py \
  --config policy/SVLR/deploy_policy.yml --overrides \
  --task_name swap_blocks \
  --task_config demo_clean_aloha \
  --policy_name SVLR \
  --ckpt_setting svlr_swap_debug \
  --seed 0 \
  --instruction_type unseen \
  --episode_num 1 \
  --global_task "Swap the positions of the two blocks. Finally press the button." \
  --sim_camera_key right_camera \
  --sim_vlm_camera_shader_dir minimal \
  --sim_save_debug_images true \
  --sim_drive true \
  --sim_home true \
  --sim_arm right \
  --sim_mirror_single_arm auto \
  --sim_keep_alive_after_actions false \
  --render_freq 1
```

`RMBENCH_SWAP_DEBUG_SUCCESS=1` is an opt-in debug success gate for bridge plumbing. It does not replace asset/config validation and should be omitted for real success-rate measurement.

## SVLR env: Grounded-SAM-2 setup (perception backend)

`run_svlr_rmbench.sh` uses the Grounded-SAM-2 segmentation backend. Three
environment prerequisites must be satisfied in the `svlr` conda env / the
`~/Documents/Grounded-SAM-2` checkout. None of these require editing SVLR code.

1. Model checkpoints (not shipped with the repo, only downloader scripts are):

   ```bash
   cd ~/Documents/Grounded-SAM-2
   # SAM 2.1 large  -> checkpoints/sam2.1_hiera_large.pt
   curl -L -C - -o checkpoints/sam2.1_hiera_large.pt \
     https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
   # GroundingDINO SwinT-OGC -> gdino_checkpoints/groundingdino_swint_ogc.pth
   curl -L -C - -o gdino_checkpoints/groundingdino_swint_ogc.pth \
     https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   ```

2. GroundingDINO runtime deps into the svlr env:

   ```bash
   python -m pip install "supervision>=0.22.0" pycocotools addict yapf
   ```

3. transformers 5.x compatibility shim for the vendored GroundingDINO text
   encoder. The SVLR env ships transformers 5.x (required by gradio 6.x /
   huggingface-hub 1.x), which removed `ModuleUtilsMixin.get_head_mask` and
   changed `get_extended_attention_mask`. transformers cannot be downgraded
   (transformers 4.x needs huggingface-hub < 1.0, but gradio 6.20 needs
   huggingface-hub >= 1.2). Install the scoped, auto-loading shim:

   ```bash
   # from the RMBench repo, with the svlr env active (or pass its python)
   bash policy/SVLR/svlr_env/install_gdino_tf5_compat.sh
   ```

   This drops `_gdino_tf5_compat.py` + `zz_gdino_tf5_compat.pth` into the env's
   site-packages so the shim loads automatically for `python main.py`. It only
   restores the two removed/renamed BERT helpers and does not affect normal
   transformers 5.x usage (sentence-transformers, etc.).

Note: `remote_rgbd_camera ... Connection refused` on the SVLR side simply means
the RMBench bridge (`:65500`) is not up yet. Start RMBench and wait for
`[sim-server] http://0.0.0.0:65500` before expecting camera frames.
