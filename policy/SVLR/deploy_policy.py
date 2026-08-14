"""
Simulation web server — RoboTwin (SAPIEN) behind the robot_server.py HTTP API
-----------------------------------------------------------------------------
Exposes the same endpoints robot_server.py exposes, so SVLR's internal robot
client talks to the RoboTwin simulation with no change. Adds an optional driver
thread that calls SVLR's Gradio API (/process_vlm, then /process_llm_command) once per episode.

Topology
--------
    RoboTwin harness (this module is its "policy": get_model/eval/reset_model)
        │  serves cam/pose/actions on :65500  ◀── SVLR's robot client
    SVLR Gradio app (:7860)  ── the brain ──────┘
        ▲
    driver thread (in this process) ── /process_vlm + /process_llm_command(instruction) per episode

Why there is no startup deadlock
--------------------------------
"Port listening" != "server has data". SimServer.start() binds :65500 during
get_model(), before the episode loop and independent of SVLR. The bridge is
pre-seeded with a placeholder frame + neutral pose, so /camera and /robot_pose
answer from the instant the port binds — SVLR can boot in any order. The driver
retries connecting to :7860, so order is fully forgiving.

Per-step policy contract (from eval_policy)
-------------------------------------------
    reset_func(model)
    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
        observation = TASK_ENV.get_obs()
        eval_func(TASK_ENV, model, observation)   # ONE call == ONE step
        if TASK_ENV.eval_success: break

One eval() step:
    publish harness frame + measured pose  ->  set end_action True (sim ready)
    ->  block until SVLR POSTs /send_action  ->  drive TASK_ENV.take_action(...,"ee")
    exactly once, then publish the fresh observation and acknowledge completion.

This is a demo bridge, not the final benchmark policy. RMBench take_action() is
already a dense motion primitive; re-sending the same target in a tolerance loop
made contact actions such as press unstable. The bridge also exits cleanly if the
SVLR Gradio driver finishes without producing any action.

Threading: eval()/reset_model() run on the harness MAIN thread (all env access
stays there); uvicorn + the SVLR driver run on background daemon threads; HTTP
endpoints only read cached bridge state.

Install: put at policy/<policy_name>.py and set policy_name in the eval config.
Config via usr_args (sim_host/sim_port/sim_arm/sim_drive/svlr_url) or env
(SIM_HOST/SIM_PORT/SIM_ARM/SIM_DRIVE/SVLR_URL).
Smoke test without SAPIEN:  python sim_server.py --mock  [--drive]

ADAPT to your build: extract_camera_payload, _endpose_from_obs (+layout/quat),
sim_list_entities.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import os
import queue
import threading
import time
from io import BytesIO
from typing import Any, Optional

import cv2 as cv
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image


# ===========================================================================
# Pydantic models (same shape SVLR already consumes)
# ===========================================================================


class ActionPayload(BaseModel):
    model_config = {"extra": "allow"}


class SegmentEntityPayload(BaseModel):
    name: str


class OkResponse(BaseModel):
    ok: bool = True


class EndActionResponse(BaseModel):
    end_action: bool


class PoseResponse(BaseModel):
    pose: list[float] | None


class StatusResponse(BaseModel):
    action_count: int
    rejected_action_count: int
    end_action: bool
    pose: list[float] | None
    instruction: str | None
    episode_id: int
    accept_actions: bool
    done: bool
    success: bool
    stop_requested: bool
    observation_request_id: int
    observation_ready_id: int
    observation_error: str | None
    mode: str


# ===========================================================================
# ====================  ADAPT THESE TO YOUR ROBOTWIN BUILD  =================
# ===========================================================================
#
# Endpose ('ee') layout: [ left_xyz(3) left_quat(4) left_grip(1)
#                          right_xyz(3) right_quat(4) right_grip(1) ] -> 16
ENDPOSE_DIM = 16
_ARM_BASE = {"left": 0, "right": 8}
CONTROLLED_ARM = "right"            # which arm SVLR's single EE target drives
FIXED_QUAT_WXYZ = (0.7035625528174423, -6.977925221139638e-06, -3.883136134669406e-06, 0.7106333331678397)
QUAT_ORDER = "wxyz"                 # quat order in each *_endpose entry + take_action; SAPIEN=wxyz
CAMERA_KEY = "right_camera"
PLACEHOLDER_W, PLACEHOLDER_H = 640, 480

DEFAULT_ENDPOSE = np.array(
    [-2.44753662e-06, -2.09625375e-01,  1.23524601e+00,  5.31254846e-01, -4.66658013e-01,  4.66638342e-01,  5.31269465e-01, 1.00000000e+00,
     -2.44753662e-06, -2.09625375e-01,  1.23524601e+00,  5.31254846e-01, -4.66658013e-01,  4.66638342e-01,  5.31269465e-01, 1.00000000e+00], dtype=np.float64,
)

# --- Completion acknowledgement ---
# RMBench env.take_action(..., action_type="ee") is already a dense planner +
# simulator execution. The bridge therefore ACKs a low-level SVLR command after
# one successful dense take_action call, and logs measured EE error for debug.
# The tolerance/substep knobs are kept as accepted config/env fields for older
# launcher compatibility, but they no longer decide whether to replay a command.
EE_POS_TOL_M = 0.02
EE_SETTLE_EPS_M = 0.002
EE_HOLD_FRAMES = 3
MAX_SUBSTEPS_PER_ACTION = 60
MIN_SUBSTEPS_PER_ACTION = 8
FINISH_IDLE_S = 5.0          # auto-drive: after SVLR Gradio call returns and no actions arrive, end episode
EPISODE_HANDOFF_TIMEOUT_S = 300.0

# --- Fixed "home"/ready pose driven on the first step of each episode, BEFORE
# SVLR is invoked (the sim analogue of RealRobotBackend._move_to_initial_position).
# Controlled-arm target: xyz(3) + quat(4, in QUAT_ORDER) + gripper(1).  ADAPT.
HOME_ON_RESET = False
HOME_CONTROLLED = np.array([0, -0.15,  1.4,  0.5, -0.5, 0.5, 0.5, 1.0], dtype=np.float64)


def map_gripper(svlr_gripper: float) -> float:
    """SVLR gripper scalar -> RoboTwin normalized gripper command.

    SVLR/PANDA uses a physical opening width (open ~= 0.08, close = 0.0), while
    RoboTwin/SAPIEN policies usually use a normalized command (open = 1.0, close = 0.0).
    Keep already-normalized values untouched, and map small Panda-style widths to binary
    open/close so pick/place actions do not run with an almost-closed gripper.
    """
    value = float(svlr_gripper)
    if not np.isfinite(value):
        return 0.0
    if 0.0 <= value <= 0.12:
        return 1.0 if value >= 0.04 else 0.0
    return float(np.clip(value, 0.0, 1.0))


def _quat_wxyz_to_xyzw(q) -> list[float]:
    if QUAT_ORDER == "xyzw":
        return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _jpeg_b64(bgr: np.ndarray) -> str:
    ok, jpg = cv.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(jpg.tobytes()).decode("ascii")


def _intrinsics(w: int, h: int, fx=1.0, fy=1.0, ppx=None, ppy=None) -> dict:
    return {"width": int(w), "height": int(h), "fx": fx, "fy": fy,
            "ppx": w / 2.0 if ppx is None else ppx,
            "ppy": h / 2.0 if ppy is None else ppy,
            "model": "none", "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]}


def placeholder_camera_payload(w: int = PLACEHOLDER_W, h: int = PLACEHOLDER_H) -> dict:
    """A valid (black) frame so /camera/rgbd never 503s before episode 0.
    Removes the boot-time coupling between SVLR and the sim server."""
    black = np.zeros((h, w, 3), np.uint8)
    return {"ok": True, "color_bgr_jpeg_b64": _jpeg_b64(black), "depth_npy_b64": None,
            "intrinsics": _intrinsics(w, h), "camera_name": "placeholder",
            "timestamp_s": time.time(), "placeholder": True}


def _select_camera(obs: Any, preferred_key: str = CAMERA_KEY) -> tuple[str, dict]:
    """Return the requested camera when available, otherwise fall back safely.

    RMBench/Franka works best with the wrist camera for this SVLR bridge, so the
    default is right_camera. The fallback is only for configs that do not expose it.
    """
    observation = obs.get("observation", {}) if isinstance(obs, dict) else {}
    preferred = [preferred_key, "right_camera", "left_camera", "head_camera", "front_camera"]
    for key in preferred:
        if key and key in observation:
            cam = observation[key]
            if isinstance(cam, dict) and "rgb" in cam:
                return key, cam
    available = list(observation.keys())
    raise KeyError(f"No usable RGB camera found. preferred={preferred_key!r}, available={available}")


def _normalise_depth_to_m(depth: Any) -> np.ndarray:
    """Return RMBench depth in meters.

    RMBench/RoboTwin get_depth() stores depth in millimeters. SVLR expects meters.
    This auto-detect keeps the bridge safe if a future config already returns meters.
    """
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if valid.size and float(np.nanmedian(valid)) > 10.0:
        arr = arr * 0.001
    return arr.astype(np.float32, copy=False)


def _camera_object_from_env(env: Any, camera_key: str):
    """Return the live SAPIEN camera object for camera_key, if available.

    This is called only on the RMBench main/eval thread, right after env.get_obs().
    It is therefore safe to read SAPIEN camera textures here.
    """
    cams = getattr(env, "cameras", None)
    if cams is None:
        return None
    if camera_key == "left_camera" and hasattr(cams, "left_camera"):
        return cams.left_camera
    if camera_key == "right_camera" and hasattr(cams, "right_camera"):
        return cams.right_camera
    for cam, name in zip(getattr(cams, "static_camera_list", []), getattr(cams, "static_camera_name", [])):
        if name == camera_key:
            return cam
    return None


def _position_texture_from_camera(camera):
    try:
        position = np.asarray(camera.get_picture("Position"), dtype=np.float32)
        units = "m"
        valid = position[..., 3] < 1.0 if position.ndim == 3 and position.shape[-1] >= 4 else None
    except Exception:
        try:
            position = np.asarray(camera.get_picture("PositionSegmentation"), dtype=np.float32)
            units = "mm"
            valid = (
                np.linalg.norm(position[..., :3], axis=-1) > 0.0
                if position.ndim == 3 and position.shape[-1] >= 3
                else None
            )
        except Exception as exc:
            raise RuntimeError(
                'SAPIEN camera did not provide "Position" or "PositionSegmentation".'
            ) from exc

    if position.ndim != 3 or position.shape[-1] < 3:
        raise RuntimeError(f"Unexpected SAPIEN position texture shape: {position.shape}")

    xyz = position[..., :3].astype(np.float32, copy=False)
    finite = np.isfinite(xyz).all(axis=-1)
    if units == "mm":
        xyz = xyz * 0.001
    else:
        finite_values = np.abs(xyz[np.isfinite(xyz)])
        if finite_values.size and float(np.nanmedian(finite_values)) > 10.0:
            xyz = xyz * 0.001
    valid = finite if valid is None else (valid & finite)
    return xyz, valid


def _world_xyz_from_camera_object(env: Any, camera_key: str, rgb_shape) -> np.ndarray | None:
    """Dense per-pixel world XYZ from the selected RMBench/SAPIEN camera.

    This is the clean simulation calibration path: instead of trying to reuse the
    real Panda wrist-camera dx/dy/dz, the bridge asks SAPIEN for the rendered
    per-pixel camera-space position texture and transforms it with the camera model
    matrix. It works for wrist cameras too, because it is recomputed every frame.
    """
    camera = _camera_object_from_env(env, camera_key)
    if camera is None:
        return None
    try:
        pts_cam, valid = _position_texture_from_camera(camera)
        model = np.asarray(camera.get_model_matrix(), dtype=np.float32)
    except Exception as exc:
        print(f"[sim-server] could not read world XYZ for {camera_key}: {exc}")
        return None

    h, w = rgb_shape[:2]
    if pts_cam.shape[0] != h or pts_cam.shape[1] != w:
        print(
            f"[sim-server] world XYZ shape mismatch for {camera_key}: "
            f"position={pts_cam.shape[:2]}, rgb={(h, w)}"
        )
        return None

    world = pts_cam @ model[:3, :3].T + model[:3, 3]
    world = world.astype(np.float32, copy=False)
    if valid is not None:
        world[~valid] = np.nan
    return world.astype(np.float32, copy=False)


def _npy_b64(array: np.ndarray) -> str:
    buf = BytesIO()
    np.save(buf, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render_vlm_bgr_with_shader(env: Any, camera_key: str, shader_dir: str | None) -> np.ndarray | None:
    """Render an RGB-only VLM view from a pre-created shader-specific camera.

    This is intentionally separate from the RGB-D payload. The normal observation
    remains depth-compatible, while this optional pass can use a minimal clone for
    a cleaner/lighter VLM image. SAPIEN binds a camera's shader at creation time,
    so changing set_camera_shader_dir after a camera exists is not sufficient.
    """
    shader_dir = str(shader_dir or "").strip()
    if not shader_dir:
        return None

    cams = getattr(env, "cameras", None)
    get_vlm_camera = getattr(cams, "get_vlm_camera", None)
    camera = get_vlm_camera(camera_key) if callable(get_vlm_camera) else None
    if camera is None:
        print(
            f"[sim-server] no VLM camera clone for {camera_key}; "
            "restart RMBench after setting sim_vlm_camera_shader_dir."
        )
        return None

    try:
        camera.take_picture()
        rgba = np.asarray(camera.get_picture("Color"))
        rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        # A newly-created SAPIEN shader camera can intermittently return its
        # zero-initialized render target even though the normal RGB-D camera is
        # already valid.  Never forward that transient black frame to the VLM:
        # returning None makes extract_camera_payload omit the optional VLM
        # image, so SVLR safely uses the synchronized depth-camera RGB instead.
        # Keep the threshold deliberately strict so legitimate dark images are
        # not replaced merely for having low contrast.
        if bgr.size == 0 or float(np.percentile(bgr, 99.0)) <= 5.0:
            print(
                f"[sim-server] rejected near-black VLM shader frame for "
                f"{camera_key}; falling back to synchronized RGB-D color"
            )
            return None
        return bgr
    except Exception as exc:
        print(f"[sim-server] could not read VLM RGB from shader '{shader_dir}': {exc}")
        return None


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray | None:
    try:
        q = np.asarray(quat, dtype=np.float64).reshape(4)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _control_metadata_from_env(env: Any, arm: str = CONTROLLED_ARM) -> dict[str, Any]:
    """Runtime control-frame geometry for SVLR action generation.

    RMBench `ee` actions are expressed in the pose returned by get_*_ee_pose(),
    while the visually useful manipulation point is the TCP/gripper center from
    get_*_tcp_pose().  Their local offset comes from the loaded robot model, so
    SVLR does not need a task-specific or object-specific Z clamp.
    """
    robot = getattr(env, "robot", None)
    if robot is None:
        return {}
    get_ee = getattr(robot, f"get_{arm}_ee_pose", None)
    get_tcp = getattr(robot, f"get_{arm}_tcp_pose", None)
    if not callable(get_ee) or not callable(get_tcp):
        return {}
    try:
        ee_pose = np.asarray(get_ee(), dtype=np.float64).reshape(-1)
        tcp_pose = np.asarray(get_tcp(), dtype=np.float64).reshape(-1)
    except Exception as exc:
        print(f"[sim-server] could not compute control metadata for {arm}: {exc}")
        return {}
    if ee_pose.size < 7 or tcp_pose.size < 3:
        return {}

    rotation = _quat_wxyz_to_matrix(ee_pose[3:7])
    if rotation is None:
        return {}
    command_to_tcp_world = tcp_pose[:3] - ee_pose[:3]
    command_to_tcp_local = rotation.T @ command_to_tcp_world
    if not np.isfinite(command_to_tcp_local).all():
        return {}
    return {
        "arm": str(arm),
        "command_frame": "rmbench_ee_pose",
        "target_frame": "tcp",
        "command_to_tcp_local_m": command_to_tcp_local.astype(float).tolist(),
        "command_to_tcp_world_m": command_to_tcp_world.astype(float).tolist(),
        "command_to_tcp_distance_m": float(np.linalg.norm(command_to_tcp_world)),
    }


def extract_camera_payload(
    obs: Any,
    camera_key: str = CAMERA_KEY,
    env: Any | None = None,
    controlled_arm: str = CONTROLLED_ARM,
    vlm_camera_shader_dir: str | None = None,
) -> dict[str, Any]:
    """Observation -> SVLR remote RGB-D payload.

    Extra field for simulation calibration:
      world_xyz_npy_b64: HxWx3 float32, per-pixel XYZ in RMBench/world frame.

    SVLR uses that field when present. Real robot workflows ignore it.
    """
    used_camera_key, cam = _select_camera(obs, camera_key)
    rgb = np.asarray(cam["rgb"])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

    depth_b64 = None
    depth = cam.get("depth") if isinstance(cam, dict) else None
    if depth is not None:
        depth_m = _normalise_depth_to_m(depth)
        depth_b64 = _npy_b64(depth_m)

    world_xyz_b64 = None
    world_xyz = _world_xyz_from_camera_object(env, used_camera_key, rgb.shape) if env is not None else None
    if world_xyz is not None:
        world_xyz_b64 = _npy_b64(world_xyz)

    vlm_bgr = (
        _render_vlm_bgr_with_shader(env, used_camera_key, vlm_camera_shader_dir)
        if env is not None and str(vlm_camera_shader_dir or "").strip()
        else None
    )

    h, w = rgb.shape[:2]
    intr = cam.get("intrinsic_cv") if isinstance(cam, dict) else None
    if intr is not None:
        intr = np.asarray(intr, dtype=np.float64)
        intrinsics = _intrinsics(w, h, float(intr[0, 0]), float(intr[1, 1]),
                                 float(intr[0, 2]), float(intr[1, 2]))
    else:
        intrinsics = _intrinsics(w, h)

    payload = {
        "ok": True,
        "color_bgr_jpeg_b64": _jpeg_b64(bgr),
        "depth_npy_b64": depth_b64,
        "world_xyz_npy_b64": world_xyz_b64,
        "intrinsics": intrinsics,
        "control_metadata": _control_metadata_from_env(env, controlled_arm) if env is not None else {},
        "camera_name": str(used_camera_key),
        "timestamp_s": time.time(),
    }
    if vlm_bgr is not None:
        payload["vlm_color_bgr_jpeg_b64"] = _jpeg_b64(vlm_bgr)
        payload["vlm_camera_shader_dir"] = str(vlm_camera_shader_dir)
    if isinstance(cam, dict):
        if cam.get("cam2world_gl") is not None:
            payload["cam2world_gl"] = np.asarray(cam["cam2world_gl"], dtype=float).tolist()
        if cam.get("extrinsic_cv") is not None:
            payload["extrinsic_cv"] = np.asarray(cam["extrinsic_cv"], dtype=float).tolist()
    return payload

def _entry_to_xyz_quat(pose: Any) -> tuple[np.ndarray, np.ndarray]:
    """One endpose entry (left_endpose / right_endpose) -> (xyz[3], quat[4]) with
    the quat in QUAT_ORDER. The EE-position gate uses xyz only, so quat handling
    is best-effort. ADAPT if your pose is not xyz+quaternion.

    Handles: length>=7 (xyz + quat, default), length 6 (xyz + euler xyz),
    length 3 (xyz only), 4x4 homogeneous matrix.
    """
    a = np.asarray(pose, dtype=np.float64).reshape(-1)
    if a.size == 16:  # 4x4 matrix
        T = a.reshape(4, 4)
        xyz = T[:3, 3].copy()
        try:
            from scipy.spatial.transform import Rotation as R
            q_xyzw = R.from_matrix(T[:3, :3]).as_quat()
            quat = q_xyzw if QUAT_ORDER == "xyzw" else np.array([q_xyzw[3], *q_xyzw[:3]])
        except Exception:
            quat = np.array(FIXED_QUAT_WXYZ, dtype=np.float64)
        return xyz, quat
    xyz = a[:3].copy()
    if a.size >= 7:
        quat = a[3:7].copy()                 # assume already in QUAT_ORDER
    elif a.size == 6:
        try:
            from scipy.spatial.transform import Rotation as R
            q_xyzw = R.from_euler("xyz", a[3:6]).as_quat()
            quat = q_xyzw if QUAT_ORDER == "xyzw" else np.array([q_xyzw[3], *q_xyzw[:3]])
        except Exception:
            quat = np.array(FIXED_QUAT_WXYZ, dtype=np.float64)
    else:
        quat = np.array(FIXED_QUAT_WXYZ, dtype=np.float64)
    return xyz, quat


def _endpose_from_obs(obs: Any) -> Optional[np.ndarray]:
    """obs -> internal 16-dim endpose [Lxyz Lquat Lgrip Rxyz Rquat Rgrip], or None.

    Your obs["endpose"] is a dict: left_endpose / left_gripper / right_endpose /
    right_gripper. Also tolerates a flat 16/14 array for other configs.
    """
    try:
        ep = obs["endpose"]
    except (KeyError, TypeError):
        return None

    out = np.empty(ENDPOSE_DIM, dtype=np.float64)
    if isinstance(ep, dict):
        try:
            lx, lq = _entry_to_xyz_quat(ep["left_endpose"])
            rx, rq = _entry_to_xyz_quat(ep["right_endpose"])
            lg = float(np.asarray(ep["left_gripper"], dtype=np.float64).reshape(-1)[0])
            rg = float(np.asarray(ep["right_gripper"], dtype=np.float64).reshape(-1)[0])
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        out[0:3], out[3:7], out[7] = lx, lq, lg
        out[8:11], out[11:15], out[15] = rx, rq, rg
        return out

    # Fallback: flat array layouts.
    try:
        flat = np.asarray(ep, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if flat.size == ENDPOSE_DIM:
        return flat.copy()
    if flat.size == 14:
        out[:] = DEFAULT_ENDPOSE
        out[0:7], out[8:15] = flat[0:7], flat[7:14]
        return out
    return None


def svlr_action_has_position(svlr_action: dict) -> bool:
    """True when an SVLR low-level command includes an EE target XYZ.

    Gripper-only commands such as {"gripper": 0.0} must still be executed for
    several SAPIEN substeps, but they should not invent or require a new target
    position.
    """
    return (
        "pos_end_effector" in svlr_action
        or all(k in svlr_action for k in ("ee.x", "ee.y", "ee.z"))
    )


def build_take_action(
    svlr_action: dict,
    cmd: np.ndarray,
    arm: str = CONTROLLED_ARM,
    mirror_single_arm: bool = False,
) -> np.ndarray:
    base = _ARM_BASE[arm]
    out = cmd.copy()
    print("[sim-server] SVLR action:", svlr_action)

    # SVLR low-level primitives may be gripper-only, e.g. {"gripper": 0.0}.
    # Keep the previous EE target in that case; do not crash and do not invent XYZ.
    if svlr_action_has_position(svlr_action):
        pos = svlr_action.get("pos_end_effector")
        if pos is None:
            pos = [svlr_action["ee.x"], svlr_action["ee.y"], svlr_action["ee.z"]]

        out[base + 0:base + 3] = [float(pos[0]), float(pos[1]), float(pos[2])]

        # IMPORTANT for wrist-camera RMBench runs:
        # The EE orientation is part of the camera view.  Do not force a generic
        # quaternion here.  The RMBENCH SVLR profile emits the calibrated init
        # quaternion for every waypoint, so movements and final go_init keep the
        # wrist camera in the same useful view frame.
        if len(pos) >= 7:
            out[base + 3:base + 7] = [float(pos[3]), float(pos[4]), float(pos[5]), float(pos[6])]
        else:
            # Legacy fallback: preserve the previously commanded orientation.
            out[base + 3:base + 7] = cmd[base + 3:base + 7]
    else:
        print("[sim-server] gripper-only action: keeping previous EE target pose")

    if "gripper" in svlr_action or "ee.gripper_pos" in svlr_action:
        raw = svlr_action.get("gripper", svlr_action.get("ee.gripper_pos"))
        out[base + 7] = map_gripper(float(raw))

    if mirror_single_arm:
        # demo_clean_franka represents one physical Franka through RMBench's
        # left/right 16D layout. If only the controlled slot changes, take_action
        # plans a "hold" trajectory for the other slot and a "move" trajectory
        # for the same underlying robot, which produces conflicting commands.
        other_arm = "left" if arm == "right" else "right"
        other_base = _ARM_BASE[other_arm]
        out[other_base:other_base + 8] = out[base:base + 8]
        print(
            f"[sim-server] mirrored {arm} command to {other_arm} "
            "for single-arm RMBench embodiment"
        )
    return out

def requested_ee_xyz(cmd: np.ndarray, arm: str = CONTROLLED_ARM) -> np.ndarray:
    base = _ARM_BASE[arm]
    return cmd[base:base + 3].astype(np.float64).copy()


def measured_ee_xyz(obs: Any, arm: str = CONTROLLED_ARM) -> Optional[np.ndarray]:
    """Measured EE position of the controlled arm from obs, or None if unavailable."""
    m = _endpose_from_obs(obs)
    if m is None:
        return None
    base = _ARM_BASE[arm]
    return m[base:base + 3].astype(np.float64).copy()


def pose_for_svlr(cmd: np.ndarray, arm: str = CONTROLLED_ARM) -> list[float]:
    base = _ARM_BASE[arm]
    xyz, quat = cmd[base:base + 3], cmd[base + 3:base + 7]
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), *_quat_wxyz_to_xyzw(quat)]


def sim_list_entities(env: Any) -> set[str]:
    names: set[str] = set()
    try:
        for actor in env.scene.get_all_actors():
            names.add(actor.get_name())
    except Exception:
        pass
    return names


# ===========================================================================
# Thread-safe bridge (pre-seeded so it answers from bind-time)
# ===========================================================================


class SimBridge:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._action_q: "queue.Queue[dict]" = queue.Queue()
        self._end_action = False
        self._instruction: Optional[str] = None
        # Pre-seed: server is fully answerable before episode 0 -> no boot deadlock.
        self._pose: list[float] = pose_for_svlr(DEFAULT_ENDPOSE)
        self._camera: dict = placeholder_camera_payload()
        self._entities: set[str] = set()
        self._action_count = 0
        self._rejected_action_count = 0
        self._episode_id = 0
        self._accept_actions = False
        self._done = False
        self._success = False
        self._observation_request_id = 0
        self._observation_ready_id = 0
        self._observation_error: str | None = None
        self.stop_requested = False

    def _clear_action_queue_locked(self) -> None:
        with self._action_q.mutex:
            self._action_q.queue.clear()

    # producer (harness/main thread)
    def begin_episode(self) -> int:
        with self._lock:
            # Drop any stale actions from the prior episode and keep the action
            # gate closed until SVLR has acknowledged /reset_episode.
            self._clear_action_queue_locked()
            self._end_action = False
            self._done = self._success = False
            self._action_count = 0
            self._rejected_action_count = 0
            self._observation_request_id = 0
            self._observation_ready_id = 0
            self._observation_error = None
            self._episode_id += 1
            self._accept_actions = False
            self.stop_requested = False
            return self._episode_id

    def publish(self, pose, camera, entities) -> None:
        with self._lock:
            self._pose, self._camera, self._entities = pose, camera, entities

    def set_end_action(self, v: bool) -> None:
        with self._lock:
            self._end_action = v

    def open_action_window(self, episode_id: int | None = None) -> bool:
        with self._lock:
            if episode_id is not None and int(episode_id) != self._episode_id:
                return False
            self._accept_actions = True
            self._end_action = True
            return True

    def set_instruction(self, s: str) -> None:
        with self._lock:
            self._instruction = s

    def set_done(self, success: bool) -> None:
        with self._lock:
            self._accept_actions = False
            self._end_action = False
            self._clear_action_queue_locked()
            self._done, self._success = True, success

    def request_stop(self) -> None:
        with self._lock:
            self.stop_requested = True
            self._accept_actions = False
            self._end_action = False
            self._clear_action_queue_locked()

    def episode_id(self) -> int:
        with self._lock:
            return int(self._episode_id)

    def pop_action(self, timeout: float) -> Optional[dict]:
        try:
            return self._action_q.get(timeout=timeout)
        except queue.Empty:
            return None

    # consumer (uvicorn thread)
    def send_action(self, payload: dict) -> bool:
        with self._lock:
            payload = dict(payload)
            payload_episode_id = payload.pop("episode_id", None)
            try:
                stale_episode = (
                    payload_episode_id is not None
                    and int(payload_episode_id) != self._episode_id
                )
            except (TypeError, ValueError):
                stale_episode = True
            if stale_episode:
                self._rejected_action_count += 1
                print(
                    "[sim-server] ignoring stale SVLR action "
                    f"for episode_id={payload_episode_id}; "
                    f"current_episode_id={self._episode_id} "
                    f"(rejected={self._rejected_action_count})"
                )
                return False
            if self.stop_requested or not self._accept_actions or self._done:
                self._rejected_action_count += 1
                print(
                    "[sim-server] ignoring SVLR action outside active episode "
                    f"(rejected={self._rejected_action_count}): {payload}"
                )
                return False
            self._end_action = False
            self._action_count += 1
            self._action_q.put(payload)
            return True

    def request_observation_pose(self, episode_id: int | None = None) -> dict:
        """Queue a camera-pose transition without recording a task action."""
        with self._lock:
            try:
                stale_episode = (
                    episode_id is not None
                    and int(episode_id) != self._episode_id
                )
            except (TypeError, ValueError):
                stale_episode = True
            if (
                stale_episode
                or self.stop_requested
                or not self._accept_actions
                or self._done
            ):
                return {
                    "ok": False,
                    "error": "observation pose requested outside active episode",
                }
            self._observation_request_id += 1
            request_id = self._observation_request_id
            self._observation_error = None
            self._action_q.put(
                {
                    "_bridge_command": "prepare_observation",
                    "episode_id": self._episode_id,
                    "request_id": request_id,
                }
            )
            return {"ok": True, "request_id": request_id}

    def finish_observation_pose(
        self,
        episode_id: int,
        request_id: int,
        error: str | None = None,
    ) -> None:
        with self._lock:
            if int(episode_id) != self._episode_id:
                return
            if error:
                self._observation_error = str(error)
                return
            self._observation_ready_id = max(
                self._observation_ready_id, int(request_id)
            )
            self._observation_error = None

    def reset_end_action(self) -> None:
        with self._lock:
            self._end_action = False

    def is_end_action(self) -> bool:
        with self._lock:
            return self._end_action

    def get_pose(self):
        with self._lock:
            return list(self._pose)

    def get_camera(self):
        with self._lock:
            return self._camera

    def segment(self, name: str) -> bool:
        with self._lock:
            return name in self._entities

    def action_count(self) -> int:
        with self._lock:
            return int(self._action_count)

    def status(self) -> dict:
        with self._lock:
            return {"action_count": self._action_count, "end_action": self._end_action,
                    "pose": list(self._pose), "instruction": self._instruction,
                    "episode_id": self._episode_id,
                    "done": self._done, "success": self._success,
                    "accept_actions": self._accept_actions,
                    "rejected_action_count": self._rejected_action_count,
                    "observation_request_id": self._observation_request_id,
                    "observation_ready_id": self._observation_ready_id,
                    "observation_error": self._observation_error,
                    "stop_requested": self.stop_requested}


# ===========================================================================
# FastAPI app
# ===========================================================================


def create_app(bridge: SimBridge) -> FastAPI:
    app = FastAPI(title="Simulation web server", version="1.0.0")

    @app.get("/robot_pose", response_model=PoseResponse)
    async def robot_pose():
        return {"pose": bridge.get_pose()}

    @app.get("/end_action", response_model=EndActionResponse)
    async def end_action():
        return {"end_action": bridge.is_end_action()}

    @app.get("/status", response_model=StatusResponse)
    async def status():
        return {**bridge.status(), "mode": "sim"}

    @app.get("/instruction")
    async def instruction():
        return {"instruction": bridge.status()["instruction"]}

    @app.get("/camera/rgbd")
    async def camera_rgbd():
        payload = bridge.get_camera()
        if payload is None:
            raise HTTPException(status_code=503, detail="no frame published yet")
        return JSONResponse(payload)

    @app.post("/send_action", response_model=OkResponse)
    async def send_action(payload: ActionPayload):
        return {"ok": bridge.send_action(payload.model_dump())}

    @app.post("/reset_end_action", response_model=OkResponse)
    async def reset_end_action():
        bridge.reset_end_action()
        return {"ok": True}

    @app.post("/prepare_observation")
    async def prepare_observation(payload: ActionPayload):
        return bridge.request_observation_pose(
            payload.model_dump().get("episode_id")
        )

    @app.post("/segment_entity")
    async def segment_entity(payload: SegmentEntityPayload):
        return {"found": bridge.segment(payload.name), "name": payload.name}

    @app.post("/stop", response_model=OkResponse)
    async def stop():
        bridge.request_stop()
        return {"ok": True, "message": "stopping current episode"}

    return app


# ===========================================================================
# Server + per-step driver + optional SVLR Gradio driver thread
# ===========================================================================


class SimServer:
    def __init__(self, host="0.0.0.0", port=65500, controlled_arm=CONTROLLED_ARM,
                 action_poll_s=0.1, drive=False, svlr_url="http://127.0.0.1:7860",
                 camera_key=CAMERA_KEY, save_debug_images=False,
                 call_vlm_before_llm=True,
                 ee_pos_tol_m=EE_POS_TOL_M, ee_settle_eps_m=EE_SETTLE_EPS_M,
                 ee_hold_frames=EE_HOLD_FRAMES,
                 max_substeps_per_action=MAX_SUBSTEPS_PER_ACTION,
                 min_substeps_per_action=MIN_SUBSTEPS_PER_ACTION,
                 finish_idle_s=FINISH_IDLE_S,
                 episode_handoff_timeout_s=EPISODE_HANDOFF_TIMEOUT_S,
                 home_on_reset=HOME_ON_RESET, home_controlled=None,
                 mirror_single_arm=False,
                 keep_alive_after_actions=False,
                 vlm_camera_shader_dir: str | None = None,
                 debug_dir=".") -> None:
        self.host, self.port = host, port
        self.controlled_arm = controlled_arm
        self.action_poll_s = float(action_poll_s)
        self.drive, self.svlr_url = drive, svlr_url
        self.camera_key = str(camera_key or CAMERA_KEY)
        self.save_debug_images = bool(save_debug_images)
        self.call_vlm_before_llm = bool(call_vlm_before_llm)
        self.ee_pos_tol_m = ee_pos_tol_m
        self.ee_settle_eps_m = ee_settle_eps_m
        self.ee_hold_frames = int(ee_hold_frames)
        self.max_substeps_per_action = int(max_substeps_per_action)
        self.min_substeps_per_action = int(min_substeps_per_action)
        self.finish_idle_s = float(finish_idle_s)
        self.episode_handoff_timeout_s = float(episode_handoff_timeout_s)
        self.mirror_single_arm = bool(mirror_single_arm)
        self.keep_alive_after_actions = bool(keep_alive_after_actions)
        self.vlm_camera_shader_dir = str(vlm_camera_shader_dir or "").strip()
        self.debug_dir = os.path.abspath(os.path.expanduser(str(debug_dir or ".")))
        if self.save_debug_images:
            os.makedirs(self.debug_dir, exist_ok=True)
        if self.finish_idle_s < 0.0:
            raise ValueError("finish_idle_s must be >= 0")
        if self.episode_handoff_timeout_s <= 0.0:
            raise ValueError("episode_handoff_timeout_s must be > 0")
        self._driver_state_lock = threading.Lock()
        self._driver_finished_event = threading.Event()
        self._driver_episode_id = 0
        self._driver_started = False
        self._driver_finished = False
        self._driver_failed = False
        if self.max_substeps_per_action < 1:
            raise ValueError("max_substeps_per_action must be >= 1")
        if self.min_substeps_per_action < 1:
            raise ValueError("min_substeps_per_action must be >= 1")
        if self.min_substeps_per_action > self.max_substeps_per_action:
            print(
                f"[sim-server] min_substeps_per_action ({self.min_substeps_per_action}) "
                f"> max_substeps_per_action ({self.max_substeps_per_action}); clamping to max"
            )
            self.min_substeps_per_action = self.max_substeps_per_action
        self.home_on_reset = bool(home_on_reset)

        # HOME_CONTROLLED is the physical start and perception pose for the
        # single arm when --sim_home is enabled.
        self.home_controlled = np.asarray(
            HOME_CONTROLLED if home_controlled is None else home_controlled,
            dtype=np.float64,
        ).reshape(-1)
        assert self.home_controlled.size == 8, "home_controlled must be xyz(3)+quat(4)+gripper(1)"
        self.bridge = SimBridge()
        self.app = create_app(self.bridge)
        self._cmd: Optional[np.ndarray] = None
        self._observation_cmd: Optional[np.ndarray] = None
        self._uv: Optional[uvicorn.Server] = None
        self._uv_thread: Optional[threading.Thread] = None
        self._episode_q: "queue.Queue[tuple[int, str]]" = queue.Queue()
        self._driver_thread: Optional[threading.Thread] = None
        self._shutdown = False

    def start(self) -> None:
        if self._uv_thread is not None:
            return
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        self._uv = uvicorn.Server(config)
        self._uv.install_signal_handlers = lambda: None
        self._uv_thread = threading.Thread(target=self._uv.run, daemon=True)
        self._uv_thread.start()
        while not getattr(self._uv, "started", False):
            time.sleep(0.01)
        print(f"[sim-server] http://{self.host}:{self.port}  "
              f"(arm: {self.controlled_arm}, camera: {self.camera_key})  "
              f"drive={'on->' + self.svlr_url if self.drive else 'off'}  "
              f"mirror_single_arm={self.mirror_single_arm}  "
              f"keep_alive_after_actions={self.keep_alive_after_actions}  "
              f"vlm_shader={self.vlm_camera_shader_dir or 'same'}")
        if self.drive:
            self._driver_thread = threading.Thread(target=self._drive_loop, daemon=True)
            self._driver_thread.start()

    def _mark_driver_started(self, episode_id: int) -> None:
        with self._driver_state_lock:
            self._driver_episode_id = int(episode_id)
            self._driver_started = True
            self._driver_finished = False
            self._driver_failed = False
            self._driver_finished_event.clear()

    def _mark_driver_finished(
        self,
        episode_id: int,
        failed: bool = False,
    ) -> None:
        with self._driver_state_lock:
            if int(episode_id) != self._driver_episode_id:
                print(
                    "[driver] ignoring stale completion for "
                    f"episode_id={episode_id}; "
                    f"current_episode_id={self._driver_episode_id}"
                )
                return
            self._driver_finished = True
            self._driver_failed = bool(failed)
            self._driver_finished_event.set()

    def _driver_done(self, episode_id: int | None = None) -> tuple[bool, bool]:
        with self._driver_state_lock:
            if (
                episode_id is not None
                and int(episode_id) != self._driver_episode_id
            ):
                return False, False
            return bool(self._driver_finished), bool(self._driver_failed)

    def _wait_for_driver_handoff(self, timeout_s: float | None = None) -> None:
        """Wait until the previous episode's blocking SVLR request has returned."""

        with self._driver_state_lock:
            if not self._driver_started or self._driver_finished:
                return
            episode_id = self._driver_episode_id
            finished_event = self._driver_finished_event

        # A new evaluator reset proves that the previous episode ended. A
        # natural step-limit exit may not have published a terminal bridge
        # verdict yet, so latch failure before waiting for SVLR to unwind.
        if not self.bridge.status().get("done", False):
            print(
                "[sim-server] previous episode ended without a terminal bridge "
                f"verdict; latching failure for episode_id={episode_id}"
            )
            self.bridge.set_done(False)

        timeout = (
            self.episode_handoff_timeout_s
            if timeout_s is None
            else float(timeout_s)
        )
        print(
            "[sim-server] waiting for SVLR driver handoff before resetting "
            f"episode_id={episode_id}"
        )
        if not finished_event.wait(timeout=max(0.0, timeout)):
            raise RuntimeError(
                "SVLR driver did not acknowledge terminal episode "
                f"episode_id={episode_id} within {timeout:.1f}s"
            )
        print(
            "[sim-server] SVLR driver handoff complete "
            f"episode_id={episode_id}"
        )

    def _reset_svlr_episode(
        self,
        client: Any,
        timeout_s: float = 20.0,
    ) -> dict[str, Any]:
        """Force-clear SVLR's per-episode state before running VLM/LLM."""
        deadline = time.monotonic() + timeout_s
        last_status: dict[str, Any] | None = None
        last_error: Exception | None = None

        while time.monotonic() < deadline and not self._shutdown:
            try:
                result = client.predict(api_name="/reset_episode")
                if isinstance(result, dict):
                    status = result
                elif (
                    isinstance(result, (list, tuple))
                    and len(result) == 1
                    and isinstance(result[0], dict)
                ):
                    status = result[0]
                else:
                    status = {
                        "ok": False,
                        "reason": f"unexpected_response:{result!r}",
                    }

                last_status = status
                if status.get("ok") is True:
                    print(
                        "[driver] SVLR episode reset confirmed "
                        f"(episode_id={status.get('episode_id')})"
                    )
                    return status

                print(f"[driver] SVLR reset not ready: {status}")
                time.sleep(0.5)
            except Exception as exc:
                last_error = exc
                print(f"[driver] SVLR reset call failed; retrying: {exc}")
                time.sleep(0.5)

        detail = last_status if last_status is not None else repr(last_error)
        raise RuntimeError(f"SVLR episode reset was not confirmed: {detail}")

    def _run_svlr_episode(
        self,
        client: Any,
        episode_id: int,
        instruction: str,
    ) -> None:
        """Run one SVLR request against frames from the matching episode."""

        print("[driver] resetting SVLR session")
        self._reset_svlr_episode(client)
        if not self.bridge.open_action_window(episode_id):
            raise RuntimeError(
                "stale SVLR driver request for "
                f"episode_id={episode_id}; "
                f"current_episode_id={self.bridge.episode_id()}"
            )
        print("[driver] SVLR action gate opened")
        if self.call_vlm_before_llm:
            print("[driver] running SVLR VLM perception")
            client.predict(api_name="/process_vlm")
        print(f"[driver] running SVLR LLM command: {instruction}")
        client.predict(prompt=instruction, api_name="/process_llm_command")

    # -- SVLR driver: retries connect, fires perception + language per episode --
    def _drive_loop(self) -> None:
        try:
            from gradio_client import Client
        except ImportError:
            print("[driver] gradio_client not installed; SVLR auto-drive disabled")
            return
        client = None
        while not self._shutdown:
            if client is None:
                try:
                    client = Client(
                        self.svlr_url,
                        httpx_kwargs={"timeout": None},
                    )
                    print(f"[driver] connected to SVLR at {self.svlr_url}")
                except Exception:
                    time.sleep(1.0)
                    continue
            try:
                episode_id, instruction = self._episode_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                # Blocks for the whole episode while SVLR drives :65500.
                # RMBench episodes can change the scene while SVLR keeps its
                # Gradio process alive. Reset SVLR's per-episode memory before
                # perception so WorldMemory/actions from the prior episode do
                # not leak into the new task.
                self._run_svlr_episode(client, episode_id, instruction)
                # The Gradio call returning means SVLR has generated its low-level
                # queue. Execution may still be draining through /send_action, so
                # do NOT stop immediately; step() will finish only after an idle
                # grace period with no new actions.
            except Exception as e:
                print(
                    "[driver] SVLR command failed: "
                    f"{type(e).__name__}: {e!r}"
                )
                client = None  # force reconnect next round
                driver_failed = True
            else:
                driver_failed = False
            finally:
                self._mark_driver_finished(episode_id, driver_failed)

    # -- per-episode reset --
    def reset_episode(self, timeout_s: float | None = None) -> int:
        self._wait_for_driver_handoff(timeout_s=timeout_s)
        self._cmd = None
        self._observation_cmd = None
        episode_id = self.bridge.begin_episode()
        with self._driver_state_lock:
            self._driver_episode_id = episode_id
            self._driver_started = False
            self._driver_finished = False
            self._driver_failed = False
            self._driver_finished_event.clear()
        return episode_id

    def _save_debug_camera_images(self, obs: Any, suffix: str) -> None:
        if not self.save_debug_images:
            return
        try:
            observation = obs.get("observation", {}) if isinstance(obs, dict) else {}
            saved_any = False
            for key, cam in observation.items():
                if not isinstance(cam, dict) or "rgb" not in cam:
                    continue
                rgb = np.asarray(cam["rgb"])
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                out_path = os.path.join(
                    self.debug_dir, f"svlr_bridge_{key}_{suffix}.png"
                )
                Image.fromarray(rgb).save(out_path)
                print(f"[sim-server] saved {out_path}")
                saved_any = True
            if not saved_any:
                print(f"[sim-server] no RGB cameras found to save; keys={list(observation.keys())}")
        except Exception as exc:
            print(f"[sim-server] could not save debug camera images: {exc}")

    def _pump_idle_viewer(self, env: Any, fps: float = 30.0) -> None:
        """Keep the SAPIEN viewer interactive while waiting for SVLR/curl actions.

        Important: this does NOT call env.take_action() and does NOT advance a
        robot command. It only refreshes renderer/viewer events so the user can
        move the SAPIEN UI camera while the bridge is idle.
        """
        if not getattr(env, "render_freq", 0):
            return
        viewer = getattr(env, "viewer", None)
        if viewer is None:
            return
        try:
            if hasattr(env, "_update_render"):
                env._update_render()
            elif hasattr(env, "scene"):
                env.scene.update_render()
            viewer.render()
        except Exception as exc:
            # Do not kill the episode just because the debug viewer had an issue.
            print(f"[sim-server] idle viewer render warning: {exc}")

    # -- one eval() == one step --
    def step(self, env: Any, observation: Any) -> None:
        self._save_debug_camera_images(observation, f"step{env.take_action_cnt}")
        print("[sim-server] end pose:", _endpose_from_obs(observation))

        if self._cmd is None:
            # First step of the episode: home the arm BEFORE engaging SVLR, then
            # publish the homed frame and hand the instruction to the driver.
            self._home_and_engage(env, observation)
            self._save_debug_camera_images(env.get_obs(), f"home{env.take_action_cnt}")
            if bool(getattr(env, "eval_success", False)):
                return
        else:
            measured = _endpose_from_obs(observation)
            pose = pose_for_svlr(measured if measured is not None else self._cmd, self.controlled_arm)
            self.bridge.publish(
                pose,
                extract_camera_payload(
                    observation,
                    self.camera_key,
                    env=env,
                    controlled_arm=self.controlled_arm,
                    vlm_camera_shader_dir=self.vlm_camera_shader_dir,
                ),
                sim_list_entities(env),
            )

        action = None
        wait_started = time.monotonic()
        last_wait_log = 0.0
        driver_finished_since: Optional[float] = None
        keep_alive_logged = False

        # Keep SAPIEN viewer responsive while waiting for SVLR/curl.
        # action_poll_s=0.1 gives only ~10 Hz max, so use a shorter queue timeout
        # during idle. Override with SIM_IDLE_RENDER_FPS=60 if needed.
        try:
            idle_render_fps = float(os.environ.get("SIM_IDLE_RENDER_FPS", "30"))
        except Exception:
            idle_render_fps = 30.0
        idle_render_fps = max(1.0, min(120.0, idle_render_fps))
        idle_render_dt = 1.0 / idle_render_fps
        poll_timeout = min(float(self.action_poll_s), idle_render_dt)
        last_idle_render = 0.0

        while not self.bridge.stop_requested:
            action = self.bridge.pop_action(timeout=poll_timeout)
            if action is not None:
                if action.get("_bridge_command") == "prepare_observation":
                    request_episode_id = int(action.get("episode_id", -1))
                    request_id = int(action.get("request_id", -1))
                    try:
                        restored_observation = self._restore_observation_pose(env)
                        self._save_debug_camera_images(
                            restored_observation,
                            f"observation{request_id}",
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        print(
                            "[sim-server] observation pose restore failed: "
                            f"{error}"
                        )
                        self.bridge.finish_observation_pose(
                            request_episode_id,
                            request_id,
                            error=error,
                        )
                    else:
                        self.bridge.finish_observation_pose(
                            request_episode_id,
                            request_id,
                        )
                    action = None
                    wait_started = time.monotonic()
                    continue
                break
            now = time.monotonic()
            if now - last_idle_render >= idle_render_dt:
                self._pump_idle_viewer(env, fps=idle_render_fps)
                last_idle_render = now
            driver_finished, driver_failed = self._driver_done(
                self.bridge.episode_id()
            )
            if self.drive and driver_finished:
                if driver_finished_since is None:
                    driver_finished_since = now
                    print(
                        "[sim-server] SVLR driver returned; waiting for queued "
                        "low-level actions before ending episode"
                    )
                idle_s = now - driver_finished_since
                if idle_s >= self.finish_idle_s:
                    actions_seen = self.bridge.action_count()
                    if (
                        self.keep_alive_after_actions
                        and actions_seen > 0
                        and not driver_failed
                    ):
                        if not keep_alive_logged:
                            print(
                                "[sim-server] SVLR action sequence is idle; "
                                "keeping RMBench alive for inspection "
                                "(POST /stop or Ctrl-C to end)"
                            )
                            keep_alive_logged = True
                        continue
                    if actions_seen == 0:
                        print(
                            f"[sim-server] SVLR driver finished but produced no action "
                            f"after {idle_s:.1f}s post-driver idle; ending episode "
                            f"(driver_failed={driver_failed})"
                        )
                    else:
                        print(
                            f"[sim-server] SVLR driver finished and no new action arrived "
                            f"for {idle_s:.1f}s post-driver idle; ending episode "
                            f"(driver_failed={driver_failed}, actions={actions_seen})"
                        )
                    self.bridge.set_done(
                        bool(getattr(env, "eval_success", False))
                        and not driver_failed
                        and actions_seen > 0
                    )
                    with contextlib.suppress(Exception):
                        env.take_action_cnt = env.step_lim
                    return
            else:
                driver_finished_since = None
                keep_alive_logged = False
            if now - last_wait_log >= 10.0:
                print(f"[sim-server] waiting for SVLR action... "
                      f"({now - wait_started:.1f}s; POST /stop or Ctrl-C to end)")
                last_wait_log = now
        if action is None:
            # Only an explicit /stop request ends the episode here.
            print("[sim-server] stop requested; ending current episode")
            self.bridge.set_done(False)
            with contextlib.suppress(Exception):
                env.take_action_cnt = env.step_lim
            return

        # From this point until _execute_until_ee_reached returns, /end_action must
        # stay false. SVLR should only receive the next completion acknowledgement
        # once RMBench/SAPIEN has physically advanced the command.
        self.bridge.set_end_action(False)
        has_position_target = svlr_action_has_position(action)
        self._cmd = build_take_action(
            action,
            self._cmd,
            self.controlled_arm,
            mirror_single_arm=self.mirror_single_arm,
        )
        run_for, action_observation = self._execute_until_ee_reached(
            env,
            has_position_target=has_position_target,
        )
        with contextlib.suppress(Exception):
            # _execute_until_ee_reached already acquired the synchronized fresh
            # observation used for the bridge publish. Reuse it for debug output
            # instead of asking SAPIEN to render the same frame a second time.
            self._save_debug_camera_images(
                action_observation, f"after{env.take_action_cnt}"
            )
        print(f"[sim-server] take_action complete: {self._cmd}  (ran {run_for} substeps)")

        if bool(getattr(env, "eval_success", False)):
            # The fresh post-action observation was already published by
            # _execute_until_ee_reached; do not perform another renderer read on
            # the terminal path before returning the validator verdict.
            self.bridge.set_done(True)
            return

        # ACK the low-level SVLR command only after the sim has advanced it.
        self.bridge.set_end_action(True)

    @staticmethod
    def _apply_initial_observation_transition(env: Any) -> bool:
        """Run an optional environment transition after its snapshot is stored."""

        hook = getattr(env, "on_initial_observation_published", None)
        if hook is None:
            return False
        if not callable(hook):
            raise TypeError("on_initial_observation_published must be callable")
        hook()
        print("[sim-server] applied initial-observation transition hook")
        return True

    # -- first step: drive to the fixed home pose, then publish + engage SVLR --
    def _home_and_engage(self, env: Any, observation: Any) -> None:
        # Seed the command from the measured pose so the un-driven arm is held put.
        measured = _endpose_from_obs(observation)
        self._cmd = (measured if measured is not None else DEFAULT_ENDPOSE).copy()

        if self.home_on_reset:
            print(f"[sim-server] homing {self.controlled_arm} arm to {self.home_controlled}")
            base = _ARM_BASE[self.controlled_arm]
            self._cmd[base:base + 8] = self.home_controlled
            if self.mirror_single_arm:
                other_arm = "left" if self.controlled_arm == "right" else "right"
                other_base = _ARM_BASE[other_arm]
                self._cmd[other_base:other_base + 8] = self.home_controlled
            _run_for, pub_obs = self._execute_until_ee_reached(
                env,
                evaluate_success=False,
            )
        else:
            pub_obs = env.get_obs()

        # Publish one synchronized observation only after the physical home
        # transition has completed. The driver is queued below, so VLM and
        # Grounded-SAM2 cannot race the home motion or read a cached pre-home cue.
        m = _endpose_from_obs(pub_obs)
        # Replaying the command that produced the first frame is more stable
        # than targeting its measured endpoint: the dense controller has a
        # small repeatable tracking offset, so feeding that measurement back as
        # the next target would shift the wrist camera on every observation.
        self._observation_cmd = np.asarray(self._cmd, dtype=np.float64).copy()
        pose = pose_for_svlr(m if m is not None else self._cmd, self.controlled_arm)
        self.bridge.publish(
            pose,
            extract_camera_payload(
                pub_obs,
                self.camera_key,
                env=env,
                controlled_arm=self.controlled_arm,
                vlm_camera_shader_dir=self.vlm_camera_shader_dir,
            ),
            sim_list_entities(env),
        )
        self._save_debug_camera_images(pub_obs, "initial_observation")
        print(
            "[sim-server] synchronized initial perception frame published; "
            f"take_action_cnt={getattr(env, 'take_action_cnt', None)}"
        )

        # The bridge owns an encoded copy of the RGB-D payload at this point.
        # Environments such as observe_and_pickup may now change the visible
        # scene without altering the snapshot consumed by the first VLM call.
        transitioned = self._apply_initial_observation_transition(env)
        if transitioned and self.save_debug_images:
            self._save_debug_camera_images(
                env.get_obs(), "after_initial_observation_transition"
            )

        # If a dense home action already satisfied the task (or an explicit debug
        # success gate did), finish immediately instead of launching SVLR and
        # waiting for unnecessary actions.
        if os.environ.get("RMBENCH_SWAP_DEBUG_SUCCESS", "").strip().lower() in {"1", "true", "yes", "on"}:
            with contextlib.suppress(Exception):
                env.max_reward = max(float(getattr(env, "max_reward", 0.0)), 1.0)
                env.eval_success = True
            self.bridge.set_done(True)
            return
        if bool(getattr(env, "eval_success", False)):
            self.bridge.set_done(True)
            return

        with contextlib.suppress(Exception):
            instr = env.get_instruction()
            self.bridge.set_instruction(instr)
            if self.drive:
                episode_id = self.bridge.episode_id()
                self._mark_driver_started(episode_id)
                self._episode_q.put(
                    (episode_id, instr)
                )   # driver fires /process_vlm + /process_llm_command now
            else:
                self.bridge.open_action_window()

    def _restore_observation_pose(self, env: Any) -> Any:
        """Replay the command that produced the first episode viewpoint."""
        if self._observation_cmd is None:
            raise RuntimeError("initial observation pose is unavailable")
        self._cmd = self._observation_cmd.copy()
        run_for, observation = self._execute_until_ee_reached(
            env,
            evaluate_success=False,
        )
        print(
            "[sim-server] restored initial observation pose "
            f"(ran {run_for} substeps)"
        )
        return observation

    # -- execute exactly one RMBench dense action per SVLR low-level command --
    def _execute_until_ee_reached(
        self,
        env: Any,
        has_position_target: bool = True,
        evaluate_success: bool = True,
    ) -> tuple[int, Any]:
        """Execute one SVLR low-level command as one RMBench dense action.

        Important RMBench/SAPIEN detail: env.take_action(..., action_type="ee")
        is already a dense execution. It calls the internal planner and runs the
        resulting joint/gripper trajectory through many simulator physics steps.

        The previous bridge repeatedly re-issued the same EE target until the
        measured EE error was below a tolerance. That made the arm look like it
        was correcting/replanning in a loop, especially near contacts such as a
        button press where the exact requested pose may be physically blocked.

        New contract:
          - one SVLR /send_action  ->  one env.take_action(..., "ee")
          - publish the fresh observation
          - acknowledge /end_action immediately after that dense execution returns

        EE error is kept only as debug information; it no longer decides whether
        to replay the same command.
        """
        target = requested_ee_xyz(self._cmd, self.controlled_arm)

        # One RMBench dense action. This may internally execute many physics
        # steps, but the bridge should not re-send the same target again.
        env.take_action(
            self._cmd,
            action_type="ee",
            evaluate_success=evaluate_success,
        )

        observation = env.get_obs()
        measured = _endpose_from_obs(observation)
        pose = pose_for_svlr(measured if measured is not None else self._cmd, self.controlled_arm)
        self.bridge.publish(
            pose,
            extract_camera_payload(
                observation,
                self.camera_key,
                env=env,
                controlled_arm=self.controlled_arm,
                vlm_camera_shader_dir=self.vlm_camera_shader_dir,
            ),
            sim_list_entities(env),
        )

        if has_position_target:
            meas = measured_ee_xyz(observation, self.controlled_arm)
            if meas is None:
                print("[sim-server] one-shot position action complete; EE pose unavailable")
            else:
                dist = float(np.linalg.norm(meas - target))
                print(
                    f"[sim-server] one-shot position action complete: "
                    f"ee_error={dist:.4f}m (debug only)"
                )
        else:
            print("[sim-server] one-shot gripper action complete")

        return 1, observation


# ===========================================================================
# RoboTwin eval-harness entry points
# ===========================================================================

_SERVER: Optional[SimServer] = None


def _cfg(usr_args, key, env_key, default, cast):
    if usr_args and key in usr_args and usr_args[key] is not None:
        return cast(usr_args[key])
    v = os.environ.get(env_key)
    return cast(v) if v is not None else default


def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _optional_bool(usr_args, key, env_key):
    if usr_args and key in usr_args and usr_args[key] is not None:
        value = str(usr_args[key]).strip().lower()
        if value in ("", "auto", "none", "null"):
            return None
        return _as_bool(value)
    value = os.environ.get(env_key)
    if value is None:
        return None
    value = value.strip().lower()
    if value in ("", "auto", "none", "null"):
        return None
    return _as_bool(value)


def _as_vec(v):
    """Accept a list/tuple or a comma-separated string -> list[float]."""
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(x) for x in str(v).split(",")]


def _configure_camera_shader_environment(usr_args=None) -> tuple[str, str]:
    """Select source and VLM camera shaders before RMBench creates the scene.

    SAPIEN binds a shader when each camera is created.  The default raster
    shader can expose orange render outlines in off-screen RGB captures, while
    the RMBench-compatible ``minimal`` shader produces clean RGB and still
    provides the packed position texture used by the depth path.  Unless the
    source shader is explicitly overridden, keep it aligned with the requested
    VLM shader so RGB-D, VLM, and segmentation observe the same clean render.
    """
    vlm_camera_shader_dir = _cfg(
        usr_args,
        "sim_vlm_camera_shader_dir",
        "SIM_VLM_CAMERA_SHADER_DIR",
        "",
        str,
    ).strip()
    source_camera_shader_dir = _cfg(
        usr_args,
        "sim_camera_shader_dir",
        "RMBENCH_CAMERA_SHADER_DIR",
        vlm_camera_shader_dir,
        str,
    ).strip()

    for env_key, shader_dir in (
        ("RMBENCH_CAMERA_SHADER_DIR", source_camera_shader_dir),
        ("RMBENCH_VLM_CAMERA_SHADER_DIR", vlm_camera_shader_dir),
    ):
        if shader_dir:
            os.environ[env_key] = shader_dir
        else:
            os.environ.pop(env_key, None)

    print(
        "[render] configured camera shaders: "
        f"source={source_camera_shader_dir or 'default'} "
        f"vlm={vlm_camera_shader_dir or 'same'}"
    )
    return source_camera_shader_dir, vlm_camera_shader_dir


def get_model(usr_args=None):
    global _SERVER
    if _SERVER is None:
        home_controlled = None
        if usr_args and usr_args.get("sim_home_controlled") is not None:
            home_controlled = _as_vec(usr_args["sim_home_controlled"])
        elif os.environ.get("SIM_HOME_CONTROLLED"):
            home_controlled = _as_vec(os.environ["SIM_HOME_CONTROLLED"])
        mirror_single_arm = _optional_bool(usr_args, "sim_mirror_single_arm", "SIM_MIRROR_SINGLE_ARM")
        if mirror_single_arm is None:
            mirror_single_arm = bool(
                usr_args
                and usr_args.get("dual_arm_embodied", False)
                and usr_args.get("single_physical_dual_slot", False)
            )
        _source_camera_shader_dir, vlm_camera_shader_dir = (
            _configure_camera_shader_environment(usr_args)
        )
        _SERVER = SimServer(
            host=_cfg(usr_args, "sim_host", "SIM_HOST", "0.0.0.0", str),
            port=_cfg(usr_args, "sim_port", "SIM_PORT", 65500, int),
            controlled_arm=_cfg(usr_args, "sim_arm", "SIM_ARM", CONTROLLED_ARM, str),
            drive=_cfg(usr_args, "sim_drive", "SIM_DRIVE", True, _as_bool),
            svlr_url=_cfg(usr_args, "svlr_url", "SVLR_URL", "http://127.0.0.1:7860", str),
            camera_key=_cfg(usr_args, "sim_camera_key", "SIM_CAMERA_KEY", CAMERA_KEY, str),
            save_debug_images=_cfg(usr_args, "sim_save_debug_images", "SIM_SAVE_DEBUG_IMAGES", False, _as_bool),
            call_vlm_before_llm=_cfg(usr_args, "sim_call_vlm", "SIM_CALL_VLM", True, _as_bool),
            ee_pos_tol_m=_cfg(usr_args, "sim_ee_pos_tol", "SIM_EE_POS_TOL", EE_POS_TOL_M, float),
            ee_settle_eps_m=_cfg(usr_args, "sim_ee_settle_eps", "SIM_EE_SETTLE_EPS", EE_SETTLE_EPS_M, float),
            ee_hold_frames=_cfg(usr_args, "sim_ee_hold_frames", "SIM_EE_HOLD_FRAMES", EE_HOLD_FRAMES, int),
            max_substeps_per_action=_cfg(usr_args, "sim_max_substeps", "SIM_MAX_SUBSTEPS",
                                         MAX_SUBSTEPS_PER_ACTION, int),
            min_substeps_per_action=_cfg(usr_args, "sim_min_substeps", "SIM_MIN_SUBSTEPS",
                                         MIN_SUBSTEPS_PER_ACTION, int),
            finish_idle_s=_cfg(usr_args, "sim_finish_idle_s", "SIM_FINISH_IDLE_S",
                               FINISH_IDLE_S, float),
            episode_handoff_timeout_s=_cfg(
                usr_args,
                "sim_episode_handoff_timeout_s",
                "SIM_EPISODE_HANDOFF_TIMEOUT_S",
                EPISODE_HANDOFF_TIMEOUT_S,
                float,
            ),
            home_on_reset=_cfg(usr_args, "sim_home", "SIM_HOME", HOME_ON_RESET, _as_bool),
            home_controlled=home_controlled,
            mirror_single_arm=mirror_single_arm,
            keep_alive_after_actions=_cfg(
                usr_args,
                "sim_keep_alive_after_actions",
                "SIM_KEEP_ALIVE_AFTER_ACTIONS",
                False,
                _as_bool,
            ),
            vlm_camera_shader_dir=vlm_camera_shader_dir,
            debug_dir=_cfg(
                usr_args,
                "sim_debug_dir",
                "SIM_DEBUG_DIR",
                ".",
                str,
            ),
        )
        _SERVER.start()
    return _SERVER


def reset_model(model):
    model.reset_episode()


def encode_obs(observation):
    return observation


def eval(TASK_ENV, model, observation):
    model.step(TASK_ENV, encode_obs(observation))


# ===========================================================================
# Mock env + harness loop (smoke test, no SAPIEN)
# ===========================================================================


class MockSimEnv:
    EE_STEP_M = 0.05  # Mock-only: expose partial motion for bridge debug logs.

    def __init__(self, w=640, h=480, step_lim=100000):
        self.w, self.h, self.step_lim = w, h, step_lim
        self.take_action_cnt = 0
        self.eval_success = False
        self._endpose = DEFAULT_ENDPOSE.copy()

    def get_instruction(self):
        return "pick up the block and place it on the plate"

    def take_action(self, action, action_type="ee", evaluate_success=True):
        target = np.asarray(action, dtype=np.float64).reshape(-1)
        # Move each arm's EE xyz toward the target by at most EE_STEP_M; snap
        # orientation + gripper. Real RMBench take_action is dense and normally
        # reaches the planned target in one call.
        for base in (0, 8):
            cur = self._endpose[base:base + 3]
            d = target[base:base + 3] - cur
            n = float(np.linalg.norm(d))
            cur += d if n <= self.EE_STEP_M else d * (self.EE_STEP_M / n)
            self._endpose[base + 3:base + 8] = target[base + 3:base + 8]
        self.take_action_cnt += 1
        rx, ry, rz = self._endpose[8:11]
        print(f"[mock-sim] take_action ({action_type}) cnt={self.take_action_cnt} "
              f"ee_r=({rx:.3f},{ry:.3f},{rz:.3f})")

    def get_obs(self):
        t = self.take_action_cnt
        x = np.linspace(0, 255, self.w, dtype=np.uint16)
        row = ((x + t * 3) % 256).astype(np.uint8)
        rgb = np.stack([np.tile(row, (self.h, 1)), np.tile(row[::-1], (self.h, 1)),
                        np.full((self.h, self.w), (t * 5) % 256, np.uint8)], axis=-1).astype(np.uint8)
        return {"observation": {CAMERA_KEY: {"rgb": rgb, "depth": np.full((self.h, self.w), 0.8, np.float32)}},
                "endpose": {
                    "left_endpose": np.concatenate([self._endpose[0:3], self._endpose[3:7]]),
                    "left_gripper": float(self._endpose[7]),
                    "right_endpose": np.concatenate([self._endpose[8:11], self._endpose[11:15]]),
                    "right_gripper": float(self._endpose[15]),
                }}


def _run_mock_harness(server: SimServer):
    env = MockSimEnv()
    reset_model(server)
    print("[mock-sim] serving; drive from SVLR (--drive) or POST /stop / Ctrl-C to end.")
    while env.take_action_cnt < env.step_lim and not server.bridge.stop_requested:
        eval(env, server, env.get_obs())
        if env.eval_success:
            break
    print("[mock-sim] episode ended.")


def parse_args():
    p = argparse.ArgumentParser(description="RoboTwin simulation web server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=65500)
    p.add_argument("--arm", choices=["left", "right"], default=CONTROLLED_ARM)
    p.add_argument("--mock", action="store_true", help="run the mock env (no SAPIEN)")
    p.add_argument("--drive", action="store_true", help="auto-drive a running SVLR Gradio app")
    p.add_argument("--svlr-url", default="http://127.0.0.1:7860")
    p.add_argument("--camera-key", default=CAMERA_KEY)
    p.add_argument("--save-debug-images", action="store_true")
    p.add_argument(
        "--mirror-single-arm",
        action="store_true",
        help="Mirror the controlled arm command into both 16D slots for one-robot RMBench embodiments.",
    )
    p.add_argument(
        "--keep-alive-after-actions",
        action="store_true",
        help="Keep the mock harness alive after SVLR actions finish; stop with POST /stop or Ctrl-C.",
    )
    p.add_argument(
        "--vlm-camera-shader-dir",
        default="",
        help="Optional RGB-only SAPIEN camera shader for the image sent to SVLR's VLM.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.mock:
        raise SystemExit("Standalone ships only the mock env. Set policy_name to this "
                         "module for real RoboTwin. Re-run with --mock to smoke-test.")
    global _SERVER
    _SERVER = SimServer(host=args.host, port=args.port, controlled_arm=args.arm,
                        drive=args.drive, svlr_url=args.svlr_url,
                        camera_key=args.camera_key,
                        save_debug_images=args.save_debug_images,
                        mirror_single_arm=args.mirror_single_arm,
                        keep_alive_after_actions=args.keep_alive_after_actions,
                        vlm_camera_shader_dir=args.vlm_camera_shader_dir)
    _SERVER.start()
    try:
        _run_mock_harness(_SERVER)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
